"""CPU regression checks for the released DeepConf confidence direction."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cognitive_probe.evaluation.deepconf_online import simulate_deepconf_trace

ROOT = Path(__file__).resolve().parents[1]


class DeepConfDirectionTest(unittest.TestCase):
    def test_preserve_blocks_tail_and_invalid_score_positions(self):
        cases = [
            ([10, 2, 2, 10], False, 4, 6),
            ([10, 10, 1], False, 3, 1),
            ([-1, 2, None, 2, 10], True, 4, 2),
            ([5, 5], False, 2, 5),
        ]
        for scores, stopped, stop_index, trace_score in cases:
            with self.subTest(scores=scores):
                result = simulate_deepconf_trace(
                    source_row_index=0, output_ids=list(range(len(scores))),
                    token_scores=scores, phase="adaptive", group_size=2,
                    lower_is_better=False, threshold=5, vote_weight="raw",
                    full_response="", decode_output_ids=None,
                )
                self.assertEqual(result.stopped_early, stopped)
                self.assertEqual(result.stop_index, stop_index)
                self.assertEqual(result.trace_score, trace_score)
                self.assertEqual(result.trace_weight, trace_score)

    def run_cli(self, script, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / script), *map(str, args)],
            env=dict(os.environ, PYTHONPATH=str(ROOT / "src"),
                     PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=""),
            capture_output=True, text=True, timeout=30,
        )

    def test_score_and_replay_cli_with_legacy_metadata(self):
        sharp = [(math.log(.99), 1, None), (math.log(.01), 2, None)]
        flat = [(math.log(.5), 1, None), (math.log(.5), 2, None)]
        sharp_score = -sum(item[0] for item in sharp) / 2
        flat_score = -sum(item[0] for item in flat) / 2
        # Warmup minima are flat_score and sharp_score. Later low-confidence
        # output must prune; the high-confidence output must survive.
        distributions = [[sharp, sharp, flat, flat], [sharp] * 4, [flat] * 4, [sharp] * 4]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            source, scored, output = (temp / name for name in ("source", "scored", "output"))
            source.mkdir()
            rows = [
                {"example_id": "q1", "input_ids": [0], "output_ids": [1, 2, 3, 4],
                 "response": r"\boxed{42}", "answer": "42", "finish_reason": "stop",
                 "output_top_logprobs": distribution}
                for distribution in distributions
            ]
            (source / "generation_results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            result = self.run_cli(
                "scripts/evaluation/prob/run_deepconf_logprob.py",
                "--input-dir", source, "--output-dir", scored, "--topk", 2,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            args_path = scored / "args.json"
            args = json.loads(args_path.read_text())
            self.assertFalse(args["probe"]["lower_is_better"])
            # Old raw score artifacts must remain reusable without regeneration.
            args["probe"]["lower_is_better"] = True
            args_path.write_text(json.dumps(args))
            result = self.run_cli(
                "scripts/evaluation/postprocess/run_deepconf_online.py",
                "--input-dir", scored, "--output-dir", output,
                "--total-budget", 4, "--warmup-traces", 2, "--group-size", 2,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((output / "deepconf_online_summary.json").read_text())
            config = summary["config"]
            self.assertFalse(config["lower_is_better"])
            self.assertEqual(config["vote_weight"], "raw")
            self.assertTrue(config["include_warmup_in_vote"])
            self.assertFalse(config["consensus_stop_enabled"])
            sample = summary["samples"][0]
            self.assertAlmostEqual(sample["threshold"], flat_score * .1 + sharp_score * .9)
            self.assertTrue(sample["consensus_reached"])
            self.assertEqual(sample["used_trace_count"], 4)
            traces = [json.loads(line) for line in (output / "deepconf_online_traces.jsonl").read_text().splitlines()]
            self.assertEqual([t["stopped_early"] for t in traces], [False, False, True, False])
            self.assertEqual([t["stop_index"] for t in traces], [4, 4, 2, 4])
            for trace in traces:
                self.assertEqual(trace["trace_weight"], trace["trace_score"])
            results = [json.loads(line) for line in (output / "generation_results.jsonl").read_text().splitlines()]
            for row in results:
                self.assertFalse(row["prune_meta"]["lower_is_better"])
                self.assertEqual(row["prune_meta"]["vote_weight_method"], "raw")
            saved_args = json.loads((output / "args.json").read_text())
            self.assertFalse(saved_args["probe_prune"]["prune_config"]["strategy_args"]["lower_is_better"])
            self.assertTrue((output / "final_outputs.jsonl").exists())

    def test_cli_rejects_decreasing_vote_weights_before_reading_data(self):
        for weight in ("exp_neg", "inverse"):
            with self.subTest(weight=weight):
                result = self.run_cli(
                    "scripts/evaluation/postprocess/run_deepconf_online.py",
                    "--input-dir", "unused", "--vote-weight", weight,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("invalid choice", result.stderr)


if __name__ == "__main__":
    unittest.main()
