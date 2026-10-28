from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROB_ROOT = PROJECT_ROOT / "scripts" / "evaluation" / "prob"
EVAL_ROOT = PROJECT_ROOT / "src" / "cognitive_probe" / "evaluation"
if str(PROB_ROOT) not in sys.path:
    sys.path.insert(0, str(PROB_ROOT))
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from deepconf_logprob import (  # noqa: E402
    INVALID_SCORE,
    align_input_top_logprobs,
    compute_token_confidence,
    compute_token_probability_proxy,
    compute_trace_confidence,
    extract_topk_logprobs,
)
from run_deepconf_logprob import _load_checkpoint_records  # noqa: E402


class DeepConfLogprobTest(unittest.TestCase):
    def test_extract_topk_logprobs_sorts_descending(self) -> None:
        step = [(-3.0, 1, None), (-0.4, 2, None), (-1.2, 3, None)]
        self.assertEqual(extract_topk_logprobs(step, topk=2), [-0.4, -1.2])

    def test_compute_token_confidence_uses_negative_mean_logprob(self) -> None:
        step = [(-0.2, 1, None), (-0.8, 2, None), (-1.5, 3, None)]
        self.assertAlmostEqual(compute_token_confidence(step, topk=2), 0.5)

    def test_compute_token_probability_proxy_is_monotonic_transform(self) -> None:
        step = [(-0.2, 1, None), (-0.8, 2, None)]
        self.assertAlmostEqual(compute_token_probability_proxy(step, topk=2), 0.6065306597)

    def test_compute_token_confidence_accepts_dict_payload(self) -> None:
        step = {"1": -0.7, "2": -0.3, "3": -2.0}
        self.assertAlmostEqual(compute_token_confidence(step, topk=2), 0.5)

    def test_align_input_top_logprobs_prefers_exact_prompt_slice(self) -> None:
        entries = ["p0", "p1", "o0", "o1", "o2"]
        self.assertEqual(
            align_input_top_logprobs(entries, input_len=2, output_len=3),
            ["o0", "o1", "o2"],
        )

    def test_align_input_top_logprobs_handles_omitted_first_prompt_token(self) -> None:
        entries = ["p1", "o0", "o1", "o2"]
        self.assertEqual(
            align_input_top_logprobs(entries, input_len=2, output_len=3),
            ["o0", "o1", "o2"],
        )

    def test_compute_trace_confidence_ignores_invalid_entries(self) -> None:
        self.assertEqual(compute_trace_confidence([INVALID_SCORE, 0.4, 0.8]), 0.4)

    def test_load_checkpoint_records_uses_latest_duplicate(self) -> None:
        records = [
            {"output_ids": [1, 2], "probs": [INVALID_SCORE, INVALID_SCORE]},
            {"output_ids": [3], "probs": [INVALID_SCORE]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.jsonl"
            checkpoint.write_text(
                '{"record_idx":0,"scores":[0.1,0.2]}\n'
                '{"record_idx":0,"scores":[0.3,0.4]}\n'
                '{"record_idx":1,"scores":[0.5]}\n'
            )

            completed, scored_tokens = _load_checkpoint_records(
                checkpoint_path=checkpoint,
                records=records,
                score_field="probs",
            )

        self.assertEqual(completed, {0, 1})
        self.assertEqual(records[0]["probs"], [0.3, 0.4])
        self.assertEqual(records[1]["probs"], [0.5])
        self.assertEqual(scored_tokens, 3)


if __name__ == "__main__":
    unittest.main()
