from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cognitive_probe.evaluation.stats import (
    apply_holm_correction,
    bootstrap_ci,
    combine_stouffer_pvalues,
    compute_per_problem_metrics_from_run,
    paired_test,
    run_full_bootstrap_evaluation,
    stratified_bootstrap_meta,
    validate_against_eval_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_tiny_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "run_overview.json",
        {
            "run_id": str(run_dir),
            "run_dir": str(run_dir),
            "dataset_name": "tiny",
            "model_name": "mock",
            "run_type": "prune",
            "strategy_name": "mock_prune",
            "strategy_label": "mock_prune",
        },
    )
    _write_jsonl(
        run_dir / "source_targets.jsonl",
        [
            {"sample_id": "p1", "source_row_index": 0, "answer": "1"},
            {"sample_id": "p1", "source_row_index": 1, "answer": "1"},
            {"sample_id": "p1", "source_row_index": 2, "answer": "1"},
            {"sample_id": "p2", "source_row_index": 3, "answer": "2"},
            {"sample_id": "p2", "source_row_index": 4, "answer": "2"},
            {"sample_id": "p2", "source_row_index": 5, "answer": "2"},
        ],
    )
    _write_jsonl(
        run_dir / "generation_segments.jsonl",
        [
            {
                "segment_id": f"s{i}",
                "sample_id": "p1" if i < 3 else "p2",
                "response_text": "",
                "source_row_index": i,
                "visible_output_tokens": 1,
                "usage": {"input_tokens": 0, "output_tokens": 1, "request_count": 1},
            }
            for i in range(6)
        ],
    )
    _write_jsonl(run_dir / "decision_log.jsonl", [])
    _write_jsonl(
        run_dir / "final_outputs.jsonl",
        [
            {
                "output_id": "p1__0",
                "sample_id": "p1",
                "final_segment_id": "s0",
                "response": "answer is \\boxed{1}",
                "k_group": 1,
                "source_row_index": 0,
                "was_pruned": False,
            },
            {
                "output_id": "p1__1",
                "sample_id": "p1",
                "final_segment_id": "s1",
                "response": "answer is \\boxed{0}",
                "k_group": 2,
                "source_row_index": 1,
                "was_pruned": True,
            },
            {
                "output_id": "p1__2",
                "sample_id": "p1",
                "final_segment_id": "s2",
                "response": "answer is \\boxed{0}",
                "k_group": 3,
                "source_row_index": 2,
                "was_pruned": False,
            },
            {
                "output_id": "p2__0",
                "sample_id": "p2",
                "final_segment_id": "s3",
                "response": "no final answer",
                "k_group": 1,
                "source_row_index": 3,
                "was_pruned": False,
            },
            {
                "output_id": "p2__1",
                "sample_id": "p2",
                "final_segment_id": "s4",
                "response": "still no final answer",
                "k_group": 2,
                "source_row_index": 4,
                "was_pruned": True,
            },
            {
                "output_id": "p2__2",
                "sample_id": "p2",
                "final_segment_id": "s5",
                "response": "invalid",
                "k_group": 3,
                "source_row_index": 5,
                "was_pruned": False,
            },
        ],
    )


class BootstrapStatsTests(unittest.TestCase):
    def test_bootstrap_ci_is_reproducible_and_percentile(self) -> None:
        values = np.asarray([0.0, 0.25, 0.75, 1.0])
        first = bootstrap_ci(values, B=500, seed=123)
        second = bootstrap_ci(values, B=500, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(first["ci_method"], "percentile_bootstrap")
        self.assertLessEqual(first["ci_lo"], first["point"])
        self.assertGreaterEqual(first["ci_hi"], first["point"])

    def test_bootstrap_ci_uses_jeffreys_for_saturated_binary_arrays(self) -> None:
        all_zero = bootstrap_ci(np.zeros(8), B=100, seed=0)
        all_one = bootstrap_ci(np.ones(8), B=100, seed=0)
        self.assertEqual(all_zero["ci_method"], "jeffreys_beta")
        self.assertEqual(all_one["ci_method"], "jeffreys_beta")
        self.assertEqual(all_zero["point"], 0.0)
        self.assertEqual(all_one["point"], 1.0)
        self.assertTrue(0.0 <= all_zero["ci_lo"] <= all_zero["ci_hi"] <= 1.0)
        self.assertTrue(0.0 <= all_one["ci_lo"] <= all_one["ci_hi"] <= 1.0)

    def test_paired_test_ci_contains_observed_diff(self) -> None:
        metrics_a = {
            "sample_ids": ["a", "b", "c", "d"],
            "per_problem": {"maj": np.asarray([1.0, 1.0, 0.0, 1.0])},
        }
        metrics_b = {
            "sample_ids": ["a", "b", "c", "d"],
            "per_problem": {"maj": np.asarray([0.0, 1.0, 0.0, 0.0])},
        }
        result = paired_test(metrics_a, metrics_b, "a", "b", "maj", B=500, seed=1)
        self.assertAlmostEqual(result["mean_diff"], 0.5)
        self.assertLessEqual(result["diff_ci_lo"], result["mean_diff"])
        self.assertGreaterEqual(result["diff_ci_hi"], result["mean_diff"])

    def test_stratified_bootstrap_meta_preserves_stratum_counts(self) -> None:
        result = stratified_bootstrap_meta(
            {
                "small": np.asarray([1.0, 0.0]),
                "large": np.asarray([0.0, 0.0, 1.0, 1.0]),
            },
            B=200,
            seed=3,
        )
        self.assertAlmostEqual(result["mean_diff"], 3.0 / 6.0)
        self.assertEqual(result["n_problems"], 6)
        self.assertEqual(result["n_problems_by_stratum"], {"small": 2, "large": 4})
        self.assertEqual(result["ci_method"], "stratified_percentile_bootstrap")
        self.assertLessEqual(result["diff_ci_lo"], result["mean_diff"])
        self.assertGreaterEqual(result["diff_ci_hi"], result["mean_diff"])

    def test_stratified_bootstrap_p_value_centers_within_stratum(self) -> None:
        result = stratified_bootstrap_meta(
            {
                "high": np.asarray([1.0, 1.0]),
                "low": np.asarray([0.0, 0.0, 0.0, 0.0]),
            },
            B=200,
            seed=11,
            alternative="greater",
        )
        self.assertAlmostEqual(result["mean_diff"], 1.0 / 3.0)
        self.assertEqual(result["bootstrap_p_raw"], 0.0)

    def test_stouffer_uses_directional_p_values_and_raw_n_weights(self) -> None:
        negative = combine_stouffer_pvalues([0.9], weights=[30])
        positive = combine_stouffer_pvalues([0.1], weights=[30])
        weighted = combine_stouffer_pvalues([0.1, 0.9], weights=[30, 1055])

        self.assertLess(negative["stat"], 0.0)
        self.assertGreater(positive["stat"], 0.0)
        self.assertEqual(weighted["weights"], [30.0, 1055.0])
        self.assertLess(weighted["stat"], 0.0)

    def test_holm_correction_handles_null_p_values(self) -> None:
        corrected = apply_holm_correction(
            [
                {"wilcoxon_p_raw": 0.01},
                {"wilcoxon_p_raw": 0.04},
                {"wilcoxon_p_raw": None},
            ],
            "wilcoxon_p_raw",
        )
        self.assertGreaterEqual(corrected[0]["wilcoxon_p_holm"], corrected[0]["wilcoxon_p_raw"])
        self.assertGreaterEqual(corrected[1]["wilcoxon_p_holm"], corrected[1]["wilcoxon_p_raw"])
        self.assertIsNone(corrected[2]["wilcoxon_p_holm"])
        self.assertFalse(corrected[2]["wilcoxon_p_holm_reject"])

    def test_v2_semantics_alignment_on_tiny_run(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_tiny_run(run_dir)
            metrics = compute_per_problem_metrics_from_run(run_dir, k=3)
            np.testing.assert_allclose(metrics["mean"], np.asarray([1.0 / 3.0, 0.0]))
            np.testing.assert_allclose(metrics["pass"], np.asarray([1.0, 0.0]))
            np.testing.assert_allclose(metrics["maj"], np.asarray([1.0, 0.0]))
            np.testing.assert_allclose(metrics["tok"], np.asarray([1.0, 1.0]))
            self.assertEqual(metrics["diagnostics"]["n_outputs_pruned"], 2)
            self.assertEqual(metrics["diagnostics"]["n_outputs_vote_eligible"], 4)
            self.assertEqual(metrics["diagnostics"]["n_problems_without_vote_answer"], 1)

            alignment = validate_against_eval_report(run_dir, k=3, per_problem_metrics=metrics)
            self.assertTrue(alignment["passed"])
            self.assertTrue(math.isclose(alignment["mean_abs_error"], 0.0, abs_tol=1e-12))
            self.assertTrue(math.isclose(alignment["pass_abs_error"], 0.0, abs_tol=1e-12))
            self.assertTrue(math.isclose(alignment["maj_abs_error"], 0.0, abs_tol=1e-12))
            self.assertTrue(math.isclose(alignment["tok_abs_error"], 0.0, abs_tol=1e-12))

    def test_full_evaluation_writes_output_contract(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            out_dir = Path(tmp) / "stats"
            _make_tiny_run(run_dir)

            run_full_bootstrap_evaluation(
                method_specs=[
                    {"method": "vanilla", "run_dir": str(run_dir)},
                    {"method": "copy", "run_dir": str(run_dir)},
                ],
                reference_method="vanilla",
                output_dir=out_dir,
                dataset_name="tiny",
                k=3,
                B=50,
                seed=7,
                command="test command",
            )

            for name in [
                "summary.json",
                "per_method_metrics.json",
                "per_method_metrics.csv",
                "paired_tests.json",
                "paired_tests.csv",
                "diagnostics.json",
                "README.md",
            ]:
                self.assertTrue((out_dir / name).exists(), name)

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["dataset"], "tiny")
            self.assertEqual(summary["methods"], ["vanilla", "copy"])
            paired = json.loads((out_dir / "paired_tests.json").read_text(encoding="utf-8"))
            self.assertEqual(paired["holm_family_size"], 3)
            self.assertEqual(len(paired["tests"]), 3)
            metrics = json.loads((out_dir / "per_method_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("point_ci_half_width", metrics["vanilla"]["metrics"]["mean"])
            self.assertIn("point_pm_ci", metrics["vanilla"]["metrics"]["mean"])
            self.assertIn("point_pm_ci_pct", metrics["vanilla"]["metrics"]["mean"])
            self.assertIn("point_pm_ci_k", metrics["vanilla"]["metrics"]["tok"])
            self.assertIn("diff_pm_ci", paired["tests"][0])
            readme = (out_dir / "README.md").read_text(encoding="utf-8")
            self.assertIn("point +- half-width", readme)


if __name__ == "__main__":
    unittest.main()
