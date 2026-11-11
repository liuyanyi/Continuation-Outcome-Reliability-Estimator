#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import warnings as py_warnings
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import orjson
from scipy.stats import binomtest, wilcoxon

from cognitive_probe.evaluation.data_model import FINAL_OUTPUTS_FILE
from cognitive_probe.evaluation.io import load_jsonl
from cognitive_probe.evaluation.stats import (
    _add_ci_display_fields,
    _dump_json,
    _git_commit,
    _mean,
    _metric_values,
    _write_paired_csv,
    apply_holm_correction,
    bootstrap_ci,
    method_pair_family,
)
from cognitive_probe.evaluation.v2 import load_generation_segments, load_run_overview

JUDGE_RESULTS_FILE = "livebench_judge_results.jsonl"
REPORT_FILE = "livebench_eval_report.json"
METRICS = ("lcb_pass", "mean", "pass", "maj", "tok")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path):
    with path.open("rb") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                yield orjson.loads(line)


def parse_method_spec(raw: str) -> dict[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--method must use METHOD=RUN_DIR.")
    method, run_dir = raw.split("=", 1)
    method = method.strip()
    run_dir = run_dir.strip()
    if not method or not run_dir:
        raise argparse.ArgumentTypeError("--method requires non-empty METHOD and RUN_DIR.")
    return {"method": method, "run_dir": run_dir}


def parse_metric_list(raw: str) -> tuple[str, ...]:
    metrics = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = [metric for metric in metrics if metric not in METRICS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown metric(s): {', '.join(unknown)}. Expected subset of {METRICS}.")
    return metrics


def _group_outputs(final_outputs: list[dict[str, Any]], k: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(
        final_outputs,
        key=lambda item: (
            str(item.get("sample_id")),
            int(item.get("k_group") or 0),
            int(item.get("source_row_index") or 0),
        ),
    ):
        grouped[str(row.get("sample_id"))].append(row)
    too_short = [sample_id for sample_id, rows in grouped.items() if len(rows) < k]
    if too_short:
        raise ValueError(f"{len(too_short)} samples have fewer than k={k} outputs; first={too_short[:5]}")
    return {sample_id: rows[:k] for sample_id, rows in grouped.items()}


def _load_judge_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        output_id = row.get("output_id")
        if output_id:
            results[str(output_id)] = row
    return results


def _collect_lineage(segment_id: str, segment_map: dict[str, Any]) -> list[Any]:
    lineage: list[Any] = []
    current = segment_map.get(segment_id)
    while current is not None:
        lineage.append(current)
        current = segment_map.get(current.parent_segment_id) if current.parent_segment_id else None
    lineage.reverse()
    return lineage


def _visible_tokens_by_output(run_dir: Path, grouped_outputs: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    segment_map = {segment.segment_id: segment for segment in load_generation_segments(run_dir)}
    result: dict[str, int] = {}
    for rows in grouped_outputs.values():
        for output in rows:
            output_id = str(output.get("output_id"))
            lineage = _collect_lineage(str(output.get("final_segment_id") or ""), segment_map)
            if lineage:
                result[output_id] = sum(int(segment.visible_output_tokens or 0) for segment in lineage)
            else:
                result[output_id] = int(output.get("source_output_tokens") or 0)
    return result


def _behavior_vector(result: dict[str, Any]) -> tuple[bool, ...] | None:
    vector = result.get("test_pass_vector")
    if not isinstance(vector, list):
        summary = result.get("test_summary") if isinstance(result.get("test_summary"), dict) else {}
        vector = summary.get("test_pass_vector")
    if not isinstance(vector, list):
        return None
    if not all(isinstance(item, bool) for item in vector):
        return None
    return tuple(vector)


def _majority_behavior_is_correct(signatures: list[tuple[bool, ...]]) -> bool:
    if not signatures:
        return False
    counts: dict[tuple[bool, ...], int] = {}
    first_seen: dict[tuple[bool, ...], int] = {}
    for idx, signature in enumerate(signatures):
        counts[signature] = counts.get(signature, 0) + 1
        first_seen.setdefault(signature, idx)
    majority_signature = min(
        counts.items(),
        key=lambda item: (-item[1], first_seen[item[0]]),
    )[0]
    return bool(majority_signature) and all(majority_signature)


def compute_per_problem_metrics(run_dir: Path, k: int) -> dict[str, Any]:
    grouped = _group_outputs(load_jsonl(run_dir / FINAL_OUTPUTS_FILE), k)
    judge_results = _load_judge_results(run_dir / JUDGE_RESULTS_FILE)
    visible_tokens = _visible_tokens_by_output(run_dir, grouped)

    sample_ids = sorted(grouped)
    mean_values: list[float] = []
    pass_values: list[float] = []
    maj_values: list[float] = []
    tok_values: list[float] = []
    missing: list[str] = []
    n_outputs_pruned = 0
    n_outputs_vote_eligible = 0
    n_problems_without_vote_signature = 0
    missing_behavior: list[str] = []
    for sample_id in sample_ids:
        scores: list[int] = []
        vote_signatures: list[tuple[bool, ...]] = []
        tokens: list[int] = []
        for output in grouped[sample_id]:
            output_id = str(output.get("output_id"))
            result = judge_results.get(output_id)
            if output.get("was_pruned"):
                # pruned-as-abstain unified convention: pruned outputs score 0
                # for Mean/Pass and are not eligible for Maj voting.
                scores.append(0)
                n_outputs_pruned += 1
                tokens.append(visible_tokens[output_id])
                continue
            if result is None:
                missing.append(output_id)
                continue
            score = 1 if result.get("passed") else 0
            scores.append(score)
            n_outputs_vote_eligible += 1
            signature = _behavior_vector(result)
            if signature is None:
                missing_behavior.append(output_id)
            else:
                vote_signatures.append(signature)
            tokens.append(visible_tokens[output_id])
        if len(scores) != k:
            raise ValueError(f"sample_id={sample_id} has {len(scores)} judged outputs, expected k={k}.")
        mean_values.append(float(sum(scores) / k))
        pass_values.append(float(max(scores)))
        if not vote_signatures:
            n_problems_without_vote_signature += 1
        maj_values.append(1.0 if _majority_behavior_is_correct(vote_signatures) else 0.0)
        tok_values.append(float(sum(tokens) / k))
    if missing:
        raise ValueError(f"Missing {len(missing)} judge results; first={missing[:5]}")
    if missing_behavior:
        raise ValueError(
            f"Missing behavior signatures for {len(missing_behavior)} judged outputs; first={missing_behavior[:5]}. "
            "Re-run run_eval_livebench.py with --run-judge to refresh LiveCodeBench judge cache."
        )

    per_problem = {
        "lcb_pass": np.asarray(mean_values, dtype=np.float64),
        "mean": np.asarray(mean_values, dtype=np.float64),
        "pass": np.asarray(pass_values, dtype=np.float64),
        "maj": np.asarray(maj_values, dtype=np.float64),
        "tok": np.asarray(tok_values, dtype=np.float64),
    }
    report = _read_json(run_dir / REPORT_FILE)
    expected = {
        "lcb_pass": report.get("lcb_pass_at_1"),
        "mean": (report.get("mean_at_k") or {}).get(str(k)),
        "pass": (report.get("best_score_mean") or {}).get(str(k)),
        "maj": (report.get("maj_at_k") or {}).get(str(k)),
        "tok": report.get("avg_final_visible_tokens"),
    }
    alignment = {
        f"{metric}_abs_error": abs(float(_mean(values)) - float(expected[metric]))
        for metric, values in per_problem.items()
        if expected.get(metric) is not None
    }
    alignment["passed"] = all(value <= 1e-9 for value in alignment.values())
    return {
        "run_dir": str(run_dir),
        "k": k,
        "sample_ids": sample_ids,
        "per_problem": per_problem,
        "diagnostics": {
            "n_problems": len(sample_ids),
            "n_outputs_selected": len(sample_ids) * k,
            "n_outputs_pruned": n_outputs_pruned,
            "n_outputs_vote_eligible": n_outputs_vote_eligible,
            "n_problems_without_vote_signature": n_problems_without_vote_signature,
            "responses_per_item_min": min(len(rows) for rows in grouped.values()),
            "responses_per_item_max": max(len(rows) for rows in grouped.values()),
        },
        "livebench_alignment": alignment,
    }


def evaluate_method(method: str, run_dir: Path, k: int, B: int, seed: int) -> dict[str, Any]:
    evaluation = compute_per_problem_metrics(run_dir, k)
    metrics: dict[str, dict[str, Any]] = {}
    for idx, metric in enumerate(METRICS):
        metric_ci = bootstrap_ci(evaluation["per_problem"][metric], B=B, seed=seed + idx)
        if metric == "tok":
            metric_ci.pop("point_pm_ci_pct", None)
            _add_ci_display_fields(
                metric_ci,
                point_key="point",
                lo_key="ci_lo",
                hi_key="ci_hi",
                prefix="point",
                include_percent=False,
                include_kilo=True,
            )
        metric_ci["livebench_report_field"] = {
            "lcb_pass": "lcb_pass_at_1",
            "mean": "mean_at_k",
            "pass": "best_score_mean",
            "maj": "maj_at_k",
            "tok": "avg_final_visible_tokens",
        }[metric]
        metrics[metric] = metric_ci
    return {"method": method, "metrics": metrics, **evaluation}


def paired_test(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    method_a: str,
    method_b: str,
    metric: str,
    B: int,
    seed: int,
    alternative: str,
) -> dict[str, Any]:
    if metrics_a["sample_ids"] != metrics_b["sample_ids"]:
        raise ValueError(f"sample_id mismatch between {method_a} and {method_b}.")
    vals_a = _metric_values(metrics_a["per_problem"], metric)
    vals_b = _metric_values(metrics_b["per_problem"], metric)
    diffs = vals_a - vals_b
    observed = _mean(diffs)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diffs.size, size=(B, diffs.size))
    boot_diffs = diffs[indices].mean(axis=1)
    diff_ci_lo, diff_ci_hi = np.quantile(boot_diffs, [0.025, 0.975])

    warnings: list[str] = []
    try:
        with py_warnings.catch_warnings():
            py_warnings.simplefilter("ignore", RuntimeWarning)
            wilcoxon_result = wilcoxon(diffs, alternative=alternative, zero_method="wilcox")
        wilcoxon_stat = float(wilcoxon_result.statistic)
        wilcoxon_p_raw = float(wilcoxon_result.pvalue)
    except ValueError as exc:
        wilcoxon_stat = None
        wilcoxon_p_raw = None
        warnings.append(f"wilcoxon_failed:{exc}")

    centered = diffs - observed
    null_indices = np.random.default_rng(seed + 1).integers(0, diffs.size, size=(B, diffs.size))
    null_boot = centered[null_indices].mean(axis=1)
    if alternative == "greater":
        bootstrap_p_raw = float(np.mean(null_boot >= observed))
    elif alternative == "less":
        bootstrap_p_raw = float(np.mean(null_boot <= observed))
    else:
        bootstrap_p_raw = float(np.mean(np.abs(null_boot) >= abs(observed)))

    mcnemar_n_a_only = mcnemar_n_b_only = mcnemar_n_discordant = mcnemar_p_raw = None
    if np.all(np.isin(vals_a, [0.0, 1.0])) and np.all(np.isin(vals_b, [0.0, 1.0])):
        mcnemar_n_a_only = int(np.sum((vals_a == 1.0) & (vals_b == 0.0)))
        mcnemar_n_b_only = int(np.sum((vals_a == 0.0) & (vals_b == 1.0)))
        mcnemar_n_discordant = mcnemar_n_a_only + mcnemar_n_b_only
        if mcnemar_n_discordant == 0:
            mcnemar_p_raw = 1.0
        else:
            mcnemar_p_raw = float(
                binomtest(mcnemar_n_a_only, n=mcnemar_n_discordant, p=0.5, alternative=alternative).pvalue
            )

    payload = {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "family": method_pair_family(method_a, method_b),
        "n_problems": int(diffs.size),
        "mean_diff": observed,
        "diff_ci_lo": float(diff_ci_lo),
        "diff_ci_hi": float(diff_ci_hi),
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p_raw": wilcoxon_p_raw,
        "bootstrap_p_raw": bootstrap_p_raw,
        "mcnemar_n_a_only": mcnemar_n_a_only,
        "mcnemar_n_b_only": mcnemar_n_b_only,
        "mcnemar_n_discordant": mcnemar_n_discordant,
        "mcnemar_p_raw": mcnemar_p_raw,
        "alternative": alternative,
        "warnings": warnings,
    }
    _add_ci_display_fields(
        payload,
        point_key="mean_diff",
        lo_key="diff_ci_lo",
        hi_key="diff_ci_hi",
        prefix="diff",
        include_percent=(metric != "tok"),
        include_kilo=(metric == "tok"),
    )
    return payload


def _apply_holm(rows: list[dict[str, Any]], p_field: str, group_key: str, out_p: str, out_reject: str) -> None:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[str(row[group_key])].append(idx)
    for indices in grouped.values():
        adjusted = apply_holm_correction([rows[idx] for idx in indices], p_field)
        holm_key = p_field.replace("_raw", "_holm")
        reject_key = f"{holm_key}_reject"
        for idx, item in zip(indices, adjusted, strict=True):
            rows[idx][out_p] = item.get(holm_key)
            rows[idx][out_reject] = item.get(reject_key)


def add_holm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(row) for row in rows]
    for p_field in ("wilcoxon_p_raw", "bootstrap_p_raw"):
        prefix = p_field.replace("_raw", "")
        _apply_holm(result, p_field, "family", f"{prefix}_holm_method", f"{prefix}_holm_method_reject")
        _apply_holm(result, p_field, "metric", f"{prefix}_holm_metric", f"{prefix}_holm_metric_reject")
        adjusted = apply_holm_correction(result, p_field)
        holm_key = p_field.replace("_raw", "_holm")
        reject_key = f"{holm_key}_reject"
        for row, item in zip(result, adjusted, strict=True):
            row[f"{prefix}_holm_global"] = item.get(holm_key)
            row[f"{prefix}_holm_global_reject"] = item.get(reject_key)
    return result


def _write_per_method_csv(path: Path, evaluations: dict[str, dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "run_dir",
        "k",
        "metric",
        "point",
        "ci_lo",
        "ci_hi",
        "point_ci_half_width",
        "point_pm_ci",
        "point_pm_ci_pct",
        "point_pm_ci_k",
        "n",
        "B",
        "ci_method",
        "livebench_report_field",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, evaluation in evaluations.items():
            for metric, payload in evaluation["metrics"].items():
                writer.writerow(
                    {
                        "method": method,
                        "run_dir": evaluation["run_dir"],
                        "k": evaluation["k"],
                        "metric": metric,
                        **{key: payload.get(key) for key in fieldnames if key in payload},
                    }
                )


def _jsonable_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in evaluation.items()
        if key not in {"sample_ids", "per_problem"}
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method_specs = args.method
    evaluations_with_arrays: dict[str, dict[str, Any]] = {}
    method_runs: dict[str, dict[str, Any]] = {}
    for spec_idx, spec in enumerate(method_specs):
        method = spec["method"]
        run_dir = Path(spec["run_dir"]).resolve()
        evaluation = evaluate_method(method, run_dir, args.k, args.B, args.seed + spec_idx * 1009)
        evaluations_with_arrays[method] = evaluation
        overview = load_run_overview(run_dir)
        method_runs[method] = {
            "run_dir": str(run_dir),
            "run_id": overview.run_id,
            "dataset_name": overview.dataset_name,
            "model_name": overview.model_name,
            "run_type": overview.run_type,
            "strategy_name": overview.strategy_name,
        }

    if args.reference not in evaluations_with_arrays:
        raise ValueError(f"Reference method {args.reference!r} not in methods.")
    methods = list(evaluations_with_arrays)
    pair_specs: list[tuple[str, str]] = []
    if args.comparison_mode == "all-pairs":
        pair_specs = [(a, b) for a in methods for b in methods if a != b]
    else:
        pair_specs = [(method, args.reference) for method in methods if method != args.reference]
    metric_alternatives = {
        metric: ("less" if metric == "tok" else args.alternative)
        for metric in args.paired_metrics
    }
    paired_tests = add_holm(
        [
            paired_test(
                evaluations_with_arrays[a],
                evaluations_with_arrays[b],
                a,
                b,
                metric,
                args.B,
                args.seed + pair_idx * 1009 + metric_idx,
                metric_alternatives[metric],
            )
            for pair_idx, (a, b) in enumerate(pair_specs)
            for metric_idx, metric in enumerate(args.paired_metrics)
        ]
    )

    per_method_metrics = {method: _jsonable_evaluation(evaluation) for method, evaluation in evaluations_with_arrays.items()}
    summary = {
        "dataset": args.dataset_name,
        "k": args.k,
        "n_problems": len(next(iter(evaluations_with_arrays.values()))["sample_ids"]),
        "B": args.B,
        "seed": args.seed,
        "methods": methods,
        "reference_method": args.reference,
        "comparison_mode": args.comparison_mode,
        "paired_metrics": list(args.paired_metrics),
        "metric_alternatives": metric_alternatives,
        "run_timestamp": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_commit": _git_commit(),
        "method_runs": method_runs,
        "files": {
            "per_method_metrics": "per_method_metrics.json",
            "paired_tests": "paired_tests.json",
            "per_method_metrics_csv": "per_method_metrics.csv",
            "paired_tests_csv": "paired_tests.csv",
            "diagnostics": "diagnostics.json",
            "readme": "README.md",
        },
    }
    paired_payload = {
        "reference_method": args.reference,
        "alternative": args.alternative,
        "metric_alternatives": metric_alternatives,
        "comparison_mode": args.comparison_mode,
        "k": args.k,
        "paired_metrics": list(args.paired_metrics),
        "tests": paired_tests,
    }
    diagnostics = {
        "dataset": args.dataset_name,
        "k": args.k,
        "per_method_diagnostics": {
            method: evaluation["diagnostics"] for method, evaluation in per_method_metrics.items()
        },
        "livebench_alignment_failures": [
            method for method, evaluation in per_method_metrics.items()
            if not evaluation["livebench_alignment"].get("passed")
        ],
    }

    _dump_json(output_dir / "summary.json", summary)
    _dump_json(output_dir / "per_method_metrics.json", per_method_metrics)
    _dump_json(output_dir / "paired_tests.json", paired_payload)
    _dump_json(output_dir / "diagnostics.json", diagnostics)
    _write_per_method_csv(output_dir / "per_method_metrics.csv", per_method_metrics)
    _write_paired_csv(output_dir / "paired_tests.csv", paired_tests, args.k)

    command = " ".join(shlex.quote(arg) for arg in sys.argv)
    readme_lines = [
        f"# LiveCodeBench Bootstrap Stats: {args.dataset_name}",
        "",
        "## Command",
        f"`{command}`",
        "",
        "## Per-Method Metrics",
        "Maj clusters non-pruned outputs by full test pass/fail behavior signature, picks the largest cluster, and scores it correct only when that signature passes every test; mean/pass/lcb_pass still score all selected outputs.",
        "",
        "| method | metric | point | 95% CI | point +- half-width | percent / K-tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method, evaluation in per_method_metrics.items():
        for metric in METRICS:
            payload = evaluation["metrics"][metric]
            display = f"{payload['point_pm_ci_k']}K" if metric == "tok" else f"{payload['point_pm_ci_pct']}%"
            readme_lines.append(
                f"| {method} | {metric} | {payload['point']:.6f} | "
                f"[{payload['ci_lo']:.6f}, {payload['ci_hi']:.6f}] | {payload['point_pm_ci']} | {display} |"
            )
    readme_lines.extend(["", "## Paired Tests", "| method_a | method_b | metric | diff | 95% CI | bootstrap raw | McNemar raw | Wilcoxon raw |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for test in paired_tests:
        readme_lines.append(
            f"| {test['method_a']} | {test['method_b']} | {test['metric']} | {test['mean_diff']:.6f} | "
            f"[{test['diff_ci_lo']:.6f}, {test['diff_ci_hi']:.6f}] | {test.get('bootstrap_p_raw')} | "
            f"{test.get('mcnemar_p_raw')} | {test.get('wilcoxon_p_raw')} |"
        )
    readme_lines.extend(["", "## Alignment", *[
        f"- {method}: {evaluation['livebench_alignment']}" for method, evaluation in per_method_metrics.items()
    ], ""])
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LiveCodeBench bootstrap confidence intervals and paired tests.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--method", action="append", type=parse_method_spec, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--comparison-mode", choices=("reference", "all-pairs"), default="reference")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--B", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alternative", choices=("greater", "less", "two-sided"), default="greater")
    parser.add_argument("--paired-metrics", type=parse_metric_list, default=METRICS)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
