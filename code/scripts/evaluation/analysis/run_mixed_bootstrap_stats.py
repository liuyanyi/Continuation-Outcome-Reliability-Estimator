#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from cognitive_probe.evaluation.stats import (
    COMPARISON_MODES,
    PAIRED_METRICS,
    PER_METHOD_METRICS,
    V2_REPORT_FIELDS,
    _add_ci_display_fields,
    _dump_json,
    _git_commit,
    _strip_per_problem,
    build_paired_tests,
    bootstrap_ci,
    combine_stouffer_pvalues,
    default_metric_alternatives,
    evaluate_method as evaluate_math_method,
    stratified_paired_test,
)

from run_livebench_bootstrap_stats import evaluate_method as evaluate_livebench_method

METHOD_SET_MODES = ("exact", "common")


def parse_input_dir(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError("Dataset label cannot be empty.")
        return label, Path(path.strip())
    path = Path(raw.strip())
    if not str(path):
        raise argparse.ArgumentTypeError("Input directory cannot be empty.")
    return path.name, path


def parse_metric_list(raw: str) -> tuple[str, ...]:
    metrics = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not metrics:
        raise argparse.ArgumentTypeError("At least one metric is required.")
    unknown = [metric for metric in metrics if metric not in PAIRED_METRICS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown metric(s): {', '.join(unknown)}. Expected a comma-separated subset of {PAIRED_METRICS}."
        )
    return metrics


def parse_metric_alternative(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--metric-alternative must use METRIC=ALTERNATIVE.")
    metric, alternative = raw.split("=", 1)
    metric = metric.strip()
    alternative = alternative.strip()
    if metric not in PAIRED_METRICS:
        raise argparse.ArgumentTypeError(f"Unknown metric {metric!r}; expected one of {PAIRED_METRICS}.")
    if alternative not in {"greater", "less", "two-sided"}:
        raise argparse.ArgumentTypeError("Alternative must be one of: greater, less, two-sided.")
    return metric, alternative


def parse_weights_mode(raw: str) -> str:
    if raw not in {"none", "n"}:
        raise argparse.ArgumentTypeError("mix weights mode must be 'none' or 'n'.")
    return raw


def parse_method_set_mode(raw: str) -> str:
    if raw not in METHOD_SET_MODES:
        raise argparse.ArgumentTypeError(f"method set mode must be one of: {', '.join(METHOD_SET_MODES)}.")
    return raw


def _load_summary(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary.json under {input_dir}.")
    return json.loads(path.read_text(encoding="utf-8"))


def _is_livebench_summary(summary: dict[str, Any]) -> bool:
    dataset = str(summary.get("dataset") or "").lower()
    input_datasets = " ".join(str(item).lower() for item in summary.get("input_datasets") or [])
    return "livecodebench" in dataset or "livebench" in dataset or "livecodebench" in input_datasets


def _evaluate_method_for_summary(
    summary: dict[str, Any],
    method: str,
    run_dir: Path,
    k: int,
    B: int,
    seed: int,
) -> dict[str, Any]:
    if _is_livebench_summary(summary):
        return evaluate_livebench_method(method, run_dir, k, B=B, seed=seed)
    return evaluate_math_method(method, run_dir, k, B=B, seed=seed)


def _alignment_payload(evaluation: dict[str, Any]) -> dict[str, Any]:
    alignment = evaluation.get("v2_alignment")
    if isinstance(alignment, dict):
        return alignment
    alignment = evaluation.get("livebench_alignment")
    if isinstance(alignment, dict):
        return alignment
    return {"passed": False, "error": "missing_alignment_payload"}


def _concat_method_evaluations(
    *,
    dataset_evaluations: list[tuple[str, dict[str, dict[str, Any]]]],
    methods: list[str],
    reference_method: str,
    k: int | str,
    B: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for method_idx, method in enumerate(methods):
        sample_ids: list[str] = []
        per_problem: dict[str, list[np.ndarray]] = {metric: [] for metric in PER_METHOD_METRICS}
        diagnostics_by_dataset: dict[str, Any] = {}
        run_dirs_by_dataset: dict[str, str] = {}

        for dataset_label, evaluations in dataset_evaluations:
            if method not in evaluations:
                raise ValueError(f"Method {method!r} is missing from dataset {dataset_label!r}.")
            evaluation = evaluations[method]
            run_dirs_by_dataset[dataset_label] = evaluation["run_dir"]
            diagnostics_by_dataset[dataset_label] = evaluation["diagnostics"]
            sample_ids.extend(f"{dataset_label}::{sample_id}" for sample_id in evaluation["sample_ids"])
            for metric in PER_METHOD_METRICS:
                per_problem[metric].append(np.asarray(evaluation["per_problem"][metric], dtype=np.float64))

        metric_arrays = {
            metric: np.concatenate(parts).astype(np.float64, copy=False)
            for metric, parts in per_problem.items()
        }
        metrics: dict[str, dict[str, Any]] = {}
        for metric_idx, metric in enumerate(PER_METHOD_METRICS):
            metric_ci = bootstrap_ci(metric_arrays[metric], B=B, seed=seed + method_idx * 1009 + metric_idx)
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
            metric_ci["v2_report_field"] = V2_REPORT_FIELDS[metric]
            metrics[metric] = metric_ci

        combined[method] = {
            "method": method,
            "run_dir": ";".join(run_dirs_by_dataset[dataset_label] for dataset_label, _ in dataset_evaluations),
            "run_dirs_by_dataset": run_dirs_by_dataset,
            "k": k,
            "metrics": metrics,
            "diagnostics": {
                "n_problems": len(sample_ids),
                "datasets": diagnostics_by_dataset,
            },
            "v2_alignment": {
                "passed": all(
                    _alignment_payload(evaluations[method]).get("passed")
                    for _, evaluations in dataset_evaluations
                ),
                "datasets": {
                    dataset_label: _alignment_payload(evaluations[method])
                    for dataset_label, evaluations in dataset_evaluations
                },
            },
            "sample_ids": sample_ids,
            "per_problem": metric_arrays,
        }

    reference_sample_ids = combined[reference_method]["sample_ids"]
    for method, evaluation in combined.items():
        if evaluation["sample_ids"] != reference_sample_ids:
            raise ValueError(f"sample_id mismatch between {method} and reference {reference_method}.")
    return combined


def _build_dataset_pairs(
    dataset_evaluations: list[tuple[str, dict[str, dict[str, Any]]]],
    *,
    methods: list[str],
    reference_method: str,
    paired_metrics: tuple[str, ...],
    metric_alternatives: dict[str, str],
    comparison_mode: str,
    B: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    per_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset_idx, (dataset_label, evaluations) in enumerate(dataset_evaluations):
        if reference_method not in evaluations:
            raise ValueError(f"Reference method {reference_method!r} missing in dataset {dataset_label!r}.")
        per_dataset[dataset_label] = build_paired_tests(
            evaluations,
            methods=methods,
            reference_method=reference_method,
            paired_metrics=paired_metrics,
            metric_alternatives=metric_alternatives,
            comparison_mode=comparison_mode,
            B=B,
            seed=seed + dataset_idx * 10007,
        )
    return per_dataset


def _write_per_method_csv(path: Path, per_method_metrics: dict[str, Any]) -> None:
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
        "v2_report_field",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, evaluation in per_method_metrics.items():
            for metric, metric_payload in evaluation["metrics"].items():
                writer.writerow(
                    {
                        "method": method,
                        "run_dir": evaluation["run_dir"],
                        "k": evaluation["k"],
                        "metric": metric,
                        **{key: metric_payload.get(key) for key in fieldnames if key in metric_payload},
                    }
                )


def _write_paired_csv(path: Path, tests: list[dict[str, Any]], k: int) -> None:
    fieldnames = [
        "method_a",
        "method_b",
        "k",
        "metric",
        "family",
        "n_problems",
        "mean_diff",
        "diff_ci_lo",
        "diff_ci_hi",
        "diff_ci_half_width",
        "diff_pm_ci",
        "diff_pm_ci_pct",
        "diff_pm_ci_k",
        "dataset_p_values",
        "wilcoxon_stat",
        "wilcoxon_p_raw",
        "wilcoxon_p_holm_method",
        "wilcoxon_p_holm_metric",
        "wilcoxon_p_holm_global",
        "wilcoxon_p_holm_method_reject",
        "wilcoxon_p_holm_metric_reject",
        "wilcoxon_p_holm_global_reject",
        "bootstrap_p_raw",
        "bootstrap_p_holm_method",
        "bootstrap_p_holm_metric",
        "bootstrap_p_holm_global",
        "bootstrap_p_holm_method_reject",
        "bootstrap_p_holm_metric_reject",
        "bootstrap_p_holm_global_reject",
        "stouffer_stat",
        "stouffer_p_raw",
        "stouffer_n",
        "stouffer_weights",
        "stouffer_base_p_field",
        "n_problems_by_stratum",
        "ci_method",
        "warnings",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for test in tests:
            row = {key: test.get(key) for key in fieldnames}
            row["k"] = k
            for key in ["dataset_p_values", "stouffer_weights", "n_problems_by_stratum"]:
                if row.get(key) is not None:
                    row[key] = json.dumps(row[key], ensure_ascii=False)
            row["warnings"] = ";".join(test.get("warnings") or [])
            writer.writerow(row)


def _write_readme(
    path: Path,
    *,
    command: str,
    dataset_name: str,
    input_summaries: list[dict[str, Any]],
    k: int | str,
    k_by_dataset: dict[str, int],
    B: int,
    seed: int,
    metrics: tuple[str, ...],
    metric_alternatives: dict[str, str],
    comparison_mode: str,
    mix_mode: str,
    mix_weights_mode: str,
    method_set_mode: str,
    dropped_methods_by_dataset: dict[str, list[str]],
    git_commit: str | None,
    per_method_metrics: dict[str, Any],
    paired_tests: list[dict[str, Any]],
) -> None:
    import scipy

    input_lines = [
        f"- {summary['dataset']}: n={summary['n_problems']}, k={summary['k']}, source={summary['source_dir']}"
        for summary in input_summaries
    ]
    dropped_lines = [
        f"- {dataset}: {', '.join(methods)}"
        for dataset, methods in sorted(dropped_methods_by_dataset.items())
        if methods
    ]
    if not dropped_lines:
        dropped_lines = ["- none"]
    metric_lines = [
        "| method | metric | point | 95% CI | point +- half-width | percent / K-tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method, evaluation in per_method_metrics.items():
        for metric in PER_METHOD_METRICS:
            payload = evaluation["metrics"][metric]
            display_value = (
                f"{payload['point_pm_ci_k']}K"
                if metric == "tok"
                else f"{payload['point_pm_ci_pct']}%"
            )
            metric_lines.append(
                f"| {method} | {metric} | {payload['point']:.6f} | "
                f"[{payload['ci_lo']:.6f}, {payload['ci_hi']:.6f}] | "
                f"{payload['point_pm_ci']} | {display_value} |"
            )

    if mix_mode == "stouffer":
        paired_lines = [
            "| method_a | method_b | metric | family | datasets | base p | Stouffer stat | Stouffer p |",
            "|---|---|---|---|---|---:|---:|---:|",
        ]
        for test in paired_tests:
            if test["metric"] != "maj":
                continue
            paired_lines.append(
                f"| {test['method_a']} | {test['method_b']} | {test['metric']} | {test['family']} | "
                f"{len(test.get('dataset_p_values') or [])} | {test.get('stouffer_base_p_field')} | "
                f"{test.get('stouffer_stat')} | {test.get('stouffer_p_raw')} |"
            )
    elif mix_mode == "stratified":
        paired_lines = [
            "| method_a | method_b | metric | family | n | diff | 95% CI | bootstrap raw |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
        for test in paired_tests:
            if test["metric"] != "maj":
                continue
            paired_lines.append(
                f"| {test['method_a']} | {test['method_b']} | {test['metric']} | {test['family']} | "
                f"{test['n_problems']} | {test['mean_diff']:.6f} | "
                f"[{test['diff_ci_lo']:.6f}, {test['diff_ci_hi']:.6f}] | "
                f"{test.get('bootstrap_p_raw')} |"
            )
    else:
        paired_lines = [
            "| method_a | method_b | metric | family | diff | 95% CI | bootstrap raw | McNemar raw | Wilcoxon raw |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for test in paired_tests:
            if test["metric"] != "maj":
                continue
            paired_lines.append(
                f"| {test['method_a']} | {test['method_b']} | {test['metric']} | {test['family']} | "
                f"{test['mean_diff']:.6f} | [{test['diff_ci_lo']:.6f}, {test['diff_ci_hi']:.6f}] | "
                f"{test.get('bootstrap_p_raw')} | {test.get('mcnemar_p_raw')} | {test.get('wilcoxon_p_raw')} |"
            )

    warning_lines: list[str] = []
    for test in paired_tests:
        for warning in test.get("warnings") or []:
            warning_lines.append(f"- {test['method_a']} vs {test['method_b']} {test['metric']}: {warning}")
    if not warning_lines:
        warning_lines.append("- none")

    content = "\n".join(
        [
            f"# Mixed Bootstrap Stats: {dataset_name}",
            "",
            "## Command",
            f"`{command}`",
            "",
            "## Inputs",
            *input_lines,
            "",
            "## Parameters",
            f"- k: {k}",
            f"- k_by_dataset: {', '.join(f'{dataset}={dataset_k}' for dataset, dataset_k in k_by_dataset.items())}",
            f"- B: {B}",
            f"- seed: {seed}",
            f"- mix_mode: {mix_mode}",
            f"- mix_weights_mode: {mix_weights_mode}",
            f"- method_set_mode: {method_set_mode}",
            f"- comparison_mode: {comparison_mode}",
            f"- paired_metrics: {', '.join(metrics)}",
            f"- metric_alternatives: {', '.join(f'{metric}={metric_alternatives[metric]}' for metric in metrics)}",
            f"- holm_families: method, metric, global",
            "- paired_rows: Maj@K only",
            f"- git_commit: {git_commit or 'unknown'}",
            f"- numpy: {np.__version__}",
            f"- scipy: {scipy.__version__}",
            "",
            "## Per-Method Metrics",
            *metric_lines,
            "",
            "## Paired Tests",
            *paired_lines,
            "",
            "## Warnings",
            *warning_lines,
            "",
            "## Dropped Methods",
            *dropped_lines,
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def run_mixed_bootstrap_evaluation(
    *,
    inputs: list[tuple[str, Path]],
    output_dir: Path,
    dataset_name: str,
    reference_method: str | None = None,
    k: int | None = None,
    B: int = 10000,
    seed: int = 0,
    alternative: str = "greater",
    paired_metrics: tuple[str, ...] = PAIRED_METRICS,
    metric_alternatives: dict[str, str] | None = None,
    comparison_mode: str = "reference",
    mix_mode: str = "concat",
    mix_weights_mode: str = "n",
    method_set_mode: str = "exact",
    command: str | None = None,
) -> None:
    if not inputs:
        raise ValueError("At least one input bootstrap directory is required.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    dataset_evaluations: list[tuple[str, dict[str, dict[str, Any]]]] = []
    k_by_dataset: dict[str, int] = {}

    for dataset_idx, (dataset_label, input_dir) in enumerate(inputs):
        input_dir = input_dir.resolve()
        summary = _load_summary(input_dir)
        summary["source_dir"] = str(input_dir)
        summaries.append(summary)

        dataset_k = int(summary["k"])
        k_by_dataset[dataset_label] = dataset_k
        if k is not None and dataset_k != k:
            raise ValueError(f"k mismatch for {input_dir}: summary has {dataset_k}, expected {k}.")

        dataset_reference = str(summary["reference_method"])
        if reference_method is None:
            reference_method = dataset_reference
        elif dataset_reference != reference_method:
            raise ValueError(
                f"reference_method mismatch for {input_dir}: summary has {dataset_reference}, expected {reference_method}."
            )

        evaluations: dict[str, dict[str, Any]] = {}
        for method_idx, method in enumerate(summary["methods"]):
            run_dir = Path(summary["method_runs"][method]["run_dir"]).resolve()
            evaluations[method] = _evaluate_method_for_summary(
                summary,
                method,
                run_dir,
                dataset_k,
                B=B,
                seed=seed + dataset_idx * 1000003 + method_idx * 1009,
            )
        dataset_evaluations.append((dataset_label, evaluations))

    if reference_method is None:
        raise ValueError("Failed to infer reference_method from inputs.")
    k_values = list(k_by_dataset.values())
    output_k: int | str
    if k is not None:
        output_k = k
    elif len(set(k_values)) == 1:
        output_k = k_values[0]
    else:
        output_k = "mixed"
    metric_alternatives = default_metric_alternatives(
        paired_metrics,
        default_alternative=alternative,
        overrides=metric_alternatives,
    )

    method_sets = [set(evaluations.keys()) for _, evaluations in dataset_evaluations]
    common_methods = set.intersection(*method_sets)
    if reference_method not in common_methods:
        raise ValueError(f"Reference method {reference_method!r} is not present in every input.")
    all_methods = set.union(*method_sets)
    dropped_methods_by_dataset = {
        dataset_label: sorted(set(evaluations.keys()) - common_methods)
        for dataset_label, evaluations in dataset_evaluations
        if set(evaluations.keys()) - common_methods
    }
    missing_by_dataset = {
        dataset_label: sorted(all_methods - set(evaluations.keys()))
        for dataset_label, evaluations in dataset_evaluations
        if all_methods - set(evaluations.keys())
    }
    if method_set_mode == "exact" and (dropped_methods_by_dataset or missing_by_dataset):
        raise ValueError(
            "Inputs must contain the same methods. "
            f"Use --method-set common to keep only the intersection. "
            f"Extra methods: {dropped_methods_by_dataset}; missing methods: {missing_by_dataset}"
        )

    first_methods = list(summaries[0]["methods"])
    methods = [method for method in first_methods if method in common_methods]
    combined = _concat_method_evaluations(
        dataset_evaluations=dataset_evaluations,
        methods=methods,
        reference_method=reference_method,
        k=output_k,
        B=B,
        seed=seed,
    )
    per_method_metrics = {
        method: _strip_per_problem(evaluation)
        for method, evaluation in combined.items()
    }

    if mix_mode == "concat":
        paired_tests = build_paired_tests(
            combined,
            methods=methods,
            reference_method=reference_method,
            paired_metrics=paired_metrics,
            metric_alternatives=metric_alternatives,
            comparison_mode=comparison_mode,
            B=B,
            seed=seed,
        )
    elif mix_mode == "stouffer":
        dataset_pairs = _build_dataset_pairs(
            dataset_evaluations,
            methods=methods,
            reference_method=reference_method,
            paired_metrics=paired_metrics,
            metric_alternatives=metric_alternatives,
            comparison_mode=comparison_mode,
            B=B,
            seed=seed,
        )
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for dataset_label, tests in dataset_pairs.items():
            for test in tests:
                grouped[(test["method_a"], test["method_b"], test["metric"])].append(
                    {
                        **test,
                        "dataset": dataset_label,
                    }
                )

        n_by_input_label = {
            dataset_label: int(summary["n_problems"])
            for dataset_label, summary in zip(
                (dataset_label for dataset_label, _ in dataset_evaluations),
                summaries,
                strict=True,
            )
        }
        paired_tests = []
        for (method_a, method_b, metric), tests in grouped.items():
            base_p_field = "wilcoxon_p_raw"
            strata = [str(test["dataset"]) for test in tests]
            p_values = [float(test[base_p_field]) for test in tests]
            weight_values = (
                [float(n_by_input_label[stratum]) for stratum in strata]
                if mix_weights_mode == "n"
                else None
            )
            meta = combine_stouffer_pvalues(p_values, weights=weight_values)
            paired_tests.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "family": tests[0]["family"],
                    "dataset_p_values": p_values,
                    "dataset_p_values_by_stratum": dict(zip(strata, p_values, strict=True)),
                    "stouffer_stat": meta["stat"],
                    "stouffer_p_raw": meta["p_raw"],
                    "stouffer_n": meta["n"],
                    "stouffer_weights": meta["weights"],
                    "stouffer_weights_by_stratum": (
                        None if weight_values is None else dict(zip(strata, weight_values, strict=True))
                    ),
                    "n_problems_by_stratum": n_by_input_label,
                    "stouffer_base_p_field": base_p_field,
                    "warnings": [],
                }
            )
    else:
        pair_specs: list[tuple[str, str]] = []
        if comparison_mode == "reference":
            pair_specs.extend((method, reference_method) for method in methods if method != reference_method)
        else:
            pair_specs.extend(
                (method_a, method_b)
                for method_a in methods
                for method_b in methods
                if method_a != method_b
            )

        paired_tests = []
        metrics_by_dataset = {
            dataset_label: evaluations
            for dataset_label, evaluations in dataset_evaluations
        }
        pair_seed = seed + 100000
        for pair_idx, (method_a, method_b) in enumerate(pair_specs):
            for metric_idx, metric in enumerate(paired_metrics):
                test = stratified_paired_test(
                    {
                        dataset_label: evaluations[method_a]
                        for dataset_label, evaluations in metrics_by_dataset.items()
                    },
                    {
                        dataset_label: evaluations[method_b]
                        for dataset_label, evaluations in metrics_by_dataset.items()
                    },
                    method_a,
                    method_b,
                    metric,
                    B=B,
                    seed=pair_seed + pair_idx * 1009 + metric_idx,
                    alternative=metric_alternatives[metric],
                )
                test["family"] = f"{method_a}_vs_{method_b}"
                paired_tests.append(test)

        from cognitive_probe.evaluation.stats import add_multi_holm_corrections

        paired_tests = add_multi_holm_corrections(paired_tests)

    n_by_dataset = {summary["dataset"]: int(summary["n_problems"]) for summary in summaries}
    methods_by_dataset = {summary["dataset"]: summary["methods"] for summary in summaries}
    git_commit = _git_commit()
    run_timestamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    summary_payload = {
        "dataset": dataset_name,
        "input_datasets": [summary["dataset"] for summary in summaries],
        "n_problems_by_dataset": n_by_dataset,
        "k": output_k,
        "k_by_dataset": k_by_dataset,
        "n_problems": sum(n_by_dataset.values()),
        "B": B,
        "seed": seed,
        "methods": methods,
        "methods_by_dataset": methods_by_dataset,
        "method_set_mode": method_set_mode,
        "dropped_methods_by_dataset": dropped_methods_by_dataset,
        "missing_methods_by_dataset": missing_by_dataset,
        "reference_method": reference_method,
        "comparison_mode": comparison_mode,
        "mix_mode": mix_mode,
        "mix_weights_mode": mix_weights_mode,
        "paired_metrics": list(paired_metrics),
        "metric_alternatives": metric_alternatives,
        "holm_families": {
            "method": sorted({t["family"] for t in paired_tests}),
            "metric": list(paired_metrics),
            "global": len(paired_tests),
        },
        "run_timestamp": run_timestamp,
        "git_commit": git_commit,
        "input_summaries": summaries,
        "files": {
            "per_method_metrics": "per_method_metrics.json",
            "paired_tests": "paired_tests.json",
            "per_dataset_paired_tests": "per_dataset_paired_tests.json",
            "per_method_metrics_csv": "per_method_metrics.csv",
            "paired_tests_csv": "paired_tests.csv",
            "diagnostics": "diagnostics.json",
            "readme": "README.md",
        },
    }

    wilcoxon_warnings = [
        {
            "method_a": test["method_a"],
            "method_b": test["method_b"],
            "metric": test["metric"],
            "warnings": test.get("warnings") or [],
        }
        for test in paired_tests
        if test.get("warnings")
    ]
    method_dataset_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for method, evaluation in combined.items():
        for dataset_label, diagnostics in evaluation["diagnostics"]["datasets"].items():
            method_dataset_counts[method][dataset_label] = int(diagnostics["n_problems"])
    diagnostics_payload = {
        "dataset": dataset_name,
        "k": output_k,
        "k_by_dataset": k_by_dataset,
        "n_problems": summary_payload["n_problems"],
        "n_problems_by_dataset": n_by_dataset,
        "per_method_dataset_counts": method_dataset_counts,
        "wilcoxon_warnings": wilcoxon_warnings,
        "v2_alignment_failures": [
            {
                "method": method,
                "dataset": dataset_label,
                "alignment": alignment,
            }
            for method, evaluation in per_method_metrics.items()
            for dataset_label, alignment in evaluation["v2_alignment"]["datasets"].items()
            if not alignment.get("passed")
        ],
    }
    per_dataset_paired_tests = {
        summary["dataset"]: {
            "source_dir": summary["source_dir"],
            "paired_tests_file": "paired_tests.json",
            "tests": json.loads((Path(summary["source_dir"]) / "paired_tests.json").read_text(encoding="utf-8"))[
                "tests"
            ],
        }
        for summary in summaries
    }
    paired_payload = {
        "reference_method": reference_method,
        "alternative": alternative,
        "metric_alternatives": metric_alternatives,
        "comparison_mode": comparison_mode,
        "mix_mode": mix_mode,
        "mix_weights_mode": mix_weights_mode,
        "k": output_k,
        "k_by_dataset": k_by_dataset,
        "holm_family_size": len(paired_tests),
        "holm_family_sizes": {
            "method": len({t["family"] for t in paired_tests}),
            "metric": len(paired_metrics),
            "global": len(paired_tests),
        },
        "holm_alpha": 0.05,
        "paired_metrics": list(paired_metrics),
        "tests": paired_tests,
    }

    _dump_json(output_dir / "summary.json", summary_payload)
    _dump_json(output_dir / "per_method_metrics.json", per_method_metrics)
    _dump_json(output_dir / "paired_tests.json", paired_payload)
    _dump_json(output_dir / "per_dataset_paired_tests.json", per_dataset_paired_tests)
    _dump_json(output_dir / "diagnostics.json", diagnostics_payload)
    _write_per_method_csv(output_dir / "per_method_metrics.csv", per_method_metrics)
    _write_paired_csv(output_dir / "paired_tests.csv", paired_tests, output_k)
    _write_readme(
        output_dir / "README.md",
        command=command or " ".join(shlex.quote(arg) for arg in sys.argv),
        dataset_name=dataset_name,
        input_summaries=summaries,
        k=output_k,
        k_by_dataset=k_by_dataset,
        B=B,
        seed=seed,
        metrics=paired_metrics,
        metric_alternatives=metric_alternatives,
        comparison_mode=comparison_mode,
        mix_mode=mix_mode,
        mix_weights_mode=mix_weights_mode,
        method_set_mode=method_set_mode,
        dropped_methods_by_dataset=dropped_methods_by_dataset,
        git_commit=git_commit,
        per_method_metrics=per_method_metrics,
        paired_tests=paired_tests,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run mixed bootstrap confidence intervals and paired tests by concatenating "
            "per-problem metrics from multiple existing bootstrap result directories."
        )
    )
    parser.add_argument("--dataset-name", required=True, help="Mixed dataset/run label to write into output artifacts.")
    parser.add_argument(
        "--input",
        action="append",
        type=parse_input_dir,
        required=True,
        help="Bootstrap result dir, optionally LABEL=DIR. Repeat for each dataset.",
    )
    parser.add_argument("--reference", default=None, help="Reference method name. Defaults to the input summaries.")
    parser.add_argument(
        "--comparison-mode",
        choices=COMPARISON_MODES,
        default="reference",
        help="reference compares every method to --reference; all-pairs emits both directions for every method pair.",
    )
    parser.add_argument(
        "--mix-mode",
        choices=["concat", "stouffer", "stratified"],
        default="concat",
        help=(
            "concat concatenates per-problem arrays; stratified resamples within each input dataset; "
            "stouffer combines per-dataset p-values."
        ),
    )
    parser.add_argument(
        "--mix-weights",
        choices=["none", "n"],
        default="n",
        help="Weight mode for Stouffer meta-analysis. n uses dataset size, none uses equal weights.",
    )
    parser.add_argument(
        "--method-set",
        type=parse_method_set_mode,
        default="exact",
        help="exact requires identical methods in every input; common keeps only the method intersection.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for mixed summary/statistics artifacts.")
    parser.add_argument("--k", type=int, default=None, help="Expected k. Defaults to the input summaries.")
    parser.add_argument("--B", type=int, default=10000, help="Bootstrap resample count.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for np.random.default_rng.")
    parser.add_argument(
        "--alternative",
        choices=["greater", "less", "two-sided"],
        default="greater",
        help="Alternative hypothesis for paired tests.",
    )
    parser.add_argument(
        "--paired-metrics",
        type=parse_metric_list,
        default=PAIRED_METRICS,
        help="Comma-separated paired-test metrics. Defaults to pass,mean,maj,tok.",
    )
    parser.add_argument(
        "--metric-alternative",
        action="append",
        type=parse_metric_alternative,
        default=[],
        help=(
            "Per-metric alternative override in METRIC=ALTERNATIVE form. "
            "By default tok uses less and other metrics use --alternative."
        ),
    )
    args = parser.parse_args()

    command = " ".join(shlex.quote(arg) for arg in sys.argv)
    run_mixed_bootstrap_evaluation(
        inputs=args.input,
        output_dir=Path(args.output_dir),
        dataset_name=args.dataset_name,
        reference_method=args.reference,
        k=args.k,
        B=args.B,
        seed=args.seed,
        alternative=args.alternative,
        paired_metrics=args.paired_metrics,
        metric_alternatives=dict(args.metric_alternative),
        comparison_mode=args.comparison_mode,
        mix_mode=args.mix_mode,
        mix_weights_mode=args.mix_weights,
        method_set_mode=args.method_set,
        command=command,
    )
    print(f"已写出 mixed bootstrap 统计结果：{Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
