from __future__ import annotations

import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
import warnings as py_warnings
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta, binomtest, combine_pvalues, wilcoxon

from cognitive_probe.evaluation.data_model import (
    FINAL_OUTPUTS_FILE,
    RUN_OVERVIEW_FILE,
    SOURCE_TARGETS_FILE,
    report_path_for_k,
)
from cognitive_probe.evaluation.v2 import (
    _extract_vote_answer,
    _grade_extracted_answer,
    _load_source_targets,
    _majority_vote_answer,
    _select_outputs_by_k,
    compute_eval_report,
    load_final_outputs,
    load_generation_segments,
    load_report,
    load_run_overview,
)
from cognitive_probe.utils.math_utils import grade_answer_verl

METRICS = ("pass", "mean", "maj")
PER_METHOD_METRICS = ("pass", "mean", "maj", "tok")
PAIRED_METRICS = PER_METHOD_METRICS
COMPARISON_MODES = ("reference", "all-pairs")
V2_REPORT_FIELDS = {
    "pass": "best_score_mean",
    "mean": "mean_at_k",
    "maj": "maj_at_k",
    "tok": "avg_final_visible_tokens",
}


def _format_pm(point: float, half_width: float, scale: float = 1.0, digits: int = 4) -> str:
    return f"{point * scale:.{digits}f} +- {half_width * scale:.{digits}f}"


def _add_ci_display_fields(
    payload: dict[str, Any],
    *,
    point_key: str,
    lo_key: str,
    hi_key: str,
    prefix: str,
    include_percent: bool = True,
    include_kilo: bool = False,
) -> None:
    point = float(payload[point_key])
    lo = float(payload[lo_key])
    hi = float(payload[hi_key])
    half_width = (hi - lo) / 2.0
    payload[f"{prefix}_ci_half_width"] = half_width
    payload[f"{prefix}_pm_ci"] = _format_pm(point, half_width)
    if include_percent:
        payload[f"{prefix}_pm_ci_pct"] = _format_pm(point, half_width, scale=100.0, digits=2)
    if include_kilo:
        payload[f"{prefix}_pm_ci_k"] = _format_pm(point, half_width, scale=1.0 / 1000.0, digits=2)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def _mean(values: np.ndarray) -> float:
    if values.size == 0:
        raise ValueError("Cannot compute statistics for an empty problem array.")
    return float(np.mean(values))


def _bootstrap_means(values: np.ndarray, B: int, seed: int) -> np.ndarray:
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}.")
    n = int(values.size)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty array.")

    rng = np.random.default_rng(seed)
    out = np.empty(B, dtype=np.float64)
    chunk_size = max(1, min(B, 2048))
    offset = 0
    while offset < B:
        chunk = min(chunk_size, B - offset)
        indices = rng.integers(0, n, size=(chunk, n))
        out[offset : offset + chunk] = values[indices].mean(axis=1)
        offset += chunk
    return out


def _is_binary(values: np.ndarray) -> bool:
    return bool(np.all((values == 0.0) | (values == 1.0)))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_default(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(item) for item in value]
    return value


def _dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_default(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _metric_values(metrics: dict[str, Any], metric: str) -> np.ndarray:
    values = metrics.get(metric)
    if not isinstance(values, np.ndarray):
        raise TypeError(f"Expected numpy array for metric {metric!r}.")
    return values.astype(np.float64, copy=False)


def _lineage_visible_tokens(segment_id: str, segment_map: dict[str, Any]) -> int:
    total = 0
    current = segment_map.get(segment_id)
    while current is not None:
        total += int(current.visible_output_tokens)
        parent_id = current.parent_segment_id
        current = segment_map.get(parent_id) if parent_id else None
    return total


def compute_per_problem_metrics_from_run(run_dir: Path, k: int) -> dict[str, Any]:
    """
    Recompute V2-compatible per-problem pass / mean / maj arrays from run artifacts.
    """
    run_dir = Path(run_dir).resolve()
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    final_outputs = load_final_outputs(run_dir)
    segments = load_generation_segments(run_dir)
    source_targets = _load_source_targets(run_dir)
    selected_outputs = _select_outputs_by_k(final_outputs, k)

    segment_map = {segment.segment_id: segment for segment in segments}
    answer_by_source_row = {item.source_row_index: item.answer for item in source_targets}

    grouped_scores: dict[str, list[int]] = defaultdict(list)
    grouped_vote_answers: dict[str, list[str]] = defaultdict(list)
    grouped_visible_tokens: dict[str, list[int]] = defaultdict(list)
    sample_ground_truths: dict[str, str] = {}
    selected_counts: dict[str, int] = defaultdict(int)
    n_outputs_pruned = 0
    n_outputs_vote_eligible = 0

    for output in selected_outputs:
        sample_id = output.sample_id
        selected_counts[sample_id] += 1
        if output.was_pruned:
            n_outputs_pruned += 1
        else:
            n_outputs_vote_eligible += 1

        source_row_idx = output.source_row_index
        if source_row_idx is None:
            final_segment = segment_map.get(output.final_segment_id)
            if final_segment is not None:
                source_row_idx = final_segment.source_row_index
        gt = answer_by_source_row.get(source_row_idx, "") if source_row_idx is not None else ""
        if gt:
            sample_ground_truths.setdefault(sample_id, gt)

        score = 0 if output.was_pruned else (1 if grade_answer_verl(output.response, gt) else 0)
        grouped_scores[sample_id].append(score)

        if not output.was_pruned:
            vote_answer = _extract_vote_answer(output.response)
            if vote_answer is not None:
                grouped_vote_answers[sample_id].append(vote_answer)
        grouped_visible_tokens[sample_id].append(_lineage_visible_tokens(output.final_segment_id, segment_map))

    sample_ids = sorted(grouped_scores.keys())
    if not sample_ids:
        raise ValueError(f"No selected outputs found under {run_dir}.")

    pass_values: list[float] = []
    mean_values: list[float] = []
    maj_values: list[float] = []
    tok_values: list[float] = []
    n_problems_without_vote_answer = 0

    for sample_id in sample_ids:
        scores = grouped_scores[sample_id]
        if len(scores) != k:
            raise ValueError(f"Sample {sample_id} has {len(scores)} selected outputs; expected k={k}.")
        vote_answers = grouped_vote_answers.get(sample_id, [])
        maj_answer = _majority_vote_answer(vote_answers)
        if maj_answer is None:
            n_problems_without_vote_answer += 1
        mean_values.append(float(sum(scores) / len(scores)))
        pass_values.append(float(max(scores)))
        maj_values.append(
            1.0 if _grade_extracted_answer(maj_answer, sample_ground_truths.get(sample_id, "")) else 0.0
        )
        visible_tokens = grouped_visible_tokens.get(sample_id, [])
        if len(visible_tokens) != k:
            raise ValueError(f"Sample {sample_id} has {len(visible_tokens)} visible-token values; expected k={k}.")
        tok_values.append(float(sum(visible_tokens) / len(visible_tokens)))

    counts = list(selected_counts.values())
    return {
        "run_dir": str(run_dir),
        "k": k,
        "sample_ids": sample_ids,
        "pass": np.asarray(pass_values, dtype=np.float64),
        "mean": np.asarray(mean_values, dtype=np.float64),
        "maj": np.asarray(maj_values, dtype=np.float64),
        "tok": np.asarray(tok_values, dtype=np.float64),
        "diagnostics": {
            "n_problems": len(sample_ids),
            "n_outputs_selected": len(selected_outputs),
            "n_outputs_pruned": n_outputs_pruned,
            "n_outputs_vote_eligible": n_outputs_vote_eligible,
            "n_problems_without_vote_answer": n_problems_without_vote_answer,
            "responses_per_item_min": min(counts),
            "responses_per_item_max": max(counts),
        },
    }


def validate_against_eval_report(
    run_dir: Path,
    k: int,
    per_problem_metrics: dict[str, Any],
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """
    Validate recomputed point estimates against eval_report_v2_{k}.json.
    """
    run_dir = Path(run_dir).resolve()
    report = load_report(run_dir, k)
    if report is None:
        report = compute_eval_report(run_dir, k)

    errors: dict[str, float] = {}
    points = {
        "mean": _mean(_metric_values(per_problem_metrics, "mean")),
        "pass": _mean(_metric_values(per_problem_metrics, "pass")),
        "maj": _mean(_metric_values(per_problem_metrics, "maj")),
        "tok": _mean(_metric_values(per_problem_metrics, "tok")),
    }
    expected = {
        "mean": float(report.mean_at_k),
        "pass": float(report.best_score_mean),
        "maj": float(report.maj_at_k),
        "tok": float(report.avg_final_visible_tokens),
    }
    for metric in PER_METHOD_METRICS:
        errors[f"{metric}_abs_error"] = abs(points[metric] - expected[metric])

    passed = all(value <= tolerance for value in errors.values())
    alignment = {
        "eval_report": report_path_for_k(run_dir, k).name,
        **errors,
        "passed": passed,
    }
    if not passed:
        raise ValueError(
            "V2 alignment failed for "
            f"{run_dir} k={k}: "
            + ", ".join(f"{key}={value:.12g}" for key, value in errors.items())
        )
    return alignment


def bootstrap_ci(
    per_problem_values: np.ndarray,
    B: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
    allow_jeffreys: bool = True,
) -> dict[str, Any]:
    """
    Single-level problem bootstrap CI for a per-problem metric array.
    """
    values = np.asarray(per_problem_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1-D array, got shape {values.shape}.")
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Bootstrap input contains NaN or Inf.")
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}.")

    point = _mean(values)
    if allow_jeffreys and _is_binary(values) and (point == 0.0 or point == 1.0):
        successes = float(values.sum())
        failures = float(values.size - successes)
        lo = _finite_float(beta.ppf(alpha / 2.0, successes + 0.5, failures + 0.5))
        hi = _finite_float(beta.ppf(1.0 - alpha / 2.0, successes + 0.5, failures + 0.5))
        payload = {
            "point": point,
            "ci_lo": 0.0 if lo is None else lo,
            "ci_hi": 1.0 if hi is None else hi,
            "n": int(values.size),
            "B": int(B),
            "ci_method": "jeffreys_beta",
        }
        _add_ci_display_fields(payload, point_key="point", lo_key="ci_lo", hi_key="ci_hi", prefix="point")
        return payload

    boot_means = _bootstrap_means(values, B, seed)
    lo, hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    payload = {
        "point": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": int(values.size),
        "B": int(B),
        "ci_method": "percentile_bootstrap",
    }
    _add_ci_display_fields(payload, point_key="point", lo_key="ci_lo", hi_key="ci_hi", prefix="point")
    return payload


def evaluate_method(
    method: str,
    run_dir: Path,
    k: int,
    B: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Evaluate one method run and return V2-aligned point estimates plus CIs.
    """
    per_problem = compute_per_problem_metrics_from_run(run_dir, k)
    alignment = validate_against_eval_report(run_dir, k, per_problem)
    metrics: dict[str, dict[str, Any]] = {}
    for offset, metric in enumerate(PER_METHOD_METRICS):
        metric_ci = bootstrap_ci(_metric_values(per_problem, metric), B=B, seed=seed + offset)
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
    return {
        "method": method,
        "run_dir": str(Path(run_dir)),
        "k": k,
        "metrics": metrics,
        "diagnostics": per_problem["diagnostics"],
        "v2_alignment": alignment,
        "sample_ids": per_problem["sample_ids"],
        "per_problem": {
            metric: per_problem[metric] for metric in PER_METHOD_METRICS
        },
    }


def paired_test(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    method_a: str,
    method_b: str,
    metric: str,
    B: int = 10000,
    seed: int = 0,
    alternative: str = "greater",
) -> dict[str, Any]:
    """
    Paired bootstrap diff CI and Wilcoxon signed-rank test.
    """
    if metric not in PAIRED_METRICS:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {PAIRED_METRICS}.")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")
    if metrics_a["sample_ids"] != metrics_b["sample_ids"]:
        raise ValueError(f"sample_id mismatch between {method_a} and {method_b}.")

    vals_a = _metric_values(metrics_a["per_problem"], metric)
    vals_b = _metric_values(metrics_b["per_problem"], metric)
    diffs = vals_a - vals_b
    observed = _mean(diffs)

    boot_diffs = _bootstrap_means(diffs, B, seed)
    diff_ci_lo, diff_ci_hi = np.quantile(boot_diffs, [0.025, 0.975])

    warnings: list[str] = []
    non_zero_diffs = int(np.count_nonzero(diffs))
    if non_zero_diffs < 6:
        warnings.append("insufficient_non_zero_diffs_for_wilcoxon")

    wilcoxon_stat: float | None
    wilcoxon_p_raw: float | None
    try:
        with py_warnings.catch_warnings():
            py_warnings.simplefilter("ignore", RuntimeWarning)
            wilcoxon_result = wilcoxon(diffs, alternative=alternative, zero_method="wilcox")
        wilcoxon_stat = _finite_float(wilcoxon_result.statistic)
        wilcoxon_p_raw = _finite_float(wilcoxon_result.pvalue)
        if wilcoxon_p_raw is None:
            wilcoxon_stat = None
            warnings.append("wilcoxon_failed:non_finite_result")
    except ValueError as exc:
        wilcoxon_stat = None
        wilcoxon_p_raw = None
        warnings.append(f"wilcoxon_failed:{exc}")

    centered = diffs - observed
    null_boot = _bootstrap_means(centered, B, seed + 1)
    if alternative == "greater":
        bootstrap_p_raw = float(np.mean(null_boot >= observed))
    elif alternative == "less":
        bootstrap_p_raw = float(np.mean(null_boot <= observed))
    else:
        bootstrap_p_raw = float(np.mean(np.abs(null_boot) >= abs(observed)))

    mcnemar_n_a_only: int | None = None
    mcnemar_n_b_only: int | None = None
    mcnemar_n_discordant: int | None = None
    mcnemar_p_raw: float | None = None
    if _is_binary(vals_a) and _is_binary(vals_b):
        mcnemar_n_a_only = int(np.sum((vals_a == 1.0) & (vals_b == 0.0)))
        mcnemar_n_b_only = int(np.sum((vals_a == 0.0) & (vals_b == 1.0)))
        mcnemar_n_discordant = mcnemar_n_a_only + mcnemar_n_b_only
        if mcnemar_n_discordant == 0:
            mcnemar_p_raw = 1.0
        else:
            mcnemar_p_raw = float(
                binomtest(
                    mcnemar_n_a_only,
                    n=mcnemar_n_discordant,
                    p=0.5,
                    alternative=alternative,
                ).pvalue
            )

    payload = {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
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
    if metric == "tok":
        payload.pop("diff_pm_ci_pct", None)
        _add_ci_display_fields(
            payload,
            point_key="mean_diff",
            lo_key="diff_ci_lo",
            hi_key="diff_ci_hi",
            prefix="diff",
            include_percent=False,
            include_kilo=True,
        )
    else:
        _add_ci_display_fields(payload, point_key="mean_diff", lo_key="diff_ci_lo", hi_key="diff_ci_hi", prefix="diff")
    return payload


def stratified_bootstrap_meta(
    diffs_by_stratum: dict[str, np.ndarray],
    B: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
    alternative: str = "greater",
) -> dict[str, Any]:
    """
    Paired stratified bootstrap over per-problem diffs grouped by dataset/stratum.

    The point estimate is the sample-size-weighted mean over all diffs. Each
    bootstrap replicate resamples within every stratum with the original stratum
    size, then concatenates the resampled diffs.
    """
    if not diffs_by_stratum:
        raise ValueError("diffs_by_stratum must not be empty.")
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}.")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")

    arrays: dict[str, np.ndarray] = {}
    for name, diffs in diffs_by_stratum.items():
        values = np.asarray(diffs, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"Expected a 1-D diff array for stratum {name!r}, got shape {values.shape}.")
        if values.size == 0:
            raise ValueError(f"Stratum {name!r} is empty.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Stratum {name!r} contains NaN or Inf.")
        arrays[str(name)] = values

    all_diffs = np.concatenate(list(arrays.values()))
    observed = _mean(all_diffs)
    n_by_stratum = {name: int(values.size) for name, values in arrays.items()}

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(B, dtype=np.float64)
    for b in range(B):
        total = 0.0
        total_n = 0
        for values in arrays.values():
            n = int(values.size)
            indices = rng.integers(0, n, size=n)
            total += float(values[indices].sum())
            total_n += n
        boot_diffs[b] = total / total_n

    diff_ci_lo, diff_ci_hi = np.quantile(boot_diffs, [alpha / 2.0, 1.0 - alpha / 2.0])

    centered = {name: values - float(values.mean()) for name, values in arrays.items()}
    rng_null = np.random.default_rng(seed + 1)
    null_boot = np.empty(B, dtype=np.float64)
    for b in range(B):
        total = 0.0
        total_n = 0
        for values in centered.values():
            n = int(values.size)
            indices = rng_null.integers(0, n, size=n)
            total += float(values[indices].sum())
            total_n += n
        null_boot[b] = total / total_n

    if alternative == "greater":
        bootstrap_p_raw = float(np.mean(null_boot >= observed))
    elif alternative == "less":
        bootstrap_p_raw = float(np.mean(null_boot <= observed))
    else:
        bootstrap_p_raw = float(np.mean(np.abs(null_boot) >= abs(observed)))

    payload = {
        "mean_diff": observed,
        "diff_ci_lo": float(diff_ci_lo),
        "diff_ci_hi": float(diff_ci_hi),
        "bootstrap_p_raw": bootstrap_p_raw,
        "n_problems": int(all_diffs.size),
        "n_problems_by_stratum": n_by_stratum,
        "B": int(B),
        "ci_method": "stratified_percentile_bootstrap",
        "alternative": alternative,
    }
    _add_ci_display_fields(payload, point_key="mean_diff", lo_key="diff_ci_lo", hi_key="diff_ci_hi", prefix="diff")
    return payload


def stratified_paired_test(
    metrics_a_by_stratum: dict[str, dict[str, Any]],
    metrics_b_by_stratum: dict[str, dict[str, Any]],
    method_a: str,
    method_b: str,
    metric: str,
    B: int = 10000,
    seed: int = 0,
    alternative: str = "greater",
) -> dict[str, Any]:
    """
    Paired stratified bootstrap diff CI and bootstrap p-value for one method pair.
    """
    if metric not in PAIRED_METRICS:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {PAIRED_METRICS}.")
    if set(metrics_a_by_stratum) != set(metrics_b_by_stratum):
        raise ValueError("metrics_a_by_stratum and metrics_b_by_stratum must contain the same strata.")

    diffs_by_stratum: dict[str, np.ndarray] = {}
    for stratum in metrics_a_by_stratum:
        metrics_a = metrics_a_by_stratum[stratum]
        metrics_b = metrics_b_by_stratum[stratum]
        if metrics_a["sample_ids"] != metrics_b["sample_ids"]:
            raise ValueError(f"sample_id mismatch between {method_a} and {method_b} in stratum {stratum!r}.")
        vals_a = _metric_values(metrics_a["per_problem"], metric)
        vals_b = _metric_values(metrics_b["per_problem"], metric)
        diffs_by_stratum[stratum] = vals_a - vals_b

    payload = stratified_bootstrap_meta(
        diffs_by_stratum,
        B=B,
        seed=seed,
        alternative=alternative,
    )
    payload.update(
        {
            "method_a": method_a,
            "method_b": method_b,
            "metric": metric,
            "wilcoxon_stat": None,
            "wilcoxon_p_raw": None,
            "warnings": [],
        }
    )
    if metric == "tok":
        _add_ci_display_fields(
            payload,
            point_key="mean_diff",
            lo_key="diff_ci_lo",
            hi_key="diff_ci_hi",
            prefix="diff",
            include_percent=False,
            include_kilo=True,
        )
    return payload


def apply_holm_correction(
    test_results: list[dict[str, Any]],
    p_field: str,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """
    Holm-Bonferroni correction over a family of test results.
    """
    adjusted_field = p_field.replace("_raw", "_holm")
    reject_field = f"{adjusted_field}_reject"
    corrected = [dict(item) for item in test_results]
    valid: list[tuple[int, float]] = []

    for idx, item in enumerate(corrected):
        p_value = item.get(p_field)
        if p_value is None:
            item[adjusted_field] = None
            item[reject_field] = False
            continue
        p_float = float(p_value)
        if not math.isfinite(p_float):
            item[adjusted_field] = None
            item[reject_field] = False
            continue
        valid.append((idx, p_float))

    valid.sort(key=lambda pair: pair[1])
    m = len(valid)
    running_adjusted = 0.0
    holm_still_rejecting = True
    for rank, (idx, p_value) in enumerate(valid):
        factor = m - rank
        adjusted = min(1.0, factor * p_value)
        running_adjusted = max(running_adjusted, adjusted)
        corrected[idx][adjusted_field] = running_adjusted
        threshold = alpha / factor
        reject = holm_still_rejecting and p_value <= threshold
        if not reject:
            holm_still_rejecting = False
        corrected[idx][reject_field] = bool(reject)
    return corrected


def default_metric_alternatives(
    metrics: tuple[str, ...] = PAIRED_METRICS,
    default_alternative: str = "greater",
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    if default_alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("default_alternative must be 'greater', 'less', or 'two-sided'.")
    overrides = overrides or {}
    out: dict[str, str] = {}
    for metric in metrics:
        if metric not in PAIRED_METRICS:
            raise ValueError(f"Unknown paired metric {metric!r}; expected one of {PAIRED_METRICS}.")
        alternative = overrides.get(metric, "less" if metric == "tok" else default_alternative)
        if alternative not in {"greater", "less", "two-sided"}:
            raise ValueError(f"Invalid alternative for {metric}: {alternative!r}.")
        out[metric] = alternative
    return out


def method_pair_family(method_a: str, method_b: str) -> str:
    return f"{method_a}_vs_{method_b}"


def combine_stouffer_pvalues(
    p_values: list[float],
    weights: list[float] | None = None,
) -> dict[str, Any]:
    if not p_values:
        raise ValueError("p_values must not be empty.")
    p_array = np.asarray(p_values, dtype=np.float64)
    if p_array.ndim != 1:
        raise ValueError(f"Expected a 1-D p-value array, got shape {p_array.shape}.")
    if not np.all(np.isfinite(p_array)):
        raise ValueError("p_values contains NaN or Inf.")
    if np.any((p_array < 0.0) | (p_array > 1.0)):
        raise ValueError("p_values must be between 0 and 1.")

    weight_array: np.ndarray | None = None
    if weights is not None:
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.shape != p_array.shape:
            raise ValueError("weights must have the same length as p_values.")
        if not np.all(np.isfinite(weight_array)):
            raise ValueError("weights contains NaN or Inf.")
        if np.any(weight_array <= 0.0):
            raise ValueError("weights must be positive.")

    stat, p_raw = combine_pvalues(p_array, method="stouffer", weights=weight_array)
    return {
        "stat": float(stat),
        "p_raw": float(p_raw),
        "n": int(p_array.size),
        "weights": None if weight_array is None else [float(item) for item in weight_array.tolist()],
    }


def build_paired_tests(
    evaluations_with_arrays: dict[str, dict[str, Any]],
    *,
    methods: list[str],
    reference_method: str,
    paired_metrics: tuple[str, ...] = PAIRED_METRICS,
    metric_alternatives: dict[str, str] | None = None,
    comparison_mode: str = "reference",
    B: int = 10000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    if comparison_mode not in COMPARISON_MODES:
        raise ValueError(f"comparison_mode must be one of {COMPARISON_MODES}.")
    if reference_method not in evaluations_with_arrays:
        raise ValueError(f"Reference method {reference_method!r} is not among method specs.")

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
    pair_seed = seed + 100000
    for pair_idx, (method_a, method_b) in enumerate(pair_specs):
        for metric_idx, metric in enumerate(paired_metrics):
            test = paired_test(
                evaluations_with_arrays[method_a],
                evaluations_with_arrays[method_b],
                method_a,
                method_b,
                metric,
                B=B,
                seed=pair_seed + pair_idx * 1009 + metric_idx,
                alternative=(metric_alternatives or {})[metric],
            )
            test["family"] = method_pair_family(method_a, method_b)
            paired_tests.append(test)
    return add_multi_holm_corrections(paired_tests)


def _apply_grouped_holm(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    p_field: str,
    out_p_field: str,
    out_reject_field: str,
) -> list[dict[str, Any]]:
    corrected = [dict(item) for item in rows]
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(corrected):
        grouped[str(item[group_key])].append(idx)
    for _, indices in grouped.items():
        subset = [corrected[idx] for idx in indices]
        adjusted = apply_holm_correction(subset, p_field)
        adjusted_p_field = p_field.replace("_raw", "_holm")
        adjusted_reject_field = f"{adjusted_p_field}_reject"
        for idx, item in zip(indices, adjusted, strict=True):
            corrected[idx][out_p_field] = item.get(adjusted_p_field)
            corrected[idx][out_reject_field] = item.get(adjusted_reject_field)
    return corrected


def add_multi_holm_corrections(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired_tests = [dict(test) for test in tests]
    for test in paired_tests:
        test.setdefault("family", method_pair_family(test["method_a"], test["method_b"]))

    paired_tests = _apply_grouped_holm(
        paired_tests,
        group_key="family",
        p_field="wilcoxon_p_raw",
        out_p_field="wilcoxon_p_holm_method",
        out_reject_field="wilcoxon_p_holm_method_reject",
    )
    paired_tests = _apply_grouped_holm(
        paired_tests,
        group_key="family",
        p_field="bootstrap_p_raw",
        out_p_field="bootstrap_p_holm_method",
        out_reject_field="bootstrap_p_holm_method_reject",
    )
    paired_tests = _apply_grouped_holm(
        paired_tests,
        group_key="metric",
        p_field="wilcoxon_p_raw",
        out_p_field="wilcoxon_p_holm_metric",
        out_reject_field="wilcoxon_p_holm_metric_reject",
    )
    paired_tests = _apply_grouped_holm(
        paired_tests,
        group_key="metric",
        p_field="bootstrap_p_raw",
        out_p_field="bootstrap_p_holm_metric",
        out_reject_field="bootstrap_p_holm_metric_reject",
    )

    adjusted_global_w = apply_holm_correction(paired_tests, "wilcoxon_p_raw")
    adjusted_global_b = apply_holm_correction(paired_tests, "bootstrap_p_raw")
    for idx, (item_w, item_b) in enumerate(zip(adjusted_global_w, adjusted_global_b, strict=True)):
        paired_tests[idx]["wilcoxon_p_holm_global"] = item_w.get("wilcoxon_p_holm")
        paired_tests[idx]["wilcoxon_p_holm_global_reject"] = item_w.get("wilcoxon_p_holm_reject")
        paired_tests[idx]["bootstrap_p_holm_global"] = item_b.get("bootstrap_p_holm")
        paired_tests[idx]["bootstrap_p_holm_global_reject"] = item_b.get("bootstrap_p_holm_reject")

    for test in paired_tests:
        for key in [
            "wilcoxon_p_holm_method",
            "wilcoxon_p_holm_metric",
            "wilcoxon_p_holm_global",
            "bootstrap_p_holm_method",
            "bootstrap_p_holm_metric",
            "bootstrap_p_holm_global",
        ]:
            test.setdefault(key, None)
        for key in [
            "wilcoxon_p_holm_method_reject",
            "wilcoxon_p_holm_metric_reject",
            "wilcoxon_p_holm_global_reject",
            "bootstrap_p_holm_method_reject",
            "bootstrap_p_holm_metric_reject",
            "bootstrap_p_holm_global_reject",
        ]:
            test.setdefault(key, False)
    return paired_tests


def _strip_per_problem(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in evaluation.items()
        if key not in {"sample_ids", "per_problem"}
    }


def _artifact_hash(run_dir: Path, k: int) -> str:
    files = [
        RUN_OVERVIEW_FILE,
        FINAL_OUTPUTS_FILE,
        SOURCE_TARGETS_FILE,
        report_path_for_k(run_dir, k).name,
    ]
    digest = hashlib.sha256()
    for name in files:
        path = run_dir / name
        if not path.exists():
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    commit = proc.stdout.strip()
    return commit or None


def _method_run_metadata(run_dir: Path, k: int, label: str | None = None) -> dict[str, Any]:
    overview = load_run_overview(run_dir)
    payload = {
        "run_dir": str(run_dir),
        "run_id": overview.run_id,
        "dataset_name": overview.dataset_name,
        "model_name": overview.model_name,
        "run_type": overview.run_type,
        "strategy_name": overview.strategy_name,
        "strategy_label": overview.strategy_label,
        "baseline_run_dir": overview.baseline_run_dir,
        "reference_run_dir": overview.reference_run_dir,
        "artifact_hash": _artifact_hash(run_dir, k),
    }
    if label:
        payload["label"] = label
    return payload


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
        "mcnemar_n_a_only",
        "mcnemar_n_b_only",
        "mcnemar_n_discordant",
        "mcnemar_p_raw",
        "warnings",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for test in tests:
            row = {key: test.get(key) for key in fieldnames}
            row["k"] = k
            row["warnings"] = ";".join(test.get("warnings") or [])
            writer.writerow(row)


def _write_readme(
    path: Path,
    *,
    command: str,
    dataset_name: str,
    k: int,
    B: int,
    seed: int,
    git_commit: str | None,
    method_runs: dict[str, Any],
    per_method_metrics: dict[str, Any],
    paired_tests: list[dict[str, Any]],
    paired_metrics: tuple[str, ...],
    metric_alternatives: dict[str, str],
    comparison_mode: str,
) -> None:
    import scipy

    warning_lines: list[str] = []
    for test in paired_tests:
        for warning in test.get("warnings") or []:
            warning_lines.append(f"- {test['method_a']} vs {test['method_b']} {test['metric']}: {warning}")
    if not warning_lines:
        warning_lines.append("- none")

    alignment_lines = []
    for method, evaluation in per_method_metrics.items():
        alignment = evaluation["v2_alignment"]
        alignment_lines.append(
            f"- {method}: passed={alignment['passed']}, "
            f"mean_abs_error={alignment['mean_abs_error']}, "
            f"pass_abs_error={alignment['pass_abs_error']}, "
            f"maj_abs_error={alignment['maj_abs_error']}, "
            f"tok_abs_error={alignment['tok_abs_error']}"
        )

    method_lines = [
        f"- {method}: {metadata['run_dir']} ({metadata['artifact_hash']})"
        for method, metadata in method_runs.items()
    ]
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
    if not paired_tests:
        paired_lines.append("| none | none | none | none |  |  |  |  |  |")

    content = "\n".join(
        [
            f"# Bootstrap Stats: {dataset_name}",
            "",
            "## Command",
            f"`{command}`",
            "",
            "## Parameters",
            f"- k: {k}",
            f"- B: {B}",
            f"- seed: {seed}",
            f"- comparison_mode: {comparison_mode}",
            f"- paired_metrics: {', '.join(paired_metrics)}",
            f"- metric_alternatives: {', '.join(f'{metric}={metric_alternatives[metric]}' for metric in paired_metrics)}",
            "- holm_families: method pair, metric, global",
            "- paired_rows: Maj@K only",
            f"- git_commit: {git_commit or 'unknown'}",
            f"- numpy: {np.__version__}",
            f"- scipy: {scipy.__version__}",
            "",
            "## Methods",
            *method_lines,
            "",
            "## Per-Method Metrics",
            *metric_lines,
            "",
            "## Paired Tests",
            *paired_lines,
            "",
            "## V2 Alignment",
            *alignment_lines,
            "",
            "## Warnings",
            *warning_lines,
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def run_full_bootstrap_evaluation(
    method_specs: list[dict[str, Any]],
    reference_method: str,
    output_dir: Path,
    dataset_name: str,
    k: int = 32,
    B: int = 10000,
    seed: int = 0,
    alternative: str = "greater",
    paired_metrics: tuple[str, ...] = METRICS,
    metric_alternatives: dict[str, str] | None = None,
    comparison_mode: str = "reference",
    command: str | None = None,
) -> None:
    """
    Run the full statistical evaluation and write JSON/CSV artifacts.
    """
    if not method_specs:
        raise ValueError("At least one method spec is required.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluations_with_arrays: dict[str, dict[str, Any]] = {}
    method_runs: dict[str, dict[str, Any]] = {}
    for spec_idx, spec in enumerate(method_specs):
        method = str(spec["method"])
        if method in evaluations_with_arrays:
            raise ValueError(f"Duplicate method spec: {method}.")
        run_dir = Path(spec["run_dir"]).resolve()
        evaluation = evaluate_method(method, run_dir, k, B=B, seed=seed + spec_idx * 1009)
        evaluations_with_arrays[method] = evaluation
        method_runs[method] = _method_run_metadata(run_dir, k, spec.get("label"))

    if reference_method not in evaluations_with_arrays:
        raise ValueError(f"Reference method {reference_method!r} is not among method specs.")

    reference_sample_ids = evaluations_with_arrays[reference_method]["sample_ids"]
    for method, evaluation in evaluations_with_arrays.items():
        if evaluation["sample_ids"] != reference_sample_ids:
            raise ValueError(f"sample_id mismatch between {method} and reference {reference_method}.")

    metric_alternatives = default_metric_alternatives(
        paired_metrics,
        default_alternative=alternative,
        overrides=metric_alternatives,
    )
    paired_tests = build_paired_tests(
        evaluations_with_arrays,
        methods=list(evaluations_with_arrays.keys()),
        reference_method=reference_method,
        paired_metrics=paired_metrics,
        metric_alternatives=metric_alternatives,
        comparison_mode=comparison_mode,
        B=B,
        seed=seed,
    )

    per_method_metrics = {
        method: _strip_per_problem(evaluation)
        for method, evaluation in evaluations_with_arrays.items()
    }

    n_problems = len(reference_sample_ids)
    git_commit = _git_commit()
    run_timestamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    summary = {
        "dataset": dataset_name,
        "k": k,
        "n_problems": n_problems,
        "B": B,
        "seed": seed,
        "methods": list(evaluations_with_arrays.keys()),
        "reference_method": reference_method,
        "comparison_mode": comparison_mode,
        "paired_metrics": list(paired_metrics),
        "metric_alternatives": metric_alternatives,
        "holm_families": {
            "method": sorted({test["family"] for test in paired_tests}),
            "metric": list(paired_metrics),
            "global": len(paired_tests),
        },
        "run_timestamp": run_timestamp,
        "git_commit": git_commit,
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

    saturation_warnings = []
    for method, evaluation in per_method_metrics.items():
        for metric, payload in evaluation["metrics"].items():
            if payload.get("ci_method") == "jeffreys_beta":
                saturation_warnings.append(f"{method}.{metric}")

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
    diagnostics = {
        "dataset": dataset_name,
        "k": k,
        "per_method_diagnostics": {
            method: evaluation["diagnostics"]
            for method, evaluation in per_method_metrics.items()
        },
        "v2_alignment_failures": [
            method
            for method, evaluation in per_method_metrics.items()
            if not evaluation["v2_alignment"].get("passed")
        ],
        "saturation_warnings": saturation_warnings,
        "wilcoxon_warnings": wilcoxon_warnings,
    }

    paired_payload = {
        "reference_method": reference_method,
        "alternative": alternative,
        "metric_alternatives": metric_alternatives,
        "comparison_mode": comparison_mode,
        "k": k,
        "holm_family_size": len(paired_tests),
        "holm_family_sizes": {
            "method": len({test["family"] for test in paired_tests}),
            "metric": len(paired_metrics),
            "global": len(paired_tests),
        },
        "holm_alpha": 0.05,
        "paired_metrics": list(paired_metrics),
        "tests": paired_tests,
    }

    _dump_json(output_dir / "summary.json", summary)
    _dump_json(output_dir / "per_method_metrics.json", per_method_metrics)
    _dump_json(output_dir / "paired_tests.json", paired_payload)
    _dump_json(output_dir / "diagnostics.json", diagnostics)
    _write_per_method_csv(output_dir / "per_method_metrics.csv", per_method_metrics)
    _write_paired_csv(output_dir / "paired_tests.csv", paired_tests, k)
    _write_readme(
        output_dir / "README.md",
        command=command or " ".join(shlex.quote(arg) for arg in sys.argv),
        dataset_name=dataset_name,
        k=k,
        B=B,
        seed=seed,
        git_commit=git_commit,
        method_runs=method_runs,
        per_method_metrics=per_method_metrics,
        paired_tests=paired_tests,
        paired_metrics=paired_metrics,
        metric_alternatives=metric_alternatives,
        comparison_mode=comparison_mode,
    )
