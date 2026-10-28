#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import orjson
from prettytable import PrettyTable

from cognitive_probe.evaluation.data_model import (
    FINAL_OUTPUTS_FILE,
    SOURCE_TARGETS_FILE,
    GenerationSegment,
)
from cognitive_probe.evaluation.io import dump_json, load_jsonl
from cognitive_probe.evaluation.livebench_code_judge import (
    extract_python_code,
    judge_code,
    stable_hash,
)
from cognitive_probe.evaluation.v2 import ensure_v2_run, load_generation_segments, load_run_overview

CUSTOM_OUTPUTS_FILE = "livebench_custom_outputs.json"
JUDGE_RESULTS_FILE = "livebench_judge_results.jsonl"
REPORT_FILE = "livebench_eval_report.json"
JUDGE_CACHE_SCHEMA_VERSION = 2


def parse_ks(k_arg: str) -> list[int]:
    ks = sorted({int(item) for item in k_arg.replace(",", " ").split() if item.strip()})
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("--k must contain positive integers")
    return ks


def _first_target_by_sample(source_targets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for target in source_targets:
        sample_id = str(target.get("sample_id"))
        result.setdefault(sample_id, target)
    return result


def _question_id(sample_id: str, target: dict[str, Any] | None) -> str:
    if not target:
        return sample_id
    judge_payload = target.get("judge_payload") if isinstance(target.get("judge_payload"), dict) else {}
    metadata = target.get("metadata") if isinstance(target.get("metadata"), dict) else {}
    for container in (judge_payload, metadata):
        value = container.get("question_id")
        if value:
            return str(value)
    return sample_id


def _validate_targets(targets_by_sample: dict[str, dict[str, Any]]) -> None:
    bad: list[str] = []
    for sample_id, target in targets_by_sample.items():
        judge_type = str(target.get("judge_type") or "math")
        if judge_type != "livecodebench_code_generation":
            bad.append(f"{sample_id}:{judge_type}")
        if len(bad) >= 5:
            break
    if bad:
        raise ValueError(
            "run_eval_livebench.py requires livecodebench_code_generation targets; "
            f"found incompatible targets: {bad}"
        )


def load_run_rows(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    ensure_v2_run(run_dir)
    final_outputs = load_jsonl(run_dir / FINAL_OUTPUTS_FILE)
    source_targets = load_jsonl(run_dir / SOURCE_TARGETS_FILE)
    targets_by_sample = _first_target_by_sample(source_targets)
    _validate_targets(targets_by_sample)
    return final_outputs, targets_by_sample


def select_outputs_by_k(final_outputs: list[dict[str, Any]], max_k: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(
        final_outputs,
        key=lambda item: (
            str(item.get("sample_id")),
            int(item.get("k_group") or 0),
            int(item.get("source_row_index") or 0),
        ),
    ):
        sample_id = str(row.get("sample_id"))
        grouped[sample_id].append(row)
    return {sample_id: rows[:max_k] for sample_id, rows in grouped.items()}


def _usage_bucket() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "request_count": 0,
        "total_tokens": 0,
    }


def _accumulate_usage(bucket: dict[str, int], segment: GenerationSegment) -> None:
    bucket["input_tokens"] += int(segment.usage.input_tokens or 0)
    bucket["output_tokens"] += int(segment.usage.output_tokens or 0)
    bucket["cached_input_tokens"] += int(segment.usage.cached_input_tokens or 0)
    bucket["request_count"] += int(segment.usage.request_count or 0)
    bucket["total_tokens"] += int(segment.usage.input_tokens or 0) + int(segment.usage.output_tokens or 0)


def _collect_lineage(segment_id: str, segment_map: dict[str, GenerationSegment]) -> list[GenerationSegment]:
    lineage: list[GenerationSegment] = []
    current = segment_map.get(segment_id)
    while current is not None:
        lineage.append(current)
        current = segment_map.get(current.parent_segment_id) if current.parent_segment_id else None
    lineage.reverse()
    return lineage


def _compute_token_summary(
    *,
    grouped_outputs: dict[str, list[dict[str, Any]]],
    segments: list[GenerationSegment],
    k: int,
) -> dict[str, Any]:
    segment_map = {segment.segment_id: segment for segment in segments}
    selected_outputs = [
        output
        for rows in grouped_outputs.values()
        for output in rows[:k]
    ]

    used_segment_ids: set[str] = set()
    per_output_visible_tokens: list[int] = []
    for output in selected_outputs:
        lineage = _collect_lineage(str(output.get("final_segment_id") or ""), segment_map)
        if not lineage:
            source_tokens = output.get("source_output_tokens")
            if isinstance(source_tokens, int):
                per_output_visible_tokens.append(source_tokens)
            continue
        used_segment_ids.update(segment.segment_id for segment in lineage)
        per_output_visible_tokens.append(sum(int(segment.visible_output_tokens or 0) for segment in lineage))

    billed_input_tokens_total = 0
    billed_output_tokens_total = 0
    cached_input_tokens_total = 0
    request_count_total = 0
    final_visible_tokens_total = 0
    usage_by_model: dict[str, dict[str, int]] = defaultdict(_usage_bucket)
    usage_by_executor: dict[str, dict[str, int]] = defaultdict(_usage_bucket)
    cost_by_model: dict[str, float] = defaultdict(float)
    cost_by_executor: dict[str, float] = defaultdict(float)

    for segment_id in used_segment_ids:
        segment = segment_map[segment_id]
        input_tokens = int(segment.usage.input_tokens or 0)
        output_tokens = int(segment.usage.output_tokens or 0)
        cached_input_tokens = int(segment.usage.cached_input_tokens or 0)
        request_count = int(segment.usage.request_count or 0)
        total_tokens = input_tokens + output_tokens

        billed_input_tokens_total += input_tokens
        billed_output_tokens_total += output_tokens
        cached_input_tokens_total += cached_input_tokens
        request_count_total += request_count
        final_visible_tokens_total += int(segment.visible_output_tokens or 0)

        model_name = segment.model_name or "unknown"
        executor_name = segment.executor_name or "unknown"
        _accumulate_usage(usage_by_model[model_name], segment)
        _accumulate_usage(usage_by_executor[executor_name], segment)
        cost_by_model[model_name] += float(total_tokens)
        cost_by_executor[executor_name] += float(total_tokens)

    billed_tokens_total = billed_input_tokens_total + billed_output_tokens_total
    num_outputs = len(selected_outputs)
    avg_final_visible_tokens = (
        sum(per_output_visible_tokens) / len(per_output_visible_tokens)
        if per_output_visible_tokens
        else 0.0
    )
    avg_cost_per_output = float(billed_tokens_total) / num_outputs if num_outputs > 0 else 0.0

    return {
        "k": k,
        "num_outputs_total": num_outputs,
        "avg_final_visible_tokens": avg_final_visible_tokens,
        "total_cost": float(billed_tokens_total),
        "total_cost_unit": "tokens",
        "billed_tokens_total": billed_tokens_total,
        "billed_input_tokens_total": billed_input_tokens_total,
        "billed_output_tokens_total": billed_output_tokens_total,
        "cached_input_tokens_total": cached_input_tokens_total,
        "request_count_total": request_count_total,
        "final_visible_tokens_total": final_visible_tokens_total,
        "avg_cost_per_output": avg_cost_per_output,
        "usage_summary": {
            "totals": {
                "input_tokens": billed_input_tokens_total,
                "output_tokens": billed_output_tokens_total,
                "cached_input_tokens": cached_input_tokens_total,
                "request_count": request_count_total,
                "total_tokens": billed_tokens_total,
                "final_visible_tokens": final_visible_tokens_total,
            },
            "by_model": {key: value for key, value in sorted(usage_by_model.items())},
            "by_executor": {key: value for key, value in sorted(usage_by_executor.items())},
        },
        "cost_summary": {
            "unit": "tokens",
            "by_model": {key: value for key, value in sorted(cost_by_model.items())},
            "by_executor": {key: value for key, value in sorted(cost_by_executor.items())},
        },
    }


def write_custom_outputs(
    run_dir: Path,
    grouped_outputs: dict[str, list[dict[str, Any]]],
    targets_by_sample: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(grouped_outputs):
        target = targets_by_sample.get(sample_id)
        code_list = [
            extract_python_code(str(output.get("response") or ""))
            for output in grouped_outputs[sample_id]
        ]
        rows.append(
            {
                "question_id": _question_id(sample_id, target),
                "code_list": code_list,
            }
        )
    (run_dir / CUSTOM_OUTPUTS_FILE).write_bytes(
        orjson.dumps(rows, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE)
    )
    return rows


def load_judge_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        output_id = row.get("output_id")
        if output_id:
            results[str(output_id)] = row
    return results


def _iter_jsonl(path: Path):
    with path.open("rb") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                yield orjson.loads(line)


def _manifest_path_from_targets(
    run_dir: Path,
    targets_by_sample: dict[str, dict[str, Any]],
    explicit_path: str | None,
) -> Path | None:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    for target in targets_by_sample.values():
        judge_payload = target.get("judge_payload") if isinstance(target.get("judge_payload"), dict) else {}
        manifest = judge_payload.get("judge_manifest_file")
        if isinstance(manifest, str) and manifest.strip():
            path = Path(manifest)
            if not path.is_absolute():
                path = path.resolve() if path.exists() else (run_dir / path).resolve()
            return path
    return None


def _load_full_judge_payloads(
    *,
    run_dir: Path,
    targets_by_sample: dict[str, dict[str, Any]],
    needed_sample_ids: set[str],
    judge_manifest: str | None,
) -> dict[str, dict[str, Any]]:
    path = _manifest_path_from_targets(run_dir, targets_by_sample, judge_manifest)
    if path is None:
        return {
            sample_id: target.get("judge_payload", {})
            for sample_id, target in targets_by_sample.items()
            if sample_id in needed_sample_ids and isinstance(target.get("judge_payload"), dict)
        }
    if not path.exists():
        raise FileNotFoundError(f"Judge manifest not found: {path}")

    remaining = set(needed_sample_ids)
    result: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        sample_id = str(row.get("sample_id") or row.get("id") or "")
        if sample_id not in remaining:
            continue
        payload = row.get("judge_payload")
        if isinstance(payload, dict):
            result[sample_id] = payload
            remaining.remove(sample_id)
        if not remaining:
            break
    if remaining:
        fallback = [
            sample_id
            for sample_id in sorted(remaining)
            if isinstance(targets_by_sample.get(sample_id, {}).get("judge_payload"), dict)
        ]
        for sample_id in fallback:
            result[sample_id] = targets_by_sample[sample_id]["judge_payload"]
    return result


def _result_cache_valid(
    result: dict[str, Any] | None,
    *,
    response_hash: str,
    judge_payload_hash: str,
) -> bool:
    if result is None:
        return False
    return (
        result.get("judge_schema_version") == JUDGE_CACHE_SCHEMA_VERSION
        and result.get("response_hash") == response_hash
        and result.get("judge_payload_hash") == judge_payload_hash
        and isinstance(result.get("test_pass_vector"), list)
        and isinstance(result.get("behavior_signature"), str)
    )


def _write_judge_results(path: Path, results: dict[str, dict[str, Any]]) -> None:
    with path.open("wb") as f:
        for output_id in sorted(results):
            f.write(orjson.dumps(results[output_id], option=orjson.OPT_APPEND_NEWLINE))


def _append_judge_result(path: Path, result: dict[str, Any]) -> None:
    with path.open("ab") as f:
        f.write(orjson.dumps(result, option=orjson.OPT_APPEND_NEWLINE))
        f.flush()
        os.fsync(f.fileno())


def run_local_judge(
    *,
    run_dir: Path,
    grouped_outputs: dict[str, list[dict[str, Any]]],
    targets_by_sample: dict[str, dict[str, Any]],
    judge_manifest: str | None,
    timeout_s: float,
    max_workers: int,
    force: bool,
    public_only: bool,
    include_test_details: bool,
    python_bin: str,
) -> dict[str, dict[str, Any]]:
    result_path = run_dir / JUDGE_RESULTS_FILE
    cached_results = load_judge_results(result_path)
    needed_sample_ids = set(grouped_outputs)
    judge_payloads = _load_full_judge_payloads(
        run_dir=run_dir,
        targets_by_sample=targets_by_sample,
        needed_sample_ids=needed_sample_ids,
        judge_manifest=judge_manifest,
    )

    jobs: list[tuple[str, str, dict[str, Any], dict[str, Any], str, str, str]] = []
    results = dict(cached_results)
    for sample_id, outputs in grouped_outputs.items():
        judge_payload = judge_payloads.get(sample_id)
        if not isinstance(judge_payload, dict):
            raise ValueError(f"Missing judge payload for sample_id={sample_id}")
        judge_payload_hash = stable_hash(judge_payload)
        for output in outputs:
            output_id = str(output.get("output_id"))
            response = str(output.get("response") or "")
            response_hash = stable_hash(response)
            if not force and _result_cache_valid(
                cached_results.get(output_id),
                response_hash=response_hash,
                judge_payload_hash=judge_payload_hash,
            ):
                continue
            code = extract_python_code(response)
            jobs.append((output_id, sample_id, output, judge_payload, code, response_hash, judge_payload_hash))

    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            future_map = {
                executor.submit(
                    judge_code,
                    code=code,
                    judge_payload=judge_payload,
                    timeout_s=timeout_s,
                    include_private=not public_only,
                    python_bin=python_bin,
                    include_test_details=include_test_details,
                ): (output_id, sample_id, output, code, response_hash, judge_payload_hash)
                for output_id, sample_id, output, judge_payload, code, response_hash, judge_payload_hash in jobs
            }
            for future in as_completed(future_map):
                output_id, sample_id, output, code, response_hash, judge_payload_hash = future_map[future]
                judge_result = future.result()
                test_summary = judge_result.to_dict(include_tests=include_test_details)
                payload = {
                    "judge_schema_version": JUDGE_CACHE_SCHEMA_VERSION,
                    "output_id": output_id,
                    "sample_id": sample_id,
                    "k_group": output.get("k_group"),
                    "passed": judge_result.passed,
                    "verdict": judge_result.verdict,
                    "error_type": judge_result.error_type,
                    "duration_s": judge_result.duration_s,
                    "response_hash": response_hash,
                    "judge_payload_hash": judge_payload_hash,
                    "extracted_code": code,
                    "test_pass_vector": list(judge_result.test_pass_vector),
                    "behavior_signature": test_summary["behavior_signature"],
                    "test_summary": test_summary,
                }
                results[output_id] = payload
                _append_judge_result(result_path, payload)
    _write_judge_results(result_path, results)
    return results


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
    counts: Counter[tuple[bool, ...]] = Counter()
    first_seen: dict[tuple[bool, ...], int] = {}
    for idx, signature in enumerate(signatures):
        counts[signature] += 1
        first_seen.setdefault(signature, idx)
    majority_signature = min(
        counts.items(),
        key=lambda item: (-item[1], first_seen[item[0]]),
    )[0]
    return bool(majority_signature) and all(majority_signature)


def compute_report(
    *,
    run_dir: Path,
    ks: list[int],
    grouped_outputs: dict[str, list[dict[str, Any]]],
    targets_by_sample: dict[str, dict[str, Any]],
    judge_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    max_k = max(ks)
    overview = load_run_overview(run_dir)
    segments = load_generation_segments(run_dir)
    token_summary_by_k = {
        str(k): _compute_token_summary(
            grouped_outputs=grouped_outputs,
            segments=segments,
            k=k,
        )
        for k in ks
    }
    token_summary = token_summary_by_k[str(max_k)]
    status = "judged" if judge_results else "pending_judge"
    mean_at_k: dict[str, float | None] = {}
    maj_at_k: dict[str, float | None] = {}
    best_score_mean: dict[str, float | None] = {}
    lcb_pass_at_1: float | None = None
    missing_output_ids: set[str] = set()

    for k in ks:
        per_item_mean: list[float] = []
        per_item_maj: list[float] = []
        per_item_best: list[float] = []
        missing_behavior_output_ids: set[str] = set()
        for sample_id, outputs in grouped_outputs.items():
            scores: list[int] = []
            vote_signatures: list[tuple[bool, ...]] = []
            for output in outputs[:max_k]:
                output_id = str(output.get("output_id"))
                result = judge_results.get(output_id)
                if result is None:
                    missing_output_ids.add(output_id)
                    continue
                score = 1 if result.get("passed") else 0
                if len(scores) < k:
                    scores.append(score)
                    if not output.get("was_pruned"):
                        signature = _behavior_vector(result)
                        if signature is None:
                            missing_behavior_output_ids.add(output_id)
                        else:
                            vote_signatures.append(signature)

            if scores:
                per_item_mean.append(sum(scores) / len(scores))
                per_item_maj.append(1.0 if _majority_behavior_is_correct(vote_signatures) else 0.0)
                per_item_best.append(float(max(scores)))

        if missing_behavior_output_ids:
            first = sorted(missing_behavior_output_ids)[:5]
            raise ValueError(
                f"Missing behavior signatures for {len(missing_behavior_output_ids)} judged outputs at k={k}; "
                f"first={first}. Re-run with --run-judge to refresh LiveCodeBench judge cache."
            )

        mean_at_k[str(k)] = (sum(per_item_mean) / len(per_item_mean)) if per_item_mean else None
        maj_at_k[str(k)] = (sum(per_item_maj) / len(per_item_maj)) if per_item_maj else None
        best_score_mean[str(k)] = (sum(per_item_best) / len(per_item_best)) if per_item_best else None

    per_item_lcb_pass_1: list[float] = []
    for outputs in grouped_outputs.values():
        scores = []
        for output in outputs[:max_k]:
            result = judge_results.get(str(output.get("output_id")))
            if result is None:
                continue
            scores.append(1 if result.get("passed") else 0)
        n = len(scores)
        c = sum(scores)
        if n >= 1:
            per_item_lcb_pass_1.append(c / n)
    if per_item_lcb_pass_1:
        lcb_pass_at_1 = sum(per_item_lcb_pass_1) / len(per_item_lcb_pass_1)

    error_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    for result in judge_results.values():
        error_type = result.get("error_type")
        verdict = result.get("verdict")
        if error_type:
            error_counts[str(error_type)] += 1
        if verdict:
            verdict_counts[str(verdict)] += 1

    target_hashes = {
        sample_id: stable_hash(target.get("judge_payload", {}))
        for sample_id, target in targets_by_sample.items()
    }
    response_hashes = {
        str(output.get("output_id")): stable_hash(str(output.get("response") or ""))
        for rows in grouped_outputs.values()
        for output in rows[:max_k]
    }

    return {
        "schema_version": 2,
        "run_id": overview.run_id,
        "run_dir": str(run_dir),
        "dataset_name": overview.dataset_name,
        "model_name": overview.model_name,
        "run_type": overview.run_type,
        "strategy_name": overview.strategy_name,
        "status": status,
        "k": ks,
        "num_items": len(grouped_outputs),
        "num_outputs_total": sum(len(rows) for rows in grouped_outputs.values()),
        "mean_at_k": mean_at_k,
        "maj_at_k": maj_at_k,
        "best_score_mean": best_score_mean,
        "lcb_pass_at_1": lcb_pass_at_1,
        "missing_judgements": len(missing_output_ids) if judge_results else None,
        "error_counts": dict(sorted(error_counts.items())),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "avg_final_visible_tokens": token_summary["avg_final_visible_tokens"],
        "total_cost": token_summary["total_cost"],
        "total_cost_unit": token_summary["total_cost_unit"],
        "billed_tokens_total": token_summary["billed_tokens_total"],
        "billed_input_tokens_total": token_summary["billed_input_tokens_total"],
        "billed_output_tokens_total": token_summary["billed_output_tokens_total"],
        "cached_input_tokens_total": token_summary["cached_input_tokens_total"],
        "request_count_total": token_summary["request_count_total"],
        "final_visible_tokens_total": token_summary["final_visible_tokens_total"],
        "avg_cost_per_output": token_summary["avg_cost_per_output"],
        "usage_summary": token_summary["usage_summary"],
        "cost_summary": token_summary["cost_summary"],
        "token_summary_by_k": token_summary_by_k,
        "responses_per_item_hist": {
            str(k1): int(v1)
            for k1, v1 in Counter(len(v) for v in grouped_outputs.values()).items()
        },
        "artifacts": {
            "custom_outputs": CUSTOM_OUTPUTS_FILE,
            "judge_results": JUDGE_RESULTS_FILE,
            "report": REPORT_FILE,
        },
        "hashes": {
            "judge_payload_by_sample": target_hashes,
            "response_by_output": response_hashes,
        },
    }


def print_report(report: dict[str, Any]) -> None:
    table = PrettyTable()
    table.field_names = ["metric", "value"]
    table.align["metric"] = "l"
    table.align["value"] = "r"
    table.add_row(["status", report["status"]])
    table.add_row(["num_items", report["num_items"]])
    table.add_row(["num_outputs_total", report["num_outputs_total"]])
    for k, value in report["mean_at_k"].items():
        table.add_row([f"Mean@{k}", "N/A" if value is None else f"{value:.6f}"])
    for k, value in report["maj_at_k"].items():
        table.add_row([f"Maj@{k}", "N/A" if value is None else f"{value:.6f}"])
    for k, value in report["best_score_mean"].items():
        table.add_row([f"BestMean@{k}", "N/A" if value is None else f"{value:.6f}"])
    value = report["lcb_pass_at_1"]
    table.add_row(["lcb_pass@1", "N/A" if value is None else f"{value:.6f}"])
    table.add_row(["avg_final_visible_tokens", f"{float(report['avg_final_visible_tokens']):.2f}"])
    table.add_row(["billed_tokens_total", str(report["billed_tokens_total"])])
    table.add_row(["billed_input_tokens_total", str(report["billed_input_tokens_total"])])
    table.add_row(["billed_output_tokens_total", str(report["billed_output_tokens_total"])])
    table.add_row(["cached_input_tokens_total", str(report["cached_input_tokens_total"])])
    table.add_row(["request_count_total", str(report["request_count_total"])])
    table.add_row(["final_visible_tokens_total", str(report["final_visible_tokens_total"])])
    table.add_row(["avg_cost_per_output", f"{float(report['avg_cost_per_output']):.2f} {report['total_cost_unit']}"])
    usage_summary = report.get("usage_summary") if isinstance(report.get("usage_summary"), dict) else {}
    by_model = usage_summary.get("by_model") if isinstance(usage_summary.get("by_model"), dict) else {}
    for model_name, usage in sorted(by_model.items()):
        if not isinstance(usage, dict):
            continue
        table.add_row([
            "model_usage",
            f"{model_name}: in={usage.get('input_tokens', 0)}, "
            f"out={usage.get('output_tokens', 0)}, req={usage.get('request_count', 0)}",
        ])
    print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LiveCodeBench evaluator entrypoint. Writes custom output rows and summarizes cached judge results.",
    )
    parser.add_argument("--dir", required=True, help="Generation run directory.")
    parser.add_argument("-k", "--k", default="1,5,10", help="Comma or space separated k values.")
    parser.add_argument("--run-judge", action="store_true", help="Run local code judge and update livebench_judge_results.jsonl.")
    parser.add_argument("--judge-manifest", default=None, help="Full judge payload sidecar JSONL. Defaults to manifest path in benchmark rows.")
    parser.add_argument("--timeout-s", type=float, default=6.0, help="Timeout per test case.")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel judge workers.")
    parser.add_argument("--force", action="store_true", help="Rejudge outputs even if cache hashes match.")
    parser.add_argument("--public-only", action="store_true", help="Use only public tests.")
    parser.add_argument("--include-test-details", action="store_true", help="Store per-test details in judge cache.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used for candidate code.")
    args = parser.parse_args()

    run_dir = Path(args.dir).resolve()
    ks = parse_ks(args.k)
    final_outputs, targets_by_sample = load_run_rows(run_dir)
    grouped_outputs = select_outputs_by_k(final_outputs, max(ks))
    write_custom_outputs(run_dir, grouped_outputs, targets_by_sample)
    if args.run_judge:
        judge_results = run_local_judge(
            run_dir=run_dir,
            grouped_outputs=grouped_outputs,
            targets_by_sample=targets_by_sample,
            judge_manifest=args.judge_manifest,
            timeout_s=args.timeout_s,
            max_workers=args.max_workers,
            force=args.force,
            public_only=args.public_only,
            include_test_details=args.include_test_details,
            python_bin=args.python_bin,
        )
    else:
        judge_results = load_judge_results(run_dir / JUDGE_RESULTS_FILE)
    report = compute_report(
        run_dir=run_dir,
        ks=ks,
        grouped_outputs=grouped_outputs,
        targets_by_sample=targets_by_sample,
        judge_results=judge_results,
    )
    dump_json(run_dir / REPORT_FILE, report)
    print_report(report)
    print(f"Wrote {run_dir / CUSTOM_OUTPUTS_FILE}")
    print(f"Wrote {run_dir / REPORT_FILE}")


if __name__ == "__main__":
    main()
