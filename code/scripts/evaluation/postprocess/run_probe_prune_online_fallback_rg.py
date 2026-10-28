#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import orjson
from tqdm.auto import tqdm

from cognitive_probe.evaluation.data_model import (
    DECISION_LOG_FILE,
    FINAL_OUTPUTS_FILE,
    GENERATION_SEGMENTS_FILE,
    RUN_OVERVIEW_FILE,
    SOURCE_TARGETS_FILE,
    DecisionRecord,
    FinalOutput,
    GenerationSegment,
    SegmentUsage,
)
from cognitive_probe.evaluation.io import dump_json, json_text, load_jsonl, write_jsonl
from cognitive_probe.evaluation.probe_prune import build_strategy
from cognitive_probe.evaluation.probe_prune.common import (
    build_source_targets,
    decode_response,
    legacy_defaults_for_strategy,
    load_original_args,
    merge_strategy_args,
    prepare_output_dir,
    resolve_input_file,
    split_input_strategy_args,
)
from cognitive_probe.evaluation.probe_prune.online_common import (
    build_group_outcomes,
    build_prune_config,
)
from cognitive_probe.evaluation.v2 import _reference_pass_map, build_run_overview


@dataclass(frozen=True)
class FallbackRegenerationConfig:
    api_url: str
    model_name: str
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    seed_offset: int
    timeout_s: float
    concurrency_limit: int
    rollback_tokens: int | None
    rollback_ratio: float | None


@dataclass(frozen=True)
class FallbackRegenerationTrace:
    row_index: int
    regenerated_row: dict[str, Any]
    regenerated_seed: int | None
    stop_index: int
    resume_index: int
    rollback_applied_tokens: int
    rollback_mode: str
    continuation_output_ids: list[int]


def _build_fallback_regeneration_config(
    *,
    original_args: dict[str, Any],
    strategy_args: dict[str, Any],
) -> FallbackRegenerationConfig:
    api_url = str(strategy_args.get("api_url") or "").strip().rstrip("/")
    if not api_url:
        raise ValueError("fallback_regenerate_if_pruned requires explicit api_url")

    model_name = str(strategy_args.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("fallback_regenerate_if_pruned requires explicit model_name")

    try:
        temperature = float(
            strategy_args.get("temperature", original_args.get("temperature", 1.0))
        )
        top_p = float(strategy_args.get("top_p", original_args.get("top_p", 1.0)))
        top_k = int(strategy_args.get("top_k", original_args.get("top_k", 20)))
        max_new_tokens = int(
            strategy_args.get("max_new_tokens", original_args.get("max_length", 1024))
        )
        seed_offset = int(strategy_args.get("seed_offset", 1000003))
        timeout_s = float(strategy_args.get("timeout_s", 300.0))
        concurrency_limit = int(strategy_args.get("concurrency_limit", 16))
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid fallback_regenerate_if_pruned args: {e}") from e

    rollback_tokens_raw = strategy_args.get("rollback_tokens")
    rollback_ratio_raw = strategy_args.get("rollback_ratio")
    if rollback_tokens_raw is not None and rollback_ratio_raw is not None:
        raise ValueError(
            "fallback_regenerate_if_pruned accepts only one of rollback_tokens or rollback_ratio"
        )

    rollback_tokens: int | None = None
    rollback_ratio: float | None = None
    try:
        if rollback_ratio_raw is not None:
            rollback_ratio = float(rollback_ratio_raw)
        else:
            rollback_tokens = int(
                rollback_tokens_raw if rollback_tokens_raw is not None else 32
            )
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid fallback_regenerate_if_pruned rollback args: {e}"
        ) from e

    if max_new_tokens <= 0:
        raise ValueError("fallback_regenerate_if_pruned requires max_new_tokens > 0")
    if timeout_s <= 0:
        raise ValueError("fallback_regenerate_if_pruned requires timeout_s > 0")
    if concurrency_limit <= 0:
        raise ValueError("fallback_regenerate_if_pruned requires concurrency_limit > 0")
    if rollback_tokens is not None and rollback_tokens <= 0:
        raise ValueError("fallback_regenerate_if_pruned requires rollback_tokens > 0")
    if rollback_ratio is not None and (rollback_ratio <= 0.0 or rollback_ratio > 1.0):
        raise ValueError(
            "fallback_regenerate_if_pruned requires rollback_ratio in (0, 1]"
        )

    return FallbackRegenerationConfig(
        api_url=api_url,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        seed_offset=seed_offset,
        timeout_s=timeout_s,
        concurrency_limit=concurrency_limit,
        rollback_tokens=rollback_tokens,
        rollback_ratio=rollback_ratio,
    )


def _compute_resume_index(
    *,
    bounded_stop_index: int,
    config: FallbackRegenerationConfig,
) -> tuple[int, int, str]:
    if config.rollback_ratio is not None:
        rollback_applied_tokens = min(
            bounded_stop_index,
            int(math.ceil(bounded_stop_index * config.rollback_ratio - 1e-12)),
        )
        rollback_mode = "ratio"
    else:
        rollback_tokens = (
            config.rollback_tokens if config.rollback_tokens is not None else 0
        )
        rollback_applied_tokens = min(bounded_stop_index, rollback_tokens)
        rollback_mode = "tokens"

    resume_index = bounded_stop_index - rollback_applied_tokens
    return resume_index, rollback_applied_tokens, rollback_mode


def _as_output_items(resp_json: Any) -> list[dict[str, Any]]:
    if isinstance(resp_json, list):
        return [item for item in resp_json if isinstance(item, dict)]
    if isinstance(resp_json, dict):
        if "meta_info" in resp_json:
            return [resp_json]
        text_list = resp_json.get("text")
        out_ids_list = resp_json.get("output_ids")
        meta_list = resp_json.get("meta_info")
        if (
            isinstance(text_list, list)
            and isinstance(out_ids_list, list)
            and isinstance(meta_list, list)
        ):
            items: list[dict[str, Any]] = []
            for idx in range(len(out_ids_list)):
                items.append(
                    {
                        "text": text_list[idx] if idx < len(text_list) else "",
                        "output_ids": out_ids_list[idx],
                        "meta_info": meta_list[idx] if idx < len(meta_list) else {},
                    }
                )
            return items
    raise ValueError("Unexpected /generate response schema")


def _derive_regen_seed(
    *, record: dict[str, Any], row_index: int, seed_offset: int
) -> int:
    raw_seed = record.get("seed")
    if raw_seed is None:
        return seed_offset + row_index
    try:
        return int(raw_seed) + seed_offset
    except (TypeError, ValueError):
        return seed_offset + row_index


def _build_fallback_payload(
    *,
    record: dict[str, Any],
    config: FallbackRegenerationConfig,
    regen_seed: int,
    rid: str,
    resume_output_ids: list[int],
    tokenizer: Any,
) -> tuple[dict[str, Any], list[int], str]:
    request_max_new_tokens = max(1, config.max_new_tokens - len(resume_output_ids))
    payload: dict[str, Any] = {
        "rid": rid,
        "sampling_params": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_new_tokens": request_max_new_tokens,
            "sampling_seed": regen_seed,
        },
        "stream": False,
    }

    input_ids = record.get("input_ids")
    if isinstance(input_ids, list):
        base_input_ids = [int(token_id) for token_id in input_ids]
        payload["input_ids"] = base_input_ids + resume_output_ids
        input_text = str(record.get("input_text") or "")
        return payload, base_input_ids, input_text

    input_text = record.get("input_text")
    if isinstance(input_text, str) and input_text:
        prefix_text = decode_response(tokenizer, resume_output_ids)
        payload["text"] = input_text + prefix_text
        return payload, [], input_text

    raise ValueError(
        "fallback_regenerate_if_pruned requires input_ids or input_text in each record"
    )


async def _fallback_regenerate_async(
    *,
    requests: list[tuple[int, int, dict[str, Any]]],
    config: FallbackRegenerationConfig,
    tokenizer: Any,
    completed_traces: dict[int, FallbackRegenerationTrace] | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_fsync_interval: int = 1,
) -> dict[int, FallbackRegenerationTrace]:
    existing_traces = dict(completed_traces or {})
    pending_requests = [
        request for request in requests if int(request[0]) not in existing_traces
    ]
    if not pending_requests:
        return existing_traces

    semaphore = asyncio.Semaphore(config.concurrency_limit)
    request_prefix = f"fallback_rg_{os.getpid()}_{time.time_ns():x}"
    async with httpx.AsyncClient(trust_env=False) as client:

        async def _run_one(
            row_index: int,
            stop_index: int,
            record: dict[str, Any],
        ) -> FallbackRegenerationTrace:
            original_output_ids_raw = record.get("output_ids")
            if not isinstance(original_output_ids_raw, list):
                raise ValueError(
                    f"fallback_regenerate_if_pruned requires output_ids for row {row_index}"
                )

            original_output_ids = [
                int(token_id) for token_id in original_output_ids_raw
            ]
            bounded_stop_index = max(0, min(stop_index, len(original_output_ids)))
            resume_index, rollback_applied_tokens, rollback_mode = (
                _compute_resume_index(
                    bounded_stop_index=bounded_stop_index,
                    config=config,
                )
            )
            resume_output_ids = original_output_ids[:resume_index]

            regen_seed = _derive_regen_seed(
                record=record,
                row_index=row_index,
                seed_offset=config.seed_offset,
            )
            rid = f"{request_prefix}_row_{row_index}"
            payload, base_input_ids, input_text = _build_fallback_payload(
                record=record,
                config=config,
                regen_seed=regen_seed,
                rid=rid,
                resume_output_ids=resume_output_ids,
                tokenizer=tokenizer,
            )

            async with semaphore:
                resp = await client.post(
                    f"{config.api_url}/generate",
                    json=payload,
                    timeout=config.timeout_s,
                )
            if resp.status_code != 200:
                raise RuntimeError(
                    "SGL /generate failed during fallback_regenerate_if_pruned. "
                    f"status={resp.status_code}, body={resp.text[:1000]}"
                )
            items = _as_output_items(resp.json())
            if len(items) != 1:
                raise RuntimeError(
                    f"Expected single fallback regeneration output, got {len(items)}"
                )
            output = items[0]

            continuation_output_ids_raw = output.get("output_ids")
            if not isinstance(continuation_output_ids_raw, list):
                raise ValueError(
                    f"Missing output_ids in fallback regeneration result for row {row_index}"
                )
            continuation_output_ids = [
                int(token_id) for token_id in continuation_output_ids_raw
            ]
            merged_output_ids = resume_output_ids + continuation_output_ids

            regenerated_row = copy.deepcopy(record)
            if base_input_ids:
                regenerated_row["input_ids"] = base_input_ids
            if input_text:
                regenerated_row["input_text"] = input_text
            regenerated_row["seed"] = regen_seed
            regenerated_row["response"] = decode_response(tokenizer, merged_output_ids)
            regenerated_row["output_ids"] = merged_output_ids
            regenerated_row["response_len_tokens"] = len(merged_output_ids)
            regenerated_row["probs"] = []
            finish_reason = (
                output.get("meta_info", {}).get("finish_reason", {}).get("type")
            )
            regenerated_row["finish_reason"] = (
                str(finish_reason) if finish_reason is not None else None
            )
            return FallbackRegenerationTrace(
                row_index=row_index,
                regenerated_row=regenerated_row,
                regenerated_seed=regen_seed,
                stop_index=bounded_stop_index,
                resume_index=resume_index,
                rollback_applied_tokens=rollback_applied_tokens,
                rollback_mode=rollback_mode,
                continuation_output_ids=continuation_output_ids,
            )

        tasks = [
            _run_one(row_index, stop_index, record)
            for row_index, stop_index, record in pending_requests
        ]
        traces = dict(existing_traces)
        checkpoint_handle = (
            checkpoint_path.open("ab") if checkpoint_path is not None else None
        )
        try:
            with tqdm(
                total=len(requests),
                initial=len(existing_traces),
                desc="Fallback regenerating",
                unit="sample",
            ) as pbar:
                completed_count = len(existing_traces)
                for future in asyncio.as_completed(tasks):
                    trace = await future
                    traces[trace.row_index] = trace
                    completed_count += 1
                    if checkpoint_handle is not None:
                        checkpoint_handle.write(
                            orjson.dumps(
                                asdict(trace), option=orjson.OPT_APPEND_NEWLINE
                            )
                        )
                        checkpoint_handle.flush()
                        if (
                            checkpoint_fsync_interval > 0
                            and completed_count % checkpoint_fsync_interval == 0
                        ):
                            os.fsync(checkpoint_handle.fileno())
                    pbar.update(1)
        finally:
            if checkpoint_handle is not None:
                checkpoint_handle.flush()
                if checkpoint_fsync_interval > 0:
                    os.fsync(checkpoint_handle.fileno())
                checkpoint_handle.close()
    return traces


def _load_fallback_checkpoint(
    checkpoint_path: Path,
) -> dict[int, FallbackRegenerationTrace]:
    if not checkpoint_path.exists():
        return {}

    traces: dict[int, FallbackRegenerationTrace] = {}
    invalid_lines = 0
    for raw_line in checkpoint_path.read_bytes().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = orjson.loads(line)
        except orjson.JSONDecodeError:
            invalid_lines += 1
            continue
        if not isinstance(payload, dict):
            invalid_lines += 1
            continue
        try:
            trace = FallbackRegenerationTrace(
                row_index=int(payload["row_index"]),
                regenerated_row=dict(payload["regenerated_row"]),
                regenerated_seed=(
                    int(payload["regenerated_seed"])
                    if payload.get("regenerated_seed") is not None
                    else None
                ),
                stop_index=int(payload["stop_index"]),
                resume_index=int(payload["resume_index"]),
                rollback_applied_tokens=int(payload["rollback_applied_tokens"]),
                rollback_mode=str(payload["rollback_mode"]),
                continuation_output_ids=[
                    int(token_id) for token_id in payload["continuation_output_ids"]
                ],
            )
        except (KeyError, TypeError, ValueError):
            invalid_lines += 1
            continue
        traces[trace.row_index] = trace

    if invalid_lines:
        print(
            f"Warning: ignored {invalid_lines} invalid fallback checkpoint line(s) in {checkpoint_path}"
        )
    return traces


def fallback_regenerate_pruned_records(
    *,
    records: list[dict[str, Any]],
    prune_stop_indices: dict[int, int],
    original_args: dict[str, Any],
    strategy_args: dict[str, Any],
    tokenizer: Any,
    checkpoint_path: Path | None = None,
    checkpoint_fsync_interval: int = 1,
) -> tuple[FallbackRegenerationConfig, dict[int, FallbackRegenerationTrace]]:
    config = _build_fallback_regeneration_config(
        original_args=original_args,
        strategy_args=strategy_args,
    )
    if not prune_stop_indices:
        return config, {}

    requests = [
        (row_index, stop_index, records[row_index])
        for row_index, stop_index in sorted(prune_stop_indices.items())
    ]
    completed_traces: dict[int, FallbackRegenerationTrace] = {}
    if checkpoint_path is not None:
        valid_indices = set(prune_stop_indices)
        completed_traces = {
            row_index: trace
            for row_index, trace in _load_fallback_checkpoint(checkpoint_path).items()
            if row_index in valid_indices
        }
        if completed_traces:
            print(
                "Resuming fallback_regenerate_if_pruned from checkpoint: "
                f"{len(completed_traces)} / {len(prune_stop_indices)} records complete"
            )
    replacements = asyncio.run(
        _fallback_regenerate_async(
            requests=requests,
            config=config,
            tokenizer=tokenizer,
            completed_traces=completed_traces,
            checkpoint_path=checkpoint_path,
            checkpoint_fsync_interval=checkpoint_fsync_interval,
        )
    )
    missing = sorted(set(prune_stop_indices) - set(replacements))
    if missing:
        raise RuntimeError(f"Missing fallback-regenerated rows for indices: {missing}")
    return config, replacements


def _write_fallback_v2_run(
    *,
    out_dir: Path,
    args_json: dict[str, Any],
    original_records: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    indexed_outcomes: dict[int, tuple[Any, str]],
    fallback_traces: dict[int, FallbackRegenerationTrace],
    fallback_config: FallbackRegenerationConfig | None,
    tokenizer: Any,
) -> None:
    overview = build_run_overview(args_json, out_dir)
    reference_map = _reference_pass_map(out_dir, args_json)
    base_model_name = overview.model_name or overview.model
    fallback_model_name = (
        str(fallback_config.model_name)
        if fallback_config is not None
        else base_model_name
    )

    segments: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    final_outputs: list[dict[str, Any]] = []
    occ_counter: dict[str, int] = defaultdict(int)

    for row_idx, final_row in enumerate(final_rows):
        sample_id = str(final_row.get("example_id"))
        occ_idx = occ_counter[sample_id]
        occ_counter[sample_id] += 1
        output_id = f"{sample_id}__{occ_idx}"
        final_segment_id = f"segment__{sample_id}__{occ_idx}"
        finish_reason = str(final_row.get("finish_reason") or "") or None
        final_output_ids = (
            [int(token_id) for token_id in final_row.get("output_ids", [])]
            if isinstance(final_row.get("output_ids"), list)
            else []
        )
        final_response = str(final_row.get("response") or "")
        final_seed = (
            int(final_row["seed"]) if final_row.get("seed") is not None else None
        )
        source_output_tokens = (
            len(original_records[row_idx]["output_ids"])
            if isinstance(original_records[row_idx].get("output_ids"), list)
            else len(final_output_ids)
        )
        trace = fallback_traces.get(row_idx)
        if trace is not None:
            outcome, _ = indexed_outcomes[row_idx]
            decision = outcome.decision
            original_output_ids_raw = original_records[row_idx].get("output_ids")
            original_probs_raw = original_records[row_idx].get("probs")
            if not isinstance(original_output_ids_raw, list):
                raise ValueError(
                    f"Missing output_ids for fallback online row {row_idx}"
                )
            original_output_ids = [
                int(token_id) for token_id in original_output_ids_raw
            ]
            original_probs = (
                original_probs_raw if isinstance(original_probs_raw, list) else []
            )

            pruned_output_ids = original_output_ids[: trace.stop_index]
            pruned_probs = [
                float(value)
                for value in original_probs[: trace.stop_index]
                if isinstance(value, (int, float))
            ]
            pruned_segment_id = f"{final_segment_id}__pruned"
            pruned_response = decode_response(tokenizer, pruned_output_ids)
            pruned_seed = (
                int(original_records[row_idx]["seed"])
                if original_records[row_idx].get("seed") is not None
                else None
            )
            resume_output_ids = original_output_ids[: trace.resume_index]
            resume_segment_id = f"{final_segment_id}__resume"
            resume_response = decode_response(tokenizer, resume_output_ids)
            prune_meta = (
                final_row.get("prune_meta", {})
                if isinstance(final_row.get("prune_meta"), dict)
                else {}
            )

            segments.append(
                GenerationSegment(
                    segment_id=pruned_segment_id,
                    sample_id=sample_id,
                    parent_segment_id=None,
                    executor_name="model",
                    model_name=base_model_name,
                    seed=pruned_seed,
                    output_ids=pruned_output_ids,
                    response_text=pruned_response,
                    probs=pruned_probs,
                    usage=SegmentUsage(
                        input_tokens=0,
                        output_tokens=len(pruned_output_ids),
                        cached_input_tokens=0,
                        request_count=1,
                    ),
                    visible_output_tokens=0,
                    source_row_index=row_idx,
                ).to_dict()
            )
            decisions.append(
                DecisionRecord(
                    decision_type="prune_stop",
                    sample_id=sample_id,
                    output_id=output_id,
                    segment_id=pruned_segment_id,
                    original_output_tokens=source_output_tokens,
                    kept_output_tokens=len(pruned_output_ids),
                    discarded_output_tokens=0,
                    reason=str(prune_meta.get("reason") or decision.reason) or None,
                    meta={
                        "postprocess_strategy": prune_meta.get("postprocess_strategy"),
                        "logical_cut_index": outcome.logical_cut_index,
                        "detect_pos": outcome.detect_pos,
                        "stop_index": trace.stop_index,
                        "resume_index": trace.resume_index,
                        "simulation_mode": "strict_online",
                    },
                ).to_dict()
            )
            segments.append(
                GenerationSegment(
                    segment_id=resume_segment_id,
                    sample_id=sample_id,
                    parent_segment_id=pruned_segment_id,
                    executor_name="model",
                    model_name=base_model_name,
                    seed=pruned_seed,
                    output_ids=resume_output_ids,
                    response_text=resume_response,
                    probs=[],
                    usage=SegmentUsage(
                        input_tokens=0,
                        output_tokens=len(resume_output_ids),
                        cached_input_tokens=0,
                        request_count=1,
                    ),
                    visible_output_tokens=0,
                    source_row_index=row_idx,
                ).to_dict()
            )
            decisions.append(
                DecisionRecord(
                    decision_type="rollback",
                    sample_id=sample_id,
                    output_id=output_id,
                    segment_id=resume_segment_id,
                    original_output_tokens=trace.stop_index,
                    kept_output_tokens=trace.resume_index,
                    discarded_output_tokens=trace.rollback_applied_tokens,
                    reason="postprocess fallback_regenerate_if_pruned rolled back online prefix",
                    meta={
                        "regenerated_seed": trace.regenerated_seed,
                        "stop_index": trace.stop_index,
                        "resume_index": trace.resume_index,
                        "rollback_applied_tokens": trace.rollback_applied_tokens,
                        "rollback_mode": trace.rollback_mode,
                        "simulation_mode": "strict_online",
                    },
                ).to_dict()
            )
            parent_segment_id = resume_segment_id
            was_pruned = True
        else:
            parent_segment_id = None
            was_pruned = False

        segments.append(
            GenerationSegment(
                segment_id=final_segment_id,
                sample_id=sample_id,
                parent_segment_id=parent_segment_id,
                executor_name="model",
                model_name=fallback_model_name
                if trace is not None
                else base_model_name,
                seed=final_seed,
                output_ids=final_output_ids,
                response_text=final_response,
                probs=[],
                usage=SegmentUsage(
                    input_tokens=0,
                    output_tokens=len(final_output_ids),
                    cached_input_tokens=0,
                    request_count=1,
                ),
                visible_output_tokens=len(final_output_ids),
                source_row_index=row_idx,
            ).to_dict()
        )
        decisions.append(
            DecisionRecord(
                decision_type="finalize_output",
                sample_id=sample_id,
                output_id=output_id,
                segment_id=final_segment_id,
                kept_output_tokens=len(final_output_ids),
                reason=finish_reason,
            ).to_dict()
        )
        final_outputs.append(
            FinalOutput(
                output_id=output_id,
                sample_id=sample_id,
                final_segment_id=final_segment_id,
                response=final_response,
                finish_reason=finish_reason,
                k_group=occ_idx + 1,
                source_row_index=row_idx,
                was_pruned=was_pruned,
                would_pass_without_prune=reference_map.get((sample_id, occ_idx)),
                source_output_tokens=source_output_tokens,
            ).to_dict()
        )

    dump_json(out_dir / RUN_OVERVIEW_FILE, overview.to_dict())
    write_jsonl(out_dir / GENERATION_SEGMENTS_FILE, segments)
    write_jsonl(out_dir / DECISION_LOG_FILE, decisions)
    write_jsonl(out_dir / FINAL_OUTPUTS_FILE, final_outputs)
    write_jsonl(out_dir / SOURCE_TARGETS_FILE, build_source_targets(final_rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict online replay for probe pruning with fallback-regenerate control. "
            "This keeps strict online stop semantics, rolls back a short suffix before the "
            "cut point, then asks model API to continue from that resumed prefix."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing exactly one '*.probed.jsonl' file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results/probe_prune_online_fallback_rg",
        help="Output base directory used when --output-dir is not provided.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="run",
        help="Experiment subdirectory name under --output.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Exact output run directory. If provided, --output and --experiment-name are ignored.",
    )
    parser.add_argument(
        "--result-file-name",
        type=str,
        default="generation_results.jsonl",
        help="Result filename inside output directory.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="first_below",
        help="Second-level generation strategy name.",
    )
    parser.add_argument(
        "--strategy-args-json",
        type=str,
        default="",
        help="Inline JSON object for second-level strategy args.",
    )
    parser.add_argument(
        "--strategy-args-file",
        type=str,
        default="",
        help="JSON file path for second-level strategy args.",
    )
    parser.add_argument(
        "--input-group-key",
        type=str,
        default="example_id",
        help="Field name used to group records as one sample for first-level online replay.",
    )
    parser.add_argument(
        "--input-strategy",
        type=str,
        default="delegate_second_level",
        help=(
            "First-level strict online strategy name. "
            "Supported: delegate_second_level, group_consensus_gate, group_quorum_snapshot_floor."
        ),
    )
    parser.add_argument(
        "--input-strategy-args-json",
        type=str,
        default="",
        help="Inline JSON object for first-level strategy args.",
    )
    parser.add_argument(
        "--input-strategy-args-file",
        type=str,
        default="",
        help="JSON file path for first-level strategy args.",
    )
    parser.add_argument(
        "--fallback-args-json",
        type=str,
        default="",
        help="Inline JSON object for fallback_regenerate_if_pruned args.",
    )
    parser.add_argument(
        "--fallback-args-file",
        type=str,
        default="",
        help="JSON file path for fallback_regenerate_if_pruned args.",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--min-keep-tokens", type=int, default=0)
    parser.add_argument("--arm-threshold", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--warmup-valid-points", type=int, default=3)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--drop-threshold", type=float, default=0.2)
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="",
        help=(
            "Tokenizer path used to decode resumed prefix and merged output text. "
            "Required for fallback_regenerate_if_pruned."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Process only first N records for debug (0 means all).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume fallback_regenerate_if_pruned API calls from an incremental checkpoint in the output directory.",
    )
    parser.add_argument(
        "--checkpoint-file-name",
        type=str,
        default=".fallback_regenerate_if_pruned.checkpoint.jsonl",
        help="Incremental fallback regeneration checkpoint JSONL filename inside the output directory.",
    )
    parser.add_argument(
        "--checkpoint-fsync-interval",
        type=int,
        default=1,
        help="fsync checkpoint every N completed fallback regenerations (0 disables fsync).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    input_path = resolve_input_file(input_dir)
    original_args = load_original_args(input_dir)

    if args.min_keep_tokens < 0:
        raise ValueError("--min-keep-tokens must be >= 0")
    if args.max_records < 0:
        raise ValueError("--max-records must be >= 0")
    if not args.input_group_key:
        raise ValueError("--input-group-key must be non-empty")
    if not args.result_file_name:
        raise ValueError("--result-file-name must be a non-empty filename")
    if not args.checkpoint_file_name.strip():
        raise ValueError("--checkpoint-file-name must be non-empty")
    if args.checkpoint_fsync_interval < 0:
        raise ValueError("--checkpoint-fsync-interval must be >= 0")
    effective_tokenizer_path = str(args.tokenizer_path or "").strip()
    if not effective_tokenizer_path:
        raise ValueError(
            "--tokenizer-path is required for fallback_regenerate_if_pruned"
        )

    strategy_args = merge_strategy_args(
        defaults=legacy_defaults_for_strategy(args),
        args_file=args.strategy_args_file,
        args_json=args.strategy_args_json,
        args_file_name="--strategy-args-file",
        args_json_name="--strategy-args-json",
    )
    raw_input_strategy_args = merge_strategy_args(
        defaults={},
        args_file=args.input_strategy_args_file,
        args_json=args.input_strategy_args_json,
        args_file_name="--input-strategy-args-file",
        args_json_name="--input-strategy-args-json",
    )
    fallback_strategy_args = merge_strategy_args(
        defaults={},
        args_file=args.fallback_args_file,
        args_json=args.fallback_args_json,
        args_file_name="--fallback-args-file",
        args_json_name="--fallback-args-json",
    )
    (
        input_strategy_args,
        embedded_generation_strategy_name,
        embedded_generation_strategy_args,
    ) = split_input_strategy_args(
        input_strategy_name=args.input_strategy,
        input_strategy_args=raw_input_strategy_args,
    )

    if embedded_generation_strategy_name is not None:
        resolved_generation_strategy_name = embedded_generation_strategy_name
        resolved_generation_strategy_args = embedded_generation_strategy_args
    else:
        resolved_generation_strategy_name = args.strategy
        resolved_generation_strategy_args = strategy_args

    generation_strategy = build_strategy(
        strategy_name=resolved_generation_strategy_name,
        strategy_args=resolved_generation_strategy_args,
    )

    records = load_jsonl(input_path)
    if args.max_records > 0:
        records = records[: args.max_records]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        effective_tokenizer_path,
        trust_remote_code=True,
    )

    indexed_outcomes, input_group_count = build_group_outcomes(
        records=records,
        input_group_key=args.input_group_key,
        input_strategy_name=args.input_strategy,
        input_strategy_args=input_strategy_args,
        generation_strategy=generation_strategy,
    )
    prune_stop_indices: dict[int, int] = {}
    for record_idx, (outcome, _) in indexed_outcomes.items():
        decision = outcome.decision
        if not decision.should_prune or decision.cut_index is None:
            continue
        output_ids = records[record_idx].get("output_ids")
        probs = records[record_idx].get("probs")
        if not isinstance(output_ids, list):
            continue
        if not isinstance(probs, list) or len(probs) != len(output_ids):
            continue
        stop_index = max(0, min(int(decision.cut_index), len(output_ids)))
        if stop_index >= len(output_ids):
            continue
        prune_stop_indices[record_idx] = stop_index

    out_dir: Path | None = None
    output_path: Path | None = None
    checkpoint_path: Path | None = None
    if not args.dry_run:
        out_dir = prepare_output_dir(args, input_path)
        output_path = out_dir / args.result_file_name
        checkpoint_path = out_dir / args.checkpoint_file_name

    fallback_config = None
    fallback_traces: dict[int, FallbackRegenerationTrace] = {}
    fallback_regenerated = 0
    if prune_stop_indices:
        fallback_config, fallback_traces = fallback_regenerate_pruned_records(
            records=records,
            prune_stop_indices=prune_stop_indices,
            original_args=original_args,
            strategy_args=fallback_strategy_args,
            tokenizer=tokenizer,
            checkpoint_path=checkpoint_path if args.resume else None,
            checkpoint_fsync_interval=args.checkpoint_fsync_interval,
        )
        fallback_regenerated = len(fallback_traces)

    total = 0
    pruned = 0
    unchanged = 0
    empty_after_prune = 0

    out_rows: list[dict[str, Any]] = []
    for record_idx, record in enumerate(records):
        total += 1
        row = copy.deepcopy(record)
        outcome, input_group_value = indexed_outcomes[record_idx]
        decision = outcome.decision

        output_ids = row.get("output_ids")
        probs = row.get("probs")
        if not isinstance(output_ids, list):
            raise ValueError("Each record must contain list[int] field: output_ids")

        if not isinstance(probs, list) or len(probs) != len(output_ids):
            unchanged += 1
            row["prune_meta"] = {
                "strategy": (
                    args.input_strategy
                    if args.input_strategy != "delegate_second_level"
                    else generation_strategy.name
                ),
                "generation_strategy": generation_strategy.name,
                "input_strategy": args.input_strategy,
                "input_group_key_field": args.input_group_key,
                "input_group_value": input_group_value,
                "decision": "keep",
                "reason": decision.reason,
                "simulation_mode": "strict_online",
                "candidate_count": outcome.candidate_count,
                "release_count": outcome.release_count,
            }
            out_rows.append(row)
            continue

        if not decision.should_prune or decision.cut_index is None:
            unchanged += 1
            row["prune_meta"] = {
                "strategy": (
                    args.input_strategy
                    if args.input_strategy != "delegate_second_level"
                    else generation_strategy.name
                ),
                "generation_strategy": generation_strategy.name,
                "input_strategy": args.input_strategy,
                "input_group_key_field": args.input_group_key,
                "input_group_value": input_group_value,
                "decision": "keep",
                "reason": decision.reason,
                "simulation_mode": "strict_online",
                "candidate_count": outcome.candidate_count,
                "release_count": outcome.release_count,
            }
            out_rows.append(row)
            continue

        cut_index = max(0, min(int(decision.cut_index), len(output_ids)))
        if cut_index >= len(output_ids):
            unchanged += 1
            row["prune_meta"] = {
                "strategy": (
                    args.input_strategy
                    if args.input_strategy != "delegate_second_level"
                    else generation_strategy.name
                ),
                "generation_strategy": generation_strategy.name,
                "input_strategy": args.input_strategy,
                "input_group_key_field": args.input_group_key,
                "input_group_value": input_group_value,
                "decision": "keep",
                "reason": f"{decision.reason}; online stop_index >= output_len after clamp",
                "simulation_mode": "strict_online",
                "candidate_count": outcome.candidate_count,
                "release_count": outcome.release_count,
                "logical_cut_index": outcome.logical_cut_index,
                "detect_pos": outcome.detect_pos,
                "stop_index": cut_index,
            }
            out_rows.append(row)
            continue

        fallback_trace = fallback_traces.get(record_idx)
        if fallback_trace is not None:
            regenerated_row = copy.deepcopy(fallback_trace.regenerated_row)
            regenerated_row["prune_meta"] = {
                "strategy": (
                    args.input_strategy
                    if args.input_strategy != "delegate_second_level"
                    else generation_strategy.name
                ),
                "generation_strategy": generation_strategy.name,
                "input_strategy": args.input_strategy,
                "input_group_key_field": args.input_group_key,
                "input_group_value": input_group_value,
                "decision": "prune_then_fallback_regenerate",
                "reason": decision.reason,
                "trigger_pos": decision.trigger_pos,
                "trigger_prob": decision.trigger_prob,
                "cut_index": cut_index,
                "resume_index": fallback_trace.resume_index,
                "rollback_applied_tokens": fallback_trace.rollback_applied_tokens,
                "rollback_mode": fallback_trace.rollback_mode,
                "configured_rollback_tokens": (
                    fallback_config.rollback_tokens
                    if fallback_config is not None
                    else None
                ),
                "configured_rollback_ratio": (
                    fallback_config.rollback_ratio
                    if fallback_config is not None
                    else None
                ),
                "original_len": len(output_ids),
                "pruned_len": cut_index,
                "logical_cut_index": outcome.logical_cut_index,
                "detect_pos": outcome.detect_pos,
                "stop_index": cut_index,
                "simulation_mode": "strict_online",
                "candidate_count": outcome.candidate_count,
                "release_count": outcome.release_count,
                "final_state": outcome.final_state,
                "postprocess_strategy": "fallback_regenerate_if_pruned",
                "postprocess_strategy_args": dict(fallback_strategy_args),
                "original_seed": row.get("seed"),
                "regenerated_seed": fallback_trace.regenerated_seed,
                "continuation_len": len(fallback_trace.continuation_output_ids),
            }
            pruned += 1
            out_rows.append(regenerated_row)
            continue

        pruned_output_ids = output_ids[:cut_index]
        pruned_probs = probs[:cut_index]

        row["output_ids"] = pruned_output_ids
        row["probs"] = pruned_probs
        row["response_len_tokens"] = len(pruned_output_ids)
        row["finish_reason"] = "prune"
        row["response"] = decode_response(tokenizer, pruned_output_ids)
        row["prune_meta"] = {
            "strategy": (
                args.input_strategy
                if args.input_strategy != "delegate_second_level"
                else generation_strategy.name
            ),
            "generation_strategy": generation_strategy.name,
            "input_strategy": args.input_strategy,
            "input_group_key_field": args.input_group_key,
            "input_group_value": input_group_value,
            "decision": "prune",
            "reason": decision.reason,
            "trigger_pos": decision.trigger_pos,
            "trigger_prob": decision.trigger_prob,
            "cut_index": cut_index,
            "original_len": len(output_ids),
            "pruned_len": len(pruned_output_ids),
            "logical_cut_index": outcome.logical_cut_index,
            "detect_pos": outcome.detect_pos,
            "stop_index": cut_index,
            "simulation_mode": "strict_online",
            "candidate_count": outcome.candidate_count,
            "release_count": outcome.release_count,
            "final_state": outcome.final_state,
        }

        if len(pruned_output_ids) == 0:
            empty_after_prune += 1
        pruned += 1
        out_rows.append(row)

    if not args.dry_run:
        assert out_dir is not None
        assert output_path is not None

        args_json = copy.deepcopy(original_args)
        args_json["probe_prune"] = {
            "input_file": str(input_path),
            "input_dir": str(input_dir),
            "output": str(out_dir),
            "result_file": args.result_file_name,
            "experiment_name": args.experiment_name,
            "output_base": args.output,
            "output_dir_override": args.output_dir or None,
            "max_records": args.max_records,
            "tokenizer_path": effective_tokenizer_path,
            "keep_prune_meta": True,
            "dry_run": args.dry_run,
            "simulation_mode": "strict_online",
            "resume": args.resume,
            "checkpoint_file": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_fsync_interval": args.checkpoint_fsync_interval,
            "prune_config": build_prune_config(
                input_group_key=args.input_group_key,
                input_strategy_name=args.input_strategy,
                input_strategy_args=input_strategy_args,
                generation_strategy_name=generation_strategy.name,
                generation_strategy_args=resolved_generation_strategy_args,
                postprocess_strategy_name="fallback_regenerate_if_pruned",
                postprocess_strategy_args=fallback_strategy_args,
            ),
        }
        if fallback_config is not None:
            args_json["probe_prune"]["postprocess_runtime"] = {
                "model_name": fallback_config.model_name,
                "temperature": fallback_config.temperature,
                "top_p": fallback_config.top_p,
                "top_k": fallback_config.top_k,
                "max_new_tokens": fallback_config.max_new_tokens,
                "api_url": fallback_config.api_url,
                "seed_offset": fallback_config.seed_offset,
                "timeout_s": fallback_config.timeout_s,
                "concurrency_limit": fallback_config.concurrency_limit,
                "rollback_tokens": fallback_config.rollback_tokens,
                "rollback_ratio": fallback_config.rollback_ratio,
            }

        dump_json(out_dir / "args.json", args_json)
        assert output_path is not None
        write_jsonl(output_path, out_rows)
        _write_fallback_v2_run(
            out_dir=out_dir,
            args_json=args_json,
            original_records=records,
            final_rows=out_rows,
            indexed_outcomes=indexed_outcomes,
            fallback_traces=fallback_traces,
            fallback_config=fallback_config,
            tokenizer=tokenizer,
        )

    root_strategy_name = (
        generation_strategy.name
        if args.input_strategy == "delegate_second_level"
        else args.input_strategy
    )
    print(f"Input records: {total}")
    print(f"Pruned records: {pruned}")
    print(f"Fallback-regenerated records: {fallback_regenerated}")
    print(f"Unchanged records: {unchanged}")
    print(f"Empty after prune: {empty_after_prune}")
    print(f"Input groups ({args.input_group_key}): {input_group_count}")

    print("-" * 30)
    print("Simulation mode: strict_online")
    print(f"Root prune strategy: {root_strategy_name}")
    if input_strategy_args:
        print(
            f"First-level strategy args: {json_text(input_strategy_args, sort_keys=True)}"
        )
    print(f"Second-level strategy: {generation_strategy.name}")
    print(
        f"Second-level strategy args: {json_text(resolved_generation_strategy_args, sort_keys=True)}"
    )
    print("Control strategy: fallback_regenerate_if_pruned")
    print(f"Control strategy args: {json_text(fallback_strategy_args, sort_keys=True)}")
    print("-" * 30)

    if args.dry_run:
        print("Dry run only, output file not written.")
    else:
        assert out_dir is not None
        assert output_path is not None
        print(f"Output directory: {out_dir}")
        print(f"Saved strict online replay results to: {output_path}")


if __name__ == "__main__":
    main()
