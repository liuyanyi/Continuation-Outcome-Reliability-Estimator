#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from pathlib import Path
from typing import Any

from cognitive_probe.evaluation.deepconf_online import run_deepconf_online_for_sample
from cognitive_probe.evaluation.io import dump_json, load_jsonl, write_jsonl
from cognitive_probe.evaluation.probe_prune.common import (
    decode_response,
    prepare_output_dir,
    resolve_input_file,
)
from cognitive_probe.evaluation.v2 import migrate_legacy_run


def _load_json(path: Path) -> dict[str, Any]:
    import orjson

    payload = orjson.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _resolve_tokenizer_path(
    cli_tokenizer_path: str,
    original_args: dict[str, Any],
) -> str:
    explicit = cli_tokenizer_path.strip()
    if explicit:
        return explicit

    candidates: list[Any] = []
    api_generation = original_args.get("api_generation")
    if isinstance(api_generation, dict):
        candidates.append(api_generation.get("tokenizer_path"))
    probe_prune = original_args.get("probe_prune")
    if isinstance(probe_prune, dict):
        candidates.append(probe_prune.get("tokenizer_path"))
    probe = original_args.get("probe")
    if isinstance(probe, dict):
        candidates.append(probe.get("tokenizer_path"))
        candidates.append(probe.get("base_model_path"))
    candidates.append(original_args.get("tokenizer_path"))
    candidates.append(original_args.get("model"))

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _load_tokenizer(tokenizer_path: str) -> Any | None:
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if Path(tokenizer_path).exists():
        kwargs["local_files_only"] = True
    return AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)


def _build_final_row(
    *,
    original_row: dict[str, Any],
    stop_index: int,
    response: str,
) -> dict[str, Any]:
    row = copy.deepcopy(original_row)
    output_ids = row.get("output_ids")
    probs = row.get("probs")
    if not isinstance(output_ids, list):
        raise ValueError("Each record must contain list[int] field: output_ids")
    if not isinstance(probs, list):
        raise ValueError("Each record must contain list[float] field: probs")

    bounded_stop_index = max(0, min(int(stop_index), len(output_ids)))
    row["output_ids"] = [int(token_id) for token_id in output_ids[:bounded_stop_index]]
    row["response_len_tokens"] = bounded_stop_index
    row["probs"] = list(probs[:bounded_stop_index])
    row["response"] = response
    row["finish_reason"] = (
        "prune" if bounded_stop_index < len(output_ids) else row.get("finish_reason")
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay DeepConf online on precomputed DeepConf token scores. "
            "This uses per-sample warmup traces to estimate a threshold, then applies "
            "group-level early stop and weighted consensus voting across a fixed per-sample budget."
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
        default="evaluation/results/deepconf_online",
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
        "--input-group-key",
        type=str,
        default="example_id",
        help="Field used to group rollout records into one DeepConf sample.",
    )
    parser.add_argument(
        "--total-budget",
        type=int,
        default=32,
        help="Maximum number of traces to use per sample, including warmup traces.",
    )
    parser.add_argument(
        "--warmup-traces",
        type=int,
        default=16,
        help="Number of full warmup traces per sample.",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=0.10,
        help=(
            "Fraction of best warmup traces kept as the pruning cutoff. "
            "DeepConf scores are higher-is-better: the threshold is the "
            "(1 - keep-ratio) percentile of minimum group scores, and adaptive "
            "traces prune when a group score falls below it."
        ),
    )
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.95,
        help=(
            "Consensus level tracked in metadata and summaries. "
            "This does not stop sample generation early; all traces up to --total-budget are kept."
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=32,
        help="Number of valid token scores per DeepConf group.",
    )
    parser.add_argument(
        "--vote-weight",
        type=str,
        default="raw",
        choices=("raw",),
        help="Use the higher-is-better trace confidence directly as the vote weight.",
    )
    parser.add_argument(
        "--exclude-warmup-from-vote",
        action="store_true",
        help="Do not add warmup traces into the weighted vote pool.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="",
        help=(
            "Tokenizer path used to decode truncated output_ids back to response text. "
            "If empty, the script tries to resolve it from the source run args.json."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Only process the first N grouped samples for debugging (0 means all).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.total_budget <= 0:
        raise ValueError("--total-budget must be > 0")
    if args.warmup_traces < 0:
        raise ValueError("--warmup-traces must be >= 0")
    if args.keep_ratio <= 0.0 or args.keep_ratio > 1.0:
        raise ValueError("--keep-ratio must be in (0, 1]")
    if args.consensus_threshold <= 0.0 or args.consensus_threshold > 1.0:
        raise ValueError("--consensus-threshold must be in (0, 1]")
    if args.group_size <= 0:
        raise ValueError("--group-size must be > 0")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")

    input_dir = Path(args.input_dir)
    input_path = resolve_input_file(input_dir)
    original_args = _load_json(input_dir / "args.json") if (input_dir / "args.json").exists() else {}
    records = load_jsonl(input_path)
    if not records:
        raise ValueError("No records found in input JSONL")

    for row_idx, row in enumerate(records):
        output_ids = row.get("output_ids")
        probs = row.get("probs")
        if not isinstance(output_ids, list):
            raise ValueError(f"Record {row_idx} missing list output_ids")
        if not isinstance(probs, list) or len(probs) != len(output_ids):
            raise ValueError(
                f"Record {row_idx} missing DeepConf scores or length mismatch: "
                f"len(probs)={len(probs) if isinstance(probs, list) else 'invalid'}, "
                f"len(output_ids)={len(output_ids)}"
            )
        if args.input_group_key not in row:
            raise ValueError(
                f"Record {row_idx} missing group key '{args.input_group_key}'"
            )

    tokenizer_path = _resolve_tokenizer_path(args.tokenizer_path, original_args)
    tokenizer = _load_tokenizer(tokenizer_path)
    decode_output_ids = None
    if tokenizer is not None:
        decode_output_ids = lambda token_ids: decode_response(tokenizer, token_ids)

    grouped_records: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for row_idx, row in enumerate(records):
        grouped_records[str(row.get(args.input_group_key))].append((row_idx, row))

    grouped_items = list(grouped_records.items())
    if args.max_samples > 0:
        grouped_items = grouped_items[: args.max_samples]

    final_rows: list[dict[str, Any]] = []
    trace_manifest_rows: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []

    for sample_id, indexed_rows in grouped_items:
        outcome = run_deepconf_online_for_sample(
            sample_id=sample_id,
            indexed_records=indexed_rows,
            total_budget=args.total_budget,
            warmup_traces=args.warmup_traces,
            keep_ratio=args.keep_ratio,
            consensus_threshold=args.consensus_threshold,
            group_size=args.group_size,
            lower_is_better=False,
            vote_weight=args.vote_weight,
            include_warmup_in_vote=not args.exclude_warmup_from_vote,
            decode_output_ids=decode_output_ids,
        )

        for trace_idx, trace in enumerate(outcome.traces):
            original_row = records[trace.source_row_index]
            final_row = _build_final_row(
                original_row=original_row,
                stop_index=trace.stop_index,
                response=trace.response,
            )
            final_row["prune_meta"] = {
                "strategy": "deepconf_online",
                "generation_strategy": "deepconf_online",
                "input_strategy": "deepconf_online",
                "input_group_key_field": args.input_group_key,
                "input_group_value": sample_id,
                "decision": "prune" if trace.stopped_early else "keep",
                "reason": trace.stop_reason,
                "phase": trace.phase,
                "source_row_index": trace.source_row_index,
                "cut_index": trace.stop_index,
                "stop_index": trace.stop_index,
                "original_len": trace.output_len,
                "pruned_len": trace.stop_index,
                "threshold": trace.threshold,
                "group_scores": trace.group_scores,
                "trace_score": trace.trace_score,
                "trace_weight": trace.trace_weight,
                "stopped_early": trace.stopped_early,
                "vote_answer": trace.answer,
                "vote_applied": trace.vote_applied,
                "consensus_answer": trace.consensus_answer,
                "consensus_beta": trace.consensus_beta,
                "warmup_keep_ratio": args.keep_ratio,
                "lower_is_better": False,
                "vote_weight_method": args.vote_weight,
                "consensus_stop_enabled": False,
            }
            final_rows.append(final_row)
            trace_manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "trace_index": trace_idx,
                    "source_row_index": trace.source_row_index,
                    "phase": trace.phase,
                    "stop_index": trace.stop_index,
                    "output_len": trace.output_len,
                    "stopped_early": trace.stopped_early,
                    "threshold": trace.threshold,
                    "group_scores": trace.group_scores,
                    "trace_score": trace.trace_score,
                    "trace_weight": trace.trace_weight,
                    "answer": trace.answer,
                    "vote_applied": trace.vote_applied,
                    "consensus_answer": trace.consensus_answer,
                    "consensus_beta": trace.consensus_beta,
                    "stop_reason": trace.stop_reason,
                }
            )

        sample_summaries.append(
            {
                "sample_id": sample_id,
                "threshold": outcome.threshold,
                "warmup_trace_count": outcome.warmup_trace_count,
                "used_trace_count": outcome.used_trace_count,
                "available_trace_count": outcome.available_trace_count,
                "total_budget": outcome.total_budget,
                "consensus_reached": outcome.consensus_reached,
                "consensus_beta": outcome.consensus_beta,
                "consensus_answer": outcome.consensus_answer,
                "total_used_output_tokens": outcome.total_used_output_tokens,
                "skipped_source_row_indices": outcome.skipped_source_row_indices,
            }
        )

    per_sample_output_counts = defaultdict(int)
    for row in final_rows:
        per_sample_output_counts[str(row.get("example_id"))] += 1
    min_outputs_per_sample = (
        min(per_sample_output_counts.values()) if per_sample_output_counts else 0
    )

    summary = {
        "input_dir": str(input_dir.resolve()),
        "input_file": str(input_path.resolve()),
        "num_input_records": len(records),
        "num_input_samples": len(grouped_records),
        "num_processed_samples": len(grouped_items),
        "num_output_records": len(final_rows),
        "min_outputs_per_sample": min_outputs_per_sample,
        "max_outputs_per_sample": (
            max(per_sample_output_counts.values()) if per_sample_output_counts else 0
        ),
        "config": {
            "input_group_key": args.input_group_key,
            "total_budget": args.total_budget,
            "warmup_traces": args.warmup_traces,
            "keep_ratio": args.keep_ratio,
            "consensus_threshold": args.consensus_threshold,
            "group_size": args.group_size,
            "lower_is_better": False,
            "vote_weight": args.vote_weight,
            "include_warmup_in_vote": not args.exclude_warmup_from_vote,
            "consensus_stop_enabled": False,
            "tokenizer_path": tokenizer_path or None,
        },
        "samples": sample_summaries,
    }

    if args.dry_run:
        print(f"Processed samples: {len(grouped_items)}")
        print(f"Output records: {len(final_rows)}")
        print(f"Min outputs per sample: {min_outputs_per_sample}")
        return

    out_dir = prepare_output_dir(args, input_path)
    write_jsonl(out_dir / args.result_file_name, final_rows)
    write_jsonl(out_dir / "deepconf_online_traces.jsonl", trace_manifest_rows)
    dump_json(out_dir / "deepconf_online_summary.json", summary)

    merged_args = copy.deepcopy(original_args)
    merged_args["probe_prune"] = {
        "method": "deepconf_online",
        "input_dir": str(input_dir.resolve()),
        "input_file": str(input_path.resolve()),
        "output": str(out_dir.resolve()),
        "result_file": args.result_file_name,
        "tokenizer_path": tokenizer_path or None,
        "keep_prune_meta": True,
        "prune_config": {
            "strategy": "deepconf_online",
            "strategy_args": {
                "input_group_key": args.input_group_key,
                "total_budget": args.total_budget,
                "warmup_traces": args.warmup_traces,
                "keep_ratio": args.keep_ratio,
                "consensus_threshold": args.consensus_threshold,
                "group_size": args.group_size,
                "lower_is_better": False,
                "vote_weight": args.vote_weight,
                "include_warmup_in_vote": not args.exclude_warmup_from_vote,
                "consensus_stop_enabled": False,
            },
        },
    }
    merged_args["budget"] = {
        "budget_type": "traces",
        "budget_limit": args.total_budget,
        "budget_scope": "per_sample",
    }
    dump_json(out_dir / "args.json", merged_args)
    migrate_legacy_run(out_dir, force=True)

    print(f"Output directory: {out_dir}")
    print(f"Output records: {len(final_rows)}")
    print(f"Min outputs per sample: {min_outputs_per_sample}")


if __name__ == "__main__":
    main()
