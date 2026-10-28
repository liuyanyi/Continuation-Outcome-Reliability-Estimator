#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from pathlib import Path
from typing import Any

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
from cognitive_probe.evaluation.probe_prune.postprocess import (
    RegenerationTrace,
    regenerate_pruned_records,
)
from cognitive_probe.evaluation.v2 import (
    _reference_pass_map,
    build_run_overview,
)


def _write_regeneration_v2_run(
    *,
    out_dir: Path,
    args_json: dict[str, Any],
    original_records: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    indexed_outcomes: dict[int, tuple[Any, str]],
    regeneration_traces: dict[int, RegenerationTrace],
    regeneration_config: Any,
    tokenizer: Any,
) -> None:
    overview = build_run_overview(args_json, out_dir)
    reference_map = _reference_pass_map(out_dir, args_json)
    base_model_name = overview.model_name or overview.model
    regen_model_name = str(regeneration_config.model_name)

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
        trace = regeneration_traces.get(row_idx)
        if trace is not None:
            outcome, _ = indexed_outcomes[row_idx]
            decision = outcome.decision
            if decision.cut_index is None:
                raise ValueError(
                    f"Missing stop_index for regenerated online row {row_idx}"
                )
            original_output_ids = original_records[row_idx].get("output_ids")
            original_probs = original_records[row_idx].get("probs")
            if not isinstance(original_output_ids, list):
                raise ValueError(
                    f"Missing output_ids for regenerated online row {row_idx}"
                )
            if not isinstance(original_probs, list):
                original_probs = []

            stop_index = max(0, min(int(decision.cut_index), len(original_output_ids)))
            pruned_output_ids = [
                int(token_id) for token_id in original_output_ids[:stop_index]
            ]
            pruned_probs = [
                float(value)
                for value in original_probs[:stop_index]
                if isinstance(value, (int, float))
            ]
            pruned_segment_id = f"{final_segment_id}__pruned"
            pruned_response = (
                decode_response(tokenizer, pruned_output_ids)
                if tokenizer is not None
                else ""
            )
            pruned_seed = (
                int(original_records[row_idx]["seed"])
                if original_records[row_idx].get("seed") is not None
                else None
            )
            prune_meta = (
                final_row.get("prune_meta")
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
                        "stop_index": stop_index,
                        "simulation_mode": "strict_online",
                    },
                ).to_dict()
            )
            decisions.append(
                DecisionRecord(
                    decision_type="rollback",
                    sample_id=sample_id,
                    output_id=output_id,
                    segment_id=pruned_segment_id,
                    original_output_tokens=len(pruned_output_ids),
                    kept_output_tokens=0,
                    discarded_output_tokens=len(pruned_output_ids),
                    reason="postprocess regenerate_if_pruned discarded online prefix",
                    meta={
                        "regenerated_seed": trace.regenerated_seed,
                        "simulation_mode": "strict_online",
                    },
                ).to_dict()
            )
            parent_segment_id = pruned_segment_id
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
                model_name=regen_model_name if trace is not None else base_model_name,
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
            "Strict online replay for probe pruning with regenerate-if-pruned control. "
            "This keeps strict online stop semantics, then calls model API to regenerate replacements."
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
        default="evaluation/results/probe_prune_online_regenerate",
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
        "--regen-args-json",
        type=str,
        default="",
        help="Inline JSON object for regenerate_if_pruned args.",
    )
    parser.add_argument(
        "--regen-args-file",
        type=str,
        default="",
        help="JSON file path for regenerate_if_pruned args.",
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
        help="Tokenizer path used to decode truncated output_ids back to response text.",
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
        help="Resume regenerate_if_pruned API calls from an incremental checkpoint in the output directory.",
    )
    parser.add_argument(
        "--checkpoint-file-name",
        type=str,
        default=".regenerate_if_pruned.checkpoint.jsonl",
        help="Incremental regeneration checkpoint JSONL filename inside the output directory.",
    )
    parser.add_argument(
        "--checkpoint-fsync-interval",
        type=int,
        default=1,
        help="fsync checkpoint every N completed regenerations (0 disables fsync).",
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
    regeneration_strategy_args = merge_strategy_args(
        defaults={},
        args_file=args.regen_args_file,
        args_json=args.regen_args_json,
        args_file_name="--regen-args-file",
        args_json_name="--regen-args-json",
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

    tokenizer = None
    effective_tokenizer_path = str(args.tokenizer_path or "").strip()
    if effective_tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            effective_tokenizer_path, trust_remote_code=True
        )

    indexed_outcomes, input_group_count = build_group_outcomes(
        records=records,
        input_group_key=args.input_group_key,
        input_strategy_name=args.input_strategy,
        input_strategy_args=input_strategy_args,
        generation_strategy=generation_strategy,
    )
    pruned_row_indices: list[int] = []
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
        pruned_row_indices.append(record_idx)

    out_dir: Path | None = None
    output_path: Path | None = None
    checkpoint_path: Path | None = None
    if not args.dry_run:
        out_dir = prepare_output_dir(args, input_path)
        output_path = out_dir / args.result_file_name
        checkpoint_path = out_dir / args.checkpoint_file_name

    regeneration_config = None
    regeneration_traces: dict[int, RegenerationTrace] = {}
    regenerated = 0
    if pruned_row_indices:
        regeneration_config, regeneration_traces = regenerate_pruned_records(
            records=records,
            pruned_row_indices=pruned_row_indices,
            original_args=original_args,
            strategy_args=regeneration_strategy_args,
            checkpoint_path=checkpoint_path if args.resume else None,
            checkpoint_fsync_interval=args.checkpoint_fsync_interval,
        )
        regenerated = len(regeneration_traces)

    total = 0
    pruned = 0
    unchanged = 0
    empty_after_prune = 0
    decode_skipped = 0

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
                "strategy": args.input_strategy
                if args.input_strategy != "delegate_second_level"
                else generation_strategy.name,
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
                "strategy": args.input_strategy
                if args.input_strategy != "delegate_second_level"
                else generation_strategy.name,
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
                "strategy": args.input_strategy
                if args.input_strategy != "delegate_second_level"
                else generation_strategy.name,
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

        regen_trace = regeneration_traces.get(record_idx)
        if regen_trace is not None:
            regenerated_row = copy.deepcopy(regen_trace.regenerated_row)
            regenerated_row["prune_meta"] = {
                "strategy": args.input_strategy
                if args.input_strategy != "delegate_second_level"
                else generation_strategy.name,
                "generation_strategy": generation_strategy.name,
                "input_strategy": args.input_strategy,
                "input_group_key_field": args.input_group_key,
                "input_group_value": input_group_value,
                "decision": "prune_then_regenerate",
                "reason": decision.reason,
                "trigger_pos": decision.trigger_pos,
                "trigger_prob": decision.trigger_prob,
                "cut_index": cut_index,
                "original_len": len(output_ids),
                "pruned_len": cut_index,
                "logical_cut_index": outcome.logical_cut_index,
                "detect_pos": outcome.detect_pos,
                "stop_index": cut_index,
                "simulation_mode": "strict_online",
                "candidate_count": outcome.candidate_count,
                "release_count": outcome.release_count,
                "final_state": outcome.final_state,
                "postprocess_strategy": "regenerate_if_pruned",
                "postprocess_strategy_args": dict(regeneration_strategy_args),
                "original_seed": row.get("seed"),
                "regenerated_seed": regen_trace.regenerated_seed,
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

        if tokenizer is not None:
            row["response"] = decode_response(tokenizer, pruned_output_ids)
        else:
            decode_skipped += 1

        row["prune_meta"] = {
            "strategy": args.input_strategy
            if args.input_strategy != "delegate_second_level"
            else generation_strategy.name,
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
            "tokenizer_path": args.tokenizer_path or None,
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
                postprocess_strategy_name="regenerate_if_pruned",
                postprocess_strategy_args=regeneration_strategy_args,
            ),
        }
        if regeneration_config is not None:
            args_json["probe_prune"]["postprocess_runtime"] = {
                "model_name": regeneration_config.model_name,
                "temperature": regeneration_config.temperature,
                "top_p": regeneration_config.top_p,
                "top_k": regeneration_config.top_k,
                "api_url": regeneration_config.api_url,
                "seed_offset": regeneration_config.seed_offset,
                "timeout_s": regeneration_config.timeout_s,
                "concurrency_limit": regeneration_config.concurrency_limit,
            }
            if regeneration_config.max_new_tokens is not None:
                args_json["probe_prune"]["postprocess_runtime"]["max_new_tokens"] = (
                    regeneration_config.max_new_tokens
                )

        dump_json(out_dir / "args.json", args_json)
        assert output_path is not None
        write_jsonl(output_path, out_rows)
        _write_regeneration_v2_run(
            out_dir=out_dir,
            args_json=args_json,
            original_records=records,
            final_rows=out_rows,
            indexed_outcomes=indexed_outcomes,
            regeneration_traces=regeneration_traces,
            regeneration_config=regeneration_config,
            tokenizer=tokenizer,
        )

    root_strategy_name = (
        generation_strategy.name
        if args.input_strategy == "delegate_second_level"
        else args.input_strategy
    )
    print(f"Input records: {total}")
    print(f"Pruned records: {pruned}")
    print(f"Regenerated records: {regenerated}")
    print(f"Unchanged records: {unchanged}")
    print(f"Empty after prune: {empty_after_prune}")
    print(f"Input groups ({args.input_group_key}): {input_group_count}")
    if decode_skipped > 0:
        print(f"Decode skipped records (no --tokenizer-path): {decode_skipped}")

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
    print("Control strategy: regenerate_if_pruned")
    print(
        f"Control strategy args: {json_text(regeneration_strategy_args, sort_keys=True)}"
    )
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
