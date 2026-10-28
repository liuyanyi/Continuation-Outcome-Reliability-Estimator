#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from cognitive_probe.evaluation.io import dump_json, json_text, load_jsonl, write_jsonl
from cognitive_probe.evaluation.probe_prune import build_input_strategy, build_strategy
from cognitive_probe.evaluation.probe_prune.common import (
    build_group_decisions,
    build_prune_config,
    decode_response,
    legacy_defaults_for_strategy,
    load_original_args,
    merge_strategy_args,
    prepare_output_dir,
    resolve_input_file,
    split_input_strategy_args,
)
from cognitive_probe.evaluation.v2 import migrate_legacy_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prune generation results based on probe probabilities. "
            "The script truncates from the prune position and writes finish_reason='prune'."
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
        default="evaluation/results/probe_prune",
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
        help=(
            "Second-level (per generation record) strategy name. "
            "When --input-strategy is not delegate_second_level, this acts as a fallback "
            "unless generation_strategy is embedded inside --input-strategy-args-json."
        ),
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
        help=(
            "Field name used to group records as one input sample for first-level pruning. "
            "Default groups all rollouts by example_id."
        ),
    )
    parser.add_argument(
        "--input-strategy",
        type=str,
        default="delegate_second_level",
        help=(
            "First-level (input-group level) strategy name. "
            "Default delegates each grouped record to second-level strategy."
        ),
    )
    parser.add_argument(
        "--input-strategy-args-json",
        type=str,
        default="",
        help=(
            "Inline JSON object for first-level strategy args. "
            "For level1 strategies, preferred schema is to embed second-level config here via "
            "generation_strategy + generation_strategy_args."
        ),
    )
    parser.add_argument(
        "--input-strategy-args-file",
        type=str,
        default="",
        help="JSON file path for first-level strategy args.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Legacy default arg for strategy_args.threshold.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2,
        help="Legacy default arg for strategy_args.window_size.",
    )
    parser.add_argument(
        "--min-keep-tokens",
        type=int,
        default=0,
        help="Legacy default arg for strategy_args.min_keep_tokens.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Legacy default arg for strategy_args.n.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=3,
        help="Legacy default arg for strategy_args.m.",
    )
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=0.2,
        help="Legacy default arg for strategy_args.drop_threshold.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="",
        help=(
            "Tokenizer path used to decode pruned output_ids back to response text. "
            "If empty, decoding is skipped and response text is kept unchanged."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report statistics without writing output file.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Process only first N records for debug (0 means all).",
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

    input_strategy = build_input_strategy(
        strategy_name=args.input_strategy,
        strategy_args=input_strategy_args,
    )
    strategy = build_strategy(
        strategy_name=resolved_generation_strategy_name,
        strategy_args=resolved_generation_strategy_args,
    )
    root_strategy_name = input_strategy.name
    if root_strategy_name == "delegate_second_level":
        root_strategy_name = strategy.name

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

    indexed_decisions, input_group_count = build_group_decisions(
        records=records,
        input_group_key=args.input_group_key,
        input_strategy=input_strategy,
        generation_strategy=strategy,
    )

    total = 0
    pruned = 0
    unchanged = 0
    empty_after_prune = 0
    decode_skipped = 0

    out_rows: list[dict[str, Any]] = []
    for record_idx, record in enumerate(records):
        total += 1
        row = copy.deepcopy(record)
        decision, input_group_value = indexed_decisions[record_idx]

        output_ids = row.get("output_ids")
        probs = row.get("probs")
        if not isinstance(output_ids, list):
            raise ValueError("Each record must contain list[int] field: output_ids")

        if not isinstance(probs, list) or len(probs) != len(output_ids):
            # Keep untouched if this row has not been probed or has malformed probs.
            unchanged += 1
            row["prune_meta"] = {
                "strategy": root_strategy_name,
                "generation_strategy": strategy.name,
                "input_strategy": input_strategy.name,
                "input_group_key_field": args.input_group_key,
                "input_group_value": input_group_value,
                "decision": "keep",
                "reason": "invalid row: probs missing or length mismatch",
            }
            out_rows.append(row)
            continue

        if not decision.should_prune or decision.cut_index is None:
            unchanged += 1
            row["prune_meta"] = {
                "strategy": root_strategy_name,
                "generation_strategy": strategy.name,
                "input_strategy": input_strategy.name,
                "input_group_key_field": args.input_group_key,
                "input_group_value": input_group_value,
                "decision": "keep",
                "reason": decision.reason,
            }
            out_rows.append(row)
            continue

        cut_index = max(0, min(int(decision.cut_index), len(output_ids)))
        if cut_index >= len(output_ids):
            unchanged += 1
            row["prune_meta"] = {
                "strategy": root_strategy_name,
                "generation_strategy": strategy.name,
                "input_strategy": input_strategy.name,
                "input_group_key_field": args.input_group_key,
                "input_group_value": input_group_value,
                "decision": "keep",
                "reason": "cut_index >= output_len after clamp",
            }
            out_rows.append(row)
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
            "strategy": root_strategy_name,
            "generation_strategy": strategy.name,
            "input_strategy": input_strategy.name,
            "input_group_key_field": args.input_group_key,
            "input_group_value": input_group_value,
            "decision": "prune",
            "reason": decision.reason,
            "trigger_pos": decision.trigger_pos,
            "trigger_prob": decision.trigger_prob,
            "cut_index": cut_index,
            "original_len": len(output_ids),
            "pruned_len": len(pruned_output_ids),
        }

        if len(pruned_output_ids) == 0:
            empty_after_prune += 1
        pruned += 1
        out_rows.append(row)

    out_dir: Path | None = None
    output_path: Path | None = None

    if not args.dry_run:
        out_dir = prepare_output_dir(args, input_path)
        output_path = out_dir / args.result_file_name

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
            "prune_config": build_prune_config(
                input_group_key=args.input_group_key,
                input_strategy_name=input_strategy.name,
                input_strategy_args=input_strategy_args,
                generation_strategy_name=strategy.name,
                generation_strategy_args=resolved_generation_strategy_args,
            ),
        }

        dump_json(out_dir / "args.json", args_json)

        assert output_path is not None
        write_jsonl(output_path, out_rows)
        migrate_legacy_run(out_dir, force=True)

    print(f"Input records: {total}")
    print(f"Pruned records: {pruned}")
    print(f"Unchanged records: {unchanged}")
    print(f"Empty after prune: {empty_after_prune}")
    print(f"Input groups ({args.input_group_key}): {input_group_count}")
    if decode_skipped > 0:
        print(f"Decode skipped records (no --tokenizer-path): {decode_skipped}")

    print("-" * 30)
    print(f"Root prune strategy: {root_strategy_name}")
    if input_strategy_args:
        print(
            "First-level strategy args: "
            f"{json_text(input_strategy_args, sort_keys=True)}"
        )
    print(f"Second-level strategy: {strategy.name}")
    print(
        "Second-level strategy args: "
        f"{json_text(resolved_generation_strategy_args, sort_keys=True)}"
    )
    print("-" * 30)

    if args.dry_run:
        print("Dry run only, output file not written.")
    else:
        assert out_dir is not None
        assert output_path is not None
        print(f"Output directory: {out_dir}")
        print(f"Saved pruned results to: {output_path}")


if __name__ == "__main__":
    main()
