from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from cognitive_probe.evaluation.data_model import SourceTarget
from cognitive_probe.evaluation.io import json_text, load_json


def decode_response(tokenizer: Any, output_ids: list[int]) -> str:
    return tokenizer.decode(
        output_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def resolve_input_file(input_dir: Path) -> Path:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input-dir must be a directory: {input_dir}")

    candidates = sorted(input_dir.glob("*.probed.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No '*.probed.jsonl' file found under directory: {input_dir}"
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(
            "Multiple '*.probed.jsonl' files found; please keep only one in input directory. "
            f"Found: {names}"
        )
    return candidates[0]


def load_original_args(input_dir: Path) -> dict[str, Any]:
    src_args_path = input_dir / "args.json"
    if not src_args_path.exists():
        return {}
    return load_json(src_args_path)


def prepare_output_dir(args: argparse.Namespace, input_path: Path) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        dataset_name = input_path.stem
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        out_dir = Path(args.output) / args.experiment_name / dataset_name / timestamp

    if out_dir.exists():
        if not out_dir.is_dir():
            raise NotADirectoryError(
                f"Output path exists and is not a directory: {out_dir}"
            )
        output_path = out_dir / args.result_file_name
        if output_path.exists():
            raise FileExistsError(f"Output file already exists: {output_path}")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_json_object(raw: str, *, arg_name: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}

    try:
        payload = orjson.loads(raw)
    except orjson.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for {arg_name}: {e}") from e

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object for {arg_name}, got {type(payload).__name__}"
        )
    return payload


def load_json_object_file(path_text: str, *, arg_name: str) -> dict[str, Any]:
    path_text = path_text.strip()
    if not path_text:
        return {}

    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"{arg_name} not found: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object in {arg_name}={path}, got {type(payload).__name__}"
        )
    return payload


def merge_strategy_args(
    *,
    defaults: dict[str, Any],
    args_file: str,
    args_json: str,
    args_file_name: str,
    args_json_name: str,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    merged.update(defaults)
    merged.update(load_json_object_file(args_file, arg_name=args_file_name))
    merged.update(load_json_object(args_json, arg_name=args_json_name))
    return merged


def legacy_defaults_for_strategy(args: argparse.Namespace) -> dict[str, Any]:
    if args.strategy == "first_below":
        return {
            "threshold": args.threshold,
            "min_keep_tokens": args.min_keep_tokens,
        }
    if args.strategy == "consecutive_below":
        return {
            "threshold": args.threshold,
            "window_size": args.window_size,
            "min_keep_tokens": args.min_keep_tokens,
        }
    if args.strategy == "armed_consecutive_below":
        return {
            "threshold": args.threshold,
            "window_size": args.window_size,
            "arm_threshold": args.arm_threshold,
            "min_keep_tokens": args.min_keep_tokens,
        }
    if args.strategy == "armed_valid_window_n_out_of_m":
        return {
            "threshold": args.threshold,
            "n": args.n,
            "m": args.m,
            "arm_threshold": args.arm_threshold,
            "min_keep_tokens": args.min_keep_tokens,
        }
    if args.strategy == "valid_window_average":
        return {
            "threshold": args.threshold,
            "window_size": args.window_size,
            "min_keep_tokens": args.min_keep_tokens,
        }
    if args.strategy == "ema_below":
        return {
            "threshold": args.threshold,
            "alpha": args.alpha,
            "warmup_valid_points": args.warmup_valid_points,
            "min_keep_tokens": args.min_keep_tokens,
        }
    if args.strategy == "armed_ema_below":
        return {
            "threshold": args.threshold,
            "alpha": args.alpha,
            "warmup_valid_points": args.warmup_valid_points,
            "arm_threshold": args.arm_threshold,
            "min_keep_tokens": args.min_keep_tokens,
        }
    if args.strategy == "valid_window_n_out_of_m":
        return {
            "threshold": args.threshold,
            "n": args.n,
            "m": args.m,
            "min_keep_tokens": args.min_keep_tokens,
        }
    if args.strategy == "drop_from_peak":
        return {
            "drop_threshold": args.drop_threshold,
            "min_keep_tokens": args.min_keep_tokens,
        }
    return {}


def split_input_strategy_args(
    *,
    input_strategy_name: str,
    input_strategy_args: dict[str, Any],
) -> tuple[dict[str, Any], str | None, dict[str, Any]]:
    resolved_input_args = dict(input_strategy_args)
    nested_generation_name_raw = resolved_input_args.pop("generation_strategy", None)
    nested_generation_args_raw = resolved_input_args.pop(
        "generation_strategy_args", None
    )

    if nested_generation_args_raw is not None and nested_generation_name_raw is None:
        raise ValueError(
            "input-strategy args include generation_strategy_args but missing generation_strategy"
        )

    if input_strategy_name == "delegate_second_level":
        if (
            nested_generation_name_raw is not None
            or nested_generation_args_raw is not None
        ):
            raise ValueError(
                "delegate_second_level should not embed generation_strategy in "
                "--input-strategy-args-json; use --strategy / --strategy-args-json instead"
            )
        return resolved_input_args, None, {}

    nested_generation_name: str | None = None
    if nested_generation_name_raw is not None:
        nested_generation_name = str(nested_generation_name_raw).strip()
        if not nested_generation_name:
            raise ValueError(
                "generation_strategy in input-strategy args must be non-empty"
            )

    if nested_generation_args_raw is None:
        nested_generation_args: dict[str, Any] = {}
    elif isinstance(nested_generation_args_raw, dict):
        nested_generation_args = dict(nested_generation_args_raw)
    else:
        raise ValueError(
            "generation_strategy_args in input-strategy args must be a JSON object"
        )

    return resolved_input_args, nested_generation_name, nested_generation_args


def build_prune_config(
    *,
    input_group_key: str,
    input_strategy_name: str,
    input_strategy_args: dict[str, Any],
    generation_strategy_name: str,
    generation_strategy_args: dict[str, Any],
    simulation_mode: str | None = None,
    postprocess_strategy_name: str | None = None,
    postprocess_strategy_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "input_group_key": input_group_key,
    }
    if simulation_mode:
        payload["simulation_mode"] = simulation_mode

    if input_strategy_name == "delegate_second_level":
        payload["strategy"] = generation_strategy_name
        payload["strategy_args"] = generation_strategy_args
    else:
        root_strategy_args = dict(input_strategy_args)
        root_strategy_args["generation_strategy"] = generation_strategy_name
        root_strategy_args["generation_strategy_args"] = generation_strategy_args
        payload["strategy"] = input_strategy_name
        payload["strategy_args"] = root_strategy_args

    if postprocess_strategy_name:
        payload["postprocess_strategy"] = postprocess_strategy_name
        payload["postprocess_strategy_args"] = dict(postprocess_strategy_args or {})
    return payload


def resolve_input_group_value(
    *,
    record: dict[str, Any],
    group_key: str,
    record_idx: int,
) -> str:
    if group_key not in record:
        raise ValueError(
            f"Missing group key '{group_key}' at record index {record_idx}. "
            "Set --input-group-key to a valid record field."
        )
    value = record.get(group_key)
    if value is None:
        raise ValueError(
            f"Group key '{group_key}' is None at record index {record_idx}. "
            "Set --input-group-key to a valid non-null field."
        )
    if isinstance(value, (dict, list)):
        return json_text(value, sort_keys=True)
    return str(value)


def build_group_decisions(
    *,
    records: list[dict[str, Any]],
    input_group_key: str,
    input_strategy: Any,
    generation_strategy: Any,
) -> tuple[dict[int, tuple[Any, str]], int]:
    grouped_records: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, record in enumerate(records):
        group_value = resolve_input_group_value(
            record=record,
            group_key=input_group_key,
            record_idx=idx,
        )
        grouped_records[group_value].append((idx, record))

    indexed_decisions: dict[int, tuple[Any, str]] = {}
    for group_value, indexed_rows in grouped_records.items():
        group_rows = [row for _, row in indexed_rows]
        decisions = input_strategy.decide_group(
            records=group_rows,
            generation_strategy=generation_strategy,
        )
        if len(decisions) != len(indexed_rows):
            raise ValueError(
                f"Input strategy '{input_strategy.name}' returned {len(decisions)} "
                f"decisions for group '{group_value}', expected {len(indexed_rows)}."
            )
        for (record_idx, _), decision in zip(indexed_rows, decisions, strict=True):
            indexed_decisions[record_idx] = (decision, group_value)

    return indexed_decisions, len(grouped_records)


def build_source_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_targets: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        source_targets.append(
            SourceTarget(
                sample_id=str(row.get("example_id")),
                source_row_index=row_idx,
                answer=str(row.get("answer") or ""),
            ).to_dict()
        )
    return source_targets
