from __future__ import annotations

from collections import defaultdict
from typing import Any

from .common import build_prune_config as build_prune_config_common
from .common import (
    resolve_input_group_value,
)
from .online import simulate_online_group


def build_prune_config(
    *,
    input_group_key: str,
    input_strategy_name: str,
    input_strategy_args: dict[str, Any],
    generation_strategy_name: str,
    generation_strategy_args: dict[str, Any],
    postprocess_strategy_name: str | None = None,
    postprocess_strategy_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_prune_config_common(
        input_group_key=input_group_key,
        input_strategy_name=input_strategy_name,
        input_strategy_args=input_strategy_args,
        generation_strategy_name=generation_strategy_name,
        generation_strategy_args=generation_strategy_args,
        simulation_mode="strict_online",
        postprocess_strategy_name=postprocess_strategy_name,
        postprocess_strategy_args=postprocess_strategy_args,
    )


def build_group_outcomes(
    *,
    records: list[dict[str, Any]],
    input_group_key: str,
    input_strategy_name: str,
    input_strategy_args: dict[str, Any],
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

    indexed_outcomes: dict[int, tuple[Any, str]] = {}
    for group_value, indexed_rows in grouped_records.items():
        group_rows = [row for _, row in indexed_rows]
        outcomes = simulate_online_group(
            records=group_rows,
            generation_strategy=generation_strategy,
            input_strategy_name=input_strategy_name,
            input_strategy_args=input_strategy_args,
        )
        if len(outcomes) != len(indexed_rows):
            raise ValueError(
                f"Strict online simulator returned {len(outcomes)} outcomes for group '{group_value}', "
                f"expected {len(indexed_rows)}."
            )
        for (record_idx, _), outcome in zip(indexed_rows, outcomes, strict=True):
            indexed_outcomes[record_idx] = (outcome, group_value)

    return indexed_outcomes, len(grouped_records)
