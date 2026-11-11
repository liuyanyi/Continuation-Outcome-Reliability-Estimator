from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

INVALID_SCORE = -1.0


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_logprob_from_item(item: Any) -> float | None:
    if isinstance(item, dict):
        for key in ("logprob", "log_prob", "lp"):
            if key in item:
                return _maybe_float(item.get(key))
        return None

    if isinstance(item, (list, tuple)) and item:
        return _maybe_float(item[0])

    return None


def extract_topk_logprobs(step_top_logprobs: Any, *, topk: int) -> list[float]:
    if topk <= 0:
        raise ValueError("topk must be > 0")

    values: list[float] = []
    if isinstance(step_top_logprobs, dict):
        for value in step_top_logprobs.values():
            maybe = _maybe_float(value)
            if maybe is not None:
                values.append(maybe)
    elif isinstance(step_top_logprobs, list):
        for item in step_top_logprobs:
            maybe = _extract_logprob_from_item(item)
            if maybe is not None:
                values.append(maybe)

    values.sort(reverse=True)
    return values[:topk]


def compute_token_confidence(step_top_logprobs: Any, *, topk: int) -> float:
    """Compute the literal DeepConf token score from top-k logprobs.

    `C_i = -mean(top-k logprobs)` is higher-is-better confidence in DeepConf.
    It is not the negative log-likelihood of the sampled token.
    """

    values = extract_topk_logprobs(step_top_logprobs, topk=topk)
    if not values:
        return INVALID_SCORE
    return -sum(values) / len(values)


def compute_token_probability_proxy(step_top_logprobs: Any, *, topk: int) -> float:
    """Geometric mean of top-k probabilities (decreases as confidence rises).

    Retained for compatibility; do not use this as a DeepConf vote weight.
    """

    confidence = compute_token_confidence(step_top_logprobs, topk=topk)
    if confidence < 0.0:
        return INVALID_SCORE
    return math.exp(-confidence)


def aggregate_group_confidence(
    token_confidences: Sequence[Any],
    *,
    group_size: int,
    invalid_value: float = INVALID_SCORE,
) -> list[float]:
    if group_size <= 0:
        raise ValueError("group_size must be > 0")

    valid_scores: list[float] = []
    for value in token_confidences:
        maybe = _maybe_float(value)
        if maybe is None or maybe < 0.0:
            continue
        valid_scores.append(maybe)

    groups: list[float] = []
    for start in range(0, len(valid_scores), group_size):
        chunk = valid_scores[start : start + group_size]
        if not chunk:
            continue
        groups.append(sum(chunk) / len(chunk))
    if not groups:
        return [invalid_value] if valid_scores else []
    return groups


def compute_trace_confidence(
    group_confidences: Sequence[Any],
    *,
    invalid_value: float = INVALID_SCORE,
) -> float:
    valid_scores: list[float] = []
    for value in group_confidences:
        maybe = _maybe_float(value)
        if maybe is None or maybe < 0.0:
            continue
        valid_scores.append(maybe)
    if not valid_scores:
        return invalid_value
    return min(valid_scores)


def align_input_top_logprobs(
    input_top_logprobs: Sequence[Any],
    *,
    input_len: int,
    output_len: int,
) -> list[Any]:
    if output_len < 0:
        raise ValueError("output_len must be >= 0")
    if input_len < 0:
        raise ValueError("input_len must be >= 0")
    if output_len == 0:
        return []

    entries = list(input_top_logprobs)
    if len(entries) == output_len:
        return entries

    start = input_len
    end = input_len + output_len
    if len(entries) >= end:
        return entries[start:end]

    # Some backends omit the first prompt token when returning prompt logprobs.
    if input_len > 0:
        shifted_start = input_len - 1
        shifted_end = shifted_start + output_len
        if len(entries) >= shifted_end:
            return entries[shifted_start:shifted_end]

    if len(entries) > output_len:
        return entries[-output_len:]

    raise ValueError(
        "Cannot align input_top_logprobs with output_ids: "
        f"entries={len(entries)}, input_len={input_len}, output_len={output_len}"
    )
