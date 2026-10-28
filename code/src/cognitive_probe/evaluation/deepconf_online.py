from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from cognitive_probe.evaluation.deepconf_logprob import INVALID_SCORE
from cognitive_probe.utils.math_utils import extract_answer, mathd_normalize_answer


def _maybe_valid_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < 0.0:
        return None
    return score


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    if percentile <= 0.0:
        return min(values)
    if percentile >= 100.0:
        return max(values)

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (percentile / 100.0) * (len(ordered) - 1)
    lower_idx = int(math.floor(position))
    upper_idx = int(math.ceil(position))
    if lower_idx == upper_idx:
        return ordered[lower_idx]
    mix = position - lower_idx
    return ordered[lower_idx] * (1.0 - mix) + ordered[upper_idx] * mix


def extract_vote_answer_from_response(response: str) -> str | None:
    answer = extract_answer(response)
    if answer is None:
        return None
    normalized = mathd_normalize_answer(answer)
    candidate = normalized if normalized is not None else answer.strip()
    candidate = candidate.strip()
    return candidate or None


def compute_trace_score(
    group_scores: list[float],
    *,
    lower_is_better: bool,
    invalid_value: float = INVALID_SCORE,
) -> float:
    valid_scores = [
        score
        for raw in group_scores
        if (score := _maybe_valid_score(raw)) is not None
    ]
    if not valid_scores:
        return invalid_value
    return max(valid_scores) if lower_is_better else min(valid_scores)


def compute_warmup_threshold(
    trace_scores: list[float],
    *,
    keep_ratio: float,
    lower_is_better: bool,
) -> float:
    valid_scores = [
        score
        for raw in trace_scores
        if (score := _maybe_valid_score(raw)) is not None
    ]
    if not valid_scores:
        return math.inf if lower_is_better else -math.inf

    percentile = keep_ratio * 100.0 if lower_is_better else (1.0 - keep_ratio) * 100.0
    return _percentile(valid_scores, percentile)


def trace_score_to_weight(
    trace_score: float,
    *,
    lower_is_better: bool,
    vote_weight: str,
) -> float:
    valid_score = _maybe_valid_score(trace_score)
    if valid_score is None:
        return 0.0

    if vote_weight == "exp_neg":
        if not lower_is_better:
            raise ValueError("vote_weight=exp_neg requires lower_is_better=True")
        return math.exp(-valid_score)
    if vote_weight == "inverse":
        if not lower_is_better:
            raise ValueError("vote_weight=inverse requires lower_is_better=True")
        return 1.0 / max(valid_score, 1e-12)
    if vote_weight == "raw":
        if lower_is_better:
            return 1.0 / max(valid_score, 1e-12)
        return max(valid_score, 0.0)
    raise ValueError(f"Unsupported vote_weight: {vote_weight}")


def should_early_stop_group(
    group_score: float,
    *,
    threshold: float,
    lower_is_better: bool,
) -> bool:
    valid_score = _maybe_valid_score(group_score)
    if valid_score is None:
        return False
    return valid_score > threshold if lower_is_better else valid_score < threshold


@dataclass(slots=True)
class DeepConfTraceOutcome:
    source_row_index: int
    phase: str
    stop_index: int
    output_len: int
    stopped_early: bool
    stop_reason: str
    threshold: float | None
    group_scores: list[float]
    trace_score: float
    trace_weight: float
    answer: str | None
    response: str
    vote_applied: bool
    consensus_beta: float | None
    consensus_answer: str | None


@dataclass(slots=True)
class DeepConfSampleOutcome:
    sample_id: str
    threshold: float
    warmup_trace_count: int
    used_trace_count: int
    available_trace_count: int
    total_budget: int
    consensus_reached: bool
    consensus_beta: float | None
    consensus_answer: str | None
    total_used_output_tokens: int
    traces: list[DeepConfTraceOutcome]
    skipped_source_row_indices: list[int]


def simulate_deepconf_trace(
    *,
    source_row_index: int,
    output_ids: list[int],
    token_scores: list[Any],
    phase: str,
    group_size: int,
    lower_is_better: bool,
    threshold: float | None,
    vote_weight: str,
    full_response: str,
    decode_output_ids: Callable[[list[int]], str] | None,
) -> DeepConfTraceOutcome:
    if group_size <= 0:
        raise ValueError("group_size must be > 0")

    if len(token_scores) != len(output_ids):
        raise ValueError(
            "token_scores length must match output_ids length: "
            f"{len(token_scores)} != {len(output_ids)}"
        )

    group_scores: list[float] = []
    buffered_valid_scores: list[float] = []
    stopped_early = False
    stop_reason = "completed full trace"
    stop_index = len(output_ids)

    for pos, raw_score in enumerate(token_scores):
        score = _maybe_valid_score(raw_score)
        if score is not None:
            buffered_valid_scores.append(score)

        if len(buffered_valid_scores) < group_size:
            continue

        group_score = sum(buffered_valid_scores) / len(buffered_valid_scores)
        group_scores.append(group_score)
        buffered_valid_scores = []

        if threshold is None:
            continue
        if not should_early_stop_group(
            group_score,
            threshold=threshold,
            lower_is_better=lower_is_better,
        ):
            continue

        stopped_early = True
        stop_index = pos + 1
        cmp_symbol = ">" if lower_is_better else "<"
        stop_reason = (
            f"group score {group_score:.6f} {cmp_symbol} threshold {threshold:.6f}"
        )
        break

    if not stopped_early and buffered_valid_scores:
        group_scores.append(sum(buffered_valid_scores) / len(buffered_valid_scores))

    trace_score = compute_trace_score(
        group_scores,
        lower_is_better=lower_is_better,
    )
    trace_weight = trace_score_to_weight(
        trace_score,
        lower_is_better=lower_is_better,
        vote_weight=vote_weight,
    )

    truncated_output_ids = output_ids[:stop_index]
    if stop_index >= len(output_ids):
        response = full_response
    elif decode_output_ids is not None:
        response = decode_output_ids(truncated_output_ids)
    else:
        response = ""

    answer = extract_vote_answer_from_response(response) if response else None
    vote_applied = answer is not None and trace_weight > 0.0

    return DeepConfTraceOutcome(
        source_row_index=source_row_index,
        phase=phase,
        stop_index=stop_index,
        output_len=len(output_ids),
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        threshold=threshold,
        group_scores=group_scores,
        trace_score=trace_score,
        trace_weight=trace_weight,
        answer=answer,
        response=response,
        vote_applied=vote_applied,
        consensus_beta=None,
        consensus_answer=None,
    )


def update_weighted_vote(
    vote_by_answer: dict[str, float],
    *,
    answer: str | None,
    weight: float,
) -> tuple[str | None, float | None]:
    if answer is not None and weight > 0.0:
        vote_by_answer[answer] = vote_by_answer.get(answer, 0.0) + weight
    if not vote_by_answer:
        return None, None
    best_answer = max(vote_by_answer, key=vote_by_answer.get)
    total_weight = sum(vote_by_answer.values())
    if total_weight <= 0.0:
        return best_answer, None
    return best_answer, vote_by_answer[best_answer] / total_weight


def run_deepconf_online_for_sample(
    *,
    sample_id: str,
    indexed_records: list[tuple[int, dict[str, Any]]],
    total_budget: int,
    warmup_traces: int,
    keep_ratio: float,
    consensus_threshold: float,
    group_size: int,
    lower_is_better: bool,
    vote_weight: str,
    include_warmup_in_vote: bool,
    decode_output_ids: Callable[[list[int]], str] | None,
) -> DeepConfSampleOutcome:
    bounded_records = indexed_records[: max(0, total_budget)]
    warmup_count = min(max(0, warmup_traces), len(bounded_records))

    vote_by_answer: dict[str, float] = {}
    traces: list[DeepConfTraceOutcome] = []

    warmup_scores: list[float] = []
    total_used_output_tokens = 0
    consensus_answer: str | None = None
    consensus_beta: float | None = None
    consensus_reached = False

    for source_row_index, record in bounded_records[:warmup_count]:
        output_ids = [int(token_id) for token_id in record.get("output_ids", [])]
        token_scores = list(record.get("probs") or [])
        outcome = simulate_deepconf_trace(
            source_row_index=source_row_index,
            output_ids=output_ids,
            token_scores=token_scores,
            phase="warmup",
            group_size=group_size,
            lower_is_better=lower_is_better,
            threshold=None,
            vote_weight=vote_weight,
            full_response=str(record.get("response") or ""),
            decode_output_ids=decode_output_ids,
        )
        traces.append(outcome)
        warmup_scores.append(outcome.trace_score)
        total_used_output_tokens += outcome.stop_index

    threshold = compute_warmup_threshold(
        warmup_scores,
        keep_ratio=keep_ratio,
        lower_is_better=lower_is_better,
    )

    if include_warmup_in_vote:
        for outcome in traces:
            consensus_answer, consensus_beta = update_weighted_vote(
                vote_by_answer,
                answer=outcome.answer,
                weight=outcome.trace_weight if outcome.vote_applied else 0.0,
            )
            outcome.consensus_answer = consensus_answer
            outcome.consensus_beta = consensus_beta
        if consensus_beta is not None and consensus_beta >= consensus_threshold:
            consensus_reached = True

    for source_row_index, record in bounded_records[warmup_count:]:
        output_ids = [int(token_id) for token_id in record.get("output_ids", [])]
        token_scores = list(record.get("probs") or [])
        outcome = simulate_deepconf_trace(
            source_row_index=source_row_index,
            output_ids=output_ids,
            token_scores=token_scores,
            phase="adaptive",
            group_size=group_size,
            lower_is_better=lower_is_better,
            threshold=threshold,
            vote_weight=vote_weight,
            full_response=str(record.get("response") or ""),
            decode_output_ids=decode_output_ids,
        )
        consensus_answer, consensus_beta = update_weighted_vote(
            vote_by_answer,
            answer=outcome.answer,
            weight=outcome.trace_weight if outcome.vote_applied else 0.0,
        )
        outcome.consensus_answer = consensus_answer
        outcome.consensus_beta = consensus_beta
        traces.append(outcome)
        total_used_output_tokens += outcome.stop_index
        if consensus_beta is not None and consensus_beta >= consensus_threshold:
            consensus_reached = True

    used_count = len(traces)
    skipped_source_row_indices = [
        source_row_index for source_row_index, _ in bounded_records[used_count:]
    ]
    return DeepConfSampleOutcome(
        sample_id=sample_id,
        threshold=threshold,
        warmup_trace_count=warmup_count,
        used_trace_count=used_count,
        available_trace_count=len(indexed_records),
        total_budget=total_budget,
        consensus_reached=consensus_reached,
        consensus_beta=consensus_beta,
        consensus_answer=consensus_answer,
        total_used_output_tokens=total_used_output_tokens,
        traces=traces,
        skipped_source_row_indices=skipped_source_row_indices,
    )
