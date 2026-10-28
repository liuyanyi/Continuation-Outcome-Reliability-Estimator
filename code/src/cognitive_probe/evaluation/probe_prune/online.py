from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .strategies import BasePruneStrategy, PruneDecision


def _keep_decision(*, reason: str) -> PruneDecision:
    return PruneDecision(
        should_prune=False,
        cut_index=None,
        trigger_pos=None,
        trigger_prob=None,
        reason=reason,
    )


def _normalize_prob(raw_prob: Any) -> float | None:
    if raw_prob is None:
        return None
    try:
        prob = float(raw_prob)
    except (TypeError, ValueError):
        return None
    if prob < 0.0:
        return None
    return prob


def _prune_count_requirement(*, total: int, min_count: int, min_fraction: float) -> int:
    return max(min_count, int(math.ceil(total * min_fraction - 1e-12)))


def _quantile_floor_value(values: list[int], quantile: float) -> int:
    if not values:
        raise ValueError("values must be non-empty")
    sorted_values = sorted(values)
    idx = int(math.ceil(quantile * (len(sorted_values) - 1)))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


@dataclass
class OnlinePruneCandidate:
    logical_cut_index: int
    detect_pos: int
    observed_stop_index: int
    trigger_prob: float | None
    reason: str


@dataclass
class OnlineReplayOutcome:
    decision: PruneDecision
    logical_cut_index: int | None
    detect_pos: int | None
    candidate_count: int
    release_count: int
    final_state: str


class BaseOnlineGenerationState(ABC):
    name: str

    @abstractmethod
    def step(self, *, pos: int, raw_prob: Any) -> OnlinePruneCandidate | None:
        raise NotImplementedError


@dataclass
class FirstBelowOnlineState(BaseOnlineGenerationState):
    threshold: float
    min_keep_tokens: int
    name: str = "first_below"

    def step(self, *, pos: int, raw_prob: Any) -> OnlinePruneCandidate | None:
        prob = _normalize_prob(raw_prob)
        if prob is None or pos < self.min_keep_tokens or prob > self.threshold:
            return None
        return OnlinePruneCandidate(
            logical_cut_index=pos,
            detect_pos=pos,
            observed_stop_index=pos + 1,
            trigger_prob=prob,
            reason=f"first p<=threshold ({prob:.6f}<={self.threshold:.6f})",
        )


@dataclass
class ConsecutiveBelowOnlineState(BaseOnlineGenerationState):
    threshold: float
    window_size: int
    min_keep_tokens: int
    valid_window: deque[tuple[int, float]] = field(default_factory=deque)
    name: str = "consecutive_below"

    def step(self, *, pos: int, raw_prob: Any) -> OnlinePruneCandidate | None:
        prob = _normalize_prob(raw_prob)
        if prob is None:
            return None

        self.valid_window.append((pos, prob))
        if len(self.valid_window) > self.window_size:
            self.valid_window.popleft()
        if len(self.valid_window) < self.window_size:
            return None

        if any(
            window_pos < self.min_keep_tokens for window_pos, _ in self.valid_window
        ):
            return None
        if not all(
            window_prob <= self.threshold for _, window_prob in self.valid_window
        ):
            return None

        trigger_pos, trigger_prob = self.valid_window[0]
        return OnlinePruneCandidate(
            logical_cut_index=trigger_pos,
            detect_pos=pos,
            observed_stop_index=pos + 1,
            trigger_prob=trigger_prob,
            reason=(
                f"{self.window_size} consecutive valid points p<=threshold "
                f"(start_prob={trigger_prob:.6f}, threshold={self.threshold:.6f})"
            ),
        )


@dataclass
class ArmedConsecutiveBelowOnlineState(BaseOnlineGenerationState):
    threshold: float
    window_size: int
    arm_threshold: float
    min_keep_tokens: int
    armed: bool = False
    arm_pos: int | None = None
    valid_window: deque[tuple[int, float]] = field(default_factory=deque)
    name: str = "armed_consecutive_below"

    def step(self, *, pos: int, raw_prob: Any) -> OnlinePruneCandidate | None:
        prob = _normalize_prob(raw_prob)
        if prob is None:
            return None

        if not self.armed:
            if prob >= self.arm_threshold:
                self.armed = True
                self.arm_pos = pos
                self.valid_window.clear()
            return None

        self.valid_window.append((pos, prob))
        if len(self.valid_window) > self.window_size:
            self.valid_window.popleft()
        if len(self.valid_window) < self.window_size:
            return None

        if any(
            window_pos < self.min_keep_tokens for window_pos, _ in self.valid_window
        ):
            return None
        if not all(
            window_prob <= self.threshold for _, window_prob in self.valid_window
        ):
            return None

        trigger_pos, trigger_prob = self.valid_window[0]
        return OnlinePruneCandidate(
            logical_cut_index=trigger_pos,
            detect_pos=pos,
            observed_stop_index=pos + 1,
            trigger_prob=trigger_prob,
            reason=(
                f"armed at pos={self.arm_pos}, then {self.window_size} consecutive valid points "
                f"p<=threshold (start_prob={trigger_prob:.6f}, threshold={self.threshold:.6f})"
            ),
        )


@dataclass
class ArmedValidWindowNOutOfMOnlineState(BaseOnlineGenerationState):
    threshold: float
    n: int
    m: int
    arm_threshold: float
    min_keep_tokens: int
    armed: bool = False
    arm_pos: int | None = None
    valid_window: deque[tuple[int, float]] = field(default_factory=deque)
    name: str = "armed_valid_window_n_out_of_m"

    def step(self, *, pos: int, raw_prob: Any) -> OnlinePruneCandidate | None:
        prob = _normalize_prob(raw_prob)
        if prob is None:
            return None

        if not self.armed:
            if prob >= self.arm_threshold:
                self.armed = True
                self.arm_pos = pos
                self.valid_window.clear()
            return None

        self.valid_window.append((pos, prob))
        if len(self.valid_window) > self.m:
            self.valid_window.popleft()
        if len(self.valid_window) < self.m:
            return None

        if any(
            window_pos < self.min_keep_tokens for window_pos, _ in self.valid_window
        ):
            return None

        below = [
            (window_pos, window_prob)
            for window_pos, window_prob in self.valid_window
            if window_prob <= self.threshold
        ]
        if len(below) < self.n:
            return None

        trigger_pos, trigger_prob = below[0]
        return OnlinePruneCandidate(
            logical_cut_index=trigger_pos,
            detect_pos=pos,
            observed_stop_index=pos + 1,
            trigger_prob=trigger_prob,
            reason=(
                f"armed at pos={self.arm_pos}, then {len(below)} out of {self.m} "
                f"valid probes p<=threshold"
            ),
        )


@dataclass
class ValidWindowAverageOnlineState(BaseOnlineGenerationState):
    threshold: float
    window_size: int
    min_keep_tokens: int
    valid_window: deque[tuple[int, float]] = field(default_factory=deque)
    name: str = "valid_window_average"

    def step(self, *, pos: int, raw_prob: Any) -> OnlinePruneCandidate | None:
        prob = _normalize_prob(raw_prob)
        if prob is None:
            return None

        self.valid_window.append((pos, prob))
        if len(self.valid_window) > self.window_size:
            self.valid_window.popleft()
        if len(self.valid_window) < self.window_size:
            return None

        if any(
            window_pos < self.min_keep_tokens for window_pos, _ in self.valid_window
        ):
            return None

        avg_prob = (
            sum(window_prob for _, window_prob in self.valid_window) / self.window_size
        )
        if avg_prob > self.threshold:
            return None

        trigger_pos, _ = self.valid_window[0]
        return OnlinePruneCandidate(
            logical_cut_index=trigger_pos,
            detect_pos=pos,
            observed_stop_index=pos + 1,
            trigger_prob=avg_prob,
            reason=(
                f"valid window avg p<=threshold "
                f"({avg_prob:.6f}<={self.threshold:.6f}, size={self.window_size})"
            ),
        )


@dataclass
class EmaBelowOnlineState(BaseOnlineGenerationState):
    threshold: float
    alpha: float
    warmup_valid_points: int
    min_keep_tokens: int
    ema: float | None = None
    seen_valid_points: int = 0
    name: str = "ema_below"

    def step(self, *, pos: int, raw_prob: Any) -> OnlinePruneCandidate | None:
        prob = _normalize_prob(raw_prob)
        if prob is None:
            return None

        self.ema = (
            prob
            if self.ema is None
            else self.alpha * prob + (1.0 - self.alpha) * self.ema
        )
        self.seen_valid_points += 1
        if (
            self.seen_valid_points < self.warmup_valid_points
            or pos < self.min_keep_tokens
        ):
            return None
        if self.ema > self.threshold or prob > self.threshold:
            return None

        return OnlinePruneCandidate(
            logical_cut_index=pos,
            detect_pos=pos,
            observed_stop_index=pos + 1,
            trigger_prob=self.ema,
            reason=(
                f"ema and current prob both <= threshold "
                f"(ema={self.ema:.6f}, current={prob:.6f}, threshold={self.threshold:.6f})"
            ),
        )


@dataclass
class ArmedEmaBelowOnlineState(BaseOnlineGenerationState):
    threshold: float
    alpha: float
    warmup_valid_points: int
    arm_threshold: float
    min_keep_tokens: int
    armed: bool = False
    arm_pos: int | None = None
    ema: float | None = None
    seen_post_arm_valid_points: int = 0
    name: str = "armed_ema_below"

    def step(self, *, pos: int, raw_prob: Any) -> OnlinePruneCandidate | None:
        prob = _normalize_prob(raw_prob)
        if prob is None:
            return None

        if not self.armed:
            if prob >= self.arm_threshold:
                self.armed = True
                self.arm_pos = pos
                self.ema = None
                self.seen_post_arm_valid_points = 0
            return None

        self.ema = (
            prob
            if self.ema is None
            else self.alpha * prob + (1.0 - self.alpha) * self.ema
        )
        self.seen_post_arm_valid_points += 1
        if (
            self.seen_post_arm_valid_points < self.warmup_valid_points
            or pos < self.min_keep_tokens
        ):
            return None
        if self.ema > self.threshold or prob > self.threshold:
            return None

        return OnlinePruneCandidate(
            logical_cut_index=pos,
            detect_pos=pos,
            observed_stop_index=pos + 1,
            trigger_prob=self.ema,
            reason=(
                f"armed at pos={self.arm_pos}, then ema and current prob both <= threshold "
                f"(ema={self.ema:.6f}, current={prob:.6f}, threshold={self.threshold:.6f})"
            ),
        )


@dataclass
class ValidWindowNOutOfMOnlineState(BaseOnlineGenerationState):
    threshold: float
    n: int
    m: int
    min_keep_tokens: int
    valid_window: deque[tuple[int, float]] = field(default_factory=deque)
    name: str = "valid_window_n_out_of_m"

    def step(self, *, pos: int, raw_prob: Any) -> OnlinePruneCandidate | None:
        prob = _normalize_prob(raw_prob)
        if prob is None:
            return None

        self.valid_window.append((pos, prob))
        if len(self.valid_window) > self.m:
            self.valid_window.popleft()
        if len(self.valid_window) < self.m:
            return None

        if any(
            window_pos < self.min_keep_tokens for window_pos, _ in self.valid_window
        ):
            return None

        below = [
            (window_pos, window_prob)
            for window_pos, window_prob in self.valid_window
            if window_prob <= self.threshold
        ]
        if len(below) < self.n:
            return None

        trigger_pos, trigger_prob = below[0]
        return OnlinePruneCandidate(
            logical_cut_index=trigger_pos,
            detect_pos=pos,
            observed_stop_index=pos + 1,
            trigger_prob=trigger_prob,
            reason=f"{len(below)} out of {self.m} valid probes p<=threshold",
        )


@dataclass
class DropFromPeakOnlineState(BaseOnlineGenerationState):
    drop_threshold: float
    min_keep_tokens: int
    peak_prob: float = 0.0
    name: str = "drop_from_peak"

    def step(self, *, pos: int, raw_prob: Any) -> OnlinePruneCandidate | None:
        prob = _normalize_prob(raw_prob)
        if prob is None:
            return None

        if prob > self.peak_prob:
            self.peak_prob = prob
        if pos < self.min_keep_tokens or (self.peak_prob - prob) < self.drop_threshold:
            return None

        return OnlinePruneCandidate(
            logical_cut_index=pos,
            detect_pos=pos,
            observed_stop_index=pos + 1,
            trigger_prob=prob,
            reason=(
                f"prob dropped from peak "
                f"({self.peak_prob:.6f} - {prob:.6f} >= {self.drop_threshold:.6f})"
            ),
        )


def build_online_generation_state(
    strategy: BasePruneStrategy,
) -> BaseOnlineGenerationState:
    if strategy.name == "first_below":
        return FirstBelowOnlineState(
            threshold=float(strategy.threshold),
            min_keep_tokens=int(strategy.min_keep_tokens),
        )
    if strategy.name == "consecutive_below":
        return ConsecutiveBelowOnlineState(
            threshold=float(strategy.threshold),
            window_size=int(strategy.window_size),
            min_keep_tokens=int(strategy.min_keep_tokens),
        )
    if strategy.name == "armed_consecutive_below":
        return ArmedConsecutiveBelowOnlineState(
            threshold=float(strategy.threshold),
            window_size=int(strategy.window_size),
            arm_threshold=float(strategy.arm_threshold),
            min_keep_tokens=int(strategy.min_keep_tokens),
        )
    if strategy.name == "armed_valid_window_n_out_of_m":
        return ArmedValidWindowNOutOfMOnlineState(
            threshold=float(strategy.threshold),
            n=int(strategy.n),
            m=int(strategy.m),
            arm_threshold=float(strategy.arm_threshold),
            min_keep_tokens=int(strategy.min_keep_tokens),
        )
    if strategy.name == "valid_window_average":
        return ValidWindowAverageOnlineState(
            threshold=float(strategy.threshold),
            window_size=int(strategy.window_size),
            min_keep_tokens=int(strategy.min_keep_tokens),
        )
    if strategy.name == "ema_below":
        return EmaBelowOnlineState(
            threshold=float(strategy.threshold),
            alpha=float(strategy.alpha),
            warmup_valid_points=int(strategy.warmup_valid_points),
            min_keep_tokens=int(strategy.min_keep_tokens),
        )
    if strategy.name == "armed_ema_below":
        return ArmedEmaBelowOnlineState(
            threshold=float(strategy.threshold),
            alpha=float(strategy.alpha),
            warmup_valid_points=int(strategy.warmup_valid_points),
            arm_threshold=float(strategy.arm_threshold),
            min_keep_tokens=int(strategy.min_keep_tokens),
        )
    if strategy.name == "valid_window_n_out_of_m":
        return ValidWindowNOutOfMOnlineState(
            threshold=float(strategy.threshold),
            n=int(strategy.n),
            m=int(strategy.m),
            min_keep_tokens=int(strategy.min_keep_tokens),
        )
    if strategy.name == "drop_from_peak":
        return DropFromPeakOnlineState(
            drop_threshold=float(strategy.drop_threshold),
            min_keep_tokens=int(strategy.min_keep_tokens),
        )
    raise ValueError(
        f"Generation strategy '{strategy.name}' is not supported by strict online replay."
    )


@dataclass
class OnlineRecordState:
    record_idx: int
    output_len: int
    generation_state: BaseOnlineGenerationState | None
    status: str = "active"
    cursor: int = 0
    candidate_count: int = 0
    release_count: int = 0
    pending_candidate: OnlinePruneCandidate | None = None
    scheduled_candidate: OnlinePruneCandidate | None = None
    scheduled_stop_index: int | None = None
    final_decision: PruneDecision | None = None
    final_logical_cut_index: int | None = None
    final_detect_pos: int | None = None

    def is_terminal(self) -> bool:
        return self.status in {"committed", "finished"}


def _commit_now(
    *,
    state: OnlineRecordState,
    candidate: OnlinePruneCandidate,
    stop_index: int,
    reason_suffix: str,
) -> None:
    reason = (
        candidate.reason
        if not reason_suffix
        else f"{candidate.reason}; {reason_suffix}"
    )
    state.status = "committed"
    state.pending_candidate = None
    state.scheduled_candidate = None
    state.scheduled_stop_index = None
    state.final_logical_cut_index = candidate.logical_cut_index
    state.final_detect_pos = candidate.detect_pos
    state.final_decision = PruneDecision(
        should_prune=True,
        cut_index=stop_index,
        trigger_pos=candidate.detect_pos,
        trigger_prob=candidate.trigger_prob,
        reason=reason,
    )


def _schedule_stop(
    *,
    state: OnlineRecordState,
    candidate: OnlinePruneCandidate,
    stop_index: int,
) -> None:
    state.status = "scheduled"
    state.pending_candidate = None
    state.scheduled_candidate = candidate
    state.scheduled_stop_index = stop_index


def _release_pending(state: OnlineRecordState) -> None:
    state.status = "active"
    state.pending_candidate = None
    state.release_count += 1


def _finalize_keep(state: OnlineRecordState, *, reason: str | None = None) -> None:
    if state.final_decision is not None:
        return
    if reason is None:
        if state.candidate_count == 0:
            reason = "sequence ended without online prune"
        else:
            reason = (
                "sequence ended without committed online prune "
                f"(candidates={state.candidate_count}, releases={state.release_count})"
            )
    state.status = "finished"
    state.final_decision = _keep_decision(reason=reason)
    state.final_logical_cut_index = None
    state.final_detect_pos = None


class BaseOnlineInputController(ABC):
    name: str

    def __init__(self, *, group_size: int) -> None:
        self.group_size = group_size

    @abstractmethod
    def observe_round(self, *, states: list[OnlineRecordState]) -> bool:
        raise NotImplementedError


class DelegateSecondLevelOnlineController(BaseOnlineInputController):
    name = "delegate_second_level"

    def observe_round(self, *, states: list[OnlineRecordState]) -> bool:
        changed = False
        for state in states:
            candidate = state.pending_candidate
            if state.status != "pending" or candidate is None:
                continue
            _commit_now(
                state=state,
                candidate=candidate,
                stop_index=candidate.observed_stop_index,
                reason_suffix="strict_online immediate commit",
            )
            changed = True
        return changed


class GroupConsensusGateOnlineController(BaseOnlineInputController):
    name = "group_consensus_gate"

    def __init__(
        self,
        *,
        group_size: int,
        min_pruned_count: int,
        min_pruned_fraction: float,
    ) -> None:
        super().__init__(group_size=group_size)
        self.required = _prune_count_requirement(
            total=group_size,
            min_count=min_pruned_count,
            min_fraction=min_pruned_fraction,
        )
        self.gate_open = False

    def observe_round(self, *, states: list[OnlineRecordState]) -> bool:
        changed = False
        pending_states = [state for state in states if state.status == "pending"]
        active_states = [state for state in states if state.status == "active"]

        if self.gate_open:
            for state in pending_states:
                assert state.pending_candidate is not None
                _commit_now(
                    state=state,
                    candidate=state.pending_candidate,
                    stop_index=state.pending_candidate.observed_stop_index,
                    reason_suffix=(
                        "strict_online group_consensus_gate committed after quorum opened "
                        f"(required={self.required})"
                    ),
                )
                changed = True
            return changed

        if len(pending_states) >= self.required:
            self.gate_open = True
            for state in pending_states:
                assert state.pending_candidate is not None
                _commit_now(
                    state=state,
                    candidate=state.pending_candidate,
                    stop_index=state.pending_candidate.observed_stop_index,
                    reason_suffix=(
                        "strict_online group_consensus_gate quorum reached "
                        f"(pending={len(pending_states)}, required={self.required})"
                    ),
                )
                changed = True
            return changed

        if len(pending_states) + len(active_states) < self.required:
            for state in pending_states:
                _release_pending(state)
                changed = True
        return changed


class GroupQuorumSnapshotFloorOnlineController(BaseOnlineInputController):
    name = "group_quorum_snapshot_floor"

    def __init__(
        self,
        *,
        group_size: int,
        min_pruned_count: int,
        min_pruned_fraction: float,
        quantile: float,
    ) -> None:
        super().__init__(group_size=group_size)
        self.required = _prune_count_requirement(
            total=group_size,
            min_count=min_pruned_count,
            min_fraction=min_pruned_fraction,
        )
        self.quantile = quantile
        self.frozen_floor: int | None = None

    def observe_round(self, *, states: list[OnlineRecordState]) -> bool:
        changed = False
        pending_states = [state for state in states if state.status == "pending"]
        active_states = [state for state in states if state.status == "active"]

        if self.frozen_floor is not None:
            for state in pending_states:
                assert state.pending_candidate is not None
                stop_index = max(
                    state.pending_candidate.observed_stop_index,
                    self.frozen_floor,
                )
                if stop_index <= state.cursor:
                    _commit_now(
                        state=state,
                        candidate=state.pending_candidate,
                        stop_index=stop_index,
                        reason_suffix=(
                            "strict_online group_quorum_snapshot_floor committed with frozen floor "
                            f"{self.frozen_floor}"
                        ),
                    )
                else:
                    _schedule_stop(
                        state=state,
                        candidate=state.pending_candidate,
                        stop_index=stop_index,
                    )
                changed = True
            return changed

        if len(pending_states) >= self.required:
            logical_cuts = [
                int(state.pending_candidate.logical_cut_index)
                for state in pending_states
                if state.pending_candidate is not None
            ]
            self.frozen_floor = _quantile_floor_value(logical_cuts, self.quantile)
            for state in pending_states:
                assert state.pending_candidate is not None
                stop_index = max(
                    state.pending_candidate.observed_stop_index,
                    self.frozen_floor,
                )
                if stop_index <= state.cursor:
                    _commit_now(
                        state=state,
                        candidate=state.pending_candidate,
                        stop_index=stop_index,
                        reason_suffix=(
                            "strict_online group_quorum_snapshot_floor froze floor "
                            f"{self.frozen_floor} at quorum={len(pending_states)}"
                        ),
                    )
                else:
                    _schedule_stop(
                        state=state,
                        candidate=state.pending_candidate,
                        stop_index=stop_index,
                    )
                changed = True
            return changed

        if len(pending_states) + len(active_states) < self.required:
            for state in pending_states:
                _release_pending(state)
                changed = True
        return changed


def build_online_input_controller(
    *,
    strategy_name: str,
    strategy_args: dict[str, Any],
    group_size: int,
) -> BaseOnlineInputController:
    if strategy_name == "delegate_second_level":
        if strategy_args:
            raise ValueError(
                "delegate_second_level does not accept strategy args for strict online replay"
            )
        return DelegateSecondLevelOnlineController(group_size=group_size)

    if strategy_name == "group_consensus_gate":
        allowed_keys = {"min_pruned_count", "min_pruned_fraction"}
        unknown = sorted(set(strategy_args) - allowed_keys)
        if unknown:
            raise ValueError(
                f"Unsupported args for strict online input strategy={strategy_name}: {unknown}"
            )
        min_pruned_count = int(strategy_args.get("min_pruned_count", 2))
        min_pruned_fraction = float(strategy_args.get("min_pruned_fraction", 0.0))
        if min_pruned_count <= 0:
            raise ValueError(
                "--min-pruned-count must be > 0 for strategy=group_consensus_gate"
            )
        if min_pruned_fraction < 0.0 or min_pruned_fraction > 1.0:
            raise ValueError(
                "--min-pruned-fraction must be in [0, 1] for strategy=group_consensus_gate"
            )
        return GroupConsensusGateOnlineController(
            group_size=group_size,
            min_pruned_count=min_pruned_count,
            min_pruned_fraction=min_pruned_fraction,
        )

    if strategy_name == "group_quorum_snapshot_floor":
        allowed_keys = {"min_pruned_count", "min_pruned_fraction", "quantile"}
        unknown = sorted(set(strategy_args) - allowed_keys)
        if unknown:
            raise ValueError(
                f"Unsupported args for strict online input strategy={strategy_name}: {unknown}"
            )
        min_pruned_count = int(strategy_args.get("min_pruned_count", 2))
        min_pruned_fraction = float(strategy_args.get("min_pruned_fraction", 0.0))
        quantile = float(strategy_args.get("quantile", 0.5))
        if min_pruned_count <= 0:
            raise ValueError(
                "--min-pruned-count must be > 0 for strategy=group_quorum_snapshot_floor"
            )
        if min_pruned_fraction < 0.0 or min_pruned_fraction > 1.0:
            raise ValueError(
                "--min-pruned-fraction must be in [0, 1] for strategy=group_quorum_snapshot_floor"
            )
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError(
                "--quantile must be in [0, 1] for strategy=group_quorum_snapshot_floor"
            )
        return GroupQuorumSnapshotFloorOnlineController(
            group_size=group_size,
            min_pruned_count=min_pruned_count,
            min_pruned_fraction=min_pruned_fraction,
            quantile=quantile,
        )

    if strategy_name == "group_quantile_floor":
        raise ValueError(
            "group_quantile_floor is hindsight-based and cannot be replayed as strict online. "
            "Use group_quorum_snapshot_floor in run_probe_prune_online.py."
        )

    raise ValueError(f"Unsupported strict online input strategy: {strategy_name}")


def simulate_online_group(
    *,
    records: list[dict[str, Any]],
    generation_strategy: BasePruneStrategy,
    input_strategy_name: str,
    input_strategy_args: dict[str, Any],
) -> list[OnlineReplayOutcome]:
    controller = build_online_input_controller(
        strategy_name=input_strategy_name,
        strategy_args=input_strategy_args,
        group_size=len(records),
    )

    states: list[OnlineRecordState] = []
    for record_idx, row in enumerate(records):
        output_ids = row.get("output_ids")
        probs = row.get("probs")
        if not isinstance(output_ids, list):
            state = OnlineRecordState(
                record_idx=record_idx,
                output_len=0,
                generation_state=None,
                status="finished",
            )
            state.final_decision = _keep_decision(
                reason="invalid row: output_ids is not a list"
            )
            states.append(state)
            continue
        if not isinstance(probs, list) or len(probs) != len(output_ids):
            state = OnlineRecordState(
                record_idx=record_idx,
                output_len=len(output_ids),
                generation_state=None,
                status="finished",
            )
            state.final_decision = _keep_decision(
                reason="invalid row: probs missing or length mismatch"
            )
            states.append(state)
            continue

        states.append(
            OnlineRecordState(
                record_idx=record_idx,
                output_len=len(output_ids),
                generation_state=build_online_generation_state(generation_strategy),
            )
        )

    while not all(state.is_terminal() for state in states):
        changed = False

        for state in states:
            if state.status != "scheduled" or state.scheduled_stop_index is None:
                continue
            if state.cursor >= state.scheduled_stop_index:
                assert state.scheduled_candidate is not None
                _commit_now(
                    state=state,
                    candidate=state.scheduled_candidate,
                    stop_index=state.scheduled_stop_index,
                    reason_suffix=(
                        "strict_online scheduled stop reached "
                        f"(stop_index={state.scheduled_stop_index})"
                    ),
                )
                changed = True

        for state in states:
            if state.status == "active" and state.cursor >= state.output_len:
                _finalize_keep(state)
                changed = True

        for state in states:
            if (
                state.status == "scheduled"
                and state.scheduled_stop_index is not None
                and state.cursor >= state.output_len
            ):
                _finalize_keep(
                    state,
                    reason=(
                        "sequence ended before scheduled online stop "
                        f"(scheduled_stop_index={state.scheduled_stop_index}, "
                        f"output_len={state.output_len}, candidates={state.candidate_count}, "
                        f"releases={state.release_count})"
                    ),
                )
                changed = True

        round_had_progress = False
        for state in states:
            if (
                state.status == "scheduled"
                and state.scheduled_stop_index is not None
                and state.cursor < min(state.scheduled_stop_index, state.output_len)
            ):
                state.cursor += 1
                changed = True
                round_had_progress = True

        for state in states:
            if state.status != "active":
                continue
            if state.cursor >= state.output_len:
                continue
            assert state.generation_state is not None
            row = records[state.record_idx]
            probs = row["probs"]
            pos = state.cursor
            candidate = state.generation_state.step(pos=pos, raw_prob=probs[pos])
            state.cursor += 1
            changed = True
            round_had_progress = True
            if candidate is not None:
                state.status = "pending"
                state.pending_candidate = candidate
                state.candidate_count += 1

        if controller.observe_round(states=states):
            changed = True

        for state in states:
            if state.status == "active" and state.cursor >= state.output_len:
                _finalize_keep(state)
                changed = True

        if not changed:
            raise RuntimeError(
                f"Strict online replay made no progress for input strategy '{controller.name}'."
            )
        if not round_had_progress:
            continue

    outcomes: list[OnlineReplayOutcome] = []
    for state in states:
        assert state.final_decision is not None
        outcomes.append(
            OnlineReplayOutcome(
                decision=state.final_decision,
                logical_cut_index=state.final_logical_cut_index,
                detect_pos=state.final_detect_pos,
                candidate_count=state.candidate_count,
                release_count=state.release_count,
                final_state=state.status,
            )
        )
    return outcomes
