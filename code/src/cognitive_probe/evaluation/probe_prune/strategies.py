from __future__ import annotations

import math
from abc import ABC
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class PruneDecision:
    should_prune: bool
    cut_index: int | None
    trigger_pos: int | None
    trigger_prob: float | None
    reason: str


class BasePruneStrategy(ABC):
    name: str

    def decide(self, *, probs: list[Any], output_len: int) -> PruneDecision:
        raise NotImplementedError(
            f"Strategy '{self.name}' does not implement per-record decide()."
        )

    def decide_group(
        self,
        *,
        records: list[dict[str, Any]],
        generation_strategy: BasePruneStrategy | None = None,
    ) -> list[PruneDecision]:
        strategy = generation_strategy or self
        return _delegate_group_decisions(records=records, strategy=strategy)


def _keep_decision(*, reason: str) -> PruneDecision:
    return PruneDecision(
        should_prune=False,
        cut_index=None,
        trigger_pos=None,
        trigger_prob=None,
        reason=reason,
    )


def _delegate_group_decisions(
    *,
    records: list[dict[str, Any]],
    strategy: BasePruneStrategy,
) -> list[PruneDecision]:
    decisions: list[PruneDecision] = []
    for row in records:
        output_ids = row.get("output_ids")
        probs = row.get("probs")

        if not isinstance(output_ids, list):
            decisions.append(
                _keep_decision(reason="invalid row: output_ids is not a list")
            )
            continue

        if not isinstance(probs, list) or len(probs) != len(output_ids):
            decisions.append(
                _keep_decision(reason="invalid row: probs missing or length mismatch")
            )
            continue

        decisions.append(strategy.decide(probs=probs, output_len=len(output_ids)))

    return decisions


def _prune_count_requirement(*, total: int, min_count: int, min_fraction: float) -> int:
    return max(min_count, int(math.ceil(total * min_fraction - 1e-12)))


def _quantile_floor_value(values: list[int], quantile: float) -> int:
    if not values:
        raise ValueError("values must be non-empty")
    sorted_values = sorted(values)
    idx = int(math.ceil(quantile * (len(sorted_values) - 1)))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def iter_valid_probs(probs: list[Any]) -> list[tuple[int, float]]:
    valid: list[tuple[int, float]] = []
    for idx, value in enumerate(probs):
        if value is None:
            continue
        try:
            p = float(value)
        except (TypeError, ValueError):
            continue
        if p < 0.0:
            continue
        valid.append((idx, p))
    return valid


@dataclass
class FirstBelowThresholdStrategy(BasePruneStrategy):
    threshold: float
    min_keep_tokens: int
    name: str = "first_below"

    def decide(self, *, probs: list[Any], output_len: int) -> PruneDecision:
        valid = iter_valid_probs(probs)
        for pos, p in valid:
            if pos < self.min_keep_tokens:
                continue
            if p <= self.threshold:
                return PruneDecision(
                    should_prune=True,
                    cut_index=pos,
                    trigger_pos=pos,
                    trigger_prob=p,
                    reason=f"first p<=threshold ({p:.6f}<={self.threshold:.6f})",
                )
        return _keep_decision(reason="no position satisfies first_below")


@dataclass
class ConsecutiveBelowThresholdStrategy(BasePruneStrategy):
    threshold: float
    window_size: int
    min_keep_tokens: int
    name: str = "consecutive_below"

    def decide(self, *, probs: list[Any], output_len: int) -> PruneDecision:
        valid = iter_valid_probs(probs)
        if len(valid) < self.window_size:
            return _keep_decision(
                reason=f"insufficient valid probe points ({len(valid)}<{self.window_size})"
            )

        for start in range(0, len(valid) - self.window_size + 1):
            window = valid[start : start + self.window_size]
            if any(pos < self.min_keep_tokens for pos, _ in window):
                continue
            if all(p <= self.threshold for _, p in window):
                trigger_pos, trigger_prob = window[0]
                return PruneDecision(
                    should_prune=True,
                    cut_index=trigger_pos,
                    trigger_pos=trigger_pos,
                    trigger_prob=trigger_prob,
                    reason=(
                        f"{self.window_size} consecutive valid points p<=threshold "
                        f"(start_prob={trigger_prob:.6f}, threshold={self.threshold:.6f})"
                    ),
                )

        return _keep_decision(reason="no window satisfies consecutive_below")


@dataclass
class ArmedConsecutiveBelowStrategy(BasePruneStrategy):
    threshold: float
    window_size: int
    arm_threshold: float
    min_keep_tokens: int
    name: str = "armed_consecutive_below"

    def decide(self, *, probs: list[Any], output_len: int) -> PruneDecision:
        valid = iter_valid_probs(probs)
        if len(valid) < self.window_size:
            return _keep_decision(
                reason=f"insufficient valid probe points ({len(valid)}<{self.window_size})"
            )

        arm_idx: int | None = None
        arm_pos: int | None = None
        for idx, (pos, p) in enumerate(valid):
            if p >= self.arm_threshold:
                arm_idx = idx
                arm_pos = pos
                break

        if arm_idx is None:
            return _keep_decision(
                reason=f"never armed because no valid probe reached arm_threshold={self.arm_threshold:.6f}"
            )

        for start in range(arm_idx + 1, len(valid) - self.window_size + 1):
            window = valid[start : start + self.window_size]
            if any(pos < self.min_keep_tokens for pos, _ in window):
                continue
            if all(p <= self.threshold for _, p in window):
                trigger_pos, trigger_prob = window[0]
                return PruneDecision(
                    should_prune=True,
                    cut_index=trigger_pos,
                    trigger_pos=trigger_pos,
                    trigger_prob=trigger_prob,
                    reason=(
                        f"armed at pos={arm_pos}, then {self.window_size} consecutive valid points "
                        f"p<=threshold (start_prob={trigger_prob:.6f}, threshold={self.threshold:.6f})"
                    ),
                )

        return _keep_decision(
            reason="no armed window satisfies armed_consecutive_below"
        )


@dataclass
class ArmedValidWindowNOutOfMStrategy(BasePruneStrategy):
    threshold: float
    n: int
    m: int
    arm_threshold: float
    min_keep_tokens: int
    name: str = "armed_valid_window_n_out_of_m"

    def decide(self, *, probs: list[Any], output_len: int) -> PruneDecision:
        valid = iter_valid_probs(probs)

        arm_idx: int | None = None
        arm_pos: int | None = None
        for idx, (pos, p) in enumerate(valid):
            if p >= self.arm_threshold:
                arm_idx = idx
                arm_pos = pos
                break

        if arm_idx is None:
            return _keep_decision(
                reason=f"never armed because no valid probe reached arm_threshold={self.arm_threshold:.6f}"
            )

        post_arm_valid = valid[arm_idx + 1 :]
        if len(post_arm_valid) < self.m:
            return _keep_decision(
                reason=(
                    "insufficient post-arm valid probe points "
                    f"({len(post_arm_valid)}<{self.m})"
                )
            )

        for start in range(0, len(post_arm_valid) - self.m + 1):
            window = post_arm_valid[start : start + self.m]
            if any(pos < self.min_keep_tokens for pos, _ in window):
                continue

            below = [(pos, p) for pos, p in window if p <= self.threshold]
            if len(below) < self.n:
                continue

            trigger_pos, trigger_prob = below[0]
            return PruneDecision(
                should_prune=True,
                cut_index=trigger_pos,
                trigger_pos=trigger_pos,
                trigger_prob=trigger_prob,
                reason=(
                    f"armed at pos={arm_pos}, then {len(below)} out of {self.m} "
                    f"valid probes p<=threshold"
                ),
            )

        return _keep_decision(
            reason="no armed window satisfies armed_valid_window_n_out_of_m"
        )


@dataclass
class ValidWindowAverageStrategy(BasePruneStrategy):
    threshold: float
    window_size: int
    min_keep_tokens: int
    name: str = "valid_window_average"

    def decide(self, *, probs: list[Any], output_len: int) -> PruneDecision:
        valid = iter_valid_probs(probs)
        if len(valid) < self.window_size:
            return _keep_decision(
                reason=f"insufficient valid probe points ({len(valid)}<{self.window_size})"
            )

        for start in range(0, len(valid) - self.window_size + 1):
            window = valid[start : start + self.window_size]
            if any(pos < self.min_keep_tokens for pos, _ in window):
                continue

            avg_prob = sum(p for _, p in window) / self.window_size
            if avg_prob <= self.threshold:
                trigger_pos, _ = window[0]
                return PruneDecision(
                    should_prune=True,
                    cut_index=trigger_pos,
                    trigger_pos=trigger_pos,
                    trigger_prob=avg_prob,
                    reason=(
                        f"valid window avg p<=threshold "
                        f"({avg_prob:.6f}<={self.threshold:.6f}, size={self.window_size})"
                    ),
                )

        return _keep_decision(reason="no window satisfies valid_window_average")


@dataclass
class EmaBelowStrategy(BasePruneStrategy):
    threshold: float
    alpha: float
    warmup_valid_points: int
    min_keep_tokens: int
    name: str = "ema_below"

    def decide(self, *, probs: list[Any], output_len: int) -> PruneDecision:
        valid = iter_valid_probs(probs)
        if len(valid) < self.warmup_valid_points:
            return _keep_decision(
                reason=(
                    "insufficient valid probe points "
                    f"({len(valid)}<{self.warmup_valid_points})"
                )
            )

        ema: float | None = None
        seen = 0
        for pos, p in valid:
            ema = p if ema is None else self.alpha * p + (1.0 - self.alpha) * ema
            seen += 1

            if seen < self.warmup_valid_points:
                continue
            if pos < self.min_keep_tokens:
                continue
            if ema <= self.threshold and p <= self.threshold:
                return PruneDecision(
                    should_prune=True,
                    cut_index=pos,
                    trigger_pos=pos,
                    trigger_prob=ema,
                    reason=(
                        f"ema and current prob both <= threshold "
                        f"(ema={ema:.6f}, current={p:.6f}, threshold={self.threshold:.6f})"
                    ),
                )

        return _keep_decision(reason="no position satisfies ema_below")


@dataclass
class ArmedEmaBelowStrategy(BasePruneStrategy):
    threshold: float
    alpha: float
    warmup_valid_points: int
    arm_threshold: float
    min_keep_tokens: int
    name: str = "armed_ema_below"

    def decide(self, *, probs: list[Any], output_len: int) -> PruneDecision:
        valid = iter_valid_probs(probs)

        arm_idx: int | None = None
        arm_pos: int | None = None
        for idx, (pos, p) in enumerate(valid):
            if p >= self.arm_threshold:
                arm_idx = idx
                arm_pos = pos
                break

        if arm_idx is None:
            return _keep_decision(
                reason=f"never armed because no valid probe reached arm_threshold={self.arm_threshold:.6f}"
            )

        ema: float | None = None
        seen_post_arm = 0
        for pos, p in valid[arm_idx + 1 :]:
            ema = p if ema is None else self.alpha * p + (1.0 - self.alpha) * ema
            seen_post_arm += 1

            if seen_post_arm < self.warmup_valid_points:
                continue
            if pos < self.min_keep_tokens:
                continue
            if ema <= self.threshold and p <= self.threshold:
                return PruneDecision(
                    should_prune=True,
                    cut_index=pos,
                    trigger_pos=pos,
                    trigger_prob=ema,
                    reason=(
                        f"armed at pos={arm_pos}, then ema and current prob both <= threshold "
                        f"(ema={ema:.6f}, current={p:.6f}, threshold={self.threshold:.6f})"
                    ),
                )

        if seen_post_arm < self.warmup_valid_points:
            return _keep_decision(
                reason=(
                    "insufficient post-arm valid probe points "
                    f"({seen_post_arm}<{self.warmup_valid_points})"
                )
            )

        return _keep_decision(reason="no position satisfies armed_ema_below")


@dataclass
class ValidWindowNOutOfMStrategy(BasePruneStrategy):
    threshold: float
    n: int
    m: int
    min_keep_tokens: int
    name: str = "valid_window_n_out_of_m"

    def decide(self, *, probs: list[Any], output_len: int) -> PruneDecision:
        valid = iter_valid_probs(probs)
        if len(valid) < self.m:
            return _keep_decision(
                reason=f"insufficient valid probe points ({len(valid)}<{self.m})"
            )

        for start in range(0, len(valid) - self.m + 1):
            window = valid[start : start + self.m]
            if any(pos < self.min_keep_tokens for pos, _ in window):
                continue

            below_count = sum(1 for _, p in window if p <= self.threshold)
            if below_count >= self.n:
                trigger_pos, trigger_prob = next(
                    (pos, p) for pos, p in window if p <= self.threshold
                )
                return PruneDecision(
                    should_prune=True,
                    cut_index=trigger_pos,
                    trigger_pos=trigger_pos,
                    trigger_prob=trigger_prob,
                    reason=f"{below_count} out of {self.m} valid probes p<=threshold",
                )

        return _keep_decision(reason="no window satisfies valid_window_n_out_of_m")


@dataclass
class DropFromPeakStrategy(BasePruneStrategy):
    drop_threshold: float
    min_keep_tokens: int
    name: str = "drop_from_peak"

    def decide(self, *, probs: list[Any], output_len: int) -> PruneDecision:
        valid = iter_valid_probs(probs)
        peak_prob = 0.0

        for pos, p in valid:
            if p > peak_prob:
                peak_prob = p

            if pos < self.min_keep_tokens:
                continue

            if (peak_prob - p) >= self.drop_threshold:
                return PruneDecision(
                    should_prune=True,
                    cut_index=pos,
                    trigger_pos=pos,
                    trigger_prob=p,
                    reason=(
                        f"prob dropped from peak "
                        f"({peak_prob:.6f} - {p:.6f} >= {self.drop_threshold:.6f})"
                    ),
                )

        return _keep_decision(reason="no position satisfies drop_from_peak")


@dataclass
class DelegateSecondLevelStrategy(BasePruneStrategy):
    """Default first-level strategy: directly delegate each record."""

    name: str = "delegate_second_level"


@dataclass
class GroupConsensusGateStrategy(BasePruneStrategy):
    min_pruned_count: int
    min_pruned_fraction: float
    name: str = "group_consensus_gate"

    def decide_group(
        self,
        *,
        records: list[dict[str, Any]],
        generation_strategy: BasePruneStrategy | None = None,
    ) -> list[PruneDecision]:
        if generation_strategy is None:
            raise ValueError(
                "group_consensus_gate requires a second-level generation strategy"
            )

        decisions = _delegate_group_decisions(
            records=records, strategy=generation_strategy
        )
        prune_count = sum(1 for decision in decisions if decision.should_prune)
        required = _prune_count_requirement(
            total=len(decisions),
            min_count=self.min_pruned_count,
            min_fraction=self.min_pruned_fraction,
        )
        if prune_count >= required:
            return decisions

        gated_reason = f"group_consensus_gate vetoed prune_count={prune_count}, required>={required}"
        gated_decisions: list[PruneDecision] = []
        for decision in decisions:
            if decision.should_prune:
                gated_decisions.append(
                    _keep_decision(reason=f"{gated_reason}; original={decision.reason}")
                )
            else:
                gated_decisions.append(decision)
        return gated_decisions


@dataclass
class GroupQuantileFloorStrategy(BasePruneStrategy):
    min_pruned_count: int
    quantile: float
    name: str = "group_quantile_floor"

    def decide_group(
        self,
        *,
        records: list[dict[str, Any]],
        generation_strategy: BasePruneStrategy | None = None,
    ) -> list[PruneDecision]:
        if generation_strategy is None:
            raise ValueError(
                "group_quantile_floor requires a second-level generation strategy"
            )

        decisions = _delegate_group_decisions(
            records=records, strategy=generation_strategy
        )
        cut_indices = [
            int(decision.cut_index)
            for decision in decisions
            if decision.should_prune and decision.cut_index is not None
        ]
        if len(cut_indices) < self.min_pruned_count:
            return decisions

        floor_cut = _quantile_floor_value(cut_indices, self.quantile)
        adjusted: list[PruneDecision] = []
        for decision in decisions:
            if not decision.should_prune or decision.cut_index is None:
                adjusted.append(decision)
                continue

            new_cut_index = max(int(decision.cut_index), floor_cut)
            if new_cut_index == decision.cut_index:
                adjusted.append(decision)
                continue

            adjusted.append(
                PruneDecision(
                    should_prune=True,
                    cut_index=new_cut_index,
                    trigger_pos=decision.trigger_pos,
                    trigger_prob=decision.trigger_prob,
                    reason=(
                        f"{decision.reason}; raised cut_index from {decision.cut_index} "
                        f"to group q={self.quantile:g} floor {new_cut_index}"
                    ),
                )
            )
        return adjusted


GenerationStrategyBuilder = Callable[[dict[str, Any]], BasePruneStrategy]
InputStrategyBuilder = Callable[[dict[str, Any]], BasePruneStrategy]

_GENERATION_STRATEGY_BUILDERS: dict[str, GenerationStrategyBuilder] = {}
_INPUT_STRATEGY_BUILDERS: dict[str, InputStrategyBuilder] = {}


def _register_generation_strategy(
    name: str,
) -> Callable[[GenerationStrategyBuilder], GenerationStrategyBuilder]:
    def _decorator(builder: GenerationStrategyBuilder) -> GenerationStrategyBuilder:
        if name in _GENERATION_STRATEGY_BUILDERS:
            raise RuntimeError(f"Duplicate generation strategy registration: {name}")
        _GENERATION_STRATEGY_BUILDERS[name] = builder
        return builder

    return _decorator


def _register_input_strategy(
    name: str,
) -> Callable[[InputStrategyBuilder], InputStrategyBuilder]:
    def _decorator(builder: InputStrategyBuilder) -> InputStrategyBuilder:
        if name in _INPUT_STRATEGY_BUILDERS:
            raise RuntimeError(f"Duplicate input strategy registration: {name}")
        _INPUT_STRATEGY_BUILDERS[name] = builder
        return builder

    return _decorator


def _ensure_no_unknown_args(
    *,
    strategy_name: str,
    strategy_args: Mapping[str, Any],
    allowed_keys: set[str],
) -> None:
    unknown = sorted(set(strategy_args.keys()) - allowed_keys)
    if unknown:
        raise ValueError(
            f"Unsupported args for strategy={strategy_name}: {unknown}. "
            f"Allowed args: {sorted(allowed_keys)}"
        )


def _get_float(
    *,
    strategy_name: str,
    strategy_args: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    raw = strategy_args.get(key, default)
    try:
        return float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid float arg for strategy={strategy_name}: {key}={raw!r}"
        ) from e


def _get_int(
    *,
    strategy_name: str,
    strategy_args: Mapping[str, Any],
    key: str,
    default: int,
) -> int:
    raw = strategy_args.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid int arg for strategy={strategy_name}: {key}={raw!r}"
        ) from e


@_register_generation_strategy("first_below")
def _build_first_below(strategy_args: dict[str, Any]) -> BasePruneStrategy:
    strategy_name = "first_below"
    _ensure_no_unknown_args(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        allowed_keys={"threshold", "min_keep_tokens"},
    )
    return FirstBelowThresholdStrategy(
        threshold=_get_float(
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            key="threshold",
            default=0.3,
        ),
        min_keep_tokens=_get_int(
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            key="min_keep_tokens",
            default=0,
        ),
    )


@_register_generation_strategy("consecutive_below")
def _build_consecutive_below(strategy_args: dict[str, Any]) -> BasePruneStrategy:
    strategy_name = "consecutive_below"
    _ensure_no_unknown_args(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        allowed_keys={"threshold", "window_size", "min_keep_tokens"},
    )
    window_size = _get_int(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="window_size",
        default=2,
    )
    if window_size <= 0:
        raise ValueError("--window-size must be > 0 for strategy=consecutive_below")
    return ConsecutiveBelowThresholdStrategy(
        threshold=_get_float(
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            key="threshold",
            default=0.3,
        ),
        window_size=window_size,
        min_keep_tokens=_get_int(
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            key="min_keep_tokens",
            default=0,
        ),
    )


@_register_generation_strategy("armed_consecutive_below")
def _build_armed_consecutive_below(strategy_args: dict[str, Any]) -> BasePruneStrategy:
    strategy_name = "armed_consecutive_below"
    _ensure_no_unknown_args(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        allowed_keys={"threshold", "window_size", "arm_threshold", "min_keep_tokens"},
    )
    threshold = _get_float(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="threshold",
        default=0.3,
    )
    arm_threshold = _get_float(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="arm_threshold",
        default=0.6,
    )
    window_size = _get_int(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="window_size",
        default=2,
    )
    if window_size <= 0:
        raise ValueError(
            "--window-size must be > 0 for strategy=armed_consecutive_below"
        )
    if arm_threshold <= threshold:
        raise ValueError(
            "--arm-threshold must be > threshold for strategy=armed_consecutive_below"
        )
    return ArmedConsecutiveBelowStrategy(
        threshold=threshold,
        window_size=window_size,
        arm_threshold=arm_threshold,
        min_keep_tokens=_get_int(
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            key="min_keep_tokens",
            default=0,
        ),
    )


@_register_generation_strategy("armed_valid_window_n_out_of_m")
def _build_armed_valid_window_n_out_of_m(
    strategy_args: dict[str, Any],
) -> BasePruneStrategy:
    strategy_name = "armed_valid_window_n_out_of_m"
    _ensure_no_unknown_args(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        allowed_keys={"threshold", "n", "m", "arm_threshold", "min_keep_tokens"},
    )
    threshold = _get_float(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="threshold",
        default=0.3,
    )
    arm_threshold = _get_float(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="arm_threshold",
        default=0.6,
    )
    n = _get_int(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="n",
        default=2,
    )
    m = _get_int(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="m",
        default=3,
    )
    if m <= 0 or n <= 0 or n > m:
        raise ValueError("Invalid n or m for strategy=armed_valid_window_n_out_of_m")
    if arm_threshold <= threshold:
        raise ValueError(
            "--arm-threshold must be > threshold for strategy=armed_valid_window_n_out_of_m"
        )
    return ArmedValidWindowNOutOfMStrategy(
        threshold=threshold,
        n=n,
        m=m,
        arm_threshold=arm_threshold,
        min_keep_tokens=_get_int(
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            key="min_keep_tokens",
            default=0,
        ),
    )


@_register_generation_strategy("valid_window_average")
def _build_valid_window_average(strategy_args: dict[str, Any]) -> BasePruneStrategy:
    strategy_name = "valid_window_average"
    _ensure_no_unknown_args(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        allowed_keys={"threshold", "window_size", "min_keep_tokens"},
    )
    window_size = _get_int(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="window_size",
        default=2,
    )
    if window_size <= 0:
        raise ValueError("--window-size must be > 0 for strategy=valid_window_average")
    return ValidWindowAverageStrategy(
        threshold=_get_float(
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            key="threshold",
            default=0.3,
        ),
        window_size=window_size,
        min_keep_tokens=_get_int(
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            key="min_keep_tokens",
            default=0,
        ),
    )


@_register_generation_strategy("ema_below")
def _build_ema_below(strategy_args: dict[str, Any]) -> BasePruneStrategy:
    strategy_name = "ema_below"
    _ensure_no_unknown_args(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        allowed_keys={"threshold", "alpha", "warmup_valid_points", "min_keep_tokens"},
    )
    alpha = _get_float(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="alpha",
        default=0.35,
    )
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("--alpha must be in (0, 1] for strategy=ema_below")
    warmup_valid_points = _get_int(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="warmup_valid_points",
        default=3,
    )
    if warmup_valid_points <= 0:
        raise ValueError("--warmup-valid-points must be > 0 for strategy=ema_below")
    return EmaBelowStrategy(
        threshold=_get_float(
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            key="threshold",
            default=0.3,
        ),
        alpha=alpha,
        warmup_valid_points=warmup_valid_points,
        min_keep_tokens=_get_int(
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            key="min_keep_tokens",
            default=0,
        ),
    )


@_register_generation_strategy("armed_ema_below")
def _build_armed_ema_below(strategy_args: dict[str, Any]) -> BasePruneStrategy:
    strategy_name = "armed_ema_below"
    _ensure_no_unknown_args(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        allowed_keys={
            "threshold",
            "alpha",
            "warmup_valid_points",
            "arm_threshold",
            "min_keep_tokens",
        },
    )
    threshold = _get_float(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="threshold",
        default=0.3,
    )
    arm_threshold = _get_float(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="arm_threshold",
        default=0.6,
    )
    alpha = _get_float(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="alpha",
        default=0.35,
    )
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("--alpha must be in (0, 1] for strategy=armed_ema_below")
    warmup_valid_points = _get_int(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="warmup_valid_points",
        default=3,
    )
    if warmup_valid_points <= 0:
        raise ValueError(
            "--warmup-valid-points must be > 0 for strategy=armed_ema_below"
        )
    if arm_threshold <= threshold:
        raise ValueError(
            "--arm-threshold must be > threshold for strategy=armed_ema_below"
        )
    return ArmedEmaBelowStrategy(
        threshold=threshold,
        alpha=alpha,
        warmup_valid_points=warmup_valid_points,
        arm_threshold=arm_threshold,
        min_keep_tokens=_get_int(
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            key="min_keep_tokens",
            default=0,
        ),
    )


@_register_generation_strategy("valid_window_n_out_of_m")
def _build_valid_window_n_out_of_m(strategy_args: dict[str, Any]) -> BasePruneStrategy:
    strategy_name = "valid_window_n_out_of_m"
    _ensure_no_unknown_args(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        allowed_keys={"threshold", "n", "m", "min_keep_tokens"},
    )
    n = _get_int(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="n",
        default=2,
    )
    m = _get_int(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="m",
        default=3,
    )
    if m <= 0 or n <= 0 or n > m:
        raise ValueError("Invalid n or m for strategy=valid_window_n_out_of_m")
    return ValidWindowNOutOfMStrategy(
        threshold=_get_float(
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            key="threshold",
            default=0.3,
        ),
        n=n,
        m=m,
        min_keep_tokens=_get_int(
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            key="min_keep_tokens",
            default=0,
        ),
    )


@_register_generation_strategy("drop_from_peak")
def _build_drop_from_peak(strategy_args: dict[str, Any]) -> BasePruneStrategy:
    strategy_name = "drop_from_peak"
    _ensure_no_unknown_args(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        allowed_keys={"drop_threshold", "min_keep_tokens"},
    )
    return DropFromPeakStrategy(
        drop_threshold=_get_float(
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            key="drop_threshold",
            default=0.2,
        ),
        min_keep_tokens=_get_int(
            strategy_name=strategy_name,
            strategy_args=strategy_args,
            key="min_keep_tokens",
            default=0,
        ),
    )


@_register_input_strategy("delegate_second_level")
def _build_delegate_second_level(strategy_args: dict[str, Any]) -> BasePruneStrategy:
    strategy_name = "delegate_second_level"
    _ensure_no_unknown_args(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        allowed_keys=set(),
    )
    return DelegateSecondLevelStrategy()


@_register_input_strategy("group_consensus_gate")
def _build_group_consensus_gate(strategy_args: dict[str, Any]) -> BasePruneStrategy:
    strategy_name = "group_consensus_gate"
    _ensure_no_unknown_args(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        allowed_keys={"min_pruned_count", "min_pruned_fraction"},
    )
    min_pruned_count = _get_int(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="min_pruned_count",
        default=2,
    )
    min_pruned_fraction = _get_float(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="min_pruned_fraction",
        default=0.0,
    )
    if min_pruned_count <= 0:
        raise ValueError(
            "--min-pruned-count must be > 0 for strategy=group_consensus_gate"
        )
    if min_pruned_fraction < 0.0 or min_pruned_fraction > 1.0:
        raise ValueError(
            "--min-pruned-fraction must be in [0, 1] for strategy=group_consensus_gate"
        )
    return GroupConsensusGateStrategy(
        min_pruned_count=min_pruned_count,
        min_pruned_fraction=min_pruned_fraction,
    )


@_register_input_strategy("group_quantile_floor")
def _build_group_quantile_floor(strategy_args: dict[str, Any]) -> BasePruneStrategy:
    strategy_name = "group_quantile_floor"
    _ensure_no_unknown_args(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        allowed_keys={"min_pruned_count", "quantile"},
    )
    min_pruned_count = _get_int(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="min_pruned_count",
        default=2,
    )
    quantile = _get_float(
        strategy_name=strategy_name,
        strategy_args=strategy_args,
        key="quantile",
        default=0.5,
    )
    if min_pruned_count <= 0:
        raise ValueError(
            "--min-pruned-count must be > 0 for strategy=group_quantile_floor"
        )
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError(
            "--quantile must be in [0, 1] for strategy=group_quantile_floor"
        )
    return GroupQuantileFloorStrategy(
        min_pruned_count=min_pruned_count,
        quantile=quantile,
    )


def available_generation_strategies() -> list[str]:
    return sorted(_GENERATION_STRATEGY_BUILDERS.keys())


def available_input_strategies() -> list[str]:
    return sorted(_INPUT_STRATEGY_BUILDERS.keys())


def build_strategy(
    *,
    strategy_name: str,
    strategy_args: Mapping[str, Any] | None = None,
    **legacy_args: Any,
) -> BasePruneStrategy:
    merged_args: dict[str, Any] = {}
    if strategy_args:
        merged_args.update(dict(strategy_args))
    for key, value in legacy_args.items():
        if key not in merged_args and value is not None:
            merged_args[key] = value

    builder = _GENERATION_STRATEGY_BUILDERS.get(strategy_name)
    if builder is None:
        raise ValueError(
            f"Unsupported strategy: {strategy_name}. "
            f"Available generation strategies: {available_generation_strategies()}"
        )
    return builder(merged_args)


def build_input_strategy(
    *,
    strategy_name: str,
    strategy_args: Mapping[str, Any] | None = None,
) -> BasePruneStrategy:
    merged_args = dict(strategy_args or {})
    builder = _INPUT_STRATEGY_BUILDERS.get(strategy_name)
    if builder is None:
        raise ValueError(
            f"Unsupported input strategy: {strategy_name}. "
            f"Available input strategies: {available_input_strategies()}"
        )
    return builder(merged_args)
