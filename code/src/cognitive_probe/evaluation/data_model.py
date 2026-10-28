from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

RUN_OVERVIEW_FILE = "run_overview.json"
GENERATION_SEGMENTS_FILE = "generation_segments.jsonl"
DECISION_LOG_FILE = "decision_log.jsonl"
FINAL_OUTPUTS_FILE = "final_outputs.jsonl"
PRICE_SNAPSHOT_FILE = "price_snapshot.json"
SOURCE_TARGETS_FILE = "source_targets.jsonl"


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value]
    return value


class JsonDataclass:
    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass(slots=True)
class PriceEntry(JsonDataclass):
    input_token_price: float | None = None
    output_token_price: float | None = None
    cached_input_token_price: float | None = None
    request_price: float | None = None
    budget_unit: str | None = None


@dataclass(slots=True)
class PriceConfig(JsonDataclass):
    currency: str | None = None
    default: PriceEntry = field(default_factory=PriceEntry)
    models: dict[str, PriceEntry] = field(default_factory=dict)


@dataclass(slots=True)
class SegmentUsage(JsonDataclass):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    request_count: int = 1


@dataclass(slots=True)
class RunOverview(JsonDataclass):
    schema_version: int = SCHEMA_VERSION
    run_id: str = ""
    timestamp: str = ""
    run_type: str = "baseline"
    dataset_name: str = ""
    model: str = ""
    model_name: str = ""
    run_dir: str = ""
    source_format: str = "legacy_jsonl"
    strategy_name: str = "baseline"
    strategy_label: str = "baseline"
    reference_run_id: str | None = None
    reference_run_dir: str | None = None
    probe_run_id: str | None = None
    probe_run_dir: str | None = None
    probe_name: str | None = None
    probe_label: str | None = None
    baseline_run_id: str | None = None
    baseline_run_dir: str | None = None
    budget_type: str | None = None
    budget_limit: float | None = None
    budget_scope: str | None = None
    price_source: str | None = None
    price_snapshot_file: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GenerationSegment(JsonDataclass):
    segment_id: str
    sample_id: str
    parent_segment_id: str | None = None
    executor_name: str = "model"
    model_name: str = ""
    seed: int | None = None
    output_ids: list[int] = field(default_factory=list)
    response_text: str = ""
    probs: list[float] = field(default_factory=list)
    usage: SegmentUsage = field(default_factory=SegmentUsage)
    visible_output_tokens: int = 0
    source_row_index: int | None = None


@dataclass(slots=True)
class DecisionRecord(JsonDataclass):
    decision_type: str
    sample_id: str
    output_id: str | None = None
    segment_id: str | None = None
    original_output_tokens: int | None = None
    kept_output_tokens: int | None = None
    discarded_output_tokens: int | None = None
    reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FinalOutput(JsonDataclass):
    output_id: str
    sample_id: str
    final_segment_id: str
    response: str
    finish_reason: str | None = None
    k_group: int | None = None
    source_row_index: int | None = None
    was_pruned: bool = False
    would_pass_without_prune: bool | None = None
    source_output_tokens: int | None = None


@dataclass(slots=True)
class SourceTarget(JsonDataclass):
    sample_id: str
    source_row_index: int
    answer: str = ""
    judge_type: str = "math"
    judge_payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalReportV2(JsonDataclass):
    schema_version: int = SCHEMA_VERSION
    run_id: str = ""
    run_dir: str = ""
    dataset_name: str = ""
    model_name: str = ""
    run_type: str = ""
    strategy_name: str = ""
    baseline_run_id: str | None = None
    baseline_run_dir: str | None = None
    k: int = 0
    num_items: int = 0
    num_outputs_total: int = 0
    mean_at_k: float = 0.0
    maj_at_k: float = 0.0
    best_score_mean: float = 0.0
    distinct_4: float = 0.0
    solve_none: int = 0
    solve_all: int = 0
    format_error_rollouts: int = 0
    avg_final_visible_tokens: float = 0.0
    total_cost: float = 0.0
    total_cost_unit: str = "tokens"
    billed_tokens_total: int = 0
    billed_input_tokens_total: int = 0
    billed_output_tokens_total: int = 0
    cached_input_tokens_total: int = 0
    request_count_total: int = 0
    final_visible_tokens_total: int = 0
    avg_cost_per_output: float = 0.0
    cost_saving_rate: float | None = None
    usage_summary: dict[str, Any] = field(default_factory=dict)
    cost_summary: dict[str, Any] = field(default_factory=dict)
    comparison_metrics: dict[str, Any] = field(default_factory=dict)
    strategy_metrics: dict[str, Any] = field(default_factory=dict)
    responses_per_item_hist: dict[str, int] = field(default_factory=dict)


def report_path_for_k(run_dir: Path, k: int) -> Path:
    return run_dir / f"eval_report_v2_{k}.json"
