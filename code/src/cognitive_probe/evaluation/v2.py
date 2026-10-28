from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import orjson

from cognitive_probe.evaluation.data_model import (
    DECISION_LOG_FILE,
    FINAL_OUTPUTS_FILE,
    GENERATION_SEGMENTS_FILE,
    PRICE_SNAPSHOT_FILE,
    RUN_OVERVIEW_FILE,
    SOURCE_TARGETS_FILE,
    DecisionRecord,
    EvalReportV2,
    FinalOutput,
    GenerationSegment,
    PriceConfig,
    PriceEntry,
    RunOverview,
    SegmentUsage,
    SourceTarget,
    report_path_for_k,
)
from cognitive_probe.utils.math_utils import (
    extract_answer,
    grade_answer_mathd,
    grade_answer_sympy,
    grade_answer_verl,
    mathd_normalize_answer,
)


def get_diverse_score(sequences: list[str], n: int = 4) -> float:
    distinct_ngrams = set()
    total_ngrams = 0
    for seq in sequences:
        tokens = seq.split()
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            distinct_ngrams.add(ngram)
            total_ngrams += 1
    return len(distinct_ngrams) / total_ngrams if total_ngrams > 0 else 0.0


def as_str_answer(ans: Any) -> str:
    if ans is None:
        return ""
    if isinstance(ans, (int, float)):
        if isinstance(ans, float) and ans.is_integer():
            return str(int(ans))
        return str(ans)
    if isinstance(ans, (list, tuple)):
        return "(" + ",".join(as_str_answer(x) for x in ans) + ")"
    return str(ans)


def _extract_vote_answer(passage: str) -> str | None:
    answer = extract_answer(passage)
    if answer is None:
        return None
    normalized = mathd_normalize_answer(answer)
    candidate = normalized if normalized is not None else answer.strip()
    candidate = candidate.strip()
    return candidate or None


def _grade_extracted_answer(answer: str | None, ground_truth: str) -> bool:
    if not ground_truth or not answer:
        return False
    if "\\boxed" in ground_truth:
        ground_truth = extract_answer(ground_truth)
        if ground_truth is None:
            return False
    return grade_answer_mathd(answer, ground_truth) or grade_answer_sympy(answer, ground_truth)


def _majority_vote_answer(answers: list[str]) -> str | None:
    if not answers:
        return None
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for idx, answer in enumerate(answers):
        counts[answer] += 1
        first_seen.setdefault(answer, idx)
    return min(counts.items(), key=lambda item: (-item[1], first_seen[item[0]]))[0]


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return default


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))


def _dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("wb") as f:
        for row in rows:
            f.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))


def _path_signature(path: Path) -> tuple[str, int, int]:
    resolved = path.resolve()
    stat = resolved.stat()
    return (str(resolved), stat.st_mtime_ns, stat.st_size)


# Cache by resolved path + mtime + size so edits invalidate automatically.
@lru_cache(maxsize=2048)
def _load_json_cached(path_str: str, _mtime_ns: int, _size: int) -> dict[str, Any]:
    payload = orjson.loads(Path(path_str).read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path_str}, got {type(payload).__name__}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    return _load_json_cached(*_path_signature(path))


@lru_cache(maxsize=128)
def _load_jsonl_cached(path_str: str, _mtime_ns: int, _size: int) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for raw_line in Path(path_str).read_bytes().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = orjson.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Expected JSON object line in {path_str}, got {type(item).__name__}")
        rows.append(item)
    return tuple(rows)


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return _load_jsonl_cached(*_path_signature(path))


def _resolve_tokenizer_path(args_data: dict[str, Any]) -> str | None:
    candidates: list[Any] = []
    probe_prune = args_data.get("probe_prune")
    if isinstance(probe_prune, dict):
        candidates.append(probe_prune.get("tokenizer_path"))
    probe = args_data.get("probe")
    if isinstance(probe, dict):
        candidates.append(probe.get("tokenizer_path"))
        candidates.append(probe.get("base_model_path"))
    candidates.append(args_data.get("tokenizer_path"))
    candidates.append(args_data.get("model"))
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


@lru_cache(maxsize=8)
def _load_decode_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    path = tokenizer_path.strip()
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if Path(path).exists():
        kwargs["local_files_only"] = True
    return AutoTokenizer.from_pretrained(path, **kwargs)


def _decode_output_ids(tokenizer_path: str, output_ids: list[int]) -> str:
    tokenizer = _load_decode_tokenizer(tokenizer_path)
    return tokenizer.decode(
        output_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


@lru_cache(maxsize=512)
def _load_run_overview_cached(path_str: str, mtime_ns: int, size: int) -> RunOverview:
    payload = dict(_load_json_cached(path_str, mtime_ns, size))
    if payload.get("reference_run_id") is None and payload.get("baseline_run_id") is not None:
        payload["reference_run_id"] = payload.get("baseline_run_id")
    if payload.get("reference_run_dir") is None and payload.get("baseline_run_dir") is not None:
        payload["reference_run_dir"] = payload.get("baseline_run_dir")
    return RunOverview(**_filter_dataclass_payload(RunOverview, payload))


@lru_cache(maxsize=128)
def _load_generation_segments_cached(path_str: str, mtime_ns: int, size: int) -> tuple[GenerationSegment, ...]:
    segments: list[GenerationSegment] = []
    for row in _load_jsonl_cached(path_str, mtime_ns, size):
        usage_payload = row.get("usage") if isinstance(row.get("usage"), dict) else {}
        usage = SegmentUsage(
            input_tokens=_safe_int(usage_payload.get("input_tokens", row.get("billed_input_tokens", 0))),
            output_tokens=_safe_int(usage_payload.get("output_tokens", row.get("billed_output_tokens", 0))),
            cached_input_tokens=_safe_int(usage_payload.get("cached_input_tokens", 0)),
            request_count=max(1, _safe_int(usage_payload.get("request_count", 1), 1)),
        )
        payload = _filter_dataclass_payload(GenerationSegment, row)
        payload["usage"] = usage
        if not payload.get("executor_name"):
            payload["executor_name"] = "model"
        segments.append(GenerationSegment(**payload))
    return tuple(segments)


@lru_cache(maxsize=128)
def _load_decision_log_cached(path_str: str, mtime_ns: int, size: int) -> tuple[DecisionRecord, ...]:
    return tuple(
        DecisionRecord(**_filter_dataclass_payload(DecisionRecord, row))
        for row in _load_jsonl_cached(path_str, mtime_ns, size)
    )


@lru_cache(maxsize=128)
def _load_final_outputs_cached(path_str: str, mtime_ns: int, size: int) -> tuple[FinalOutput, ...]:
    return tuple(
        FinalOutput(**_filter_dataclass_payload(FinalOutput, row))
        for row in _load_jsonl_cached(path_str, mtime_ns, size)
    )


@lru_cache(maxsize=128)
def _load_source_targets_cached(path_str: str, mtime_ns: int, size: int) -> tuple[SourceTarget, ...]:
    return tuple(
        SourceTarget(**_filter_dataclass_payload(SourceTarget, row))
        for row in _load_jsonl_cached(path_str, mtime_ns, size)
    )


@lru_cache(maxsize=512)
def _load_report_cached(path_str: str, mtime_ns: int, size: int) -> EvalReportV2:
    payload = dict(_load_json_cached(path_str, mtime_ns, size))
    if payload.get("comparison_metrics") is None:
        payload["comparison_metrics"] = {}
    if payload.get("strategy_metrics") is None:
        payload["strategy_metrics"] = {}
    if payload.get("usage_summary") is None:
        payload["usage_summary"] = {}
    if payload.get("cost_summary") is None:
        payload["cost_summary"] = {}
    if payload.get("cost_saving_rate") is not None and not payload["comparison_metrics"].get("cost_saving_rate"):
        payload["comparison_metrics"]["cost_saving_rate"] = payload["cost_saving_rate"]
    return EvalReportV2(**_filter_dataclass_payload(EvalReportV2, payload))


def _filter_dataclass_payload(cls: type[Any], payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    return {key: value for key, value in payload.items() if key in allowed}


def _model_basename(model: str) -> str:
    return os.path.basename(model.rstrip("/")) if model else ""


def _infer_timestamp(args_data: dict[str, Any], run_dir: Path) -> str:
    raw = str(args_data.get("timestamp") or run_dir.name)
    if raw.endswith(".probe"):
        raw = raw[:-6]
    return raw


def _make_run_id(run_dir: Path) -> str:
    return str(run_dir.resolve())


def _extract_dataset_name(args_data: dict[str, Any]) -> str:
    raw = str(args_data.get("dataset_name") or args_data.get("data") or "")
    if raw:
        return Path(raw).stem
    return ""


def _format_strategy_arg_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (int, bool)):
        return str(value)
    if value is None:
        return "None"
    return str(value)


def _build_strategy_label(strategy_name: str, strategy_args: dict[str, Any] | None) -> str:
    if not strategy_args:
        return strategy_name
    parts = [f"{key}={_format_strategy_arg_value(value)}" for key, value in strategy_args.items()]
    return f"{strategy_name} ({', '.join(parts)})"


def _extract_probe_descriptor(args_data: dict[str, Any]) -> tuple[str | None, str | None]:
    probe = args_data.get("probe")
    if not isinstance(probe, dict):
        return (None, None)

    explicit_name = None
    for key in ("probe_method", "prob_method", "method", "strategy", "name", "type", "kind", "score_source"):
        value = probe.get(key)
        if isinstance(value, str) and value.strip():
            explicit_name = value.strip()
            break

    probe_model = probe.get("probe_model")
    probe_model_name = (
        Path(probe_model.strip()).name
        if isinstance(probe_model, str) and probe_model.strip()
        else None
    )

    if explicit_name and probe_model_name:
        return (explicit_name, f"{explicit_name} ({probe_model_name})")
    if explicit_name:
        return (explicit_name, explicit_name)
    if probe_model_name:
        return ("probe_model", f"probe_model ({probe_model_name})")
    return ("probe", "probe")


def _probe_metadata_from_run_dir(run_dir: str | None) -> tuple[str | None, str | None, str | None, str | None]:
    if not run_dir:
        return (None, None, None, None)

    path = Path(run_dir).resolve()
    run_id = str(path)
    baseline_run_dir = None
    probe_name = None
    probe_label = None

    overview_path = path / RUN_OVERVIEW_FILE
    if overview_path.exists():
        try:
            overview = load_run_overview(path)
            run_id = overview.run_id or run_id
            baseline_run_dir = overview.baseline_run_id
            probe_name = overview.probe_name
            probe_label = overview.probe_label
        except Exception:
            pass

    args_path = path / "args.json"
    if args_path.exists():
        try:
            args_data = _load_json(args_path)
        except Exception:
            args_data = None
        if isinstance(args_data, dict):
            fallback_probe_name, fallback_probe_label = _extract_probe_descriptor(args_data)
            if baseline_run_dir is None:
                baseline_run_dir = _extract_reference_run_dir(args_data, "probe")
            if probe_name is None:
                probe_name = fallback_probe_name
            if probe_label is None:
                probe_label = fallback_probe_label

    return (run_id, baseline_run_dir, probe_name, probe_label)


def _extract_strategy_name(args_data: dict[str, Any], run_type: str) -> tuple[str, str]:
    if run_type == "baseline":
        return ("baseline", "baseline")
    if run_type == "probe":
        probe_name, probe_label = _extract_probe_descriptor(args_data)
        return (probe_name or "probe", probe_label or "probe")

    probe_prune = args_data.get("probe_prune")
    if not isinstance(probe_prune, dict):
        return ("prune", "prune")
    prune_config = probe_prune.get("prune_config")
    if not isinstance(prune_config, dict):
        return ("prune", "prune")
    strategy = prune_config.get("strategy")
    if isinstance(strategy, str) and strategy.strip():
        strategy_name = strategy.strip()
        strategy_args = prune_config.get("strategy_args")
        if not isinstance(strategy_args, dict):
            strategy_args = None
        strategy_label = _build_strategy_label(strategy_name, strategy_args)
        postprocess_strategy = prune_config.get("postprocess_strategy")
        if isinstance(postprocess_strategy, str) and postprocess_strategy.strip():
            postprocess_name = postprocess_strategy.strip()
            postprocess_args = prune_config.get("postprocess_strategy_args")
            if not isinstance(postprocess_args, dict):
                postprocess_args = None
            postprocess_label = _build_strategy_label(postprocess_name, postprocess_args)
            return (
                f"{strategy_name}+{postprocess_name}",
                f"{strategy_label} + {postprocess_label}",
            )
        return (strategy_name, strategy_label)
    return ("prune", "prune")


def _extract_run_type(args_data: dict[str, Any]) -> str:
    if isinstance(args_data.get("probe_prune"), dict):
        return "prune"
    if isinstance(args_data.get("probe"), dict):
        return "probe"
    return "baseline"


def _extract_reference_run_dir(args_data: dict[str, Any], run_type: str) -> str | None:
    if run_type == "baseline":
        return None

    if run_type == "probe":
        output_path = args_data.get("output")
        if isinstance(output_path, str) and output_path.strip():
            return str(Path(output_path).resolve())
        return None

    probe_prune = args_data.get("probe_prune")
    if not isinstance(probe_prune, dict):
        return None

    input_dir = probe_prune.get("input_dir")
    if isinstance(input_dir, str) and input_dir.strip():
        return str(Path(input_dir.strip()).resolve())

    input_file = probe_prune.get("input_file")
    if isinstance(input_file, str) and input_file.strip():
        return str(Path(input_file.strip()).resolve().parent)
    return None


def _extract_probe_run_dir(args_data: dict[str, Any], run_type: str, run_dir: Path) -> str | None:
    if run_type == "probe":
        return str(run_dir.resolve())
    if run_type == "prune":
        return _extract_reference_run_dir(args_data, run_type)
    return None


def _extract_baseline_run_dir(args_data: dict[str, Any], run_type: str) -> str | None:
    if run_type == "baseline":
        return None

    if run_type == "probe":
        return _extract_reference_run_dir(args_data, run_type)

    probe_run_dir = _extract_reference_run_dir(args_data, run_type)
    if probe_run_dir is None:
        return None

    _, baseline_run_dir, _, _ = _probe_metadata_from_run_dir(probe_run_dir)
    if baseline_run_dir:
        return baseline_run_dir

    path = Path(probe_run_dir)
    name = path.name
    if name.endswith(".probe"):
        return str(path.with_name(name[:-6]).resolve())
    return None


def build_run_overview(args_data: dict[str, Any], run_dir: Path) -> RunOverview:
    run_type = _extract_run_type(args_data)
    model = str(args_data.get("model") or "")
    timestamp = _infer_timestamp(args_data, run_dir)
    reference_run_dir = _extract_reference_run_dir(args_data, run_type)
    probe_run_dir = _extract_probe_run_dir(args_data, run_type, run_dir)
    probe_run_id = probe_run_dir
    probe_name = None
    probe_label = None
    if run_type == "probe":
        probe_name, probe_label = _extract_probe_descriptor(args_data)
    elif run_type == "prune":
        probe_run_id, _, probe_name, probe_label = _probe_metadata_from_run_dir(probe_run_dir)
    baseline_run_dir = _extract_baseline_run_dir(args_data, run_type)
    strategy_name, strategy_label = _extract_strategy_name(args_data, run_type)
    budget_type = None
    budget_limit = None
    budget_scope = None
    if isinstance(args_data.get("budget"), dict):
        budget = args_data["budget"]
        budget_type = str(budget.get("budget_type") or "") or None
        budget_limit = _safe_float(budget.get("budget_limit"))
        budget_scope = str(budget.get("budget_scope") or "") or None

    return RunOverview(
        run_id=_make_run_id(run_dir),
        timestamp=timestamp,
        run_type=run_type,
        dataset_name=_extract_dataset_name(args_data),
        model=model,
        model_name=_model_basename(model),
        run_dir=str(run_dir.resolve()),
        strategy_name=strategy_name,
        strategy_label=strategy_label,
        reference_run_id=reference_run_dir,
        reference_run_dir=reference_run_dir,
        probe_run_id=probe_run_id,
        probe_run_dir=probe_run_dir,
        probe_name=probe_name,
        probe_label=probe_label,
        baseline_run_id=baseline_run_dir,
        baseline_run_dir=baseline_run_dir,
        budget_type=budget_type,
        budget_limit=budget_limit,
        budget_scope=budget_scope,
    )


def is_v2_run_dir(run_dir: Path) -> bool:
    return (run_dir / RUN_OVERVIEW_FILE).exists()


def load_price_config(path: Path) -> PriceConfig:
    payload = _load_json(path)
    default_payload = payload.get("default") if isinstance(payload.get("default"), dict) else {}
    models_payload = payload.get("models") if isinstance(payload.get("models"), dict) else {}
    models: dict[str, PriceEntry] = {}
    for model_name, entry in models_payload.items():
        if not isinstance(model_name, str) or not isinstance(entry, dict):
            continue
        models[model_name] = PriceEntry(
            input_token_price=_safe_float(entry.get("input_token_price")),
            output_token_price=_safe_float(entry.get("output_token_price")),
            cached_input_token_price=_safe_float(entry.get("cached_input_token_price")),
            request_price=_safe_float(entry.get("request_price")),
            budget_unit=str(entry.get("budget_unit")) if entry.get("budget_unit") is not None else None,
        )
    return PriceConfig(
        currency=str(payload.get("currency")) if payload.get("currency") is not None else None,
        default=PriceEntry(
            input_token_price=_safe_float(default_payload.get("input_token_price")),
            output_token_price=_safe_float(default_payload.get("output_token_price")),
            cached_input_token_price=_safe_float(default_payload.get("cached_input_token_price")),
            request_price=_safe_float(default_payload.get("request_price")),
            budget_unit=str(default_payload.get("budget_unit")) if default_payload.get("budget_unit") is not None else None,
        ),
        models=models,
    )


def resolve_price_entry(price_config: PriceConfig | None, model_name: str) -> PriceEntry:
    if price_config is None:
        return PriceEntry()
    if model_name in price_config.models:
        return price_config.models[model_name]
    basename = _model_basename(model_name)
    if basename in price_config.models:
        return price_config.models[basename]
    return price_config.default


def ensure_price_snapshot(run_dir: Path, overview: RunOverview, price_config: PriceConfig | None, price_source: str | None) -> None:
    if price_config is None:
        return
    snapshot_path = run_dir / PRICE_SNAPSHOT_FILE
    _dump_json(snapshot_path, price_config.to_dict())
    overview.price_source = price_source
    overview.price_snapshot_file = snapshot_path.name
    _dump_json(run_dir / RUN_OVERVIEW_FILE, overview.to_dict())


def _resolve_generation_file(run_dir: Path) -> Path:
    candidates = [
        run_dir / "generation_results.jsonl",
        run_dir / "generation_results.probed.jsonl",
        run_dir / "output.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No generation file found under {run_dir}")


def _resolve_reference_file(run_dir: Path, args_data: dict[str, Any]) -> Path | None:
    probe_prune = args_data.get("probe_prune")
    if not isinstance(probe_prune, dict):
        return None

    input_file = probe_prune.get("input_file")
    if isinstance(input_file, str) and input_file.strip():
        candidate = Path(input_file.strip())
        if candidate.exists():
            return candidate

    input_dir = probe_prune.get("input_dir")
    if isinstance(input_dir, str) and input_dir.strip():
        base = Path(input_dir.strip())
        for name in ("generation_results.probed.jsonl", "generation_results.jsonl", "output.jsonl"):
            candidate = base / name
            if candidate.exists():
                return candidate
    return None


def _reference_pass_map(run_dir: Path, args_data: dict[str, Any]) -> dict[tuple[str, int], bool]:
    reference_file = _resolve_reference_file(run_dir, args_data)
    if reference_file is None:
        return {}

    rows = _load_jsonl(reference_file)
    occ_counter: dict[str, int] = defaultdict(int)
    result: dict[tuple[str, int], bool] = {}
    for row in rows:
        sample_id = str(row.get("example_id"))
        occ_idx = occ_counter[sample_id]
        occ_counter[sample_id] += 1
        gt = as_str_answer(row.get("answer"))
        resp = str(row.get("response") or "")
        result[(sample_id, occ_idx)] = bool(grade_answer_verl(resp, gt))
    return result


def _source_targets_from_generation_rows(rows: tuple[dict[str, Any], ...]) -> list[SourceTarget]:
    targets: list[SourceTarget] = []
    for row_idx, row in enumerate(rows):
        sample_id = str(row.get("example_id"))
        judge_payload = row.get("judge_payload") if isinstance(row.get("judge_payload"), dict) else {}
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        targets.append(
            SourceTarget(
                sample_id=sample_id,
                source_row_index=row_idx,
                answer=as_str_answer(row.get("answer")),
                judge_type=str(row.get("judge_type") or "math"),
                judge_payload=judge_payload,
                metadata=metadata,
            )
        )
    return targets


def _ensure_source_targets_file(
    run_dir: Path,
    *,
    generation_rows: tuple[dict[str, Any], ...] | None = None,
) -> Path:
    path = run_dir / SOURCE_TARGETS_FILE
    if path.exists():
        return path

    rows = generation_rows
    if rows is None:
        rows = _load_jsonl(_resolve_generation_file(run_dir))
    _dump_jsonl(path, [item.to_dict() for item in _source_targets_from_generation_rows(rows)])
    return path


def _load_source_targets(run_dir: Path) -> list[SourceTarget]:
    path = _ensure_source_targets_file(run_dir)
    return list(_load_source_targets_cached(*_path_signature(path)))


def _build_v2_rows(
    run_dir: Path,
    *,
    args_data: dict[str, Any],
    rows: tuple[dict[str, Any], ...],
    strip_segment_probs: bool = False,
    strip_segment_output_ids: bool = False,
) -> tuple[RunOverview, list[GenerationSegment], list[DecisionRecord], list[FinalOutput], list[SourceTarget]]:
    overview = build_run_overview(args_data, run_dir)
    segments: list[GenerationSegment] = []
    decisions: list[DecisionRecord] = []
    final_outputs: list[FinalOutput] = []
    reference_map = _reference_pass_map(run_dir, args_data)
    occ_counter: dict[str, int] = defaultdict(int)

    for row_idx, row in enumerate(rows):
        sample_id = str(row.get("example_id"))
        occ_idx = occ_counter[sample_id]
        occ_counter[sample_id] += 1
        output_id = f"{sample_id}__{occ_idx}"
        segment_id = f"segment__{sample_id}__{occ_idx}"
        output_ids = row.get("output_ids") if isinstance(row.get("output_ids"), list) else []
        response = str(row.get("response") or "")
        billed_output_tokens = len(output_ids) if output_ids else _safe_int(row.get("response_len_tokens"))
        probs = row.get("probs") if isinstance(row.get("probs"), list) else []
        seed = _safe_int(row.get("seed"), 0) if row.get("seed") is not None else None
        source_output_tokens = billed_output_tokens
        was_pruned = False
        prune_reason = None

        prune_meta = row.get("prune_meta") if isinstance(row.get("prune_meta"), dict) else {}
        if overview.run_type == "prune":
            source_output_tokens = _safe_int(prune_meta.get("original_len"), billed_output_tokens)
            was_pruned = str(prune_meta.get("decision") or "") == "prune"
            prune_reason = str(prune_meta.get("reason") or "") or None

        segments.append(
            GenerationSegment(
                segment_id=segment_id,
                sample_id=sample_id,
                parent_segment_id=None,
                executor_name="model",
                model_name=overview.model_name or overview.model,
                seed=seed,
                output_ids=[] if strip_segment_output_ids else [_safe_int(x) for x in output_ids],
                response_text=response,
                probs=[] if strip_segment_probs else [float(x) for x in probs if isinstance(x, (int, float))],
                usage=SegmentUsage(
                    input_tokens=0,
                    output_tokens=billed_output_tokens,
                    cached_input_tokens=0,
                    request_count=1,
                ),
                visible_output_tokens=billed_output_tokens,
                source_row_index=row_idx,
            )
        )
        if was_pruned:
            decisions.append(
                DecisionRecord(
                    decision_type="prune_stop",
                    sample_id=sample_id,
                    output_id=output_id,
                    segment_id=segment_id,
                    original_output_tokens=source_output_tokens,
                    kept_output_tokens=billed_output_tokens,
                    discarded_output_tokens=0,
                    reason=prune_reason,
                    meta={k: v for k, v in prune_meta.items() if k not in {"original_len", "pruned_len", "reason"}},
                )
            )
        decisions.append(
            DecisionRecord(
                decision_type="finalize_output",
                sample_id=sample_id,
                output_id=output_id,
                segment_id=segment_id,
                kept_output_tokens=billed_output_tokens,
                reason=str(row.get("finish_reason") or "") or None,
            )
        )
        final_outputs.append(
            FinalOutput(
                output_id=output_id,
                sample_id=sample_id,
                final_segment_id=segment_id,
                response=response,
                finish_reason=str(row.get("finish_reason") or "") or None,
                k_group=occ_idx + 1,
                source_row_index=row_idx,
                was_pruned=was_pruned,
                would_pass_without_prune=reference_map.get((sample_id, occ_idx)),
                source_output_tokens=source_output_tokens,
            )
        )

    source_targets = _source_targets_from_generation_rows(rows)
    return overview, segments, decisions, final_outputs, source_targets


def materialize_v2_run_from_rows(
    run_dir: Path,
    *,
    args_data: dict[str, Any],
    rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    strip_segment_probs: bool = False,
    strip_segment_output_ids: bool = False,
) -> RunOverview:
    run_dir = run_dir.resolve()
    row_tuple = tuple(rows)
    overview, segments, decisions, final_outputs, source_targets = _build_v2_rows(
        run_dir,
        args_data=args_data,
        rows=row_tuple,
        strip_segment_probs=strip_segment_probs,
        strip_segment_output_ids=strip_segment_output_ids,
    )
    _dump_json(run_dir / RUN_OVERVIEW_FILE, overview.to_dict())
    _dump_jsonl(run_dir / GENERATION_SEGMENTS_FILE, [item.to_dict() for item in segments])
    _dump_jsonl(run_dir / DECISION_LOG_FILE, [item.to_dict() for item in decisions])
    _dump_jsonl(run_dir / FINAL_OUTPUTS_FILE, [item.to_dict() for item in final_outputs])
    _dump_jsonl(run_dir / SOURCE_TARGETS_FILE, [item.to_dict() for item in source_targets])
    return overview


def compact_v2_run_artifacts(
    run_dir: Path,
    *,
    remove_legacy_generation: bool = True,
    strip_segment_probs: bool = True,
    strip_segment_output_ids: bool = False,
) -> dict[str, int]:
    run_dir = run_dir.resolve()
    stats = {
        "legacy_bytes_removed": 0,
        "segment_bytes_saved": 0,
    }

    legacy_path = run_dir / "generation_results.jsonl"
    if remove_legacy_generation and legacy_path.exists():
        stats["legacy_bytes_removed"] = legacy_path.stat().st_size
        legacy_path.unlink()

    segment_path = run_dir / GENERATION_SEGMENTS_FILE
    if not segment_path.exists() or (not strip_segment_probs and not strip_segment_output_ids):
        return stats

    original_rows = list(_load_jsonl(segment_path))
    compacted_rows: list[dict[str, Any]] = []
    changed = False
    for row in original_rows:
        updated = dict(row)
        if strip_segment_probs and "probs" in updated:
            updated.pop("probs", None)
            changed = True
        if strip_segment_output_ids and "output_ids" in updated:
            updated.pop("output_ids", None)
            changed = True
        compacted_rows.append(updated)

    if not changed:
        return stats

    before_size = segment_path.stat().st_size
    _dump_jsonl(segment_path, compacted_rows)
    after_size = segment_path.stat().st_size
    stats["segment_bytes_saved"] = max(0, before_size - after_size)
    return stats


def migrate_legacy_run(run_dir: Path, *, force: bool = False) -> RunOverview:
    run_dir = run_dir.resolve()
    if is_v2_run_dir(run_dir) and not force:
        return load_run_overview(run_dir)

    args_path = run_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing args.json under {run_dir}")

    args_data = _load_json(args_path)
    generation_file = _resolve_generation_file(run_dir)
    rows = _load_jsonl(generation_file)
    return materialize_v2_run_from_rows(run_dir, args_data=args_data, rows=rows)


def _refresh_run_overview(run_dir: Path, overview: RunOverview) -> RunOverview:
    args_path = run_dir / "args.json"
    if not args_path.exists():
        return overview

    args_data = _load_json(args_path)
    expected = build_run_overview(args_data, run_dir)
    updates = {
        "run_type": expected.run_type,
        "strategy_name": expected.strategy_name,
        "strategy_label": expected.strategy_label,
        "reference_run_id": expected.reference_run_id,
        "reference_run_dir": expected.reference_run_dir,
        "probe_run_id": expected.probe_run_id,
        "probe_run_dir": expected.probe_run_dir,
        "probe_name": expected.probe_name,
        "probe_label": expected.probe_label,
        "baseline_run_id": expected.baseline_run_id,
        "baseline_run_dir": expected.baseline_run_dir,
    }
    if any(getattr(overview, key) != value for key, value in updates.items()):
        overview = RunOverview(**{**overview.to_dict(), **updates})
        _dump_json(run_dir / RUN_OVERVIEW_FILE, overview.to_dict())
    return overview


def refresh_run_overview(run_dir: Path, *, force_migrate: bool = False) -> RunOverview:
    run_dir = run_dir.resolve()
    overview = migrate_legacy_run(run_dir, force=force_migrate)
    overview = _refresh_run_overview(run_dir, overview)
    _ensure_source_targets_file(run_dir)
    return overview


def load_run_overview(run_dir: Path) -> RunOverview:
    return _load_run_overview_cached(*_path_signature(run_dir / RUN_OVERVIEW_FILE))


def load_generation_segments(run_dir: Path) -> list[GenerationSegment]:
    return list(_load_generation_segments_cached(*_path_signature(run_dir / GENERATION_SEGMENTS_FILE)))


def load_decision_log(run_dir: Path) -> list[DecisionRecord]:
    return list(_load_decision_log_cached(*_path_signature(run_dir / DECISION_LOG_FILE)))


def load_final_outputs(run_dir: Path) -> list[FinalOutput]:
    return list(_load_final_outputs_cached(*_path_signature(run_dir / FINAL_OUTPUTS_FILE)))


def _repair_pruned_response_texts(run_dir: Path, overview: RunOverview) -> bool:
    if overview.run_type != "prune":
        return False

    args_path = run_dir / "args.json"
    if not args_path.exists():
        return False
    tokenizer_path = _resolve_tokenizer_path(_load_json(args_path))
    if not tokenizer_path:
        return False

    try:
        segments = load_generation_segments(run_dir)
        final_outputs = load_final_outputs(run_dir)
    except Exception:
        return False

    segment_map = {segment.segment_id: segment for segment in segments}
    updated_segments: list[GenerationSegment] = []
    updated_final_outputs: list[FinalOutput] = []
    segments_changed = False
    outputs_changed = False

    for segment in segments:
        response_text = segment.response_text
        if segment.output_ids:
            try:
                decoded = _decode_output_ids(tokenizer_path, segment.output_ids)
            except Exception:
                decoded = response_text
            if decoded != response_text:
                response_text = decoded
                segments_changed = True
        updated_segments.append(
            segment if response_text == segment.response_text else GenerationSegment(
                segment_id=segment.segment_id,
                sample_id=segment.sample_id,
                parent_segment_id=segment.parent_segment_id,
                executor_name=segment.executor_name,
                model_name=segment.model_name,
                seed=segment.seed,
                output_ids=list(segment.output_ids),
                response_text=response_text,
                probs=list(segment.probs),
                usage=segment.usage,
                visible_output_tokens=segment.visible_output_tokens,
                source_row_index=segment.source_row_index,
            )
        )

    for output in final_outputs:
        response = output.response
        final_segment = segment_map.get(output.final_segment_id)
        if final_segment is not None and final_segment.output_ids:
            try:
                decoded = _decode_output_ids(tokenizer_path, final_segment.output_ids)
            except Exception:
                decoded = response
            if decoded != response:
                response = decoded
                outputs_changed = True
        updated_final_outputs.append(
            output if response == output.response else FinalOutput(
                output_id=output.output_id,
                sample_id=output.sample_id,
                final_segment_id=output.final_segment_id,
                response=response,
                finish_reason=output.finish_reason,
                k_group=output.k_group,
                source_row_index=output.source_row_index,
                was_pruned=output.was_pruned,
                would_pass_without_prune=output.would_pass_without_prune,
                source_output_tokens=output.source_output_tokens,
            )
        )

    if segments_changed:
        _dump_jsonl(run_dir / GENERATION_SEGMENTS_FILE, [item.to_dict() for item in updated_segments])
    if outputs_changed:
        _dump_jsonl(run_dir / FINAL_OUTPUTS_FILE, [item.to_dict() for item in updated_final_outputs])
    return segments_changed or outputs_changed


def ensure_v2_run(run_dir: Path, *, force_migrate: bool = False) -> RunOverview:
    overview = refresh_run_overview(run_dir, force_migrate=force_migrate)
    if _repair_pruned_response_texts(run_dir, overview):
        for report_path in run_dir.glob("eval_report_v2_*.json"):
            report_path.unlink(missing_ok=True)
    return overview


def _collect_lineage(segment_id: str, segment_map: dict[str, GenerationSegment]) -> list[GenerationSegment]:
    lineage: list[GenerationSegment] = []
    current = segment_map.get(segment_id)
    while current is not None:
        lineage.append(current)
        current = segment_map.get(current.parent_segment_id) if current.parent_segment_id else None
    lineage.reverse()
    return lineage


def _compute_cost(price_entry: PriceEntry, billed_input_tokens: int, billed_output_tokens: int) -> float:
    request_cost = price_entry.request_price or 0.0
    input_cost = (price_entry.input_token_price or 0.0) * billed_input_tokens
    output_cost = (price_entry.output_token_price or 0.0) * billed_output_tokens
    return request_cost + input_cost + output_cost


def _usage_total(segment: GenerationSegment) -> int:
    return segment.usage.input_tokens + segment.usage.output_tokens


def _usage_bucket() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "request_count": 0,
        "total_tokens": 0,
    }


def _accumulate_usage(bucket: dict[str, int], segment: GenerationSegment) -> None:
    bucket["input_tokens"] += segment.usage.input_tokens
    bucket["output_tokens"] += segment.usage.output_tokens
    bucket["cached_input_tokens"] += segment.usage.cached_input_tokens
    bucket["request_count"] += segment.usage.request_count
    bucket["total_tokens"] += _usage_total(segment)


def _select_outputs_by_k(final_outputs: list[FinalOutput], k: int) -> list[FinalOutput]:
    grouped: dict[str, list[FinalOutput]] = defaultdict(list)
    for item in sorted(final_outputs, key=lambda x: (x.sample_id, x.k_group or 0, x.source_row_index or 0)):
        grouped[item.sample_id].append(item)

    selected: list[FinalOutput] = []
    for sample_outputs in grouped.values():
        if len(sample_outputs) < k:
            raise ValueError(
                f"Sample {sample_outputs[0].sample_id} has only {len(sample_outputs)} outputs, fewer than required k={k}."
            )
        selected.extend(sample_outputs[:k])
    return selected


def _compute_report_object(
    run_dir: Path,
    k: int,
    price_config: PriceConfig | None,
    *,
    include_baseline_compare: bool,
) -> EvalReportV2:
    overview = load_run_overview(run_dir)
    segments = load_generation_segments(run_dir)
    decisions = load_decision_log(run_dir)
    final_outputs = load_final_outputs(run_dir)
    source_targets = _load_source_targets(run_dir)
    selected_outputs = _select_outputs_by_k(final_outputs, k)
    segment_map = {segment.segment_id: segment for segment in segments}
    answer_by_source_row = {item.source_row_index: item.answer for item in source_targets}

    grouped_texts: dict[str, list[str]] = defaultdict(list)
    grouped_scores: dict[str, list[int]] = defaultdict(list)
    grouped_vote_answers: dict[str, list[str]] = defaultdict(list)
    sample_ground_truths: dict[str, str] = {}
    total_outputs = 0
    format_error_rollouts = 0
    selected_sample_ids = {output.sample_id for output in selected_outputs}

    used_segment_ids: set[str] = set()
    shared_segment_hits: Counter[str] = Counter()
    per_output_visible_tokens: list[int] = []

    for output in selected_outputs:
        lineage = _collect_lineage(output.final_segment_id, segment_map)
        lineage_ids = [segment.segment_id for segment in lineage]
        for segment_id in lineage_ids:
            shared_segment_hits[segment_id] += 1
        used_segment_ids.update(lineage_ids)

        response = output.response
        total_outputs += 1
        if "boxed" not in response:
            format_error_rollouts += 1

        sample_id = output.sample_id
        grouped_texts[sample_id].append(response)
        gt = ""
        final_segment = segment_map.get(output.final_segment_id)
        source_row_idx = output.source_row_index
        if source_row_idx is None and final_segment is not None:
            source_row_idx = final_segment.source_row_index
        # Ground truth remains available from legacy generation file only; recover lazily.
        # We keep this path local to evaluation to avoid duplicating large fields into V2 files.
        # The file is guaranteed to exist after migration because V2 is layered on top of existing run dirs.
        if source_row_idx is not None:
            gt = answer_by_source_row.get(source_row_idx, "")
        if gt:
            sample_ground_truths.setdefault(sample_id, gt)
        score = 1 if grade_answer_verl(response, gt) else 0
        grouped_scores[sample_id].append(score)
        if not output.was_pruned:
            vote_answer = _extract_vote_answer(response)
            if vote_answer is not None:
                grouped_vote_answers[sample_id].append(vote_answer)

        visible_tokens = sum(segment.visible_output_tokens for segment in lineage)
        per_output_visible_tokens.append(visible_tokens)

    billed_input_tokens_total = 0
    billed_output_tokens_total = 0
    cached_input_tokens_total = 0
    request_count_total = 0
    final_visible_tokens_total = 0
    total_cost = 0.0
    usage_by_model: dict[str, dict[str, int]] = defaultdict(_usage_bucket)
    usage_by_executor: dict[str, dict[str, int]] = defaultdict(_usage_bucket)
    cost_by_model: dict[str, float] = defaultdict(float)
    cost_by_executor: dict[str, float] = defaultdict(float)
    for segment_id in used_segment_ids:
        segment = segment_map[segment_id]
        billed_input_tokens_total += segment.usage.input_tokens
        billed_output_tokens_total += segment.usage.output_tokens
        cached_input_tokens_total += segment.usage.cached_input_tokens
        request_count_total += segment.usage.request_count
        final_visible_tokens_total += segment.visible_output_tokens
        price_entry = resolve_price_entry(price_config, segment.model_name)
        segment_cost = _compute_cost(price_entry, segment.usage.input_tokens, segment.usage.output_tokens)
        total_cost += segment_cost
        _accumulate_usage(usage_by_model[segment.model_name or "unknown"], segment)
        _accumulate_usage(usage_by_executor[segment.executor_name or "unknown"], segment)
        cost_by_model[segment.model_name or "unknown"] += segment_cost
        cost_by_executor[segment.executor_name or "unknown"] += segment_cost

    shared_prefix_tokens_total = sum(
        segment_map[segment_id].visible_output_tokens
        for segment_id, hit_count in shared_segment_hits.items()
        if hit_count > 1 and segment_id in segment_map
    )
    discarded_tokens_total = sum(
        _safe_int(decision.discarded_output_tokens)
        for decision in decisions
        if decision.sample_id in selected_sample_ids
    )
    rollback_count = sum(1 for decision in decisions if decision.decision_type == "rollback")
    model_switch_count = sum(1 for decision in decisions if decision.decision_type == "model_switch")
    budget_stop_count = sum(1 for decision in decisions if decision.decision_type == "budget_stop")
    prune_stop_count = sum(1 for decision in decisions if decision.decision_type == "prune_stop")

    diverse_scores: list[float] = []
    avg_scores: list[float] = []
    maj_scores: list[float] = []
    best_scores: list[float] = []
    solve_none = 0
    solve_all = 0
    for sample_id in sorted(grouped_texts.keys()):
        responses = grouped_texts[sample_id]
        scores = grouped_scores[sample_id]
        maj_answer = _majority_vote_answer(grouped_vote_answers.get(sample_id, []))
        maj_scores.append(
            1.0 if _grade_extracted_answer(maj_answer, sample_ground_truths.get(sample_id, "")) else 0.0
        )
        diverse_scores.append(get_diverse_score(responses))
        avg = sum(scores) / len(scores) if scores else 0.0
        avg_scores.append(avg)
        best_scores.append(max(scores) if scores else 0.0)
        if avg == 0:
            solve_none += 1
        elif avg == 1:
            solve_all += 1

    billed_tokens_total = billed_input_tokens_total + billed_output_tokens_total
    if price_config is None:
        total_cost = float(billed_tokens_total)
        total_cost_unit = "tokens"
        cost_by_model = {key: float(value["total_tokens"]) for key, value in usage_by_model.items()}
        cost_by_executor = {key: float(value["total_tokens"]) for key, value in usage_by_executor.items()}
    else:
        total_cost_unit = price_config.currency or "cost"

    avg_cost_per_output = total_cost / total_outputs if total_outputs > 0 else 0.0
    avg_final_visible_tokens = (
        sum(per_output_visible_tokens) / len(per_output_visible_tokens) if per_output_visible_tokens else 0.0
    )
    usage_summary = {
        "totals": {
            "input_tokens": billed_input_tokens_total,
            "output_tokens": billed_output_tokens_total,
            "cached_input_tokens": cached_input_tokens_total,
            "request_count": request_count_total,
            "total_tokens": billed_tokens_total,
            "final_visible_tokens": final_visible_tokens_total,
        },
        "by_model": {key: value for key, value in sorted(usage_by_model.items())},
        "by_executor": {key: value for key, value in sorted(usage_by_executor.items())},
    }
    cost_summary = {
        "unit": total_cost_unit,
        "by_model": {key: value for key, value in sorted(cost_by_model.items())},
        "by_executor": {key: value for key, value in sorted(cost_by_executor.items())},
    }
    comparison_metrics: dict[str, Any] = {}
    strategy_metrics: dict[str, Any] = {
        "strategy_type": overview.run_type,
        "event_counts": {
            "prune_stop": prune_stop_count,
            "rollback": rollback_count,
            "model_switch": model_switch_count,
            "budget_stop": budget_stop_count,
        },
    }
    if shared_prefix_tokens_total > 0:
        strategy_metrics["tree"] = {
            "shared_prefix_tokens_total": shared_prefix_tokens_total,
        }
    if discarded_tokens_total > 0 or rollback_count > 0:
        strategy_metrics["rollback"] = {
            "discarded_tokens_total": discarded_tokens_total,
            "rollback_count": rollback_count,
        }
    if model_switch_count > 0:
        strategy_metrics["routing"] = {
            "model_switch_count": model_switch_count,
        }
    if budget_stop_count > 0:
        strategy_metrics["budget"] = {
            "budget_stop_count": budget_stop_count,
        }
    if overview.run_type == "prune":
        pruned_rollouts = sum(1 for output in selected_outputs if output.was_pruned)
        pruned_would_fail_count = sum(
            1 for output in selected_outputs if output.was_pruned and output.would_pass_without_prune is False
        )
        pruned_would_pass_count = sum(
            1 for output in selected_outputs if output.was_pruned and output.would_pass_without_prune is True
        )
        would_pass_total = sum(1 for output in selected_outputs if output.would_pass_without_prune is True)
        prune_rate = (pruned_rollouts / total_outputs) if total_outputs > 0 else 0.0
        pruned_labeled_total = pruned_would_fail_count + pruned_would_pass_count
        pruning_precision = (
            pruned_would_fail_count / pruned_labeled_total if pruned_labeled_total > 0 else None
        )
        false_positive_rate = (
            pruned_would_pass_count / would_pass_total if would_pass_total > 0 else None
        )
        strategy_metrics["prune"] = {
            "reference_run_id": overview.reference_run_id,
            "baseline_run_id": overview.baseline_run_id,
            "pruned_outputs": pruned_rollouts,
            "prune_rate": prune_rate,
            "pruned_would_fail_count": pruned_would_fail_count,
            "pruned_would_pass_count": pruned_would_pass_count,
            "would_pass_total": would_pass_total,
            "pruning_precision": pruning_precision,
            "false_positive_rate": false_positive_rate,
        }

    report = EvalReportV2(
        run_id=overview.run_id,
        run_dir=overview.run_dir,
        dataset_name=overview.dataset_name,
        model_name=overview.model_name,
        run_type=overview.run_type,
        strategy_name=overview.strategy_name,
        baseline_run_id=overview.baseline_run_id,
        baseline_run_dir=overview.baseline_run_dir,
        k=k,
        num_items=len(grouped_texts),
        num_outputs_total=total_outputs,
        mean_at_k=sum(avg_scores) / len(avg_scores) if avg_scores else 0.0,
        maj_at_k=sum(maj_scores) / len(maj_scores) if maj_scores else 0.0,
        best_score_mean=sum(best_scores) / len(best_scores) if best_scores else 0.0,
        distinct_4=sum(diverse_scores) / len(diverse_scores) if diverse_scores else 0.0,
        solve_none=solve_none,
        solve_all=solve_all,
        format_error_rollouts=format_error_rollouts,
        avg_final_visible_tokens=avg_final_visible_tokens,
        total_cost=total_cost,
        total_cost_unit=total_cost_unit,
        billed_tokens_total=billed_tokens_total,
        billed_input_tokens_total=billed_input_tokens_total,
        billed_output_tokens_total=billed_output_tokens_total,
        cached_input_tokens_total=cached_input_tokens_total,
        request_count_total=request_count_total,
        final_visible_tokens_total=final_visible_tokens_total,
        avg_cost_per_output=avg_cost_per_output,
        usage_summary=usage_summary,
        cost_summary=cost_summary,
        comparison_metrics=comparison_metrics,
        strategy_metrics=strategy_metrics,
        responses_per_item_hist={str(k1): int(v1) for k1, v1 in Counter(len(v) for v in grouped_texts.values()).items()},
    )

    if include_baseline_compare and overview.baseline_run_dir:
        baseline_dir = Path(overview.baseline_run_dir)
        if baseline_dir.exists() and baseline_dir.resolve() != run_dir.resolve():
            ensure_v2_run(baseline_dir)
            baseline_report = _compute_report_object(
                baseline_dir,
                k,
                price_config,
                include_baseline_compare=False,
            )
            baseline_total = baseline_report.total_cost
            if baseline_total > 0:
                report.cost_saving_rate = (baseline_total - report.total_cost) / baseline_total
            report.comparison_metrics = {
                "baseline_run_id": baseline_report.run_id,
                "baseline_total_cost": baseline_report.total_cost,
                "baseline_avg_cost_per_output": baseline_report.avg_cost_per_output,
                "baseline_mean_at_k": baseline_report.mean_at_k,
                "baseline_maj_at_k": baseline_report.maj_at_k,
                "total_cost_delta": report.total_cost - baseline_report.total_cost,
                "avg_cost_per_output_delta": report.avg_cost_per_output - baseline_report.avg_cost_per_output,
                "mean_at_k_delta": report.mean_at_k - baseline_report.mean_at_k,
                "maj_at_k_delta": report.maj_at_k - baseline_report.maj_at_k,
                "cost_saving_rate": report.cost_saving_rate,
            }

    return report


def compute_eval_report(
    run_dir: Path,
    k: int,
    *,
    price_config: PriceConfig | None = None,
    price_source: str | None = None,
) -> EvalReportV2:
    run_dir = run_dir.resolve()
    overview = ensure_v2_run(run_dir)
    ensure_price_snapshot(run_dir, overview, price_config, price_source)
    report = _compute_report_object(run_dir, k, price_config, include_baseline_compare=True)
    _dump_json(report_path_for_k(run_dir, k), report.to_dict())
    return report


def scan_run_dirs(root_dir: Path, recursive: bool = True) -> list[Path]:
    if not root_dir.exists():
        return []
    iterator = root_dir.rglob("*") if recursive else root_dir.iterdir()
    run_dirs: list[Path] = []
    for path in iterator:
        if not path.is_dir():
            continue
        if is_v2_run_dir(path):
            run_dirs.append(path)
            continue
        if (path / "args.json").exists() and (
            (path / "generation_results.jsonl").exists() or (path / "output.jsonl").exists()
            or (path / "generation_results.probed.jsonl").exists()
        ):
            run_dirs.append(path)
    return sorted(set(run_dirs))


def load_report(run_dir: Path, k: int) -> EvalReportV2 | None:
    path = report_path_for_k(run_dir, k)
    if not path.exists():
        return None
    return _load_report_cached(*_path_signature(path))


def report_choices(run_dir: Path) -> list[int]:
    choices: list[int] = []
    for path in run_dir.glob("eval_report_v2_*.json"):
        suffix = path.stem.rsplit("_", 1)[-1]
        try:
            choices.append(int(suffix))
        except ValueError:
            continue
    return sorted(set(choices))


def summarize_run_for_ui(run_dir: Path) -> dict[str, Any]:
    overview = refresh_run_overview(run_dir)
    reports = [load_report(run_dir, k) for k in report_choices(run_dir)]
    reports = [report for report in reports if report is not None]
    return {
        "run_id": overview.run_id,
        "dataset_name": overview.dataset_name,
        "model_name": overview.model_name,
        "run_type": overview.run_type,
        "strategy_name": overview.strategy_name,
        "strategy_label": overview.strategy_label,
        "timestamp": overview.timestamp,
        "probe_run_id": overview.probe_run_id,
        "probe_label": overview.probe_label,
        "baseline_run_id": overview.baseline_run_id,
        "run_dir": overview.run_dir,
        "reports": [report.to_dict() for report in reports],
    }
