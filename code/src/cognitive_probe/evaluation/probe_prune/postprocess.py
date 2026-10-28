from __future__ import annotations

import asyncio
import copy
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import orjson
from tqdm.auto import tqdm


@dataclass(frozen=True)
class RegenerationTrace:
    row_index: int
    regenerated_row: dict[str, Any]
    regenerated_seed: int | None


@dataclass(frozen=True)
class RegenerationConfig:
    api_url: str
    model_name: str
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int | None
    seed_offset: int
    timeout_s: float
    concurrency_limit: int


def available_postprocess_strategies() -> list[str]:
    return ["regenerate_if_pruned"]


def validate_postprocess_strategy(
    *,
    strategy_name: str,
    strategy_args: dict[str, Any],
) -> None:
    if strategy_name != "regenerate_if_pruned":
        raise ValueError(
            f"Unsupported postprocess strategy: {strategy_name}. "
            f"Available postprocess strategies: {available_postprocess_strategies()}"
        )

    allowed_keys = {
        "api_url",
        "model_name",
        "seed_offset",
        "temperature",
        "top_p",
        "top_k",
        "max_new_tokens",
        "omit_max_new_tokens",
        "timeout_s",
        "concurrency_limit",
    }
    unknown = sorted(set(strategy_args.keys()) - allowed_keys)
    if unknown:
        raise ValueError(
            f"Unsupported args for postprocess strategy={strategy_name}: {unknown}. "
            f"Allowed args: {sorted(allowed_keys)}"
        )


def build_regeneration_config(
    *,
    original_args: dict[str, Any],
    strategy_args: dict[str, Any],
) -> RegenerationConfig:
    api_url = str(strategy_args.get("api_url") or "").strip().rstrip("/")
    if not api_url:
        raise ValueError("regenerate_if_pruned requires explicit api_url")

    model_name = str(strategy_args.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("regenerate_if_pruned requires explicit model_name")

    try:
        temperature = float(
            strategy_args.get("temperature", original_args.get("temperature", 1.0))
        )
        top_p = float(strategy_args.get("top_p", original_args.get("top_p", 1.0)))
        top_k = int(strategy_args.get("top_k", original_args.get("top_k", 20)))
        omit_max_new_tokens = bool(strategy_args.get("omit_max_new_tokens", False))
        if omit_max_new_tokens:
            max_new_tokens = None
        else:
            max_new_tokens = int(
                strategy_args.get("max_new_tokens", original_args.get("max_length", 1024))
            )
        seed_offset = int(strategy_args.get("seed_offset", 1000003))
        timeout_s = float(strategy_args.get("timeout_s", 300.0))
        concurrency_limit = int(strategy_args.get("concurrency_limit", 16))
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid regenerate_if_pruned args: {e}") from e

    if max_new_tokens is not None and max_new_tokens <= 0:
        raise ValueError("regenerate_if_pruned requires max_new_tokens > 0")
    if timeout_s <= 0:
        raise ValueError("regenerate_if_pruned requires timeout_s > 0")
    if concurrency_limit <= 0:
        raise ValueError("regenerate_if_pruned requires concurrency_limit > 0")

    return RegenerationConfig(
        api_url=api_url,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        seed_offset=seed_offset,
        timeout_s=timeout_s,
        concurrency_limit=concurrency_limit,
    )


def _as_output_items(resp_json: Any) -> list[dict[str, Any]]:
    if isinstance(resp_json, list):
        return [item for item in resp_json if isinstance(item, dict)]
    if isinstance(resp_json, dict):
        if "meta_info" in resp_json:
            return [resp_json]
        text_list = resp_json.get("text")
        out_ids_list = resp_json.get("output_ids")
        meta_list = resp_json.get("meta_info")
        if (
            isinstance(text_list, list)
            and isinstance(out_ids_list, list)
            and isinstance(meta_list, list)
        ):
            items: list[dict[str, Any]] = []
            for idx in range(len(out_ids_list)):
                items.append(
                    {
                        "text": text_list[idx] if idx < len(text_list) else "",
                        "output_ids": out_ids_list[idx],
                        "meta_info": meta_list[idx] if idx < len(meta_list) else {},
                    }
                )
            return items
    raise ValueError("Unexpected /generate response schema")


def _build_generation_payload(
    *,
    record: dict[str, Any],
    config: RegenerationConfig,
    regen_seed: int,
    rid: str,
) -> tuple[dict[str, Any], list[int] | None, str | None]:
    sampling_params: dict[str, Any] = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "sampling_seed": regen_seed,
    }
    if config.max_new_tokens is not None:
        sampling_params["max_new_tokens"] = config.max_new_tokens

    payload: dict[str, Any] = {
        "rid": rid,
        "sampling_params": sampling_params,
        "stream": False,
    }
    input_ids = record.get("input_ids")
    if isinstance(input_ids, list):
        resolved_input_ids = [int(token_id) for token_id in input_ids]
        payload["input_ids"] = resolved_input_ids
        return payload, resolved_input_ids, None

    input_text = record.get("input_text")
    if isinstance(input_text, str) and input_text:
        payload["text"] = input_text
        return payload, None, input_text

    raise ValueError(
        "regenerate_if_pruned requires input_ids or input_text in each record"
    )


def _derive_regen_seed(
    *, record: dict[str, Any], row_index: int, seed_offset: int
) -> int:
    raw_seed = record.get("seed")
    if raw_seed is None:
        return seed_offset + row_index
    try:
        return int(raw_seed) + seed_offset
    except (TypeError, ValueError):
        return seed_offset + row_index


def _load_regeneration_checkpoint(
    checkpoint_path: Path,
) -> dict[int, RegenerationTrace]:
    if not checkpoint_path.exists():
        return {}

    traces: dict[int, RegenerationTrace] = {}
    invalid_lines = 0
    for raw_line in checkpoint_path.read_bytes().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = orjson.loads(line)
        except orjson.JSONDecodeError:
            invalid_lines += 1
            continue
        if not isinstance(payload, dict):
            invalid_lines += 1
            continue
        try:
            trace = RegenerationTrace(
                row_index=int(payload["row_index"]),
                regenerated_row=dict(payload["regenerated_row"]),
                regenerated_seed=(
                    int(payload["regenerated_seed"])
                    if payload.get("regenerated_seed") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError):
            invalid_lines += 1
            continue
        traces[trace.row_index] = trace

    if invalid_lines:
        print(
            f"Warning: ignored {invalid_lines} invalid regeneration checkpoint line(s) in {checkpoint_path}"
        )
    return traces


def _append_regeneration_checkpoint(
    *,
    handle: Any,
    trace: RegenerationTrace,
    completed_count: int,
    fsync_interval: int,
) -> None:
    handle.write(orjson.dumps(asdict(trace), option=orjson.OPT_APPEND_NEWLINE))
    handle.flush()
    if fsync_interval > 0 and completed_count % fsync_interval == 0:
        os.fsync(handle.fileno())


async def _generate_replacements_async(
    *,
    requests: list[tuple[int, dict[str, Any]]],
    config: RegenerationConfig,
    completed_traces: dict[int, RegenerationTrace] | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_fsync_interval: int = 1,
) -> dict[int, RegenerationTrace]:
    existing_traces = dict(completed_traces or {})
    pending_requests = [
        request for request in requests if int(request[0]) not in existing_traces
    ]
    if not pending_requests:
        return existing_traces

    semaphore = asyncio.Semaphore(config.concurrency_limit)
    request_prefix = f"regen_{os.getpid()}_{time.time_ns():x}"
    async with httpx.AsyncClient(trust_env=False) as client:

        async def _run_one(row_index: int, record: dict[str, Any]) -> RegenerationTrace:
            regen_seed = _derive_regen_seed(
                record=record,
                row_index=row_index,
                seed_offset=config.seed_offset,
            )
            rid = f"{request_prefix}_row_{row_index}"
            payload, input_ids, input_text = _build_generation_payload(
                record=record,
                config=config,
                regen_seed=regen_seed,
                rid=rid,
            )
            async with semaphore:
                resp = await client.post(
                    f"{config.api_url}/generate",
                    json=payload,
                    timeout=config.timeout_s,
                )
            if resp.status_code != 200:
                raise RuntimeError(
                    "SGL /generate failed during regenerate_if_pruned. "
                    f"status={resp.status_code}, body={resp.text[:1000]}"
                )
            items = _as_output_items(resp.json())
            if len(items) != 1:
                raise RuntimeError(
                    f"Expected single regeneration output, got {len(items)}"
                )
            output = items[0]

            regenerated_row = copy.deepcopy(record)
            if input_ids is not None:
                regenerated_row["input_ids"] = input_ids
            if input_text is not None:
                regenerated_row["input_text"] = input_text
            regenerated_row["seed"] = regen_seed
            regenerated_row["response"] = str(output.get("text") or "")
            regenerated_output_ids = output.get("output_ids")
            if not isinstance(regenerated_output_ids, list):
                raise ValueError(
                    f"Missing output_ids in regeneration result for row {row_index}"
                )
            regenerated_row["output_ids"] = [
                int(token_id) for token_id in regenerated_output_ids
            ]
            regenerated_row["response_len_tokens"] = len(regenerated_row["output_ids"])
            regenerated_row["probs"] = []
            finish_reason = (
                output.get("meta_info", {}).get("finish_reason", {}).get("type")
            )
            regenerated_row["finish_reason"] = (
                str(finish_reason) if finish_reason is not None else None
            )
            return RegenerationTrace(
                row_index=row_index,
                regenerated_row=regenerated_row,
                regenerated_seed=regen_seed,
            )

        tasks = [_run_one(row_index, record) for row_index, record in pending_requests]
        traces = dict(existing_traces)
        checkpoint_handle = (
            checkpoint_path.open("ab") if checkpoint_path is not None else None
        )
        try:
            with tqdm(
                total=len(requests),
                initial=len(existing_traces),
                desc="Regenerating",
                unit="sample",
            ) as pbar:
                completed_count = len(existing_traces)
                for future in asyncio.as_completed(tasks):
                    trace = await future
                    traces[trace.row_index] = trace
                    completed_count += 1
                    if checkpoint_handle is not None:
                        _append_regeneration_checkpoint(
                            handle=checkpoint_handle,
                            trace=trace,
                            completed_count=completed_count,
                            fsync_interval=checkpoint_fsync_interval,
                        )
                    pbar.update(1)
        finally:
            if checkpoint_handle is not None:
                checkpoint_handle.flush()
                if checkpoint_fsync_interval > 0:
                    os.fsync(checkpoint_handle.fileno())
                checkpoint_handle.close()
    return traces


def regenerate_pruned_records(
    *,
    records: list[dict[str, Any]],
    pruned_row_indices: list[int],
    original_args: dict[str, Any],
    strategy_args: dict[str, Any],
    checkpoint_path: Path | None = None,
    checkpoint_fsync_interval: int = 1,
) -> tuple[RegenerationConfig, dict[int, RegenerationTrace]]:
    config = build_regeneration_config(
        original_args=original_args,
        strategy_args=strategy_args,
    )
    if not pruned_row_indices:
        return config, {}

    requests = [(row_index, records[row_index]) for row_index in pruned_row_indices]
    completed_traces: dict[int, RegenerationTrace] = {}
    if checkpoint_path is not None:
        completed_traces = _load_regeneration_checkpoint(checkpoint_path)
        completed_traces = {
            row_index: trace
            for row_index, trace in completed_traces.items()
            if row_index in set(pruned_row_indices)
        }
        if completed_traces:
            print(
                "Resuming regenerate_if_pruned from checkpoint: "
                f"{len(completed_traces)} / {len(pruned_row_indices)} records complete"
            )
    replacements = asyncio.run(
        _generate_replacements_async(
            requests=requests,
            config=config,
            completed_traces=completed_traces,
            checkpoint_path=checkpoint_path,
            checkpoint_fsync_interval=checkpoint_fsync_interval,
        )
    )
    missing = sorted(set(pruned_row_indices) - set(replacements.keys()))
    if missing:
        raise RuntimeError(f"Missing regenerated rows for indices: {missing}")
    return config, replacements
