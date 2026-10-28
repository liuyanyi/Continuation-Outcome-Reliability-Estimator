#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from transformers import AutoTokenizer

from cognitive_probe.evaluation.io import dump_json, load_jsonl, write_jsonl
from cognitive_probe.evaluation.probe_prune.common import (
    decode_response,
    prepare_output_dir,
    resolve_input_file,
)
from cognitive_probe.evaluation.refrain import (
    DIRECT_ANSWER_PROMPT_STYLE_DEFAULT,
    DIRECT_ANSWER_PROMPT_STYLES,
    build_direct_answer_prompt,
)
from cognitive_probe.evaluation.v2 import migrate_legacy_run


@dataclass(frozen=True)
class ContinuationConfig:
    api_url: str
    model_name: str
    direct_answer_prompt: str
    direct_answer_prompt_style: str
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int | None
    seed_offset: int
    timeout_s: float
    concurrency_limit: int
    min_keep_tokens: int


@dataclass(frozen=True)
class ContinuationTrace:
    row_index: int
    continued_row: dict[str, Any]
    continued_seed: int | None


def _load_continuation_checkpoint(
    checkpoint_path: Path,
) -> dict[int, ContinuationTrace]:
    if not checkpoint_path.exists():
        return {}

    traces: dict[int, ContinuationTrace] = {}
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
            trace = ContinuationTrace(
                row_index=int(payload["row_index"]),
                continued_row=dict(payload["continued_row"]),
                continued_seed=(
                    int(payload["continued_seed"])
                    if payload.get("continued_seed") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError):
            invalid_lines += 1
            continue
        traces[trace.row_index] = trace

    if invalid_lines:
        print(
            f"Warning: ignored {invalid_lines} invalid REFRAIN continuation checkpoint line(s) in {checkpoint_path}"
        )
    return traces


def _append_continuation_checkpoint(
    *,
    handle: Any,
    trace: ContinuationTrace,
    completed_count: int,
    fsync_interval: int,
) -> None:
    handle.write(orjson.dumps(asdict(trace), option=orjson.OPT_APPEND_NEWLINE))
    handle.flush()
    if fsync_interval > 0 and completed_count % fsync_interval == 0:
        os.fsync(handle.fileno())


def _load_original_args(input_dir: Path) -> dict[str, Any]:
    src_args_path = input_dir / "args.json"
    if not src_args_path.exists():
        return {}

    import orjson

    payload = orjson.loads(src_args_path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {src_args_path}")
    return payload


def _resolve_tokenizer_path(
    cli_tokenizer_path: str,
    original_args: dict[str, Any],
) -> str:
    explicit = cli_tokenizer_path.strip()
    if explicit:
        return explicit

    candidates: list[Any] = []
    api_generation = original_args.get("api_generation")
    if isinstance(api_generation, dict):
        candidates.append(api_generation.get("tokenizer_path"))
    probe_prune = original_args.get("probe_prune")
    if isinstance(probe_prune, dict):
        candidates.append(probe_prune.get("tokenizer_path"))
    probe = original_args.get("probe")
    if isinstance(probe, dict):
        candidates.append(probe.get("tokenizer_path"))
        candidates.append(probe.get("base_model_path"))
    candidates.append(original_args.get("tokenizer_path"))
    candidates.append(original_args.get("model"))

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _resolve_api_url(cli_api_url: str, original_args: dict[str, Any]) -> str:
    explicit = cli_api_url.strip().rstrip("/")
    if explicit:
        return explicit

    api_generation = original_args.get("api_generation")
    if isinstance(api_generation, dict):
        value = api_generation.get("api_url")
        if isinstance(value, str) and value.strip():
            return value.strip().rstrip("/")
    raise ValueError(
        "Unable to resolve generation api_url. Pass --api-url explicitly or ensure "
        "the source run contains api_generation.api_url in args.json."
    )


def _resolve_model_name(cli_model_name: str, original_args: dict[str, Any]) -> str:
    explicit = cli_model_name.strip()
    if explicit:
        return explicit

    api_generation = original_args.get("api_generation")
    if isinstance(api_generation, dict):
        value = api_generation.get("model_name")
        if isinstance(value, str) and value.strip():
            return value.strip()
    model = original_args.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    raise ValueError(
        "Unable to resolve model name. Pass --model-name explicitly or ensure the "
        "source run contains api_generation.model_name in args.json."
    )


def _build_config(args: argparse.Namespace, original_args: dict[str, Any]) -> ContinuationConfig:
    api_url = _resolve_api_url(args.api_url, original_args)
    model_name = _resolve_model_name(args.model_name, original_args)

    try:
        temperature = float(
            args.temperature
            if args.temperature is not None
            else original_args.get("temperature", 1.0)
        )
        top_p = float(
            args.top_p if args.top_p is not None else original_args.get("top_p", 1.0)
        )
        top_k = int(
            args.top_k if args.top_k is not None else original_args.get("top_k", 20)
        )
        if args.omit_max_new_tokens:
            max_new_tokens = None
        elif args.max_new_tokens is not None:
            max_new_tokens = int(args.max_new_tokens)
        else:
            max_new_tokens = int(original_args.get("max_length", 1024))
        seed_offset = int(args.seed_offset)
        timeout_s = float(args.timeout_s)
        concurrency_limit = int(args.concurrency_limit)
        min_keep_tokens = int(args.min_keep_tokens)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid REFRAIN postprocess args: {exc}") from exc

    if max_new_tokens is not None and max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0 when provided")
    if timeout_s <= 0:
        raise ValueError("--timeout-s must be > 0")
    if concurrency_limit <= 0:
        raise ValueError("--concurrency-limit must be > 0")
    if min_keep_tokens < 0:
        raise ValueError("--min-keep-tokens must be >= 0")

    return ContinuationConfig(
        api_url=api_url,
        model_name=model_name,
        direct_answer_prompt=build_direct_answer_prompt(args.direct_answer_prompt_style),
        direct_answer_prompt_style=args.direct_answer_prompt_style,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        seed_offset=seed_offset,
        timeout_s=timeout_s,
        concurrency_limit=concurrency_limit,
        min_keep_tokens=min_keep_tokens,
    )


def _derive_seed(*, record: dict[str, Any], row_index: int, seed_offset: int) -> int:
    raw_seed = record.get("seed")
    if raw_seed is None:
        return seed_offset + row_index
    try:
        return int(raw_seed) + seed_offset
    except (TypeError, ValueError):
        return seed_offset + row_index


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


def _build_augmented_prompt(
    *,
    record: dict[str, Any],
    stop_index: int,
    stop_char_end: int | None,
    tokenizer: Any | None,
    direct_answer_prompt: str,
) -> tuple[list[int] | None, str | None, str]:
    output_ids_raw = record.get("output_ids")
    if not isinstance(output_ids_raw, list):
        raise ValueError("Each record must contain list[int] field: output_ids")
    output_ids = [int(token_id) for token_id in output_ids_raw]
    kept_output_ids = output_ids[:stop_index]

    prefix_text = ""
    if tokenizer is not None:
        prefix_text = decode_response(tokenizer, kept_output_ids)
    else:
        response_text = str(record.get("response") or "")
        if stop_char_end is None:
            raise ValueError(
                "REFRAIN postprocess requires tokenizer or stored stop_char_end to "
                "reconstruct the prefix."
            )
        prefix_text = response_text[:stop_char_end]

    input_ids_raw = record.get("input_ids")
    if tokenizer is not None and isinstance(input_ids_raw, list):
        control_ids = tokenizer.encode(
            direct_answer_prompt,
            add_special_tokens=False,
        )
        augmented_input_ids = [int(token_id) for token_id in input_ids_raw]
        augmented_input_ids.extend(kept_output_ids)
        augmented_input_ids.extend(int(token_id) for token_id in control_ids)
        return augmented_input_ids, None, prefix_text

    input_text = record.get("input_text")
    if isinstance(input_text, str) and input_text:
        return None, input_text + prefix_text + direct_answer_prompt, prefix_text

    if tokenizer is not None and isinstance(input_ids_raw, list):
        prompt_text = tokenizer.decode(
            [int(token_id) for token_id in input_ids_raw],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return None, prompt_text + prefix_text + direct_answer_prompt, prefix_text

    raise ValueError(
        "REFRAIN postprocess requires input_ids or input_text in each record"
    )


async def _continue_async(
    *,
    requests: list[tuple[int, dict[str, Any], int, int | None]],
    config: ContinuationConfig,
    tokenizer: Any | None,
    completed_traces: dict[int, ContinuationTrace],
    checkpoint_path: Path | None,
    checkpoint_fsync_interval: int,
) -> dict[int, ContinuationTrace]:
    pending_requests = [
        request for request in requests if int(request[0]) not in completed_traces
    ]
    if not pending_requests:
        return dict(completed_traces)

    semaphore = asyncio.Semaphore(config.concurrency_limit)
    request_prefix = f"refrain_{os.getpid()}_{time.time_ns():x}"
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(config.timeout_s), trust_env=False
    ) as client:

        async def _run_one(
            row_index: int,
            record: dict[str, Any],
            stop_index: int,
            stop_char_end: int | None,
        ) -> ContinuationTrace:
            continued_seed = _derive_seed(
                record=record,
                row_index=row_index,
                seed_offset=config.seed_offset,
            )
            augmented_input_ids, augmented_text, prefix_text = _build_augmented_prompt(
                record=record,
                stop_index=stop_index,
                stop_char_end=stop_char_end,
                tokenizer=tokenizer,
                direct_answer_prompt=config.direct_answer_prompt,
            )
            sampling_params: dict[str, Any] = {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "sampling_seed": continued_seed,
            }
            if config.max_new_tokens is not None:
                sampling_params["max_new_tokens"] = config.max_new_tokens

            payload: dict[str, Any] = {
                "rid": f"{request_prefix}_row_{row_index}",
                "sampling_params": sampling_params,
                "stream": False,
            }
            if augmented_input_ids is not None:
                payload["input_ids"] = augmented_input_ids
            elif augmented_text is not None:
                payload["text"] = augmented_text
            else:
                raise ValueError("Failed to construct augmented prompt")

            async with semaphore:
                resp = await client.post(f"{config.api_url}/generate", json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    "SGL /generate failed during REFRAIN continuation. "
                    f"status={resp.status_code}, body={resp.text[:1000]}"
                )
            items = _as_output_items(resp.json())
            if len(items) != 1:
                raise RuntimeError(
                    f"Expected single continuation output, got {len(items)}"
                )
            output = items[0]

            continued_row = copy.deepcopy(record)
            continued_row["seed"] = continued_seed
            if augmented_input_ids is not None:
                continued_row["input_ids"] = augmented_input_ids
            if augmented_text is not None:
                continued_row["input_text"] = augmented_text
            continued_row["response"] = str(output.get("text") or "")
            output_ids_raw = output.get("output_ids")
            if not isinstance(output_ids_raw, list):
                raise ValueError(
                    f"Missing output_ids in continuation result for row {row_index}"
                )
            continued_row["output_ids"] = [int(token_id) for token_id in output_ids_raw]
            continued_row["response_len_tokens"] = len(continued_row["output_ids"])
            continued_row["probs"] = []
            finish_reason = output.get("meta_info", {}).get("finish_reason")
            if isinstance(finish_reason, dict):
                finish_reason = finish_reason.get("type")
            continued_row["finish_reason"] = (
                str(finish_reason) if finish_reason is not None else None
            )
            continued_row["refrain_postprocess"] = {
                "decision": "continue_with_direct_answer",
                "stop_index": stop_index,
                "kept_prefix_tokens": stop_index,
                "original_output_tokens": len(record.get("output_ids", []))
                if isinstance(record.get("output_ids"), list)
                else None,
                "prompt_injection": config.direct_answer_prompt,
                "prompt_injection_style": config.direct_answer_prompt_style,
                "prefix_preview": prefix_text[:200],
                "continued_seed": continued_seed,
            }
            continued_row["prune_meta"] = {
                "strategy": "refrain",
                "decision": "prune_then_regenerate",
                "reason": "REFRAIN stop trigger followed by direct-answer continuation",
                "cut_index": stop_index,
                "original_len": len(record.get("output_ids", []))
                if isinstance(record.get("output_ids"), list)
                else None,
                "pruned_len": stop_index,
                "postprocess_strategy": "direct_answer_if_triggered",
            }
            return ContinuationTrace(
                row_index=row_index,
                continued_row=continued_row,
                continued_seed=continued_seed,
            )

        tasks = [
            _run_one(row_index, record, stop_index, stop_char_end)
            for row_index, record, stop_index, stop_char_end in pending_requests
        ]
        traces = dict(completed_traces)
        checkpoint_handle = (
            checkpoint_path.open("ab") if checkpoint_path is not None else None
        )
        try:
            with tqdm(
                total=len(requests),
                initial=len(completed_traces),
                desc="REFRAIN continue",
                unit="sample",
            ) as pbar:
                completed_count = len(completed_traces)
                for future in asyncio.as_completed(tasks):
                    trace = await future
                    traces[trace.row_index] = trace
                    completed_count += 1
                    if checkpoint_handle is not None:
                        _append_continuation_checkpoint(
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply REFRAIN postprocess by truncating at the detected stop point, "
            "injecting a direct-answer user turn, and continuing generation."
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
        default="evaluation/results/refrain_postprocess",
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
        "--tokenizer-path",
        type=str,
        default="",
        help="Tokenizer path used to rebuild the augmented prompt when input_ids exist.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="",
        help="Override generation API URL. Defaults to source run api_generation.api_url.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Override logical model name. Defaults to source run metadata.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--omit-max-new-tokens", action="store_true")
    parser.add_argument("--seed-offset", type=int, default=2000003)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--concurrency-limit", type=int, default=16)
    parser.add_argument("--min-keep-tokens", type=int, default=0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume REFRAIN continuations from an incremental checkpoint in the output directory.",
    )
    parser.add_argument(
        "--checkpoint-file-name",
        type=str,
        default=".refrain_continuation.checkpoint.jsonl",
        help="Incremental continuation checkpoint JSONL filename inside the output directory.",
    )
    parser.add_argument(
        "--checkpoint-fsync-interval",
        type=int,
        default=1,
        help="fsync checkpoint every N completed continuations (0 disables fsync).",
    )
    parser.add_argument(
        "--direct-answer-prompt-style",
        type=str,
        choices=DIRECT_ANSWER_PROMPT_STYLES,
        default=DIRECT_ANSWER_PROMPT_STYLE_DEFAULT,
        help=(
            "Assistant prefix used after the REFRAIN direct-answer user turn. "
            "'assistant' keeps the Qwen-compatible default; "
            "'assistant_empty_think' appends <think></think> for Nemotron-style templates."
        ),
    )
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_records < 0:
        raise ValueError("--max-records must be >= 0")
    if not args.checkpoint_file_name.strip():
        raise ValueError("--checkpoint-file-name must be non-empty")
    if args.checkpoint_fsync_interval < 0:
        raise ValueError("--checkpoint-fsync-interval must be >= 0")

    input_dir = Path(args.input_dir)
    input_path = resolve_input_file(input_dir)
    original_args = _load_original_args(input_dir)
    config = _build_config(args, original_args)

    tokenizer = None
    tokenizer_path = _resolve_tokenizer_path(args.tokenizer_path, original_args)
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )

    records = load_jsonl(input_path)
    if args.max_records > 0:
        records = records[: args.max_records]

    requests: list[tuple[int, dict[str, Any], int, int | None]] = []
    kept = 0
    triggerable = 0
    for row_index, record in enumerate(
        tqdm(records, desc="REFRAIN scan", unit="record")
    ):
        refrain = record.get("refrain")
        if not isinstance(refrain, dict):
            kept += 1
            record["refrain_postprocess"] = {
                "decision": "keep",
                "reason": "missing refrain metadata",
            }
            continue

        stop_index_raw = refrain.get("stop_index")
        stop_char_end_raw = refrain.get("stop_char_end")
        output_ids = record.get("output_ids")
        if not isinstance(output_ids, list):
            kept += 1
            record["refrain_postprocess"] = {
                "decision": "keep",
                "reason": "missing output_ids",
            }
            continue

        output_len = len(output_ids)
        if stop_index_raw is None:
            kept += 1
            record["refrain_postprocess"] = {
                "decision": "keep",
                "reason": "no REFRAIN trigger",
            }
            continue

        stop_index = max(0, min(int(stop_index_raw), output_len))
        stop_char_end = int(stop_char_end_raw) if stop_char_end_raw is not None else None
        if stop_index < config.min_keep_tokens:
            kept += 1
            record["refrain_postprocess"] = {
                "decision": "keep",
                "reason": (
                    f"triggered at {stop_index} < min_keep_tokens {config.min_keep_tokens}"
                ),
                "stop_index": stop_index,
            }
            continue
        if stop_index >= output_len:
            kept += 1
            record["refrain_postprocess"] = {
                "decision": "keep",
                "reason": "trigger already at or beyond full output length",
                "stop_index": stop_index,
            }
            continue

        triggerable += 1
        requests.append((row_index, record, stop_index, stop_char_end))

    out_dir: Path | None = None
    output_path: Path | None = None
    checkpoint_path: Path | None = None
    completed_traces: dict[int, ContinuationTrace] = {}
    if not args.dry_run:
        out_dir = prepare_output_dir(args, input_path)
        output_path = out_dir / args.result_file_name
        checkpoint_path = out_dir / args.checkpoint_file_name
        if args.resume:
            completed_traces = _load_continuation_checkpoint(checkpoint_path)
            if completed_traces:
                print(
                    "Resuming REFRAIN continuations from checkpoint: "
                    f"{len(completed_traces)} / {len(requests)} continuations complete"
                )

    continuation_traces = asyncio.run(
        _continue_async(
            requests=requests,
            config=config,
            tokenizer=tokenizer,
            completed_traces=completed_traces,
            checkpoint_path=checkpoint_path if args.resume else None,
            checkpoint_fsync_interval=args.checkpoint_fsync_interval,
        )
    )

    replaced = 0
    out_rows: list[dict[str, Any]] = []
    for row_index, record in enumerate(
        tqdm(records, desc="REFRAIN finalize", unit="record")
    ):
        trace = continuation_traces.get(row_index)
        if trace is None:
            out_rows.append(record)
            continue
        replaced += 1
        out_rows.append(trace.continued_row)

    if not args.dry_run:
        assert out_dir is not None
        assert output_path is not None
        args_json = copy.deepcopy(original_args)
        args_json["probe_prune"] = {
            "input_file": str(input_path),
            "input_dir": str(input_dir),
            "output": str(out_dir),
            "result_file": args.result_file_name,
            "experiment_name": args.experiment_name,
            "output_base": args.output,
            "output_dir_override": args.output_dir or None,
            "tokenizer_path": tokenizer_path or None,
            "simulation_mode": "refrain_direct_answer",
            "prune_config": {
                "schema_version": 2,
                "strategy": "refrain",
                "strategy_args": {
                    "min_keep_tokens": config.min_keep_tokens,
                },
                "postprocess_strategy": "direct_answer_if_triggered",
                "postprocess_strategy_args": {
                    "prompt_injection": config.direct_answer_prompt,
                    "prompt_injection_style": config.direct_answer_prompt_style,
                    "seed_offset": config.seed_offset,
                },
            },
            "postprocess_runtime": {
                "api_url": config.api_url,
                "model_name": config.model_name,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "timeout_s": config.timeout_s,
                "concurrency_limit": config.concurrency_limit,
                "resume": args.resume,
                "checkpoint_file": str(checkpoint_path) if checkpoint_path else None,
                "checkpoint_fsync_interval": args.checkpoint_fsync_interval,
                "resumed_continuations": len(completed_traces),
            },
        }
        if config.max_new_tokens is not None:
            args_json["probe_prune"]["postprocess_runtime"]["max_new_tokens"] = (
                config.max_new_tokens
            )

        dump_json(out_dir / "args.json", args_json)
        assert output_path is not None
        write_jsonl(output_path, out_rows)
        migrate_legacy_run(out_dir, force=True)

    print(f"Input records: {len(records)}")
    print(f"Triggered records: {triggerable}")
    print(f"Continued records: {replaced}")
    print(f"Kept records: {kept}")
    print(f"Tokenizer path: {tokenizer_path or 'unresolved'}")
    print(f"Generation API URL: {config.api_url}")
    if args.dry_run:
        print("Dry run only, output file not written.")
    else:
        assert out_dir is not None
        assert output_path is not None
        print(f"Output directory: {out_dir}")
        print(f"Saved REFRAIN postprocess results to: {output_path}")


if __name__ == "__main__":
    main()
