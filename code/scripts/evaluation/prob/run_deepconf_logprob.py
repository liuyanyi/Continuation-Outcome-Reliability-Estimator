#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import os
from pathlib import Path
from typing import Any

import httpx
import orjson
from tqdm.auto import tqdm

from cognitive_probe.evaluation.deepconf_logprob import (
    INVALID_SCORE,
    align_input_top_logprobs,
    compute_token_confidence,
)
from cognitive_probe.evaluation.io import dump_json, load_json, load_jsonl, write_jsonl


def _resolve_input_file(input_dir: Path) -> Path:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input-dir must be a directory: {input_dir}")

    candidates = [input_dir / "generation_results.jsonl", input_dir / "output.jsonl"]
    existing = [path for path in candidates if path.exists()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        return input_dir / "generation_results.jsonl"
    raise FileNotFoundError(
        "No generation file found in input directory. Expected one of: "
        "generation_results.jsonl, output.jsonl"
    )


def _load_original_args(input_dir: Path) -> dict[str, Any]:
    src_args_path = input_dir / "args.json"
    if not src_args_path.exists():
        return {}
    return load_json(src_args_path)


def _normalize_token_ids(value: Any, *, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"Each record must contain list[int] field: {field_name}")
    return [int(token_id) for token_id in value]


def _maybe_existing_output_top_logprobs(record: dict[str, Any]) -> list[Any] | None:
    for key in ("output_top_logprobs", "top_logprobs"):
        value = record.get(key)
        if isinstance(value, list):
            return value
    meta_info = record.get("meta_info")
    if isinstance(meta_info, dict):
        value = meta_info.get("output_top_logprobs")
        if isinstance(value, list):
            return value
    return None


async def _fetch_output_top_logprobs(
    *,
    client: httpx.AsyncClient,
    url: str,
    record: dict[str, Any],
    topk: int,
    request_max_new_tokens: int,
    timeout_s: float,
    max_retries: int,
) -> list[Any]:
    input_ids = _normalize_token_ids(record.get("input_ids"), field_name="input_ids")
    output_ids = _normalize_token_ids(record.get("output_ids"), field_name="output_ids")
    input_text = record.get("input_text")
    response_text = record.get("response")

    # Prefer token-id replay when available; otherwise fall back to raw text replay.
    if input_ids:
        full_ids = input_ids + output_ids
        payload: dict[str, Any] = {
            "input_ids": full_ids,
            "stream": False,
            "max_tokens": request_max_new_tokens,
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": topk,
        }
    elif isinstance(input_text, str) and isinstance(response_text, str):
        payload = {
            "text": input_text + response_text,
            "stream": False,
            "max_tokens": request_max_new_tokens,
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": topk,
        }
    else:
        raise ValueError("Record must contain input_ids or both input_text/response")

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.post(
                f"{url}/generate", json=payload, timeout=timeout_s
            )
            if response.status_code != 200:
                raise RuntimeError(
                    "SGL /generate failed when fetching DeepConf logprobs. "
                    f"status={response.status_code}, body={response.text[:1000]}"
                )
            payload_json = response.json()
            if not isinstance(payload_json, dict):
                raise ValueError(
                    f"Unexpected /generate response type: {type(payload_json)}"
                )
            meta_info = payload_json.get("meta_info")
            if not isinstance(meta_info, dict):
                raise ValueError("Missing meta_info in /generate response")
            input_top_logprobs = meta_info.get("input_top_logprobs")
            if not isinstance(input_top_logprobs, list):
                raise ValueError("Missing input_top_logprobs in /generate response")
            return align_input_top_logprobs(
                input_top_logprobs,
                input_len=len(input_ids),
                output_len=len(output_ids),
            )
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            if attempt >= max_retries:
                break
            await asyncio.sleep(min(30.0, 2.0**attempt))

    raise RuntimeError("Failed to fetch DeepConf logprobs") from last_error


def _score_output_top_logprobs(
    *, output_top_logprobs: list[Any], topk: int
) -> list[float]:
    return [
        float(compute_token_confidence(entry, topk=topk))
        for entry in output_top_logprobs
    ]


def _load_checkpoint_records(
    *,
    checkpoint_path: Path,
    records: list[dict[str, Any]],
    score_field: str,
) -> tuple[set[int], int]:
    if not checkpoint_path.exists():
        return set(), 0

    completed_indices: set[int] = set()
    scored_tokens = 0
    invalid_lines = 0

    for line_no, raw_line in enumerate(checkpoint_path.read_bytes().splitlines(), 1):
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
            record_idx = int(payload["record_idx"])
        except (KeyError, TypeError, ValueError):
            invalid_lines += 1
            continue
        if record_idx < 0 or record_idx >= len(records):
            invalid_lines += 1
            continue

        scores = payload.get("scores")
        if not isinstance(scores, list):
            invalid_lines += 1
            continue

        output_ids = _normalize_token_ids(
            records[record_idx].get("output_ids"), field_name="output_ids"
        )
        if len(scores) != len(output_ids):
            invalid_lines += 1
            continue

        normalized_scores = [float(score) for score in scores]
        if record_idx in completed_indices:
            scored_tokens -= sum(
                1
                for score in records[record_idx].get(score_field, [])
                if score != INVALID_SCORE
            )
        records[record_idx][score_field] = normalized_scores
        completed_indices.add(record_idx)
        scored_tokens += sum(1 for score in normalized_scores if score != INVALID_SCORE)

    if invalid_lines:
        print(
            f"Warning: ignored {invalid_lines} invalid checkpoint line(s) in {checkpoint_path}"
        )
    return completed_indices, scored_tokens


def _write_checkpoint_row(
    *,
    handle: Any,
    record_idx: int,
    scores: list[float],
    used_existing: bool,
    completed_count: int,
    fsync_interval: int,
) -> None:
    payload = {
        "record_idx": record_idx,
        "scores": scores,
        "used_existing": used_existing,
    }
    handle.write(orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE))
    handle.flush()
    if fsync_interval > 0 and completed_count % fsync_interval == 0:
        os.fsync(handle.fileno())


async def _score_records(
    *,
    records: list[dict[str, Any]],
    completed_indices: set[int],
    checkpoint_path: Path | None,
    checkpoint_fsync_interval: int,
    url: str,
    topk: int,
    request_max_new_tokens: int,
    timeout_s: float,
    max_retries: int,
    request_concurrency: int,
    score_field: str,
) -> tuple[int, int, int]:
    semaphore = asyncio.Semaphore(request_concurrency)
    fetched_records = 0
    existing_records = 0
    scored_tokens = 0
    pending_indices = [
        record_idx
        for record_idx in range(len(records))
        if record_idx not in completed_indices
    ]

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s), trust_env=False
    ) as client:

        async def _run_one(record_idx: int) -> tuple[int, list[float], bool]:
            record = records[record_idx]
            async with semaphore:
                output_ids = _normalize_token_ids(
                    record.get("output_ids"), field_name="output_ids"
                )
                existing_output_top_logprobs = _maybe_existing_output_top_logprobs(
                    record
                )
                if existing_output_top_logprobs is not None:
                    scored = _score_output_top_logprobs(
                        output_top_logprobs=existing_output_top_logprobs,
                        topk=topk,
                    )
                    if len(scored) != len(output_ids):
                        raise ValueError(
                            "Existing output_top_logprobs length mismatch: "
                            f"record_idx={record_idx}, scored={len(scored)}, output_ids={len(output_ids)}"
                        )
                    return record_idx, scored, True

                fetched_output_top_logprobs = await _fetch_output_top_logprobs(
                    client=client,
                    url=url,
                    record=record,
                    topk=topk,
                    request_max_new_tokens=request_max_new_tokens,
                    timeout_s=timeout_s,
                    max_retries=max_retries,
                )
                scored = _score_output_top_logprobs(
                    output_top_logprobs=fetched_output_top_logprobs,
                    topk=topk,
                )
                if len(scored) != len(output_ids):
                    raise ValueError(
                        "Fetched input_top_logprobs length mismatch after alignment: "
                        f"record_idx={record_idx}, scored={len(scored)}, output_ids={len(output_ids)}"
                )
                return record_idx, scored, False

        tasks = [asyncio.create_task(_run_one(record_idx)) for record_idx in pending_indices]
        checkpoint_handle = (
            checkpoint_path.open("ab") if checkpoint_path is not None else None
        )
        try:
            with tqdm(
                total=len(records), desc="DeepConf logprob scoring", unit="record"
            ) as pbar:
                if completed_indices:
                    pbar.update(len(completed_indices))
                completed_count = len(completed_indices)
                for future in asyncio.as_completed(tasks):
                    try:
                        record_idx, scores, used_existing = await future
                    except Exception:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        raise

                    records[record_idx][score_field] = scores
                    scored_tokens += sum(
                        1 for score in scores if score != INVALID_SCORE
                    )
                    if used_existing:
                        existing_records += 1
                    else:
                        fetched_records += 1
                    completed_count += 1
                    if checkpoint_handle is not None:
                        _write_checkpoint_row(
                            handle=checkpoint_handle,
                            record_idx=record_idx,
                            scores=scores,
                            used_existing=used_existing,
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

    return fetched_records, existing_records, scored_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute DeepConf-style token scores from top-k logprobs and write them "
            "back as a prob run aligned with output_ids."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing generation_results.jsonl or output.jsonl and optional args.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for args.json and scored results.",
    )
    parser.add_argument(
        "--result-file-name",
        type=str,
        default="generation_results.probed.jsonl",
        help="Output result filename inside output directory.",
    )
    parser.add_argument(
        "--score-field",
        type=str,
        default="probs",
        help="Field name used to store per-token DeepConf scores.",
    )
    parser.add_argument(
        "--sgl-url",
        type=str,
        default="http://127.0.0.1:30000",
        help="SGL /generate base URL used when records do not already store top logprobs.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Use top-k logprobs to compute token confidence.",
    )
    parser.add_argument(
        "--request-max-new-tokens",
        type=int,
        default=1,
        help="Extra generation tokens requested when replaying full sequences for prompt logprobs.",
    )
    parser.add_argument(
        "--request-concurrency",
        type=int,
        default=32,
        help="Max concurrent /generate requests while fetching prompt logprobs.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Process only first N records for debug (0 means all).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from an incremental checkpoint in --output-dir when present.",
    )
    parser.add_argument(
        "--checkpoint-file-name",
        type=str,
        default=".generation_results.probed.checkpoint.jsonl",
        help="Incremental checkpoint JSONL filename inside --output-dir.",
    )
    parser.add_argument(
        "--checkpoint-fsync-interval",
        type=int,
        default=1,
        help="fsync checkpoint every N completed records (0 disables fsync).",
    )
    args = parser.parse_args()

    if args.topk <= 0:
        raise ValueError("--topk must be > 0")
    if args.request_max_new_tokens < 0:
        raise ValueError("--request-max-new-tokens must be >= 0")
    if args.request_concurrency <= 0:
        raise ValueError("--request-concurrency must be > 0")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")
    if not args.score_field.strip():
        raise ValueError("--score-field must be non-empty")
    if not args.result_file_name.strip():
        raise ValueError("--result-file-name must be non-empty")
    if not args.checkpoint_file_name.strip():
        raise ValueError("--checkpoint-file-name must be non-empty")
    if args.checkpoint_fsync_interval < 0:
        raise ValueError("--checkpoint-fsync-interval must be >= 0")

    input_dir = Path(args.input_dir)
    input_path = _resolve_input_file(input_dir)
    original_args = _load_original_args(input_dir)

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        if not out_dir.is_dir():
            raise NotADirectoryError(
                f"Output path exists but is not a directory: {out_dir}"
            )
        if (out_dir / args.result_file_name).exists():
            raise FileExistsError(
                f"Output result file already exists: {out_dir / args.result_file_name}"
            )
        if any(out_dir.iterdir()) and not args.resume:
            raise FileExistsError(
                "Output directory already exists and is not empty; pass --resume "
                f"to continue from a checkpoint: {out_dir}"
            )
    else:
        out_dir.mkdir(parents=True, exist_ok=False)

    output_path = out_dir / args.result_file_name
    checkpoint_path = out_dir / args.checkpoint_file_name

    records = load_jsonl(input_path)
    if args.max_records > 0:
        records = records[: args.max_records]
    if not records:
        raise ValueError("No records found in input JSONL")

    for record in records:
        output_ids = _normalize_token_ids(
            record.get("output_ids"), field_name="output_ids"
        )
        record[args.score_field] = [INVALID_SCORE] * len(output_ids)

    completed_indices: set[int] = set()
    resumed_scored_tokens = 0
    if args.resume:
        completed_indices, resumed_scored_tokens = _load_checkpoint_records(
            checkpoint_path=checkpoint_path,
            records=records,
            score_field=args.score_field,
        )
        if completed_indices:
            print(
                "Resuming DeepConf logprob from checkpoint: "
                f"{len(completed_indices)} / {len(records)} records already scored"
            )

    fetched_records, existing_records, scored_tokens = asyncio.run(
        _score_records(
            records=records,
            completed_indices=completed_indices,
            checkpoint_path=checkpoint_path if args.resume else None,
            checkpoint_fsync_interval=args.checkpoint_fsync_interval,
            url=args.sgl_url.rstrip("/"),
            topk=args.topk,
            request_max_new_tokens=args.request_max_new_tokens,
            timeout_s=args.timeout,
            max_retries=args.max_retries,
            request_concurrency=args.request_concurrency,
            score_field=args.score_field,
        )
    )

    missing_indices = [
        record_idx
        for record_idx, record in enumerate(records)
        if args.score_field not in record
        or len(record[args.score_field])
        != len(_normalize_token_ids(record.get("output_ids"), field_name="output_ids"))
    ]
    if missing_indices:
        raise RuntimeError(
            "DeepConf logprob scoring did not complete all records. "
            f"missing_or_mismatched_records={len(missing_indices)}, "
            f"first_missing={missing_indices[:10]}, checkpoint={checkpoint_path}"
        )

    write_jsonl(output_path, records)

    merged_args = copy.deepcopy(original_args)
    merged_args["probe"] = {
        "prob_method": "deepconf_logprob",
        "input_dir": str(input_dir),
        "input_file": str(input_path),
        "output": str(out_dir),
        "result_file": args.result_file_name,
        "score_field": args.score_field,
        "formula": "neg_avg_topk_logprob",
        "invalid_value": INVALID_SCORE,
        "lower_is_better": False,
        "logprob_source": "meta_info.input_top_logprobs",
        "request_mode": "prompt_logprob",
        "sgl_url": args.sgl_url,
        "topk": args.topk,
        "request_max_new_tokens": args.request_max_new_tokens,
        "request_concurrency": args.request_concurrency,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "max_records": args.max_records,
        "resume": args.resume,
        "checkpoint_file": str(checkpoint_path),
        "checkpoint_fsync_interval": args.checkpoint_fsync_interval,
        "resumed_records": len(completed_indices),
        "fetched_logprob_records": fetched_records,
        "used_existing_logprob_records": existing_records,
        "scored_records": len(records),
        "scored_tokens": resumed_scored_tokens + scored_tokens,
    }
    dump_json(out_dir / "args.json", merged_args)

    print(f"Input records: {len(records)}")
    print(f"Resumed checkpoint records: {len(completed_indices)}")
    print(f"Fetched logprob records: {fetched_records}")
    print(f"Used existing logprob records: {existing_records}")
    print(f"Scored tokens: {resumed_scored_tokens + scored_tokens}")
    print(f"Output directory: {out_dir}")
    print(f"Saved DeepConf-scored results to: {output_path}")


if __name__ == "__main__":
    main()
