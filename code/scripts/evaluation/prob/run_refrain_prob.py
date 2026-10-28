#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import orjson
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from cognitive_probe.evaluation.io import dump_json, load_json, load_jsonl, write_jsonl
from cognitive_probe.evaluation.refrain import (
    DEFAULT_PROVISIONAL_ANSWER_CUES,
    DEFAULT_REFLECTION_TRIGGER_VOCAB,
    analyze_refrain,
    build_step_ranges,
)


@dataclass(slots=True)
class PreparedRecord:
    row_index: int
    step_texts: list[str]


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


def _load_refrain_checkpoint(
    *,
    checkpoint_path: Path,
    records: list[dict[str, Any]],
) -> tuple[set[int], int, int, int]:
    if not checkpoint_path.exists():
        return set(), 0, 0, 0

    latest_rows: dict[int, dict[str, Any]] = {}
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
            row_index = int(payload["row_index"])
        except (KeyError, TypeError, ValueError):
            invalid_lines += 1
            continue
        if row_index < 0 or row_index >= len(records):
            invalid_lines += 1
            continue
        probs = payload.get("probs")
        refrain = payload.get("refrain")
        output_ids = records[row_index].get("output_ids")
        if (
            not isinstance(probs, list)
            or not isinstance(refrain, dict)
            or not isinstance(output_ids, list)
            or len(probs) != len(output_ids)
        ):
            invalid_lines += 1
            continue
        latest_rows[row_index] = payload

    completed_indices: set[int] = set()
    triggered_records = 0
    scored_tokens = 0
    total_steps = 0
    for row_index, payload in latest_rows.items():
        probs = [float(value) for value in payload["probs"]]
        records[row_index]["probs"] = probs
        records[row_index]["refrain"] = payload["refrain"]
        completed_indices.add(row_index)
        total_steps += int(payload.get("num_steps") or 0)
        scored_tokens += int(payload.get("scored_tokens") or 0)
        if bool(payload.get("triggered")):
            triggered_records += 1

    if invalid_lines:
        print(
            f"Warning: ignored {invalid_lines} invalid REFRAIN checkpoint line(s) in {checkpoint_path}"
        )
    return completed_indices, triggered_records, scored_tokens, total_steps


def _append_refrain_checkpoint(
    *,
    handle: Any,
    row_index: int,
    record: dict[str, Any],
    triggered: bool,
    num_steps: int,
    scored_tokens: int,
    completed_count: int,
    fsync_interval: int,
) -> None:
    payload = {
        "row_index": row_index,
        "probs": record["probs"],
        "refrain": record["refrain"],
        "triggered": triggered,
        "num_steps": num_steps,
        "scored_tokens": scored_tokens,
    }
    handle.write(orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE))
    handle.flush()
    if fsync_interval > 0 and completed_count % fsync_interval == 0:
        os.fsync(handle.fileno())


async def _embed_text_batches(
    *,
    texts: list[str],
    api_url: str,
    model_name: str,
    batch_size: int,
    timeout_s: float,
    request_concurrency: int,
    max_retries: int,
) -> list[list[float]]:
    if not texts:
        return []

    results: list[list[float] | None] = [None] * len(texts)
    semaphore = asyncio.Semaphore(request_concurrency)
    timeout = httpx.Timeout(timeout_s)

    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:

        async def _request_batch(start: int, end: int) -> None:
            batch = texts[start:end]
            payload = {"model": model_name, "input": batch}
            last_error: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    async with semaphore:
                        resp = await client.post(
                            f"{api_url}/v1/embeddings",
                            json=payload,
                        )
                    if resp.status_code != 200:
                        raise RuntimeError(
                            "Embedding request failed. "
                            f"status={resp.status_code}, body={resp.text[:1000]}"
                        )
                    resp_json = resp.json()
                    data = resp_json.get("data")
                    if not isinstance(data, list) or len(data) != len(batch):
                        raise ValueError(
                            "Unexpected embeddings response length: "
                            f"expected={len(batch)}, got={len(data) if isinstance(data, list) else 'invalid'}"
                        )
                    for offset, item in enumerate(data):
                        if not isinstance(item, dict):
                            raise ValueError(
                                f"Invalid embedding item type: {type(item).__name__}"
                            )
                        embedding = item.get("embedding")
                        if not isinstance(embedding, list):
                            raise ValueError("Missing embedding vector in response item")
                        results[start + offset] = [float(value) for value in embedding]
                    return
                except Exception as exc:  # noqa: PERF203
                    last_error = exc
                    if attempt >= max_retries:
                        break
                    await asyncio.sleep(min(1.0, 0.2 * (attempt + 1)))
            raise RuntimeError(
                "Failed to embed batch "
                f"[{start}, {end}) size={end - start}"
            ) from last_error

        async def _run_batch(start: int, end: int) -> None:
            try:
                await _request_batch(start, end)
                return
            except Exception as exc:
                if end - start <= 1:
                    raise RuntimeError(
                        "Failed to embed single step "
                        f"idx={start}, chars={len(texts[start])}"
                    ) from exc

                mid = start + ((end - start) // 2)
                tqdm.write(
                    "REFRAIN embeddings fallback split: "
                    f"[{start}, {end}) -> [{start}, {mid}) + [{mid}, {end})"
                )
                await _run_batch(start, mid)
                await _run_batch(mid, end)

        starts = list(range(0, len(texts), batch_size))
        tasks = [_run_batch(start, min(start + batch_size, len(texts))) for start in starts]
        with tqdm(total=len(tasks), desc="REFRAIN embeddings", unit="batch") as pbar:
            for future in asyncio.as_completed(tasks):
                await future
                pbar.update(1)

    missing = [idx for idx, value in enumerate(results) if value is None]
    if missing:
        raise RuntimeError(f"Missing embeddings for step indices: {missing[:20]}")
    return [[float(value) for value in vector] for vector in results if vector is not None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute REFRAIN-style semantic redundancy scores for generated reasoning "
            "traces and write them back as token-aligned probs."
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
        help="Result filename inside output directory.",
    )
    parser.add_argument(
        "--embedding-url",
        type=str,
        default="http://219.216.69.50:10001",
        help="OpenAI-compatible embedding service base URL.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="bge-m3",
        help="Embedding model name exposed by the embedding service.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Semantic redundancy threshold used by REFRAIN.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="",
        help="Tokenizer path used to align step spans with output_ids.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of reasoning steps per embedding request.",
    )
    parser.add_argument(
        "--request-concurrency",
        type=int,
        default=8,
        help="Max concurrent embedding requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry count for failed embedding requests.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Only process the first N records for debugging (0 means all).",
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

    if args.threshold < 0.0 or args.threshold > 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.request_concurrency <= 0:
        raise ValueError("--request-concurrency must be > 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")
    if args.max_records < 0:
        raise ValueError("--max-records must be >= 0")
    if not args.checkpoint_file_name.strip():
        raise ValueError("--checkpoint-file-name must be non-empty")
    if args.checkpoint_fsync_interval < 0:
        raise ValueError("--checkpoint-fsync-interval must be >= 0")

    input_dir = Path(args.input_dir)
    input_path = _resolve_input_file(input_dir)
    original_args = _load_original_args(input_dir)

    tokenizer_path = _resolve_tokenizer_path(args.tokenizer_path, original_args)
    if not tokenizer_path:
        raise ValueError(
            "Unable to resolve tokenizer path. Pass --tokenizer-path explicitly or "
            "ensure it exists in the source run args.json."
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        if not out_dir.is_dir():
            raise NotADirectoryError(
                f"Output path exists but is not a directory: {out_dir}"
            )
        if (out_dir / args.result_file_name).exists():
            raise FileExistsError(
                f"Output file already exists: {out_dir / args.result_file_name}"
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

    completed_indices: set[int] = set()
    resumed_triggered_records = 0
    resumed_scored_tokens = 0
    resumed_total_steps = 0
    if args.resume:
        (
            completed_indices,
            resumed_triggered_records,
            resumed_scored_tokens,
            resumed_total_steps,
        ) = _load_refrain_checkpoint(
            checkpoint_path=checkpoint_path,
            records=records,
        )
        if completed_indices:
            print(
                "Resuming REFRAIN prob from checkpoint: "
                f"{len(completed_indices)} / {len(records)} records complete"
            )

    prepared: list[PreparedRecord] = []
    all_step_texts: list[str] = []
    total_steps = resumed_total_steps

    for row_index, record in enumerate(
        tqdm(records, desc="REFRAIN prepare", unit="record")
    ):
        output_ids_raw = record.get("output_ids")
        if not isinstance(output_ids_raw, list):
            raise ValueError(f"Record {row_index} missing list output_ids")
        output_ids = [int(token_id) for token_id in output_ids_raw]
        response_text = str(record.get("response") or "")

        if row_index in completed_indices:
            continue

        record["probs"] = [0.0] * len(output_ids)
        step_ranges = build_step_ranges(
            response_text=response_text,
            output_ids=output_ids,
            tokenizer=tokenizer,
        )
        step_texts = [step.text for step in step_ranges]
        record["_refrain_step_ranges"] = step_ranges
        prepared.append(PreparedRecord(row_index=row_index, step_texts=step_texts))
        all_step_texts.extend(step_texts)
        total_steps += len(step_texts)

    all_embeddings = asyncio.run(
        _embed_text_batches(
            texts=all_step_texts,
            api_url=args.embedding_url.rstrip("/"),
            model_name=args.embedding_model,
            batch_size=args.batch_size,
            timeout_s=args.timeout,
            request_concurrency=args.request_concurrency,
            max_retries=args.max_retries,
        )
    )

    global_cursor = 0
    triggered_records = resumed_triggered_records
    scored_tokens = resumed_scored_tokens
    checkpoint_handle = checkpoint_path.open("ab") if args.resume else None
    try:
        score_iter = tqdm(
            prepared,
            total=len(records),
            initial=len(completed_indices),
            desc="REFRAIN score",
            unit="record",
        )
        completed_count = len(completed_indices)
        for item in score_iter:
            record = records[item.row_index]
            step_ranges = record.pop("_refrain_step_ranges")
            if not isinstance(step_ranges, list):
                raise ValueError("Internal REFRAIN step range state corrupted")

            step_embeddings = all_embeddings[global_cursor : global_cursor + len(step_ranges)]
            global_cursor += len(step_ranges)

            decision = analyze_refrain(
                step_ranges=step_ranges,
                embeddings=step_embeddings,
                threshold=args.threshold,
                reflection_vocab=DEFAULT_REFLECTION_TRIGGER_VOCAB,
                answer_cues=DEFAULT_PROVISIONAL_ANSWER_CUES,
            )
            triggered = decision.stop_step_index is not None
            if triggered:
                triggered_records += 1

            probs = record.get("probs")
            if not isinstance(probs, list):
                raise ValueError("Internal REFRAIN probs state corrupted")
            record_scored_tokens = 0
            for signal in decision.steps:
                for token_idx in range(signal.token_start, min(signal.token_end, len(probs))):
                    probs[token_idx] = float(signal.similarity)
                    scored_tokens += 1
                    record_scored_tokens += 1

            record["refrain"] = decision.to_dict()
            completed_count += 1
            if checkpoint_handle is not None:
                _append_refrain_checkpoint(
                    handle=checkpoint_handle,
                    row_index=item.row_index,
                    record=record,
                    triggered=triggered,
                    num_steps=len(step_ranges),
                    scored_tokens=record_scored_tokens,
                    completed_count=completed_count,
                    fsync_interval=args.checkpoint_fsync_interval,
                )
    finally:
        if checkpoint_handle is not None:
            checkpoint_handle.flush()
            if args.checkpoint_fsync_interval > 0:
                os.fsync(checkpoint_handle.fileno())
            checkpoint_handle.close()

    for record in records:
        record.pop("_refrain_step_ranges", None)

    if global_cursor != len(all_embeddings):
        raise RuntimeError(
            f"REFRAIN embedding cursor mismatch: consumed={global_cursor}, available={len(all_embeddings)}"
        )

    missing_indices = [
        row_index
        for row_index, record in enumerate(records)
        if "refrain" not in record
    ]
    if missing_indices:
        raise RuntimeError(
            "REFRAIN scoring did not complete all records. "
            f"missing_records={len(missing_indices)}, first_missing={missing_indices[:10]}, "
            f"checkpoint={checkpoint_path}"
        )

    merged_args = copy.deepcopy(original_args)
    merged_args["probe"] = {
        "prob_method": "refrain",
        "input_dir": str(input_dir),
        "input_file": str(input_path),
        "output": str(out_dir),
        "result_file": args.result_file_name,
        "tokenizer_path": tokenizer_path,
        "embedding_url": args.embedding_url.rstrip("/"),
        "embedding_api_path": "/v1/embeddings",
        "embedding_model": args.embedding_model,
        "threshold": args.threshold,
        "step_delimiter": "blank_line_then_fallback",
        "reflection_trigger_vocab_size": len(DEFAULT_REFLECTION_TRIGGER_VOCAB),
        "provisional_answer_cue_count": len(DEFAULT_PROVISIONAL_ANSWER_CUES),
        "batch_size": args.batch_size,
        "request_concurrency": args.request_concurrency,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "max_records": args.max_records,
        "scored_records": len(records),
        "triggered_records": triggered_records,
        "total_steps": total_steps,
        "scored_tokens": scored_tokens,
        "resume": args.resume,
        "checkpoint_file": str(checkpoint_path),
        "checkpoint_fsync_interval": args.checkpoint_fsync_interval,
        "resumed_records": len(completed_indices),
        "resumed_triggered_records": resumed_triggered_records,
        "resumed_total_steps": resumed_total_steps,
        "resumed_scored_tokens": resumed_scored_tokens,
    }

    write_jsonl(output_path, records)
    dump_json(out_dir / "args.json", merged_args)

    print(f"Input records: {len(records)}")
    print(f"Total REFRAIN steps: {total_steps}")
    print(f"Triggered records: {triggered_records}")
    print(f"Scored tokens: {scored_tokens}")
    print(f"Output directory: {out_dir}")
    print(f"Saved REFRAIN-scored results to: {output_path}")


if __name__ == "__main__":
    main()
