#!/usr/bin/env python3
"""
Probe encoding pipeline with producer-consumer architecture.

This script uses a decoupled producer-consumer pattern to maximize throughput:
- Producer: Asynchronously fetches hidden states from SGL encode endpoint
- Consumer: Processes hidden states through probe head with dynamic batching
- Queue: Decouples SGL calls from head inference for maximum parallelism

Key optimizations:
1. Producer continuously fetches from SGL without waiting for head inference
2. Consumer uses dynamic batch sizes (e.g., 2,4,8,16,32,...) for efficient head inference
3. Preferred batch sizes allow trade-off between latency and throughput
4. Queue enables maximum SGL concurrency while batching head calls efficiently
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import os
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import orjson
import torch
from safetensors import safe_open
from tqdm.auto import tqdm

from cognitive_probe.evaluation.io import dump_json, load_json, load_jsonl, write_jsonl
from cognitive_probe.probe_model.probe_model import SingleProbeHead


def _resolve_input_file(input_dir: Path) -> Path:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input-dir must be a directory: {input_dir}")

    candidates = [input_dir / "generation_results.jsonl", input_dir / "output.jsonl"]
    existing = [p for p in candidates if p.exists()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        # Prefer generation_results.jsonl for this probe pipeline.
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


def _load_track_token_ids(path: Path) -> set[int]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(
            f"track token file must be a JSON object mapping token_id->text, got {type(payload)}"
        )

    ids: set[int] = set()
    for key in payload:
        ids.add(int(key))
    if not ids:
        raise ValueError(f"No track token ids found in: {path}")
    return ids


def _extract_lm_head_weight(model_path: str) -> torch.Tensor:
    """Load the input embedding projection used during probe training.

    GenericForProbing computes logit features from input embeddings, including
    models whose output LM head is untied. An LM-head fallback is valid only
    when the checkpoint explicitly declares tied embeddings.
    """
    config_path = Path(model_path) / "config.json"
    config = load_json(config_path) if config_path.exists() else {}
    tied = any(config.get(key, False) for key in ("tie_word_embeddings", "tie_weights", "tie_weight"))

    def pick_key(keys: Any) -> str:
        candidates = (
            "model.embed_tokens.weight", "language_model.model.embed_tokens.weight",
            "backbone.embeddings.weight", "model.embeddings.weight",
            "embed_tokens.weight", "transformer.wte.weight", "tok_embeddings.weight",
            "word_embeddings.weight",
        )
        for key in candidates:
            if key in keys:
                return key
        for key in keys:
            if str(key).endswith(("embed_tokens.weight", "word_embeddings.weight", "wte.weight")):
                return str(key)
        if tied:
            for key in keys:
                if str(key).endswith("lm_head.weight"):
                    return str(key)
        raise KeyError("Cannot find input embedding weights used by probe training")

    root = Path(model_path)
    index = root / "model.safetensors.index.json"
    if index.exists():
        weight_map = load_json(index)["weight_map"]
        key = pick_key(weight_map.keys())
        with safe_open(root / weight_map[key], framework="pt", device="cpu") as weights:
            return weights.get_tensor(key).float().contiguous()
    single = root / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(f"No safetensors checkpoint found under {root}; provide a local base-model directory")
    with safe_open(single, framework="pt", device="cpu") as weights:
        key = pick_key(weights.keys())
        return weights.get_tensor(key).float().contiguous()


def _build_track_tasks(
    records: list[dict[str, Any]],
    track_token_ids: set[int],
    track_token_stride: int,
) -> list[dict[str, Any]]:
    """Build task list with only positions, prefix_ids will be constructed on-demand.

    Only keep one tracked token every `track_token_stride` output positions per record.
    """
    if track_token_stride <= 0:
        raise ValueError("track_token_stride must be > 0")

    tasks: list[dict[str, Any]] = []
    for ridx, record in enumerate(
        tqdm(records, desc="Build track tasks", unit="record")
    ):
        input_ids = record.get("input_ids")
        output_ids = record.get("output_ids")
        if not isinstance(input_ids, list) or not isinstance(output_ids, list):
            raise ValueError(
                "Each record must contain list[int] fields: input_ids and output_ids"
            )

        # Keep one tracked token every N output positions for this record.
        last_kept_pos = -track_token_stride
        for pos, token_id in enumerate(output_ids):
            if int(token_id) not in track_token_ids:
                continue
            if pos - last_kept_pos < track_token_stride:
                continue
            tasks.append(
                {
                    "record_idx": ridx,
                    "token_pos": pos,
                }
            )
            last_kept_pos = pos
    return tasks


def _task_key(record_idx: int, token_pos: int) -> tuple[int, int]:
    return int(record_idx), int(token_pos)


def _load_probe_checkpoint(
    *,
    checkpoint_path: Path,
    records: list[dict[str, Any]],
) -> tuple[set[tuple[int, int]], int, int]:
    if not checkpoint_path.exists():
        return set(), 0, 0

    latest_rows: dict[tuple[int, int], dict[str, Any]] = {}
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
            record_idx = int(payload["record_idx"])
            token_pos = int(payload["token_pos"])
        except (KeyError, TypeError, ValueError):
            invalid_lines += 1
            continue
        if record_idx < 0 or record_idx >= len(records):
            invalid_lines += 1
            continue
        probs = records[record_idx].get("probs")
        if not isinstance(probs, list) or token_pos < 0 or token_pos >= len(probs):
            invalid_lines += 1
            continue
        latest_rows[_task_key(record_idx, token_pos)] = payload

    completed: set[tuple[int, int]] = set()
    scored_positions = 0
    skipped_positions = 0
    for key, payload in latest_rows.items():
        record_idx, token_pos = key
        if bool(payload.get("skipped")):
            skipped_positions += 1
        else:
            try:
                prob = float(payload["prob"])
            except (KeyError, TypeError, ValueError):
                invalid_lines += 1
                continue
            records[record_idx]["probs"][token_pos] = prob
            scored_positions += 1
        completed.add(key)

    if invalid_lines:
        print(
            f"Warning: ignored {invalid_lines} invalid probe checkpoint line(s) in {checkpoint_path}"
        )
    return completed, scored_positions, skipped_positions


def _write_probe_checkpoint_row(
    *,
    handle: Any,
    task: dict[str, Any],
    prob: float | None,
    skipped: bool,
    completed_count: int,
    fsync_interval: int,
) -> None:
    payload: dict[str, Any] = {
        "record_idx": int(task["record_idx"]),
        "token_pos": int(task["token_pos"]),
        "skipped": bool(skipped),
    }
    if prob is not None:
        payload["prob"] = float(prob)
    handle.write(orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE))
    handle.flush()
    if fsync_interval > 0 and completed_count % fsync_interval == 0:
        os.fsync(handle.fileno())


def _extract_target_hidden(
    hidden_states: Any, *, debug_context: str = ""
) -> torch.Tensor:
    """Extract target token hidden state from encode response.

    The encode endpoint returns embeddings directly, typically as a list of vectors.
    """
    if hidden_states is None:
        raise RuntimeError(
            f"Missing hidden_states in SGL encode response{debug_context}"
        )

    hs = torch.tensor(hidden_states, dtype=torch.float32)

    # Common case: list of embeddings, take the last one
    if hs.dim() == 2:  # [seq_len, hidden_dim]
        if hs.shape[0] == 0:
            raise RuntimeError(f"Empty hidden states (seq_len=0){debug_context}")
        return hs[-1]

    # Single embedding
    if hs.dim() == 1:  # [hidden_dim]
        if hs.shape[0] == 0:
            raise RuntimeError(f"Empty hidden state vector{debug_context}")
        return hs

    # Wrapped in batch dimension
    if hs.dim() == 3 and hs.shape[0] == 1:  # [1, seq_len, hidden_dim]
        if hs.shape[1] == 0:
            raise RuntimeError(
                f"Empty hidden states (seq_len=0 in batch){debug_context}"
            )
        return hs[0, -1]

    raise RuntimeError(
        f"Unexpected hidden state shape: {tuple(hs.shape)}{debug_context}"
    )


def _is_context_length_encode_error(resp: httpx.Response) -> bool:
    if resp.status_code != 400:
        return False
    body = str(resp.text or "").lower()
    return "longer than the model's context length" in body


async def _fetch_single_hidden_state(
    *,
    client: httpx.AsyncClient,
    url: str,
    records: list[dict[str, Any]],
    task: dict[str, Any],
    task_idx: int,
    timeout_s: float,
    max_hidden_state_retries: int,
) -> tuple[int, torch.Tensor | None]:
    """Fetch hidden state for a single task using encode endpoint, returns (task_idx, hidden_state)."""
    # Construct prefix_ids on-demand from position
    ridx = int(task["record_idx"])
    pos = int(task["token_pos"])
    record = records[ridx]

    input_ids = record.get("input_ids")
    output_ids = record.get("output_ids")
    if not isinstance(input_ids, list) or not isinstance(output_ids, list):
        raise ValueError(
            "Each record must contain list[int] fields: input_ids and output_ids"
        )
    if pos < 0 or pos >= len(output_ids):
        raise IndexError(
            f"Invalid task token_pos={pos} for record_idx={ridx} with output_len={len(output_ids)}"
        )

    # Include current track token (pos), then take tail token hidden state
    prefix_ids = input_ids + output_ids[: pos + 1]

    if not prefix_ids:
        raise ValueError(
            f"Empty prefix_ids for record_idx={ridx}, token_pos={pos}, "
            f"input_len={len(input_ids)}, output_len={len(output_ids)}"
        )

    # Use encode endpoint instead of generate
    payload = {
        "input_ids": prefix_ids,
    }

    last_err: Exception | None = None
    for attempt in range(max_hidden_state_retries + 1):
        try:
            resp = await client.post(f"{url}/encode", json=payload, timeout=timeout_s)
            if resp.status_code != 200:
                if _is_context_length_encode_error(resp):
                    return (task_idx, None)
                raise RuntimeError(
                    f"SGL /encode failed for record_idx={ridx}, token_pos={pos}. "
                    f"status={resp.status_code}, body={resp.text[:1000]}"
                )

            resp_data = resp.json()
            # encode endpoint typically returns embeddings directly or wrapped in "embeddings" key
            if isinstance(resp_data, dict):
                if "embeddings" in resp_data:
                    hidden_states = resp_data["embeddings"]
                elif "embedding" in resp_data:
                    hidden_states = resp_data["embedding"]
                else:
                    # Try to use the whole response as hidden states
                    raise RuntimeError(
                        f"Unexpected encode response format: {list(resp_data.keys())}"
                    )
            elif isinstance(resp_data, list):
                hidden_states = resp_data
            else:
                raise RuntimeError(
                    f"Unexpected encode response type: {type(resp_data)}"
                )

            debug_ctx = (
                f" (record_idx={ridx}, token_pos={pos}, prefix_len={len(prefix_ids)})"
            )
            hs = _extract_target_hidden(hidden_states, debug_context=debug_ctx)

            # Validate the returned hidden state
            if hs.numel() == 0:
                raise RuntimeError(
                    f"Got empty hidden state tensor for record_idx={ridx}, token_pos={pos}, "
                    f"prefix_len={len(prefix_ids)}"
                )

            return (task_idx, hs)
        except RuntimeError as err:
            retryable = "Unexpected hidden state shape" in str(
                err
            ) or "Missing hidden_states" in str(err)
            if not retryable or attempt >= max_hidden_state_retries:
                # Add more context before re-raising
                raise RuntimeError(
                    f"Failed to fetch hidden state for record_idx={ridx}, token_pos={pos}, "
                    f"prefix_len={len(prefix_ids)}: {err}"
                ) from err
            last_err = err
            await asyncio.sleep(0.05)
        except httpx.HTTPError as err:
            if attempt >= max_hidden_state_retries:
                raise RuntimeError(
                    f"Failed to fetch hidden state for record_idx={ridx}, token_pos={pos}, "
                    f"prefix_len={len(prefix_ids)}: transient HTTP error from {url}: {err}"
                ) from err
            last_err = err
            await asyncio.sleep(0.2 * (attempt + 1))

    if last_err is not None:
        raise last_err
    raise RuntimeError("Unreachable")


async def _run_sgl_batch(
    *,
    client: httpx.AsyncClient,
    url: str,
    records: list[dict[str, Any]],
    batch_tasks: list[dict[str, Any]],
    timeout_s: float,
    max_hidden_state_retries: int,
    max_concurrent: int = 16,
) -> list[torch.Tensor | None]:
    """Fetch hidden states for a batch of tasks using async concurrent encode requests."""
    semaphore = asyncio.Semaphore(max_concurrent)
    urls = [item.strip().rstrip("/") for item in url.split(",") if item.strip()]
    if not urls:
        raise ValueError("At least one SGL URL must be provided")

    async def fetch_with_semaphore(
        idx: int, task: dict[str, Any]
    ) -> tuple[int, torch.Tensor | None]:
        async with semaphore:
            return await _fetch_single_hidden_state(
                client=client,
                url=urls[idx % len(urls)],
                records=records,
                task=task,
                task_idx=idx,
                timeout_s=timeout_s,
                max_hidden_state_retries=max_hidden_state_retries,
            )

    tasks = [fetch_with_semaphore(idx, task) for idx, task in enumerate(batch_tasks)]

    results_list = await asyncio.gather(*tasks)
    results = {task_idx: hs for task_idx, hs in results_list}

    # Return in original order.
    return [results[i] for i in range(len(batch_tasks))]


def _get_dynamic_batch_size(queue_size: int, preferred_sizes: list[int]) -> int:
    """Get optimal batch size based on queue size using predefined preferred sizes."""
    for size in reversed(preferred_sizes):
        if queue_size >= size:
            return size
    return min(queue_size, preferred_sizes[0]) if queue_size > 0 else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decoupled run_probe using SGLang encode endpoint: include track token in input_ids, "
            "fetch hidden states from SGL /encode, then run local SingleProbeHead and write probs aligned with output_ids."
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
        help="Output directory for args.json and probed results.",
    )
    parser.add_argument(
        "--result-file-name",
        type=str,
        default="generation_results.probed.jsonl",
        help="Output probed result filename inside output directory.",
    )
    parser.add_argument(
        "--track-token-map",
        type=str,
        default="track_token_id2text.json",
        help="JSON file mapping token_id->decoded_text (only keys are used)",
    )
    parser.add_argument(
        "--track-token-stride",
        type=int,
        default=128,
        help="Keep at most one tracked token every N output tokens per record.",
    )
    parser.add_argument(
        "--head-model-path",
        type=str,
        required=True,
        help="Path to exported SingleProbeHead (config.json + model.safetensors)",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="",
        help="Base model path for lm_head extraction when head uses logit features",
    )
    parser.add_argument("--sgl-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--head-batch-size",
        type=int,
        default=256,
        help="Maximum batch size for local probe head inference (actual size determined dynamically).",
    )
    parser.add_argument(
        "--head-preferred-batch-sizes",
        type=str,
        default="2,4,8,16,32,64,128,256",
        help="Comma-separated list of preferred batch sizes for dynamic batching (e.g., '2,4,8,16,32').",
    )
    parser.add_argument(
        "--consumer-wait-timeout",
        type=float,
        default=0.1,
        help="Max time (seconds) consumer waits for queue items before processing current batch.",
    )
    parser.add_argument(
        "--queue-max-size",
        type=int,
        default=2048,
        help="Maximum queue size for producer-consumer pipeline.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--max-hidden-state-retries",
        type=int,
        default=2,
        help="Retry count for transient hidden_states issues from SGL.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Process only first N records for debug (0 means all)",
    )
    parser.add_argument(
        "--sgl-max-workers",
        type=int,
        default=512,
        help="Max concurrent async tasks for SGL encode requests per batch (default: 16)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for faster probe head inference (requires PyTorch >= 2.0)",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
        help="torch.compile mode: 'reduce-overhead' for latency, 'max-autotune' for throughput",
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
        help="fsync checkpoint every N completed positions (0 disables fsync).",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.head_batch_size <= 0:
        raise ValueError("--head-batch-size must be > 0")
    if args.max_hidden_state_retries < 0:
        raise ValueError("--max-hidden-state-retries must be >= 0")
    if args.track_token_stride <= 0:
        raise ValueError("--track-token-stride must be > 0")
    if args.sgl_max_workers <= 0:
        raise ValueError("--sgl-max-workers must be > 0")
    if args.consumer_wait_timeout < 0:
        raise ValueError("--consumer-wait-timeout must be >= 0")
    if args.queue_max_size <= 0:
        raise ValueError("--queue-max-size must be > 0")
    if not args.checkpoint_file_name.strip():
        raise ValueError("--checkpoint-file-name must be non-empty")
    if args.checkpoint_fsync_interval < 0:
        raise ValueError("--checkpoint-fsync-interval must be >= 0")

    # Parse preferred batch sizes
    try:
        preferred_batch_sizes = sorted(
            [int(x.strip()) for x in args.head_preferred_batch_sizes.split(",")]
        )
        if not preferred_batch_sizes or any(
            size <= 0 for size in preferred_batch_sizes
        ):
            raise ValueError("All preferred batch sizes must be > 0")
        if preferred_batch_sizes[-1] > args.head_batch_size:
            raise ValueError(
                f"Maximum preferred batch size ({preferred_batch_sizes[-1]}) cannot exceed --head-batch-size ({args.head_batch_size})"
            )
    except ValueError as e:
        raise ValueError(f"Invalid --head-preferred-batch-sizes: {e}") from e

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

    if not args.result_file_name:
        raise ValueError("--result-file-name must be a non-empty filename")
    output_path = out_dir / args.result_file_name
    checkpoint_path = out_dir / args.checkpoint_file_name

    track_token_ids = _load_track_token_ids(Path(args.track_token_map))
    records = load_jsonl(input_path)
    if args.max_records > 0:
        records = records[: args.max_records]
    if not records:
        raise ValueError("No records found in input JSONL")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    compile_options = None
    if args.compile:
        compile_options = {"mode": args.compile_mode, "dynamic_shapes": True}
        print(
            f"[INFO] Loading probe head with torch.compile (mode={args.compile_mode})..."
        )
    else:
        print("[INFO] Loading probe head without compilation...")

    head_model = SingleProbeHead.from_pretrained(
        args.head_model_path,
        device=device,
        compile=args.compile,
        compile_options=compile_options,
    )
    head_model.eval()

    if head_model.is_compiled():
        print(
            f"[INFO] ✓ Probe head compiled successfully with mode={args.compile_mode}"
        )
    elif args.compile:
        print("[WARNING] Compilation was requested but failed. Using eager execution.")

    if str(head_model.pooling_mode) == "segment_mean":
        raise ValueError(
            "Unsupported pooling_mode=segment_mean for decoupled run_probe. "
            "Current pipeline fetches one hidden state per position and only supports last_token head."
        )

    lm_head_weight: torch.Tensor | None = None
    if int(head_model.logit_feature_dim) > 0:
        if not args.base_model_path:
            raise ValueError(
                "Head model enables logit features, but --base-model-path is empty. "
                "Please provide base model path to load lm_head weight."
            )
        lm_head_weight = _extract_lm_head_weight(args.base_model_path).to(device=device)

    tasks = _build_track_tasks(
        records=records,
        track_token_ids=track_token_ids,
        track_token_stride=args.track_token_stride,
    )

    for record in records:
        output_ids = record.get("output_ids")
        if not isinstance(output_ids, list):
            raise ValueError("Each record must contain list[int] field: output_ids")
        record["probs"] = [-1.0] * len(output_ids)

    completed_task_keys: set[tuple[int, int]] = set()
    resumed_scored_positions = 0
    resumed_skipped_positions = 0
    if args.resume:
        (
            completed_task_keys,
            resumed_scored_positions,
            resumed_skipped_positions,
        ) = _load_probe_checkpoint(
            checkpoint_path=checkpoint_path,
            records=records,
        )
        if completed_task_keys:
            print(
                "Resuming probe encode from checkpoint: "
                f"{len(completed_task_keys)} / {len(tasks)} positions complete"
            )
    pending_tasks = [
        task
        for task in tasks
        if _task_key(int(task["record_idx"]), int(task["token_pos"]))
        not in completed_task_keys
    ]

    if not tasks:
        write_jsonl(output_path, records)
        merged_args = copy.deepcopy(original_args)
        merged_args["probe"] = {
            "input_dir": str(input_dir),
            "input_file": str(input_path),
            "output": str(out_dir),
            "result_file": args.result_file_name,
            "probe_model": args.head_model_path,
            "track_token_map": args.track_token_map,
            "track_token_stride": args.track_token_stride,
            "base_model_path": args.base_model_path or None,
            "sgl_url": args.sgl_url,
            "batch_size": args.batch_size,
            "head_batch_size": args.head_batch_size,
            "head_preferred_batch_sizes": args.head_preferred_batch_sizes,
            "consumer_wait_timeout": args.consumer_wait_timeout,
            "queue_max_size": args.queue_max_size,
            "timeout": args.timeout,
            "max_hidden_state_retries": args.max_hidden_state_retries,
            "device": args.device,
            "max_records": args.max_records,
            "track_token_ids": len(track_token_ids),
            "scored_track_positions": 0,
            "skipped_context_window_positions": 0,
            "resume": args.resume,
            "checkpoint_file": str(checkpoint_path),
            "checkpoint_fsync_interval": args.checkpoint_fsync_interval,
            "resumed_positions": 0,
        }
        dump_json(out_dir / "args.json", merged_args)

        print(f"Input records: {len(records)}")
        print(f"Track token ids: {len(track_token_ids)}")
        print("Scored track positions: 0")
        print("Skipped context-window positions: 0")
        print(f"Output directory: {out_dir}")
        print(f"Saved probed results to: {output_path}")
        return

    scored_positions = resumed_scored_positions
    skipped_context_window_positions = resumed_skipped_positions

    async def run_probe() -> None:
        nonlocal scored_positions, skipped_context_window_positions

        # Producer-consumer queue
        queue: asyncio.Queue[tuple[dict[str, Any], torch.Tensor | None] | None] = (
            asyncio.Queue(maxsize=args.queue_max_size)
        )
        checkpoint_handle = checkpoint_path.open("ab") if args.resume else None
        completed_count = len(completed_task_keys)

        def write_checkpoint(
            task: dict[str, Any],
            *,
            prob: float | None = None,
            skipped: bool = False,
        ) -> None:
            nonlocal completed_count
            completed_count += 1
            if checkpoint_handle is None:
                return
            _write_probe_checkpoint_row(
                handle=checkpoint_handle,
                task=task,
                prob=prob,
                skipped=skipped,
                completed_count=completed_count,
                fsync_interval=args.checkpoint_fsync_interval,
            )

        async def producer() -> None:
            """Producer: Fetch hidden states from SGL and put into queue."""
            async with httpx.AsyncClient() as client:
                with tqdm(
                    total=len(tasks),
                    initial=len(completed_task_keys),
                    desc="SGL encode (producer)",
                    unit="item",
                    position=0,
                ) as pbar:
                    for start in range(0, len(pending_tasks), args.batch_size):
                        batch_tasks = pending_tasks[start : start + args.batch_size]
                        token_hiddens = await _run_sgl_batch(
                            client=client,
                            url=args.sgl_url.rstrip("/"),
                            records=records,
                            batch_tasks=batch_tasks,
                            timeout_s=args.timeout,
                            max_hidden_state_retries=args.max_hidden_state_retries,
                            max_concurrent=args.sgl_max_workers,
                        )

                        # Put items into queue
                        for task, hidden in zip(
                            batch_tasks, token_hiddens, strict=True
                        ):
                            await queue.put((task, hidden))
                        pbar.update(len(batch_tasks))

            # Put sentinel to signal end of data
            await queue.put(None)

        async def consumer() -> None:
            """Consumer: Get items from queue with dynamic batching and run head inference."""
            nonlocal scored_positions

            with tqdm(
                total=len(tasks),
                initial=len(completed_task_keys),
                desc="Head inference (consumer)",
                unit="item",
                position=1,
            ) as pbar:
                while True:
                    batch_tasks: list[dict[str, Any]] = []
                    batch_hiddens: list[torch.Tensor] = []

                    # Try to get first item (blocking)
                    try:
                        item = await queue.get()
                    except asyncio.CancelledError:
                        break

                    if item is None:  # Sentinel value
                        break

                    task, hidden = item
                    if hidden is None:
                        skipped_context_window_positions += 1
                        write_checkpoint(task, skipped=True)
                        pbar.update(1)
                        continue
                    batch_tasks.append(task)
                    batch_hiddens.append(hidden)

                    # Collect more items from queue with dynamic batching
                    while len(batch_tasks) < args.head_batch_size:
                        # Determine target batch size based on queue size
                        current_queue_size = queue.qsize()
                        target_size = _get_dynamic_batch_size(
                            current_queue_size + len(batch_tasks), preferred_batch_sizes
                        )

                        # If we've reached a preferred batch size, process immediately
                        if (
                            len(batch_tasks) in preferred_batch_sizes
                            and len(batch_tasks) >= target_size
                        ):
                            break

                        # Try to get more items with timeout
                        try:
                            item = await asyncio.wait_for(
                                queue.get(), timeout=args.consumer_wait_timeout
                            )
                            if item is None:  # Sentinel
                                await queue.put(
                                    None
                                )  # Put it back for other consumers (if any)
                                break
                            task, hidden = item
                            if hidden is None:
                                skipped_context_window_positions += 1
                                write_checkpoint(task, skipped=True)
                                pbar.update(1)
                                continue
                            batch_tasks.append(task)
                            batch_hiddens.append(hidden)
                        except TimeoutError:
                            # No more items available within timeout, process current batch
                            break

                    # Process the batch
                    if batch_tasks:
                        with torch.no_grad():
                            hs_tensor = torch.stack(batch_hiddens, dim=0).to(
                                device=device, dtype=torch.float32
                            )
                            probs_tensor = head_model._score_from_state(
                                hs_tensor, lm_head_weight
                            ).squeeze(-1)
                            probs_np = (
                                probs_tensor.detach().cpu().numpy().astype(np.float64)
                            )

                        for task, prob in zip(batch_tasks, probs_np, strict=True):
                            ridx = int(task["record_idx"])
                            pos = int(task["token_pos"])
                            records[ridx]["probs"][pos] = float(prob)
                            scored_positions += 1
                            write_checkpoint(task, prob=float(prob))

                        pbar.update(len(batch_tasks))

        try:
            # Run producer and consumer concurrently
            await asyncio.gather(producer(), consumer())
        finally:
            if checkpoint_handle is not None:
                checkpoint_handle.flush()
                if args.checkpoint_fsync_interval > 0:
                    os.fsync(checkpoint_handle.fileno())
                checkpoint_handle.close()

    asyncio.run(run_probe())

    write_jsonl(output_path, records)

    merged_args = copy.deepcopy(original_args)
    merged_args["probe"] = {
        "input_dir": str(input_dir),
        "input_file": str(input_path),
        "output": str(out_dir),
        "result_file": args.result_file_name,
        "probe_model": args.head_model_path,
        "track_token_map": args.track_token_map,
        "track_token_stride": args.track_token_stride,
        "base_model_path": args.base_model_path or None,
        "sgl_url": args.sgl_url,
        "batch_size": args.batch_size,
        "head_batch_size": args.head_batch_size,
        "head_preferred_batch_sizes": args.head_preferred_batch_sizes,
        "consumer_wait_timeout": args.consumer_wait_timeout,
        "queue_max_size": args.queue_max_size,
        "timeout": args.timeout,
        "max_hidden_state_retries": args.max_hidden_state_retries,
        "device": args.device,
        "max_records": args.max_records,
        "track_token_ids": len(track_token_ids),
        "scored_track_positions": scored_positions,
        "skipped_context_window_positions": skipped_context_window_positions,
        "resume": args.resume,
        "checkpoint_file": str(checkpoint_path),
        "checkpoint_fsync_interval": args.checkpoint_fsync_interval,
        "resumed_positions": len(completed_task_keys),
        "resumed_scored_positions": resumed_scored_positions,
        "resumed_skipped_context_window_positions": resumed_skipped_positions,
    }
    dump_json(out_dir / "args.json", merged_args)

    print(f"Input records: {len(records)}")
    print(f"Track token ids: {len(track_token_ids)}")
    print(f"Scored track positions: {scored_positions}")
    print(f"Skipped context-window positions: {skipped_context_window_positions}")
    print(f"Output directory: {out_dir}")
    print(f"Saved probed results to: {output_path}")


if __name__ == "__main__":
    main()
