#!/usr/bin/env python3
"""Encode each prefix once and share features across compatible probe heads.

--heads-json maps run names to exported head directories. Each run is written
under output-dir/name using the existing single-head result format. A shared
journal commits all heads together and supports safe restart after interruption.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import httpx
import orjson
import torch
import torch.nn.functional as F

from cognitive_probe.evaluation.io import load_jsonl, write_jsonl
from cognitive_probe.probe_model.probe_model import SingleProbeHead
from scripts.evaluation.prob.run_probe_encode import (
    _build_probe_tasks,
    _extract_lm_head_weight,
    _iter_probe_batches,
    _load_original_args,
    _load_track_token_map,
    _resolve_input_file,
    _run_sgl_batch,
)


def fingerprint(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check_compatible(heads, base_config):
    keys = (
        "pooling_mode",
        "use_hidden_state_feature",
        "normalize_hidden_states_before_head",
        "use_logit_margin_feature",
        "use_logit_entropy_feature",
        "logit_feature_dim",
    )
    first = next(iter(heads.values()))
    signature = tuple(getattr(first, k) for k in keys)
    if first.pooling_mode != "last_token":
        raise ValueError("Shared encoding currently supports last_token only")
    for name, head in heads.items():
        if tuple(getattr(head, k) for k in keys) != signature:
            raise ValueError(f"{name}: incompatible feature configuration")
        for key in ("hidden_size", "model_type", "vocab_size"):
            if getattr(head.config, key) != base_config[key]:
                raise ValueError(f"{name}: incompatible backbone {key}")
    return first


def shared_probabilities(heads, hidden, lm_weight):
    first = next(iter(heads.values()))
    features = []
    if first.use_hidden_state_feature:
        features.append(
            F.layer_norm(hidden, hidden.shape[-1:])
            if first.normalize_hidden_states_before_head
            else hidden
        )
    if first.logit_feature_dim:
        features.append(first._compute_logit_features(hidden, lm_weight))
    features = features[0] if len(features) == 1 else torch.cat(features, dim=-1)
    return {
        name: torch.sigmoid(head.score(features)).squeeze(-1)
        for name, head in heads.items()
    }


def read_journal(path, names, valid_keys):
    """Reject invalid commits; truncate only an interrupted final write."""
    completed = {}
    if not path.exists():
        return completed
    with path.open("rb+") as f:
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                f.truncate(start)
                break
            row = orjson.loads(line)
            key = (row["record_idx"], row["token_pos"])
            if key not in valid_keys or key in completed:
                raise ValueError(f"Invalid/duplicate checkpoint key {key}")
            if not row["skipped"]:
                if set(row["probs"]) != set(names):
                    raise ValueError("Checkpoint head names mismatch")
                if any(not 0 <= p <= 1 for p in row["probs"].values()):
                    raise ValueError("Invalid probability in checkpoint")
            completed[key] = row
    return completed


async def evaluate(args):
    start = time.monotonic()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = json.loads(Path(args.heads_json).read_text())
    reserved = {".", "..", ".lock", "manifest.json", "checkpoint.jsonl", "summary.json", "reference"}
    if not paths or any(not n or Path(n).name != n or n in reserved for n in paths):
        raise ValueError("Head names must be nonempty directory names and must not use reserved output names")
    input_path = _resolve_input_file(Path(args.input_dir))
    records = load_jsonl(input_path)
    if not records:
        raise ValueError("No records found in input JSONL")
    if args.max_records:
        records = records[: args.max_records]
    tasks = _build_probe_tasks(
        records,
        probe_point_mode=args.probe_point_mode,
        track_token_map=_load_track_token_map(Path(args.track_token_map)),
        track_token_stride=args.track_token_stride,
    )
    manifest = dict(
        input_sha256=fingerprint(input_path),
        heads={
            n: {
                "path": str(Path(p).resolve()),
                "config_sha256": fingerprint(Path(p) / "config.json"),
                "weights_sha256": fingerprint(Path(p) / "model.safetensors"),
            }
            for n, p in paths.items()
        },
        base_model=str(Path(args.base_model_path).resolve()),
        base_config_sha256=fingerprint(Path(args.base_model_path) / "config.json"),
        track_map_sha256=fingerprint(args.track_token_map),
        mode=args.probe_point_mode,
        stride=args.track_token_stride,
        max_records=args.max_records,
        verify_single_head=args.verify_single_head,
        tasks=len(tasks),
    )
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume manifest mismatch: inputs/heads/settings changed")
    elif any(p.name != ".lock" for p in out.iterdir()):
        raise ValueError("Nonempty output directory has no manifest")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    valid_keys = {(t["record_idx"], t["token_pos"]) for t in tasks}
    completed = read_journal(out / "checkpoint.jsonl", paths, valid_keys)
    resumed = len(completed)
    pending = [t for t in tasks if (t["record_idx"], t["token_pos"]) not in completed]
    verification_max_error = {name: 0.0 for name in paths}
    old_summary = out / "summary.json"
    if old_summary.exists():
        verification_max_error.update(
            json.loads(old_summary.read_text()).get("max_probability_error", {})
        )
    encoded = 0
    if pending:
        heads = {
            n: SingleProbeHead.from_pretrained(
                p, device=torch.device(args.device), compile=False
            ).eval()
            for n, p in paths.items()
        }
        first = check_compatible(
            heads, json.loads((Path(args.base_model_path) / "config.json").read_text())
        )
        lm_weight = (
            _extract_lm_head_weight(args.base_model_path).to(args.device)
            if first.logit_feature_dim
            else None
        )
        queue = asyncio.Queue(maxsize=2)

        async def producer():
            nonlocal encoded
            async with httpx.AsyncClient() as client:
                for batch in _iter_probe_batches(
                    pending, batch_size=args.batch_size, interleave_records=True
                ):
                    hidden = await _run_sgl_batch(
                        client=client,
                        url=args.sgl_url.rstrip("/"),
                        records=records,
                        batch_tasks=batch,
                        timeout_s=300,
                        max_hidden_state_retries=2,
                        max_concurrent=args.batch_size,
                    )
                    encoded += len(batch)
                    await queue.put((batch, hidden))
            await queue.put(None)

        async def consumer():
            with (out / "checkpoint.jsonl").open("ab") as journal:
                while (item := await queue.get()) is not None:
                    batch, hidden = item
                    positions = [i for i, h in enumerate(hidden) if h is not None]
                    probs, references = {}, {}
                    if positions:
                        with torch.no_grad():
                            tensor = torch.stack([hidden[i] for i in positions]).to(
                                args.device, dtype=torch.float32
                            )
                            result = shared_probabilities(heads, tensor, lm_weight)
                            for name, head in heads.items():
                                if not torch.isfinite(result[name]).all():
                                    raise ValueError(
                                        f"{name}: non-finite probabilities"
                                    )
                                if args.verify_single_head:
                                    ref = head._score_from_state(
                                        tensor, lm_weight
                                    ).squeeze(-1)
                                    torch.testing.assert_close(
                                        result[name], ref, rtol=1e-6, atol=1e-7
                                    )
                                    verification_max_error[name] = max(
                                        verification_max_error[name],
                                        float((result[name] - ref).abs().max()),
                                    )
                                    references[name] = dict(
                                        zip(positions, ref.cpu().tolist())
                                    )
                                probs[name] = dict(
                                    zip(positions, result[name].cpu().tolist())
                                )
                    for i, task in enumerate(batch):
                        row = {**task, "skipped": hidden[i] is None}
                        if not row["skipped"]:
                            row["probs"] = {n: probs[n][i] for n in paths}
                            if args.verify_single_head:
                                row["references"] = {n: references[n][i] for n in paths}
                        journal.write(orjson.dumps(row) + b"\n")
                        completed[(task["record_idx"], task["token_pos"])] = row
                    journal.flush()
                    os.fsync(journal.fileno())
                    print(
                        f"{len(completed)}/{len(tasks)} prefixes; shared by {len(paths)} heads",
                        flush=True,
                    )

        async with asyncio.TaskGroup() as group:
            group.create_task(producer())
            group.create_task(consumer())
    assert len(completed) == len(tasks)
    if args.verify_single_head:
        for row in completed.values():
            if not row["skipped"]:
                for name in paths:
                    verification_max_error[name] = max(
                        verification_max_error[name],
                        abs(row["probs"][name] - row["references"][name]),
                    )
    original = _load_original_args(Path(args.input_dir))
    for name, path in paths.items():
        for reference in [False, True] if args.verify_single_head else [False]:
            dest = out / ("reference" if reference else "") / name
            dest.mkdir(parents=True, exist_ok=True)
            output = [{**r, "probs": [-1.0] * len(r["output_ids"])} for r in records]
            for (ridx, pos), row in completed.items():
                if not row["skipped"]:
                    output[ridx]["probs"][pos] = row[
                        "references" if reference else "probs"
                    ][name]
            temp = dest / "generation_results.probed.jsonl.tmp"
            write_jsonl(temp, output)
            temp.replace(dest / "generation_results.probed.jsonl")
            metadata = copy.deepcopy(original)
            metadata["probe"] = dict(
                probe_model=path,
                input_dir=args.input_dir,
                base_model_path=args.base_model_path,
                probe_point_mode=args.probe_point_mode,
                track_token_stride=args.track_token_stride,
                track_token_map=args.track_token_map,
                shared_manifest=str(manifest_path.resolve()),
                scored_track_positions=sum(
                    not r["skipped"] for r in completed.values()
                ),
                skipped_context_window_positions=sum(
                    r["skipped"] for r in completed.values()
                ),
            )
            (dest / "args.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = dict(
        records=len(records),
        tasks=len(tasks),
        heads=list(paths),
        resumed_positions=resumed,
        encoded_positions_this_run=encoded,
        elapsed_seconds=time.monotonic() - start,
        max_probability_error=verification_max_error,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("input-dir", "output-dir", "heads-json", "base-model-path", "sgl-url"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--track-token-map", default="track_token_id2text.json")
    p.add_argument(
        "--probe-point-mode", choices=["track_token", "stride"], default="track_token"
    )
    p.add_argument("--track-token-stride", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--verify-single-head", action="store_true")
    args = p.parse_args()
    if args.batch_size <= 0 or args.track_token_stride <= 0 or args.max_records < 0:
        p.error("batch-size and track-token-stride must be positive; max-records must be nonnegative")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
