#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from cognitive_probe.evaluation.utils import (
    GenerationConfig,
    GenerationResult,
    load_evaluation_samples,
    prepare_experiment,
    save_all_results,
)
from cognitive_probe.utils.log_utils import init_loguru_logger


def build_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch generation against a running SGL /generate endpoint."
    )
    parser.add_argument(
        "--api-url",
        type=str,
        required=True,
        help=(
            "SGL server base URL. Provide a comma-separated list to use "
            "client-side round-robin across independent services."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Logical model name written into run metadata.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Tokenizer path used to build prompt input_ids.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Input benchmark JSONL path.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=1.4,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=20,
        help="Sampling top-k.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Sampling top-p.",
    )
    parser.add_argument(
        "-n",
        "--num-generation",
        type=int,
        default=32,
        help="Roll-outs per prompt.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=32768,
        help="Max new tokens per request.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="run",
        help="Experiment name under evaluation/results.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results",
        help="Output base directory.",
    )
    parser.add_argument(
        "--concurrency-limit",
        type=int,
        default=32,
        help="Max concurrent /generate requests.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=3000.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=2,
        help="Extra retry attempts per generation request.",
    )
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=2.0,
        help="Base retry backoff in seconds.",
    )
    parser.add_argument(
        "--eval-k",
        type=str,
        default="4,8,16,32",
        help="k values passed to run_eval.py after generation.",
    )
    parser.add_argument(
        "--no-run-eval",
        action="store_true",
        help="Skip automatic run_eval.py after generation.",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help=(
            "Resume a previous API generation run directory. The directory must "
            "contain args.json with saved seeds; completed rows in "
            "generation_results.jsonl are skipped."
        ),
    )
    return parser


def _prepare_args(args: argparse.Namespace) -> tuple[GenerationConfig, Path]:
    if args.resume_dir:
        return _prepare_resume_args(args)

    prepare_ns = Namespace(
        data=args.data,
        experiment_name=args.experiment_name,
        num_generation=args.num_generation,
        model=args.model_name,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        output=args.output,
    )
    gen_config, out_dir = prepare_experiment(prepare_ns, engine_kwargs=None)

    args_path = out_dir / "args.json"
    payload = json.loads(args_path.read_text(encoding="utf-8"))
    api_urls = _parse_api_urls(args.api_url)
    payload["api_generation"] = {
        "api_url": ",".join(api_urls),
        "api_urls": api_urls,
        "model_name": args.model_name,
        "tokenizer_path": args.tokenizer_path,
        "concurrency_limit": args.concurrency_limit,
        "timeout_s": args.timeout_s,
        "request_retries": args.request_retries,
        "retry_backoff_s": args.retry_backoff_s,
        "resume_dir": None,
    }
    payload["seeds"] = gen_config.seeds
    args_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return gen_config, out_dir


def _prepare_resume_args(args: argparse.Namespace) -> tuple[GenerationConfig, Path]:
    out_dir = Path(args.resume_dir).expanduser().resolve()
    args_path = out_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Cannot resume without args.json: {args_path}")

    payload = json.loads(args_path.read_text(encoding="utf-8"))
    seeds_raw = payload.get("seeds")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError(
            "Cannot resume this run reproducibly: args.json does not contain "
            "'seeds'. Start a new run with the updated script first."
        )

    data_path = str(payload.get("data") or args.data)
    samples = load_evaluation_samples(data_path)
    api_urls = _parse_api_urls(args.api_url)
    payload["api_generation"] = {
        "api_url": ",".join(api_urls),
        "api_urls": api_urls,
        "model_name": args.model_name,
        "tokenizer_path": args.tokenizer_path,
        "concurrency_limit": args.concurrency_limit,
        "timeout_s": args.timeout_s,
        "request_retries": args.request_retries,
        "retry_backoff_s": args.retry_backoff_s,
        "resume_dir": str(out_dir),
    }
    args_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return GenerationConfig(
        samples=samples,
        seeds=[int(seed) for seed in seeds_raw],
        model_path=str(payload.get("model") or args.model_name),
        temperature=float(payload["temperature"]),
        top_p=float(payload.get("top_p", 1.0)),
        top_k=int(payload["top_k"]),
        max_tokens=int(payload["max_length"]),
        engine_kwargs=payload.get("engine_kwargs"),
    ), out_dir


def _parse_api_urls(raw: str) -> list[str]:
    urls = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    if not urls:
        raise ValueError("--api-url must contain at least one URL")
    return urls


def _result_key(example_id: str | int, seed: int) -> tuple[str, int]:
    return str(example_id), int(seed)


def _load_existing_results(results_path: Path) -> dict[tuple[str, int], GenerationResult]:
    results: dict[tuple[str, int], GenerationResult] = {}
    if not results_path.exists():
        return results

    with results_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                result = GenerationResult(**row)
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse checkpoint row {line_no} in {results_path}: {exc}"
                ) from exc
            results[_result_key(result.example_id, result.seed)] = result
    return results


def _append_result(results_path: Path, result: GenerationResult) -> None:
    with results_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def _as_output_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        output_ids = payload.get("output_ids")
        if isinstance(output_ids, list):
            return [payload]

        text_list = payload.get("text")
        out_ids_list = payload.get("output_ids")
        meta_list = payload.get("meta_info")
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


async def generate_async(
    *,
    config: GenerationConfig,
    out_dir: Path,
    api_urls: list[str],
    tokenizer_path: str,
    timeout_s: float,
    concurrency_limit: int,
    request_retries: int,
    retry_backoff_s: float,
) -> list[GenerationResult]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    prompt_items: list[
        tuple[
            str | int,
            list[dict[str, str]],
            str,
            list[int],
            str,
            str,
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []
    for sample in config.samples:
        prompt = list(sample["prompt"])
        input_text = str(
            tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        input_ids = [
            int(token_id)
            for token_id in tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
            )
        ]
        prompt_items.append(
            (
                sample["example_id"],
                prompt,
                input_text,
                input_ids,
                str(sample["answer"]),
                str(sample.get("judge_type") or "math"),
                sample.get("judge_payload") if isinstance(sample.get("judge_payload"), dict) else {},
                sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {},
            )
        )

    endpoint_count = len(api_urls)
    per_endpoint_limit = max(
        1, (concurrency_limit + endpoint_count - 1) // endpoint_count
    )
    endpoint_semaphores = [
        asyncio.Semaphore(per_endpoint_limit) for _ in range(endpoint_count)
    ]
    timeout = httpx.Timeout(timeout_s)
    results_path = out_dir / "generation_results.jsonl"
    results_map = _load_existing_results(results_path)
    errors: list[str] = []
    request_prefix = f"api_{os.getpid()}_{time.time_ns():x}"
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:

        async def _run_one(
            request_index: int,
            example_id: str | int,
            prompt: list[dict[str, str]],
            input_text: str,
            input_ids: list[int],
            answer: str,
            judge_type: str,
            judge_payload: dict[str, Any],
            metadata: dict[str, Any],
            seed: int,
        ) -> tuple[str, int, GenerationResult]:
            max_attempts = request_retries + 1
            output: dict[str, Any] | None = None
            for attempt in range(max_attempts):
                endpoint_index = (request_index + attempt) % endpoint_count
                api_url = api_urls[endpoint_index]
                payload = {
                    "rid": (
                        f"{request_prefix}_{example_id}_seed{seed}"
                        f"_try{attempt}"
                    ),
                    "input_ids": input_ids,
                    "sampling_params": {
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "top_k": config.top_k,
                        "max_new_tokens": config.max_tokens,
                        "sampling_seed": seed,
                    },
                    "stream": False,
                }
                try:
                    async with endpoint_semaphores[endpoint_index]:
                        resp = await client.post(f"{api_url}/generate", json=payload)
                    if resp.status_code != 200:
                        raise RuntimeError(
                            "SGL /generate failed during baseline generation. "
                            f"endpoint={api_url}, status={resp.status_code}, "
                            f"body={resp.text[:1000]}"
                        )
                    items = _as_output_items(resp.json())
                    if len(items) != 1:
                        raise RuntimeError(
                            f"Expected single generation output, got {len(items)}"
                        )
                    output = items[0]
                    break
                except Exception as exc:
                    if attempt + 1 >= max_attempts:
                        raise
                    logger.warning(
                        "Generation request failed; retrying. "
                        "example_id={} seed={} endpoint={} attempt={}/{} error={}",
                        example_id,
                        seed,
                        api_url,
                        attempt + 1,
                        max_attempts,
                        repr(exc),
                    )
                    await asyncio.sleep(retry_backoff_s * (attempt + 1))
            if output is None:
                raise RuntimeError(
                    f"Generation request failed without output: "
                    f"example_id={example_id!r}, seed={seed}"
                )
            output_ids_raw = output.get("output_ids")
            if not isinstance(output_ids_raw, list):
                raise ValueError(
                    f"Missing output_ids for example_id={example_id!r}, seed={seed}"
                )
            output_ids = [int(token_id) for token_id in output_ids_raw]
            finish_reason_raw = output.get("meta_info", {}).get("finish_reason")
            if isinstance(finish_reason_raw, dict):
                finish_reason = finish_reason_raw.get("type")
            else:
                finish_reason = finish_reason_raw
            result = GenerationResult(
                example_id=example_id,
                prompt=prompt,
                input_text=input_text,
                input_ids=input_ids,
                answer=answer,
                seed=seed,
                response=str(output.get("text") or ""),
                output_ids=output_ids,
                response_len_tokens=len(output_ids),
                probs=[],
                finish_reason=str(finish_reason) if finish_reason is not None else None,
                judge_type=judge_type,
                judge_payload=judge_payload,
                metadata=metadata,
            )
            return str(example_id), seed, result

        request_specs = [
            (
                example_id,
                prompt,
                input_text,
                input_ids,
                answer,
                judge_type,
                judge_payload,
                metadata,
                seed,
            )
            for (
                example_id,
                prompt,
                input_text,
                input_ids,
                answer,
                judge_type,
                judge_payload,
                metadata,
            ) in prompt_items
            for seed in config.seeds
        ]
        expected_keys = {
            _result_key(request_spec[0], request_spec[-1])
            for request_spec in request_specs
        }
        stale_checkpoint_keys = set(results_map) - expected_keys
        if stale_checkpoint_keys:
            logger.warning(
                "Ignoring {} checkpoint rows that do not match this run.",
                len(stale_checkpoint_keys),
            )
            results_map = {
                key: result
                for key, result in results_map.items()
                if key in expected_keys
            }
        completed_before_start = len(results_map)
        if completed_before_start:
            logger.info(
                "Loaded {} completed generation checkpoints from {}",
                completed_before_start,
                results_path,
            )
        pending_specs = [
            request_spec
            for request_spec in request_specs
            if _result_key(request_spec[0], request_spec[-1]) not in results_map
        ]
        if not pending_specs:
            logger.info("All {} generation requests are already complete.", len(request_specs))

        coros = [
            _run_one(request_index, *request_spec)
            for request_index, request_spec in enumerate(pending_specs)
        ]
        for fut in tqdm(
            asyncio.as_completed(coros),
            total=len(request_specs),
            initial=len(request_specs) - len(pending_specs),
            desc="Generating",
        ):
            try:
                example_id_key, seed, result = await fut
            except Exception as exc:
                logger.exception("Generation request failed after retries: {}", exc)
                errors.append(repr(exc))
                continue
            results_map[_result_key(example_id_key, seed)] = result
            _append_result(results_path, result)

    if errors:
        preview = "\n".join(errors[:5])
        raise RuntimeError(
            f"{len(errors)} generation requests failed after retries. "
            f"First failures:\n{preview}"
        )

    ordered_results: list[GenerationResult] = []
    for example_id, _prompt, _input_text, _input_ids, _answer, _judge_type, _judge_payload, _metadata in prompt_items:
        example_id_key = str(example_id)
        for seed in config.seeds:
            ordered_results.append(results_map[(example_id_key, seed)])
    return ordered_results


def run_eval(out_dir: Path, eval_k: str) -> None:
    run_eval_path = Path(__file__).resolve().parents[1] / "analysis" / "run_eval.py"
    subprocess.run(
        [
            sys.executable,
            str(run_eval_path),
            "--dir",
            str(out_dir),
            "--k",
            eval_k,
        ],
        check=True,
    )


def main() -> int:
    init_loguru_logger()
    args = build_arguments().parse_args()
    gen_config, out_dir = _prepare_args(args)
    api_urls = _parse_api_urls(args.api_url)

    logger.info("Generating via {}", ", ".join(api_urls))
    logger.info(
        "Endpoint count: {} | total concurrency: {} | per-endpoint concurrency: {}",
        len(api_urls),
        args.concurrency_limit,
        max(1, (args.concurrency_limit + len(api_urls) - 1) // len(api_urls)),
    )
    logger.info("Logical model name: {}", args.model_name)
    logger.info("Tokenizer path: {}", args.tokenizer_path)
    logger.info("Output directory: {}", out_dir)

    results = asyncio.run(
        generate_async(
            config=gen_config,
            out_dir=out_dir,
            api_urls=api_urls,
            tokenizer_path=args.tokenizer_path,
            timeout_s=args.timeout_s,
            concurrency_limit=args.concurrency_limit,
            request_retries=args.request_retries,
            retry_backoff_s=args.retry_backoff_s,
        )
    )
    save_all_results(out_dir=out_dir, all_results=results)

    if not args.no_run_eval:
        run_eval(out_dir=out_dir, eval_k=args.eval_k)

    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
