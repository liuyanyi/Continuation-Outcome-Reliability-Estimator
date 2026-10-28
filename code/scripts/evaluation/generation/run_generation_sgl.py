#!/usr/bin/env python3
"""
Batch inference on our standard JSONL dataset (BenchmarkData), single-process.

Highlights:
- No global state: all configs passed explicitly.
- Output is a directory containing:
  - args.json    (all run configs)
  - output.jsonl (generation results)

Input JSONL format per line:
{
  "id": str,
  "source": str,
  "index": int,
  "prompt": [{"role": str, "content": str}, ...],
  "solution": str | list | number
}
"""

from __future__ import annotations

import asyncio
import json
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from cognitive_probe.evaluation.utils import (
    AsyncGenerator,
    GenerationConfig,
    GenerationResult,
    build_evaluation_arguments,
    load_evaluation_samples,
    parse_extra_engine_args,
    prepare_experiment,
    save_all_results,
)
from cognitive_probe.utils.cuda_utils import is_volta
from cognitive_probe.utils.log_utils import init_loguru_logger

# debugpy.connect(5678)

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


def extend_args_with_sgl(parser: ArgumentParser) -> ArgumentParser:
    # sgl specific startup arguments (pass-through)
    sgl_group = parser.add_argument_group("sgl", "sgl startup parameters")
    sgl_group.add_argument(
        "--sgl-args",
        nargs="*",
        default=[],
        help=(
            "Pass-through sgl startup args. "
            "Provide flags or key=val pairs, e.g. --sgl-args --numa-off --log-verbose"
        ),
    )
    # explicit parallelism args for sgl: tensor, pipeline, data
    sgl_group.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Tensor parallel size (will be passed as tensor_parallel_size to LLM)",
    )
    sgl_group.add_argument(
        "--pp",
        type=int,
        default=None,
        help="Pipeline parallel size (will be passed as pipeline_parallel_size to LLM)",
    )
    sgl_group.add_argument(
        "--dp",
        type=int,
        default=None,
        help="Data parallel size (will be passed as data_parallel_size to LLM)",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help=(
            "Resume a previous in-process SGL generation run directory. The "
            "directory must contain args.json with saved seeds; completed rows "
            "in generation_results.jsonl are skipped."
        ),
    )
    return parser


def _prepare_run(
    args,
    parsed_sgl_kwargs: dict[str, Any],
) -> tuple[GenerationConfig, Path]:
    if not args.resume_dir:
        gen_config, out_dir = prepare_experiment(
            args,
            engine_kwargs=parsed_sgl_kwargs,
        )
        args_path = out_dir / "args.json"
        payload = json.loads(args_path.read_text(encoding="utf-8"))
        payload["seeds"] = gen_config.seeds
        args_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return gen_config, out_dir

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
    samples = load_evaluation_samples(str(payload.get("data") or args.data))
    payload["resume_dir"] = str(out_dir)
    args_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return GenerationConfig(
        samples=samples,
        seeds=[int(seed) for seed in seeds_raw],
        model_path=str(payload.get("model") or args.model),
        temperature=float(payload["temperature"]),
        top_p=float(payload.get("top_p", 1.0)),
        top_k=int(payload["top_k"]),
        max_tokens=int(payload["max_length"]),
        engine_kwargs=payload.get("engine_kwargs"),
    ), out_dir


class SGLGenerator(AsyncGenerator):
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir

    async def generate_async(
        self,
        config: GenerationConfig,
        # samples: list[dict[str, Any]],
        # seeds: list[int],
        # model_path: str,
        # temperature: float,
        # top_p: float,
        # top_k: int,
        # max_tokens: int,
        # sgl_kwargs: dict[str, Any] | None = None,
    ) -> list[GenerationResult]:
        samples = config.samples
        seeds = config.seeds
        model_path = config.model_path
        temperature = config.temperature
        top_p = config.top_p
        top_k = config.top_k
        max_tokens = config.max_tokens
        sgl_kwargs = config.engine_kwargs

        from sglang.srt.entrypoints.engine import Engine
        from sglang.srt.server_args import ServerArgs

        logger.info("Loading model from {}", model_path)

        if is_volta():
            raise RuntimeError("Volta GPUs are not supported by sglang backend.")

        merged = dict(sgl_kwargs or {})
        server_args = ServerArgs(
            model_path=model_path,
            **merged,
        )
        llm = Engine(server_args=server_args)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer = cast(PreTrainedTokenizerBase, tokenizer)

        # Attempt to send all requests in one batch by duplicating messages for each seed
        # and passing a list of SamplingParams (one per request). If the vllm API
        # doesn't accept a list of SamplingParams, fall back to the per-seed loop.
        try:
            # The above is a trick to satisfy type checkers; rebuild properly:
            # all_messages = [s["prompt"] for _ in seeds for s in samples]
            prompt_items: list[
                tuple[str, list[dict[str, str]], str, list[int], str, dict, dict]
            ] = [
                (
                    str(s["example_id"]),
                    cast(list[dict[str, str]], s["prompt"]),
                    cast(
                        str,
                        tokenizer.apply_chat_template(
                            s["prompt"],
                            tokenize=False,
                            add_generation_prompt=True,
                        ),
                    ),
                    cast(
                        list[int],
                        tokenizer.apply_chat_template(
                            s["prompt"],
                            tokenize=True,
                            add_generation_prompt=True,
                        ),
                    ),
                    str(s.get("judge_type") or "math"),
                    s.get("judge_payload") if isinstance(s.get("judge_payload"), dict) else {},
                    s.get("metadata") if isinstance(s.get("metadata"), dict) else {},
                )
                for s in samples
            ]
            results_path = self.out_dir / "generation_results.jsonl"
            results_map = _load_existing_results(results_path)

            # repeat for each seed
            all_sampling: list[dict[str, Any]] = []
            all_indexes: list[str] = []
            all_prompts_repeated: list[str] = []
            rid_lookup: dict[str, tuple[str, int]] = {}
            # build all sampling params
            for seed in seeds:
                for example_id, _prompt_messages, input_text, _input_ids, _judge_type, _judge_payload, _metadata in prompt_items:
                    if _result_key(example_id, seed) in results_map:
                        continue
                    rid = f"{example_id}_seed{seed}"
                    all_indexes.append(f"{example_id}_seed{seed}")
                    all_prompts_repeated.append(input_text)
                    rid_lookup[rid] = (example_id, int(seed))
                    all_sampling.append(
                        {
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": top_k,
                            "max_new_tokens": max_tokens,
                            "sampling_seed": seed,
                        }
                    )

            expected_keys = {
                _result_key(example_id, seed)
                for example_id, *_rest in prompt_items
                for seed in seeds
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
            if results_map:
                logger.info(
                    "Loaded {} completed generation checkpoints from {}",
                    len(results_map),
                    results_path,
                )

            all_futures = []
            for pmt, samp, idx in zip(all_prompts_repeated, all_sampling, all_indexes):
                all_futures.append(
                    llm.async_generate(
                        prompt=pmt,
                        sampling_params=samp,
                        rid=idx,
                    )
                )

            prompt_lookup: dict[str, tuple[list[dict[str, str]], str, list[int], str, dict, dict]] = {
                example_id: (
                    prompt_messages,
                    input_text,
                    input_ids,
                    judge_type,
                    judge_payload,
                    metadata,
                )
                for (
                    example_id,
                    prompt_messages,
                    input_text,
                    input_ids,
                    judge_type,
                    judge_payload,
                    metadata,
                ) in prompt_items
            }
            answer_lookup = {str(sample["example_id"]): sample["answer"] for sample in samples}

            for future in tqdm(
                asyncio.as_completed(all_futures),
                total=len(samples) * len(seeds),
                initial=len(results_map),
                desc="Generating",
            ):
                output = await future
                out_id = str(output["meta_info"]["id"])
                example_id, seed = rid_lookup[out_id]
                (
                    prompt_messages,
                    input_text,
                    input_ids,
                    judge_type,
                    judge_payload,
                    metadata,
                ) = prompt_lookup[example_id]
                result = GenerationResult(
                    example_id=example_id,
                    prompt=prompt_messages,
                    input_text=input_text,
                    input_ids=input_ids,
                    answer=answer_lookup[example_id],
                    seed=seed,
                    response=output["text"],
                    output_ids=output["output_ids"],
                    finish_reason=output["meta_info"].get("finish_reason", {}).get("type", None),
                    response_len_tokens=len(output["output_ids"]),
                    judge_type=judge_type,
                    judge_payload=judge_payload,
                    metadata=metadata,
                )
                results_map[_result_key(example_id, seed)] = result
                _append_result(results_path, result)

            # Reconstruct results
            results: list[GenerationResult] = []

            for sample in samples:
                example_id = str(sample["example_id"])
                for seed in seeds:
                    results.append(results_map[_result_key(example_id, seed)])
        except TypeError as e:
            # Fallback: call per-seed as before
            logger.exception(e)
            return []
        finally:
            llm.shutdown()
        return results


def main() -> int:
    init_loguru_logger()
    parser = build_evaluation_arguments()
    parser = extend_args_with_sgl(parser)
    args = parser.parse_args()

    # Parse sgl args and pass as kwargs to LLM
    parsed_sgl_kwargs = parse_extra_engine_args(
        getattr(args, "sgl_args", []),
        blacklist={"model", "tp_size", "pp_size", "dp_size"},
    )

    # If explicit parallelism args provided, they override any sgl-args
    if getattr(args, "tp", None) is not None:
        parsed_sgl_kwargs["tp_size"] = int(args.tp)
    if getattr(args, "pp", None) is not None:
        parsed_sgl_kwargs["pp_size"] = int(args.pp)
    if getattr(args, "dp", None) is not None:
        parsed_sgl_kwargs["dp_size"] = int(args.dp)

    gen_config, out_dir = _prepare_run(args, parsed_sgl_kwargs)

    generator = SGLGenerator(out_dir=out_dir)

    results = asyncio.run(generator.generate_async(gen_config))

    logger.info("Total generations collected: {}", len(results))
    save_all_results(out_dir=out_dir, all_results=results)

    return 0


if __name__ == "__main__":
    main()
