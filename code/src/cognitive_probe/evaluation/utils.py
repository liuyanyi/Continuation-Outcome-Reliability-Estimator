import argparse
import random
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import orjson
from loguru import logger

from cognitive_probe.evaluation.v2 import migrate_legacy_run


def parse_extra_engine_args(
    arg_list: list[str], blacklist: set[str] | None = None
) -> dict[str, Any]:
    """Parse a list of pass-through sgl args into a kwargs dict.

    Rules:
    - items like key=val become {key: typed_val}
    - items starting with '-' or '--' are treated as flags and set to True
    - hyphens in keys are converted to underscores
    - values are cast to int/float/bool/None when possible
    """

    def _cast_val(v: str):
        if v.lower() in {"true", "false"}:
            return v.lower() == "true"
        if v.lower() == "none":
            return None
        # int then float
        try:
            iv = int(v)
            return iv
        except Exception:
            pass
        try:
            fv = float(v)
            return fv
        except Exception:
            pass
        return v

    kwargs: dict[str, Any] = {}
    if not arg_list:
        return kwargs
    for item in arg_list:
        if not item:
            continue
        # flag like --numa-off or -f -> set numa_off=True or f=True
        if item.startswith("--") or (item.startswith("-") and "=" not in item):
            key = item.lstrip("-").replace("-", "_")
            kwargs[key] = True
            continue
        if "=" in item:
            key, val = item.split("=", 1)
            key = key.lstrip("-").replace("-", "_")
            kwargs[key] = _cast_val(val)
            continue
        # bare word without '=': treat as flag (set True)
        key = item.replace("-", "_")
        kwargs[key] = True

    if blacklist:
        forbidden_keys = set(kwargs.keys()).intersection(blacklist)
        if forbidden_keys:
            raise ValueError(f"Forbidden extra engine args found: {forbidden_keys}")
    return kwargs


def build_evaluation_arguments() -> ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch generation on JSONL BenchmarkData"
    )
    parser.add_argument("--model", type=str, required=True, help="vLLM model path")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="输入 JSONL 数据路径 (BenchmarkData)",
    )
    parser.add_argument(
        "-t", "--temperature", type=float, default=1.4, help="temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0, help="top-p"
    )
    parser.add_argument("-k", "--top-k", type=int, default=20, help="top-k")
    parser.add_argument(
        "-n", "--num-generation", type=int, default=32, help="roll-outs per prompt"
    )
    parser.add_argument("--max_length", type=int, default=90000, help="max new tokens")
    parser.add_argument(
        "--experiment_name", type=str, default="run", help="experiment name"
    )
    parser.add_argument(
        "--output", type=str, default="evaluation/results", help="output base directory"
    )
    return parser


def prepare_experiment(
    args: Namespace,
    engine_kwargs: dict[str, Any] | None = None,
) -> tuple["GenerationConfig", Path]:
    data_path = args.data
    if not Path(data_path).exists():
        raise FileNotFoundError(f"{data_path} does not exist")

    name = args.experiment_name
    n = args.num_generation
    model_path = args.model
    max_tokens = args.max_length
    temperature = args.temperature
    top_p = getattr(args, "top_p", 1.0)
    top_k = args.top_k
    dataset_name = Path(data_path).stem

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = Path(f"{args.output}/{name}/{dataset_name}/{timestamp}")
    if out_dir.exists():
        logger.error(
            "Output directory {} already exists; aborting to avoid overwrite.", out_dir
        )
        raise FileExistsError(f"Output directory {out_dir} already exists")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Outputs will be saved to {}", out_dir)

    samples = load_evaluation_samples(data_path)

    logger.info("Starting generation on dataset: {}", data_path)

    args_json = {
        "model": model_path,
        "data": data_path,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_generation": n,
        "max_length": max_tokens,
        "experiment_name": name,
        "output": str(out_dir),
        "dataset_name": dataset_name,
        "timestamp": timestamp,
        "engine_kwargs": engine_kwargs,
    }

    with (out_dir / "args.json").open("w", encoding="utf-8") as f:
        f.write(
            orjson.dumps(
                args_json, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE
            ).decode("utf-8")
        )

    random_seeds = random.sample(range(2**31 - 1), n)
    logger.info("Prepared {} seeds", len(random_seeds))

    return GenerationConfig(
        samples=samples,
        seeds=random_seeds,
        model_path=model_path,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        engine_kwargs=engine_kwargs,
    ), out_dir


def save_all_results(
    out_dir: Path,
    all_results: list["GenerationResult"],
) -> None:
    results_path = out_dir / "generation_results.jsonl"
    with results_path.open("wb") as f:
        for result in all_results:
            result_json = asdict(result)
            f.write(orjson.dumps(result_json, option=orjson.OPT_APPEND_NEWLINE))
    logger.info("Saved all generation results to {}", results_path)
    try:
        migrate_legacy_run(out_dir, force=True)
        logger.info("Wrote V2 evaluation artifacts to {}", out_dir)
    except Exception as exc:
        logger.warning("Failed to write V2 evaluation artifacts for {}: {}", out_dir, exc)


def load_evaluation_samples(filepath: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = orjson.loads(line)
            samples.append(
                {
                    "example_id": item.get("id", i),
                    "prompt": item.get("prompt", []),
                    "answer": item.get("solution"),
                    "judge_type": item.get("judge_type", "math"),
                    "judge_payload": item.get("judge_payload", {}),
                    "metadata": item.get("metadata", {}),
                }
            )
    logger.info("Loaded {} samples from {}", len(samples), filepath)
    return samples


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for generation - eliminates parameter duplication."""

    samples: list[dict[str, Any]]
    seeds: list[int]
    model_path: str
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    engine_kwargs: dict[str, Any] | None = None


class SyncGenerator(Protocol):
    """Protocol for synchronous generators.

    Implement this protocol to create a synchronous generator.
    No inheritance required - just implement the interface.
    """

    @property
    def is_async(self) -> bool:
        return False

    def generate(self, config: GenerationConfig) -> list[dict[str, Any]]:
        """Generate outputs synchronously."""
        ...


@dataclass
class GenerationResult:
    # {
    #     "example_id": sample["example_id"],
    #     "prompt": sample["prompt"],
    #     "answer": sample["answer"],
    #     "seed": seed,
    #     "response": response["text"],
    #     "response_len_tokens": len(response["output_ids"]),
    # }
    example_id: str | int
    # Original chat messages from dataset sample["prompt"].
    prompt: list[dict[str, str]]
    # Flattened chat template text fed into the generation engine.
    input_text: str
    # Token ids of input_text from tokenizer.apply_chat_template(..., tokenize=True).
    input_ids: list[int]
    answer: str
    seed: int
    response: str
    output_ids: list[int]
    response_len_tokens: int
    # 能计算的地方是0~1，不能计算的位置是-1
    probs: list[float] = field(default_factory=list)
    finish_reason: str | None = None
    judge_type: str = "math"
    judge_payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class AsyncGenerator(Protocol):
    """Protocol for asynchronous generators.

    Implement this protocol to create an asynchronous generator.
    No inheritance required - just implement the interface.
    """

    @property
    def is_async(self) -> bool:
        return True

    async def generate_async(self, config: GenerationConfig) -> list[GenerationResult]:
        """Generate outputs asynchronously."""
        ...
