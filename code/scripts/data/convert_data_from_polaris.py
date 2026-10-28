"""
将 POLARIS 基准数据（parquet，经由 HF datasets 读取）转换为标准 jsonl。

按需精简：
- 仅两个参数：-i 输入 parquet，-o 输出 jsonl
- 使用 loguru 记录日志
- 自动根据输入文件名推断 source（如 aime24.parquet -> "aime24"）

示例：
    python scripts/data/convert_data_from_polaris.py -i POLARIS/evaluation/benchmarks/aime24.parquet -o POLARIS/evaluation/benchmarks/aime24.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import TypedDict, cast

from datasets import load_dataset
from loguru import logger
from tqdm.auto import tqdm

from cognitive_probe.utils.log_utils import init_loguru_logger

# -----------------------------
# 数据结构
# -----------------------------


class Conversation(TypedDict):
    role: str
    content: str


class RewardData(TypedDict):
    ground_truth: str | int | float | list


class RawBenchmarkData(TypedDict, total=False):
    data_source: str
    prompt: list[Conversation]
    ability: str
    reward_model: RewardData
    extra_info: dict


class BenchmarkData(TypedDict):
    id: str
    source: str
    index: int
    prompt: list[Conversation]
    solution: str | list[str]


# -----------------------------
# 工具函数
# -----------------------------


def infer_source_from_path(path: str) -> str:
    """从文件名推断 source，例如 aime24.parquet -> "aime24"。"""
    stem = Path(path).stem
    token = stem.split(".")[0] if stem else ""
    return token or "unknown"


def ensure_parent_dir(path: str) -> None:
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def to_benchmark_item(
    item: RawBenchmarkData, fallback_source: str, i: int
) -> BenchmarkData:
    data_source = item.get("data_source")
    if not data_source:
        data_source = fallback_source
    prompt = item.get("prompt", [])
    index = item.get("extra_info", {}).get("index", i)
    reward_model = cast(dict, item.get("reward_model", {}))
    if "ground_truth" not in reward_model:
        raise KeyError("reward_model.ground_truth 缺失")
    if not isinstance(reward_model["ground_truth"], (str, int, float, list)):
        raise ValueError("reward_model.ground_truth 类型错误")
    solution = cast("str | int | float | list", reward_model["ground_truth"])
    if isinstance(solution, list):
        assert all(isinstance(s, (str, int, float)) for s in solution), (
            "reward_model.ground_truth 列表元素类型错误"
        )
    return BenchmarkData(
        id=f"{data_source}_{index}",
        source=data_source,
        index=index,
        prompt=prompt,
        solution=solution,
    )


def iter_benchmark_items(data_files: str, default_source: str):
    dataset = load_dataset("parquet", data_files=data_files, split="train")
    total = None
    try:
        total = len(dataset)  # 部分 dataset 支持 __len__
    except Exception:
        pass
    for i, row in enumerate(tqdm(dataset, total=total, desc="Converting", unit="it")):
        try:
            item = cast(RawBenchmarkData, row)
            yield to_benchmark_item(item, default_source, i)
        except Exception as e:  # noqa: BLE001
            logger.exception("转换单条样本失败 index=%d: %s", i, e)
            continue


def write_jsonl(items, output_file: str) -> int:
    ensure_parent_dir(output_file)
    # 若 items 非 list，尝试获取总数失败则为 None
    try:
        total = len(items)  # type: ignore[arg-type]
    except Exception:
        total = None
    written = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for obj in tqdm(items, total=total, desc="Writing", unit="it"):
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")
            written += 1
    return written


def convert(input_path: str, output_path: str) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    source = infer_source_from_path(input_path)
    logger.info(
        "开始转换: input={}, output={}, source={}", input_path, output_path, source
    )

    items = list(iter_benchmark_items(input_path, source))
    total = len(items)
    logger.info("解析完成，共 {} 条", total)

    written = write_jsonl(items, output_path)
    logger.info("写入完成：{} 条 -> {}", written, output_path)


# -----------------------------
# CLI
# -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 POLARIS parquet 基准数据转换为 jsonl",
    )
    parser.add_argument(
        "-i", dest="input_path", required=True, help="输入 parquet 文件路径"
    )
    parser.add_argument(
        "-o", dest="output_path", required=True, help="输出 jsonl 文件路径"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    init_loguru_logger()
    try:
        convert(input_path=args.input_path, output_path=args.output_path)
        logger.info("完成")
        return 0
    except Exception as e:  # noqa: BLE001
        logger.exception("执行失败: {}", e)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
