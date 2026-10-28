"""
将 GPQA benchmark 数据（仅 gpqa_diamond）转换为项目评测 jsonl 格式。

输出格式对齐 scripts/data/convert_data_from_polaris.py：
- id
- source
- index
- prompt: [{"role": "user", "content": "..."}]
- solution

并将题目构造成标准多项选择题（A/B/C/D），按 seed 打乱选项并记录正确答案：
- input: Question
- choices: [Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3, Correct Answer]
- target: the correct answer letter after shuffling

示例：
    python scripts/data/process_gpqa_diamond.py \
      -i data/gpqa/gpqa_diamond.csv \
      -o data/benchmarks/gpqa_diamond.jsonl
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import TypedDict

from loguru import logger
from tqdm.auto import tqdm

from cognitive_probe.utils.log_utils import init_loguru_logger


class Conversation(TypedDict):
    role: str
    content: str


class BenchmarkData(TypedDict):
    id: str
    source: str
    index: int
    prompt: list[Conversation]
    solution: str


def ensure_parent_dir(path: str) -> None:
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def build_mcq_prompt(question: str, choices: list[tuple[str, str]]) -> str:
    choice_map = {label: text for label, text in choices}
    return (
        f"What is the correct answer to this question: {question}\n\n"
        "Choices:\n"
        f"(A) {choice_map['A']}\n"
        f"(B) {choice_map['B']}\n"
        f"(C) {choice_map['C']}\n"
        f"(D) {choice_map['D']}\n\n"
        "Let's think step by step and output the final answer letter (A, B, C, or D) within \\boxed{}."
    )


def shuffle_choices(
    incorrect_1: str,
    incorrect_2: str,
    incorrect_3: str,
    correct: str,
    rng: random.Random,
) -> tuple[list[tuple[str, str]], str]:
    labeled_choices = [
        ("Incorrect", incorrect_1),
        ("Incorrect", incorrect_2),
        ("Incorrect", incorrect_3),
        ("Correct", correct),
    ]
    rng.shuffle(labeled_choices)

    letters = ["A", "B", "C", "D"]
    rendered: list[tuple[str, str]] = []
    answer_letter = ""
    for i, (tag, text) in enumerate(labeled_choices):
        letter = letters[i]
        rendered.append((letter, text))
        if tag == "Correct":
            answer_letter = letter

    if not answer_letter:
        raise ValueError("未找到正确选项，无法生成答案字母")
    return rendered, answer_letter


def iter_gpqa_diamond_items(input_csv: str, rng: random.Random):
    source = "gpqa_diamond"
    with open(input_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_fields = [
            "Question",
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
            "Correct Answer",
        ]

        missing_fields = [x for x in required_fields if x not in (reader.fieldnames or [])]
        if missing_fields:
            raise KeyError(f"CSV 缺少必要字段: {missing_fields}")

        for i, row in enumerate(tqdm(reader, desc="Converting", unit="it")):
            question = clean_text(row.get("Question"))
            a = clean_text(row.get("Incorrect Answer 1"))
            b = clean_text(row.get("Incorrect Answer 2"))
            c = clean_text(row.get("Incorrect Answer 3"))
            d = clean_text(row.get("Correct Answer"))

            if not question:
                logger.warning("跳过空题干样本 index={}", i)
                continue
            if not a or not b or not c or not d:
                logger.warning("跳过选项缺失样本 index={}", i)
                continue

            choices, answer_letter = shuffle_choices(
                incorrect_1=a,
                incorrect_2=b,
                incorrect_3=c,
                correct=d,
                rng=rng,
            )
            content = build_mcq_prompt(question=question, choices=choices)
            yield BenchmarkData(
                id=f"{source}_{i}",
                source=source,
                index=i,
                prompt=[{"role": "user", "content": content}],
                solution=answer_letter,
            )


def write_jsonl(items, output_file: str) -> int:
    ensure_parent_dir(output_file)
    written = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for obj in tqdm(items, desc="Writing", unit="it"):
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")
            written += 1
    return written


def validate_input_is_gpqa_diamond(input_path: str) -> None:
    name = Path(input_path).name.lower()
    if "gpqa_diamond" not in name:
        raise ValueError(
            f"仅支持 gpqa_diamond 子集，当前输入文件名不是 gpqa_diamond: {input_path}"
        )


def convert(input_path: str, output_path: str, seed: int | None = None) -> None:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    validate_input_is_gpqa_diamond(input_path)
    logger.info(
        "开始转换: input={}, output={}, seed={}",
        input_path,
        output_path,
        seed,
    )

    rng = random.Random(seed)

    items = list(iter_gpqa_diamond_items(input_path, rng=rng))
    logger.info("解析完成，共 {} 条", len(items))

    written = write_jsonl(items, output_path)
    logger.info("写入完成：{} 条 -> {}", written, output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 gpqa_diamond.csv 转换为项目 benchmark jsonl（多项选择题）",
    )
    parser.add_argument("-i", dest="input_path", required=True, help="输入 gpqa_diamond.csv")
    parser.add_argument("-o", dest="output_path", required=True, help="输出 jsonl 文件路径")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（设置后可复现选项打乱结果）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    init_loguru_logger()
    try:
        convert(input_path=args.input_path, output_path=args.output_path, seed=args.seed)
        logger.info("完成")
        return 0
    except Exception as e:  # noqa: BLE001
        logger.exception("执行失败: {}", e)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
