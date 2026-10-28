#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import pickle
import random
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson
from loguru import logger
from tqdm.auto import tqdm

THINKING_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Solve the problem exactly and output "
    "only the final Python code in one markdown code block."
)

INSTRUCT_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Solve the problem exactly.\n\n"
    "Your response must use this structure:\n"
    "Thinking Process:\n"
    "Briefly reason through the algorithm, edge cases, and complexity.\n\n"
    "Final Code:\n"
    "Provide exactly one fenced Python code block containing the final solution."
)

RELEASE_FILES: dict[str, list[str]] = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    "release_latest": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    "v1": ["test.jsonl"],
    "v2": ["test2.jsonl"],
    "v3": ["test3.jsonl"],
    "v4": ["test4.jsonl"],
    "v5": ["test5.jsonl"],
    "v6": ["test6.jsonl"],
}


def _json_default(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _as_str(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value)


def _load_maybe_json(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def _decode_private_test_cases(value: Any) -> tuple[Any, str | None]:
    parsed = _load_maybe_json(value)
    if isinstance(parsed, (dict, list)) or parsed is None:
        return parsed, None

    if not isinstance(parsed, str):
        return parsed, None

    text = parsed.strip()
    if not text:
        return None, None

    try:
        decoded = base64.b64decode(text)
        inflated = zlib.decompress(decoded)
        payload = pickle.loads(inflated)
        return _load_maybe_json(payload), None
    except Exception as exc:  # noqa: BLE001
        return value, f"{type(exc).__name__}: {exc}"


def _normalize_test_cases(value: Any) -> list[dict[str, Any]]:
    parsed = _load_maybe_json(value)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("inputs"), list) and isinstance(parsed.get("outputs"), list):
            return [
                {"input": inp, "output": out}
                for inp, out in zip(parsed["inputs"], parsed["outputs"])
            ]
        return [parsed]
    if isinstance(parsed, list):
        result: list[dict[str, Any]] = []
        for item in parsed:
            if isinstance(item, dict):
                result.append(item)
            else:
                result.append({"input": item, "output": None})
        return result
    return []


def _build_input_output(
    public_cases: list[dict[str, Any]],
    private_cases: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    tests = public_cases + private_cases
    inputs: list[Any] = []
    outputs: list[Any] = []
    for case in tests:
        inputs.append(case.get("input", case.get("inputs")))
        outputs.append(case.get("output", case.get("outputs")))

    input_output: dict[str, Any] = {
        "inputs": inputs,
        "outputs": outputs,
    }
    func_name = metadata.get("func_name") or metadata.get("function_name")
    if func_name:
        input_output["fn_name"] = str(func_name)
    return input_output


def _build_prompt(
    question: str,
    starter_code: str,
    metadata: dict[str, Any],
    *,
    prompt_profile: str,
) -> list[dict[str, str]]:
    func_name = metadata.get("func_name") or metadata.get("function_name")
    mode_hint = (
        f"Implement the function `{func_name}`. Do not read from stdin."
        if func_name
        else "Write a complete program that reads from stdin and writes to stdout."
    )
    if prompt_profile == "instruct":
        output_instruction = (
            "Use the required response structure from the system message. "
            "Put the final solution code only in the Final Code fenced Python block."
        )
        system_prompt = INSTRUCT_SYSTEM_PROMPT
    elif prompt_profile == "thinking":
        output_instruction = "Return only one fenced Python code block."
        system_prompt = THINKING_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unsupported prompt profile: {prompt_profile}")

    parts = [
        question.strip(),
        mode_hint,
        output_instruction,
    ]
    if starter_code.strip():
        parts.append("Starter code:\n```python\n" + starter_code.strip() + "\n```")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _row_question_id(row: dict[str, Any], index: int) -> str:
    for key in ("question_id", "id", "problem_id"):
        value = row.get(key)
        if not _is_missing(value) and str(value).strip():
            return str(value).strip()
    return str(index)


def _normalize_record(
    row: dict[str, Any],
    index: int,
    *,
    source: str,
    decode_private_tests: bool,
    prompt_profile: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_raw = _load_maybe_json(row.get("metadata"))
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    public_cases = _normalize_test_cases(row.get("public_test_cases"))
    private_cases: list[dict[str, Any]] = []
    private_decode_error = None
    private_test_cases_raw = _as_str(row.get("private_test_cases"))
    if decode_private_tests:
        private_raw, private_decode_error = _decode_private_test_cases(row.get("private_test_cases"))
        private_cases = _normalize_test_cases(private_raw)
    input_output = _build_input_output(public_cases, private_cases, metadata)

    question_id = _row_question_id(row, index)
    question = _as_str(row.get("question_content") or row.get("question") or row.get("prompt"))
    starter_code = _as_str(row.get("starter_code"))

    full_judge_payload: dict[str, Any] = {
        "question_id": question_id,
        "input_output": input_output,
        "public_test_cases": public_cases,
        "starter_code": starter_code,
        "metadata": metadata,
    }
    if decode_private_tests:
        full_judge_payload["private_test_cases"] = private_cases
        if private_decode_error:
            full_judge_payload["private_decode_error"] = private_decode_error
            full_judge_payload["private_test_cases_raw"] = private_test_cases_raw
    else:
        full_judge_payload["private_test_cases_encoded"] = private_test_cases_raw
        full_judge_payload["private_test_cases_encoding"] = "base64_zlib_pickle"
        full_judge_payload["private_test_cases_decoded"] = False

    row_metadata = {
        "question_id": question_id,
        "platform": _as_str(row.get("platform")),
        "contest_id": _as_str(row.get("contest_id")),
        "contest_date": _as_str(row.get("contest_date")),
        "difficulty": _as_str(row.get("difficulty")),
        "release_date": _as_str(row.get("release_date")),
        "metadata": metadata,
    }

    benchmark_judge_payload = {
        "question_id": question_id,
        "input_output": input_output,
        "public_test_cases": public_cases,
        "starter_code": starter_code,
        "metadata": metadata,
        "private_test_cases_externalized": True,
    }

    benchmark_row = {
        "id": f"{source}_{question_id}",
        "source": source,
        "index": index,
        "prompt": _build_prompt(
            question,
            starter_code,
            metadata,
            prompt_profile=prompt_profile,
        ),
        "solution": "",
        "judge_type": "livecodebench_code_generation",
        "judge_payload": benchmark_judge_payload,
        "metadata": row_metadata,
    }
    return benchmark_row, full_judge_payload


def _iter_json_file(path: Path) -> Iterable[dict[str, Any]]:
    payload = orjson.loads(path.read_bytes())
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        rows = payload.get("rows") or payload.get("data")
        if isinstance(rows, list):
            for item in rows:
                if isinstance(item, dict):
                    yield item
            return
        yield payload
        return
    raise ValueError(f"Unsupported JSON payload in {path}: {type(payload).__name__}")


def _iter_jsonl_file(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            item = orjson.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object line in {path}")
            yield item


def _iter_parquet_file(path: Path) -> Iterable[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    yield from frame.to_dict(orient="records")


def _iter_input_file(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl_file(path)
    elif suffix == ".json":
        yield from _iter_json_file(path)
    elif suffix == ".parquet":
        yield from _iter_parquet_file(path)
    else:
        raise ValueError(f"Unsupported input file extension: {path}")


def _release_allowed_files(version_tag: str) -> set[str] | None:
    version_tag = version_tag.strip()
    if not version_tag or version_tag == "all":
        return None
    if version_tag in RELEASE_FILES:
        return set(RELEASE_FILES[version_tag])
    if "_" in version_tag:
        start, end = version_tag.split("_", 1)
        if start in RELEASE_FILES and end in RELEASE_FILES and start.startswith("v") and end.startswith("v"):
            start_idx = int(start[1:])
            end_idx = int(end[1:])
            if start_idx <= end_idx:
                files = ["test.jsonl" if idx == 1 else f"test{idx}.jsonl" for idx in range(start_idx, end_idx + 1)]
                return set(files)
    raise ValueError(f"Unsupported version tag: {version_tag}")


def _discover_input_files(input_path: Path, *, version_tag: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files: list[Path] = []
    for pattern in ("*.jsonl", "*.json", "*.parquet"):
        files.extend(input_path.rglob(pattern))
    allowed = _release_allowed_files(version_tag)
    if allowed is not None:
        files = [path for path in files if path.name in allowed]
    return sorted(files)


def _iter_rows(input_path: Path, *, version_tag: str) -> Iterable[dict[str, Any]]:
    files = _discover_input_files(input_path, version_tag=version_tag)
    if not files:
        raise FileNotFoundError(f"No matching json/jsonl/parquet files found under {input_path} for {version_tag}")
    for path in files:
        logger.info("Reading {}", path)
        yield from _iter_input_file(path)


def load_rows(input_path: Path, *, version_tag: str) -> list[dict[str, Any]]:
    return list(_iter_rows(input_path, version_tag=version_tag))


def convert(
    *,
    input_path: Path,
    output_path: Path,
    source: str,
    version_tag: str,
    limit: int | None,
    offset: int,
    shuffle: bool,
    seed: int,
    decode_private_tests: bool,
    judge_output_path: Path | None,
    prompt_profile: str,
) -> int:
    if shuffle:
        rows = load_rows(input_path, version_tag=version_tag)
        rng = random.Random(seed)
        rng.shuffle(rows)
        if offset:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        row_iter: Iterable[dict[str, Any]] = rows
    else:
        row_iter = _iter_rows(input_path, version_tag=version_tag)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if judge_output_path is not None:
        judge_output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    seen = 0
    with output_path.open("wb") as f, (
        judge_output_path.open("wb") if judge_output_path is not None else open(os.devnull, "wb")
    ) as judge_f:
        for row in tqdm(row_iter, desc="Converting", unit="row"):
            if not shuffle:
                if seen < offset:
                    seen += 1
                    continue
                if limit is not None and written >= limit:
                    break
            item, full_judge_payload = _normalize_record(
                row,
                written,
                source=source,
                decode_private_tests=decode_private_tests,
                prompt_profile=prompt_profile,
            )
            if judge_output_path is not None:
                item["judge_payload"]["judge_manifest_file"] = str(judge_output_path)
                judge_f.write(
                    orjson.dumps(
                        {
                            "id": item["id"],
                            "sample_id": item["id"],
                            "question_id": full_judge_payload["question_id"],
                            "judge_type": "livecodebench_code_generation",
                            "judge_payload": full_judge_payload,
                            "metadata": item["metadata"],
                        },
                        default=_json_default,
                        option=orjson.OPT_APPEND_NEWLINE,
                    )
                )
            f.write(
                orjson.dumps(
                    item,
                    default=_json_default,
                    option=orjson.OPT_APPEND_NEWLINE,
                )
            )
            written += 1
            seen += 1
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert local LiveCodeBench code generation files to project benchmark JSONL.",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Local LiveCodeBench file or directory containing json/jsonl/parquet files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/benchmarks/livecodebench_code_generation.jsonl",
        help="Output benchmark JSONL path.",
    )
    parser.add_argument(
        "--source",
        default="livecodebench_code_generation",
        help="Source name written to benchmark rows.",
    )
    parser.add_argument(
        "--version-tag",
        default="release_v6",
        help=(
            "LiveCodeBench local version tag to select files. "
            "Use release_v6 for cumulative v6, v6 for only test6.jsonl, or all for all files."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Keep only the first N rows after offset/shuffle.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many rows before applying limit.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle rows before offset/limit.")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    parser.add_argument(
        "--judge-output",
        default=None,
        help=(
            "Sidecar JSONL for full judge payloads. "
            "Defaults to <output stem>.judge.jsonl; use empty string to disable."
        ),
    )
    parser.add_argument(
        "--decode-private-tests",
        action="store_true",
        help="Decode compressed private tests during conversion. Default keeps them encoded to avoid huge files.",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=["thinking", "instruct"],
        default="thinking",
        help=(
            "Prompt style. 'thinking' keeps the original code-only prompt for thinking models; "
            "'instruct' asks for Thinking Process plus Final Code."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output)
    if args.judge_output is None:
        judge_output_path = output_path.with_name(output_path.stem + ".judge.jsonl")
    elif str(args.judge_output).strip():
        judge_output_path = Path(args.judge_output)
    else:
        judge_output_path = None
    written = convert(
        input_path=Path(args.input),
        output_path=output_path,
        source=args.source,
        version_tag=args.version_tag,
        limit=args.limit,
        offset=args.offset,
        shuffle=args.shuffle,
        seed=args.seed,
        decode_private_tests=args.decode_private_tests,
        judge_output_path=judge_output_path,
        prompt_profile=args.prompt_profile,
    )
    logger.info("Wrote {} rows to {}", written, args.output)
    if judge_output_path is not None:
        logger.info("Wrote judge payload sidecar to {}", judge_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
