from __future__ import annotations

import base64
import hashlib
import json
import os
import pickle
import re
import subprocess
import tempfile
import textwrap
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

PYTHON_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*(?P<code>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)

FUNCTION_RUNNER = r"""
import importlib.util
import json
import sys
from pathlib import Path

payload = json.loads(sys.stdin.read())
solution_path = Path(payload["solution_path"])
fn_name = payload["fn_name"]
raw_input = payload["input"]

spec = importlib.util.spec_from_file_location("candidate_solution", solution_path)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)

if hasattr(module, fn_name):
    fn = getattr(module, fn_name)
else:
    solution_cls = getattr(module, "Solution")
    fn = getattr(solution_cls(), fn_name)

args = json.loads(raw_input)
if not isinstance(args, list):
    args = [args]
result = fn(*args)
sys.stdout.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
"""


@dataclass(slots=True)
class TestCaseResult:
    index: int
    passed: bool
    verdict: str
    error_type: str | None = None
    actual: str | None = None
    expected: str | None = None
    duration_s: float = 0.0
    stderr: str = ""


@dataclass(slots=True)
class JudgeResult:
    passed: bool
    verdict: str
    error_type: str | None = None
    duration_s: float = 0.0
    num_tests: int = 0
    passed_tests: int = 0
    failed_test_index: int | None = None
    test_pass_vector: list[bool] = field(default_factory=list)
    tests: list[TestCaseResult] = field(default_factory=list)

    def to_dict(self, *, include_tests: bool = False) -> dict[str, Any]:
        pass_bits = "".join("1" if passed else "0" for passed in self.test_pass_vector)
        payload: dict[str, Any] = {
            "passed": self.passed,
            "verdict": self.verdict,
            "error_type": self.error_type,
            "duration_s": self.duration_s,
            "num_tests": self.num_tests,
            "passed_tests": self.passed_tests,
            "failed_test_index": self.failed_test_index,
            "test_pass_vector": self.test_pass_vector,
            "behavior_signature": f"{len(self.test_pass_vector)}:{pass_bits}",
        }
        if include_tests:
            payload["tests"] = [
                {
                    "index": test.index,
                    "passed": test.passed,
                    "verdict": test.verdict,
                    "error_type": test.error_type,
                    "actual": test.actual,
                    "expected": test.expected,
                    "duration_s": test.duration_s,
                    "stderr": test.stderr,
                }
                for test in self.tests
            ]
        return payload


def extract_python_code(response: str) -> str:
    """Extract the first fenced Python code block, falling back to raw text."""
    if not response:
        return ""

    matches = list(PYTHON_FENCE_RE.finditer(response))
    if matches:
        return matches[0].group("code").strip()

    stripped = response.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return stripped.strip("`").strip()
    return stripped


def stable_hash(payload: Any) -> str:
    data = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(data).hexdigest()


def decode_private_test_cases(encoded: str) -> Any:
    decoded = base64.b64decode(encoded)
    inflated = zlib.decompress(decoded)
    payload = pickle.loads(inflated)
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload
    return payload


def normalize_cases(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return [{"input": value, "output": None}]
    if isinstance(value, dict):
        if isinstance(value.get("inputs"), list) and isinstance(value.get("outputs"), list):
            return [
                {"input": inp, "output": out}
                for inp, out in zip(value["inputs"], value["outputs"])
            ]
        return [value]
    if isinstance(value, list):
        result: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                result.append(item)
            else:
                result.append({"input": item, "output": None})
        return result
    return [{"input": value, "output": None}]


def cases_from_judge_payload(judge_payload: dict[str, Any], *, include_private: bool) -> list[dict[str, Any]]:
    cases = normalize_cases(judge_payload.get("public_test_cases"))
    if include_private:
        if isinstance(judge_payload.get("private_test_cases"), list):
            cases.extend(normalize_cases(judge_payload.get("private_test_cases")))
        elif encoded := judge_payload.get("private_test_cases_encoded"):
            cases.extend(normalize_cases(decode_private_test_cases(str(encoded))))
    if not cases:
        cases = normalize_cases(judge_payload.get("input_output"))
    return cases


def _stdio_equal(actual: str, expected: str) -> bool:
    if actual.rstrip() == expected.rstrip():
        return True
    return actual.split() == expected.split()


def _loads_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _functional_equal(actual: str, expected: Any) -> bool:
    actual_obj = _loads_jsonish(actual)
    expected_obj = _loads_jsonish(expected)
    return actual_obj == expected_obj


def _subprocess_env() -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return env


def _resolve_python_bin(python_bin: str) -> str:
    path = Path(python_bin)
    if path.exists():
        return str(path.resolve())
    return python_bin


def _run_subprocess(
    cmd: list[str],
    *,
    input_text: str,
    cwd: Path,
    timeout_s: float,
) -> tuple[str, str, int, float, str | None]:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            cwd=cwd,
            env=_subprocess_env(),
            timeout=timeout_s,
            check=False,
        )
        duration_s = time.monotonic() - start
        return proc.stdout, proc.stderr, proc.returncode, duration_s, None
    except subprocess.TimeoutExpired as exc:
        duration_s = time.monotonic() - start
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return stdout, stderr, -1, duration_s, "timeout"


def _run_stdio_case(
    *,
    python_bin: str,
    solution_path: Path,
    case: dict[str, Any],
    index: int,
    timeout_s: float,
) -> TestCaseResult:
    expected = "" if case.get("output") is None else str(case.get("output"))
    actual, stderr, returncode, duration_s, error = _run_subprocess(
        [python_bin, str(solution_path)],
        input_text=str(case.get("input") or ""),
        cwd=solution_path.parent,
        timeout_s=timeout_s,
    )
    if error == "timeout":
        return TestCaseResult(index=index, passed=False, verdict="timeout", error_type="timeout", duration_s=duration_s, stderr=stderr[:2000])
    if returncode != 0:
        return TestCaseResult(index=index, passed=False, verdict="runtime_error", error_type="runtime_error", actual=actual[:2000], expected=expected[:2000], duration_s=duration_s, stderr=stderr[:2000])
    passed = _stdio_equal(actual, expected)
    return TestCaseResult(index=index, passed=passed, verdict="accepted" if passed else "wrong_answer", error_type=None if passed else "wrong_answer", actual=actual[:2000], expected=expected[:2000], duration_s=duration_s, stderr=stderr[:2000])


def _run_function_case(
    *,
    python_bin: str,
    solution_path: Path,
    runner_path: Path,
    fn_name: str,
    case: dict[str, Any],
    index: int,
    timeout_s: float,
) -> TestCaseResult:
    expected = case.get("output")
    payload = {
        "solution_path": str(solution_path),
        "fn_name": fn_name,
        "input": str(case.get("input") or ""),
    }
    actual, stderr, returncode, duration_s, error = _run_subprocess(
        [python_bin, str(runner_path)],
        input_text=json.dumps(payload, ensure_ascii=False),
        cwd=solution_path.parent,
        timeout_s=timeout_s,
    )
    if error == "timeout":
        return TestCaseResult(index=index, passed=False, verdict="timeout", error_type="timeout", duration_s=duration_s, stderr=stderr[:2000])
    if returncode != 0:
        return TestCaseResult(index=index, passed=False, verdict="runtime_error", error_type="runtime_error", actual=actual[:2000], expected=str(expected)[:2000], duration_s=duration_s, stderr=stderr[:2000])
    passed = _functional_equal(actual, expected)
    return TestCaseResult(index=index, passed=passed, verdict="accepted" if passed else "wrong_answer", error_type=None if passed else "wrong_answer", actual=actual[:2000], expected=str(expected)[:2000], duration_s=duration_s, stderr=stderr[:2000])


def judge_code(
    *,
    code: str,
    judge_payload: dict[str, Any],
    timeout_s: float,
    include_private: bool = True,
    python_bin: str = "python3",
    include_test_details: bool = False,
) -> JudgeResult:
    cases = cases_from_judge_payload(judge_payload, include_private=include_private)
    num_tests = len(cases)
    if not code.strip():
        return JudgeResult(
            passed=False,
            verdict="format_error",
            error_type="format_error",
            num_tests=num_tests,
            passed_tests=0,
            failed_test_index=0 if num_tests else None,
            test_pass_vector=[False] * num_tests,
        )

    python_bin = _resolve_python_bin(python_bin)
    metadata = judge_payload.get("metadata") if isinstance(judge_payload.get("metadata"), dict) else {}
    input_output = judge_payload.get("input_output") if isinstance(judge_payload.get("input_output"), dict) else {}
    fn_name = metadata.get("func_name") or metadata.get("function_name") or input_output.get("fn_name")
    test_results: list[TestCaseResult] = []
    start = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="lcb_judge_") as tmp:
        tmp_path = Path(tmp)
        solution_path = tmp_path / "solution.py"
        solution_path.write_text(code, encoding="utf-8")
        runner_path = tmp_path / "function_runner.py"
        if fn_name:
            runner_path.write_text(textwrap.dedent(FUNCTION_RUNNER), encoding="utf-8")

        for index, case in enumerate(cases):
            if fn_name:
                result = _run_function_case(
                    python_bin=python_bin,
                    solution_path=solution_path,
                    runner_path=runner_path,
                    fn_name=str(fn_name),
                    case=case,
                    index=index,
                    timeout_s=timeout_s,
                )
            else:
                result = _run_stdio_case(
                    python_bin=python_bin,
                    solution_path=solution_path,
                    case=case,
                    index=index,
                    timeout_s=timeout_s,
                )
            test_results.append(result)

    duration_s = time.monotonic() - start
    passed_tests = sum(1 for result in test_results if result.passed)
    test_pass_vector = [result.passed for result in test_results]
    failed = next((result for result in test_results if not result.passed), None)
    if failed is None:
        return JudgeResult(
            passed=True,
            verdict="accepted",
            duration_s=duration_s,
            num_tests=num_tests,
            passed_tests=passed_tests,
            test_pass_vector=test_pass_vector,
            tests=test_results if include_test_details else [],
        )
    return JudgeResult(
        passed=False,
        verdict=failed.verdict,
        error_type=failed.error_type,
        duration_s=duration_s,
        num_tests=num_tests,
        passed_tests=passed_tests,
        failed_test_index=failed.index,
        test_pass_vector=test_pass_vector,
        tests=test_results if include_test_details else [],
    )
