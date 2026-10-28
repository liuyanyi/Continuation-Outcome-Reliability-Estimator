#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import traceback
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cognitive_probe.evaluation.v2 import compact_v2_run_artifacts


@dataclass(frozen=True)
class SuiteCase:
    output_subdir: str
    description: str
    input_strategy: str = "delegate_second_level"
    input_strategy_args: dict[str, Any] = field(default_factory=dict)
    strategy: str = ""
    strategy_args: dict[str, Any] = field(default_factory=dict)
    postprocess_strategy: str = ""
    postprocess_strategy_args: dict[str, Any] = field(default_factory=dict)
    regen_args: dict[str, Any] = field(default_factory=dict)
    fallback_args: dict[str, Any] = field(default_factory=dict)
    input_group_key: str = "example_id"


@dataclass(frozen=True)
class SuiteGroup:
    title: str
    cases: list[SuiteCase]


@dataclass(frozen=True)
class SuiteConfig:
    suite_name: str
    mode: str
    default_output_suffix: str
    default_output_mode: str
    default_tokenizer_path: str
    result_file_name: str
    groups: list[SuiteGroup]


@dataclass(frozen=True)
class RuntimeOptions:
    tokenizer_path: str
    min_keep_tokens: int
    max_records: int
    run_eval_after_prune: bool
    eval_k_values: str
    eval_k_array: list[str]
    case_regex: str
    max_workers: int
    compact_completed_cases: bool
    remove_legacy_generation: bool
    strip_segment_probs: bool
    strip_segment_output_ids: bool
    continue_on_case_error: bool


def _load_toml(path: Path) -> dict[str, Any]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected TOML table in {path}")
    return payload


def _require_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _optional_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a table/object")
    return dict(value)


def _parse_case(payload: Any, *, group_title: str, case_index: int) -> SuiteCase:
    if not isinstance(payload, dict):
        raise ValueError(f"group '{group_title}' case #{case_index} must be a table")

    output_subdir = _require_str(
        payload.get("output_subdir"), field_name="output_subdir"
    )
    description = _require_str(payload.get("description"), field_name="description")
    input_strategy = str(payload.get("input_strategy", "delegate_second_level")).strip()
    if not input_strategy:
        raise ValueError(f"case '{output_subdir}' has empty input_strategy")

    input_group_key = str(payload.get("input_group_key", "example_id")).strip()
    if not input_group_key:
        raise ValueError(f"case '{output_subdir}' has empty input_group_key")

    strategy = str(payload.get("strategy", "")).strip()
    strategy_args = _optional_dict(
        payload.get("strategy_args"), field_name="strategy_args"
    )
    postprocess_strategy = str(payload.get("postprocess_strategy", "")).strip()
    postprocess_strategy_args = _optional_dict(
        payload.get("postprocess_strategy_args"),
        field_name="postprocess_strategy_args",
    )
    regen_args = _optional_dict(payload.get("regen_args"), field_name="regen_args")
    fallback_args = _optional_dict(
        payload.get("fallback_args"), field_name="fallback_args"
    )
    input_strategy_args = _optional_dict(
        payload.get("input_strategy_args"),
        field_name="input_strategy_args",
    )

    if input_strategy == "delegate_second_level" and not strategy:
        raise ValueError(
            f"case '{output_subdir}' uses delegate_second_level and must define strategy"
        )
    if not strategy and strategy_args:
        raise ValueError(
            f"case '{output_subdir}' defines strategy_args without strategy"
        )
    if not postprocess_strategy and postprocess_strategy_args:
        raise ValueError(
            f"case '{output_subdir}' defines postprocess_strategy_args without postprocess_strategy"
        )
    if regen_args:
        missing_regen_keys = [
            key for key in ("api_url", "model_name") if key not in regen_args
        ]
        if missing_regen_keys:
            raise ValueError(
                f"case '{output_subdir}' regen_args missing required keys: {missing_regen_keys}"
            )
    if fallback_args:
        missing_fallback_keys = [
            key for key in ("api_url", "model_name") if key not in fallback_args
        ]
        if missing_fallback_keys:
            raise ValueError(
                f"case '{output_subdir}' fallback_args missing required keys: {missing_fallback_keys}"
            )

    return SuiteCase(
        output_subdir=output_subdir,
        description=description,
        input_strategy=input_strategy,
        input_strategy_args=input_strategy_args,
        strategy=strategy,
        strategy_args=strategy_args,
        postprocess_strategy=postprocess_strategy,
        postprocess_strategy_args=postprocess_strategy_args,
        regen_args=regen_args,
        fallback_args=fallback_args,
        input_group_key=input_group_key,
    )


def _parse_group(payload: Any, *, group_index: int) -> SuiteGroup:
    if not isinstance(payload, dict):
        raise ValueError(f"group #{group_index} must be a table")

    title = _require_str(payload.get("title"), field_name="title")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"group '{title}' must define a non-empty cases array")

    cases = [
        _parse_case(case_payload, group_title=title, case_index=case_index)
        for case_index, case_payload in enumerate(raw_cases, start=1)
    ]
    return SuiteGroup(title=title, cases=cases)


def _load_suite_config(path: Path) -> SuiteConfig:
    payload = _load_toml(path)
    suite_name = _require_str(payload.get("suite_name"), field_name="suite_name")
    mode = _require_str(payload.get("mode"), field_name="mode")
    if mode not in {"offline", "online", "online_regenerate", "online_fallback_rg"}:
        raise ValueError(f"Unsupported mode '{mode}' in {path}")

    default_output_suffix = _require_str(
        payload.get("default_output_suffix"),
        field_name="default_output_suffix",
    )
    default_output_mode = _require_str(
        payload.get("default_output_mode", "append_suffix"),
        field_name="default_output_mode",
    )
    if default_output_mode not in {"append_suffix", "replace_probe_suffix"}:
        raise ValueError(
            f"Unsupported default_output_mode '{default_output_mode}' in {path}"
        )

    result_file_name = _require_str(
        payload.get("result_file_name", "generation_results.jsonl"),
        field_name="result_file_name",
    )
    default_tokenizer_path = str(payload.get("default_tokenizer_path", ""))

    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError(f"{path} must define a non-empty groups array")

    groups = [
        _parse_group(group_payload, group_index=i)
        for i, group_payload in enumerate(raw_groups, start=1)
    ]
    return SuiteConfig(
        suite_name=suite_name,
        mode=mode,
        default_output_suffix=default_output_suffix,
        default_output_mode=default_output_mode,
        default_tokenizer_path=default_tokenizer_path,
        result_file_name=result_file_name,
        groups=groups,
    )


def _env_text(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value


def _env_int(name: str, default: int) -> int:
    raw = _env_text(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from e
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_text(name, "1" if default else "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 0/1/true/false/yes/no/on/off")


def _resolve_runtime_options(config: SuiteConfig) -> RuntimeOptions:
    tokenizer_path = _env_text("TOKENIZER_PATH", config.default_tokenizer_path)
    min_keep_tokens = _env_int("MIN_KEEP_TOKENS", 0)
    max_records = _env_int("MAX_RECORDS", 0)
    run_eval_after_prune = _env_bool("RUN_EVAL_AFTER_PRUNE", True)
    eval_k_values = _env_text("EVAL_K_VALUES", "4,8,16,32").strip()
    case_regex = _env_text("CASE_REGEX", "").strip()
    max_workers = max(1, _env_int("SUITE_MAX_WORKERS", 1))
    compact_completed_cases = _env_bool("PRUNE_COMPACT_COMPLETED_CASES", False)
    remove_legacy_generation = _env_bool("PRUNE_REMOVE_LEGACY_GENERATION", compact_completed_cases)
    strip_segment_probs = _env_bool("PRUNE_STRIP_SEGMENT_PROBS", compact_completed_cases)
    strip_segment_output_ids = _env_bool("PRUNE_STRIP_SEGMENT_OUTPUT_IDS", False)
    continue_on_case_error = _env_bool("SUITE_CONTINUE_ON_CASE_ERROR", True)
    eval_k_array = [item.strip() for item in eval_k_values.split(",") if item.strip()]
    if not eval_k_array:
        raise ValueError("EVAL_K_VALUES must contain at least one k")
    return RuntimeOptions(
        tokenizer_path=tokenizer_path,
        min_keep_tokens=min_keep_tokens,
        max_records=max_records,
        run_eval_after_prune=run_eval_after_prune,
        eval_k_values=eval_k_values,
        eval_k_array=eval_k_array,
        case_regex=case_regex,
        max_workers=max_workers,
        compact_completed_cases=compact_completed_cases,
        remove_legacy_generation=remove_legacy_generation,
        strip_segment_probs=strip_segment_probs,
        strip_segment_output_ids=strip_segment_output_ids,
        continue_on_case_error=continue_on_case_error,
    )


def _filter_groups(
    groups: list[SuiteGroup],
    *,
    case_regex: str,
) -> list[SuiteGroup]:
    if not case_regex:
        return groups

    pattern = re.compile(case_regex)
    filtered_groups: list[SuiteGroup] = []
    for group in groups:
        filtered_cases = [
            case
            for case in group.cases
            if pattern.search(group.title)
            or pattern.search(case.output_subdir)
            or pattern.search(case.description)
        ]
        if filtered_cases:
            filtered_groups.append(SuiteGroup(title=group.title, cases=filtered_cases))
    return filtered_groups


def _infer_output_dir(config: SuiteConfig, input_dir: str) -> str:
    cleaned = input_dir.rstrip("/")
    if config.default_output_mode == "replace_probe_suffix" and cleaned.endswith(
        ".probe"
    ):
        return f"{cleaned[: -len('.probe')]}{config.default_output_suffix}"
    return f"{cleaned}{config.default_output_suffix}"


def _json_arg(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _script_path(config: SuiteConfig) -> Path:
    root = Path(__file__).resolve().parent
    if config.mode == "online_regenerate":
        return root / "run_probe_prune_online_regenerate.py"
    if config.mode == "online_fallback_rg":
        return root / "run_probe_prune_online_fallback_rg.py"
    if config.mode == "online":
        return root / "run_probe_prune_online.py"
    return root / "run_probe_prune.py"


def _run_eval(case_dir: Path, runtime: RuntimeOptions, *, python_bin: str) -> None:
    if not runtime.run_eval_after_prune:
        return

    missing_reports = any(
        not (case_dir / f"eval_report_v2_{k}.json").exists()
        for k in runtime.eval_k_array
    )
    if not missing_reports:
        print(
            f"    skip eval: existing eval_report_*.json cover k={runtime.eval_k_values}"
        )
        return

    print(f"    eval: running run_eval.py (k={runtime.eval_k_values})")
    subprocess.run(
        [
            python_bin,
            "scripts/evaluation/analysis/run_eval.py",
            "--dir",
            str(case_dir),
            "--k",
            runtime.eval_k_values,
        ],
        check=True,
    )


def _case_has_v2_artifacts(case_dir: Path) -> bool:
    required = (
        case_dir / "run_overview.json",
        case_dir / "generation_segments.jsonl",
        case_dir / "decision_log.jsonl",
        case_dir / "final_outputs.jsonl",
        case_dir / "source_targets.jsonl",
    )
    return all(path.exists() for path in required)


def _compact_case_artifacts(case_dir: Path, runtime: RuntimeOptions) -> None:
    if not runtime.compact_completed_cases:
        return
    if not _case_has_v2_artifacts(case_dir):
        return
    stats = compact_v2_run_artifacts(
        case_dir,
        remove_legacy_generation=runtime.remove_legacy_generation,
        strip_segment_probs=runtime.strip_segment_probs,
        strip_segment_output_ids=runtime.strip_segment_output_ids,
    )
    saved = stats["legacy_bytes_removed"] + stats["segment_bytes_saved"]
    if saved > 0:
        print(f"    compacted artifacts: saved {saved / (1024 ** 2):.1f} MiB")


def _build_case_command(
    *,
    config: SuiteConfig,
    runtime: RuntimeOptions,
    input_dir: str,
    case_dir: Path,
    case: SuiteCase,
    python_bin: str,
) -> list[str]:
    cmd = [
        python_bin,
        str(_script_path(config)),
        "--input-dir",
        input_dir,
        "--output-dir",
        str(case_dir),
        "--result-file-name",
        config.result_file_name,
        "--input-group-key",
        case.input_group_key,
        "--input-strategy",
        case.input_strategy,
        "--input-strategy-args-json",
        _json_arg(case.input_strategy_args),
        "--min-keep-tokens",
        str(runtime.min_keep_tokens),
    ]

    if case.strategy:
        cmd.extend(["--strategy", case.strategy])
    if case.strategy_args:
        cmd.extend(["--strategy-args-json", _json_arg(case.strategy_args)])
    if case.postprocess_strategy:
        cmd.extend(["--postprocess-strategy", case.postprocess_strategy])
    if case.postprocess_strategy_args:
        cmd.extend(
            [
                "--postprocess-strategy-args-json",
                _json_arg(case.postprocess_strategy_args),
            ]
        )
    if config.mode == "online_regenerate":
        cmd.extend(["--regen-args-json", _json_arg(case.regen_args)])
    if config.mode == "online_fallback_rg":
        cmd.extend(["--fallback-args-json", _json_arg(case.fallback_args)])
    if runtime.max_records > 0:
        cmd.extend(["--max-records", str(runtime.max_records)])
    if runtime.tokenizer_path:
        cmd.extend(["--tokenizer-path", runtime.tokenizer_path])
    return cmd


def _run_case(
    *,
    config: SuiteConfig,
    runtime: RuntimeOptions,
    input_dir: str,
    output_dir: Path,
    case: SuiteCase,
    python_bin: str,
) -> None:
    case_dir = output_dir / case.output_subdir
    result_path = case_dir / config.result_file_name
    cmd = _build_case_command(
        config=config,
        runtime=runtime,
        input_dir=input_dir,
        case_dir=case_dir,
        case=case,
        python_bin=python_bin,
    )

    if result_path.exists() or _case_has_v2_artifacts(case_dir):
        print(f"    skip: {case.output_subdir} already has {config.result_file_name}")
    elif case_dir.exists():
        print(f"    resume: reusing existing directory {case_dir}")
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(cmd, check=True)

    _run_eval(case_dir, runtime, python_bin=python_bin)
    _compact_case_artifacts(case_dir, runtime)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a config-driven probe prune suite."
    )
    parser.add_argument("--config", required=True, help="Path to suite TOML config.")
    parser.add_argument("--input-dir", required=True, help="Input probe directory.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional explicit output directory. Defaults follow the suite config rule.",
    )
    parser.add_argument(
        "--case-regex",
        default="",
        help="Optional regex filter matched against group title, case output_subdir, or description.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_suite_config(config_path)
    runtime = _resolve_runtime_options(config)
    case_regex = args.case_regex.strip() or runtime.case_regex
    filtered_groups = _filter_groups(config.groups, case_regex=case_regex)

    input_dir = args.input_dir
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(_infer_output_dir(config, input_dir))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable
    total_cases = sum(len(group.cases) for group in filtered_groups)
    if total_cases == 0:
        raise ValueError(
            f"No suite cases matched case regex {case_regex!r} in config {config_path}"
        )

    print("==========================================")
    print(f"Running {config.suite_name}")
    print("==========================================")
    print(f"Config: {config_path}")
    print(f"Mode: {config.mode}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("Incremental mode: enabled (existing results will be skipped)")
    print(f"Python command: {python_bin}")
    print(f"Min keep tokens: {runtime.min_keep_tokens}")
    print(f"Max records: {runtime.max_records}")
    print(f"Run eval after prune: {int(runtime.run_eval_after_prune)}")
    print(f"Eval k values: {runtime.eval_k_values}")
    print(f"Suite workers: {runtime.max_workers}")
    print(f"OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS', '(unset)')}")
    print(f"MKL_NUM_THREADS: {os.getenv('MKL_NUM_THREADS', '(unset)')}")
    print(f"OPENBLAS_NUM_THREADS: {os.getenv('OPENBLAS_NUM_THREADS', '(unset)')}")
    print(f"TOKENIZERS_PARALLELISM: {os.getenv('TOKENIZERS_PARALLELISM', '(unset)')}")
    print(f"Compact completed cases: {int(runtime.compact_completed_cases)}")
    print(f"Continue on case error: {int(runtime.continue_on_case_error)}")
    print(f"Suite size: {total_cases} cases")
    if case_regex:
        print(f"Case regex: {case_regex}")
    if runtime.tokenizer_path:
        print(f"Tokenizer path: {runtime.tokenizer_path}")
    else:
        print("Tokenizer path: (not set, response text won't be decoded)")
    print("==========================================")
    print()

    scheduled_cases: list[tuple[int, str, SuiteCase]] = []
    case_index = 0
    for group in filtered_groups:
        print(f">>> {group.title}")
        for case in group.cases:
            case_index += 1
            scheduled_cases.append((case_index, group.title, case))
        print()

    failures: list[tuple[str, str]] = []

    def _run_case_entry(item: tuple[int, str, SuiteCase]) -> None:
        index, _group_title, case = item
        print(f"  - [{index}/{total_cases}] {case.description}")
        try:
            _run_case(
                config=config,
                runtime=runtime,
                input_dir=input_dir,
                output_dir=output_dir,
                case=case,
                python_bin=python_bin,
            )
        except Exception as exc:
            if not runtime.continue_on_case_error:
                raise
            message = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            print(f"    case failed: {case.output_subdir}: {message}")
            failures.append((case.output_subdir, message))

    if runtime.max_workers == 1:
        for item in scheduled_cases:
            _run_case_entry(item)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=runtime.max_workers) as executor:
            future_map = {
                executor.submit(_run_case_entry, item): item
                for item in scheduled_cases
            }
            for future in concurrent.futures.as_completed(future_map):
                future.result()

    print("==========================================")
    if failures:
        print(f"Completed with {len(failures)} failed case(s).")
        failure_log = output_dir / "suite_failures.log"
        with failure_log.open("w", encoding="utf-8") as f:
            for case_name, message in failures:
                f.write(f"{case_name}\t{message}\n")
        print(f"Failure log saved to: {failure_log}")
    else:
        print("All suite cases completed successfully!")
    print(f"Results saved to: {output_dir}/")
    print("==========================================")


if __name__ == "__main__":
    main()
