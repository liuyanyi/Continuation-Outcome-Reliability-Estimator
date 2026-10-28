#!/usr/bin/env python3
"""Check the local SGLang Qwen3 /encode pooler patch.

The probe-encode pipeline expects raw last-token hidden states from SGLang's
/encode endpoint. For Qwen3 CausalLM served with --is-embedding, SGLang routes
through Qwen3ForCausalLM.pooler. Upstream sglang==0.5.9 normalizes that pooler;
this project requires normalize=False.
"""

from __future__ import annotations

import ast
import argparse
import importlib.metadata as metadata
import sys
from pathlib import Path


def _sglang_qwen3_path() -> tuple[str, Path]:
    try:
        dist = metadata.distribution("sglang")
    except metadata.PackageNotFoundError:
        raise RuntimeError("sglang is not installed in the active environment")

    version = dist.version
    qwen3_path = Path(dist.locate_file("sglang/srt/models/qwen3.py"))
    if not qwen3_path.exists():
        raise RuntimeError(f"Cannot find SGLang Qwen3 model file: {qwen3_path}")
    return version, qwen3_path


def _literal_keyword(call: ast.Call, name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _target_is_self_pooler(target: ast.expr) -> bool:
    return (
        isinstance(target, ast.Attribute)
        and target.attr == "pooler"
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    )


def _find_qwen3_pooler_normalize(path: Path) -> tuple[bool | None, int | None]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "Qwen3ForCausalLM":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
                continue
            for stmt in ast.walk(item):
                if not isinstance(stmt, ast.Assign):
                    continue
                if not any(_target_is_self_pooler(target) for target in stmt.targets):
                    continue
                if not isinstance(stmt.value, ast.Call):
                    return None, stmt.lineno
                normalize_value = _literal_keyword(stmt.value, "normalize")
                if isinstance(normalize_value, ast.Constant) and isinstance(
                    normalize_value.value, bool
                ):
                    return normalize_value.value, normalize_value.lineno
                return None, stmt.lineno

    return None, None


def apply_patch(path: Path, version: str) -> bool:
    """Apply only the known 0.5.9 Qwen3 pooler edit; safe to repeat."""
    normalize, _ = _find_qwen3_pooler_normalize(path)
    if normalize is False:
        return False
    if version != "0.5.9" or normalize is not True:
        raise RuntimeError("Refusing to patch an unknown SGLang version or pooler layout")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name != "Qwen3ForCausalLM":
            continue
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name != "__init__":
                continue
            for stmt in ast.walk(method):
                if not isinstance(stmt, ast.Assign) or not any(
                    _target_is_self_pooler(t) for t in stmt.targets
                ):
                    continue
                call = stmt.value
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "Pooler":
                    raise RuntimeError("Unexpected pooler constructor")
                value = _literal_keyword(call, "normalize")
                if not isinstance(value, ast.Constant) or value.value is not True:
                    raise RuntimeError("Unexpected normalize expression")
                lines = source.splitlines(keepends=True)
                line = lines[value.lineno - 1].encode("utf-8")
                lines[value.lineno - 1] = (line[:value.col_offset] + b"False" + line[value.end_col_offset:]).decode("utf-8")
                updated = "".join(lines)
                ast.parse(updated)
                backup = path.with_suffix(path.suffix + ".probe-original")
                if not backup.exists():
                    backup.write_text(source, encoding="utf-8")
                path.write_text(updated, encoding="utf-8")
                return True
    raise RuntimeError("Qwen3 pooler assignment not found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Patch installed SGLang 0.5.9; saves a .probe-original backup. Restart the service afterwards.")
    args = parser.parse_args()
    try:
        version, qwen3_path = _sglang_qwen3_path()
        if args.apply:
            changed = apply_patch(qwen3_path, version)
            print("Patch applied; restart running Qwen3 services." if changed else "Patch already present.")
        normalize, lineno = _find_qwen3_pooler_normalize(qwen3_path)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    location = f"{qwen3_path}:{lineno}" if lineno is not None else str(qwen3_path)
    if normalize is False:
        print(
            "OK: SGLang Qwen3 /encode pooler patch is present "
            f"(sglang=={version}, {location}, normalize=False)."
        )
        return 0

    if normalize is True:
        print(
            "FAIL: SGLang Qwen3 /encode pooler is normalized "
            f"(sglang=={version}, {location}, normalize=True). "
            "run_probe_encode.py expects raw last-token hidden states; restore "
            "`Pooler(pooling_type=PoolingType.LAST, normalize=False)`.",
            file=sys.stderr,
        )
        return 1

    print(
        "FAIL: Could not verify SGLang Qwen3 pooler normalize setting "
        f"(sglang=={version}, {location}). Expected an explicit normalize=False.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
