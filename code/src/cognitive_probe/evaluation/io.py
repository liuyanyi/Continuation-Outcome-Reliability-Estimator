from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_bytes().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(orjson.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for row in rows:
            f.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))


def load_json(path: Path) -> dict[str, Any]:
    payload = orjson.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}, got {type(payload)}")
    return payload


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(
        orjson.dumps(
            payload,
            option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE,
        )
    )


def json_text(payload: Any, *, sort_keys: bool = False) -> str:
    option = orjson.OPT_SORT_KEYS if sort_keys else 0
    return orjson.dumps(payload, option=option).decode("utf-8")
