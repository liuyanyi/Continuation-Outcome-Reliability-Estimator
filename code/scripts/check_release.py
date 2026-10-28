"""Static release checks; no GPU, model downloads or third-party imports."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import tomllib


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = list(root.joinpath("src").rglob("*.py")) + list(root.joinpath("scripts").rglob("*.py"))
    for path in sources:
        ast.parse(path.read_text(), filename=str(path))
    for name, prompt_key, answer_key in (
        ("train_questions_tiny.jsonl", "messages", "answer"),
        ("benchmark_tiny.jsonl", "prompt", "solution"),
    ):
        rows = [json.loads(line) for line in (root / "data/examples" / name).read_text().splitlines()]
        assert len(rows) >= 2 and len({r["id"] for r in rows}) == len(rows), name
        for row in rows:
            assert row[answer_key] and row[prompt_key], name
            assert all(m["role"] and m["content"] for m in row[prompt_key]), name
    config = json.loads((root / "configs/train_probe.json").read_text())
    assert config["pooling_mode"] == "last_token"
    assert not any(config[k] for k in ("full_finetune_backbone", "use_lora", "unfreeze_last_layer_norm"))
    tomllib.loads((root / "pyproject.toml").read_text())
    tomllib.loads((root / "uv.lock").read_text())
    for folder in ("src", "scripts", "configs"):
        for path in (root / folder).rglob("*"):
            if path.suffix in {".py", ".sh", ".json", ".toml"}:
                text = path.read_text()
                assert "/large-" + "storage/" not in text, path
                assert "/workspace/" + "math_rl/" not in text, path
    print(f"OK: {len(sources)} Python sources, tiny examples, config and lock syntax")


if __name__ == "__main__":
    main()
