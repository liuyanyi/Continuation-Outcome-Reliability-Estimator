# Environment and dependencies

Use Python 3.12 and `uv`, from the release root:

```bash
uv sync --frozen --no-dev
```

GPU execution requires CUDA compatible with PyTorch 2.9.1/cu129. Installing
FlashAttention from source requires a compatible CUDA compiler. The public
PyTorch index is configured in `pyproject.toml`; no private index or local wheel
is required by the configuration.

`pyproject.toml` declares direct dependencies and `uv.lock` pins the full graph.
Model execution dependencies are pinned, including PyTorch, Transformers, SGLang
and FlashAttention. The Qwen3 SGLang patch is a separate required step for probe
encoding; see the README. Reinstalling SGLang may remove that patch.

To update dependencies, use `uv add` / `uv remove`, then refresh exported locks:

```bash
uv export --frozen --no-dev --format requirements.txt --no-emit-project \
  --output-file requirements.lock.txt
uv export --frozen --only-dev --format requirements.txt --no-emit-project \
  --output-file requirements-dev.lock.txt
uv lock --check
```

Use `.venv/bin/python` or `uv run python` for scripts. `uv sync --frozen` also
installs the dev tools. Test results and installation limits are in
[VALIDATION.md](VALIDATION.md).
