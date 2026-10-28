# Validation (2026-10-28)

Validated using the parent project virtual environment with release/src on PYTHONPATH.

- 31 CPU unit tests passed, including Qwen3 patch idempotency/refusal checks,
  exported-head parity and input-embedding selection for untied/sharded checkpoints.
- A fake local SGLang service completed the real benchmark generation, probe
  encode, online pruning and evaluation scripts on CPU. Token-aligned scores and
  final V2 report artifacts were checked. This does not validate SGLang kernels.
- A wheel built successfully in a temporary copy and contained the tree, probe,
  evaluation and Nemotron extension modules.
- All 12 core tree, scoring, dataset, training, inference, export and benchmark
  CLI entrypoints accepted --help.
- A temporary random tiny Qwen3 model completed four CPU training steps, checkpoint
  inference and standalone head export. The CPU smoke run used SDPA attention,
  use_cpu=true and bf16=false; no model weights were downloaded.
- Static source/example/config checks and startup shell syntax checks passed.
- uv lock --check --offline passed; uv sync --frozen --no-dev --dry-run succeeded.
  This is an installation plan check, not a fresh CUDA environment installation.

Not yet validated: real GPU tree generation, SGLang encode output parity against
Transformers, Nemotron runtime, and a full benchmark run against live services.

Compatibility changes: the training/head files include current normalized-sqrt-N
loss support and distributed head-save guarding. Probe encode uses input embedding
weights for logit features, matching training even when LM-head weights are untied.
Other experiment-only changes were not wholesale copied into the release.

Additional audit fixes: added-token coverage and parent-directory creation in the
token-map tool; nonzero failure exits for batch scoring/tree generation; independent
pruning-test imports; tree configuration range validation; explicit tiny probe
stride and instructions to disable automatic math evaluation for code benchmarks.

Publishing metadata still requires an owner-selected LICENSE.

## Packaged evaluation data

- AIME 2024/2025: 30 rows each. Copies are byte-identical
  to the parent workspace's benchmark inputs.
- LCB thinking/instruct inputs: 1055 rows each, identical targets and test references.
  Question IDs match all six raw splits of cumulative release_v6.
- The complete judge sidecar is byte-identical to its source (SHA-256 verified),
  with unique IDs covering every benchmark question. The real evaluator resolves
  the release-relative sidecar and loads private-test payloads.
- All four packaged compact inputs load through load_evaluation_samples; their packaged
  hashes match data/benchmarks/manifest.json. Primary documentation links resolve.

These are data-integrity checks, not new model benchmark results.

GPQA question/answer data was removed before public packaging. Its acquisition
and seeded local conversion remain documented; both archive types exclude it.

Split-package filtering was tested with synthetic code/data/GPQA/cache files.
GPQA local conversion was checked in temporary directories: 198 unique rows,
valid answer letters, and identical output hashes across two seed-42 conversions.
No converted GPQA data remains in the release.
