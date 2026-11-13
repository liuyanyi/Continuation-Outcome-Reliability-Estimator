# Validation

## CPU validation

All 54 CPU unittest cases passed. Static checks passed for 67 Python sources,
examples and configuration/lock syntax.

Coverage includes:

- Shared encoding across multiple heads: one request per selected prefix, parity
  with the training projection on untied weights, context-limit skips, complete
  and interrupted-journal recovery, and rejection of changed head manifests.
- Absolute sqrt(N) loss scaling and gradients at batch size one, the supported
  weighting modes, and rejection of the removed normalized weighting mode.
- Interleaved prefix scheduling with unequal rollout lengths and resumed gaps.

- A local fake SGLang service running generation, probe encoding, online pruning
  and evaluation, with token-aligned scores and final V2 report checks.
- Qwen3 patch idempotency and refusal checks, exported-head parity, and input
  embedding selection for untied and sharded checkpoints.
- Exact REFRAIN step mapping with split Unicode bytes, added tokens, hidden EOS,
  truncated UTF-8, non-ByteLevel tokenizers and rejection of unverifiable alignment.
- Math, LiveCodeBench and bootstrap abstention scoring, including pruned prefixes
  containing correct answers, absent judge records and the fixed K denominator.
- REFRAIN prefix and continuation costs across math, LiveCodeBench and bootstrap:
  a 100-token prefix plus a 3-token answer costs 103 tokens. Continued answers
  retain normal scoring.
- Token-map added-token coverage, batch scoring failure exits, pruning behavior,
  tree configuration validation and CLI overrides.

The tests use local fixtures without model or tokenizer downloads. They do not
validate real GPU tree generation, SGLang kernels or encode parity against
Transformers, Nemotron runtime, or full benchmarks against live services.

Publishing metadata requires an owner-selected LICENSE.

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

GPQA question/answer data is not included. Its acquisition and seeded local
conversion are documented; both archive types exclude it.

Split-package filtering was tested with synthetic code/data/GPQA/cache files.
GPQA local conversion was checked in temporary directories: 198 unique rows,
valid answer letters, and identical output hashes across two seed-42 conversions.
No converted GPQA data remains in the release.
