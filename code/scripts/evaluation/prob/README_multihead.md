# Shared encoding for multiple probe heads

`run_probe_encode_multihead.py` encodes each selected prefix once, computes
normalization and logit features once, and applies all score heads to the shared
features. Logit features use the backbone's input embedding weights, matching
training even when input embeddings and the output LM head are untied.

All heads must use the same frozen backbone/tokenizer, last-token pooling and
identical feature settings. The program checks feature settings and backbone
dimensions; these checks do not establish that different models with the same
dimensions have identical weights. Start the encode service using the same
local model directory passed as `--base-model-path`.

Create `configs/probe_heads.json`:

```json
{
  "seed1709": "outputs/g3_multiseed/seed1709_head",
  "seed5821": "outputs/g3_multiseed/seed5821_head",
  "seed9137": "outputs/g3_multiseed/seed9137_head",
  "seed27431": "outputs/g3_multiseed/seed27431_head"
}
```

From the repository root, after starting the encode service and building the
track-token map as described in the main README:

```bash
.venv/bin/python -m scripts.evaluation.prob.run_probe_encode_multihead \
  --input-dir "$RUN_DIR" --output-dir outputs/shared_probe \
  --heads-json configs/probe_heads.json --base-model-path "$MODEL_PATH" \
  --track-token-map outputs/track_token_id2text.json \
  --sgl-url http://127.0.0.1:30000 --track-token-stride 128 --batch-size 64
```

Here `MODEL_PATH` must point to a local safetensors checkpoint. Set
`--track-token-stride` to the same interval as the corresponding single-head
experiment. `--probe-point-mode stride` selects periodic token positions;
the default `track_token` selects tracked tokens separated by that interval.

Each `outputs/shared_probe/seed*/` directory contains
`generation_results.probed.jsonl` and `args.json`. Prune and evaluate each seed
using the regular online entrypoint, for example:

```bash
.venv/bin/python scripts/evaluation/postprocess/run_probe_prune_online.py \
  --input-dir outputs/shared_probe/seed1709 --output-dir outputs/pruned_seed1709 \
  --strategy ema_below --threshold 0.15 --alpha 0.15 --warmup-valid-points 2 \
  --tokenizer-path "$MODEL_PATH"
.venv/bin/python scripts/evaluation/analysis/run_eval.py \
  --dir outputs/pruned_seed1709 -k 32
```

The shared journal commits all heads for each prefix together. A file lock
prevents concurrent writers, interrupted final rows are recomputed, and repeating
the command skips completed prefixes. The manifest checks input/head hashes,
base-model path/config, token map and probe settings. Keep the base weights and
encode service fixed throughout a run. Context-limit skips retain `-1` scores.

For the full four-seed AIME25 evaluation, the wrapper connects shared encoding,
online EMA pruning and K=32 evaluation:

```bash
BASE_MODEL_PATH="$MODEL_PATH" BASE_INPUT_DIR="$RUN_DIR" \
  bash scripts/evaluation/postprocess/run_qwen_core_multiseed_aime25.sh
```

It reads heads from `outputs/g3_multiseed/` and writes results under
`outputs/g3_multiseed_aime25/`. Override `HEAD_ROOT`, `OUT_ROOT`,
`TRACK_TOKEN_MAP`, `SGL_URL`, `DEVICE` or `BATCH_SIZE` as needed. Positional
arguments select seeds, as in `...sh 42 1709 5821 9137 27431`. The input must
contain 32 outputs per question. Completed shared prefixes and pruning runs
are reused when repeating the same experiment; evaluation reports are recomputed.

For a small validation run, add `--max-records 8 --verify-single-head` and use a
separate output directory. This compares each head against its own scoring path
on the same hidden states, writes `reference/seed*/` outputs and reports maximum
probability errors. Disable verification for production timing because it
repeats feature computation.
