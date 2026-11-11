# Confidence-Aware

Tree generation, frozen-backbone probe training/inference, and benchmark evaluation.
Run commands from the repository root (the directory containing `pyproject.toml`).
Code is distributed through GitHub; data is distributed separately through
Hugging Face. GPU steps require a CUDA GPU and a base model.

## Setup

Python 3.12, CUDA compatible with PyTorch 2.9.1/cu129, and a CUDA compiler for
FlashAttention are required. Install with `uv`:

```bash
uv sync --frozen --no-dev
export MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
```

`MODEL_PATH` may also be a local model directory. Use the same model/tokenizer for
all stages. If changing it, update `model_name_or_path` and `tokenizer_name` in
`configs/train_probe.json` too.

## Data preparation

**Download and prepare the data before running any pipeline below**, including
the tiny example. The GitHub code repository does not include the data files.

Download the files from [liuyanyi/CORE-Data](https://huggingface.co/datasets/liuyanyi/CORE-Data) into the repository’s data/ directory, preserving the directory structure; no extraction is needed.

| Data | Location after extraction | Purpose |
| --- | --- | --- |
| Tiny arithmetic examples | `data/examples/` | Pipeline smoke checks |
| DAPO train/test | `data/dapo/{train,test}.jsonl` | 15,711 / 1,687 training questions |
| AIME 2024/2025 | `data/benchmarks/aime24.jsonl`, `aime25.jsonl` | 30 questions each |
| LiveCodeBench cumulative v6 | `data/benchmarks/livecodebench_code_generation*.jsonl` | 1055 questions, two prompt variants and shared judge tests |
| GPQA Diamond | Prepared locally | Not included in the download |

Check the basic layout after extraction:

```bash
.venv/bin/python scripts/check_release.py
```

DAPO is already split by question ID with seed 42 and a 90% training threshold;
keep that assignment when building probe prefix datasets. LCB evaluation requires
the complete `.judge.jsonl` sidecar, which accounts for most of the data size.
The data download contains input questions and tests, not model weights, trained
probe heads or generated rollouts. Source notices and hashes are included with
the data. Prepare GPQA separately using the
[official acquisition and local conversion instructions](docs/EVALUATION.md#gpqa-local-preparation).

## Start the generation service

In a separate terminal, start generation (leave it running for tree and benchmark
generation):

```bash
bash scripts/start_sglang.sh generate Qwen/Qwen3-4B-Instruct-2507
```

Append SGLang options as needed, e.g. `--tp-size 2` or `--port 30001`.
The service binds to localhost by default.

## 1. Generate trees and training data

```bash
.venv/bin/python scripts/run_tree_gen_batch.py \
  --config configs/tree_gen/tiny.json --model-path "$MODEL_PATH"

.venv/bin/python scripts/score_batch_file.py outputs/tree_gen_tiny --score-only

.venv/bin/python scripts/build_seq_reg_dataset.py \
  --input-dir outputs/tree_gen_tiny \
  --questions-file data/examples/train_questions_tiny.jsonl \
  --output-dir outputs/seq_reg_tiny --tokenizer-name-or-path "$MODEL_PATH" \
  --prefer-precomputed-scores --train-ratio 0.75 --seed 42 --min-support 1
```

Tree settings live in `configs/tree_gen/tiny.json`; `configs/tree_gen/qwen3_4b.json`
is a larger reference preset (512 branches, 16000 tokens). CLI arguments override
config values; paths are relative to the release working directory. Append
`--print-config` to inspect resolved settings without loading a model.

The full DAPO question set is pre-split in `data/dapo/train.jsonl` (15,711) and
`data/dapo/test.jsonl` (1,687). The larger tree preset uses the DAPO training split.
See [data/dapo/README.md](data/dapo/README.md) for generating both splits and
building prefix datasets without changing their assignment.

The toy run uses support 1 to retain short prefixes. Research runs should use their
intended threshold (normally at least 2). Check that `train_seq.jsonl` and
`test_seq.jsonl` are nonempty: extraction needs newline boundaries, so regenerate
with a larger token budget if none are emitted. The fixed example IDs split into
three training questions and one validation question.

## 2. Train and use the probe

Stop generation before training if GPU memory is limited. The tiny config freezes
the backbone, uses last-token pooling and runs four training steps.

```bash
.venv/bin/python scripts/training/run_seq_regression.py configs/train_probe.json
.venv/bin/python scripts/training/infer_seq_regression.py \
  --model-path outputs/probe_model --input-file outputs/seq_reg_tiny/test_seq.jsonl \
  --output-file outputs/probe_predictions.jsonl
.venv/bin/python scripts/training/export_probe_head.py \
  --model-path outputs/probe_model --output-dir outputs/probe_head
```

For full training, adjust the step/epoch limits and data paths. Keep the backbone
frozen for the separate base-model-plus-head inference workflow below.

## 3. Benchmark: generate, score, prune and evaluate

Start generation again if stopped. Generate four rollouts per question and evaluate
the unpruned baseline:

```bash
.venv/bin/python scripts/evaluation/generation/run_generation_api.py \
  --api-url http://127.0.0.1:30000 --model-name base-model \
  --tokenizer-path "$MODEL_PATH" --data data/examples/benchmark_tiny.jsonl \
  --experiment-name tiny_baseline --output outputs/evaluation \
  --num-generation 4 --max-length 512 --eval-k 1,4
```

Set `RUN_DIR` to the directory printed by generation, containing
`generation_results.jsonl` and `args.json`:

```bash
export RUN_DIR=outputs/evaluation/tiny_baseline/benchmark_tiny/REPLACE_WITH_TIMESTAMP
```

Stop generation. Qwen3's SGLang `/encode` pooler must return raw last-token hidden
states without L2 normalization. Apply the idempotent patch **in the service's
environment**, then start encoding:

```bash
.venv/bin/python scripts/evaluation/prob/check_sglang_qwen3_encode_patch.py --apply
bash scripts/start_sglang.sh encode "$MODEL_PATH"
```

The patch supports SGLang 0.5.9 and saves a `.probe-original` backup beside the
modified file. Reapply after reinstalling SGLang and restart existing services.
Startup checks the patch. Probe-head LayerNorm is separate and remains controlled
by the checkpoint. For `nemotron_h`, skip the Qwen3 patch: the same startup script
loads the included `cognitive_probe.sglang_ext` override.

With encoding running in a separate terminal:

```bash
.venv/bin/python scripts/evaluation/prob/build_all_track_tokens.py \
  --tokenizer-path "$MODEL_PATH" --output outputs/track_token_id2text.json
.venv/bin/python scripts/evaluation/prob/run_probe_encode.py \
  --input-dir "$RUN_DIR" --output-dir "${RUN_DIR}.probe" \
  --track-token-map outputs/track_token_id2text.json \
  --head-model-path outputs/probe_head --base-model-path "$MODEL_PATH" \
  --sgl-url http://127.0.0.1:30000 --device cuda --track-token-stride 1
.venv/bin/python scripts/evaluation/postprocess/run_probe_prune_online.py \
  --input-dir "${RUN_DIR}.probe" --output-dir outputs/tiny_probe_prune \
  --input-strategy delegate_second_level --strategy consecutive_below \
  --threshold 0.15 --window-size 2 --tokenizer-path "$MODEL_PATH"
.venv/bin/python scripts/evaluation/analysis/run_eval.py \
  --dir outputs/tiny_probe_prune -k 1,4
```

Pruning replays online decisions against scored rollouts; it does not cancel live
server requests. Regenerate/fallback workflows, DeepConf, REFRAIN and bootstrap
scripts remain under `scripts/evaluation/`. The four-step head and threshold are
only for checking the workflow. When using logit features, `--base-model-path`
must be a local model directory because inference reads its weight files.

DeepConf uses higher-is-better confidence `-mean(top-k logprobs)`. Its replay
entry point takes the minimum group confidence per warmup trace, sets the cutoff
at the `(1 - keep_ratio)` percentile (90th percentile for `keep_ratio=0.1`), and
prunes a trace when a full group falls below that cutoff. Vote weights use raw
trace confidence; the CLI accepts only `--vote-weight raw`. Disjoint 32-token
groups, 16 complete warmup traces, inclusion of warmup votes, and a fixed budget
of 32 traces remain the defaults. Consensus is recorded but does not stop replay.
The generic `Maj@K` evaluator reports unweighted majority voting.

After the data preparation step, AIME 2024/2025 and cumulative LiveCodeBench v6
inputs are available under `data/benchmarks/`, with the complete LCB judge sidecar. See [EVALUATION.md](docs/EVALUATION.md) for input selection and the
code-judge command and local GPQA acquisition/conversion steps. GPQA questions and
answers are not bundled. Dataset provenance, checksums and source notices live in
[data/benchmarks/](data/benchmarks/README.md).

## CPU checks

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_release.py
```

These check examples, configuration, syntax and CPU behavior. GPU training and
end-to-end services need separate validation. See [VALIDATION.md](docs/VALIDATION.md)
for checked behavior and remaining limits.

## Packaging for maintainers

Publish the code archive contents to GitHub and the data archive plus its SHA-256
file to the Hugging Face dataset repository
[liuyanyi/CORE-Data](https://huggingface.co/datasets/liuyanyi/CORE-Data).
Pin the data revision when publishing paper results.

```bash
# Build both archives.
.venv/bin/python scripts/package_release.py --output-dir ..
# Rebuild only code after README/source edits, without recompressing the data.
.venv/bin/python scripts/package_release.py --output-dir .. --only code
```

The packager excludes caches, generated outputs and locally prepared GPQA data,
validates each ZIP, and writes a SHA-256 file for each archive. Downloaded data
and token maps are ignored by Git so they remain outside the code repository.
