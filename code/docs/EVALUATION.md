# Benchmark evaluation

Run from the repository root after the [data preparation](../README.md#data-preparation)
and service setup steps. AIME and LCB are downloaded from Hugging Face, not shipped
in the GitHub code repository.
AIME and LCB inputs are ready for `--data`. GPQA is acquired and converted locally.

| Input under `data/benchmarks/` | Questions | Evaluation |
| --- | ---: | --- |
| `aime24.jsonl` | 30 | Math answer matching |
| `aime25.jsonl` | 30 | Math answer matching |
| `gpqa_diamond.jsonl` (local only; not bundled) | 198 | Answer-letter matching |
| `livecodebench_code_generation.jsonl` | 1055 | LCB cumulative `release_v6`, thinking prompt |
| `livecodebench_code_generation_instruct.jsonl` | 1055 | Same LCB questions/tests, instruct prompt |

LCB `release_v6` combines `test.jsonl` through `test6.jsonl`; it is not only the
175-question `v6` increment. Both prompt variants share the complete
`livecodebench_code_generation.judge.jsonl` sidecar (about 4.2 GiB). Keep it with
the inputs; compact generation files alone do not contain the private tests.
See [data provenance](../data/benchmarks/README.md) and `manifest.json` for counts
and SHA-256 checksums. Preserve prompts and record the GPQA conversion seed.

## AIME and GPQA

```bash
.venv/bin/python scripts/evaluation/generation/run_generation_api.py \
  --api-url http://127.0.0.1:30000 --model-name base-model \
  --tokenizer-path "$MODEL_PATH" --data data/benchmarks/aime24.jsonl \
  --experiment-name aime24 --output outputs/evaluation \
  --num-generation 4 --max-length 16000 --eval-k 1,4
```

Substitute `aime25.jsonl` or `gpqa_diamond.jsonl` as needed. Generation automatically
runs `run_eval.py`. These sampling settings are examples, not paper-result settings.
For existing or pruned runs:

```bash
.venv/bin/python scripts/evaluation/analysis/run_eval.py --dir "$RUN_DIR" -k 1,4
```

## GPQA local preparation

GPQA question/answer files are not bundled. Its upstream card lists CC BY 4.0
and separately asks users not to reveal examples online. Obtain access directly
at https://huggingface.co/datasets/Idavidrein/gpqa and follow the current upstream
conditions. Download `gpqa_diamond.csv` to `data/private/gpqa_diamond.csv` locally.

```bash
.venv/bin/python scripts/data/process_gpqa_diamond.py \
  -i data/private/gpqa_diamond.csv \
  -o data/benchmarks/gpqa_diamond.jsonl --seed 42
```

The converter shuffles answer choices deterministically with this seed and writes
198 rows for the full Diamond subset. Then use the AIME generation command above
with `--data data/benchmarks/gpqa_diamond.jsonl` and `--experiment-name gpqa_diamond`.
The fixed seed makes this preparation reproducible.

Both the downloaded CSV and converted JSONL are excluded by `.gitignore` and the
release packager. Keep model outputs containing GPQA prompts local too.

## LiveCodeBench v6

Generate with the appropriate prompt profile and skip automatic math evaluation:

```bash
.venv/bin/python scripts/evaluation/generation/run_generation_api.py \
  --api-url http://127.0.0.1:30000 --model-name base-model \
  --tokenizer-path "$MODEL_PATH" \
  --data data/benchmarks/livecodebench_code_generation.jsonl \
  --experiment-name lcb_v6 --output outputs/evaluation \
  --num-generation 4 --max-length 16000 --no-run-eval
```

Set `RUN_DIR` to the printed output directory, then run the code judge:

```bash
.venv/bin/python scripts/evaluation/analysis/run_eval_livebench.py \
  --dir "$RUN_DIR" -k 1,4 --run-judge \
  --judge-manifest data/benchmarks/livecodebench_code_generation.judge.jsonl \
  --max-workers 4 --timeout-s 6
```

The same command evaluates a pruned LCB run. `--run-judge` executes generated
Python against the test cases; omitting it only summarizes existing judge results.
`--public-only` is diagnostic and does not reproduce full benchmark scoring.
The report includes correctness, token usage and `lcb_pass@1`.

## Scoring and token accounting

Pure pruning is an abstention: its slot scores zero for Mean/Pass, remains in
the K denominator, and casts no majority vote. Its generated prefix still costs
tokens. This applies to math, LiveCodeBench and their bootstrap summaries.
A REFRAIN direct-answer continuation is scored normally, with both the retained
reasoning prefix and the new answer counted in output-token cost.

REFRAIN aligns steps against the original generated token bytes, including
split UTF-8 characters and hidden special tokens. Inputs that cannot be aligned
exactly are rejected.

Probe scoring and pruning follow the README. DeepConf, REFRAIN and bootstrap
entrypoints remain under `scripts/evaluation/`; use their `--help` for optional runs.
