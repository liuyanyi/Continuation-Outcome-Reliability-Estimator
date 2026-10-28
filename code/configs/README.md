# Configs

`train_probe.json` is the four-step frozen-backbone smoke configuration used by
the main README. Set its model/tokenizer paths and training data paths before
using another model. `training/` and `evaluation/` contain larger reference
configurations; their dataset/output paths must be adapted to your experiment.

Tree generation: `tree_gen/tiny.json` is the smoke preset; `tree_gen/qwen3_4b.json`
is the larger reference preset. Pass `--config FILE` to `run_tree_gen_batch.py`.
Keys use CLI names with underscores; CLI overrides JSON, which overrides defaults.
Paths are relative to the working directory. Use `--print-config` to inspect the
merged settings. `--enable-refill` and `--enable-timeout` override the corresponding
`disable_*: true` config values. The larger preset uses the packaged `data/dapo/train.jsonl`; override
`--input-file data/dapo/test.jsonl` for held-out questions.
