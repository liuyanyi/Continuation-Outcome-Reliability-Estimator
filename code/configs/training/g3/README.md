# G3 training

These configurations use `soft_bce` weighted by `sqrt(N)`, last-token pooling,
a three-layer score head with hidden size 8192, normalized hidden states and
margin/entropy features. The backbone is frozen. Loss weighting supports
`none`, `sqrt_n` and `n`.

Prepare `data/seq_reg_dataset_v1_optimized_8k/{train,test}_seq.jsonl` first.
Set the model/tokenizer paths in each configuration to the same backbone used
for tree generation and encoding. The default is Qwen3-4B-Instruct-2507.

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/training/run_qwen_core_multiseed_train.sh
# Include the seed-42 reference when training all five heads:
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/training/run_qwen_core_multiseed_train.sh 42 1709 5821 9137 27431
```

The default run trains seeds 1709, 5821, 9137 and 27431, then exports each head to
`outputs/g3_multiseed/seed<seed>_head`. All configurations fix `data_seed=42`,
train for one epoch and evaluate the full held-out split. They use three GPU
processes, per-device batch size 1 and gradient accumulation 85 (effective
batch size 255). Changing `NUM_PROCESSES` changes that effective batch size
unless accumulation is adjusted. `CONFIG_DIR`, `LOG_DIR` and
`MAIN_PROCESS_PORT` can be set in the environment. Existing output directories
are protected by `overwrite_output_dir=false`.

Use the exported heads with [shared encoding](../../../scripts/evaluation/prob/README_multihead.md).
The AIME25 training-seed comparison uses K=32 and online EMA pruning with
`threshold=0.15`, `alpha=0.15`, `warmup_valid_points=2`, and abstention scoring.
Summary standard deviations use `ddof=0`.
