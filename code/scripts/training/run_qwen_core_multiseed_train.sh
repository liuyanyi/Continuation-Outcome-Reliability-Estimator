#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
CONFIG_DIR="${CONFIG_DIR:-configs/training/g3}"
LOG_DIR="${LOG_DIR:-logs/g3_multiseed}"
NUM_PROCESSES="${NUM_PROCESSES:-3}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29646}"
mkdir -p "${LOG_DIR}"

# Seed 42 is the G3 reference; pass it explicitly to train it as well.
seeds=("$@")
if (( ${#seeds[@]} == 0 )); then
  seeds=(1709 5821 9137 27431)
fi

for seed in "${seeds[@]}"; do
  cfg="${CONFIG_DIR}/seed${seed}.json"
  train_dir=$(.venv/bin/python -c 'import json, sys; print(json.load(open(sys.argv[1]))["output_dir"])' "${cfg}")
  head_dir="${train_dir}_head"
  log_path="${LOG_DIR}/seed${seed}.log"

  echo "[g3-multiseed] train seed=${seed} config=${cfg}"
  .venv/bin/python -m accelerate.commands.launch --multi_gpu \
    --num_processes "${NUM_PROCESSES}" --num_machines 1 --machine_rank 0 \
    --mixed_precision bf16 --main_process_port "${MAIN_PROCESS_PORT}" \
    scripts/training/run_seq_regression.py "${cfg}" > "${log_path}" 2>&1

  echo "[g3-multiseed] export seed=${seed} -> ${head_dir}"
  .venv/bin/python scripts/training/export_probe_head.py \
    --model-path "${train_dir}" --output-dir "${head_dir}" \
    --trust-remote-code >> "${log_path}" 2>&1
done
