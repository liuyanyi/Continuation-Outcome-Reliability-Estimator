#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
: "${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to the local Qwen3 backbone directory}"
: "${BASE_INPUT_DIR:?Set BASE_INPUT_DIR to the AIME25 run with 32 outputs per question}"
SGL_URL="${SGL_URL:-http://127.0.0.1:30000}"
HEAD_ROOT="${HEAD_ROOT:-outputs/g3_multiseed}"
OUT_ROOT="${OUT_ROOT:-outputs/g3_multiseed_aime25}"
TRACK_TOKEN_MAP="${TRACK_TOKEN_MAP:-outputs/track_token_id2text.json}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
mkdir -p "$OUT_ROOT"
seeds=("$@")
if (( ${#seeds[@]} == 0 )); then
  seeds=(1709 5821 9137 27431)
fi

.venv/bin/python - "$HEAD_ROOT" "$OUT_ROOT/heads.json" "${seeds[@]}" <<'PY'
import json
import sys
from pathlib import Path
head_root, destination, *seeds = sys.argv[1:]
heads = {f'seed{int(seed)}': str(Path(head_root) / f'seed{int(seed)}_head') for seed in seeds}
for name, directory in heads.items():
    for filename in ('config.json', 'model.safetensors'):
        if not (Path(directory) / filename).is_file():
            raise FileNotFoundError(f'{name}: missing {directory}/{filename}')
Path(destination).write_text(json.dumps(heads, indent=2) + '\n')
PY

.venv/bin/python -m scripts.evaluation.prob.run_probe_encode_multihead \
  --input-dir "$BASE_INPUT_DIR" --output-dir "$OUT_ROOT/shared_encoding" \
  --heads-json "$OUT_ROOT/heads.json" --base-model-path "$BASE_MODEL_PATH" \
  --sgl-url "$SGL_URL" --track-token-map "$TRACK_TOKEN_MAP" \
  --track-token-stride 128 --batch-size "$BATCH_SIZE" --device "$DEVICE"

for seed in "${seeds[@]}"; do
  run_dir="$OUT_ROOT/seed${seed}/ema_t0p15_a0p15_w2"
  if [[ ! -f "$run_dir/final_outputs.jsonl" ]]; then
    .venv/bin/python scripts/evaluation/postprocess/run_probe_prune_online.py \
    --input-dir "$OUT_ROOT/shared_encoding/seed${seed}" --output-dir "$run_dir" \
    --strategy ema_below --threshold 0.15 --alpha 0.15 --warmup-valid-points 2 \
    --min-keep-tokens 0 --input-group-key example_id --tokenizer-path "$BASE_MODEL_PATH"
  fi
  .venv/bin/python scripts/evaluation/analysis/run_eval.py --dir "$run_dir" -k 32
done
