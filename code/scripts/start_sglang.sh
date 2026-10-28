#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/start_sglang.sh generate|encode MODEL_PATH [SGLang options]" >&2
  exit 2
fi
mode="$1"
model_path="$2"
shift 2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
args=(--model-path "$model_path" --host 127.0.0.1 --port 30000 --trust-remote-code)
case "$mode" in
  generate) ;;
  encode)
    model_type="$(.venv/bin/python -c 'import sys; from transformers import AutoConfig; print(AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True).model_type)' "$model_path")"
    case "$model_type" in
      qwen3) .venv/bin/python scripts/evaluation/prob/check_sglang_qwen3_encode_patch.py ;;
      nemotron_h) export SGLANG_EXTERNAL_MODEL_PACKAGE=cognitive_probe.sglang_ext ;;
      *) echo "Unsupported encode model type: $model_type" >&2; exit 2 ;;
    esac
    args+=(--is-embedding)
    ;;
  *) echo "Mode must be generate or encode" >&2; exit 2 ;;
esac
exec .venv/bin/python -m sglang.launch_server "${args[@]}" "$@"
