#!/usr/bin/env bash
set -euo pipefail

# Recommended generation config for reducing sample count while preserving diversity.
# You can override vars via environment, e.g.:
#   OUTPUT_DIR=data/seq_reg_dataset_v1_optimized ./scripts/run_build_seq_reg_optimized.sh

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
INPUT_DIR="${INPUT_DIR:-data/tree_gen_results_v2}"
QUESTIONS_FILE="${QUESTIONS_FILE:-data/dapo_question_with_suffix_17398.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data/seq_reg_dataset_v1_optimized_8k}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-Qwen/Qwen3-4B-Instruct-2507}"

TRAIN_RATIO="${TRAIN_RATIO:-0.9}"
SEED="${SEED:-42}"

# Core reduction knobs.
MIN_SUPPORT="${MIN_SUPPORT:-8}"
MIN_PREFIX_GAP_TOKENS="${MIN_PREFIX_GAP_TOKENS:-128}"
LABEL_ROUND_DECIMALS="${LABEL_ROUND_DECIMALS:-3}"
MAX_SAMPLES_PER_QUESTION="${MAX_SAMPLES_PER_QUESTION:-1500}"
MAX_GENERATED_PREFIX_TOKENS="${MAX_GENERATED_PREFIX_TOKENS:-8192}"
SPLIT_POINT_MODE="${SPLIT_POINT_MODE:-newline}"
TRACK_CHARACTER_MATCH_MODE="${TRACK_CHARACTER_MATCH_MODE:-token}"
FALLBACK_SPLIT_POINT_MODE_IF_EMPTY="${FALLBACK_SPLIT_POINT_MODE_IF_EMPTY:-}"
FALLBACK_TRACK_CHARACTER_MATCH_MODE_IF_EMPTY="${FALLBACK_TRACK_CHARACTER_MATCH_MODE_IF_EMPTY:-}"
FALLBACK_MIN_SUPPORT_IF_EMPTY="${FALLBACK_MIN_SUPPORT_IF_EMPTY:-0}"

# Diversity-cap strategy knobs.
MAX_SAMPLES_STRATEGY="${MAX_SAMPLES_STRATEGY:-diversity}"
CAP_LABEL_BINS="${CAP_LABEL_BINS:-10}"
CAP_LENGTH_BINS="${CAP_LENGTH_BINS:-5}"
CAP_LABEL_RATIO="${CAP_LABEL_RATIO:-0.5}"
CAP_LENGTH_RATIO="${CAP_LENGTH_RATIO:-0.3}"

echo "[INFO] Building seq-reg dataset with optimized reduction config"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/build_seq_reg_dataset.py \
  --input-dir "${INPUT_DIR}" \
  --questions-file "${QUESTIONS_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --tokenizer-name-or-path "${TOKENIZER_NAME_OR_PATH}" \
  --prefer-precomputed-scores \
  --skip-unscored-files \
  --train-ratio "${TRAIN_RATIO}" \
  --seed "${SEED}" \
  --min-support "${MIN_SUPPORT}" \
  --split-point-mode "${SPLIT_POINT_MODE}" \
  --track-character-match-mode "${TRACK_CHARACTER_MATCH_MODE}" \
  --fallback-split-point-mode-if-empty "${FALLBACK_SPLIT_POINT_MODE_IF_EMPTY}" \
  --fallback-track-character-match-mode-if-empty "${FALLBACK_TRACK_CHARACTER_MATCH_MODE_IF_EMPTY}" \
  --fallback-min-support-if-empty "${FALLBACK_MIN_SUPPORT_IF_EMPTY}" \
  --min-prefix-gap-tokens "${MIN_PREFIX_GAP_TOKENS}" \
  --label-round-decimals "${LABEL_ROUND_DECIMALS}" \
  --max-generated-prefix-tokens "${MAX_GENERATED_PREFIX_TOKENS}" \
  --max-samples-per-question "${MAX_SAMPLES_PER_QUESTION}" \
  --max-samples-strategy "${MAX_SAMPLES_STRATEGY}" \
  --cap-label-bins "${CAP_LABEL_BINS}" \
  --cap-length-bins "${CAP_LENGTH_BINS}" \
  --cap-label-ratio "${CAP_LABEL_RATIO}" \
  --cap-length-ratio "${CAP_LENGTH_RATIO}"

echo "[INFO] Done. See ${OUTPUT_DIR}/summary.json"
