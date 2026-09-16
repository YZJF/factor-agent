#!/usr/bin/env bash
# LoRA SFT for one stage. Extract and evolve adapters both start from MODEL=
# (the frozen base). Do not pass the extract adapter as MODEL for evolve.
#
#   python -m factor_agent build-sft --stage extract --out data/sft/extract_train.jsonl
#   python -m factor_agent build-sft --stage evolve --out data/sft/evolve_train.jsonl
#   python scripts/jsonl_to_parquet.py --in data/sft/extract_train.jsonl --out data/sft/extract_train.parquet
#   MODEL=/path/to/Qwen3-4B STAGE=extract bash scripts/run_sft.sh
#   MODEL=/path/to/Qwen3-4B STAGE=evolve bash scripts/run_sft.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c "import verl" 2>/dev/null; then
  echo "verl is not importable. pip install verl in this env; do not copy verl into this repo."
  exit 1
fi

MODEL="${MODEL:-}"
if [[ -z "$MODEL" ]]; then
  echo "set MODEL=/path/to/Qwen3-4B (frozen base, shared by both adapters)"
  exit 1
fi

STAGE="${STAGE:-extract}"
if [[ "$STAGE" != "extract" && "$STAGE" != "evolve" ]]; then
  echo "STAGE must be extract or evolve"
  exit 1
fi

LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET="${LORA_TARGET:-all-linear}"

if [[ "$STAGE" == "extract" ]]; then
  TRAIN_FILE="${TRAIN_FILE:-data/sft/extract_train.parquet}"
  SAVE_PATH="${SAVE_PATH:-checkpoints/extract_lora/sft}"
else
  TRAIN_FILE="${TRAIN_FILE:-data/sft/evolve_train.parquet}"
  SAVE_PATH="${SAVE_PATH:-checkpoints/evolve_lora/sft}"
fi
VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
N_GPUS="${N_GPUS:-1}"
NNODES="${NNODES:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LR="${LR:-1e-5}"
LOGGER="${LOGGER:-console}"

mkdir -p "$SAVE_PATH" runs/train_logs
"$PYTHON" -m torch.distributed.run \
  --standalone --nnodes="$NNODES" --nproc_per_node="$N_GPUS" \
  --module verl.trainer.sft_trainer \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.messages_key=messages \
  data.tools_key=tools \
  data.enable_thinking_key=enable_thinking \
  data.enable_thinking_default=False \
  data.ignore_input_ids_mismatch=True \
  data.pad_mode=no_padding \
  data.max_length="$MAX_LENGTH" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  data.use_dynamic_bsz=False \
  engine=fsdp \
  engine.model_dtype=bfloat16 \
  model.path="$MODEL" \
  model.lora_rank="$LORA_RANK" \
  model.lora_alpha="$LORA_ALPHA" \
  model.target_modules="$LORA_TARGET" \
  optim.lr="$LR" \
  trainer.default_local_dir="$SAVE_PATH" \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.logger="[$LOGGER]" \
  "$@"
