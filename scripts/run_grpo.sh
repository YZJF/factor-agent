#!/usr/bin/env bash
# LoRA GRPO for one stage. MODEL is the frozen base. ADAPTER_PATH should be
# that stage's SFT adapter. Evolve must not load extract_lora.
#
#   STAGE=extract MODEL=/path/to/Qwen3-4B ADAPTER_PATH=checkpoints/extract_lora/sft bash scripts/run_grpo.sh
#   STAGE=evolve  MODEL=/path/to/Qwen3-4B ADAPTER_PATH=checkpoints/evolve_lora/sft \
#     FACTOR_BACKTEST_URL=http://localhost:8001/backtest bash scripts/run_grpo.sh
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
  echo "set MODEL=/path/to/Qwen3-4B (frozen base)"
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
ADAPTER_PATH="${ADAPTER_PATH:-}"

if [[ "$STAGE" == "extract" ]]; then
  TRAIN_FILE="${TRAIN_FILE:-data/rl/extract_train.parquet}"
  SAVE_PATH="${SAVE_PATH:-checkpoints/extract_lora/grpo}"
  if [[ -z "$ADAPTER_PATH" && -d checkpoints/extract_lora/sft ]]; then
    ADAPTER_PATH="checkpoints/extract_lora/sft"
  fi
else
  TRAIN_FILE="${TRAIN_FILE:-data/rl/evolve_train.parquet}"
  SAVE_PATH="${SAVE_PATH:-checkpoints/evolve_lora/grpo}"
  if [[ -z "$ADAPTER_PATH" && -d checkpoints/evolve_lora/sft ]]; then
    ADAPTER_PATH="checkpoints/evolve_lora/sft"
  fi
fi

VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
N_GPUS="${N_GPUS:-1}"
NNODES="${NNODES:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-4}"
MAX_PROMPT="${MAX_PROMPT:-2048}"
MAX_RESPONSE="${MAX_RESPONSE:-1024}"
LR="${LR:-1e-6}"
LOGGER="${LOGGER:-console}"
AGENT_LOOP_CONFIG="${AGENT_LOOP_CONFIG:-$ROOT/configs/verl_agent_loop.yaml}"
ROLLOUT_CORRECTION="${ROLLOUT_CORRECTION:-0}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"
REWARD_PATH="${REWARD_PATH:-$ROOT/factor_agent/train/verl_reward.py}"
REWARD_NAME="${REWARD_NAME:-compute_score_fn}"

mkdir -p "$SAVE_PATH" runs/train_logs
cmd=(
  "$PYTHON" -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
  data.prompt_key=prompt
  data.max_prompt_length="$MAX_PROMPT"
  data.max_response_length="$MAX_RESPONSE"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  actor_rollout_ref.model.path="$MODEL"
  actor_rollout_ref.model.lora_rank="$LORA_RANK"
  actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"
  actor_rollout_ref.model.target_modules="$LORA_TARGET"
  actor_rollout_ref.actor.optim.lr="$LR"
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  custom_reward_function.path="$REWARD_PATH"
  custom_reward_function.name="$REWARD_NAME"
  trainer.n_gpus_per_node="$N_GPUS"
  trainer.nnodes="$NNODES"
  trainer.total_training_steps="$TOTAL_STEPS"
  trainer.default_local_dir="$SAVE_PATH"
  trainer.logger="[$LOGGER]"
  trainer.val_before_train=False
)

if [[ -n "$ADAPTER_PATH" ]]; then
  cmd+=(actor_rollout_ref.model.lora_adapter_path="$ADAPTER_PATH")
fi

if [[ "$STAGE" == "evolve" ]]; then
  cmd+=(
    actor_rollout_ref.rollout.multi_turn.enable=True
    "actor_rollout_ref.rollout.agent.agent_loop_config_path=$AGENT_LOOP_CONFIG"
    actor_rollout_ref.rollout.agent.default_agent_loop=factor_evolve_agent
  )
fi

if [[ "$ROLLOUT_CORRECTION" == "1" ]]; then
  if [[ "$STAGE" != "evolve" ]]; then
    echo "rollout correction is only supported for STAGE=evolve" >&2
    exit 1
  fi
  cmd+=(
    actor_rollout_ref.rollout.calculate_log_probs=True
    algorithm.rollout_correction.rollout_is=sequence
    "algorithm.rollout_correction.rollout_is_threshold=$ROLLOUT_IS_THRESHOLD"
    algorithm.rollout_correction.rollout_is_batch_normalize=False
    algorithm.rollout_correction.bypass_mode=False
  )
fi

cmd+=("$@")
"${cmd[@]}"
