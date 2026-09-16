"""Two LoRA adapters, one per stage. Both grow from the same frozen base.

Extract and evolve rewards/tools are nearly orthogonal, so a second full-weight
GRPO run would overwrite the first. Adapters isolate the updates. Never continue
the evolve adapter from the extract adapter — that recouples the tasks.
"""

from __future__ import annotations

ADAPTERS = {
    "extract": {
        "adapter_name": "extract_lora",
        "sft_jsonl": "data/sft/extract_train.jsonl",
        "sft_parquet": "data/sft/extract_train.parquet",
        "grpo_jsonl": "data/rl/extract_train.jsonl",
        "grpo_parquet": "data/rl/extract_train.parquet",
        "sft_dir": "checkpoints/extract_lora/sft",
        "grpo_dir": "checkpoints/extract_lora/grpo",
    },
    "evolve": {
        "adapter_name": "evolve_lora",
        "sft_jsonl": "data/sft/evolve_train.jsonl",
        "sft_parquet": "data/sft/evolve_train.parquet",
        "grpo_jsonl": "data/rl/evolve_train.jsonl",
        "grpo_parquet": "data/rl/evolve_train.parquet",
        "sft_dir": "checkpoints/evolve_lora/sft",
        "grpo_dir": "checkpoints/evolve_lora/grpo",
    },
}

DEFAULT_RANK = 32
DEFAULT_ALPHA = 64
DEFAULT_TARGET = "all-linear"


def paths(stage: str) -> dict[str, str]:
    if stage not in ADAPTERS:
        raise ValueError(f"unknown LoRA stage: {stage}")
    return ADAPTERS[stage]


def sft_overrides(rank: int = DEFAULT_RANK, alpha: int = DEFAULT_ALPHA, target: str = DEFAULT_TARGET) -> list[str]:
    return [
        f"model.lora_rank={rank}",
        f"model.lora_alpha={alpha}",
        f"model.target_modules={target}",
    ]


def grpo_overrides(
    rank: int = DEFAULT_RANK,
    alpha: int = DEFAULT_ALPHA,
    target: str = DEFAULT_TARGET,
    adapter_path: str = "",
) -> list[str]:
    flags = [
        f"actor_rollout_ref.model.lora_rank={rank}",
        f"actor_rollout_ref.model.lora_alpha={alpha}",
        f"actor_rollout_ref.model.target_modules={target}",
    ]
    if adapter_path:
        flags.append(f"actor_rollout_ref.model.lora_adapter_path={adapter_path}")
    return flags
