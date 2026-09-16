"""Stage B may change expr / windows only."""

from __future__ import annotations

from factor_agent.schemas import FactorSpec

FROZEN_FIELDS = ("universe", "frequency", "neutralization")


def spec_drift(seed: FactorSpec, pred: FactorSpec) -> list[str]:
    return [k for k in FROZEN_FIELDS if getattr(pred, k) != getattr(seed, k)]
