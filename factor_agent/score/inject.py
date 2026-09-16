"""Build negative specs from gold. Used to check the verifier, not as RM pairs."""

from __future__ import annotations

from factor_agent.schemas import FactorSpec

_FREQ_FLIP = {"daily": "monthly", "weekly": "daily", "monthly": "weekly"}


def corrupt(gold: FactorSpec, kind: str) -> FactorSpec:
    spec = gold.model_copy(deep=True)
    if kind == "drop_neutralization":
        spec.neutralization = "none"
    elif kind == "wrong_frequency":
        spec.frequency = _FREQ_FLIP[gold.frequency]
    elif kind == "wrong_window":
        spec.windows = [w * 3 if w else 60 for w in (gold.windows or [20])]
        spec.expr = (
            spec.expr.replace(str(gold.windows[0]), str(spec.windows[0]))
            if gold.windows
            else spec.expr
        )
    elif kind == "hallucinated_expr":
        spec.expr = "RANK($close)"
        spec.windows = []
        spec.universe = "UNKNOWN"
    elif kind == "illegal_expr":
        spec.expr = "import os; os.system('x')"
    else:
        raise ValueError(kind)
    return spec


INJECT_KINDS = (
    "drop_neutralization",
    "wrong_frequency",
    "wrong_window",
    "hallucinated_expr",
    "illegal_expr",
)
