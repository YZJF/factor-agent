from __future__ import annotations

from collections.abc import Callable
from typing import Any

from factor_agent.probe.integrity import check_case_integrity
from factor_agent.probe.metrics import compute_group_metrics
from factor_agent.routing.route_case import RoutingThresholds, route_case
from factor_agent.schemas import FactorCase
from factor_agent.split import group_key


def run_probe(
    cases: list[FactorCase],
    rollout_fn: Callable[[FactorCase, int], dict[str, Any]],
    *,
    k: int = 4,
    thresholds: RoutingThresholds | None = None,
) -> list[dict[str, Any]]:
    if k < 1:
        raise ValueError("probe k must be positive")
    classifications: list[dict[str, Any]] = []
    for case in cases:
        integrity = check_case_integrity(case)
        records = [rollout_fn(case, index) for index in range(k)] if integrity["ok"] else []
        metrics = compute_group_metrics(records, high_reward=(thresholds or RoutingThresholds()).high_reward)
        decision = route_case(metrics, integrity=integrity, thresholds=thresholds, split=case.split)
        classifications.append(
            {
                "case_id": case.case_id,
                "group_id": group_key(case),
                "split": case.split,
                **decision,
                "integrity": integrity,
                "metrics": metrics,
                "rollouts_ref": [record.get("artifact_path") for record in records if record.get("artifact_path")],
            }
        )
    return classifications
