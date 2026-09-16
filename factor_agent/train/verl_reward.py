"""Thin verl-style adapter. Do not import verl here."""

from __future__ import annotations

from factor_agent.score.verifier import compute_score


def compute_score_fn(data_source, solution_str, ground_truth, extra_info=None):
    """Same signature as typical verl reward_score hooks."""
    return float(compute_score(data_source, solution_str, ground_truth, extra_info))
