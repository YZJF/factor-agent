"""Compatibility shim. Use factor_agent.score.verifier."""

from factor_agent.score.verifier import compute_score, score_prediction

__all__ = ["compute_score", "score_prediction"]
