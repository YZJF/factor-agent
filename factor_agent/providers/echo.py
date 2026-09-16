"""Dry-run provider: echo the gold spec. Used to test the pipeline without an API."""

from __future__ import annotations

from factor_agent.providers.base import CompleteFn
from factor_agent.schemas import FactorCase, FactorSpec


def gold_echo(spec: FactorSpec) -> CompleteFn:
    text = spec.model_dump_json()

    def complete(_messages: list[dict[str, str]]) -> str:
        return text

    return complete


def echo_complete(case: FactorCase) -> CompleteFn:
    return gold_echo(case.gold)
