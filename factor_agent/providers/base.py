"""LLM completion is injected. Wire OpenAI / vLLM later; no vendor lock here."""

from __future__ import annotations

from collections.abc import Callable

# messages -> assistant text
CompleteFn = Callable[[list[dict[str, str]]], str]
