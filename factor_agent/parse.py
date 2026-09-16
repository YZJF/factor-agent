"""Pull a FactorSpec JSON out of model text."""

from __future__ import annotations

import json
import re
from typing import Any

from factor_agent.schemas import FactorSpec

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def parse_spec(text: str) -> FactorSpec | None:
    if not text:
        return None
    blob = text.strip()
    m = _FENCE.search(blob)
    if m:
        blob = m.group(1)
    else:
        start, end = blob.find("{"), blob.rfind("}")
        if start < 0 or end <= start:
            return None
        blob = blob[start : end + 1]
    try:
        data: dict[str, Any] = json.loads(blob)
        return FactorSpec.model_validate(data)
    except Exception:
        return None
