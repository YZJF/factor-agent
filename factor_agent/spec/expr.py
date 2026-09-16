"""Expression whitelist. Time-series ops are defined as-of yesterday (t-1)."""

from __future__ import annotations

import re

# Bars already known at the close of day t. TS_* windows end at t-1, not t.
AS_OF_BARS = 1

ALLOWED_FUNCS = {
    "RANK",
    "TS_MEAN",
    "TS_STD",
    "TS_MIN",
    "TS_MAX",
    "TS_PCTCHANGE",
    "TS_SUM",
    "COUNT",
    "INDUSTRY_NEUTRALIZE",
    "DELAY",
    "DELTA",
    "LOG",
    "ABS",
    "SIGN",
}

ALLOWED_VARS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "return",
    "industry",
    "net_mf_amount",
}

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\$[A-Za-z_][A-Za-z0-9_]*")


def expr_ok(expr: str) -> tuple[bool, str]:
    if not expr or not expr.strip():
        return False, "empty expr"
    if any(bad in expr for bad in ("import ", "exec", "eval", "__", "os.", "open(")):
        return False, "forbidden token"
    for tok in _TOKEN.findall(expr):
        if tok.startswith("$"):
            if tok[1:] not in ALLOWED_VARS:
                return False, f"unknown var {tok}"
            continue
        if tok.isupper() and tok not in ALLOWED_FUNCS and tok not in {"E"}:
            if tok not in ALLOWED_FUNCS:
                return False, f"unknown func {tok}"
    return True, "ok"


_NUMBER = re.compile(r"(?<!\d)\d+(?!\d)")


def numbers_in_text(text: str) -> set[int]:
    # `\b` would fail here: CJK characters are word characters, so "过去20个交易日"
    # has no word boundary around the 20 and every Chinese report scored 0 grounding.
    return {int(x) for x in _NUMBER.findall(text)}
