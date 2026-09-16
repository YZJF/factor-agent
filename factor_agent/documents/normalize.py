from __future__ import annotations

import re

_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_NUMBER = re.compile(r"[零〇一二两三四五六七八九十百千]+")


def chinese_number_to_int(value: str) -> int | None:
    if not value:
        return None
    if all(char in _DIGITS for char in value):
        return int("".join(str(_DIGITS[char]) for char in value))
    total = current = 0
    for char in value:
        if char in _DIGITS:
            current = _DIGITS[char]
        elif char in _UNITS:
            unit = _UNITS[char]
            total += (current or 1) * unit
            current = 0
        else:
            return None
    return total + current


def normalize_chinese_numbers(text: str) -> str:
    """Append Arabic forms without destroying the source wording."""

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        number = chinese_number_to_int(raw)
        return f"{raw}({number})" if number is not None else raw

    return _CN_NUMBER.sub(replace, text)
