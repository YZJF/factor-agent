"""Contrast backtests. Need a real backtest hook; stubs record the request."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from factor_agent.evolve.periods import DateWindows, DEFAULT_WINDOWS
from factor_agent.schemas import FactorSpec

BacktestFn = Callable[..., dict[str, Any]]


def lag_one(
    spec: FactorSpec,
    backtest: BacktestFn,
    windows: DateWindows = DEFAULT_WINDOWS,
) -> dict[str, Any]:
    start, end = windows.score()
    try:
        return backtest(spec, start=start, end=end, shift_bars=1)
    except TypeError:
        return backtest(spec)


def shuffle_cross_section(
    spec: FactorSpec,
    backtest: BacktestFn,
    windows: DateWindows = DEFAULT_WINDOWS,
) -> dict[str, Any]:
    start, end = windows.score()
    try:
        return backtest(spec, start=start, end=end, shuffle=True)
    except TypeError:
        return backtest(spec)
