"""Qlib-backed, look-ahead-safe factor portfolio backtest."""

from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from factor_agent.backtest.data import MarketDataManifest, validate_manifest
from factor_agent.backtest.expression import (
    Call,
    ExpressionError,
    Variable,
    evaluate_expression,
    max_lookback,
    parse_expression,
    required_fields,
    stack_all,
)
from factor_agent.spec.expr import AS_OF_BARS


class EngineConfig(BaseModel):
    manifest_path: str = "data/market/manifest.json"
    topk_ratio: float = Field(default=0.10, gt=0.0, le=1.0)
    open_cost: float = Field(default=0.0005, ge=0.0)
    close_cost: float = Field(default=0.0015, ge=0.0)
    annualization: int = Field(default=252, gt=0)
    shuffle_seed: int = 20260913


FrameLoader = Callable[[str, set[str], str, str], Any]


def _pandas():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError('backtest support requires: pip install -e ".[backtest]"') from exc
    return np, pd


class QlibFrameLoader:
    """Read fields and dynamic index membership from a local Qlib provider."""

    def __init__(self, provider_uri: str | Path):
        self.provider_uri = str(Path(provider_uri).resolve())
        self._initialized = False

    def _initialize(self) -> None:
        if self._initialized:
            return
        try:
            import qlib
            from qlib.constant import REG_CN
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise RuntimeError('Qlib loader requires: pip install -e ".[backtest]"') from exc
        qlib.init(provider_uri=self.provider_uri, region=REG_CN)
        self._initialized = True

    def __call__(self, universe: str, fields: set[str], start: str, end: str):
        self._initialize()
        from qlib.data import D

        market = universe.lower()
        expressions = [f"${field}" for field in sorted(fields)]
        try:
            frame = D.features(
                D.instruments(market=market),
                expressions,
                start_time=start,
                end_time=end,
                freq="day",
                disk_cache=0,
            )
        except Exception as exc:
            raise RuntimeError(f"Qlib failed to load {universe}: {exc}") from exc
        if frame.empty:
            raise RuntimeError(f"Qlib returned no rows for {universe} in {start}..{end}")
        frame = frame.rename(columns={column: column.lstrip("$") for column in frame.columns})
        frame.index = frame.index.set_names(["instrument", "datetime"])
        frame = frame.reorder_levels(["datetime", "instrument"]).sort_index()
        return frame


class QlibBacktestEngine:
    SUPPORTED_UNIVERSES = {"CSI300", "CSI500"}

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        frame_loader: FrameLoader | None = None,
        manifest: MarketDataManifest | None = None,
    ):
        self.config = config or EngineConfig()
        if manifest is None and frame_loader is None:
            manifest = validate_manifest(
                self.config.manifest_path,
                required_start="2017-01-01",
                required_end="2024-12-31",
            )
        self.manifest = manifest
        self.frame_loader = frame_loader or QlibFrameLoader(manifest.qlib_uri)  # type: ignore[union-attr]

    @staticmethod
    def validate_expression(expr: str) -> dict[str, Any]:
        try:
            node = parse_expression(expr)
            return {
                "ok": True,
                "fields": sorted(required_fields(node)),
                "max_lookback": max_lookback(node),
            }
        except ExpressionError as exc:
            return {"ok": False, "error": str(exc)}

    def _load_frame(self, universe: str, fields: set[str], start: str, end: str):
        warmup_start = self.manifest.warmup_start if self.manifest else start
        return self.frame_loader(universe, fields, warmup_start, end)

    def backtest(
        self,
        *,
        exprs: dict[str, str],
        backtest_start_time: str,
        backtest_end_time: str,
        update_freq: int = 5,
        stock_pool: str = "CSI500",
        position_size: float = 1.0,
        max_pos_each_stock: float = 0.2,
        industry_neutralization: str | None = None,
        shift_bars: int = 0,
        shuffle_cross_section: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        np, pd = _pandas()
        universe = stock_pool.upper()
        if universe not in self.SUPPORTED_UNIVERSES:
            raise ValueError(f"unsupported universe: {stock_pool}; supported: CSI300, CSI500")
        if len(exprs) != 1:
            raise ValueError("MVP backtest accepts exactly one factor expression")
        if update_freq < 1:
            raise ValueError("update_freq must be positive")
        if not 0.0 < position_size <= 1.0:
            raise ValueError("position_size must be in (0, 1]")
        if not 0.0 < max_pos_each_stock <= 1.0:
            raise ValueError("max_pos_each_stock must be in (0, 1]")
        factor_name, expression = next(iter(exprs.items()))
        node = parse_expression(expression)
        max_lookback(node)
        fields = required_fields(node) | {"return", "tradestatus"}
        neutralization = (industry_neutralization or "none").lower()
        if neutralization in {"size", "industry_size"}:
            raise ValueError("MVP backtest does not support size neutralization")
        if neutralization not in {"none", "industry"}:
            raise ValueError(f"unsupported neutralization: {industry_neutralization}")
        if neutralization == "industry":
            fields.add("industry")
        frame = self._load_frame(universe, fields, backtest_start_time, backtest_end_time)
        if neutralization == "industry" and "INDUSTRY_NEUTRALIZE" not in expression.upper():
            node = Call("INDUSTRY_NEUTRALIZE", (node, Variable("industry")))
        score = evaluate_expression(node, frame)
        total_shift = AS_OF_BARS + shift_bars
        if total_shift:
            score = stack_all(score.unstack("instrument").shift(total_shift)).reindex(score.index)
        if shuffle_cross_section:
            rng = np.random.default_rng(self.config.shuffle_seed)

            def shuffle(values: Any):
                array = values.to_numpy(copy=True)
                valid = np.flatnonzero(~pd.isna(array))
                array[valid] = rng.permutation(array[valid])
                return pd.Series(array, index=values.index)

            score = score.groupby(level="datetime", group_keys=False).apply(shuffle)
            score.index = frame.index

        start = pd.Timestamp(backtest_start_time)
        end = pd.Timestamp(backtest_end_time)
        selected = frame.loc[
            (frame.index.get_level_values("datetime") >= start)
            & (frame.index.get_level_values("datetime") <= end)
        ].copy()
        selected["score"] = score.reindex(selected.index)
        selected = selected[selected["tradestatus"].fillna(0) > 0]
        if selected.empty:
            raise RuntimeError(f"no tradable rows in backtest window {backtest_start_time}..{backtest_end_time}")

        returns = selected["return"].unstack("instrument").sort_index()
        scores = selected["score"].unstack("instrument").reindex(returns.index)
        dates = returns.index
        previous = pd.Series(0.0, index=returns.columns)
        net_returns: list[float] = []
        gross_returns: list[float] = []
        benchmark_returns: list[float] = []
        turnovers: list[float] = []
        holdings: list[int] = []

        for day_index, current_date in enumerate(dates):
            day_returns = returns.loc[current_date]
            valid_returns = day_returns.dropna()
            rebalance = day_index % update_freq == 0
            target = previous.copy()
            if rebalance:
                candidates = scores.loc[current_date].dropna().index.intersection(valid_returns.index)
                target = pd.Series(0.0, index=returns.columns)
                if len(candidates):
                    count = max(1, math.ceil(len(candidates) * self.config.topk_ratio))
                    chosen = scores.loc[current_date, candidates].nlargest(count).index
                    weight = min(position_size / len(chosen), max_pos_each_stock)
                    target.loc[chosen] = weight
            target.loc[day_returns.isna()] = 0.0
            delta = target - previous
            buys = float(delta.clip(lower=0).sum())
            sells = float((-delta.clip(upper=0)).sum())
            cost = buys * self.config.open_cost + sells * self.config.close_cost
            gross = float((target * day_returns.fillna(0.0)).sum())
            benchmark = float(valid_returns.mean()) if len(valid_returns) else 0.0
            net_returns.append(gross - cost)
            gross_returns.append(gross)
            benchmark_returns.append(benchmark)
            turnovers.append((buys + sells) / 2.0)
            holdings.append(int((target > 0).sum()))
            previous = target

        net = pd.Series(net_returns, index=dates, dtype=float)
        gross = pd.Series(gross_returns, index=dates, dtype=float)
        benchmark = pd.Series(benchmark_returns, index=dates, dtype=float)
        excess = net - benchmark
        nav = (1.0 + net).cumprod()
        gross_nav = (1.0 + gross).cumprod()
        benchmark_nav = (1.0 + benchmark).cumprod()
        periods = max(len(net), 1)
        annualized_return = float(nav.iloc[-1] ** (self.config.annualization / periods) - 1.0)
        annualized_volatility = float(net.std(ddof=0) * math.sqrt(self.config.annualization))
        excess_std = float(excess.std(ddof=0))
        information_ratio = (
            float(excess.mean() / excess_std * math.sqrt(self.config.annualization))
            if excess_std > 0
            else 0.0
        )
        drawdown = nav / nav.cummax() - 1.0
        metrics = {
            "Information_Ratio_with_cost": information_ratio,
            "Annualized_Return_with_cost": annualized_return,
            "Annualized_Volatility_with_cost": annualized_volatility,
            "Max_Drawdown_with_cost": float(drawdown.min()),
            "Average_Daily_Turnover": float(np.mean(turnovers)),
            "Total_Return_with_cost": float(nav.iloc[-1] - 1.0),
            "Total_Return_without_cost": float(gross_nav.iloc[-1] - 1.0),
            "Benchmark_Total_Return": float(benchmark_nav.iloc[-1] - 1.0),
        }
        return {
            "metrics": metrics,
            "portfolio": {
                "dates": [item.strftime("%Y-%m-%d") for item in dates],
                "nav": [float(item) for item in nav],
                "benchmark_nav": [float(item) for item in benchmark_nav],
                "daily_return": net_returns,
            },
            "diagnostics": {
                "factor_name": factor_name,
                "expression": expression,
                "universe": universe,
                "signal_lag_bars": total_shift,
                "shuffle_cross_section": shuffle_cross_section,
                "rebalance_every_bars": update_freq,
                "mean_holdings": float(np.mean(holdings)),
                "benchmark": f"{universe}_equal_weight",
                "rows": int(len(selected)),
            },
        }
