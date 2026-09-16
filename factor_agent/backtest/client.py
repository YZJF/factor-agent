from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel

from factor_agent.backtest.ledger import BacktestLedger, LedgerEntry, stable_hash
from factor_agent.schemas import BacktestResult, FactorSpec

Transport = Callable[[str, dict[str, Any], float], dict[str, Any]]


class BacktestTransportError(RuntimeError):
    def __init__(self, message: str, *, source: str = "environment"):
        super().__init__(message)
        self.source = source


class BacktestConfig(BaseModel):
    endpoint: str = "http://localhost:8001/backtest"
    metric: str = "Information_Ratio_with_cost"
    timeout_seconds: float = 600.0
    start_cash: float = 10_000_000.0
    label_forward_days: int = 5
    stop_loss_rate: float = 0.5
    stop_profit_rate: float = 0.5
    max_pos_each_stock: float = 0.2
    use_cache: bool = True


def _http_transport(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        source = "model" if 400 <= exc.code < 500 else "environment"
        try:
            detail = json.loads(body).get("detail", {})
            message = detail.get("error", body) if isinstance(detail, dict) else str(detail)
            source = detail.get("source", source) if isinstance(detail, dict) else source
        except json.JSONDecodeError:
            message = body
        raise BacktestTransportError(f"backtest HTTP {exc.code}: {message}", source=source) from exc
    except URLError as exc:
        raise BacktestTransportError(f"backtest unavailable: {exc.reason}") from exc


class BacktestClient:
    def __init__(
        self,
        config: BacktestConfig | None = None,
        *,
        transport: Transport | None = None,
        ledger: BacktestLedger | None = None,
        namespace: str = "default",
    ):
        self.config = config or BacktestConfig()
        self.transport = transport or _http_transport
        self.ledger = ledger or BacktestLedger()
        self.namespace = namespace

    def _payload(
        self,
        spec: FactorSpec,
        *,
        start: str,
        end: str,
        shift_bars: int = 0,
        shuffle: bool = False,
    ) -> dict[str, Any]:
        frequency = {"daily": 1, "weekly": 5, "monthly": 20}
        payload: dict[str, Any] = {
            "exprs": {spec.name: spec.expr},
            "backtest_start_time": start,
            "backtest_end_time": end,
            "start_cash": self.config.start_cash,
            "update_freq": frequency[spec.rebalance],
            "label_forward_days": self.config.label_forward_days,
            "stock_pool": spec.universe,
            "stop_loss_rate": self.config.stop_loss_rate,
            "stop_profit_rate": self.config.stop_profit_rate,
            "position_size": 1.0,
            "max_pos_each_stock": self.config.max_pos_each_stock,
            "use_cache": self.config.use_cache,
            "layer_start": 0,
            "layer_end": 1,
            "industry_neutralization": spec.neutralization,
            "pred_score_industry_neutralization": spec.neutralization != "none",
        }
        if shift_bars:
            payload["shift_bars"] = shift_bars
        if shuffle:
            payload["shuffle_cross_section"] = True
        return payload

    def evaluate(
        self,
        spec: FactorSpec,
        *,
        case_id: str,
        period: str,
        start: str,
        end: str,
        shift_bars: int = 0,
        shuffle: bool = False,
    ) -> BacktestResult:
        payload = self._payload(spec, start=start, end=end, shift_bars=shift_bars, shuffle=shuffle)
        expression_hash = stable_hash({"name": spec.name, "expr": spec.expr})
        config_hash = stable_hash({"config": self.config.model_dump(), "payload": payload})
        cached = self.ledger.cached(
            namespace=self.namespace,
            expression_hash=expression_hash,
            period=period,
            config_hash=config_hash,
        )
        if cached:
            return BacktestResult.model_validate({**cached.result, "cached": True, "run_id": cached.run_id})
        started = perf_counter()
        try:
            response = self.transport(self.config.endpoint, payload, self.config.timeout_seconds)
            metrics = (response.get("data") or {}).get("metrics") or response.get("metrics") or {}
            value = metrics.get(self.config.metric)
            if value is None:
                raise RuntimeError(f"metric missing from response: {self.config.metric}")
            result = BacktestResult(
                ok=True,
                metric=float(value),
                metric_name=self.config.metric,
                period=period,
                metrics={key: float(item) for key, item in metrics.items() if isinstance(item, (int, float))},
            )
        except Exception as exc:
            result = BacktestResult(
                ok=False,
                metric_name=self.config.metric,
                period=period,
                error=str(exc),
                error_source=getattr(exc, "source", "environment"),
            )
        payload["_elapsed_seconds"] = round(perf_counter() - started, 6)
        entry = LedgerEntry(
            namespace=self.namespace,
            case_id=case_id,
            expression_hash=expression_hash,
            period=period,
            config_hash=config_hash,
            request=payload,
            result=result.model_dump(),
        )
        self.ledger.append(entry)
        result.run_id = entry.run_id
        return result
