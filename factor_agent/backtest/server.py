"""FastAPI service compatible with AlphaAgentEvo's backtest contract."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from factor_agent.backtest.engine import EngineConfig, QlibBacktestEngine
from factor_agent.backtest.expression import ExpressionError


class BacktestRequest(BaseModel):
    exprs: dict[str, str]
    backtest_start_time: str
    backtest_end_time: str
    start_cash: float = 10_000_000.0
    update_freq: int = Field(default=5, ge=1)
    label_forward_days: int = Field(default=5, ge=1)
    stock_pool: str = "CSI500"
    stop_loss_rate: float | None = 0.5
    stop_profit_rate: float | None = 0.5
    position_size: float = Field(default=1.0, gt=0.0, le=1.0)
    max_pos_each_stock: float = Field(default=0.2, gt=0.0, le=1.0)
    industry_neutralization: str | None = None
    use_cache: bool = True
    layer_start: int = 0
    layer_end: int = 1
    pred_score_industry_neutralization: bool = False
    shift_bars: int = Field(default=0, ge=0, le=20)
    shuffle_cross_section: bool = False


class ExprTestRequest(BaseModel):
    name: str
    expr: str


def create_app(engine: QlibBacktestEngine | None = None):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError('API support requires: pip install -e ".[backtest]"') from exc

    app = FastAPI(
        title="Factor Research Qlib Backtest API",
        description="Look-ahead-safe CSI300/CSI500 factor backtesting",
        version="1.0.0",
    )
    backend = engine

    def get_engine() -> QlibBacktestEngine:
        nonlocal backend
        if backend is None:
            backend = QlibBacktestEngine(
                EngineConfig(
                    manifest_path=os.getenv(
                        "FACTOR_MARKET_MANIFEST",
                        "data/market/manifest.json",
                    )
                )
            )
        return backend

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            current = get_engine()
            return {
                "status": "healthy",
                "engine": "qlib",
                "data_source": current.manifest.source if current.manifest else "injected-fixture",
                "data_range": (
                    [current.manifest.start, current.manifest.end] if current.manifest else None
                ),
            }
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={"source": "environment", "error": str(exc)},
            ) from exc

    @app.post("/test_expr")
    def test_expression(request: ExprTestRequest) -> dict[str, Any]:
        result = QlibBacktestEngine.validate_expression(request.expr)
        return {
            "success": bool(result["ok"]),
            "message": "expression is supported" if result["ok"] else "expression is invalid",
            "exe_feedback": None if result["ok"] else result["error"],
            "analysis": result,
        }

    @app.post("/backtest")
    def backtest(request: BacktestRequest) -> dict[str, Any]:
        try:
            result = get_engine().backtest(**request.model_dump())
            return {
                "success": True,
                "message": "Qlib backtest completed",
                "data": result,
            }
        except (ExpressionError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail={"source": "model", "error": str(exc)},
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={"source": "environment", "error": str(exc)},
            ) from exc

    return app


app = create_app()
