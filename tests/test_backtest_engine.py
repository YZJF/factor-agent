import hashlib
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from factor_agent.backtest.data import (
    MarketDataManifest,
    download_public_data,
    export_qlib_dataset,
    validate_manifest,
)
from factor_agent.backtest.engine import EngineConfig, QlibBacktestEngine, QlibFrameLoader
from factor_agent.backtest.expression import ExpressionError, evaluate_expression, parse_expression


def _market_frame():
    dates = pd.date_range("2024-01-02", periods=12, freq="B")
    instruments = ["sh600001", "sh600002", "sz000001", "sz000002"]
    rows = []
    for day, current_date in enumerate(dates):
        for rank, instrument in enumerate(instruments):
            close = 10.0 + day + (3 - rank) * 3.0
            rows.append(
                {
                    "datetime": current_date,
                    "instrument": instrument,
                    "close": close,
                    "return": [0.02, 0.01, -0.01, -0.02][rank],
                    "tradestatus": 1.0,
                    "industry": float(rank % 2),
                }
            )
    return pd.DataFrame(rows).set_index(["datetime", "instrument"]).sort_index()


def test_safe_expression_parser_and_vectorized_evaluation():
    frame = _market_frame()
    node = parse_expression("RANK($close / TS_MEAN($close, 2))")
    result = evaluate_expression(node, frame)
    assert result.notna().any()
    assert result.groupby(level="datetime").max().dropna().eq(1.0).all()
    with pytest.raises(ExpressionError, match="unsupported function"):
        parse_expression("COUNT($close, 2)")
    with pytest.raises(ExpressionError):
        parse_expression("__import__('os')")


def test_engine_computes_nav_cost_metrics_and_real_contrasts():
    frame = _market_frame()

    def loader(universe, fields, start, end):
        assert universe == "CSI500"
        assert {"close", "return", "tradestatus"} <= fields
        return frame

    engine = QlibBacktestEngine(
        EngineConfig(topk_ratio=0.5, open_cost=0.001, close_cost=0.002),
        frame_loader=loader,
    )
    request = {
        "exprs": {"rank_close": "RANK($close)"},
        "backtest_start_time": "2024-01-02",
        "backtest_end_time": "2024-01-17",
        "update_freq": 2,
        "stock_pool": "CSI500",
    }
    baseline = engine.backtest(**request)
    repeated = engine.backtest(**request)
    lagged = engine.backtest(**request, shift_bars=2)
    shuffled = engine.backtest(**request, shuffle_cross_section=True)

    assert baseline == repeated
    assert baseline["portfolio"]["nav"][-1] > 1.0
    assert baseline["metrics"]["Total_Return_with_cost"] < baseline["metrics"]["Total_Return_without_cost"]
    assert baseline["diagnostics"]["signal_lag_bars"] == 1
    assert lagged["diagnostics"]["signal_lag_bars"] == 3
    assert lagged["portfolio"]["nav"] != baseline["portfolio"]["nav"]
    assert shuffled["portfolio"]["nav"] != baseline["portfolio"]["nav"]


def test_future_rows_cannot_change_past_backtest():
    frame = _market_frame()

    def bounded_loader(universe, fields, start, end):
        return frame.loc[frame.index.get_level_values("datetime") <= pd.Timestamp(end)]

    engine = QlibBacktestEngine(
        EngineConfig(topk_ratio=0.5),
        frame_loader=bounded_loader,
    )
    common = {
        "exprs": {"mean_close": "RANK(TS_MEAN($close, 2))"},
        "backtest_start_time": "2024-01-04",
        "update_freq": 2,
        "stock_pool": "CSI500",
    }
    early = engine.backtest(**common, backtest_end_time="2024-01-11")
    extended = engine.backtest(**common, backtest_end_time="2024-01-17")
    assert early["portfolio"]["nav"] == extended["portfolio"]["nav"][: len(early["portfolio"]["nav"])]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_public_source_writes_hashed_manifest(tmp_path: Path):
    class Source:
        name = "fixture-public"

        def open(self):
            self.opened = True

        def close(self):
            self.opened = False

        def memberships(self, start, end, universes):
            frame = pd.DataFrame(
                [
                    {"universe": universe, "date": start, "instrument": instrument}
                    for universe in universes
                    for instrument in ["sh600001", "sz000001"]
                ]
            )
            return frame, {universe: 1 for universe in universes}

        def industries(self, start, end):
            return pd.DataFrame(
                [
                    {"date": start, "instrument": "sh600001", "industry": "bank"},
                    {"date": start, "instrument": "sz000001", "industry": "tech"},
                ]
            )

        def prices(self, instruments, start, end, *, cache_dir=None):
            result = _market_frame().reset_index().rename(columns={"datetime": "date"})
            result = result[result["instrument"].isin(instruments)].copy()
            result["date"] = result["date"].dt.strftime("%Y-%m-%d")
            for field in ["open", "high", "low", "volume", "amount"]:
                result[field] = result["close"] if field != "volume" else 1000.0
            result["isST"] = 0
            return result

    source = Source()
    manifest = download_public_data(
        tmp_path,
        start="2024-01-02",
        end="2024-01-17",
        source=source,
    )
    assert manifest.source == "fixture-public"
    assert not source.opened
    assert set(manifest.content_hashes) == {"market", "memberships", "industries"}
    with pytest.raises(FileNotFoundError, match="Qlib dataset"):
        validate_manifest(
            tmp_path / "manifest.json",
            required_start="2024-01-02",
            required_end="2024-01-17",
        )


def test_exported_binary_is_readable_by_qlib(tmp_path: Path):
    pytest.importorskip("qlib")
    raw = tmp_path / "raw"
    raw.mkdir()
    frame = _market_frame().reset_index().rename(columns={"datetime": "date"})
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    for field in ["open", "high", "low", "volume", "amount"]:
        frame[field] = frame["close"] if field != "volume" else 1000.0
    frame["isST"] = 0
    market = raw / "market.csv"
    frame.to_csv(market, index=False)
    memberships = raw / "memberships.csv"
    pd.DataFrame(
        [
            {"universe": "CSI500", "date": frame["date"].min(), "instrument": instrument}
            for instrument in frame["instrument"].unique()
        ]
    ).to_csv(memberships, index=False)
    industries = raw / "industries.csv"
    pd.DataFrame(
        [
            {"date": frame["date"].min(), "instrument": instrument, "industry": "test"}
            for instrument in frame["instrument"].unique()
        ]
    ).to_csv(industries, index=False)
    manifest = MarketDataManifest(
        start=frame["date"].min(),
        end=frame["date"].max(),
        warmup_start=frame["date"].min(),
        universes=["CSI500"],
        instruments=4,
        rows=len(frame),
        membership_snapshots={"CSI500": 1},
        fields=["close", "return", "tradestatus"],
        files={
            "market": str(market),
            "memberships": str(memberships),
            "industries": str(industries),
        },
        content_hashes={
            "market": _digest(market),
            "memberships": _digest(memberships),
            "industries": _digest(industries),
        },
    )
    (raw / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    exported = export_qlib_dataset(raw, tmp_path / "qlib")
    loaded = QlibFrameLoader(exported.qlib_uri)(
        "CSI500",
        {"close", "return", "tradestatus"},
        frame["date"].min(),
        frame["date"].max(),
    )
    assert not loaded.empty
    assert {"close", "return", "tradestatus"} <= set(loaded.columns)


def test_fastapi_contract_uses_computed_engine():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from factor_agent.backtest.server import create_app

    frame = _market_frame()
    engine = QlibBacktestEngine(frame_loader=lambda universe, fields, start, end: frame)
    client = TestClient(create_app(engine))
    response = client.post(
        "/backtest",
        json={
            "exprs": {"rank_close": "RANK($close)"},
            "backtest_start_time": "2024-01-02",
            "backtest_end_time": "2024-01-17",
            "update_freq": 2,
            "stock_pool": "CSI500",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"]
    assert len(payload["data"]["portfolio"]["nav"]) == 12
    invalid = client.post(
        "/backtest",
        json={
            "exprs": {"bad": "COUNT($close, 2)"},
            "backtest_start_time": "2024-01-02",
            "backtest_end_time": "2024-01-17",
            "stock_pool": "CSI500",
        },
    )
    assert invalid.status_code == 422
    assert invalid.json()["detail"]["source"] == "model"
