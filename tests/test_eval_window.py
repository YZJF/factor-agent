import pytest

from factor_agent.evolve.eval_window import evaluate_eval_window
from factor_agent.evolve.periods import DEFAULT_WINDOWS
from factor_agent.schemas import FactorSpec

EVAL_METRICS = {
    "-TS_PCTCHANGE($close, 20)": 0.42,
    "-TS_PCTCHANGE($close, 40)": 0.55,
    "RANK($amount / TS_MEAN($amount, 60))": 0.10,
    "RANK($amount / TS_MEAN($amount, 30))": 0.25,
}


def _spec(expr: str, name: str = "f") -> FactorSpec:
    return FactorSpec(
        name=name,
        universe="CSI500",
        frequency="daily",
        neutralization="none",
        rebalance="weekly",
        expr=expr,
    )


def _backend(spec: FactorSpec, start: str, end: str) -> dict:
    assert (start, end) == DEFAULT_WINDOWS.eval()
    return {
        "metrics": {
            "Information_Ratio_with_cost": EVAL_METRICS[spec.expr],
            "turnover": 0.3,
            "max_drawdown": -0.12,
            "sharpe_without_cost": 9.9,
        }
    }


def test_median_after_cost_ir_delta_on_the_frozen_window():
    pairs = [
        ("c1", _spec("-TS_PCTCHANGE($close, 20)"), _spec("-TS_PCTCHANGE($close, 40)")),
        (
            "c2",
            _spec("RANK($amount / TS_MEAN($amount, 60))"),
            _spec("RANK($amount / TS_MEAN($amount, 30))"),
        ),
    ]
    report = evaluate_eval_window(pairs, _backend)
    assert report.window == DEFAULT_WINDOWS.eval()
    assert report.n_scored == 2
    assert report.median_delta == pytest.approx(0.14)
    assert report.improved_share == 1.0
    assert "sharpe_without_cost" not in report.rows[0].evolved_metrics
    assert report.rows[0].evolved_metrics["turnover"] == 0.3


def test_failed_backtest_is_excluded_from_the_median():
    def flaky(spec: FactorSpec, start: str, end: str) -> dict:
        if spec.expr.startswith("RANK"):
            raise RuntimeError("qlib provider missing")
        return _backend(spec, start, end)

    pairs = [
        ("c1", _spec("-TS_PCTCHANGE($close, 20)"), _spec("-TS_PCTCHANGE($close, 40)")),
        (
            "c2",
            _spec("RANK($amount / TS_MEAN($amount, 60))"),
            _spec("RANK($amount / TS_MEAN($amount, 30))"),
        ),
    ]
    report = evaluate_eval_window(pairs, flaky)
    assert report.n == 2
    assert report.n_scored == 1
    assert report.median_delta == pytest.approx(0.13)
    assert report.rows[1].ok is False
    assert "qlib provider missing" in report.rows[1].error
