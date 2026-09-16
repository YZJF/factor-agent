import json

from factor_agent.backtest.client import BacktestClient
from factor_agent.evolve.run import run_evolve
from factor_agent.schemas import FactorSpec


def _spec(expr: str = "-TS_PCTCHANGE($close, 20)", universe: str = "CSI500") -> FactorSpec:
    return FactorSpec(
        name="reversal",
        universe=universe,
        frequency="daily",
        neutralization="none",
        rebalance="weekly",
        expr=expr,
        windows=[20],
    )


def test_backtest_client_maps_factor_and_caches():
    calls = []

    def transport(url, payload, timeout):
        calls.append(payload)
        return {"data": {"metrics": {"Information_Ratio_with_cost": 0.4}}}

    client = BacktestClient(transport=transport)
    first = client.evaluate(_spec(), case_id="case", period="search", start="2020-01-01", end="2020-12-31")
    second = client.evaluate(_spec(), case_id="case", period="search", start="2020-01-01", end="2020-12-31")
    assert first.ok and second.cached
    assert calls[0]["stock_pool"] == "CSI500"
    assert len(calls) == 1


def test_evolve_runs_two_visible_search_and_hidden_score_rounds():
    seen_periods = []

    def backtest(spec, *, start, end):
        period = "search" if start.startswith("2018") else "score"
        seen_periods.append(period)
        value = 0.2 if spec.expr.endswith("20)") else 0.6
        return {"metrics": {"Information_Ratio_with_cost": value}}

    candidate = _spec("-TS_PCTCHANGE($close, 10)")
    calls = [
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "v1",
                    "function": {
                        "name": "validate_expr",
                        "arguments": json.dumps({"spec": candidate.model_dump()}),
                    },
                }
            ],
        },
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "e1",
                    "function": {
                        "name": "evaluate_factor",
                        "arguments": json.dumps({"spec": candidate.model_dump()}),
                    },
                }
            ],
        },
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "e2",
                    "function": {
                        "name": "evaluate_factor",
                        "arguments": json.dumps({"spec": candidate.model_dump()}),
                    },
                }
            ],
        },
        {"content": "done", "tool_calls": []},
    ]

    def complete(messages, tools=None):
        assert tools
        return calls.pop(0)

    trajectory = run_evolve(_spec(), complete, backtest, max_rounds=2)
    assert seen_periods.count("search") == 2
    assert seen_periods.count("score") == 3  # hidden baseline + two hidden candidates
    assert len(trajectory.extras["hidden_score_results"]) == 2
    assert all("score" not in step.content for step in trajectory.steps if step.role == "tool")
    assert trajectory.reward > 0.0


def test_evolve_drift_is_hard_capped():
    drifted = _spec("-TS_PCTCHANGE($close, 10)", universe="CSI300")

    def backtest(spec, *, start, end):
        return {"metrics": {"Information_Ratio_with_cost": 0.5}}

    replies = iter(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "e1",
                        "function": {
                            "name": "evaluate_factor",
                            "arguments": json.dumps({"spec": drifted.model_dump()}),
                        },
                    }
                ],
            },
            {"content": "done", "tool_calls": []},
        ]
    )
    trajectory = run_evolve(_spec(), lambda messages, tools=None: next(replies), backtest, max_rounds=1)
    assert trajectory.extras["evolve_reward"]["reward"] <= 0.30
    assert "spec_drift_cap" in trajectory.extras["evolve_reward"]["active_caps"]
