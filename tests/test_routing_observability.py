from pathlib import Path

from factor_agent.probe.metrics import compute_group_metrics
from factor_agent.rollout.store import RolloutStore
from factor_agent.routing.pool_writer import write_pools
from factor_agent.routing.route_case import route_case
from factor_agent.train.reports import summarize_run
from factor_agent.trajectory import Step, Trajectory


def test_rollout_store_hashes_and_persists(tmp_path: Path):
    trajectory = Trajectory(
        case_id="case-1",
        stage="extract",
        steps=[Step(role="user", content="report"), Step(role="assistant", content="{}")],
        reward=0.8,
        final_text="{}",
    )
    record = RolloutStore(tmp_path, run_id="run-1").write(
        trajectory,
        score={"reward": 0.8, "active_caps": []},
        prompt=[{"role": "user", "content": "report"}],
        tool_schemas=[],
        rollout_id="rollout-1",
    )
    assert len(record["prompt_hash"]) == 64
    assert Path(record["trajectory_path"]).exists()
    assert (tmp_path / "run-1" / "scores.jsonl").exists()


def test_probe_metrics_and_routes():
    learnable = compute_group_metrics([{"reward": 0.2}, {"reward": 0.9}])
    assert route_case(learnable)["route"] == "rl_main"
    format_failures = compute_group_metrics(
        [
            {"reward": 0.0, "active_caps": ["invalid_json_cap"]},
            {"reward": 0.1, "active_caps": ["invalid_json_cap"]},
        ]
    )
    assert route_case(format_failures)["route"] == "sft_format"
    environment = compute_group_metrics(
        [{"reward": 0.0, "tool_error_environment": 1}, {"reward": 0.0, "tool_error_environment": 1}]
    )
    assert route_case(environment)["route"] == "backtest_gap"


def test_pool_writer_and_report(tmp_path: Path):
    classifications = [
        {"case_id": "a", "route": "rl_main", "metrics": {}},
        {"case_id": "b", "route": "sft_format", "metrics": {}},
    ]
    counts = write_pools(classifications, tmp_path / "pools")
    assert counts == {"rl_main": 1, "sft_format": 1}
    summary = summarize_run(
        [
            {"case_id": "a", "reward": 0.9, "active_caps": []},
            {"case_id": "a", "reward": 0.2, "active_caps": ["illegal_expr_cap"]},
        ],
        tmp_path / "report",
        classifications=classifications,
    )
    assert summary["rollout_count"] == 2
    assert (tmp_path / "report" / "report.md").exists()
