"""Local commands. No LLM required except dry-run with --echo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from factor_agent.eval.inject_set import build_inject_eval, inject_rows
from factor_agent.eval.metrics import compare, summarize, summarize_by_group
from factor_agent.io import load_jsonl
from factor_agent.pipeline import run_case
from factor_agent.probe.metrics import compute_group_metrics
from factor_agent.providers.echo import echo_complete
from factor_agent.routing.pool_writer import write_pools
from factor_agent.routing.route_case import route_case
from factor_agent.score.verifier import score_prediction
from factor_agent.split import group_map, leak_ids
from factor_agent.train.grpo_builder import build_grpo
from factor_agent.train.reports import summarize_scores_file
from factor_agent.train.rollout_correction import correction_diagnostics
from factor_agent.train.sft_builder import build_sft


def _gold(path: str):
    cases = load_jsonl(path)
    if not cases:
        raise SystemExit(f"no cases in {path}")
    return cases


def cmd_score(args: argparse.Namespace) -> None:
    cases = _gold(args.gold)
    leaked = sorted(leak_ids(cases))
    results = [score_prediction(c, c.gold.model_dump_json()) for c in cases]
    summary = summarize(results)
    summary["leak_ids"] = leaked
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if leaked:
        raise SystemExit("train/holdout leak")


def cmd_holdout_report(args: argparse.Namespace) -> None:
    cases = [case for case in _gold(args.gold) if case.split == args.split]
    if not cases:
        raise SystemExit(f"no {args.split} cases in {args.gold}")
    results = [score_prediction(c, c.gold.model_dump_json()) for c in cases]
    report = summarize_by_group(results, group_map(cases))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def cmd_compare_metrics(args: argparse.Namespace) -> None:
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    print(
        json.dumps(
            compare(before.get("pooled", before), after.get("pooled", after)),
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_eval_window_report(args: argparse.Namespace) -> None:
    from factor_agent.backtest.client import BacktestClient, BacktestConfig
    from factor_agent.evolve.eval_window import evaluate_eval_window
    from factor_agent.schemas import FactorSpec

    pairs = []
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pairs.append(
            (
                row["case_id"],
                FactorSpec.model_validate(row["seed"]),
                FactorSpec.model_validate(row["evolved"]),
            )
        )
    client = BacktestClient(BacktestConfig(endpoint=args.endpoint), namespace="eval_window")
    report = evaluate_eval_window(pairs, client)
    print(report.model_dump_json(indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def cmd_inject(args: argparse.Namespace) -> None:
    cases = _gold(args.gold)
    n = build_inject_eval(cases, args.out)
    rows = inject_rows(cases)
    golds = [r["reward"] for r in rows if r["kind"] == "gold"]
    negs = [r["reward"] for r in rows if r["kind"] != "gold"]
    print(
        json.dumps(
            {
                "wrote": args.out,
                "n": n,
                "gold_mean": sum(golds) / len(golds),
                "inject_mean": sum(negs) / len(negs) if negs else 0.0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_build_sft(args: argparse.Namespace) -> None:
    n = build_sft(_gold(args.gold), args.out, stage=args.stage)
    print(json.dumps({"wrote": args.out, "n": n, "stage": args.stage}, ensure_ascii=False))


def cmd_build_grpo(args: argparse.Namespace) -> None:
    n = build_grpo(_gold(args.gold), args.out, stage=args.stage)
    print(json.dumps({"wrote": args.out, "n": n}, ensure_ascii=False))


def cmd_dry_run(args: argparse.Namespace) -> None:
    cases = _gold(args.gold)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rewards = []
    audits = []
    for case in cases:
        payload = run_case(case, echo_complete(case), extract_reward_min=args.gate)
        rewards.append(payload["reward"])
        for stage in ("extract", "evolve"):
            extras = (payload.get(stage) or {}).get("extras") or {}
            if extras.get("tool_audit"):
                audits.append(extras["tool_audit"])
        (out_dir / f"{case.case_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    from factor_agent.schemas import RewardResult

    print(
        json.dumps(
            summarize([RewardResult.model_validate(r) for r in rewards], tool_audits=audits),
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_route_probe(args: argparse.Namespace) -> None:
    groups: dict[str, list[dict]] = {}
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            groups.setdefault(row["case_id"], []).append(row)
    classifications = []
    for case_id, records in groups.items():
        metrics = compute_group_metrics(records)
        classifications.append({"case_id": case_id, **route_case(metrics), "metrics": metrics})
    counts = write_pools(classifications, args.out)
    print(json.dumps({"out": args.out, "counts": counts}, ensure_ascii=False, indent=2))


def cmd_summarize_run(args: argparse.Namespace) -> None:
    summary = summarize_scores_file(args.run_dir, args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_diagnose_correction(args: argparse.Namespace) -> None:
    samples = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = correction_diagnostics(samples, threshold=args.threshold, enable_delta=args.enable_delta)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_prepare_market_data(args: argparse.Namespace) -> None:
    from factor_agent.backtest.data import download_public_data, export_qlib_dataset

    manifest = download_public_data(
        args.raw_dir,
        start=args.start,
        end=args.end,
        universes=args.universes,
    )
    manifest = export_qlib_dataset(args.raw_dir, args.qlib_dir)
    print(manifest.model_dump_json(indent=2))


def cmd_serve_backtest(args: argparse.Namespace) -> None:
    import os

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit('install API dependencies with: pip install -e ".[backtest]"') from exc
    os.environ["FACTOR_MARKET_MANIFEST"] = str(Path(args.manifest).resolve())
    uvicorn.run(
        "factor_agent.backtest.server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


def cmd_backtest_smoke(args: argparse.Namespace) -> None:
    from factor_agent.backtest.engine import EngineConfig, QlibBacktestEngine

    engine = QlibBacktestEngine(EngineConfig(manifest_path=args.manifest))
    result = engine.backtest(
        exprs={args.name: args.expr},
        backtest_start_time=args.start,
        backtest_end_time=args.end,
        update_freq=args.update_freq,
        stock_pool=args.universe,
        shift_bars=args.shift_bars,
        shuffle_cross_section=args.shuffle,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="factor-agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score gold specs against themselves")
    s.add_argument("--gold", default="data/gold/gold.jsonl")
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("holdout-report", help="held-out metrics grouped by institution/series")
    s.add_argument("--gold", default="data/gold/gold.jsonl")
    s.add_argument("--split", choices=["train", "holdout"], default="holdout")
    s.add_argument("--out", default="")
    s.set_defaults(func=cmd_holdout_report)

    s = sub.add_parser("compare-metrics", help="before/after headline rates in percentage points")
    s.add_argument("--before", required=True)
    s.add_argument("--after", required=True)
    s.set_defaults(func=cmd_compare_metrics)

    s = sub.add_parser("eval-window-report", help="seed vs evolved factors on the frozen eval window")
    s.add_argument("--input", required=True, help="jsonl of {case_id, seed, evolved}")
    s.add_argument("--endpoint", default="http://localhost:8001/backtest")
    s.add_argument("--out", default="")
    s.set_defaults(func=cmd_eval_window_report)

    s = sub.add_parser("inject", help="write injected-error rows for verifier tuning")
    s.add_argument("--gold", default="data/gold/gold.jsonl")
    s.add_argument("--out", default="data/eval_inject/inject.jsonl")
    s.set_defaults(func=cmd_inject)

    s = sub.add_parser("build-sft", help="gold -> SFT jsonl for one LoRA stage")
    s.add_argument("--gold", default="data/gold/gold.jsonl")
    s.add_argument("--out", default="data/sft/extract_train.jsonl")
    s.add_argument("--stage", choices=["extract", "evolve"], default="extract")
    s.set_defaults(func=cmd_build_sft)

    s = sub.add_parser("build-grpo", help="gold -> GRPO prompt rows")
    s.add_argument("--gold", default="data/gold/gold.jsonl")
    s.add_argument("--out", default="data/rl/extract_train.jsonl")
    s.add_argument("--stage", choices=["extract", "evolve"], default="extract")
    s.set_defaults(func=cmd_build_grpo)

    s = sub.add_parser("dry-run", help="run extract pipeline with gold echo")
    s.add_argument("--gold", default="data/gold/gold.jsonl")
    s.add_argument("--out", default="data/trajectories")
    s.add_argument("--gate", type=float, default=0.7)
    s.set_defaults(func=cmd_dry_run)

    s = sub.add_parser("route-probe", help="route grouped scored rollouts into training pools")
    s.add_argument("--input", required=True)
    s.add_argument("--out", default="data/pools")
    s.set_defaults(func=cmd_route_probe)

    s = sub.add_parser("summarize-run", help="summarize rollout scores.jsonl")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_summarize_run)

    s = sub.add_parser("diagnose-correction", help="compare rollout and training log-prob paths")
    s.add_argument("--input", required=True)
    s.add_argument("--threshold", type=float, default=2.0)
    s.add_argument("--enable-delta", type=float, default=0.02)
    s.set_defaults(func=cmd_diagnose_correction)

    s = sub.add_parser("prepare-market-data", help="download BaoStock data and export a Qlib provider")
    s.add_argument("--raw-dir", default="data/market")
    s.add_argument("--qlib-dir", default="data/qlib/cn_data")
    s.add_argument("--start", default="2017-01-01")
    s.add_argument("--end", default="2024-12-31")
    s.add_argument("--universes", nargs="+", default=["CSI300", "CSI500"])
    s.set_defaults(func=cmd_prepare_market_data)

    s = sub.add_parser("serve-backtest", help="serve the local Qlib backtest API")
    s.add_argument("--manifest", default="data/market/manifest.json")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8001)
    s.set_defaults(func=cmd_serve_backtest)

    s = sub.add_parser("backtest-smoke", help="run one factor directly against local Qlib data")
    s.add_argument("--manifest", default="data/market/manifest.json")
    s.add_argument("--name", default="reversal_20d")
    s.add_argument("--expr", default="-TS_PCTCHANGE($close, 20)")
    s.add_argument("--universe", choices=["CSI300", "CSI500"], default="CSI500")
    s.add_argument("--start", default="2024-01-01")
    s.add_argument("--end", default="2024-12-31")
    s.add_argument("--update-freq", type=int, default=5)
    s.add_argument("--shift-bars", type=int, default=0)
    s.add_argument("--shuffle", action="store_true")
    s.set_defaults(func=cmd_backtest_smoke)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
