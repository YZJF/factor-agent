# Pipeline

Public install and CLI notes live in the root [README.md](../README.md) (English first, Chinese toggle).

One example = report text + a human-annotated 7-field spec.

`report → extract → rule score → gate → multi-turn mutation → SFT / GRPO`

```text
factor_agent/
  documents/          PDF/Markdown parse, evidence chunks, search/read/validate tools
  batch/              case/report/verifier/gold four-file manifest
  schemas.py            7-field FactorSpec / FactorCase
  split.py              case IDs and institution/series groups must not leak across train/holdout
  spec/expr.py          14 operators + 9 fields; time-series as-of yesterday
  spec/consistency.py   neutralization field vs expression operator
  spec/requirements.py  required evidence; params must be declared, cited, and locatable
  score/verifier.py     four subscores + three caps
  score/inject.py       five error families for verifier calibration only
  protocol.py           six illegal tool-call classes and per-trajectory audit
  extract/run.py        Stage A single-shot / long-report tool loop
  gate.py               admit only if total ≥ 0.7 and no cap
  backtest/             BaoStock → Qlib data, live portfolio backtest, API, append-only ledger
  evolve/run.py         Stage B, up to 4 live backtest turns
  evolve/drift.py       freeze universe / frequency / neutralization
  evolve/periods.py     mutate on search, score on hidden window, never touch eval
  evolve/contrast.py    one-bar lag / cross-section shuffle (backtest hooks)
  evolve/eval_window.py frozen eval-window after-cost IR median delta, seed vs mutated
  eval/metrics.py       three headline metrics, grouped holdout report, percentage-point compare
  probe/ + routing/     K-sample metrics and routing pools
  rollout/              trajectories, scores, consistency hashes
  train/                SFT / GRPO + verl AgentLoop + rollout correction
  pipeline.py           wires the pieces
```

Legacy files `verifier.py` / `inject.py` / `expr_check.py` are shims.

---

## 1. Spec and split

Fields: name / universe / frequency / neutralization / rebalance / expr / lookback windows.
One `case_id` per example; `split` = train | holdout.
`split.assert_no_leak` hard-fails in `build-sft` / `build-grpo`; `score` prints `leak_ids`.

Group key `split.group_key` falls back: explicit `group_id` → `institution::series` → `doc_id`.
Reports from the same desk and factor series reuse wording and parameters, so a case-id-only split leaks.
Holdout is grouped by institution/series; `holdout-report` prints per-group metrics and a cross-group median so one high-volume desk cannot dominate the mean.

Every parameter that appears in the expression must be declared in `windows` and cited with a chunk plus page/character span.
Gold that fails `spec/requirements.py` cannot enter the four-file batch or SFT replay.

Model-extracted specs are never treated as gold.

---

## 2. Rule scoring

No LLM judge.

| Subscore | Weight |
|---|---|
| JSON parses as a spec | 0.25 |
| Expression is executable on the whitelist | 0.25 |
| Universe / frequency / neutralization / rebalance / expr match gold | 0.40 |
| Expr and window numbers appear in cited spans | 0.10 |

Grounding uses only chunks the model actually cited. No citation → 0; there is no fallback to "it appeared somewhere in the document".
Numbers are normalized so Chinese numerals such as "过去二十个交易日" match `windows=[20]`.
(The old `numbers_in_text` used `\b\d+\b`. CJK characters are word characters, so Chinese reports scored 0 grounding forever.)
"No citation" and "hallucinated formula" are different: the hallucination cap still uses a looser full-document check, and only seals at 0.35 when the report has no support.

Caps: unparseable JSON → 0; off-whitelist operator → ≤ 0.15; expr mismatches gold **and** universe is wrong or numbers are missing from the source → ≤ 0.35.

Five injected errors (`score/inject.py`): drop neutralization, flip frequency, multiply windows by 3, unrelated formula, illegal call. Eval-set only; never SFT, never pairwise.

Declaration vs implementation mismatches (neutralization says industry but the expr has no `INDUSTRY_NEUTRALIZE`) go into `diagnostics.declaration_issues` and are not yet reweighted.

---

## 3. Two stages and the gate

**A** Short reports may emit JSON directly; long reports use `search_report` / `read_report_pages` and `validate_factor_spec` for a structural preflight.
**Gate** (`gate.py`): total ≥ 0.7, no cap, executable expr. Failures stay in A as negatives.
**B** Up to 4 live eval turns: edit expr → `validate_expr` → `evaluate_factor` → edit again. Only expr and windows may change; editing frozen fields triggers `spec_drift_cap`.

Stage A scores do not consume Stage B backtest metrics.

Every tool call in both stages goes through `protocol.classify_call`. Illegal means exactly six classes: unknown tool, unparseable args, illegal FactorSpec, over backtest-turn budget, frozen-field edit, multiple calls in one step.
`validate_expr` saying "operator not on the whitelist" is not illegal — that is expression quality and already hits a subscore.
A downed backtest service is not illegal either; it is counted in `environment_failures`. Per-trajectory results land in `Trajectory.extras["tool_audit"]`, then the rollout store writes `scores.jsonl`.

---

## 4. Reproducibility checks

| Check | Module | Status |
|---|---|---|
| Time-series ops as-of yesterday | `spec/expr.py` + `backtest/engine.py` | Signal is lagged one trading day before entering that day's portfolio return |
| Declaration vs implementation | `spec/consistency.py` | Logged, not weighted |
| Search / score / eval windows | `evolve/periods.py` defaults 2018–2021 / 2022–2023 / 2024 | search visible; score hidden; eval unseen in training |
| One-bar lag, cross-section shuffle | `evolve/run.py` + `backtest/engine.py` | Backend actually shifts/shuffles; fixed seed for replay |
| Per-turn backtest delta | `evolve/run.py` + `evolve/reward.py` | Reward is best-so-far delta on the hidden score window |
| Frozen eval window | `evolve/eval_window.py` | Only place that reads eval; offline; after-cost IR median delta, seed vs mutated |

---

## 5. Post-training

`sample → rule score → two LoRAs, each with SFT / GRPO`.
Both adapters grow from the same frozen Qwen3-4B. **The evolve LoRA is not stacked on the extract LoRA.**
Extract: gold tool-call SFT replay → rule-reward GRPO.
Evolve: protocol warmup SFT (`validate_expr` then emit the seed as-is, no fake IR) → AgentLoop backtest GRPO.
Swap adapters at the gate during inference.

| In this repo | Not in this repo |
|---|---|
| `build-sft --stage` / `build-grpo --stage` jsonl | Full `verl/` sources |
| LoRA hydra knobs in `scripts/run_sft.sh` / `run_grpo.sh` | Weights and checkpoints |

`python -c "import verl"` must succeed before launch. TIS is Stage B only, after response-mask checks and a systematic vLLM vs FSDP log-prob gap.

Each round watch three things: diversity under the same prompt, gold mean vs injected-negative mean, empty `leak_ids`.

Checkpoint compares use three headline metrics from `holdout-report` on institution/series-grouped holdout, then `compare-metrics` for percentage-point deltas: `task_success_rate` (parseable + executable + all 5 fields + no cap), `evidence_grounding_rate` (every parameter sits in a valid citation), `illegal_tool_call_rate`.

---

## Commands

```bash
pytest -q
python -m factor_agent score
python -m factor_agent holdout-report --out runs/base/holdout.json
python -m factor_agent compare-metrics --before runs/base/holdout.json --after runs/grpo/holdout.json
python -m factor_agent inject
python -m factor_agent build-sft --stage extract --out data/sft/extract_train.jsonl
python -m factor_agent build-sft --stage evolve --out data/sft/evolve_train.jsonl
python -m factor_agent build-grpo --stage extract --out data/rl/extract_train.jsonl
python -m factor_agent build-grpo --stage evolve --out data/rl/evolve_train.jsonl
python -m factor_agent dry-run

# Download public 2017–2024 data, then run a local live backtest
pip install -e ".[backtest]"
python -m factor_agent prepare-market-data
python -m factor_agent backtest-smoke
python -m factor_agent serve-backtest --port 8001
python -m factor_agent eval-window-report --input data/rl/evolved_pairs.jsonl --out runs/grpo/eval_window.json
```

Demo gold is still the 5 rows in `data/gold/gold.jsonl`, not the final labeled set. Rows already carry institution/series and per-parameter citations so the evidence and grouped-split paths run on the sample.

The live-backtest MVP covers CSI300/CSI500 and the core price-volume DSL. Data files ship with source, range, and SHA-256 manifests. search/score/eval are 2018–2021, 2022–2023, 2024; 2017 is warmup for 252-day windows. The unit-test tape only checks arithmetic, not resume performance.
