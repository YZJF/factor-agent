# Resume entry: Factor Research Agent

Shape: **one-line pitch → owner role (0-to-1) → what you built → measured results**. Bold the technical terms. Chain steps with `→`.

**Three rules:**

1. **No background essays.** Write what you built and how far it got. Do not write "why this is the only correct design". Save that for the interview (see the Q&A at the bottom).
2. **No jargon that only one tribe knows.** Readers may come from quant or from AI. Spell out "expression-operator whitelist", "score cap", "JSON structure", "pairwise preference data", "excluded from the training loss". Keep GRPO / SFT / verl / rLLM / vLLM / IR. First IR mention: "after-cost information ratio (IR)".
3. **Give counts.** Name the five error families, the operator/field counts, the max turns.

**Term map:**

- Human-labeled answer — `gold` in code, `ground_truth` on the verl interface; in prose say "human annotation" or `FactorSpec`. Do not say "gold spec" in mixed Chinese/English.
- Auto-extracted labels (RD-Agent style) — **silver standard**: usable as weak supervision, never as ground truth.
- To a quant interviewer: "structured factor definition". To an AI interviewer: "gold label / ground truth".

The HTML snippet below is the public resume block. Package mapping: `docs/pipeline.md`.

| Resume bullet | Package | Status |
|---|---|---|
| 7-field spec + holdout assert | `schemas.py` `split.py` | Split and leak checks wired into `build-sft` / `build-grpo` / `score` |
| Four subscores + three caps + 14/9 whitelist + five negatives | `score/` `spec/expr.py` | Shipped; `AS_OF_BARS=1` is the language contract, honored by the backtest engine |
| Stage A / gate / Stage B frozen fields | `extract/` `gate.py` `evolve/drift.py` | Gate and frozen fields shipped; B uses a live backtest |
| Search/score windows, lag, shuffle, staged reward | `evolve/periods.py` `evolve/contrast.py` | Windows and hooks are live |
| GRPO loop + three adoption metrics | `train/` `split.py` `scripts/run_sft.sh` `scripts/run_grpo.sh` | Row format, leak checks, launch scripts exist; verl sources stay out of repo |

A is the honest material dump. B is a numbers template. **Do not ship B placeholders.**

---

## Version A: honest copy (framework shipped, labels still demo-scale)

This dump is too long for one resume page. The HTML fragment below is the five-bullet version.

> **Factor Research Agent (Agentic RL)** | report extraction + rule scoring + post-training loop — 2026.x – present | `github.com/YZJF/factor-agent`
>
> **Role:** project owner. Designed the **extract → source check → gate → multi-turn mutation** path and the post-training loop from scratch.
>
> - **Task and data:** a **7-field structured factor spec** (name / universe / frequency / neutralization / rebalance / expression / lookback windows). One example = report + human-labeled spec. Train/holdout split by case ID with an empty-intersection assert. Auto-extracted output is weak supervision only.
> - **Rule scorer (no LLM judge):** **JSON 0.25 + executable expr 0.25 + field match 0.40 + numbers must appear in the report 0.10**. Caps: unparseable JSON **0**, off-whitelist operator **0.15**, expr mismatch plus wrong universe or missing numbers **0.35**.
> - **Expression whitelist:** **14 operators** and **9 market fields**. Time-series ops default to yesterday. No arbitrary code.
> - **Extract gate:** total **≥ 0.7**, no cap, executable expr, or the case stays a Stage A negative.
> - **Two stages:** Stage A single-shot extract. Stage B up to **4 turns**, each turn mutates the expr then backtests after-cost IR. Freeze universe / frequency / neutralization.
> - **Negatives:** five mechanical edits (drop neutralization, flip frequency, ×3 windows, unrelated formula, illegal call) for verifier calibration only.
> - **Post-training:** `sample → rule score → SFT / GRPO`. GRPO via **rLLM → verl**, sampling on **vLLM**. Adoption checks: sample diversity, gold vs negative score gap, empty leak set.

> Reward attribution is a **design**, not a measured lift. Interview as "how we would layer it", not "it gained x%".

---

## Version B: fill after you train

| Placeholder | How to get it | Resume wording |
|---|---|---|
| `<n_labels>` | line count of `data/gold/gold.jsonl` | "human-labeled **N** cases" |
| `<gold_mean>` / `<neg_mean>` | `python -m factor_agent score` + `inject` | "gold self-score **0.9x**, five injected families **below 0.3x**, gap **0.6+**" |
| `<gate_pass_base>` | Stage A on the untrained base | "base model gate pass **x%**" |
| `<gate_pass_trained>` | same holdout after training | "SFT **x%**, GRPO **y%**" |
| `<hallucination_catch>` | `hallucinated_expr_cap` / human-judged fabrications | "fabricated-formula catch **x%**" |
| `<IR_lift>` | Stage B after live backtest | "after-cost IR median **a → b**" |
| `<model / gpus>` | training script | "**Qwen3-4B**, **N-GPU** FSDP" |

Prefer these three measured claims:

1. Fabricated-formula catch rate
2. Gold vs negative score gap
3. Gate pass rate, base → trained

---

## HTML fragment

```html
      <div class="proj">
        <div class="proj-title">
          <span>Factor Research Agent (Agentic RL)</span>
          <a href="https://github.com/YZJF/factor-agent">github.com/YZJF/factor-agent</a>
        </div>
        <p class="pitch">Sell-side report → structured factor spec → rule score → gate, then multi-turn mutation with programmatic GRPO.</p>
        <p class="owner"><b>Role:</b> project owner. Designed extraction, scoring, reproducibility checks, and the post-training loop from scratch.</p>
        <ul>
          <li>Designed a <b>7-field structured factor spec</b> (name / universe / frequency / neutralization / rebalance / expr / windows). One example = report + human label. Train/holdout split with an empty-intersection assert.</li>
          <li>Rule scorer (<b>no LLM judge</b>): <b>JSON 0.25 + executable 0.25 + field match 0.40 + numbers must appear in the source 0.10</b>. Caps: unparseable JSON <b>0</b>, off-whitelist op <b>0.15</b>, mismatch plus wrong universe or missing numbers <b>0.35</b>. Custom DSL: <b>14 operators + 9 fields</b>; time-series as-of yesterday. Five mechanical error families calibrate the score gap.</li>
          <li>Two stages: <b>Stage A</b> extract; admit only if <b>total ≥ 0.7 and no cap</b>. <b>Stage B</b> mutates for up to 4 turns and backtests after-cost IR. Freeze universe / frequency / neutralization.</li>
          <li>Reproducibility: mutate on the <b>search window</b>, reward on the <b>score window</b>, never touch eval; <b>one-bar lag</b> and <b>cross-section shuffle</b>; per-turn delta reward; the two stages do not leak reward.</li>
          <li>Post-training is <b>GRPO (rLLM → verl)</b> with <b>vLLM</b> sampling. Checkpoint adoption uses <b>sample diversity, gold vs negative gap, holdout leak check</b>.</li>
        </ul>
      </div>
```

---

## Interview follow-ups

### 1. Why a truth check before mutation?

Existing factor-evolution agents (AlphaAgentEvo and similar) start from a validated library such as Alpha158. Seeds extracted from sell-side notes can be absent from the report.

The failure does not look like a failure: the agent will honestly optimize a factor that was never in the note. The backtest is real, IR goes up, the reward curve goes up. You cannot see the miss from metrics. The gate exists to stop that before mutation.

### 2. Why not an LLM judge?

Gold fields are discrete enums plus a parseable expression. A judge is another noise source, and ~40 cases will not calibrate one.

### 3. Is 40 cases too few?

Yes, that is a cold-start size. Two design consequences:

- **Leak checks are an automatic gate.** Holdout is small; two leaked rows inflate the score by 20% with no trace on the training curve.
- **Injected negatives do not pad the train set.** `inject.py` can turn 40 rows into 200, but the model would learn the scripted fingerprint (windows always ×3). Negatives calibrate the verifier only.

Scale labels only after the verifier separates gold from negatives on these 40.

### 4. Is number matching too coarse for grounding?

Yes. It catches the common miss: wrong windows/params, invented numbers. Two holes: Chinese numerals such as "二十日" vs `20` (false negative, now normalized); an unrelated `60` in the document (false positive). Next step: require neighborhood words such as "day / lookback / window", not a whole-document hit.

### 5. Credit assignment on long traces?

Reward is decomposable (`Subscores` and per-field hits) and Stage B has a real IR each step. Token-level and step-level credit can be read, not learned. Pivot selection at `max_steps=4` may not be worth a mechanism designed for dozens of steps.

### 6. Can the rule scorer detect look-ahead?

No. The four subscores measure "does it copy the label". `TS_MEAN($close, 20)` can still include today's close. Fix it in the language (ops as-of yesterday) and in the backtest (lag every input one day; shuffle IC to 0). Search/score/eval are **time** splits, not factor splits.
