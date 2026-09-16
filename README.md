# Factor Research Agent

**[English](#english)** · **[中文](#中文)**

Sell-side reports → evidence-grounded FactorSpec → programmatic gate → two LoRA adapters (SFT then GRPO) → multi-turn evolution on a frozen spec.

> Shipped gold is **5 demo cases**. The code path is complete; do not treat demo scores as trained results.

---

## English

### What this is

An agentic pipeline that turns sell-side research reports into executable equity factors, then post-trains two LoRA adapters with programmatic rewards. Unlike [AlphaAgentEvo](https://openreview.net/forum?id=lNmZrawUMu), which evolves already-validated library seeds (e.g. Alpha158), this project must **extract** the seed from a report. A hallucinated formula still produces a real IR curve, so extraction is gated before evolution is allowed.

One example = report text + a human-annotated 7-field `FactorSpec`:

`name` · `universe` · `frequency` · `neutralization` · `rebalance` · `expr` · `windows`

plus chunk/page/character evidence. Model-extracted specs are never treated as gold.

The extractor still understands Chinese sell-side notes (including Chinese numerals such as 「二十个交易日」). Demo gold in this repo is English.

### Necessary components

verl sources, model weights, and CSI market dumps stay out of this repo.

```text
report PDF/MD
    │  documents/          pagination, Chinese-numeral normalization, literal search/read
    ▼
FactorSpec JSON            schemas.py + schema/factor_spec.json
    │  extract/            single-shot for short reports; search → read → validate for long ones
    │  protocol.py         six illegal tool-call classes
    ▼
rule scorer                score/verifier.py   no LLM judge
    │  spec/               14 ops · 9 fields · required evidence · declaration checks
    │  gate.py             score ≥ 0.7 and no caps before Stage B
    ▼
two LoRAs on the same frozen Qwen3-4B (evolve is not stacked on extract)
    │  extract LoRA        gold tool-call SFT replay → rule-reward GRPO
    │  evolve  LoRA        protocol warmup SFT → backtest-delta GRPO
    ▼
Stage B evolution          evolve/  up to 4 turns of validate_expr → evaluate_factor
    │  freeze universe / frequency / neutralization
    │  search visible · score hidden for reward · eval never seen in training
    ▼
backtest service           backtest/  BaoStock → Qlib → FastAPI + append-only ledger
```

| Piece | Path | Role |
|---|---|---|
| Document tools | `factor_agent/documents/` | Paginate PDF/Markdown; normalize Chinese numerals; literal `search_report` / `read_report_pages` |
| Spec | `schemas.py` · `schema/factor_spec.json` | 7-field FactorSpec |
| Asset batch | `batch/` | `case / report / verifier / gold` with content hashes |
| Split | `split.py` | Institution/series groups; hard-fail on train/holdout leak |
| Whitelist | `spec/expr.py` | 14 operators, 9 fields; time-series as-of yesterday |
| Evidence rules | `spec/requirements.py` | Params must be declared, cited, and locatable |
| Verifier | `score/verifier.py` | JSON 0.25 + executable 0.25 + field match 0.40 + grounding 0.10 |
| Negatives | `score/inject.py` | Five mechanical error families for verifier calibration only |
| Protocol audit | `protocol.py` | Unknown tool, bad args, illegal spec, over budget, frozen-field edit, multi-call in one step |
| Extract + gate | `extract/` · `gate.py` | Stage A; admit only if ≥0.7, no cap, executable expr |
| Evolve | `evolve/` | ≤4 turns; freeze three fields; search/score/eval time split |
| Backtest | `backtest/` | CSI300/CSI500, one-bar lag, cross-section shuffle, append-only ledger |
| Metrics | `eval/metrics.py` | Task success, evidence grounding, illegal tool-call rate |
| Routing | `probe/` · `routing/` | K probes → SFT / GRPO / eval / quarantine pools |
| Rollout store | `rollout/` | Trajectories, rewards, prompt/tool hashes; tool observations masked out of loss |
| Training | `train/` · `scripts/` | Per-stage LoRA SFT/GRPO builders and hydra launchers |

Package-level map: [`docs/pipeline.md`](docs/pipeline.md).

### Stage A reward

No LLM judge. Grounding is checked **only inside cited chunks**. Numerals are normalized so 「二十个交易日」 matches `windows=[20]`.

| Subscore | Weight |
|---|---|
| JSON parses as FactorSpec | 0.25 |
| Expression is executable on the whitelist | 0.25 |
| Universe / frequency / neutralization / rebalance / expr match gold | 0.40 |
| Expr and window numbers appear in cited spans | 0.10 |

Caps: illegal JSON → 0; off-whitelist op ≤ 0.15; bad citation ≤ 0.20; hallucinated expr ≤ 0.35.

Gate: total ≥ 0.7 **and** no cap **and** executable. Failures stay in Stage A as negatives.

### Stage B

Only gated specs may mutate. Up to four turns: edit `expr` / `windows` → `validate_expr` → `evaluate_factor`. Editing frozen fields is an illegal call and triggers `spec_drift_cap`.

| Window | Default | Visibility |
|---|---|---|
| search | 2018–2021 | Shown to the model |
| score | 2022–2023 | Hidden; training reward is after-cost IR delta vs seed |
| eval | 2024 | Never used in training; checkpoint offline eval only |

Robustness hooks: one-bar lag; cross-sectional shuffle (IC should collapse to 0).

Both LoRAs grow from the **same frozen Qwen3-4B**. The evolve adapter is not stacked on the extract adapter.

### In this repo / not in this repo

| Included | Not included |
|---|---|
| `factor_agent/`, `tests/`, `configs/`, `scripts/` | `verl/` sources (install so `import verl` works) |
| 5 demo gold rows | Model weights and LoRA checkpoints |
| JSON Schema and training configs | BaoStock / Qlib market dumps (~650MB+; run `prepare-market-data`) |
| Launch scripts | Proprietary sell-side PDFs |

### Install

Python ≥ 3.10.

```bash
git clone https://github.com/YZJF/factor-agent.git
cd factor-agent
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Optional extras:

```bash
pip install -e ".[pdf]"        # PDF parsing
pip install -e ".[train]"      # jsonl → parquet
pip install -e ".[backtest]"   # BaoStock + Qlib + FastAPI
```

### Commands

```bash
python -m factor_agent score
python -m factor_agent holdout-report --out runs/base/holdout.json
python -m factor_agent compare-metrics --before runs/base/holdout.json --after runs/grpo/holdout.json
python -m factor_agent inject
python -m factor_agent dry-run

python -m factor_agent build-sft --stage extract --out data/sft/extract_train.jsonl
python -m factor_agent build-sft --stage evolve --out data/sft/evolve_train.jsonl
python -m factor_agent build-grpo --stage extract --out data/rl/extract_train.jsonl
python -m factor_agent build-grpo --stage evolve --out data/rl/evolve_train.jsonl
```

Real backtests (downloads public 2017–2024 data):

```bash
pip install -e ".[backtest]"
python -m factor_agent prepare-market-data
python -m factor_agent backtest-smoke
python -m factor_agent serve-backtest --port 8001
python -m factor_agent eval-window-report --input data/rl/evolved_pairs.jsonl \
  --endpoint http://localhost:8001/backtest --out runs/grpo/eval_window.json
```

Post-training (verl must be installed; weights stay on disk):

```bash
python scripts/jsonl_to_parquet.py --in data/sft/extract_train.jsonl --out data/sft/extract_train.parquet
python scripts/jsonl_to_parquet.py --in data/sft/evolve_train.jsonl --out data/sft/evolve_train.parquet
MODEL=/path/to/Qwen3-4B STAGE=extract bash scripts/run_sft.sh
MODEL=/path/to/Qwen3-4B STAGE=evolve  bash scripts/run_sft.sh
STAGE=extract MODEL=/path/to/Qwen3-4B ADAPTER_PATH=checkpoints/extract_lora/sft bash scripts/run_grpo.sh
STAGE=evolve  MODEL=/path/to/Qwen3-4B ADAPTER_PATH=checkpoints/evolve_lora/sft \
  FACTOR_BACKTEST_URL=http://localhost:8001/backtest bash scripts/run_grpo.sh
```

Enable TIS / rollout correction only on Stage B after response-mask checks and a confirmed log-prob gap between vLLM and FSDP.

### Backtest scope

CSI300 / CSI500, core OHLCV / amount / return / industry, and the listed operators. CSI1000, ChiNext, all-A, money-flow, and `COUNT` fail loudly rather than silently degrading. Unit tests use a tiny deterministic tape and are not performance evidence.

---

## 中文

### 这是什么

从卖方研报抽出可执行的量化因子，再用程序化奖励做后训练。和 [AlphaAgentEvo](https://openreview.net/forum?id=lNmZrawUMu) 的差别：他们从 Alpha158 这类**已验证种子**出发改公式；这里的种子是模型读研报读出来的，可能根本不在原文里。假因子一旦进回测，IR 曲线一样会很好看，所以必须先过抽取门禁，再允许演化。

一条样本 = 研报原文 + 人工标注的 7 字段 `FactorSpec`：

`name` · `universe` · `frequency` · `neutralization` · `rebalance` · `expr` · `windows`

另附 chunk / 页 / 字符级证据。自动抽取结果不当标注。

### 必要组成部分

整条链路只依赖下面这些模块。verl 源码、模型权重、CSI 行情都不进本仓。

```text
研报 PDF/MD
    │  documents/          分页、中文数字归一化、字面检索/读页
    ▼
FactorSpec JSON            schemas.py + schema/factor_spec.json
    │  extract/            短研报单轮；长研报 search → read → validate
    │  protocol.py         六类非法工具调用审计
    ▼
规则打分                   score/verifier.py   不用 LLM Judge
    │  spec/               14 算子 · 9 字段 · 证据必填 · 声明对齐
    │  gate.py             ≥0.7 且无分数上限，才进阶段 B
    ▼
两套 LoRA（同一冻结 Qwen3-4B，互不叠加）
    │  extract LoRA        gold 工具回放 SFT → 规则奖励 GRPO
    │  evolve  LoRA        协议预热 SFT → 回测增量 GRPO
    ▼
阶段 B 演化                evolve/  最多 4 轮 validate_expr → evaluate_factor
    │  冻结 universe / frequency / neutralization
    │  search 可见 · score 隐藏给奖 · eval 训练不可见
    ▼
回测服务                   backtest/  BaoStock → Qlib → FastAPI + 不可变 ledger
```

| 组件 | 路径 | 职责 |
|---|---|---|
| 文档工具 | `factor_agent/documents/` | PDF/Markdown 分页；中文数字归一化；`search_report` / `read_report_pages` 只做字面搜索 |
| 说明书 | `schemas.py` · `schema/factor_spec.json` | 7 字段 FactorSpec |
| 四件套 | `batch/` | `case / report / verifier / gold` + 内容 hash |
| 切分 | `split.py` | 按机构/系列分组；train/holdout 泄漏硬停 |
| 白名单 | `spec/expr.py` | 14 算子、9 字段；时序默认截止昨日 |
| 证据校验 | `spec/requirements.py` | 参数须声明、须被引用、引用须可定位 |
| 打分器 | `score/verifier.py` | JSON 0.25 + 可执行 0.25 + 字段对齐 0.40 + grounding 0.10 |
| 负例 | `score/inject.py` | 五类机械改错，只标定打分器，不进 SFT |
| 协议审计 | `protocol.py` | 未知工具 / 参数无法解析 / 结构非法 / 超预算 / 改冻结字段 / 单步并发多调用 |
| 抽取 | `extract/` · `gate.py` | 阶段 A；门禁：≥0.7、无 cap、表达式可执行 |
| 演化 | `evolve/` | 最多 4 轮；冻结三字段；search/score/eval 时间切开 |
| 回测 | `backtest/` | CSI300/CSI500、滞后一根 K 线、截面打乱、append-only ledger |
| 指标 | `eval/metrics.py` | 任务成功率、证据落地率、非法工具调用率 |
| 路由 | `probe/` · `routing/` | K 次 Probe 后分流到 SFT / GRPO / eval / quarantine |
| 落盘 | `rollout/` | 轨迹、reward、prompt/tool hash；工具 observation 的 loss mask = 0 |
| 训练 | `train/` · `scripts/` | 两套 LoRA 的 SFT/GRPO 数据与 hydra 启动脚本 |

更细的包对照见 [`docs/pipeline.md`](docs/pipeline.md)。

### 规则奖励（阶段 A）

不用大模型评判。grounding 只在模型**实际引用的 chunk** 内核验，数字先做中文归一化（「二十个交易日」能对上 `windows=[20]`）。

| 子分 | 权重 |
|---|---|
| JSON 能解析成 FactorSpec | 0.25 |
| 表达式在白名单内可执行 | 0.25 |
| 股票池 / 频率 / 中性化 / 再平衡 / 表达式对齐标注 | 0.40 |
| 表达式与窗口中的数字出现在被引用片段内 | 0.10 |

分数上限：非法 JSON → 0；白名单外算子 ≤ 0.15；无效引用 ≤ 0.20；编造公式 ≤ 0.35。

门禁：总分 ≥ 0.7 **且** 未触发任何上限 **且** 表达式可执行。未过的样本留在阶段 A 当负样本。

### 阶段 B

过门禁之后才允许改公式。模型最多 4 轮：改 `expr` / `windows` → `validate_expr` → `evaluate_factor`。改冻结字段记为非法调用并触发 `spec_drift_cap`。

| 窗口 | 默认区间 | 谁能看见 |
|---|---|---|
| search | 2018–2021 | 模型可见，用来改 |
| score | 2022–2023 | 隐藏，只给训练奖励（相对 seed 的含成本 IR 增量） |
| eval | 2024 | 训练全程不碰，只在 checkpoint 离线评估 |

对照：信号整体滞后一根交易日；截面打乱后 IC 应趋近 0。

两个 LoRA 都从**同一份冻结的 Qwen3-4B** 长出来，进化适配器不叠在抽取适配器上。

### 仓库里有 / 没有

| 进本仓 | 不进本仓 |
|---|---|
| `factor_agent/` 代码、`tests/`、`configs/`、`scripts/` | `verl/` 源码（本机 `import verl`） |
| 5 条演示 gold | 模型权重、LoRA checkpoint |
| JSON Schema、奖励与训练配置 | BaoStock / Qlib 行情（约 650MB+，本地 `prepare-market-data`） |
| 启动脚本 | 真实卖方研报 PDF |

### 安装

Python ≥ 3.10。命令与英文部分相同。

### 命令

命令与英文部分相同。真实回测需联网下载 2017–2024 公共数据；后训练需本机已安装 verl 与权重。

TIS / rollout 校正只在阶段 B、response mask 校验通过、且双 log-prob 诊断确认存在系统偏差后再开。

### 回测边界

当前引擎覆盖 CSI300 / CSI500、核心 OHLCV / 成交额 / 收益 / 行业，以及白名单量价算子。CSI1000、创业板、全 A、资金流、`COUNT` 会明确报不支持，不会静默降级。单元测试用确定性微型行情，不能当效果证据。

---

## License

Code in this repository is provided for research and portfolio use. Add a license file if you need a specific grant.
