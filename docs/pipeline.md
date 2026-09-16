# 流程（对齐简历五条）

公共说明与安装命令见仓库根目录 [README.md](../README.md)（中英双语）。

一条样本 = 研报原文 + 人工标注的 7 字段说明书。  
`研报 → 抽取 → 规则打分 → 门禁 → 多轮变形 → 监督微调 / GRPO`

```text
factor_agent/
  documents/          PDF/Markdown 解析、证据 chunk、检索/读页/校验工具
  batch/              case/report/verifier/gold 四件套 manifest
  schemas.py            7 字段 FactorSpec / FactorCase
  split.py              样本 ID 与机构/系列 group 均不得跨 train/holdout
  spec/expr.py          14 算子 + 9 字段；时序默认截止昨日
  spec/consistency.py   中性化字段 vs 表达式算子
  spec/requirements.py  证据必填、参数须声明且被引用、引用须可定位
  score/verifier.py     四子分 + 三级上限
  score/inject.py       五类改错，只标定打分器
  protocol.py           六类非法工具调用与逐轨迹审计
  extract/run.py        阶段 A 单轮 / 长研报多轮工具模式
  gate.py               总分 ≥ 0.7 且无上限才放行
  backtest/             BaoStock → Qlib 数据、真实组合回测、API 与不可变 ledger
  evolve/run.py         阶段 B，最多 4 轮真实回测
  evolve/drift.py       冻结股票池 / 频率 / 中性化
  evolve/periods.py     搜索期改、打分期给分、评测期不碰
  evolve/contrast.py    滞后一期 / 截面打乱（回测钩子）
  evolve/eval_window.py 冻结评测窗上 seed vs 变形后的含成本 IR 中位数增量
  eval/metrics.py       三个头条指标、分组留出集报告、前后百分点对比
  probe/ + routing/     K 次采样指标与样本路由池
  rollout/              轨迹、分数与一致性 hash 落盘
  train/                SFT / GRPO + verl AgentLoop + rollout correction
  pipeline.py           串起来
```

旧文件 `verifier.py` / `inject.py` / `expr_check.py` 只做转发。

---

## 1. 说明书与切分

字段：因子名 / 股票池 / 计算频率 / 中性化 / 再平衡频率 / 表达式 / 回看窗口。  
一条样本一个 `case_id`，`split` = train | holdout。  
`split.assert_no_leak` 在 `build-sft` / `build-grpo` 里硬停；`score` 命令会打印 `leak_ids`。

分组键 `split.group_key` 依次回退：显式 `group_id` → `机构::系列` → `doc_id`。
同一家机构同一个因子系列的研报措辞和参数高度重复，只按 `case_id` 切就会泄漏，
所以留出集按机构/系列成组切，`holdout-report` 同时给出逐组指标与跨组中位数
（否则一家产量高的机构会主导平均值）。

表达式里出现的参数必须在 `windows` 里声明，并且每个参数都要有一条带 chunk 与页/字符定位的引用；
`spec/requirements.py` 不过的标注不能进四件套，也不能回放成 SFT 轨迹。

自动抽取结果不当标注。

---

## 2. 规则打分

不用大模型评判。

| 子分 | 权重 |
|---|---|
| JSON 能解析成说明书 | 0.25 |
| 表达式在白名单内可执行 | 0.25 |
| 股票池 / 频率 / 中性化 / 再平衡 / 表达式对齐标注 | 0.40 |
| 表达式与窗口里的数字出现在被引用的片段内 | 0.10 |

grounding 只认模型自己引用的 chunk，没有引用直接 0，不再退化成"全文出现过就算落地"。
数字匹配前先做中文数字归一化，"过去二十个交易日"和 `windows=[20]` 能对上。
（之前 `numbers_in_text` 用 `\b\d+\b`，而中文字符也是词字符，中文研报的 grounding 恒为 0。）
"没给引用"和"编造公式"是两件事：编造公式上限仍按全文宽松核验判定，
只有确实在研报里找不到依据才封顶 0.35。

分数上限：JSON 无法解析 → 0；白名单外算子 → ≤ 0.15；表达式与标注不符且股票池错或数字不见于原文 → ≤ 0.35。

五类注入错误（`score/inject.py`）：中性化置空、频率跨档翻转、窗口乘 3、无关公式、非法调用。只生成评测集，不进监督微调、不成对。

声明与实现不一致（中性化写了 industry 但表达式没有 `INDUSTRY_NEUTRALIZE`）记入 `diagnostics.declaration_issues`，暂不改权重。

---

## 3. 两阶段与门禁

**A** 短研报可直接吐 JSON；长研报通过 `search_report` / `read_report_pages` 读取证据，并用 `validate_factor_spec` 做结构预检。  
**门禁**（`gate.py`）：总分 ≥ 0.7、未触发任何上限、表达式可执行。未过的样本留在 A 当负样本。  
**B** 最多 4 轮真实评估：改表达式 → `validate_expr` → `evaluate_factor` → 再改。只准动表达式和窗口；改冻结字段触发 `spec_drift_cap`。

A 的分不吃 B 的回测指标。

两阶段的每次工具调用都过 `protocol.classify_call`，非法只认六类：未知工具、参数无法解析、
FactorSpec 结构非法、超出回测轮次预算、改冻结字段、单步并发多调用。
`validate_expr` 判"算子不在白名单"不算非法 —— 那是表达式质量，子分已经扣过；
回测服务挂掉也不算，单独记 `environment_failures`。逐轨迹结果写进
`Trajectory.extras["tool_audit"]`，再由 rollout store 落到 `scores.jsonl`。

---

## 4. 复现体检

| 项 | 模块 | 状态 |
|---|---|---|
| 时序算子截止昨日 | `spec/expr.py` + `backtest/engine.py` | 信号整体滞后一根交易日后再进入当期组合收益 |
| 声明 vs 实现 | `spec/consistency.py` | 已记录，未加权 |
| 搜索期 / 打分期 / 评测期 | `evolve/periods.py` 默认 2018–2021 / 2022–2023 / 2024 | search 对模型可见；score 隐藏；eval 训练不可见 |
| 滞后一期、截面打乱 | `evolve/run.py` + `backtest/engine.py` | 后端真实移动/打乱信号，固定随机种子保证复现 |
| 每轮回测增量 | `evolve/run.py` + `evolve/reward.py` | 已按隐藏 score 期 best-so-far 增量给分 |
| 冻结评测窗 | `evolve/eval_window.py` | 唯一读 eval 期的入口，离线跑，给 seed vs 变形后的含成本 IR 中位数增量 |

---

## 5. 后训练

`样本 → 采样 → 规则打分 → 两套 LoRA 各自 SFT / GRPO`。  
两个适配器都从同一份冻结的 Qwen3-4B 长出来，**进化 LoRA 不叠在抽取 LoRA 上**。  
抽取：gold 工具回放 SFT → 规则奖励 GRPO。  
变形：协议预热 SFT（`validate_expr` 后原样交出 seed，不编假 IR）→ AgentLoop 回测 GRPO。  
推理时在门禁处切换适配器。

| 本仓有的 | 不进本仓的 |
|---|---|
| `build-sft --stage` / `build-grpo --stage` 造 jsonl | `verl/` 整棵源码 |
| `scripts/run_sft.sh` / `run_grpo.sh` 的 LoRA hydra 项 | 模型权重、checkpoint |

启动前 `python -c "import verl"` 必须成功。TIS 只在阶段 B、response mask 校验通过且双 log-prob 诊断显示系统偏差后开启。

每轮看三件事：同一提示下输出是否还分散、标注均分与注入负例均分的间距、`leak_ids` 为空。

checkpoint 之间的对比只看三个头条指标，由 `holdout-report` 在按机构/系列分组的留出集上产出，
`compare-metrics` 给百分点差：`task_success_rate`（可解析 + 可执行 + 5 字段全对 + 无上限）、
`evidence_grounding_rate`（参数全部落在有效引用内）、`illegal_tool_call_rate`。

---

## 命令

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

# 显式联网准备 2017–2024 公共数据，随后运行本地真实回测
pip install -e ".[backtest]"
python -m factor_agent prepare-market-data
python -m factor_agent backtest-smoke
python -m factor_agent serve-backtest --port 8001
python -m factor_agent eval-window-report --input data/rl/evolved_pairs.jsonl --out runs/grpo/eval_window.json
```

演示数据仍是 `data/gold/gold.jsonl` 的 5 条，不是最终标注；已带机构/系列与逐条引用片段，
目的是让证据必填和分组切分这两条链路在示例数据上就能跑通。

真实回测 MVP 仅覆盖 CSI300/CSI500 和核心量价 DSL。数据文件带来源、范围与
SHA-256 manifest；search/score/eval 分别为 2018–2021、2022–2023、2024，
2017 只用于 252 日窗口预热。测试行情只用于验证计算正确性，不作为简历效果证据。
