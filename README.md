# 多模态内容可信度核查系统

[![CI](https://github.com/fjjca/multimodal-credibility-assessor/actions/workflows/ci.yml/badge.svg)](https://github.com/fjjca/multimodal-credibility-assessor/actions/workflows/ci.yml)

图文帖子 → 原子断言拆解 → VLM 图文取证 → 证据检索 → 证据图 → 控辩裁多模型裁决 → **结论 + 证据链 + 可视化证据图**。

面向大模型应用 / FDE 岗位的全栈 demo：FastAPI + SSE 流式进度、OpenAI 兼容多模型分工、NetworkX 证据图接地裁决、置信度阈值路由、LangGraph 可选编排（循环多轮辩论）、可复现评测与消融、Docker 一键部署。默认模型全走免费档，零充值可跑通。

---

## 核心流水线

```
文本 + 图像
   │
   ├─ 1. 断言拆解 (decompose)      把帖子拆成可独立核查的原子断言
   ├─ 2. 多模态取证 (forensics)     VLM 做 OCR + 逐断言图文一致性(一致/矛盾/无法判断)
   ├─ 3. 证据检索 (retrieve)        Tavily 检索网页证据，按来源分级
   ├─ 4. 证据图构建 (graph)          断言↔证据连 支持/矛盾 边(来源加权)，证据↔证据连 同源 边
   └─ 5. 多模型裁决 (adjudicate)     控方/辩方/裁判三角色；低置信度升级强裁判
          │
       结论(supported/refuted/misleading/unverifiable) + 证据链 + Cytoscape 证据图
```

前端为原生 JS 单页（`static/index.html`），通过 SSE 实时展示每个阶段的状态与细节，并渲染取证面板与证据图。

## 设计亮点

- **图是裁决的数据源，而非装饰**：裁判读取的是证据图里的加权关系（支持/矛盾/同源），与"向量 RAG 拼 prompt"有本质区别。
- **允许弃权**：证据不足时诚实输出 `unverifiable`，而非硬猜——校准弃权比刷高准确率更重要。
- **多模型分工 + 置信度路由**：便宜模型初筛，裁判置信度低于阈值（默认 0.6）自动升级强模型；内容审核拒答时降级不触发无谓升级。
- **全组件配置开关 = 免费消融实验**：`ENABLE_RETRIEVAL / ENABLE_DEBATE / ENABLE_ESCALATION` 一行开关即可做对照，无需改代码。
- **OpenAI 兼容接口**：所有模型调用走统一 provider:model 配置，为私有化部署埋伏笔。
- **评测与一键部署**：可复现的指标（准确率/弃权率/macro-F1/成本）+ `docker compose up` 一条命令起服务。
- **轻量可观测性**：结构化 JSON 日志 + Prometheus `/metrics`，token/延迟/失败/升级四类信号零外部服务落地，token 只计实时调用与成本口径对齐。
- **LangGraph 可选编排 + 循环辩论**：`ENABLE_LANGGRAPH` 一行切换 StateGraph 与手写直线 runner，单轮逐字节等价（parity 测试），多轮控辩裁成环收敛——把"循环条件路由 + 状态归约"做成可 A/B、可回退的一等结构，而非为用框架而重写。

## 快速开始

### 方式一：Docker 一键部署（推荐）

```bash
cp .env.example .env        # 填入 ZHIPU_API_KEY（可选 TAVILY_API_KEY）
docker compose up --build   # 境内构建默认走清华 pip 镜像；海外加 --build-arg PIP_INDEX_URL=https://pypi.org/simple
# 打开 http://localhost:8000
docker compose down         # 停止
```

> 需先启动 Docker Desktop。`.cache/`、`.data/`（缓存/trace/上传）以卷持久化。

### 方式二：本地运行

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Linux/macOS 用 .venv/bin/pip
cp .env.example .env                             # 填入 key
.venv/Scripts/python -m app                       # 起服务，默认 http://127.0.0.1:8000
```

## 配置说明

配置全部经 `.env`（或环境变量）驱动，见 `.env.example`。

| 角色 | 默认模型（全免费） | 说明 |
|---|---|---|
| decompose / prosecutor / defender | `zhipu:glm-4-flash` | 便宜文本档 |
| forensics | `zhipu:glm-4.6v-flash` | 视觉档（OCR + 图文一致性） |
| judge | `zhipu:glm-4.7-flash` | 强文本档（裁判） |
| judge_strong | `zhipu:glm-4.5-flash` | 低置信度升级目标 |

关键开关：

- `ENABLE_RETRIEVAL` / `ENABLE_DEBATE` / `ENABLE_ESCALATION`：组件消融开关。
- `ENABLE_LANGGRAPH`（默认 false）/ `DEBATE_MAX_ROUNDS`（默认 1）：编排开关与辩论轮数，详见[编排小节](#编排langgraphflag-门控可选)。
- `JUDGE_CONFIDENCE_THRESHOLD`（默认 0.6）：低于此值升级强裁判。
- `LLM_CACHE_MODE`：`off` / `readwrite`（默认） / `replay`（无 key 时回放已录制响应）。
- 换模型 / 升级付费裁判：只改 `MODEL_*` 一行，不改代码。

## 评测

评测与流水线同口径，支持断点续跑（`out/checkpoint_*.jsonl`）与单条容错。

```bash
# 单模型基线（强制二选一 / 允许弃权）
.venv/Scripts/python -m scripts.evaluate --data data/eval_seed.jsonl --system baseline_fc
# 完整流水线
.venv/Scripts/python -m scripts.evaluate --data data/eval_seed.jsonl --system pipeline
# 多模态合成用例（图像取证链路）
ENABLE_RETRIEVAL=false LLM_CACHE_MODE=off \
  .venv/Scripts/python -m scripts.evaluate --data data/multimodal_demo.jsonl --system pipeline
```

数据集：`data/eval_seed.jsonl`（14 条设计文本断言）、`data/fakeddit_subset.jsonl`（100 条 Fakeddit 真实样本）、`data/multimodal_demo.jsonl`（6 条 Pillow 合成图文用例，`scripts/make_demo_images.py` 生成）。

### 结果

**M1 单模型基线（Fakeddit 平衡 100 条）**

| 系统 | 准确率 | macro-F1 | 弃权率 |
|---|---|---|---|
| baseline_fc（强制二选一） | ≈0.505 | ≈0.526 | 13% |
| baseline（允许弃权） | — | — | 95% |

纯文本单模型对 Fakeddit ≈ 随机，作为下限基线。

**M2 流水线 vs 基线（Fakeddit 20 条子集）**

| 系统 | 准确率 | 弃权率 |
|---|---|---|
| pipeline | 0.10 | 0.80 |
| baseline_fc | 0.55 | 0.10 |

诚实结论：Fakeddit 的 `refuted` 标签多依赖**图文矛盾**，纯文本检索看不到 → 流水线正确弃权而非误判。这正是引入 M3 图像取证的动机。

**M3 多模态合成用例（6 条，检索 off / cache off）**：准确率 **1.00**、弃权 **0.00**、macro-F1 **1.00**（supported 与 refuted 均 P/R/F1=1.0）。

**M4 消融（6 条合成多模态，全免费，cache off）**

| 配置 | 图像取证 | 控辩 | 检索 | 准确率 | 弃权率 | tokens |
|---|---|---|---|---|---|---|
| full | ✓ | ✓ | off | **1.00** | 0.00 | 16,925 |
| no_image | ✗ | ✓ | off | **0.00** | 0.83 | 6,983 |
| no_debate | ✓ | ✗ | off | 1.00 | 0.00 | 9,168 |
| retrieval_on | ✓ | ✓ | on | 1.00 | 0.00 | 33,597 |

- **图像取证是承重墙**：去掉后准确率 1.0→0.0，5/6 弃权，且出现"零证据却幻觉出 supported"的样本——印证证据接地防幻觉。
- **控辩 / 检索**在图像接地的清晰用例上无准确率增益，但分别占 ~54% / ~2× tokens；检索还会注入虚假支持边需裁判覆盖。其价值在文本接地 claims。

> 口径说明：合成集是受控 demo，用于验证多模态链路与消融对照，非真实世界基准；真实多模态泛化仍待可下载基准（Fakeddit 真实图因智谱境内服务器拉取境外 URL 受限而不可用，故用 base64 合成图）。

## 可观测性

零外部服务、一个轻依赖（`prometheus-client`）、纯增量埋点：**结构化 JSON 日志（stdout）+ Prometheus `/metrics`**。不接 Langfuse / OTel 云（境内网络受限且重），先把"成本、延迟、失败、升级"四类信号落地。

```bash
curl http://localhost:8000/metrics      # Prometheus 文本 exposition，可直接被抓取
```

指标（`app/obs.py`，模块级注册，根 `/metrics` 暴露）：

| 指标 | 类型 | 标签 | 含义 |
|---|---|---|---|
| `llm_calls_total` | Counter | role / model / cached | 调用次数，缓存命中单独计数 |
| `llm_tokens_total` | Counter | role | **仅实时调用**消耗 token，缓存命中不重复计费 |
| `llm_latency_seconds` | Histogram | role | 实时调用耗时（秒级桶 0.5–120s） |
| `stage_latency_seconds` | Histogram | stage | 五阶段各自耗时 |
| `verify_requests_total` | Counter | has_image | 收到的核查请求 |
| `verify_latency_seconds` | Histogram | — | 整条流水线耗时 |
| `verify_errors_total` | Counter | — | 失败次数 |
| `verdicts_total` | Counter | verdict | 结论分布（supported/refuted/…） |
| `escalations_total` | Counter | — | 低置信度升级强裁判次数 |

结构化日志走应用 logger（`credibility`），经根 JSON handler 输出到 stdout，事件含 `verify accepted/done/failed`、`stage done`、`llm retry`，字段如 `task_id / verdict / confidence / latency_s / total_tokens / stage / attempt / error`。容器里即 `docker logs` 可采的 JSON 行。

> 埋点分层：LLM / 阶段指标在客户端与 runner 内部（评测与 serving 共用）；`verify_*` / `verdicts` / `escalations` 只在 `api._run`，评测跑批不污染 serving 计数。token 只计实时调用，与"成本敏感"口径一致。评测入口 `scripts.evaluate` 同样调用 `setup_logging`，跑批输出结构化 JSON 日志（含 `llm retry` 的 `error/delay_s`，便于诊断免费档限流）。

## 编排：LangGraph（flag 门控，可选）

直线 runner 能跑固定单轮控辩裁，但**多轮循环辩论**（裁判低置信 → 反馈控辩 → 修正论证 → 重裁，直到收敛或触顶）用直线代码手写循环 + 状态穿线很别扭。LangGraph 把"成环的条件边 + 状态归约"变成一等公民：

- **不可替代点**：`add_conditional_edges` 成环（`judge → next_round → prosecutor`）+ `Annotated[list, operator.add]` reducer 跨轮累积 results/trace——这正是直线 runner 难以干净表达的结构，不是为用框架而用框架。
- **flag 门控共存**：`ENABLE_LANGGRAPH=true` 走 `StateGraph`（`app/pipeline/graph_runner.py`），`false`（默认）走手写直线 runner。节点**复用**现有阶段函数与 `adjudicate` 共享助手（`debate_base / judge_once / needs_escalation`），结果组装走同一个 `runner._build_result`——只重排编排，不重写业务逻辑。
- **等价保证**：`DEBATE_MAX_ROUNDS=1` 时，graph 的返回 dict / SSE 事件 / 调用顺序 / trace 落盘与 linear **逐字节一致**（parity 测试断言 `lin_res == gra_res`、`lin_calls == gra_calls`）。可随时 A/B 回退，零风险。

裁决子图（整个辩论对外仍是单个 `adjudicate` 阶段 running→done）：

```
START → decompose → forensics → retrieve → graph → debate_start
                                                        │
                                        enable_debate ──┴── false ──┐
                                            │ true                  │
                                            ↓                       ↓
                               ┌──→ prosecutor → defender ─────→ judge
                               │                                  │
                               │                   route_after_judge
                               │              ┌─────────┼──────────┐
                               │       低置信&round<max  需升级     收敛/触顶
                               │              ↓         ↓          ↓
                               └───── next_round     escalate → debate_end
                                                                    │
                                                           finalize → END
```

- **收敛**：裁判置信度 ≥ 阈值即停；否则进入下一轮，把上一轮裁判意见注入控辩 prompt（`judge_feedback`），受 `DEBATE_MAX_ROUNDS` 硬上限约束（默认 1 = 现行为）。
- **升级在循环之后**：触顶或收敛后若仍低置信，再走 `escalate` 升级强裁判；内容审核降级（`judge_model=""`）既不循环也不升级。

A/B 消融（同口径、分目录、cache off 独立计 token）。多模态 `multimodal_demo` 依赖 forensics VLM，跑时若遇 `glm-4.6v-flash` 免费档 429 冷却，可改用文本 `eval_seed`（无图跳过 forensics，检索 on 接地）：

```bash
# 基线：直线 runner
ENABLE_LANGGRAPH=false LLM_CACHE_MODE=off ENABLE_RETRIEVAL=true \
  .venv/Scripts/python -m scripts.evaluate --data data/eval_seed.jsonl --system pipeline --limit 6 --out out/abl_text_linear
# LangGraph 单轮（应与基线等价）
ENABLE_LANGGRAPH=true DEBATE_MAX_ROUNDS=1 LLM_CACHE_MODE=off ENABLE_RETRIEVAL=true \
  .venv/Scripts/python -m scripts.evaluate --data data/eval_seed.jsonl --system pipeline --limit 6 --out out/abl_text_graph_r1
# LangGraph ≤3 轮循环辩论
ENABLE_LANGGRAPH=true DEBATE_MAX_ROUNDS=3 LLM_CACHE_MODE=off ENABLE_RETRIEVAL=true \
  .venv/Scripts/python -m scripts.evaluate --data data/eval_seed.jsonl --system pipeline --limit 6 --out out/abl_text_graph_r3
```

**实测结果（eval_seed 前 6 条，检索 on / cache off，全免费档）**

| 配置 | 编排 | n | 准确率 | 弃权率 | macro-F1 | tokens | 多轮 | 升级 |
|---|---|---|---|---|---|---|---|---|
| linear | 手写直线 | 6 | 0.50 | 0.167 | 0.489 | 32,811 | — | 0 |
| graph_r1 | LangGraph 单轮 | 6 | 0.50 | 0.167 | 0.489 | 31,525 | rounds=1 | 0 |
| graph_r3 | LangGraph ≤3 轮 | 5\* | 0.20 | 0.20 | 0.167 | 22,484 | rounds=1（全） | 0 |

\* r3 的 s01 因 `glm-4.7-flash` 持续 429 失败未计入；扣除该条后 r1/r3 在同 5 条上 tokens 22,356 vs 22,484（+0.6%，纯抖动），无额外轮次开销。

诚实结论：

- **live parity 成立**：graph_r1 与 linear 本次 6/6 结论逐条一致、聚合指标完全相同（acc / 弃权 / macro-F1），在离线 mock parity 之外再用真实模型佐证编排等价。
- **循环未触发**：r3 全部 `rounds=1`、裁决 `n_calls=3`——免费裁判 `glm-4.7-flash` 过度自信（置信度 0.85–1.0，连判错也 0.95），从不跌破 0.6 阈值，故控辩裁循环与置信度升级都不激活。r3 与 r1 的结论差异（s03/s04/s05）是 cache off + temperature>0 的**调用抖动**，非多轮效果；准确率下降源于 s01 限流失败 + 抖动。
- **真正瓶颈是校准，不是编排**：循环机制本身正确（离线 mock 低置信测试已证收敛 / 硬上限 / 升级），但其置信度触发器在过度自信的免费裁判下很少激活。多轮辩论的准确率增益需在裁判会表达不确定的数据 / 模型上才显现——这是下一步（置信度校准 / 更强裁判）要解决的，而非编排层。

> 成本：多轮 = 控/辩/裁各多调用 N−1 次，token 随轮数近似线性增长；本次因裁判过度自信未触发，r3 与 r1 同成本。`DEBATE_MAX_ROUNDS=1` 与直线 runner 同成本，是"零代价拿到 LangGraph 编排能力"的默认档。检查点续跑（M2）/ HITL 人工介入（M3）尚未引入，见 v2 路线。

## 测试

```bash
.venv/Scripts/python -m pytest        # 102 项，覆盖客户端/缓存/各流水线阶段/取证/路由/评测/API/可观测性/LangGraph 编排与循环辩论
```

## 项目结构

```
app/
  config.py            # pydantic-settings 配置（角色模型、开关、缓存）
  api.py / main.py     # FastAPI 路由 + SSE + 静态托管 + 根 /metrics
  obs.py               # 结构化 JSON 日志 + Prometheus 指标埋点
  llm/                 # OpenAI 兼容客户端、退避重试、录制回放缓存
  pipeline/            # decompose / forensics / retrieve / evidence_graph / graph / adjudicate / runner / graph_runner / models
  eval/                # 数据集、harness（断点续跑）、指标、基线/流水线 predictor
scripts/               # evaluate / make_demo_images / prepare_fakeddit / smoke_llm
static/                # 单页前端 + vendored cytoscape.min.js
data/                  # 评测数据集与合成图像
```

每次核查产出 JSONL trace 与证据图 JSON（`.data/traces/`），便于复盘与调试。

## 成本与限制

- **成本**：默认全免费三模型分工，预算 ≈ ¥0；升级付费裁判单条约 ¥0.003–0.005。免费档有账户级并发限制，客户端带并发上限 + 指数退避；评测支持断点续跑。
- **限制**：多模态评测基于小规模合成用例；单图输入；检索依赖 Tavily 免费档质量；裁决质量受免费模型能力约束。
- **v2 路线**：LangGraph 检查点续跑（SqliteSaver）/ HITL 人工介入（interrupt）、反向搜图 / 图片挪用检测、视频与多图、Neo4j + 可信度传播 / 多跳推理、Redis/Celery、OTel/Langfuse 分布式追踪（在现有 Prometheus+JSON 之上）、本地 vLLM、React 前端。
