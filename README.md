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

前端为原生 JS 单页（`static/index.html`），通过 SSE 实时展示每个阶段的状态与细节。结果默认渲染**人读报告视图**：断言核查清单（每条断言的支持/矛盾证据 + 来源分级 + 原文链接）、裁决依据（裁判证据链 + 可折叠控辩论证）、取证面板、交互证据图、可折叠运行信息（模型/token/追踪 ID）；原始 JSON 收进「查看原始 JSON」切换按钮，`graph_elements` 等冗余字段不再直接糊在页面上。证据图（本地 vendored Cytoscape，无 CDN）支持：**点击节点/边出详情卡**（证据的来源分级/检索分/原文链接/内容摘录/关联断言，断言的支持/矛盾证据列表可点击跳转）、**选中邻域高亮**其余淡出、**图例即过滤器**（支持/矛盾/同源 × 官方/主流媒体/社交/未知）、**边宽 ∝ 来源 tier 权重**、**一键高亮裁决引用的证据链**（`evidence_chain` 定位居中）、适应画布/重新布局。节点数据经 `graph_elements` 携带 url/content/score（只读展示）；外链仅放行 http(s) 协议。

## 设计亮点

- **图是裁决的数据源，而非装饰**：裁判读取的是证据图里的加权关系（支持/矛盾/同源），与"向量 RAG 拼 prompt"有本质区别。
- **允许弃权**：证据不足时诚实输出 `unverifiable`，而非硬猜——校准弃权比刷高准确率更重要。
- **多模型分工 + 置信度路由**：便宜模型初筛，裁判置信度低于阈值（默认 0.6）自动升级强模型；内容审核拒答时降级不触发无谓升级。
- **全组件配置开关 = 免费消融实验**：`ENABLE_RETRIEVAL / ENABLE_DEBATE / ENABLE_ESCALATION` 一行开关即可做对照，无需改代码。
- **OpenAI 兼容接口**：所有模型调用走统一 provider:model 配置，为私有化部署埋伏笔。
- **评测与一键部署**：可复现的指标（准确率/弃权率/macro-F1/成本）+ `docker compose up` 一条命令起服务。
- **轻量可观测性**：结构化 JSON 日志 + Prometheus `/metrics`，token/延迟/失败/升级四类信号零外部服务落地，token 只计实时调用与成本口径对齐。
- **LangGraph 可选编排 + 循环辩论 + 检查点续跑 + HITL**：`ENABLE_LANGGRAPH` 一行切换 StateGraph 与手写直线 runner，单轮逐字节等价（parity 测试），多轮控辩裁成环收敛；`ENABLE_CHECKPOINT` 叠加 SqliteSaver 按 thread_id 持久化、崩溃后同 thread_id 续跑不重算；`ENABLE_HITL` 在弃权/低置信处 `interrupt` 暂停等人工复核、`Command(resume=...)` 注入裁决续跑——把"循环条件路由 + 状态归约 + 持久化 + 人工介入"做成可 A/B、可回退的一等结构，而非为用框架而重写。
- **事后置信度校准 + 评测 Dashboard**：温度缩放（纯 Python，无 numpy/scipy）软化免费裁判的过度自信，报 ECE/MCE/Brier 与 **LOO-CV** 诚实小样本指标；`/dashboard.html` 纯 SVG（无 CDN）渲染可靠性曲线、置信度分布与消融对比。校准只变换上报置信度，编排路由仍用原始 conf，parity 不被破坏。

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

| 角色 | 当前实配（2026-11-21 实测可达） | 说明 |
|---|---|---|
| decompose / prosecutor / defender | `zhipu:glm-4-flash` | 便宜文本档，体检 ~1.0s 稳定 |
| forensics | `siliconflow:Qwen/Qwen3-VL-8B-Instruct` | 视觉档（OCR + 图文一致性），zhipu VLM 持续 429 后前移 |
| judge | `zhipu:glm-4.5-flash` | 强文本档（裁判），zhipu `glm-4.7-flash` 持续 429 后前移 |
| judge_strong | `siliconflow:THUDM/GLM-4-9B-0414` | 低置信度升级目标（消融实测最优） |

> 上表是**实测后的实配**（`.env` 里的当前值）；`.env.example` 已按同一套写好，复制即用。代码内置默认值见 `app/config.py`（供应商原始搭配，未做体检重排）。两者差异与依据见[模型编排小节](#模型编排先免费后付费链上顺序--优先级)。

关键开关：

- `ENABLE_RETRIEVAL` / `ENABLE_DEBATE` / `ENABLE_ESCALATION`：组件消融开关。
- `ENABLE_LANGGRAPH`（默认 false）/ `DEBATE_MAX_ROUNDS`（默认 1）：编排开关与辩论轮数，详见[编排小节](#编排langgraphflag-门控可选)。
- `ENABLE_CHECKPOINT`（默认 false）/ `CHECKPOINT_PATH`：LangGraph 检查点续跑（M2），仅 `ENABLE_LANGGRAPH=true` 时生效，详见[检查点小节](#检查点续跑m2enable_checkpoint-门控)。
- `ENABLE_HITL`（默认 false）：人工介入（M3），弃权/低置信裁决暂停等人工复核，隐含启用检查点；**必须同时开 `ENABLE_LANGGRAPH=true`**（review 节点只在图路径里存在），详见[HITL 小节](#hitl-人工介入m3enable_hitl-门控)。
- `ENABLE_CALIBRATION`（默认 false）/ `CALIBRATOR_PATH`（默认 `.data/calibrator.json`）：事后置信度校准开关与 artifact 路径，详见[校准小节](#置信度校准与评测-dashboard)。
- `JUDGE_CONFIDENCE_THRESHOLD`（默认 0.6）：低于此值升级强裁判；HITL 开启时也是触发人工复核的阈值。
- `LLM_CACHE_MODE`：`off` / `readwrite`（默认） / `replay`（无 key 时回放已录制响应）。
- `LLM_ROLE_FALLBACKS` / `LLM_CIRCUIT_THRESHOLD` / `LLM_RETRY_BUDGETS`：限流止血三件套（重试预算、熔断、兜底候选链），详见[限流止血小节](#限流止血重试预算--熔断--兜底候选链)。
- `LLM_NO_THINKING_MODELS`：声明不支持 `thinking` 参数的模型（视觉模型对 `enable_thinking` 返回 400），客户端自动剔除该参数，详见[参数兼容小节](#参数兼容模型不支持-thinking-时的自愈)。
- 换模型 / 升级付费裁判：只改 `MODEL_*` 一行，不改代码；链上顺序即优先级，付费模型放最后，详见[模型编排小节](#模型编排先免费后付费链上顺序--优先级)。

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

> 上表 M4 消融是**链上重排之前**的配置跑的（旧裁判）。当前实配下的最新数字见[模型编排小节](#模型编排先免费后付费链上顺序--优先级)；两者口径均为检索 off / 缓存 off（`retrieval_on` 一行除外）。

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
| `llm_rate_limited_total` | Counter | role / model | 429 次数 |
| `llm_fallbacks_total` | Counter | role / from_model / to_model | 兜底候选切换次数 |
| `llm_circuit_open_total` | Counter | role / model | 熔断触发次数 |
| `llm_param_stripped_total` | Counter | role / model | 因模型不支持参数而剔除参数后重试的次数 |

结构化日志走应用 logger（`credibility`），经根 JSON handler 输出到 stdout，事件含 `verify accepted/done/failed`、`stage done`、`llm retry`，字段如 `task_id / verdict / confidence / latency_s / total_tokens / stage / attempt / error`。容器里即 `docker logs` 可采的 JSON 行。

> 埋点分层：LLM / 阶段指标在客户端与 runner 内部（评测与 serving 共用）；`verify_*` / `verdicts` / `escalations` 只在 `api._run`，评测跑批不污染 serving 计数。token 只计实时调用，与"成本敏感"口径一致。评测入口 `scripts.evaluate` 同样调用 `setup_logging`，跑批输出结构化 JSON 日志（含 `llm retry` 的 `error/delay_s`，便于诊断免费档限流）。

## 限流止血：重试预算 / 熔断 / 兜底候选链

免费档最常见的故障是 `429 Too Many Requests`（账户级配额，非瞬时抖动）。旧行为是"同一模型 6 次重试 × 120s 退避"，一次限流就把整条流水线拖死几分钟，还把所有请求都打在同一个已冷却的模型上。本节是四层止血 + 一层诊断。

| 层 | 机制 | 配置 | 效果 |
|---|---|---|---|
| L1 | **按角色重试预算** | `LLM_MAX_RETRIES`（默认 3）、`LLM_RETRY_BUDGETS`（按角色覆盖） | 视觉档/裁判档限流多，单独调小，不拖全局 |
| L1 | **退避封顶** | `LLM_RATE_LIMIT_BACKOFF_SECONDS`（起步 15s）、`LLM_RATE_LIMIT_BACKOFF_MAX_SECONDS`（上限 60s） | 指数增长与 `Retry-After` 都封顶，单次调用不会卡几分钟 |
| L2 | **按模型熔断** | `LLM_CIRCUIT_THRESHOLD`（连续 3 次 429 触发）、`LLM_CIRCUIT_COOLDOWN_SECONDS`（300s） | 熔断期内直接跳过该模型，不再浪费调用；冷却结束自动复位重试 |
| L3 | **兜底候选链** | `LLM_ROLE_FALLBACKS`（按角色的 `provider:model` 数组） | 主模型限流时切备胎，**跨平台**（SiliconFlow）配额池独立，互不拖累 |
| L4 | **取证降级 + 体检脚本 + 观测** | `scripts/probe_models`、dashboard 面板、`/metrics` 计数器 | VLM 不可用时降级为"图文一致性未评估"并继续出结论，而不是整条流水线崩掉 |

- **降级语义**：取证阶段遇 429/熔断，返回 `overall=uncertain` 的取证结果并继续跑（文本核查照常出结论）。前端"取证面板"会显式标注降级原因——**降级不等于通过**，不会静默伪装成"无矛盾"。
- **兜底链不退回主模型**：显式指定 `model_override` 时只用备胎、不回退主模型，避免把重试打在已冷却的主模型上。
- **跨平台兜底**（2026-11-21 体检实测可达）：`LLM_ROLE_FALLBACKS={"judge":["zhipu:glm-4.5-flash","siliconflow:zai-org/GLM-4.5-Air"],"forensics":["siliconflow:Qwen/Qwen3-VL-8B-Instruct","siliconflow:PaddlePaddle/PaddleOCR-VL-1.5"]}`。SiliconFlow 需要自己在 `.env` 填 `SILICONFLOW_API_KEY`（不同平台配额池独立，才是真兜底而非同池互搏）。视觉备胎选 `Qwen3-VL-8B-Instruct`（通用 VLM，能按取证 prompt 输出 JSON）而非纯 OCR 型号——纯 OCR 型号回不了 `{ocr_text, claims[]}` 结构。
- **体检脚本**（每个模型只打一次，输出 `out/model_probe.json`）：

  ```bash
  .venv/Scripts/python -m scripts.probe_models
  ```

  输出每个候选模型的 `ok / rate_limited / error_NNN`，据此决定谁是主、谁是备胎——不要靠猜。

- **观测**：新增 `llm_rate_limited_total`（role/model）、`llm_fallbacks_total`（role/from_model/to_model）、`llm_circuit_open_total`（role/model）、`llm_param_stripped_total`（role/model）四个计数器，`/dashboard.html` 的"限流与降级（本次进程）"面板直接展示 429 / 熔断 / 备胎切换 / 参数剔除次数与明细表。
- **兜底触发条件**：限流/熔断，以及重试耗尽后的**超时 / 连接失败 / 空响应 / 5xx**——免费档"卡住"和 429 一样致命，都该换人。**一般 4xx 不触发**：请求本身有问题（鉴权失败等），换模型同样会失败，只白烧一次配额；**例外是"该模型不支持某参数"**（如 400/20015），那是模型自身限制，同一请求对备胎可能合法，值得换人。取证阶段额外覆盖 429（降级继续，见上）。

### 体检实测（2026-11-21，免费档，逐模型单次调用）

| 角色 | 模型 | 结果 |
|---|---|---|
| decompose / prosecutor / defender | `zhipu:glm-4-flash` | ✅ ok（~1.0s） |
| forensics | `zhipu:glm-4.6v-flash` | ❌ 429 配额受限 |
| forensics 备胎 | `siliconflow:Qwen/Qwen3-VL-8B-Instruct` | ✅ ok（~1.1s） |
| forensics 备胎 | `siliconflow:PaddlePaddle/PaddleOCR-VL-1.5` | ✅ ok（~0.1s） |
| judge | `zhipu:glm-4.7-flash` | ❌ 429 配额受限 |
| judge 备胎 | `zhipu:glm-4.5-flash` | ✅ ok（~0.9s） |
| judge 备胎 | `siliconflow:zai-org/GLM-4.5-Air` | ✅ ok（~15s，较慢） |

结论与实配置：**限流不是"全部挂了"，而是集中在 VLM 档（`glm-4.6v-flash`）与裁判档（`glm-4.7-flash`）**——两者恰好是链路上最关键的两个模型。

- 据此把 `glm-4.5-flash`（体检可用）配成 judge 的兜底候选：`LLM_ROLE_FALLBACKS={"judge":["zhipu:glm-4.5-flash","siliconflow:zai-org/GLM-4.5-Air"]}`，并给视觉/裁判档收紧重试预算（`LLM_RETRY_BUDGETS={"forensics":1,"judge":1}`）——限流是账户级冷却，原地多等 45s 也等不到额度，不如留一次 15s 重试后立刻换备胎。
- **线上验证**：曾因 `glm-4.7-flash` 持续 429 而失败的样本 `s01`，在兜底链生效后跑通，trace 记录 `judge_model=glm-4.5-flash`、裁决正常产出（该次检索 off，故为 `unverifiable` 弃权而非误判）。原本"整条流水线崩掉"变成"换个模型继续跑"。
- **收紧重试预算的实测收益**（同一样本 s01，缓存关闭）：judge 预算 2→1 后，429 只留一次 15s 重试即切备胎，端到端延迟 **128s → 51s**（-60%），裁决结论与 token 消耗基本不变（7661 vs 7647）。限流是账户级冷却，原地多等 45s 等不到额度，纯属浪费。
- **forensics 已配跨平台备胎**：同平台 VLM 档一起限流，故走 SiliconFlow——填好 `SILICONFLOW_API_KEY` 后用 `LLM_ROLE_FALLBACKS={"forensics":["siliconflow:Qwen/Qwen3-VL-8B-Instruct","siliconflow:PaddlePaddle/PaddleOCR-VL-1.5"]}`；两档都没配时取证走**降级**（`uncertain` + 显式标注，文本核查照常出结论）。
- **后续处置**：体检结论直接推动了链上重排——把体检可用的模型提到主位、429 的留在链上做备胎，见下节[模型编排](#模型编排先免费后付费链上顺序--优先级)。

## 模型编排：先免费、后付费（链上顺序 = 优先级）

免费档最贵的成本不是钱，是**额度**：zhipu 的 VLM 档（`glm-4.6v-flash`）与裁判档（`glm-4.7-flash`）持续 429，把链路最关键的两个角色卡死。与其等它冷却，不如把**能用的免费模型直接提到主位**，把原来那个留在链上做备胎。

**一条规则，无需改代码**：`MODEL_<ROLE>` 是链首（主），`LLM_ROLE_FALLBACKS[role]` 是链身，**链上顺序就是降级优先级**。所以：

- 想"先免费、后付费"：把付费模型放在兜底链**最后一位**即可，代码零改动。
- 想把付费模型当"重核"位：放 `MODEL_JUDGE_STRONG`（低置信度升级目标），主链仍全免费。
- 升级位（`model_override`）只向备胎降级、不回退主模型（见[兜底链不退回主模型](#限流止血重试预算--熔断--兜底候选链)），所以付费模型挂升级位不会污染常规路径的额度。

### 实配（2026-11-21 实测可达）

```
forensics     siliconflow:Qwen/Qwen3-VL-8B-Instruct
              → siliconflow:Qwen/Qwen3-VL-8B-Thinking
              → zhipu:glm-4.6v-flash
judge         zhipu:glm-4.5-flash
              → siliconflow:zai-org/GLM-4.5-Air
              → siliconflow:THUDM/GLM-4-9B-0414
              → zhipu:glm-4.7-flash
judge_strong  siliconflow:THUDM/GLM-4-9B-0414
```

两家平台的配额池独立，所以主备**跨平台交错**：zhipu 冷却时切 SiliconFlow，SiliconFlow 冷却时还有 zhipu 兜着。

### 选型依据（实测，不靠猜）

**裁判档候选**（15 条带标注文本种子，开启升级）：

| 候选 | 准确率 | macro-F1 | 观察 |
|---|---|---|---|
| `zhipu:glm-4.5-flash` | 0.533 | 0.556 | 体检可用、~0.9s，取作主位 |
| `siliconflow:zai-org/GLM-4.5-Air` | 0.533 | 0.556 | 同为 15s 级慢模型，列为备胎 |
| `siliconflow:THUDM/GLM-4-9B-0414` | 0.533 | 0.556 | 文本档准确率与前两者持平，但多模态明显更强 → 改放升级位 |

三者准确率**打平**，差别在弃权率与延迟；因此裁判主位给最快最稳的 `glm-4.5-flash`，强模型留给升级位。

**视觉档取证质量**（6 条多模态合成用例，对照 gold 断言）：

| 取证模型 | 与 gold 一致 |
|---|---|
| `siliconflow:Qwen/Qwen3-VL-8B-Instruct` | **6 / 6** |

OCR 与图文一致性描述均正确，故取代 429 的 `glm-4.6v-flash` 上主位。

**升级目标消融**（多模态 6 条 + 文本种子 15 条）：

| 升级目标 | 多模态 准确率 | 多模态 macro-F1 | 文本种子 准确率 / macro-F1 |
|---|---|---|---|
| `siliconflow:THUDM/GLM-4-9B-0414` | **0.833** | **0.900** | **0.600 / 0.644** |
| 同模型自升级（judge 自身） | 0.667 | 0.750 | 0.533 / 0.556 |
| `siliconflow:zai-org/GLM-4.5-Air` 作主 | 0.500 | 0.500 | — |

`THUDM/GLM-4-9B-0414` 在两个口径上都最优，故放 `MODEL_JUDGE_STRONG`。

> 诚实口径：文本种子只有 15 条，0.600 vs 0.533 的差距在单条样本以内，只能说"不劣且多模态明显更好"；多模态只有 6 条，0.833 即 5/6，与历史 1.000（用现 429 的 `glm-4.7-flash` 裁判跑出）差一条样本。样本量小，结论用于**排序**而非宣称提升幅度。

> 上述多模态口径为 **检索 off / 缓存 off**（`ENABLE_RETRIEVAL=false LLM_CACHE_MODE=off`）。开启检索时旧实现会退步（图像断言被检索噪声带偏，见下节"多模态证据的两个修正"），修正后检索 on/off 口径一致。

### 加付费模型的路径

将来要接付费档（如 `zhipu:glm-5.3-flash`，约 ¥0.003–0.005/条）时：把 `MODEL_JUDGE` 保持免费档，把付费模型加到 `LLM_ROLE_FALLBACKS["judge"]` 的**末位**，或直接写进 `MODEL_JUDGE_STRONG`。链路顺序即优先级，不需要改任何代码。

## 多模态证据的两个修正：图像是主体，不是"社交内容"

**真实案例（2026-11-21）**：上传一张红烧鸡的照片，断言"这张照片拍摄于 2020 年东京奥运会现场"，系统判 `unverifiable`（置信度 0.35）——理由"支持/矛盾证据均衡"。但照片里除了鸡和青椒什么都没有，答案显然是 `refuted`。

从 LangGraph 检查点里取出当时的证据图，问题一目了然：

| 断言 | 支持边 | 矛盾边 |
|---|---|---|
| c1 拍摄于 2020 年 | e1 新华网《2020年新华社年度照片》 `support` **1.0**（官方） | img1 `contradict` **0.4** |
| c2 拍摄地点是东京 | e4 东京旅游博客 `support` **0.5**（未知） | img1 `contradict` **0.4** |
| c3 地点是东京奥运会现场 | — | img1 `contradict` **0.4** |

支持总权重 1.5 > 图像矛盾总权重 1.2 → 裁判按看到的加权证据判"均衡" → 弃权。**它算得没错，是证据权重设错了**，两个根因：

1. **上传图像被标成"社交"档**（权重 0.4），比一条新华网（官方 1.0）还轻。但对"这张照片拍于 X"这类断言，**被核查的对象就是这张照片**，它理应是最高权重证据。修正：新增 `uploaded` 档（权重 1.0，与官方并列），前端徽章显示「用户图像」。
2. **检索只用断言原文当 query**：`"照片拍摄地点是东京奥运会现场"` → 检索回**真实的东京奥运会网页** → 关系判定器判成"支持" → 凭空造出支持边（就是 M4 消融里那条"检索注入虚假支持边"）。修正：检索 query 带上**图像 OCR/描述**，让检索去查"图里是什么"；只对 VLM 能与图像对照的断言生效（`consistency != uncertain`），文本类断言的检索行为完全不变。

修正后同一条输入：证据图 `n_support=0 / n_contradict=3`，判 **`refuted` conf 1.0**，证据链 `[img1]`，理由"证据显示照片为食物特写，无任何奥运相关背景或标识，与断言描述完全不符"。

**验证**（6 条多模态合成用例，cache off；文本种子 15 条回归）：

| 配置 | 准确率 | 弃权率 | macro-F1 | tokens |
|---|---|---|---|---|
| 检索 on（修正后） | 0.833 | 0.167 | 0.900 | 28,118 |
| 检索 off（对照组） | 0.833 | 0.167 | 0.900 | 16,214 |
| 文本种子 15 条（回归检查，无图） | 0.600 | 0.267 | 0.644 | 67,956（缓存回放） |

- **检索 on/off 口径一致了**：修正前开启检索会把 `mm01` 拖到弃权（0.667 口径），现在两种口径都是 0.833。
- **文本路径零回归**：15 条文本种子的准确率/macro-F1 与修正前**逐位相同**（0.600 / 0.644）——无图即无 `uploaded` 节点、无检索 hint，代码路径不变。

**代价（诚实口径）**：图像权重提到 1.0 后，**VLM 误判"不一致"会被放大**——图像断言的结论更依赖 VLM 取证质量，这是把"图像是这类断言的唯一判据"显式化的必然代价。合成集只有 6 条，0.833 即 5/6（未命中的 `mm06` 仍弃权），样本量小，结论用于**定性**（噪声消失、方向正确）而非宣称提升幅度。

## 参数兼容：模型不支持 thinking 时的自愈

取证阶段一度**永远拿不到图像证据**：无论主模型还是跨平台备胎，都被同一个 `HTTP 400 {"code":20015}` 挡回来，且报给前端的是"内容审核拒答"——排查方向被完全带偏。

**根因（实测，非猜测）**：调用层无条件下发 `extra_body={"thinking":{"type":"disabled"}}`，两家视觉模型都把它翻译成 `enable_thinking` 并直接拒收。

| 模型 | 带 `thinking` | 不带 `thinking` |
|---|---|---|
| `siliconflow:Qwen/Qwen3-VL-8B-Instruct` | ❌ 400 code=20015 `does not support parameter 'enable_thinking'` | ✅ 200，取证 JSON 正常返回 |
| `zhipu:glm-4.6v-flash` | ❌ 400 code=20015（同上） | ⚠️ 当时 429 配额受限，未能直连验证 |

三重伤害叠加：**主模型被打死** → 4xx 不切备胎，**跨平台备胎也用不上**（备胎同样带这个参数） → `except BadRequestError` 把所有 400 都写成"内容审核拒答"，**把参数问题伪装成了审核问题**。

**修复三层**：

1. **客户端按候选模型裁剪参数**：`LLM_NO_THINKING_MODELS`（默认含 3 个视觉模型）声明不支持 thinking 的 `provider:model`，调用前自动剔除 `thinking`/`enable_thinking`；文本模型照常下发，不受影响。
2. **运行时自愈**：遇到 `400/20015` 时自动剔除 thinking 原地重试一次，并把该模型记进本进程的黑名单（`llm_param_stripped_total` 计数器可观测），未登记的模型第一次踩坑即自愈。
3. **错误如实分类**：`forensics` 区分"内容审核拒答 / 参数不被支持 / 其它 400"三种，notes 各写各的，并打出 status + code + body 日志，不再静默吞掉 body。

**验证**：真实链路跑通——zhipu 主模型 429 → 切 SiliconFlow 备胎 → 200 取证 JSON（`overall=consistent`，图像描述正确），图像证据节点不再为空。

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

### 检查点续跑（M2，`ENABLE_CHECKPOINT` 门控）

`ENABLE_LANGGRAPH=true` 且 `ENABLE_CHECKPOINT=true` 时，`AsyncSqliteSaver` 按 `thread_id`（=trace_id）把每个超步的状态持久化到 `CHECKPOINT_PATH`（sqlite）。崩溃/中断后用**同一 thread_id** 重跑即续跑：已有检查点则传 `None` 恢复待执行节点，**已完成超步不重算**（省 token、省时间）。默认 `false` 零开销、不建文件。

- **serde**：state 携带 `nx.Graph` / dataclass / pydantic，msgpack 无法直接编码，故用 `JsonPlusSerializer(pickle_fallback=True)`；pickle 仅作用于**本地自有检查点文件**（可信 artifact），不接外部输入。
- **parity 不受影响**：检查点只加持久化，不改节点逻辑与输出契约；测试断言开/关检查点与 linear 三者输出一致（剔除 trace_id）。
- **续跑测试**：mock judge 首次抛异常模拟崩溃，同 thread_id 重跑后仅补 judge，decompose/prosecutor/defender 调用次数不增。
- 这是 HITL（M3 `interrupt` 人工介入）的前置：有了持久化状态，才能在低置信/弃权处暂停等待人工再恢复（已落地，见下）。

### HITL 人工介入（M3，`ENABLE_HITL` 门控）

> **前置条件：`ENABLE_HITL` 只在 LangGraph 路径生效，必须同时开 `ENABLE_LANGGRAPH=true`。** 直线 runner（`app/pipeline/runner.py`）里没有 `review` 节点，`pending_review` 永不产生，API 的 `review` 分支也就永不命中——只开 `ENABLE_HITL` 会表现为"配了但没反应"。`ENABLE_HITL` 隐含启用检查点 saver（`graph_runner.py`），暂停期间进程重启也能按 `thread_id=trace_id` 续跑。

`ENABLE_HITL=true`（隐含启用检查点）时，裁决子图在 `debate_end → finalize` 之间插入 `review` 节点：**弃权或置信度 < `JUDGE_CONFIDENCE_THRESHOLD`** 的裁决调用 `interrupt(payload)` 暂停整图，状态落检查点；API 把任务置为 `review` 态、SSE 推送 `review` 事件（保持连接），前端弹出复核面板展示拟裁决/置信度/理由/断言，人工可**维持原判**或**改判**（verdict + confidence + reason），`POST /api/verify/{id}/resume` 把裁决作为 `Command(resume=...)` 注入 `review` 节点续跑到 `finalize`。高置信裁决直接穿过 `review`，零打扰。

最小可用配置（三行）：

```bash
ENABLE_LANGGRAPH=true
ENABLE_HITL=true
JUDGE_CONFIDENCE_THRESHOLD=0.99   # 仅为演示：免费裁判过度自信(conf≈0.9)，默认 0.6 几乎不触发
```

- **为什么用 interrupt 而非轮询/外部队列**：暂停点就是图中的一个节点，恢复值即节点内 `interrupt()` 的返回值——人工裁决无缝汇入同一条状态流，无需把流水线拆成两半或在外部存中间态；检查点（M2）保证暂停期间进程重启也能恢复。
- **契约**：结果 dict 增加 `human_review: true` 标记；人工 override 只改 verdict/confidence/reason，证据链/图/trace 不变。`review` 态任务重复 resume 返回 409，未知任务 404。
- **一个真实坑**：本版本 LangGraph 把 `Command(resume={})`（空 dict）判为「中断ID→值」空映射（空键 `all()` 为真），不注入恢复值导致再次 interrupt；"维持原判"改用显式 `{"approve": true}` 标记规避（`run_graph_resume` 内归一化）。
- **测试**：interrupt 暂停 + 改判恢复（verdict/confidence/human_review 生效）、空裁决维持原判、HITL 关闭不暂停、高置信不暂停，共 4 项；API 404/409 各 1 项。
- **实测（2026-11-21）**：带图请求 → SSE 收到 `review` 事件、任务 `status=review`、`review.claims` 带 3 条断言；`POST /api/verify/{id}/resume` 改判 `refuted/0.9` → 任务续跑至 `done`，结果带 `human_review: true` 且 `evidence_chain` 保持不变；浏览器端复核面板正确渲染断言列表、提交后结果页出现「人工复核」徽章。
- **跑评测时必须关掉 `ENABLE_HITL`**：触发复核时 `run_pipeline` 返回的是 `pending_review` 字典（没有 `usage` 键），评测样本无法计 token。现在会显式抛 `RuntimeError("HITL 人工复核已触发…评测需关闭 ENABLE_HITL")` 并把该样本记为 failed，而不是抛 `KeyError` 让人一头雾水。
- **诚实口径**：免费裁判过度自信（见 A/B 结论），默认阈值 0.6 下真实流量很少触发复核；演示可临时调高 `JUDGE_CONFIDENCE_THRESHOLD`（如 0.99）让低置信路径可见。**注意该阈值同时是"升级强裁判"的阈值**——调到 0.99 会让几乎每次裁决都升级一次强裁判，token 接近翻倍，演示后记得调回 0.6。HITL 的价值在"模型表达不确定时把决定权交还人"，而非刷准确率。

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
- **真正瓶颈是校准，不是编排**：循环机制本身正确（离线 mock 低置信测试已证收敛 / 硬上限 / 升级），但其置信度触发器在过度自信的免费裁判下很少激活。多轮辩论的准确率增益需在裁判会表达不确定的数据 / 模型上才显现——置信度校准已在[校准小节](#置信度校准与评测-dashboard)落地（温度缩放 + LOO-CV），更强裁判仍待验证。

> 成本：多轮 = 控/辩/裁各多调用 N−1 次，token 随轮数近似线性增长；本次因裁判过度自信未触发，r3 与 r1 同成本。`DEBATE_MAX_ROUNDS=1` 与直线 runner 同成本，是"零代价拿到 LangGraph 编排能力"的默认档。检查点续跑（M2）与 HITL 人工介入（M3）均已落地（见上）。

## 置信度校准与评测 Dashboard

免费裁判（glm-4.7-flash）**系统性过度自信**：A/B 中判对样本平均 conf≈0.93、判错也≈0.90，判别力弱，导致低置信循环/升级触发器几乎不激活。编排不是瓶颈，**校准才是**。D1 用事后温度缩放直接处理上报置信度。

**方法（`app/eval/calibration.py`，纯 Python，无 numpy/scipy/sklearn）**

- 把裁判的单标量 confidence 视为二分类 `P(correct)`，温度缩放 `p_cal = sigmoid(logit(p_raw)/T)`；`T>1` 软化过度自信。
- 网格搜索 `T` 最小化 NLL；仅在 **attempted（非弃权）且 confidence 非空** 的样本上拟合与评估。
- 指标 **ECE / MCE / Brier** + 10 箱可靠性分箱；另报 **LOO-CV**（留一拟合再校准）给出诚实的小样本 out-of-fold 估计，避免"同批数据既拟合又评估"的乐观偏差。
- **事后、不重跑推理**：`scripts.calibrate` 直接读已采集的 eval `.jsonl` 拟合，省额度。

```bash
# 1) 采集原始置信度（校准 off，文本避开 VLM 冷却；cache readwrite 复用已录响应）
ENABLE_CALIBRATION=false ENABLE_RETRIEVAL=true LLM_CACHE_MODE=readwrite \
  .venv/Scripts/python -m scripts.evaluate --data data/eval_seed.jsonl --system pipeline --out out/calib_collect
# 2) 拟合温度，落盘 .data/calibrator.json + .data/reliability.json
.venv/Scripts/python -m scripts.calibrate --records "out/calib_collect/eval_pipeline_*.jsonl"
# 3) 启用流水线校准（只变换上报置信度；路由/升级仍用原始 conf）
ENABLE_CALIBRATION=true .venv/Scripts/python -m app
```

**接入语义（保 parity）**：校准在两条 runner 共用的 `runner._build_result` 里、adjudicate（升级已用原始 conf 决策）**之后**应用，结果同时带 `confidence`（校准后）与 `confidence_raw`（原始）。因此 `ENABLE_CALIBRATION` 开/关不改变编排路由与调用顺序，linear/graph parity 不受影响；artifact 缺失时 `T=1.0` 静默不校准，流水线不中断。

**Dashboard（`/dashboard.html`，纯 SVG 无 CDN）**：`GET /api/evals` 扫描 `out/**/eval_*.json` 汇总、`GET /api/calibration` 读 artifact，前端渲染①校准摘要卡（T、ECE/Brier 原始→校准→LOO）②**可靠性曲线**（校准前/后 vs 对角线 y=x，点大小∝样本数）③置信度分布直方图（原始 vs 校准后）④消融对比表 + accuracy/macro-F1 条形图。只读本地 artifact，不接受用户路径输入。

### 校准结果（诚实口径）

| 数据集 | attempted | 拟合 T | ECE 原始→校准 | ECE LOO | Brier 原始→校准 | Brier LOO |
|---|---|---|---|---|---|---|
| 硬子集（A/B linear 6 条，判对率 0.6） | 5 | **5.62** | 0.320→0.188 | 0.410 | 0.329→0.212 | 0.280 |
| 全 seed（calib_collect 12 完成/3 失败，判对率 0.9） | 10 | **0.85** | 0.095→0.092 | 0.087 | 0.084→0.085 | 0.081 |

- **过度自信是子集/模型依赖的，不是全局常数**：硬子集（多 misleading 难例）裁判 conf≈0.9 但判对率仅 0.6 → 严重过度自信，T=5.6 大幅软化；全 seed 以易例为主、判对率 0.9≈平均 conf 0.88 → 已接近校准，T=0.85 反而轻微**抬升**（略欠自信）。
- **LOO-CV 是小样本的诚实阀**：硬子集 n=5 时 in-sample ECE 减半但 LOO 反升（0.410），说明温度拟合方差大、改善不可信；全 seed n=10 时 LOO（0.087）优于原始（0.095），改善才站得住。Dashboard 对 n<20 显式告警。
- **结论**：D1 的价值不是"免费涨点"，而是把"裁判到底可不可信"变成**可测量、可校正、可回退**的一等能力——ECE/Brier/LOO + 温度缩放 + 可靠性曲线，在过度自信时校正、在已校准时如实报告"无需校正"。这与"校准弃权比刷高准确率更重要"的取向一致。
- **已知失效**：`.data/calibrator.json`（T=0.85）是在**链上重排前的旧裁判**上拟合的，换裁判后温度已失配（每个裁判的自信曲线不同）。当前 `ENABLE_CALIBRATION` 默认 false，故该 artifact 处于**未启用**状态、不影响线上输出；换过裁判后应重跑 `scripts.calibrate` 重新拟合，README 里不要把旧 T 当作当前裁判的校准结论。

## 测试

```bash
.venv/Scripts/python -m pytest        # 155 项，覆盖客户端/缓存/重试与熔断/兜底链/参数兼容自愈/模型体检脚本/各流水线阶段/取证/路由/评测/API/可观测性/LangGraph 编排/循环辩论/检查点续跑/HITL 人工介入/置信度校准
```

## 项目结构

```
app/
  config.py            # pydantic-settings 配置（角色模型、开关、缓存）
  api.py / main.py     # FastAPI 路由 + SSE + 静态托管 + 根 /metrics
  obs.py               # 结构化 JSON 日志 + Prometheus 指标埋点
  llm/                 # OpenAI 兼容客户端、退避重试、录制回放缓存
  pipeline/            # decompose / forensics / retrieve / evidence_graph / graph / adjudicate / runner / graph_runner / models
  eval/                # 数据集、harness（断点续跑）、指标、基线/流水线 predictor、calibration（温度缩放/ECE/LOO）
scripts/               # evaluate / calibrate / probe_models（模型配额体检）/ make_demo_images / prepare_fakeddit / smoke_llm
static/                # 单页前端 + dashboard.html（纯 SVG）+ vendored cytoscape.min.js
data/                  # 评测数据集与合成图像
```

每次核查产出 JSONL trace 与证据图 JSON（`.data/traces/`），便于复盘与调试。

## 成本与限制

- **成本**：默认全免费、预算 ≈ ¥0；升级付费裁判单条约 ¥0.003–0.005（付费模型放兜底链末位或 `MODEL_JUDGE_STRONG` 即可，见[模型编排](#模型编排先免费后付费链上顺序--优先级)）。免费档有账户级并发限制，客户端带并发上限 + 按角色重试预算 + 退避封顶 + 按模型熔断 + 跨平台兜底候选链；评测支持断点续跑。
- **限制**：多模态评测基于小规模合成用例；单图输入；检索依赖 Tavily 免费档质量；**图像断言更依赖 VLM 取证质量**（图像证据已提权到与官方并列，VLM 误判会被放大）；裁决质量受免费模型能力约束；免费档 429 是常态，靠兜底链与链上重排缓解而非根治（跨平台兜底需自备 SiliconFlow key）。
- **v2 路线**：反向搜图 / 图片挪用检测、视频与多图、Neo4j + 可信度传播 / 多跳推理、Redis/Celery、OTel/Langfuse 分布式追踪（在现有 Prometheus+JSON 之上）、本地 vLLM、React 前端。
