# Agent Evaluation 通用调研 + 优化方向备选方案（v1）

> 日期：2026-09-25。调研基于公开网页检索；arXiv 在当前环境被网络策略屏蔽，论文细节来自检索摘要与项目主页，标注"待核"的条目需要读原文确认。

> 已被 [README.md](./README.md) 的数学建模 agent 评测方案取代为主线；本文件保留通用框架调研（第 1、2 节）和优化方向备选。

---

## 1. 开源项目速览

### 1.1 通用评测基础设施（harness / runner）

| 项目 | 定位 | 技术选型要点 |
|---|---|---|
| [AgentCompass](https://github.com/open-compass/agentcompass)（OpenCompass，EMNLP 2026 Demo） | 统一 agent 评测基础设施 | 把 **Model / Benchmark / Harness / Environment 四者解耦**；Python + CLI 入口，async 并发；local / Docker / 远程 sandbox 三种执行后端；增量持久化、失败重试、断点续评；记录轨迹、tool call、token、延迟，带可插拔 analyzer 做失败归因；内置 20+ benchmark、10+ harness（Claude Code、Codex、OpenHands、OpenClaw 等） |
| [Harbor](https://github.com/harbor-framework)（Terminal-Bench 团队，2026-01） | 容器内 agent 评测 + RL rollout 生成 | **任务 = 容器镜像 + 指令 + 测试脚本** 的标准任务格式；registry 分发数据集（Terminal-Bench 2.0/2.1、terminal-bench-science 都走它）；对接多家云 sandbox，常用 32–100 容器并行；同一套任务既做 eval 又做 RL 环境 |
| [Inspect AI](https://inspect.aisi.org.uk/) + [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals)（UK AISI） | 前沿安全评测事实标准 | **Task = Dataset + Solver + Scorer** 三段式；内置 ReAct/多 agent 原语，也能直接跑 Claude Code / Codex CLI / Gemini CLI；`SandboxEnvironment` 抽象支持 Docker、K8s、Modal、Proxmox 等；配套 [aisi-sandboxing](https://github.com/UKGovernmentBEIS/aisi-sandboxing) 与日志查看器 |
| Holistic Agent Leaderboard (HAL) | 跨 benchmark 标准化排行 | 统一 scaffold × model × benchmark 网格；主要问题是成本——9 个 benchmark 约 4 万美元，且每个配置只跑 1 次（引自 *Efficient Benchmarking of AI Agents*, 2026） |

### 1.2 代表性 benchmark

| Benchmark | 评什么 | 对你有用的设计 |
|---|---|---|
| Terminal-Bench 2.0 / 2.1 | 容器内长程命令行任务（2.1 共 89 题） | Harbor 任务格式；结果对 harness 极其敏感 |
| τ-bench / τ²-bench | 多轮客服 agent，LLM 模拟用户 + 有类型的 API + 数据库状态 | **pass^k**（k 次独立 rollout 全部成功）作为可靠性指标；同时校验最终 DB 状态和是否遵守 policy |
| [ALE-Bench](https://sakana.ai/ale-bench/)（Sakana × AtCoder） | 40 道 AtCoder 启发式赛题，长时程、按分数排名的算法工程 | **没有 pass/fail，只有分数**；带 code sandbox，允许在时间预算内反复提交拿 public 反馈；与人类排名对齐 |
| [CO-Bench](https://github.com/sunnweiwei/CO-Bench)（AAAI 2026）/ [FrontierCO](https://huggingface.co/datasets/CO-Bench/FrontierCO)（ICLR 2026） | agent 为组合优化设计算法 | 8 类 CO 问题、真实大规模实例（TSP 至千万节点）；**LLM4AD 已内置 `llm4ad/task/optimization/co_bench`** |
| HeuriGym | agent 写 CO 启发式 | 反复执行-反馈的 agentic 协议 |
| Opti-Agent-Bench / ORAgentBench / FrontierOR（2026，待核） | 端到端运筹 R&D agent：建模 → 求解 → 算法设计 | 来自真实业务问题，考察完整研发闭环而非单次建模 |
| Harness-Bench（2026，待核） | 专门测 harness 效应 | 同一模型换 harness 分数可差 10–20 个百分点（例：LangChain 仅改 harness，Terminal-Bench 从 52.8% 升到 66.5%；CORE-Bench 同模型 42% vs 78%） |

### 1.3 可观测性与打分层

| 工具 | 角色 |
|---|---|
| OpenTelemetry GenAI 语义约定（2026 初稳定） | `gen_ai.*` 属性统一记录 prompt、模型、token、tool/agent 调用——**先对这个 schema 埋点，后端随时可换** |
| [Langfuse](https://langfuse.com)（MIT，可自托管） | trace + 数据集 + 打分面板 |
| Arize Phoenix（OpenInference） | trace/session 级 trajectory 评估、漂移检测 |
| DeepEval / Ragas | 指标引擎（LLM-as-judge 等），分数回写到 trace |

### 1.4 2026 研究热点（决定你能在哪儿做出贡献）

1. **Harness 是隐藏变量**：*The Scaffold Effect in Coding Agents*、Harness-Bench、StateM（Terminal-Bench 2.1 靠 harness scaling 到 95.3%）——把"模型分数"和"harness 分数"拆开成了共识需求。
2. **评测成本与统计可靠性**：*Efficient Benchmarking of AI Agents*、*Beyond Outcomes: Dual-View Relational Learning for Efficient Agent Benchmarking*——用少量任务子集预测全量排名。
3. **自动优化 harness**：Task-CoEvolve（验证任务选择与 harness 协同进化）、CHILL-Harness、HarnessBridge——**用进化/搜索改 agent 本身**，与你的 LLM4AD 背景直接同构。
4. **超越最终分数**：*Beyond Final Scores*（长程 AI R&D agent 的过程性评估）、AgentCompass 的 trajectory analyzer——关注过程而非只看结果。

---

## 2. 技术选型归纳

| 层 | 主流做法 | 推荐 |
|---|---|---|
| 任务格式 | Harbor 任务目录 / Inspect Task / 自定义 YAML | **Harbor 格式**：与 Terminal-Bench 生态兼容，天然容器化，一套任务同时可做 RL 环境 |
| 被测 agent | Claude Code、Codex CLI、OpenHands、mini-swe-agent、自研 ReAct | 至少 3 个 CLI agent + 1 个极简 ReAct 基线 + **LLM4AD 的 EoH/FunSearch 作为"非 agent"进化基线** |
| 执行隔离 | Docker（本地）→ Modal/E2B/K8s（扩容） | 本地 Docker 起步；LLM4AD 现有 evaluator 的超时/沙箱逻辑可复用为容器内打分器 |
| 调度 | asyncio 并发 + 断点续跑 + 失败重试 | 抄 AgentCompass 的设计：每个 (task, agent, model, seed) 一行持久化，幂等可续 |
| 追踪 | OTel GenAI 语义约定 | OTel 埋点 → 自托管 Langfuse |
| 打分 | 单测 pass/fail；分数型；LLM-as-judge | 以**客观分数**为主（gap to BKS），LLM-judge 仅用于过程标注 |
| 统计 | 单次运行、均值 | pass^k、多 seed bootstrap CI、混合效应模型拆 model × harness × task 方差 |

---

## 3. 结合你学术背景的业务场景

### 你的背景画像（从本仓库推断）
- LLM 驱动的自动算法设计（EoH / FunSearch 类进化框架）；
- VRP/LNS 的 destroy–repair 算子协同进化，CVRP 相关工作；
- **LLM 变异坍缩**（生成算子趋同）与多样性度量的**效度审计**（L2 probe 输出距离通过 held-out 检验）；
- 信用分配；
- 统计上较严谨：噪声偏差修正、置换检验、功效分析、预注册决策规则、构念效度。

这些正好对应 agent evaluation 里最缺的三件事：**分数型（非 pass/fail）任务、过程行为度量、统计可信度**。

### 推荐业务背景：物流调度"优化工程 agent"的采购与上线评测

> **场景**：一家城配/即时物流公司每天面对不断变化的 VRP 变体（时间窗、多车型、临时封路、订单波峰）。算法团队人手有限，准备让 coding agent（Claude Code / Codex / OpenHands 等）承担"根据新约束改写和调优路由启发式"的工作。管理层要回答：
> 1. **选哪个 agent + 模型 + harness 组合？** 花多少钱能换来多少 gap 改善？
> 2. **agent 交付的启发式可信吗？** 会不会只在给定实例上过拟合、换了规模/分布就崩？
> 3. **agent 是在真正搜索，还是在反复提交换汤不换药的代码？** 何时应该停止、换策略或交给人？

项目名暂定 **OptAgentEval**：面向"优化算法研发 agent"的评测平台与方法学。

#### 为什么这个场景和你匹配
| 业务问题 | 对应的你已有的能力 | 可产出的研究贡献 |
|---|---|---|
| 选型：model × harness × 预算 | 统计复核、功效分析 | **Harness 效应分解**：混合效应模型给出 model / harness / task 的方差占比与置信区间；成本-性能 Pareto 前沿 |
| 交付物可信度 | 构念效度、held-out 审计 | **泛化审计协议**：public 实例开发 / private 实例打分，并加规模外推（100 → 1k → 10k 节点）与分布偏移（聚类 vs 均匀客户点） |
| agent 在搜索还是在打转 | 变异坍缩、L2 probe 距离度量 | **探索坍缩度量**：把 agent 每次提交的启发式看作一个"个体"，用你已验证过的行为距离度量 trajectory 内多样性，检测坍缩并预测最终分数 |
| 哪一步带来提升 | 信用分配 | **轨迹级信用分配**：把分数提升归因到具体动作（读文献、改 destroy、调参、加 local search） |
| 评测太贵 | 自适应选择（UCB 等） | **自适应任务/实例子集选择**：少量实例预测全量排名（对标 Efficient Benchmarking、Task-CoEvolve） |

#### 关键设计：把 LLM4AD 当作"对照组"
同一个任务，同一份 LLM 调用预算，分别交给：
- (A) 通用 coding agent（开放式、自主工具调用）；
- (B) LLM4AD 的进化方法（EoH / FunSearch / 你的 HypoEvo）。

"通用 agent 与专用进化搜索在算法设计上孰优孰劣、差在哪里"是当前没人系统回答过的问题，也是你最有资格回答的问题。

### 备选场景（若想离开物流）
- **芯片 EDA 布局 / 调度启发式 agent**：同样是分数型优化，工业味更浓，但数据获取难。
- **供应链计划 agent（建模 + 求解）**：对标 Opti-Agent-Bench / ORAgentBench，偏 MIP 建模而非启发式设计，与你的 LNS 背景距离稍远。

---

## 4. MVP 路线（约 8 周）

| 周 | 目标 | 产物 |
|---|---|---|
| 1–2 | 任务层：把 LLM4AD 的 CVRP/VRPTW/OVRP + co_bench 子集封装为 Harbor 格式；public/private/scale-shift 三套实例；打分 = gap to BKS | `tasks/` 10–15 个容器化任务 |
| 3 | Runner：并发调度、断点续跑、OTel 埋点 → Langfuse；每次提交的代码快照入库 | 可复现的 runner |
| 4–5 | 基线网格：3 agent × 2 模型 × 5 seed + LLM4AD EoH 基线，统一 token/时间预算 | 第一版结果表 + 成本曲线 |
| 6 | 过程度量：anytime 曲线、提交多样性（L2 probe 距离）、坍缩检测、泛化 gap | 行为分析报告 |
| 7 | 统计：pass^k 的分数版（best-of-k / worst-of-k）、bootstrap CI、混合效应方差分解；**预注册**主要假设 | 预注册文档 + 统计脚本 |
| 8 | 评测降本：自适应实例子集预测全量排名 | 论文初稿骨架 |

### 预注册的候选主假设
- H1：同一模型下，harness 引起的 gap 方差 ≥ 模型引起的方差。
- H2：agent 轨迹前 30% 的提交多样性（L2 距离）显著预测最终 gap（控制模型与任务）。
- H3：通用 coding agent 在 public 实例上不劣于 EoH，但 private / 规模外推的泛化 gap 更大。

---

## 5. 待确认事项
1. 算力与 API 预算（决定网格大小与 seed 数）。
2. 是否需要闭源 agent（Claude Code / Codex）还是只用开源 agent + 开源模型（Qwen 等）。
3. 目标产出：论文（投 NeurIPS D&B / ICLR）还是工程作品集，两者侧重不同。

## 参考链接
- AgentCompass: https://github.com/open-compass/agentcompass ，论文页 https://huggingface.co/papers/2607.13705
- Harbor: https://github.com/harbor-framework ，Terminal-Bench 2: https://github.com/harbor-framework/terminal-bench-2
- Inspect: https://inspect.aisi.org.uk/ ，inspect_evals: https://github.com/UKGovernmentBEIS/inspect_evals
- ALE-Bench: https://sakana.ai/ale-bench/
- CO-Bench: https://github.com/sunnweiwei/CO-Bench ，FrontierCO: https://huggingface.co/datasets/CO-Bench/FrontierCO
- τ-bench 方法学: https://benchmarkingagents.com/tau-bench/
- Efficient Benchmarking of AI Agents: https://arxiv.org/html/2603.23749v1
- Scaffold Effect: https://arxiv.org/pdf/2607.22585 ；Harness-Bench: https://arxiv.org/html/2605.27922v1
- Task-CoEvolve: https://arxiv.org/pdf/2608.20169
- Opti-Agent-Bench: https://arxiv.org/html/2607.10768 ；ORAgentBench: https://arxiv.org/html/2606.19787
- LLM 可观测性对比（OTel GenAI）: https://signoz.io/comparisons/llm-observability-tools/
