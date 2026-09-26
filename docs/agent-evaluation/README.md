# ModelEval：数学建模 Agent 评测平台（工程作品方案 v2）

> 日期：2026-09-25。目标是一个**工程作品**，不做算法自进化。通用评测框架调研见 [survey-and-optimization-idea.md](./survey-and-optimization-idea.md)。
> 下文对 MathModelAgent 和 MM-Agent 的描述来自 2026-09 两个仓库的实际代码（浅克隆阅读），其余项目来自公开网页检索。

---

## 1. 被测对象：开源数学建模 agent

| 项目 | 形态 | 输出 | 技术栈（读代码确认） |
|---|---|---|---|
| [**MathModelAgent**](https://github.com/jihe520/MathModelAgent)（jihe520，社区热度高，2026-09 仍在更新） | **两代并存**：① 旧版 multi-agent 工作流（Coordinator → Modeler → Coder → Writer）；② 新版**纯 SKILLS**，跑在 Claude Code / Codex 上（`/1start-mathmodel`），作者已声明"不再做 harness 层" | Typst/LaTeX 排版的整篇竞赛论文（PDF）+ notebook 代码 + `reports/VERIFY_REPORT.md` | 旧版：FastAPI + WebSocket + Redis，LiteLLM 接任意模型，本地 Jupyter 或 E2B/Daytona 云端代码解释器，ChromaDB+Rerank RAG，Tavily 搜索；新版：6 个阶段 skill（分析建模 → 编码可视化 → draw.io → 写作 → 验收），17 套赛事 Typst 模板，9 步自动验收脚本 |
| [**MM-Agent**](https://github.com/usail-hkust/LLM-MM-Agent)（HKUST，NeurIPS 2025） | 固定 4 阶段流水线：问题分析 → 建模 → 求解 → 报告；HMML 分层建模知识库 + actor-critic 选方法 | 结构化 JSON 解 + 报告 | 纯 Python；自带 **MM-Bench**：111 道 MCM/ICM（2000–2025）题目 JSON + 22 套数据集 |
| ModelingAgent（UIUC，EMNLP 2025） | 多 agent 框架；配套 **ModelingBench**：68 道 COMAP 系列题（MCM/ICM、HiMCM、IM2C） | 报告 | 代码是否完整开源**待核** |
| 裸 harness 基线 | Claude Code / Codex **不装任何建模 skill**，只给题目 | 自由形式 | 用来回答"skill 本身值多少分" |

**关键发现（这是你的切入点）：**

1. MathModelAgent 作者在 README 里明确写着：*"Harness SKILL 的优化需要大量黑盒测试和调优"*，欢迎贡献者*"在不同的 Harness 上测试不同的 LLM，提供反馈和案例"*。**上游已经公开需要一个评测系统，没人做。**
2. MM-Bench 现有打分脚本（`MMBench/evaluation/`，约 780 行）的局限：
   - 单个 GPT-4o 评委，`temperature=0.7`，单次采样，分数由正则 `<score>(\d+)</score>` 抽取 → **同一份解两次打分可能不同**；
   - 只吃 MM-Agent 自己的 JSON 格式 → **评不了输出 PDF 论文的 MathModelAgent**；
   - **不执行代码**：论文里写的数字是否真由代码算出，完全不校验；
   - 没有评委与人工打分的一致性校准。
3. MCM/ICM 历年题目和获奖论文都在网上，**数据污染**是真实风险：需要用模型知识截止之后的新题做留出集。

---

## 2. 业务背景

> **场景：数模竞赛辅导 / 在线建模服务平台的"模型与 skill 质量门禁"**
>
> 一家做数学建模教学与竞赛辅导的平台（形态类似 MathModelAgent 作者运营的在线版），把建模 agent 作为付费功能提供给学生。每逢国赛、美赛前，平台面对三个工程问题：
> 1. **选型与成本**：底座用哪个模型、跑在 Claude Code 还是 Codex 上？每篇论文成本 $1 还是 $15，质量差多少？
> 2. **回归测试**：每次改 skill/prompt/模板，质量是升了还是降了？现在全靠人工看几篇 PDF。
> 3. **可信度**：论文里的数字是不是代码真算出来的？有没有编造数据、答漏小问、引用不存在的文献？学生直接提交会出事。
>
> ModelEval 为此提供：**统一的运行器 + 可复现的分层打分 + 排行榜 + CI 回归门禁**。

这个场景的好处：
- 有真实上游用户（MathModelAgent 社区），作品可以直接提 PR / issue 反馈，简历上是"给 X star 项目搭了评测体系"；
- 覆盖了 agent eval 工程里最硬的几块：**异构 agent 适配、沙箱执行、LLM-as-judge 校准、成本核算、统计显著性、CI 集成**；
- 与你的背景仍有连接（统计严谨性、效度审计、运筹建模），但不依赖算法自进化。

---

## 3. 评测设计：三层打分

```
题目 (MM-Bench / ModelingBench / 留出新题)
   │
   ▼
Adapter ──► 各 agent 在容器中运行 ──► 产物 (PDF/Typst/LaTeX/JSON, 代码, 数据, 图)
   │                                           │
   │  所有 LLM 调用经 LiteLLM proxy ────────────┤──► 统一 token/成本/延迟 (OTel → Langfuse)
   ▼                                           ▼
                       Normalizer → Solution Bundle（章节、小问答案、数值结论、代码、图表）
                                               │
          ┌────────────────────────────────────┼─────────────────────────────┐
          ▼                                    ▼                             ▼
   L1 硬检查（程序化，确定性）          L2 评委打分（LLM panel）        L3 过程与成本
```

### L1 硬检查（确定性，最有说服力，优先做）
| 检查 | 做法 |
|---|---|
| 完成度 | 是否产出论文；每个小问是否有对应章节与结论（题目要求 → 论文章节的覆盖率） |
| 可编译 | Typst/LaTeX 能否编译出 PDF |
| **可复现** | 在干净沙箱里重跑 agent 交付的代码，检查能否运行成功 |
| **数值一致性** | 从论文抽取关键数值（LLM 抽取 + 正则），与重跑代码的输出比对（相对误差阈值）→ 编造/不一致率 |
| 数据使用 | 题目给了数据集时，代码是否真的读了它 |
| 引用真实性 | 参考文献能否在 Crossref / Semantic Scholar 检索到 |
| 格式违规 | 占位符、内部路径泄露、图表引用断链（可复用 MathModelAgent `6verity` 脚本的检查项） |

### L2 评委打分（主观质量）
- 维度沿用 MM-Bench 的 4 项（问题分析、建模严谨性、实用与科学性、结果与偏差分析），并补充 ModelingBench 维度中可落地的"创造性""真实性"；每个维度写**带锚点的 rubric**（1/3/5/7/10 分各给示例）。
- **评委团**：2–3 个不同厂商的模型，`temperature=0`，每份解打 3 次取中位数，报告评委间一致性。
- **成对比较**：与 O 奖 / Finalist 人类论文做 pairwise，胜率 → Bradley-Terry 排名，比绝对分更稳。
- **评委校准**（详见 [human-annotation.md](./human-annotation.md)）：你和 2–3 位有数模经验的同学人工标注约 36 篇，计算与评委的 Spearman / Krippendorff α。**这一步决定平台可信度，也是作品里最能体现方法论的部分。**
- 评委位置偏差（A/B 顺序互换）、长度偏差（字数与分数的相关）要做检测。

### L3 过程与成本
Token、美元成本、墙钟时间、工具调用数、代码报错与重试次数、是否人工介入。输出**质量-成本 Pareto 图**。

### 统计与防污染
- 每个配置至少 3 个种子；排行榜全部带 bootstrap 95% CI；两配置差异做配对检验。
- **留出集**：选模型知识截止之后的赛题（例如 2026 年的 MCM/ICM 和国赛题），对比"旧题 vs 新题"的分差，量化污染。赛题版权属于主办方，仓库只存题目链接与下载脚本，不直接分发原文。

---

## 4. 技术选型

| 层 | 选型 | 理由 |
|---|---|---|
| 评测框架 | **Inspect AI**（Task = Dataset + Solver + Scorer） | 原生支持运行 Claude Code / Codex 等外部 agent、Docker 沙箱、自带日志查看器；scorer 可以直接写成上面的 L1/L2 |
| 备选 | Harbor 任务格式 | 如果想把任务发布到 Terminal-Bench 生态；两者可以共存 |
| 沙箱 | Docker（镜像内预装 Python 科学栈、Typst、TeX Live 精简版） | 复现性检查必须在干净环境 |
| LLM 网关 | **LiteLLM proxy** | 所有被测 agent（含 MathModelAgent 旧版本身就用 LiteLLM）都指向同一个网关 → 成本/token 统一记账、限速、切换模型不改 agent 代码 |
| 追踪 | OpenTelemetry GenAI 约定 → 自托管 Langfuse | 轨迹、成本、评委打分挂在同一条 trace 上 |
| PDF/文档解析 | Typst/LaTeX 源码优先；只有 PDF 时用 PyMuPDF / marker | 数值抽取与章节切分 |
| 存储 | 每次运行一个目录（产物 + `result.json`），汇总到 DuckDB / Parquet | 易 diff、易复现 |
| 可视化 | Streamlit 或静态 leaderboard 页面 | 排行榜、Pareto 图、单篇论文逐项检查报告 |
| CI | GitHub Actions：对 skill 仓库的 PR 跑 5 题"冒烟集"，L1 必须全过，L2 不得显著下降 | 对应"回归测试"业务需求 |

---

## 5. 仓库结构草图

```
modeleval/
  adapters/            # mathmodelagent_skills.py, mathmodelagent_legacy.py, mm_agent.py, bare_harness.py
  tasks/               # 题目加载：mmbench.py, modelingbench.py, holdout.py
  normalize/           # PDF/Typst/JSON → SolutionBundle
  scorers/
    l1_checks/         # compile.py, reproduce.py, numeric_consistency.py, coverage.py, citations.py
    l2_judge/          # rubric.yaml, panel.py, pairwise.py, calibration.py
    l3_cost.py
  analysis/            # bootstrap CI, Bradley-Terry, Pareto
  dashboard/
  docker/
  human_labels/        # 人工校准标注（匿名化）
```

---

## 6. 里程碑（约 6–7 周，可随时停在一个能展示的版本）

| 周 | 交付 | 可展示点 |
|---|---|---|
| 1 | 跑通 1 题 × 2 agent（MathModelAgent skills on Claude Code、MM-Agent），全部经 LiteLLM 网关 | 第一张成本对比表 |
| 2 | Normalizer + L1 全部硬检查 | "X% 的论文数值与代码不一致"——**第一个有传播力的发现** |
| 3 | 扩到 15–20 题子集 × 4 配置 × 3 种子；裸 harness 基线 | Skill 消融：skill 带来多少提升 |
| 4 | L2 评委团 + 人工校准 30 篇 | 评委-人工一致性报告；对比 MM-Bench 原脚本的打分方差 |
| 5 | 排行榜 + Pareto 图 + 单篇报告 dashboard | 可演示的网页 |
| 6 | 留出新题污染分析；CI 冒烟门禁 | 给 MathModelAgent 提 issue/PR 附报告 |
| 7 | 文档、README、技术博客 | 作品集收尾 |

### 首个实验矩阵（建议）
| 配置 | agent | harness | 模型 |
|---|---|---|---|
| A | MathModelAgent skills | Claude Code | 模型 1 |
| B | MathModelAgent skills | Codex | 模型 2 |
| C | 裸 harness（无 skill） | Claude Code | 模型 1 |
| D | MM-Agent | 自带流水线 | 模型 1（经 LiteLLM） |

A vs C 得出 skill 的价值，A vs B 得出 harness 与模型的差异，A vs D 比较 skill 驱动与固定流水线两种架构。

---

## 7. 需要你确认
1. API 预算：一题一配置一次大约 $1–15（MM-Agent 论文报告 GPT-4o 下约 $0.88/题；Claude Code 全流程写论文会贵得多），4 配置 × 20 题 × 3 种子 = 240 次运行，要按预算缩放。
2. 能否找到 2–3 位有数模竞赛经验的同学做人工校准标注。
3. 是否要中文国赛题（MathModelAgent 主打国赛模板）还是先只做美赛英文题（MM-Bench 现成）。建议先英文，第二阶段再加国赛。

## 参考
- MathModelAgent: https://github.com/jihe520/MathModelAgent
- MM-Agent / MM-Bench: https://github.com/usail-hkust/LLM-MM-Agent ，论文 https://huggingface.co/papers/2505.14148
- ModelingAgent / ModelingBench: https://arxiv.org/abs/2505.15068 ，https://www.alphaxiv.org/benchmarks/university-of-illinois-at-urbana-champaign/modelingbench
- Inspect AI: https://inspect.aisi.org.uk/
- 通用框架调研（AgentCompass、Harbor、τ-bench 等）: [survey-and-optimization-idea.md](./survey-and-optimization-idea.md)
