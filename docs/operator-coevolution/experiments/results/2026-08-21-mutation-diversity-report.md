# 第一轮主线探索综合报告：LLM 变异坍缩与多样性度量

> 日期：2026-08-21  
> 模型：Qwen 3.5 Flash  
> 研究主线：LLM repair 算子的变异退化为相似  
> 配套实验：[变异链报告](./mutation-collapse/pilot/REPORT.md)｜[度量审计报告](./diversity-metrics/pilot/REPORT.md)

## 1. 执行摘要

今晚选择“**LLM 变异退化为相似**”作为主线，并先解决它的测量前提：什么距离能可靠地区分代码表面变化、实际 repair 行为变化和有用性变化。

本轮得到三个层次不同的结论：

1. **L2 probe 输出距离通过了 pilot 级效度门禁。** 重测信度 0.981；对独立 validation 输出行为的相关为 0.958，对 held-out 性能足迹的相关为 0.861。它可以作为下一轮变异坍缩实验的核心测量工具。
2. **代码 token/AST 不可作为行为多样性。** 它们与两个 held-out 目标均为负相关；`sequential_insertion` 与语义等价重写的 AST 距离为 0.772，但 held-out 行为距离为 0。
3. **本轮尚不能判定 Qwen 变异是否造成跨链群体坍缩。** 19/32 个子代合法，失败率 40.6%；链内后半程重访率较高，但群体 validation 有效自由度从 1.68 上升到 1.78，没有出现预注册要求的下降。由于起始有效自由度不足 2，本轮按规则判为证据不足。

因此，主线不切换，但研究顺序要调整为：

> **先用 L2 行为门禁构造真正多样的种子和合法变异接口，再比较普通选择与 L2 引导选择是否阻止坍缩。**

不能把本轮写成“已经证明 Qwen 会坍缩”，也不能写成“Qwen 不会坍缩”。可以写的是：**我们已经找到一把可信的行为尺子，并发现当前 baseline 同时受起点退化和接口无效率污染。**

---

## 2. 实验 A：多样性度量综合审计

### 2.1 设计

样本包含 13 个 repair 算子：

- Stage 0.5 的 6 个经典 repair；
- 1 个与 sequential 行为等价、代码结构不同的阴性对照；
- block-reinsert 与 new-route 两个手工“专才/极端”对照；
- 4 条 Qwen 变异链最后一个合法候选。

probe 共 80 个：CVRP/VRPTW 各 4 个实例 × 2 个状态 × 5 种 destroy 残解来源。前半实例只用于 calibration，后半实例只用于 validation；每个算子另用第二套 repair RNG 重复测量。

比较的度量：

- L0 token 3-gram Jaccard；
- L0 AST node-type 3-gram Jaccard；
- Qwen 机制描述 + TF-IDF（L1 proxy）；
- Qwen 三次随机顺序的直接成对机制判断（L1+）；
- calibration probe 输出边集合 Jaccard（L2）；
- calibration 收益秩足迹（L3 pilot）。

validation 输出行为与 validation 性能足迹只作为独立目标，不参与构造上述候选距离。

### 2.2 主结果

| 度量 | 重测信度 | held-out L2 ρ / Mantel p | L2 最近邻一致 | held-out L3 ρ / Mantel p | L3 最近邻一致 |
|---|---:|---:|---:|---:|---:|
| **probe output L2** | **0.981** | **0.958 / <0.001** | **92%** | **0.861 / <0.001** | **77%** |
| performance footprint L3 | 0.899 | 0.744 / 0.001 | 54% | 0.816 / <0.001 | 46% |
| Qwen direct L1+ | **0.203** | 0.262 / 0.153 | 31% | 0.452 / 0.038 | 31% |
| Qwen descriptor TF-IDF | 未估 | 0.036 / 0.379 | 54% | 0.171 / 0.134 | 46% |
| AST L0 | 1.000 | -0.133 / 0.838 | 31% | -0.091 / 0.732 | 23% |
| token L0 | 1.000 | -0.193 / 0.921 | 15% | -0.112 / 0.754 | 8% |

结论不是“L2 天然正确”，而是更具体的：**在当前 repair 接口、经典/手工/Qwen 混合算子集和独立实例划分上，固定残解后的输出边集合差异稳定地迁移到了 held-out 行为与性能足迹。** 它达到了预先建议的信度 ≥0.8、相关 ≥0.5、Mantel p<0.05 门槛。

### 2.3 三个决定性反例

1. `sequential_insertion × sequential_rewrite`：AST 0.772，held-out 输出 0.000。代码重写被 L0 当成很远，实际行为完全一致。
2. `sequential_insertion × block_reinsert`：AST 0.785，held-out 输出 0.008。名为 block 专才不等于行为真的专门化；在这批残解上它几乎退化成 sequential/fallback。
3. `regret_2 × regret_3`：AST 0.000，held-out 输出 0.190。只改变参数 `k` 不会改变 AST node-type 结构，却能产生可观测行为差异。

这三个例子同时覆盖“假多样性”和“隐藏多样性”，足以否定把 AST/代码结构直接当作核心行为距离。

### 2.4 Qwen 作为多样性裁判

Qwen 的三次机制描述文字相当一致，但直接差异矩阵的跨重复相关只有 **0.203**。随机改变算子排列顺序后，其数值判断大幅变化。

因此：

- 可以让 Qwen 生成机制描述、变异假设或 interestingness 理由；
- 当前不能把一次 Qwen 0–100 分数直接用于去重、聚类、fitness sharing 或生存选择；
- 若以后继续这条线，至少要做固定 rubric、pairwise 独立判断、顺序随机化和 judge ensemble，而不是单次整矩阵评分。

### 2.5 当前推荐度量栈

| 层 | 用途 | 当前决策 |
|---|---|---|
| 精确 hash/AST/token | 免费过滤完全重复、报告换皮程度 | **保留，但不代表行为** |
| Qwen 机制描述/判断 | 生成假设、人工审阅、可选 interestingness gate | **不进入自动选择** |
| L2 probe 输出边距离 | 近克隆识别、行为簇、变异链、新颖性与有效自由度 | **升为核心仪表** |
| L3 性能足迹 | 判断差异是否有用、组合库边际贡献、最终验证 | **作为较贵验证层** |
| 有效自由度 | 汇总某表示下的群体维度 | **只在 L2/L3 上解释，不作性能目标** |

本轮没有运行 CodeBERT/GraphCodeBERT：当前 Python 环境没有 torch/transformers。Qwen descriptor TF-IDF 明确标为 L1 proxy，不能冒充代码 embedding。鉴于 L0 与 L2 的差距已经非常大，是否投入代码模型应作为低优先级补充，而不是阻塞主线。

---

## 3. 实验 B：Qwen repair 变异链坍缩审计

### 3.1 设计

- 4 条链 × 8 代，共 32 次基础 Qwen 调用；
- 起点为 sequential、regret-2、block-reinsert、new-route；
- 每个合法子代无论性能都接替父代，不使用 fitness selection；
- 每代只抽一次，不对失败链补抽；
- L0 测 AST 变化，L2 测固定 validation 残解上的输出边变化；
- 预注册强证据：calibration 与 validation 有效自由度均下降至少 20%，且后半程行为重访率至少 50%。

### 3.2 结果

| 指标 | 结果 |
|---|---:|
| 合法子代 | 19 / 32 |
| 无效/失败 | **40.6%** |
| 合法子代 parent AST 距离中位数 | 0.627 |
| 合法子代 parent L2 距离中位数 | 0.246 |
| parent L0–L2 Spearman | 0.442 |
| validation gain 中位数 | +4.90% |
| validation gain 范围 | -9.36% 至 +5.61% |
| 后半程行为重访率 | 81.8% |
| calibration 有效自由度 | 1.50 → 1.80 |
| validation 有效自由度 | 1.68 → 1.78 |
| validation 局部噪声校正有效自由度（敏感性） | 1.68 → 1.75 |

四条链最终合法节点：

| 起点 | 合法子代数 | 起点 gain | 最终合法代 | 最终 gain |
|---|---:|---:|---:|---:|
| sequential | 5 | +4.69% | g8 | **-9.36%** |
| regret-2 | 5 | +5.65% | g7 | +5.08% |
| block-reinsert | 4 | +4.20% | g8 | +5.02% |
| new-route | 5 | -33.57% | g6 | -1.30% |

### 3.3 正确解释

本轮出现了大量链内重访，但**没有出现跨链群体有效维度下降**。相反，两个数据划分上的原始有效自由度都略有上升。高重访率与不下降的群体维度并不矛盾：每条链可能在自身少数模式间反复，但不同链仍停留在不同区域。

不能下强结论的两个原因：

1. **起点退化。** validation 初始有效自由度只有 1.68，低于预注册的 2.0 门槛。后续度量审计证实 block-reinsert 与 sequential 的 held-out 距离仅 0.008；四个“名字不同”的起点实际没有四个维度。
2. **接口无效率。** 13/32 个候选执行失败，主要因为 Qwen 虚构 `problem.l`、`tw`、`customers`、`distances`、`customer_locations` 等字段，另有不可行路线和未定义变量。这种不同代码不能计为有效多样性。

因此本轮的正式判定保持 **INSUFFICIENT_FOR_SCIENTIFIC_DECISION**。更细的探索性描述是：

> Qwen 单亲变异在合法子代内部表现出高链内行为重访，但当前没有证据表明四条链收敛成同一个群体吸引子；测量受起点行为退化和 40.6% 接口失败污染。

### 3.4 一个意外但有用的发现

不做 fitness selection 时，Qwen 可以产生行为距离很大的候选，但其中一些性能迅速崩坏，例如 sequential 链最终从 +4.69% 走到 -9.36%。这说明：

- “避免坍缩”不能等价为无条件最大化距离；
- 下一轮必须使用**质量约束下的行为新颖性**，例如先保证 calibration gain 不低于父代容忍线，再最大化到档案的 L2 距离；
- L2 是生成/去重工具，L3 或基础性能仍需承担“有用”的门禁。

---

## 4. 对三个原始问题的当前回答

### #1 信用分配

本轮没有直接比较信用算法，暂不宣称解决。新证据是：L3 性能足迹具有 0.899 重测信度，并能预测 held-out L3（ρ=0.816），可以作为后续组合库边际信用的上下文表示。信用实验应在变异/距离仪器稳定后进行。

### #2 变异退化为相似

主线保留，但当前只得到“链内重访高、跨链坍缩未证实”。下一次实验必须先提高起始有效维度、提供精确接口契约，并在相同候选预算下比较质量选择与质量约束的 L2 新颖性选择。

### #3 算子距离

L2 输出边 Jaccard 已通过本测试床的 pilot 门禁，是当前最扎实的进展。L0 只适合快速过滤；Qwen judge 不稳定；L3 适合验证有用性。后续还需加入搜索阶段分层和完整轨迹，才能把 pilot 距离升级为通用距离。

---

## 5. 下一步主实验：质量约束的 L2 引导变异

下一步不直接实现完整协同进化，先做一个能明确归因的两臂实验。

### 5.1 先决门禁

1. 新增至少 3 个经 probe 验证的行为专才，而不是按名字认定专才；
2. 用 calibration L2 做 max-min 种子选择，并要求独立 validation 初始有效自由度达到预注册门槛；
3. prompt 明确列出 `Problem` 的真实字段和允许 helper，目标是把合法率从 59.4% 提高到至少 85%；
4. 仍将生成合法率单独报告，不把失败代码算作多样性。

### 5.2 两个实验臂

每个父代生成同样数量的候选、使用同样 token 上限和 probe 预算：

| arm | 候选存活规则 | 回答什么 |
|---|---|---|
| M0 quality-only | 选 calibration gain 最好的合法候选 | 改进导向是否自然坍缩 |
| M1 quality-constrained L2 | 在 gain 不低于父代容忍线的候选中，选离行为档案最远者 | L2 引导能否维持有用多样性 |

建议先用 2 个经过门禁的种子 × 4 代 × 每代 3 候选 × 2 arms，共 48 次调用做 pilot。若合法率和起点门禁通过，再扩成 4×8。

### 5.3 主指标与决策

- 主指标：validation L2 有效自由度曲线；
- 共同主指标：held-out L3 组合库价值/性能足迹覆盖；
- 次指标：行为重访率、每次 LLM 调用的独特合法行为产出率、最终 gain、无效率；
- 成功条件：M1 的最终有效自由度和 L3 覆盖显著高于 M0，且最终求解质量不劣于预设容忍线。

这一步直接检验最核心的机制命题：**可信行为距离进入变异/存活后，能否阻断“改进压力 → 行为收缩”的闭环，同时不奖励无效怪异算子。**

---

## 6. 产物与复现

| 产物 | 路径 |
|---|---|
| 变异链预注册 | [`../mutation-collapse-pilot.md`](../mutation-collapse-pilot.md) |
| 变异链脚本 | [`../mutation_collapse_audit.py`](../mutation_collapse_audit.py) |
| 变异链完整结果 | [`./mutation-collapse/pilot/`](./mutation-collapse/pilot/) |
| 度量审计说明 | [`../diversity-metric-pilot.md`](../diversity-metric-pilot.md) |
| 度量审计脚本 | [`../diversity_metric_audit.py`](../diversity_metric_audit.py) |
| 度量审计完整结果 | [`./diversity-metrics/pilot/`](./diversity-metrics/pilot/) |

所有 Qwen prompt、原始 response、通过安全过滤后的候选代码、逐代观测、距离矩阵和图片均已保存。API key 只从根目录 `.env` 加载，未写入结果。
