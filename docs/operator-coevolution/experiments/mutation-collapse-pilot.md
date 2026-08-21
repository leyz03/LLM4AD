# 今晚首个实验：LLM repair 变异链坍缩审计

> 状态：预注册 pilot  
> 日期：2026-08-21  
> 脚本：[`mutation_collapse_audit.py`](./mutation_collapse_audit.py)

## 为什么先做这个

三个问题中，信用分配需要真实双种群账本，距离的完整效度审计需要专才算子和独立 L3 真值；两者今晚都容易扩成数天工程。变异退化可以先脱离完整协同进化单独测量，而且允许调用 LLM 后，4 条链 × 8 代只需要 32 次基础生成。

这个实验回答一个足以改变下一步优先级的问题：

> 在我们实际使用的模型、EoH 风格单亲变异 prompt 和 repair 接口上，LLM 变异是否会在没有 selection 的情况下自行收缩到重复行为？

若答案为是，下一步优先实现 verbalized sampling、近父代/远档案双约束和缺口驱动生成；若答案为否，而闭环种群仍坍缩，优先调查信用和存活选择。

## 假设与隔离设计

- H1：多条行为不同的 repair 变异链会逐渐收缩到少数行为模式。
- 不使用 fitness selection；每个合法子代无论表现都接替父代。
- 不向 LLM 提供其它链、历史档案或当前分数，避免选择和跨链提示造成混淆。
- 四个起点依次为 sequential、regret-2、block-reinsert、new-route repair，包含高质量近邻和行为明显不同的专才/低质量控制。
- calibration 与 validation 使用不同实例；判定必须在两者上方向一致。

## 预算

正式 pilot：

```text
4 chains × 8 generations = 32 base LLM calls
```

默认每代只调用一次，不因无效输出偷偷增加某些链的预算。每次输出上限 1400 tokens。若模型合法率太低，先如实报告，再统一把 `--attempts-per-generation` 改为 2 重跑所有链。

probe 成本为纯 CPU：CVRP/VRPTW 各 4 个实例 × 2 个状态 × 5 种经典 destroy，共 80 个固定残解；每个 repair 用两套 RNG 重复测量。

## 指标

| 指标 | 定义 | 作用 |
|---|---|---|
| L0 parent/ancestor distance | AST node-type 3-gram Jaccard | 判断代码结构是否变化 |
| L2 parent/ancestor distance | 固定残解上最终边集合的平均 Jaccard | 判断行为是否变化 |
| behavior revisit | 到本链历史最近 L2 距离不超过重复测量噪声 95% 分位 | 判断是否回到已有行为 |
| effective dimension | L2 RBF kernel 特征值的参与比 | 判断四条链是否整体坍缩 |
| validation gain | 独立 probe 上的平均相对改进 | 区分新颖但无效与有用变化 |
| invalid rate | 生成、AST 安全过滤或执行失败比例 | 单独报告工程有效性 |

L2 当前是适合 pilot 的输出行为距离，不宣称已经完成问题 #3 的最终距离效度验证。

## 预注册判定

只有同时满足以下条件才记为“强变异坍缩证据”：

1. 初始 validation 有效自由度至少为 2；
2. calibration 与 validation 的最终有效自由度相对初始都下降至少 20%；
3. 后半程合法子代中，至少 50% 是 behavior revisit。

只满足一部分记为“部分证据，需要增加链数/代数或分析换皮”；都不满足记为“当前预算未检出”，不能外推为 LLM 变异普遍不会坍缩。

## 今晚运行顺序

### 1. 管线自检（不调用 LLM）

```powershell
python docs/operator-coevolution/experiments/mutation_collapse_audit.py `
  --self-test --chains 2 --generations 2 --probe-instances 2 --probe-repeats 1 `
  --output docs/operator-coevolution/experiments/results/mutation-collapse/self-test
```

自检只验证 prompt 保存、代码解析、子进程执行、probe、距离、报告和图片，结果不能用于研究结论。

### 2. 设置 OpenAI-compatible 后端

```powershell
$env:LLM_BASE_URL='https://YOUR-ENDPOINT/v1'
$env:LLM_API_KEY='YOUR-KEY'
$env:LLM_MODEL='YOUR-MODEL'
```

脚本默认加载仓库根目录 `.env`，也可以使用上述环境变量；密钥不写入配置或结果。

### 3. 一次调用的 smoke test

```powershell
python docs/operator-coevolution/experiments/mutation_collapse_audit.py `
  --chains 1 --generations 1 --probe-instances 2 --probe-repeats 1 `
  --output docs/operator-coevolution/experiments/results/mutation-collapse/smoke
```

smoke test 成功的标准是 `chains/.../candidate.py` 存在且 `observations.csv` 中 generation 1 为 `valid`。它的统计规模不足，脚本会明确输出 `INSUFFICIENT_FOR_SCIENTIFIC_DECISION`。

### 4. 正式 pilot

```powershell
python docs/operator-coevolution/experiments/mutation_collapse_audit.py `
  --output docs/operator-coevolution/experiments/results/mutation-collapse/pilot
```

中断后使用相同命令加 `--resume`，已经保存的原始 response/candidate 不会再次调用。

## 输出与下一步

| 文件 | 内容 |
|---|---|
| `REPORT.md` | 预注册规则下的判定和解释 |
| `summary.json` | 有效自由度曲线、噪声线、重访率、失败率 |
| `observations.csv` | 每条链逐代 L0/L2/性能指标 |
| `trajectories.png` | 到祖先的行为距离和有效自由度曲线 |
| `chains/` | 每次 prompt、原始 LLM response、候选代码和错误 |

后续分支：

- 强证据：比较 baseline、verbalized sampling、双约束、缺口驱动四个 mutation arms；
- 部分证据：先检查“L0 变远但 L2 重访”的换皮比例，再扩到 8 条链 × 12 代；
- 未检出且初始多样性足够：把首要资源转向离线信用剪枝重放；
- 初始多样性不足：改进 seed 专才和 probe 后重跑，不解释收缩曲线。
