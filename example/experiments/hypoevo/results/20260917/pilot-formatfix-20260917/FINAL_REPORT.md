# 格式修复最终结果（复用旧 EoH）

用户要求停止 EoH 重跑并复用旧结果。本报告的 HypoEvo、关闭记忆各 6 组来自修复后批次，EoH 6 组全部来自 pilot-real-v3-20260917。本轮已经生成的 EoH 结果一律不参与最终比较。

每方法每任务 3 个种子，每次 28 次模型调用；qwen3.5-flash；搜索集 16 个、独立测试集 64 个均匀随机实例。路径长度越小越好。

| 任务 | 方法 | 测试均值 ± 标准差 | 合法候选/尝试 |
|---|---|---:|---:|
| tsp | hypoevo | 6.7841 ± 0.2000 | 48/51 |
| tsp | no_memory | 6.6814 ± 0.3443 | 51/57 |
| tsp | eoh | 6.8325 ± 0.0229 | 71/84 |
| cvrp | hypoevo | 14.0742 ± 0.0494 | 41/50 |
| cvrp | no_memory | 14.2961 ± 0.0124 | 47/57 |
| cvrp | eoh | 14.0729 ± 0.1330 | 47/84 |

## 解释与限制

- 修复只将元数据合规与代码合法性分开。缺失有效假设不会被算为假设检验；接口和路线合法性门槛保留。39 项回归测试通过。
- EoH 候选分母包含全部生成请求，含提前解析失败；反思与摘要调用计入 HypoEvo 调用预算但不是单独候选。
- EoH 是历史对照而非同期重新采样。三种子 pilot 不构成显著性或普遍优势证据。
- 新旧批次使用相同测试集，未按测试分数调参；正式泛化验证仍需新保留数据。
- 原始 REPORT.md 是本轮执行监控，包含被排除的新 EoH；最终对照以此文件和 FINAL_RESULTS.json 为准。

## 调用账目

{
  "new_recorded_calls": 433,
  "new_recorded_tokens": 1181577,
  "new_call_upper_bound": 435,
  "formal_new_calls": 336,
  "reused_old_calls": 168,
  "note": "New EoH runs excluded by user request; old EoH results reused. Interrupted requests may be billable."
}
