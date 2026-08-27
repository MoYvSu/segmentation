# E7c 保守双语义实验（2026-08-28）

## 判断

E7b-A 不是完全无效：测试集目检显示每张图约有 1--2 个类别判断被修正。但将 E7b
checkpoint 整体替换进 G4b 后，它同时改变前景门控和实例分类，并在 6178 个实例中产生
486 次类别翻转，其中 365 次为 pearlite→ferrite。局部修正与全局偏置因此不能分开。

## E7c-0：先解耦，不重新训练

固定路径为：

`V6 semantic foreground → G4b affinity/watershed → V6 hard vote → E7b conservative challenge`

E7b 只读取与 V6 完全相同的冻结 encoder/LoRA 特征，只加载自己的 `seg_fpn + seg_branch`。
它不能进入前景、marker、affinity 或 watershed，因此 E7c 与 G4b 的实例图必须逐像素一致。

默认保留 V6 类别。只有两模型不一致时才考虑覆盖：

- pearlite→ferrite：V6 hard ratio ≥0.35、E7b core ≥0.85，并且核心相对整实例至少提高
  0.08，显式针对“亮核心 + 粗黑边”；
- ferrite→pearlite：V6 hard ratio ≤0.65 且 E7b core ≤0.15；
- 不固定每图翻转数、实例数或平均面积，不改变 255 ID 上限。

配置：`config/experiments/affinity_g4b_high065_semantic_dual_e7c.yaml`。

## 判定顺序

1. 六张有标签验证图跑完整部署闭环，和 G4b high0.65 比较有效匹配 mIoU、铁素体面积
   误差、总分和逐图宏平均；
2. 68 张测试图确认实例图逐像素一致，统计翻转方向、面积和门控原因；
3. 目标翻转规模约 50--150 个，即每图约 1--2 个，不把该范围设为硬约束；
4. 由目检确认已知修正仍被保留，且没有新的系统性 ferrite 偏置。

## E7c-1：仅在 E7c-0 有正信号后训练

若保守门控有效，再训练 V6 锚定的稀疏语义修正器：冻结 encoder、LoRA、G4b geometry
和 `seg_fpn`，只训练零初始化 residual classifier；在 V6 已正确的实例上蒸馏 V6，在 V6
错误或低置信实例上使用人工 GT 纠错。checkpoint 继续按完整部署指标选择。若 E7c-0
无法保留目检收益，则不进入该训练，避免再次扩大模型自由度。

## E7c-0 首轮结果

- 六张验证图：代理总分 `79.1800`、有效匹配 mIoU `0.83684`、铁素体面积误差
  `0.25324`，与当前 G4b hard 基线完全一致；严格门控没有引入验证集误翻；
- 68 张测试图：6178 个实例图逐像素完全一致，无 geometry mismatch；
- 只翻转 15 个实例，其中 14 个 pearlite→ferrite、1 个 ferrite→pearlite；
- 翻转面积中位数 2607 px，唯一 ferrite→pearlite 的面积为 200960 px。大实例也可能是
  V6 的真实类别错误，因此不按面积禁止翻转；改由有标签闭环指标和逐实例目检约束风险。

结论：解耦方向成立，但首轮门控偏严。下一步不重新跑网络，直接利用已缓存的 V6/E7b
实例分数做小规模阈值扫描；目标是把测试翻转提高到约每图 1--2 个，同时保持六图代理分数
不显著低于 hard 基线。扫参工具为 `tools/sweep_semantic_dual_gate.py`，只复算类别映射，
不改变实例几何，也不设置任何实例面积门槛。

## E7c-R 缓存扫参结果

三档门控在相同缓存上重算，结果如下：

| 门控 | 六图翻转 | 六图代理总分 | 测试翻转 | p→f / f→p | 覆盖测试图 |
|---|---:|---:|---:|---:|---:|
| strict | 0 | 79.1800 | 15 | 14 / 1 | 13 / 68 |
| medium | 0 | 79.1800 | 38 | 36 / 2 | 26 / 68 |
| relaxed | 1 | 79.1193 | 67 | 63 / 4 | 37 / 68 |

relaxed 在验证集翻转的 `train_623 / instance 142` 从 pearlite 改为 ferrite 后，与 ferrite
GT 新增一个 IoU `0.631` 的有效匹配。总分下降约 `0.061` 并非该类别错误，而是新增的小
铁素体使当前验证集已偏小的 ferrite mean area 进一步下降。考虑到六图验证覆盖不足且目检
已确认 E7b 能逐图纠正少量实例，选择 relaxed 作为下一目检候选；strict 和 medium 均保留。

部署配置为
`config/experiments/affinity_g4b_high065_semantic_dual_e7c_relaxed.yaml`。它只改变
67/6178 个测试实例的类别，不改变任何实例像素、实例数或 `<=255` 输出约束。完整 68 图
重跑已验证：实例图 geometry mismatch 为 0，翻转数与缓存重放均严格等于 67。产物位于
`outputs/submission_affinity_g4b_high065_semantic_dual_e7c_relaxed/`，审计报告为
`outputs/analysis_e7c_relaxed_vs_g4b.json`。最终是否晋级
仍由成对可视化和黑盒提交决定，不能只看六图代理总分。
