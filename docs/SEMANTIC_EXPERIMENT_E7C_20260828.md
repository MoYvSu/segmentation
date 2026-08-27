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

