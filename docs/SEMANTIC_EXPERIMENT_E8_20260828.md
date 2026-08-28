# E8 几何隔离语义残差实验（2026-08-28）

## 已确认基线

G4b high0.65 官方黑盒成绩为 `80.67`，其中实例 mIoU `0.8441`、铁素体平均面积项
`0.7693`。最后的 V6 边界方案为 `79.76 / 0.8420 / 0.7532`。G4b 因此成为当前唯一
晋级基线，后续语义实验不得改变其 affinity、分水岭、实例图或 `<=255` 约束。

E7b/E7c 的 pearlite→ferrite 修正方向有真实价值，但覆盖率不足；少量反向误判说明不能在
推理侧硬编码单向翻转。下一步应提高语义头对灰暗、低对比和非均匀照明铁素体的召回，仍保留
对两类错误的正常惩罚。

## E8 架构

固定以下模块：

- SAM2 encoder 与 V6 LoRA；
- V6 `seg_fpn + seg_branch`；
- 完整 boundary 分支；
- G4b affinity checkpoint 与全部后处理参数。

只增加并训练约 0.25M 参数的 `SemanticResidualAdapter`：

`V6 seg feature + adaptive luminance/local contrast → bounded residual logit`。

输出卷积零初始化，因此 epoch 0 的语义 logits 与 V6 逐元素相同。亮度特征在每张图内独立
标准化，并加入局部对比度；它是可学习软特征，不使用固定 Lab/灰度阈值。残差最大幅度限制为
`±2.0 logit`，避免小数据下完全覆盖 V6。

## 训练目标

- 实例核心 BCE 权重由 `0.50` 提高到 `0.75`；
- 两类均启用难实例聚焦；
- ferrite 实例权重仅为 `1.15`，提供召回趋势而非单向分类规则；
- 使用小权重 asymmetric Tversky（alpha `0.40`、beta `0.60`）提高漏检代价；
- 光照增强重点覆盖 gamma、白平衡和低频照明，同时保留 20% 干净样本；
- 无标签流只做低权重高置信一致性，不把测试集或 monitor 用于训练/选模。

训练配置：`config/train/stage2_semantic_e8_residual20.yaml`。

```bash
conda activate sam2_env
python train_stage2.py \
  --config config/train/stage2_semantic_e8_residual20.yaml
```

## 判定协议

1. 初始 monitor 必须与 V6 一致，冻结 decoder/LoRA 最大变化必须为 0；
2. residual 梯度应非零，输出绝对均值应从 0 平滑增长且不迅速饱和到 2.0；
3. 有标签验证同时报告 ferrite/pearlite IoU，不能只看 ferrite recall；
4. 训练完成后使用固定 G4b high0.65 完整推理：

```bash
python tools/run_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_e8.yaml
```

5. E8 与 G4b 实例图必须逐像素一致，只允许类别 JSON 改变；最终晋级标准是目检与官方黑盒
   分数超过 `80.67`，训练 mIoU 或六图代理只能作筛选证据。
