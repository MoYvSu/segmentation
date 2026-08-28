# E10a：冻结 V6 特征，冷启动完整语义解码器

## 目的

E9 epoch 20 只把临界像素轻微推向铁素体，说明冻结 V6 语义 FPN/head 后，可学习的
高分辨率残差仍受旧语义决策约束。E10a 验证一个更彻底但保持几何不变的方案：冻结
已验证的 SAM2 + V6 LoRA 特征和全部边界/affinity 路径，随机重置并训练完整语义解码器。

E9 epoch 20 probability-mean 继续保留为提交候选；E10a 只有在固定部署口径和目检均更好时
才允许晋级。

## 架构与训练约束

```text
冻结 SAM2 + V6 LoRA 特征
          |
随机 semantic FPN + coarse semantic head
          |
256 -> 512 -> 1024 图像引导高分辨率语义路径
          |
      绝对语义 logits
```

- 在重置学生语义路径前复制 V6，作为整个训练期固定不更新的教师。
- 随机重置 `seg_fpn`、`seg_branch` 和 `semantic_residual`；高分辨率输出层从零开始。
- 只训练上述三个语义模块；LoRA、boundary/refine/center 和 affinity geometry 全部冻结。
- 有标签数据使用像素 BCE/Dice 与整实例概率目标。
- 无标签数据只在 V6 教师置信度 `>=0.90` 的区域蒸馏，权重从 `0.12` 线性降至 `0.03`，
  避免随机冷启动漂移，同时不锁死教师不确定区域。
- monitor 使用固定无标签 holdout，不参与 checkpoint 选择。

## 训练

```bash
conda activate sam2_env
python train_stage2.py \
  --config config/train/stage2_semantic_e10a_cold20.yaml
```

输出目录：`outputs/stage2_semantic_e10a_cold20/`。

## 固定几何推理

```bash
python tools/run_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml
```

推理仍由 V6 提供前景、G4b 提供 affinity/watershed 几何；E10a 只替换实例类别概率，
因此与 G4b 的实例 ID 和数量应完全一致。优先检查 test009 下边缘临界实例、纤细铁素体召回、
大块铁素体误判，以及是否新增系统性 pearlite-to-ferrite 偏置。

## 晋级标准

1. 固定有标签验证集的类别感知实例 mIoU 和铁素体平均面积代理不低于 E9/V6；
2. 无标签 monitor 不出现大面积类别翻转或置信度塌缩；
3. 固定 G4b 几何下，test 目检的铁素体漏判减少且错误翻转没有同步放大；
4. 最终仍以竞赛黑盒得分决定是否替换 `80.67` 的 G4b 基线。
