# E9 高分辨率细实例语义实验（2026-08-28）

## 动机

G4b high0.65 + V6 的官方黑盒基线为 `80.67 / 0.8441 / 0.7693`。目检发现测试集中的
纤细铁素体会被周围珠光体拉低。V6/E8 语义决策实际发生在 stride-4 特征图，1024 输入只产生
256 级语义 logits；末端插值不能恢复已经丢失的窄结构证据。E8 的低分辨率残差还快速用满
`+/-2.0` logit，验证集总体 mIoU 未提升。

## 架构

E9 冻结 V6 encoder、LoRA、`seg_fpn/seg_branch`、边界分支和完整 G4b affinity，只训练
`SemanticHighResolutionResidualAdapter`：

```text
V6 stride-4 semantic feature/logit
        + RGB / normalized luminance / local contrast
        -> stride-2 refine
        + full-resolution image cues
        -> stride-1 bounded residual (+/-0.75 logit)
```

粗语义概率和局部变化只作为输入特征，不作为高置信硬锁。输出层零初始化，所以 epoch 0 在完整
分辨率上严格复现 V6 插值结果。新增参数保持轻量，不开放 affinity LoRA。

## 非凸与纤细实例目标

不使用几何质心，也不选择最高置信像素。训练在原有逐像素 BCE/Dice 和实例等权 core BCE 外，
增加“实例完整区域平均概率”的 BCE；这与部署侧 `probability_mean` 投票一致。腐蚀后没有 core
的实例视为纤细实例代理，仅以 `1.5x` 温和增权，实例仍保留全部像素。该定义适用于任意连通或
非凸形状。

## 受控判定

训练：

```bash
conda activate sam2_env
python train_stage2.py --config config/train/stage2_semantic_e9_highres20.yaml
```

训练后对同一 checkpoint 分别生成概率均值和历史 hard-majority 两组：

```bash
python tools/run_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_e9_highres.yaml
python tools/run_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_e9_highres_hard.yaml
```

必须检查：epoch 0 与 V6 一致；LoRA/旧 decoder 零漂移；E9 residual 不快速饱和；两类 IoU、
铁素体 precision/recall、实例翻转修正/回退数，以及按实例厚度分层的错误率。最终晋级仍要求
目检和黑盒成绩超过 `80.67`，六张有标签验证只作为筛选证据。
