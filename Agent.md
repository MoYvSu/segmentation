# Agent.md — 低碳钢金相图像相区分割开发指南

> 本文档是 AI 协作者 / 开发者进入本仓库的指导手册，描述当前（2026-07 边界预测版本）的架构、约定与已知问题。修改代码前请先完整阅读。

Python 环境：`conda activate sam2_env`（或 `D:\Anaconda\envs\sam2_env\python.exe`）。

## 1. 项目目标

对低碳钢金相图像进行**相区分割 + 晶粒实例分割**：

- 语义：区分珠光体（0）/ 铁素体（1）；
- 边界：独立通道预测晶界（grain boundary）；
- 实例：边界骨架化 + 核心剥离 + 受阻分水岭 → 实例 ID + 类别映射 JSON。

## 2. 硬约束（不可违反）

1. 总参数量 < 500M（当前冻结 encoder 约 80M + 解码头约 10.8M，通过）。
2. 零预训练解码器：FPN 必须随机初始化，禁止加载 SAM 2 原生 Mask Decoder / neck / memory 等权重。
3. 本地化隔离：权重只放 `weights/`，禁止 `~/.cache/` 等全局隐式路径。
4. 长宽比保真：一律 Letterbox（等比缩放 + BORDER_REFLECT 镜像填充），禁止挤压变形。
5. 禁止离线改图：所有变换/增强必须在线完成。

## 3. 当前架构（两阶段）

### Stage 1（`train.py`，全监督，已完成）

- 冻结 SAM 2 Hiera trunk（base+），输出 4 尺度特征：112 / 224 / 448 / 896 ch，分辨率 256→32。
- **独立双 FPN 解码头**（`models/fpn_decoder.py`）：`seg_fpn` + `boundary_fpn` 各自拥有独立的 lateral_convs + ResidualBlocks + top-down 融合（256ch），再各接独立输出头（GroupNorm + ReLU + Dropout + 1×1 Conv）。
- 输出 2 通道：`[:,0]` 语义 logits，`[:,1]` 边界 logits。
- 损失：`BoundaryLoss` = 语义 BCE + `alpha_boundary` × FocalLoss(边界) × EDT 权重（权重 clamp [1, 4]）。
- 优化：AdamW(1.5e-4) + warmup(25) → cosine；梯度裁剪 1.0；best 模型按复合评分 `0.2×mIoU + 0.8×BndIoU` 保存。

### Stage 2（`train_stage2.py`，半监督 Mean Teacher，进行中）

- 从 Stage-1 checkpoint 初始化 decoder；维护 EMA 教师（decay 0.999，可自适应）。
- 双流 batch：有标签流（BoundaryLoss）+ 无标签流（一致性损失，权重 `unsup_weight` × sigmoid ramp-up 10 ep）。
- 边界伪标签源 `boundary_teacher_mode`：
  - `ema`（默认）：EMA 教师 + Stage-1 锚点渐进混合（`anchor_alpha` 从 1.0 线性衰减至 `anchor_floor=0.3`，20 ep）；
  - `stage1_direct`：Stage-1 冻结模型直接输出（无 EMA 滞后）；
  - `self_consistency`：学生弱增强预测 stop-gradient（无 EMA 依赖）。
- 无监督损失（`utils/loss_semi.py`）：
  - 语义：弱图伪标签 vs 强图学生预测 MSE，Random Patch Masking 加权（掩码区 seg 权重 2.0、boundary 权重 0.3）；
  - 边界：MSE + `sobel_weight`×Sobel 梯度一致性 + `tv_weight`×各向异性 TV + `bg_suppress_weight`×背景抑制；
  - **骨架过滤**（`skeleton_filter`）：阈值 0.3 二值化 → Zhang-Suen 骨架 → 膨胀 1px → 高斯模糊软目标；**必须施加在锚点混合之后**。
- 渐进式外观增强（`utils/progressive_aug.py`）：仅施加于学生输入（亮度/对比度/锐度/噪声），prob 0→0.8 线性 ramp 10 ep；启用时自动禁用 UnlabeledDataset 内置外观增强。
- 分支冻结：`semi_supervised.freeze.seg_branch` / `freeze.boundary_branch`（默认联合训练；两个都为 true 会被强制取消）。
- 优化器：三分组 AdamW（seg lr / boundary lr = base×0.1）。
- 调度：`flat_decay`（warmup 5 → flat 30 → 温和线性衰减至 0.2×base）或 cosine；无监督权重 sigmoid ramp-up；自适应 EMA。
- 监控：每 `checkpoint_interval`(5) epoch 对 `data/test` 前 3 张输出语义/边界概率图到 `outputs/stage2/monitor/`。

## 4. 模块地图

| 文件 | 职责 |
|------|------|
| `models/sam2_encoder.py` | 冻结 Hiera trunk，返回 4 尺度特征；禁全局缓存 |
| `models/fpn_decoder.py` | 独立双 FPN + 输出头；`SegmentationModel`；`freeze_seg_branch` / `freeze_boundary_branch` |
| `data/dataset.py` | `letterbox`（BORDER_REFLECT）、Labelme 解析、边界权重、`BoundaryDataset` |
| `data/dataset_semi.py` | `LabeledDataset` / `UnlabeledDataset`（弱强双路 + patch masking）|
| `data/active_learning.py` | 不确定性采样 + mask→Labelme JSON 反向网关 |
| `tools/preprocess_labels.py` | 离线净化 GT 生成（CLAHE + Canny + 内部掩码腐蚀）|
| `utils/loss.py` | `BoundaryLoss` |
| `utils/loss_semi.py` | 半监督一致性损失、骨架过滤、EMA 更新 |
| `utils/post_process.py` | 边界分水岭实例分割、距离场补偿（旧）等 |
| `utils/metrics.py` | `SegMetrics`（mIoU / mDice / Boundary IoU）|
| `utils/progressive_aug.py` | 渐进式外观增强 |
| `train.py` / `train_stage2.py` | 两阶段训练入口 |
| `inference.py` | Letterbox 推理 + 后处理管线 |

## 5. 数据管线

- 有标签流：`data/raw/` 图像 + `data/purified_gt/{name}_gt.npz`（`semantic` + `boundary`）；在线 letterbox 1024、随机翻转/旋转、可选 random crop(512)；在线计算 EDT 边界权重图（范围 [1, 4]）。
- 无标签流：`data/unlabeled/`；弱图（仅 letterbox）+ 强图（空间 + 外观增强 + patch masking）。
- 训练/验证划分：`train_ratio=0.8`，`seed=42`。
- 类别约定：语义 0=珠光体，1=铁素体；边界 1=晶界。Labelme label 支持 `ferrite` / `ferrite_core` / `铁素体` / `1` 与 `pearlite` / `珠光体` / `0`。
- 净化 GT 生成：`python tools/preprocess_labels.py`（标注变更后必须重新执行）。

## 6. 配置参考（`config/default_config.yaml`）

`config/default_config.yaml` 是唯一主配置（含 Stage 2 的 `semi_supervised` 段）；`config/stage2_config.yaml` 为早期独立配置，已被取代，不要回退使用。

- `paths.project_root`：硬编码绝对路径（YAML anchor 复用），迁移目录时需同步修改。
- `inference`：test_dir / output_dir / threshold / boundary_threshold / checkpoint_stage / stage1|stage2_checkpoint。
- `semi_supervised`：boundary_teacher_mode / unsup_weight / unsup_rampup_epochs / ema_decay / adaptive_ema / lr_schedule（flat_decay）/ flat_epochs / freeze / boundary_anchor / boundary_consistency / skeleton_filter / patch_mask / monitor / checkpoint_interval。
- `progressive_aug`：学生输入外观增强参数（enabled / ramp_epochs / max_prob / 各抖动范围）。
- `boundary`：净化 GT 目录、EDT 权重范围、净化参数。

新增超参数时务必附中文注释说明动机（参考现有 YAML 的注释风格）。

## 7. 训练与推理命令

```bash
conda activate sam2_env

# Stage 1 全监督
python train.py --config config/default_config.yaml
python train.py --config config/default_config.yaml --resume outputs/stage1/checkpoint_epoch50.pth

# 离线净化 GT
python tools/preprocess_labels.py

# Stage 2 半监督
python train_stage2.py --config config/default_config.yaml
python train_stage2.py --config config/default_config.yaml --resume outputs/stage2/stage2_epoch30.pth
# 分支切换（仅加载 decoder，重置优化器/调度器/epoch）
python train_stage2.py --config config/default_config.yaml --init_from_checkpoint outputs/stage2/best_model_stage2.pth

# 推理
python inference.py --config config/default_config.yaml --checkpoint outputs/stage2/best_model_stage2.pth
```

Checkpoint 统一格式：`decoder_state_dict` + `optimizer_state_dict` + `scheduler_state_dict` + `epoch` + `best_composite_score` + `config`。读取时用 `.get()` 兼容旧 key（如 `best_val_iou`）。

## 8. 调试与验证

- `debug_pipeline.py`：数据管线诊断（letterbox 比例、掩码生成、分水岭）。
- `debug_iou.py`：零 epoch IoU 硬审计（数据源 / 前向数值 / 二值化门限三项闭环）。
- `test_skeleton_watershed.py`：纯图像处理的骨架 + 受阻分水岭验证。
- `test_watershed.py`：分水岭单元测试。
- `visualize_instances.py`：实例图着色（`_inst.png` + `_class.json`）。
- Stage 2 训练每 5 epoch 自动产出 monitor 概率图，直接目检语义/边界头是否收敛、是否出现雾状热力图。

改代码后的最小验证顺序：

1. 改数据管线 → 跑 `debug_pipeline.py`；
2. 改损失 → 小步数训练并盯 sup/unsup 各项数值与 monitor 图；
3. 改后处理 → 跑 `test_skeleton_watershed.py` + `inference.py` 单图目检。

## 9. 开发规范

- 注释与提交信息使用中文（`<type>: 描述`，如 `feat:` / `fix:` / `refactor:` / `docs:` / `para:`）。
- 所有文件 UTF-8 编码；`.py` 文件头保持 `# -*- coding: utf-8 -*-`。
- 新增依赖需同步更新 `requirements.txt`；权重一律本地化（`weights/`）。
- 不破坏既有接口：checkpoint 新 key 向后兼容；`letterbox` 长宽比与 BORDER_REFLECT 约定不可改。
- 配置改动需注释动机，避免"魔法数字"无说明。

## 10. Git 工作流

- 分支前缀 `codex/`。
- `.gitignore` 忽略：`outputs/`、`weights/`、`segment-anything-2/`、`data/raw|test|unlabeled|smoketest/`、`config/*`（但保留 `default_config.yaml` 与 `stage2_config.yaml`）。
- 提交前先 `git status` 确认工作区；工作区可能存在未提交 WIP（当前为 `default_config.yaml` / `train_stage2.py` / `utils/loss_semi.py` 的背景抑制与多模式伪标签改动），属于正常迭代，不要随意丢弃或 `git checkout --` 还原。
- 每次有意义的实验/修复单独提交，便于回溯（本项目历史即按此组织）。

## 11. 已知问题与实验方向（来自迭代历史）

- **边界头崩塌**：早期硬门控导致，已用 Stage-1 锚点渐进混合根治，不要再退回硬门控方案。
- **雾状热力图**：像素级 BCE 软目标缺乏空间结构约束，现用 Sobel 梯度一致性 + 各向异性 TV + 背景抑制解决；若复现需先检查这三项权重。
- **圆斑过拟合**：patch masking 区边界权重降至 0.3。
- **背景膨胀**：温度锐化 + 背景抑制损失。
- **三分类全盲预测死锁**：已改为二分类 + 边界通道，不要退回三分类。
- **半监督初段伪标签不可靠**：unsup 权重 sigmoid ramp-up 10 ep。
- **patch_mask 与 output_size 尺寸一致性**：曾有专门修复，改动时注意。
- **调度器迁移**：曾从 SequentialLR 迁移到 LambdaLR，旧 checkpoint 恢复时会从头重新调度（代码已兼容）。
- 待探索：语义/边界分支进一步解耦训练、锚点混合调度曲线、更大规模无标签数据利用。
