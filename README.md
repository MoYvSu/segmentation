# 低碳钢金相图像相区分割

> 技术路线：**冻结 SAM 2 Image Encoder + 自制独立双 FPN 解码头（语义/边界双分支）+ 离线边界净化 GT + 两阶段训练（Stage 1 全监督 → Stage 2 半监督 Mean Teacher）+ 边界骨架化 + 受阻分水岭实例分割**。

## 当前进度

- **Stage 1（全监督）已跑通**：双 FPN 解码头、语义 BCE + 边界 Focal×EDT 权重损失，checkpoint 位于 `outputs/stage1/`（`best_model.pth` 等）。
- **Stage 2（半监督微调）进行中**：Mean Teacher（EMA）+ 多模式边界伪标签源 + 梯度感知一致性损失（Sobel + TV + 背景抑制）+ 骨架过滤 + Random Patch Masking + 渐进式外观增强 + 分支冻结 + 三分组优化器 + flat_decay 三阶段调度。checkpoint 位于 `outputs/stage2/`（`stage2_epoch30/50/100.pth`、`语义优化版模型/` 等）。
- **推理后处理已跑通**：边界骨架化 → 核心剥离 → 受阻分水岭 → 语义投票，已在 `data/test`、`data/smoketest` 批量输出实例图与类别映射。

## 环境配置

### Python 环境

```bash
conda create -n sam2_env python=3.11 -y
conda activate sam2_env
pip install -r requirements.txt
```

GPU 版 PyTorch 请按 [pytorch.org](https://pytorch.org/) 的 CUDA 匹配命令安装（如 `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`）。

### SAM 2 本地权重

将 `sam2_hiera_base_plus.pt` 放入 `weights/` 目录，`segment-anything-2/` 为本地 SAM 2 源码仓库：

```
weights/
└── sam2_hiera_base_plus.pt
```

> **约束**：严禁依赖 `~/.cache/` 等全局隐式路径，所有第三方权重必须存放在项目 `weights/` 目录。

## 数据准备

| 目录 | 内容 | 用途 |
|------|------|------|
| `data/raw/` | 有标注图像 + 同名 Labelme `.json`（label 为 `ferrite` / `pearlite`） | Stage 1 / Stage 2 有标签流 |
| `data/purified_gt/` | 离线净化 GT（`.npz`：semantic + boundary） | 训练真值 |
| `data/unlabeled/` | 无标注图像（约 1000 张） | Stage 2 半监督无标签流 |
| `data/test/` | 测试图像（68 张） | 推理 / 训练监控 |
| `data/smoketest/` | 冒烟测试图像（10 张） | 快速验证 |

有标签数据准备（标注变更后需重新生成净化 GT）：

```bash
python tools/preprocess_labels.py            # 生成 data/purified_gt/*_gt.npz
python tools/preprocess_labels.py --visualize  # 可视化净化结果
```

净化流程：CLAHE 增强 → Canny 边缘检测 → 晶粒内部掩码腐蚀剪裁 → 边界带膨胀，得到纯净、无划痕的晶界真值（二值语义掩码 + 二值边界掩码）。

## 项目结构

```
segmentationv2/
├── config/
│   ├── default_config.yaml      # 主配置（含 Stage 1 + Stage 2 半监督全量参数）
│   └── stage2_config.yaml       # Stage 2 早期独立配置（已被 default_config.yaml 取代）
├── data/
│   ├── raw/                     # 有标注图像 + Labelme .json
│   ├── purified_gt/             # 离线净化 GT（_gt.npz）
│   ├── unlabeled/               # 无标注图像（半监督）
│   ├── test/  smoketest/        # 测试 / 冒烟测试图像
│   ├── dataset.py               # 在线数据管道（letterbox / 边界权重 / BoundaryDataset）
│   ├── dataset_semi.py          # 半监督双流数据集（Labeled / Unlabeled）
│   └── active_learning.py       # 不确定性采样 + mask→Labelme JSON 反向网关
├── models/
│   ├── sam2_encoder.py          # 冻结 SAM 2 Hiera trunk（4 尺度特征）
│   └── fpn_decoder.py           # 独立双 FPN 解码头（seg_fpn + boundary_fpn）
├── utils/
│   ├── loss.py                  # BoundaryLoss（语义 BCE + 边界 Focal×EDT 权重）
│   ├── loss_semi.py             # 半监督一致性损失 + EMA 更新 + 骨架过滤
│   ├── metrics.py               # SegMetrics 评估（mIoU / mDice / Boundary IoU）
│   ├── post_process.py          # 骨架化 / 受阻分水岭 / 实例 ID
│   └── progressive_aug.py       # 渐进式外观增强（学生输入专用）
├── tools/
│   ├── preprocess_labels.py     # 离线边界净化 GT 生成
│   └── precompute_pseudo_labels.py  # Stage-1 边界伪标签离线预计算（TTA + 质量报告）
├── weights/                     # 本地权重（sam2_hiera_base_plus.pt）
├── segment-anything-2/          # SAM 2 源码（本地仓库）
├── train.py                     # Stage 1 训练入口
├── train_stage2.py              # Stage 2 半监督训练入口
├── inference.py                 # 推理入口
├── debug_iou.py                 # 零 epoch IoU 硬审计
├── debug_pipeline.py            # 数据管线诊断（letterbox / 边界权重 / 受阻分水岭）
├── test_skeleton_watershed.py   # 骨架 + 分水岭纯图像验证
└── visualize_instances.py       # 实例图着色可视化
```

## 训练

### Stage 1 全监督

```bash
conda activate sam2_env
python train.py --config config/default_config.yaml
```

恢复训练：

```bash
python train.py --config config/default_config.yaml --resume outputs/stage1/checkpoint_epoch50.pth
```

### Stage 2 半监督微调

```bash
# 推荐：预计算 Stage-1 边界伪标签缓存（stage1_direct 模式使用，
# 免去每 step 的 ref_model 前向，并剔除无边界响应的低质量图）
python tools/precompute_pseudo_labels.py --config config/default_config.yaml

python train_stage2.py --config config/default_config.yaml
```

当前 Stage 2 采用 `boundary_teacher_mode=stage1_direct` + 冻结语义分支
(`freeze.seg_branch: true`)：语义通道低频块状信息由 EMA 形式充分训练后冻结，
训练集中优化边界头；边界一致性损失含正样本加权与边界-背景 margin 项，
用于拉开边界输出区间。每 epoch 会打印 `bnd_output: max/>0.5占比/gap`，
用于观察微调是否成功泛化。

`boundary_teacher_mode` 可选四种：`ema`（EMA 教师+锚点混合）、
`stage1_direct`（Stage-1 直接输出，当前默认）、`self_consistency`
（学生自一致性）、`anchor_self`（学生自一致性 + Stage-1 锚点混合，
用于突破 Stage-1 的 recall 天花板）。预测占比上限正则
（`rate_regularizer_weight`）默认随训练从 0.1 退火到 0.4，既压雾复现
又不会像骨架阈值退火那样把弱边界从目标中剔除。

从 checkpoint 恢复：

```bash
python train_stage2.py --config config/default_config.yaml --resume outputs/stage2/stage2_epoch30.pth
```

分支切换（仅加载 decoder 权重，重置优化器/调度器/epoch，用于单独训练语义/边界头后切换）：

```bash
python train_stage2.py --config config/default_config.yaml --init_from_checkpoint outputs/stage2/best_model_stage2.pth
```

## 推理

```bash
conda activate sam2_env
python inference.py --config config/default_config.yaml --checkpoint outputs/stage2/best_model_stage2.pth
```

指定测试目录与输出目录：

```bash
python inference.py --config config/default_config.yaml --test_dir data/test --output_dir outputs/inference
```

推理增强选项（配置于 `inference` 段）：

- **双阈值滞后二值化**：`boundary_threshold: 0.4`（弱阈值）+ `boundary_threshold_high: 0.6`（强阈值），弱真实边界与强边界连通时保留、孤立噪声剔除，缓解欠分割；
- **边界 logits 缩放**：`boundary_logit_scale > 1` 增强弱边界响应；
- **推理 TTA**：`python inference.py --tta`（hflip/vflip/rot180 logits 平均），提升弱边界召回与稳定性。

不指定 `--checkpoint` 时，按配置 `inference.checkpoint_stage`（stage1/stage2）选择默认权重（当前默认指向 `outputs/stage2/stage2_epoch30.pth`）。

## 输出文件

推理后在输出目录生成：

- `{basename}_inst.png` : 单通道 uint8 实例图（1~255，按面积降序编号）
- `{basename}_class.json` : `{"实例ID": 类别标签}` 映射（0=珠光体，1=铁素体）
- `{basename}_mask.png` : 语义掩码可视化（`post_process.save_visualization=true` 时）
- `{basename}_boundary.png` : 边界概率热力图

## 技术约束

| 约束 | 说明 |
|------|------|
| 参数量 < 500M | 冻结 encoder 约 80M + 解码头约 10.8M，满足约束 |
| 零预训练解码器 | FPN 全随机初始化，禁止加载 SAM 2 原生 Mask Decoder 权重 |
| 本地化隔离 | 权重存放于 `weights/`，禁止全局缓存 |
| 长宽比保真 | Letterbox 等比缩放 + BORDER_REFLECT 镜像填充，禁止挤压变形 |

## 类别定义

| 通道 | ID | 名称 | 说明 |
|------|----|------|------|
| 语义 | 0 | pearlite | 珠光体 |
| 语义 | 1 | ferrite | 铁素体 |
| 边界 | 1 | grain_boundary | 晶界（独立通道二值预测） |
