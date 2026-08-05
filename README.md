# 低碳钢金相图像相区分割

> 技术路线：**冻结 SAM 2 Image Encoder + 自制独立双 FPN 解码头（语义/边界双分支）+ 离线边界净化 GT + 两阶段训练（Stage 1 全监督 → Stage 2 半监督）+ 边界骨架化 + 受阻分水岭实例分割 + LoRA 低秩适配 trunk（协议 C：自监督 LoRA 预训练 → Stage-1(LoRA) → 联合微调）**。

## 当前进度

- **协议 C（LoRA 全链路）已跑通**：自监督 LoRA 预训练（1000 无标签，MAE 掩码重建）→
  Stage-1 监督（LoRA）→ Stage-2 联合微调（LoRA）。当前最优
  `outputs/stage2_lora/best_model_stage2.pth`：**val mIoU 0.8315 / bndIoU 0.5009**
  （对比 v4.0 基线的 0.7811 / 0.4082，语义 +0.050、边界 +0.093）。
- **语义头已接近直接可用**：test 域 `>0.8` 高置信 72%、`0.4-0.6` 模糊带 2.9%、
  掩码碎片化约 35 个连通域（v4.0 时代 86~112）；实例类别由分水岭 + 语义投票产生。
- **边界头**：bndIoU 0.5009 为历史最高，但 test 目检仍有欠分割（边界高概率平台窄，
  `>0.7` 仅 ~1%），推理端调参中（`boundary_logit_scale` 1.3~1.8、阈值 0.35 等）。
- **v4.0 实例分类器已废弃**：语义头已足够，不再需要实例级分类器
  （git 标签 `v4.0-instance-clf` 可回溯历史实现）。

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
├── inference.py                 # 推理入口（语义投票实例分类）
├── debug_iou.py                 # 零 epoch IoU 硬审计
├── debug_pipeline.py            # 数据管线诊断（letterbox / 边界权重 / 受阻分水岭）
├── test_skeleton_watershed.py   # 骨架 + 分水岭纯图像验证
└── visualize_instances.py       # 实例图着色可视化
```

## 训练（协议 C：LoRA 全链路，当前主线）

> 架构：**自监督 LoRA 预训练 → Stage-1 监督（LoRA）→ Stage-2 联合微调（LoRA）**。
> 文献：LoRA (2021) / Conv-LoRA (2024) / TopoLoRA-SAM (2026)。trunk 梯度检查点已启用。

**① 自监督 LoRA 预训练**（1000 张无标签，MAE 掩码重建，只训 LoRA + 重建头）：

```bash
python tools/pretrain_lora_ssl.py --config config/default_config.yaml --epochs 30 --batch_size 8
```

产物：`outputs/lora_pretrain/lora_state_dict.pth`（Stage-1 的 LoRA 初始化）。

**② Stage-1 监督**（冻结 trunk + LoRA，双 FPN 随机初始化，语义/边界一起学）：

```bash
python train.py --config config/stage1_lora.yaml
```

要点：`lora.enabled: true` + `lora.init_from` 加载预训练状态；`lora.lr_ratio: 0.5`
（26 张有标签图不宜过大）；batch 8（trunk 梯度检查点控显存）。

**③ Stage-2 联合微调**（双分支联合 + LoRA，有标签监督 + 无标签边界一致性）：

```bash
python tools/precompute_pseudo_labels.py --config config/default_config.yaml   # 边界伪标签缓存（一次性）
python train_stage2.py --config config/stage2_lora.yaml \
    --init_from_checkpoint outputs/stage1_lora/best_model.pth --phase joint --tag lora
```

要点：`freeze.seg_branch / boundary_branch` 均 false（联合）；`unsup_seg_weight=0`
（语义只走监督）、`unsup_weight=0.3`（边界一致性，stage1_direct 缓存伪标签）；
LoRA 随训练继续适配，两个头与特征共同收敛，避免"特征动了、头冻结"的漂移。

**当前最优**：`outputs/stage2_lora/best_model_stage2.pth`（val mIoU 0.8315 / bndIoU 0.5009）。

### 半监督机制速览（Stage-2 可选项）

- `boundary_teacher_mode`：`stage1_direct`（当前，Stage-1 伪标签缓存）/ `ema`
  （EMA 教师+锚点混合）/ `self_consistency`（学生自一致性）/ `anchor_self`
  （自一致性 + Stage-1 锚点，可突破 recall 天花板）。
- 边界一致性损失：MSE + Sobel 梯度 + 各向异性 TV + 背景抑制 + margin + 占比上限正则
  （`rate_regularizer_weight` 0.1→0.4 退火）。
- 每 epoch 打印 `bnd_output: max/>0.5占比/gap`；每 5 epoch 输出 test 语义置信度
  （`>0.8` 占比 / `0.4-0.6` 模糊带）与监控图。
- 运行记录：`outputs/runs/<时间戳>_<phase>_<tag>/`
  （run_info.json / config_snapshot.yaml / metrics.csv / best_model.pth）。

恢复训练：

```bash
python train.py --config config/stage1_lora.yaml --resume outputs/stage1_lora/checkpoint_epoch50.pth
python train_stage2.py --config config/stage2_lora.yaml --resume outputs/stage2_lora/stage2_epoch30.pth
```

> 历史：v4.0 时代的"两阶段协议（Phase S 冻结边界 / Phase B 冻结语义）"与实例分类器
> 管线已废弃（见 git 标签 `v4.0-instance-clf`）；当前统一为协议 C 联合训练。

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

- **单阈值二值化**：`boundary_threshold: 0.5`（已移除 Canny 式滞后——边界概率在邻域连续，滞后会把强脊坡脚也纳入，导致边界带宽度沿脊线变化、轮廓崎岖）；
- **边界 logits 缩放**：`boundary_logit_scale > 1` 增强弱边界响应；
- **推理 TTA**：`python inference.py --tta`（hflip/vflip/rot180 logits 平均），提升弱边界召回与稳定性。

不指定 `--checkpoint` 时，按配置 `inference.checkpoint_stage`（stage1/stage2）选择默认权重（当前默认指向 `outputs/stage2/stage2_epoch30.pth`）。

## 实例级分类器（已废弃）

> v4.0 实例分类器推理管线（对照实验曾证实优于语义投票）已在协议 C（LoRA）落地后废弃：
> 当前语义头在 LoRA 特征上已接近直接可用，实例类别由分水岭 + 语义投票产生。
> 历史实现见 git 标签 `v4.0-instance-clf`。

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
