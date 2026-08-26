# 低碳钢金相图像相区分割

> 当前候选路线：冻结 V6 语义锚点 + 8 通道 affinity geometry + SAM2 无类别伪实例监督
> → affinity boundary → 受阻分水岭 → 语义投票分类。V6/B2 边界路线保留为基线与回退。
> 当前状态、入口与产物约定以 [docs/PIPELINE.md](docs/PIPELINE.md) 为准。

颜色先验分析已记录在 [docs/COLOR_SEPARABILITY.md](docs/COLOR_SEPARABILITY.md)。后续对话进程在
讨论语义阈值、颜色辅助或数据增强前应先阅读该文档。

## 当前进度

- **语义锚点与参考基线**：`outputs/stage2_v6/best_model_stage2.pth`。其语义分支仍是当前
  可复现锚点，原边界分支作为回退基线。
- **当前几何候选**：G3 native-crop affinity。完整部署复核显示小幅正信号，但测试图仍有
  明显欠分割，尚未证明黑盒竞赛分数提升。详见
  [docs/AFFINITY_DEPLOYMENT_EVALUATION.md](docs/AFFINITY_DEPLOYMENT_EVALUATION.md)。
- **中心热图实验已降级为负面对照**：共享 `boundary_fpn` 的中心辅助任务破坏了边界表征；
  `outputs/stage2_center_heatmap/best_model_stage2.pth` 不作为后续初始化主线。
- **当前目标**：先以完整部署路径闭环 checkpoint 选择，再改进 affinity 的断边召回与
  合并错误；Oracle GT 前景图重建指标仅作诊断，不得用于模型晋级。

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
│   ├── default_config.yaml      # 当前 V6 参考基线
│   ├── inference/               # 同架构推理配置
│   └── experiments/             # 改架构/训练目标的实验配置
├── data/
│   ├── raw/                     # 有标注图像 + Labelme .json
│   ├── purified_gt/             # 离线净化 GT（_gt.npz）
│   ├── unlabeled/               # 无标注图像（半监督）
│   ├── test/  smoketest/        # 测试 / 冒烟测试图像
│   ├── dataset.py               # 在线数据管道（letterbox / 边界权重 / BoundaryDataset）
│   ├── dataset_semi.py          # 半监督双流数据集（Labeled / Unlabeled）
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
│   ├── precompute_pseudo_labels.py  # Stage-1 边界伪标签离线预计算（TTA + 质量报告）
│   └── tmp_color_separability.py # 临时 GT 颜色可分性分析脚本（未接入训练）
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

## 训练

LoRA 自监督预训练仍可复用；新的 Stage 1 与 Stage 2 配置已分开，避免推理配置中的冻结规则
误用于从头训练：

```bash
python tools/pretrain_lora_ssl.py --config config/train/stage1_lora.yaml --epochs 30 --batch_size 8
python train.py --config config/train/stage1_lora.yaml
python train_stage2.py --config config/train/stage2_refine_v6.yaml \
    --phase boundary --tag refine_v6_b2
```

Stage 2 B2 以 V6 checkpoint 为 `base_checkpoint`，冻结语义分支和 LoRA；前 5 epoch
只训练零初始化高分辨率 refine head，随后以低学习率解冻 V6 coarse boundary 基座。
每次运行在 `outputs/runs/` 保存配置、环境和逐 epoch 指标；best 权重使用硬链接，避免重复占盘。
恢复训练必须使用相同架构，代码会在加载 optimizer 前执行严格检查。

## 推理

```bash
conda activate sam2_env
python inference.py --config config/inference/v6_reference.yaml
```

指定测试目录与输出目录：

```bash
python inference.py --config config/inference/v6_reference.yaml \
  --test_dir data/test --output_dir outputs/inference/v6_reference
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
