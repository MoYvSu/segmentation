# 低碳钢金相图像相区分割

> 当前黑盒最佳：单一 E10a 语义解码器 + G4b `high=0.65` 封边/受阻分水岭，mIoU
> `0.8381`、铁素体面积项 `0.8408`、总分 `83.94`。E9 保留为历史回退，不做连续融合。
> 当前状态、入口与产物约定以 [docs/PIPELINE.md](docs/PIPELINE.md) 为准。

颜色先验分析已记录在 [docs/COLOR_SEPARABILITY.md](docs/COLOR_SEPARABILITY.md)。后续对话进程在
讨论语义阈值、颜色辅助或数据增强前应先阅读该文档。

## 当前进度

- **语义锚点与参考基线**：`outputs/stage2_v6/best_model_stage2.pth`。其语义分支仍是当前
  可复现锚点，原边界分支作为回退基线。
- **当前几何基线**：G4b affinity + `high=0.65` + 封边/受阻分水岭。相较 G3，G4b 是当前
  更稳定的部署基线；测试集仍以欠分割和局部合并为主要风险，尚未证明黑盒竞赛分数提升。
  详见 [docs/AFFINITY_DEPLOYMENT_EVALUATION.md](docs/AFFINITY_DEPLOYMENT_EVALUATION.md)。
- **当前语义选择**：E10a 黑盒结果为 mIoU `0.8381`、铁素体平均面积项 `0.8408`、总分
  `83.94`，已取代 E9 成为单模型主线；E9 的 `0.8421/0.7917/81.69` 仅作稳定历史回退。
  详见 [docs/SEMANTIC_EXPERIMENT_E10A_20260828.md](docs/SEMANTIC_EXPERIMENT_E10A_20260828.md)。
- **统一部署模型**：`outputs/deployment/e10a_g4b_fused.pth` 把共享 SAM2+LoRA encoder、
  E10a semantic decoder 与 G4b affinity decoder 打包为一个 81.67M 参数模型；加载时不再依赖
  三份源 checkpoint，`test_009` 与原管线最终实例图/类别 JSON 字节级一致。
- **方向图切分结论**：graph-v1 `area200` 黑盒为 `0.8268/0.8365/83.17`，未超过 E10a
  watershed；graph-v2 `area150` 目检出现不自然的笔直边界，已停止晋级，不再提交。详见
  [docs/AFFINITY_GRAPH_AB_20260828.md](docs/AFFINITY_GRAPH_AB_20260828.md)。
- **中心热图实验已降级为负面对照**：共享 `boundary_fpn` 的中心辅助任务破坏了边界表征；
  `outputs/stage2_center_heatmap/best_model_stage2.pth` 不作为后续初始化主线。
- **当前目标**：G7 冻结 E10a、V6 LoRA 与 G4b affinity 基座，只在 1024 网格训练四个
  短程 affinity 的零初始化残差；SAM2 未覆盖区 ignore，伪实例跨 mask 负边降权。现有
  `high=0.65`、seal2 与分水岭不变。详见
  [docs/AFFINITY_G7_HIGHRES_SHORT_20260829.md](docs/AFFINITY_G7_HIGHRES_SHORT_20260829.md)。
- **审计结论（2026-08-27）**：G3 的验证提升尚不能等同于竞赛增益；G3 相比 G2 同时改变了
  native crop、采样、训练时长、学习率和外观增强，需在统一部署评估链上做单变量复验。详见
  [docs/AFFINITY_DEPLOYMENT_EVALUATION.md](docs/AFFINITY_DEPLOYMENT_EVALUATION.md)。

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
│   ├── loss.py                  # BoundaryLoss（语义/实例核心 + 边界 Focal×EDT）
│   ├── loss_semi.py             # 半监督一致性损失 + EMA 更新 + 骨架过滤
│   ├── semantic_training.py     # E7b 实例等权核心损失 + target-aware 暗边增强
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
├── debug_pipeline.py            # 数据管线诊断（letterbox / 边界权重 / 受阻分水岭）
├── test_skeleton_watershed.py   # 骨架 + 分水岭纯图像验证
└── visualize_instances.py       # 实例图着色可视化
```

## 当前训练与基线

G4b high0.65 + V6 语义的黑盒结果为总分 `80.67`、实例 mIoU `0.8441`、铁素体平均面积项
`0.7693`。E9 进一步得到 `0.8421/0.7917/81.69`；E10a 最终得到
`0.8381/0.8408/83.94`，以显著面积收益成为当前黑盒最佳。E7b-A 已完成验证但不晋级：它从 V6
初始化只更新语义 decoder，训练期 semantic loss 虽下降；严格同部署口径下，代理分数由
当前 hard 基线的 `79.1800` 小幅降至 `78.8379`（阈值 `0.65`）。详见
[docs/SEMANTIC_TRAINING_E7B_20260827.md](docs/SEMANTIC_TRAINING_E7B_20260827.md)。

E7c 不丢弃 E7b 的局部纠错能力，而是让 V6 固定实例几何与默认类别，仅允许 E7b
通过门控修正低置信/黑边实例。缓存扫参已选择 E7c-R 进入目检：六图验证仅翻转 1 个且
新增一个正确铁素体匹配，68 张测试图翻转 67/6178 个实例；不设置实例面积门槛。详见
[docs/SEMANTIC_EXPERIMENT_E7C_20260828.md](docs/SEMANTIC_EXPERIMENT_E7C_20260828.md)。

E8 已证明低分辨率残差能够产生局部纠错，但 `+/-2` logit 很快饱和且验证 mIoU 未提升，故不
晋级。E9 保留 V6 零漂移锚点，把可学习决策提高到输入分辨率，并增加与实例投票一致的
整实例概率损失；epoch 20 probability-mean 已通过黑盒验证并保留为稳定回退。详见
[docs/SEMANTIC_EXPERIMENT_E9_20260828.md](docs/SEMANTIC_EXPERIMENT_E9_20260828.md)。

当前 E10a 冻结 V6 LoRA、边界和 G4b affinity，随机重置完整 semantic FPN/head 与高分辨率
解码路径；固定 V6 只在无标签高置信区域提供衰减蒸馏。E10a 已获黑盒总分 `83.94`，不与
E9 连续融合；标准几何仍固定为 G4b watershed。详见
[docs/SEMANTIC_EXPERIMENT_E10A_20260828.md](docs/SEMANTIC_EXPERIMENT_E10A_20260828.md)。

如需复现实验，按既定约定可直接在训练服务器运行：

```bash
conda activate sam2_env
python train_stage2.py --config config/train/stage2_semantic_e7b_decoder20.yaml
```

E9 当前训练命令：

```bash
python train_stage2.py --config config/train/stage2_semantic_e9_highres20.yaml
```

E10a 冷启动完整语义解码器：

```bash
python train_stage2.py --config config/train/stage2_semantic_e10a_cold20.yaml
```

G7 高分辨率短程 affinity 残差：

```bash
python train_affinity_geometry_g1.py \
  --config config/train/affinity_geometry_g7_highres_short.yaml
```

E9 输出目录为 `outputs/stage2_semantic_e9_highres20/`。训练固定 G4b geometry、V6 decoder
与 LoRA，仅更新零初始化 high-resolution semantic residual；checkpoint 按验证 semantic mIoU 选择，无标签 holdout monitor 只检查
泛化稳定性。历史 Stage 1、B2、affinity 训练命令见 [docs/PIPELINE.md](docs/PIPELINE.md) 和
`config/README.md`。

## 推理

```bash
conda activate sam2_env
python tools/run_affinity_submission.py \
  --config config/inference/final_affinity_g4b_high065.yaml
```

E7b 训练完成后的固定几何对照：

```bash
python tools/run_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_e7b.yaml
```

当前 E7c-R 类别纠错目检候选（实例几何仍与 G4b 完全一致）：

```bash
python tools/run_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_dual_e7c_relaxed.yaml
```

E9 训练完成后的整实例概率聚合：

```bash
python tools/run_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_e9_highres.yaml
```

E10a 训练完成后的固定 V6/G4b 几何 challenger：

```bash
python tools/run_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml
```

推荐的单主干部署方式：

```bash
# 只需在源 checkpoint 更新后重新生成一次组合包
python tools/build_fused_deployment_checkpoint.py

# 日常推理仅加载一个组合 checkpoint
python tools/run_fused_affinity_submission.py \
  --checkpoint outputs/deployment/e10a_g4b_fused.pth
```

组合包保存完整 encoder 权重及两个任务分支，约 327 MB；源 checkpoint 路径与 SHA256 仍写入
包内供审计，但不是推理依赖。禁止用参数平均代替分支拼接。

E7b 整体替换配置只替换 semantic decoder；E7c-R 则只允许 E7b 改实例类别。两者都会验证
E7b 与 V6 的 LoRA 张量逐字节一致；若 LoRA 发生变化，将拒绝复用 G4b affinity checkpoint。V6 双头历史推理仍可使用
`python inference.py --config config/inference/v6_reference.yaml`。

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
