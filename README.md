# 低碳钢金相图像无监督相区分割

> 基于"冻结 SAM 2 Image Encoder + 自制三分类轻量 FPN 解码头 + 在线 Letterbox 管道 + 拓扑剥离后处理"技术路线的模块化分割系统。

## 环境配置

### Python 环境

```bash
conda activate sam2_env
```

### 依赖安装

```bash
pip install torch torchvision opencv-python numpy pyyaml hydra-core omegaconf iopath
```

### SAM 2 本地权重

将 `sam2_hiera_base_plus.pt` 下载并放入 `weights/` 目录：

```
weights/
└── sam2_hiera_base_plus.pt
```

> **约束**：严禁依赖 `~/.cache/` 等全局隐式路径，所有第三方权重必须存放在项目 `weights/` 目录。

## 项目结构

```
segmentationv2/
├── config/
│   └── default_config.yaml      # 全局超参数
├── data/
│   ├── raw/                     # 原始图像与同名 labelme .json
│   ├── dataset.py               # 在线数据管道
│   └── active_learning.py       # 伪标签生成与反向网关
├── models/
│   ├── __init__.py
│   ├── sam2_encoder.py          # 冻结 SAM 2 Image Encoder
│   └── fpn_decoder.py           # 轻量 FPN 解码头
├── utils/
│   ├── metrics.py               # 评估指标
│   └── post_process.py          # 尺寸还原与拓扑剥离
├── weights/                     # 本地权重
├── segment-anything-2/          # SAM 2 源码
├── train.py                     # 训练入口
├── inference.py                 # 推理入口
└── README.md
```

## 数据准备

将原始金相图像与同名的 Labelme `.json` 标注放入 `data/raw/`：

```
data/raw/
├── sample_001.jpg
├── sample_001.json
├── sample_002.png
├── sample_002.json
└── ...
```

Labelme JSON 中 `label` 字段使用 `ferrite` 和 `pearlite` 标注。

## 训练

```bash
conda activate sam2_env
python train.py --config config/default_config.yaml
```

恢复训练：
```bash
python train.py --resume outputs/checkpoint_epoch50.pth
```

## 推理

```bash
conda activate sam2_env
python inference.py --config config/default_config.yaml --checkpoint outputs/best_model.pth
```

指定测试目录：
```bash
python inference.py --test_dir data/test --output_dir outputs/inference
```

## 输出文件

推理后在输出目录生成：
- `{basename}_inst.png` : 单通道 uint8 实例图 (1~255)
- `{basename}_class.json` : `{"实例ID": 类别标签}` 映射
- `{basename}_mask.png` : 三分类可视化掩码

## 技术约束

| 约束 | 说明 |
|------|------|
| 参数量 < 500M | 总参数量严格低于 500M |
| 零预训练解码器 | 禁止加载 SAM 2 原生 Mask Decoder 权重 |
| 本地化隔离 | 权重存放于 `weights/`，禁止全局缓存 |
| 长宽比保真 | Letterbox 等比缩放，禁止挤压变形 |

## 类别定义

| ID | 名称 | 说明 |
|----|------|------|
| 0 | pearlite | 珠光体 |
| 1 | ferrite_core | 铁素体核 |
| 2 | grain_boundary | 晶界 |