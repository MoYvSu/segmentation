# Agent.md: 《低碳钢金相图像无监督相区分割》项目架构搭建指南
python环境依赖: sam2_env 采用 conda activate sam2_env
## 一、 核心目标
在当前根目录下，围绕“冻结 SAM 2 Image Encoder + 自制三分类轻量 FPN 解码头 + 在线 Letterbox 管道 + 拓扑剥离后处理”的技术路线，搭建一个模块化的半监督闭环项目系统。

## 二、 刚性约束
1. **参数量限制**：总参数量必须低于 500M。
2. **零预训练权重解码器**：禁止调用任何 SAM 2 原生的 Mask Decoder 权重。自制解码头必须完全随机初始化。
3. **本地化隔离**：项目运行所需的任何第三方权重文件（包括 SAM 2 底座权重），必须下载并存放于项目文件夹内部的 `weights/` 目录中，严禁依赖系统的 `~/.cache/` 等全局隐式路径。
4. **长宽比保真**：禁止在数据读取时对原始图像进行强行挤压变形缩放。

## 三、 项目树架构设计
请在根目录下建立对应的文件和文件夹：

project/
├── config/
│   └── default_config.yaml      # 全局超参数（路径、学习率、权重、模型版本等）
├── data/
│   ├── raw/                      # 存放原始图像与同名的 labelme .json 文件
│   ├── dataset.py                # 在线数据管道（Letterbox、多边形转三分类掩码、形态学晶界剥离）
│   └── active_learning.py        # 伪标签生成与反向转 labelme .json 矢量网关
├── models/
│   ├── __init__.py
│   ├── sam2_encoder.py           # 显式冻结的 SAM 2 Image Encoder 提取器
│   └── fpn_decoder.py            # 自研轻量级随机初始化特征金字塔解码头
├── utils/
│   ├── metrics.py                # 评估指标监控
│   └── post_process.py           # 原图自适应尺寸特征还原与拓扑连通域实例剥离
├── weights/                      # 存放下载的本地权重（如 sam2_hiera_base_plus.pt）
├── train.py                      # 微调训练主入口
├── inference.py                  # 测试集交卷预测主入口
└── README.md                     # 环境配置与运行说明

## 四、 各核心模块构建需求

### 1. 模型加载模块 (`models/`)
* **`sam2_encoder.py`**：加载指定的 `sam2_hiera_base_plus`（或 `sam2_hiera_large`）。显式设置 `requires_grad=False`。重写前向流，使其返回 Stage 1 至 Stage 4 的多尺度特征图。
* **`fpn_decoder.py`**：编写轻量级全卷积 FPN 头。通过 1*1 卷积将提取的四个尺度特征统一对齐到相同通道数（如 128 或 256），自上而下通过双线性插值融合，最后输出通道数固定为 3（三分类：0=珠光体, 1=铁素体核, 2=晶界）。
* **参数计数器**：在模型初始化完毕后，必须使用 `sum(p.numel() for p in model.parameters())` 确认总参数量小于 500M。

### 2. 在线数据管道 (`data/dataset.py`)
* **在线处理**：通过 PyTorch `Dataset` 在内存中在线处理图像，禁止离线改图。
* **在线 Letterbox 变换**：将任意非标准分辨率图像的长边等比例缩放至 1024，短边按相同比例缩放后，在右侧/下方利用 0 像素补齐（Padding）到 1024*1024。
* **多边形在线转三分类**：读取 Labelme 的 json 文件。将 ferrite 标签在内存中转为掩码。对该掩码使用 `cv2.erode` 向向内收缩 1~2 像素得到铁素体核心（类别1）；原掩码减去核心得到铁素体晶界（类别2）；将 pearlite 多边形内部设为珠光体（类别0）。

### 3. 主动学习与反向网关 (`data/active_learning.py`)
* **不确定性采样**：编写基于信息熵或边界响应方差的采样逻辑，记录筛选记录。
* **Mask to JSON 矢量反向网关**：编写 `mask_to_labelme_json` 函数。利用 `cv2.findContours` 将模型推理出的原图尺寸三分类掩码转换为有序坐标点，遵循 Labelme 的官方 JSON Schema 写出为矢量文件，用于人工微调。

### 4. 自适应尺寸还原与后处理还原 (`utils/post_process.py`)
* **网络内部动态上采样**：在推理时记录测试图原始尺寸 (H, W)。全卷积特征流通过解码头后，在最后一层使用 `F.interpolate(..., size=(H, W), mode='bilinear', align_corners=True)` 动态直接对齐原图尺寸。随后进行 `torch.argmax` 消除过渡带模糊。
* **拓扑剥离与全局 ID 分配**：
  * 对铁素体核（类别1）运行 `cv2.connectedComponentsWithStats`，切开被“晶界（类别2）”阻断的粘连晶粒，分派唯一 ID。
  * 对珠光体区域（类别0）独立运行连通域分析，将分离的珠光体团簇切分成独立实例并赋予唯一 ID。
  * 过滤面积小于 50 像素的噪点。所有实例统一共享 1~255 的整型 ID 编号并按面积降序排列写入单通道 uint8 图像 `_inst.png`。同步将 `{"实例ID": 类别标签}`（1=铁素体，0=珠光体）写入对应的 `_class.json`。

## 五、 执行步骤
1. 建立上述所有文件夹及空白脚本。
2. 优先编写并完善 `config/default_config.yaml` 和 `data/dataset.py` 模块。
3. 提示用户放入第一张金相分割原始图像与标注进行管道测试。