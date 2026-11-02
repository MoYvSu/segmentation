# 金相图像实例分割：训练与推理

| 提交信息 | 内容 |
|---|---|
| 参赛团队编号 | AIC-2026-59388388 |
| 团队名称 | 426队 |
| 队长姓名 | 苏宸锐 |
| 技术方案PDF | AIC-2026-59388388-技术报告.pdf |
| 官方回分 | 82.68 |

## 目录

```text
提交根目录/
├─ AIC-2026-59388388-模型文件/
│  ├─ engines/                 五个TensorRT引擎及engines.json
│  ├─ weights/                 backend.pt、restorer.pt、naf.pt
│  └─ pretrained/              sam2_hiera_base_plus.pt
├─ AIC-2026-59388388-代码与数据/
│  ├─ infer.py                 推理入口
│  ├─ train.py                 训练入口
│  ├─ build_engines.py         训练权重转引擎
│  ├─ README.md
│  ├─ CHECKS.md                训练、推理和构建检查清单
│  ├─ requirements.txt         推理依赖
│  ├─ inference/               引擎运行、图像处理和实例后处理
│  ├─ training/                训练编排、配方、模型与数据加载代码
│  └─ data/                    训练原图与训练标注
└─ AIC-2026-59388388-赛题结果文档/
   └─ results/                 实例PNG与类别JSON
```

根目录命名格式：`AIC-2026-59388388-团队名称-半决赛-队长姓名`。

所有命令在`AIC-2026-59388388-代码与数据`目录执行。代码中的默认路径以脚本位置解析。

## 环境

已使用的环境：Python 3.12.3、PyTorch 2.8.0+cu128、CUDA 12.8、TensorRT 10.12.0.36、OpenCV 5.0.0、RTX 4080 SUPER。

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

训练与引擎重建增加以下依赖：

```bash
python -m pip install -r training/requirements.txt
```

## 推理

```bash
python infer.py --input-dir /path/to/images --output-dir /path/to/predictions
```

`--input-dir`接受图片目录或一张图片。默认引擎目录为`../AIC-2026-59388388-模型文件/engines`，也可显式选择：

```bash
python infer.py --input-dir /path/to/images --output-dir /path/to/predictions --engine-dir ../AIC-2026-59388388-模型文件/engines
```

引擎已包含模型权重。推理直接读取五个engine，图像张量、插值和概率计算使用PyTorch，实例处理使用NumPy、OpenCV和scikit-image。

TensorRT报告引擎与当前操作系统、GPU或TensorRT版本不兼容时，`infer.py`会自动调用随包`build_engines.py`，使用`../AIC-2026-59388388-模型文件/weights`中的三份权重重建五个FP32引擎，随后继续本次推理。旧推理进程先退出，构建进程结束后再重新开始推理，避免两者同时占用显存。

自动构建需要安装上方`training/requirements.txt`中的依赖。新引擎保存在所选引擎目录的`local/`中，之后优先复用；使用`--engine-manifest`时，保存在清单所在目录的`local/`中。原有引擎保留。每次启动最多自动重建一次。

RTX 5060 Laptop 8GB上的重建约需7分钟。覆盖三种尺寸的三张样本相对RTX 4080 SUPER结果，F/P翻转率为0，442个实例的数量及对应类别不变；实验中队对三张图像进行推理测试，只有一张图有少量边界像素变化，另外两张完全一致。同一引擎重复推理结果完全一致。

使用其他训练权重对应的引擎时，同时通过`--model-dir`指定重建来源，例如：

```bash
python infer.py --input-dir /path/to/images --output-dir /path/to/predictions --engine-dir runs/train_01/engines --model-dir runs/train_01/models
```

D5a负责图像恢复，语义头判断珠光体/铁素体，affinity头判断邻近像素是否属于同一实例，NAF整理拼接后的边界。TensorRT引擎是针对目标GPU构建的可执行模型。

推理流程：

1. 原图等比缩放、reflect补齐为1024输入。
2. D5a零噪声首步恢复，全图恢复语义提供分水岭底图；原图全图语义提供最终类别概率。
3. 原生1024滑窗，重叠25%；每窗经过D5a、共享编码器和affinity头，按原坐标加权拼接边界概率。
4. 拼接边界经过细化与8像素椭圆膨胀，形成NAF输入；NAF按256窗口、64像素重叠、32像素reflect边缘处理，满批16块并保留尾批。
5. 使用原种子规则，按原边界证据筛选分水岭种子（S0）并生成实例，以原图语义概率逐实例投票，再执行200像素末端邻居合并。
6. 保存与原图同尺寸的实例PNG和类别JSON。

输出：

```text
predictions/
├─ results/
│  ├─ 图名_inst.png
│  └─ 图名_class.json
└─ summary.json
```

实例图为单通道uint16，正实例ID为1至65535；JSON中珠光体为0、铁素体为1。`summary.json`记录处理数量、耗时和引擎调用数。

`AIC-2026-59388388-赛题结果文档/results`保存150张测试图的全模块TensorRT FP32加速主线结果，包含150份实例PNG和150份类别JSON。该全加速版本已确认取得当前最高分。

## 训练数据

| 路径 | 内容 |
|---|---|
| `data/unlabeled` | 1000张赛方训练原图 |
| `data/raw` | 32张人工训练原图及LabelMe标注 |
| `data/purified_gt` | 32源早期语义与边界目标 |
| `data/completed_gt` | 32源r8实例目标、类别表和数据清单 |
| `data/sam2_geometry` | 64源SAM2无类别几何目标 |

## 训练

查看阶段及检查输入：

```bash
python train.py --plan
python train.py --check
```

从官方SAM2底座开始训练：

```bash
python train.py --config training/config.yaml --output-dir runs/train_01
```

以下顺序与`train.py`的实际执行顺序一致，各阶段完成后执行下一阶段。

| 顺序 | 阶段 | 输入与初始化 | 主要产物 |
|---|---|---|---|
| 1 | LoRA SSL | 官方SAM2主干、1000张无标签训练图；训练LoRA与随机重建头 | `lora_ssl/lora_state_dict.pth` |
| 2 | Stage1 | SSL LoRA、随机任务decoder及人工训练目标 | `stage1/best_model.pth` |
| 3 | Stage1边界缓存 | 冻结本轮Stage1，处理1000张训练原图 | `stage1_boundary/boundary_probs.npy` |
| 4 | joint | 本轮Stage1；联合使用人工监督与无标签边界一致性 | `joint/best_model_stage2.pth` |
| 5 | S-align语义头 | 本轮joint作为固定教师；随机重置学生语义头，冻结LoRA与边界头 | `semantic/best_model_stage2.pth` |
| 6 | V affinity头 | 本轮joint的边界FPN初始化；冻结编码器、LoRA与语义头 | `affinity/best_affinity.pth` |
| 7 | D5a | 随机初始化恢复器，使用1000张无标签训练图 | `d5a/epoch_060.pt` |
| 8 | R1 affinity适应 | 组装joint共享编码器、S-align语义头、V affinity头与冻结D5a；只更新affinity | `r1/final.pt` |
| 9 | backend/restorer导出 | 本轮R1与D5a，严格导出FP32状态 | `models/backend.pt`、`models/restorer.pt` |
| 10 | 零噪N1训练缓存 | 用本轮backend/restorer对32个人工训练源作局部零噪推理 | `n1_cache/manifest.json`及逐图缓存 |
| 11 | NAF训练 | 随机初始化生成器G与判别器D，使用真实N1缓存及对应GT | `naf_train/latest.pt` |
| 12 | NAF导出 | 严格加载本轮NAF，保存生成器G | `models/naf.pt` |
| 13 | 三模型汇总 | 核对本轮三份导出权重并汇总文件信息 | `training_export.json` |

官方预训练权重仅载入Hiera编码器主干；初始任务decoder采用随机初始化。S-align使用本轮joint固定教师并重置学生语义头，V affinity继承joint边界FPN，R1只更新affinity头。训练使用包内训练原图与标注，过程缩略图保存在对应训练输出目录。

LoRA是在固定编码器中加入少量可训练参数；SSL用无标签训练图进行自监督预训练。joint联合训练语义与边界，S-align调整语义头，R1让affinity适应恢复图，N1是零噪局部推理产生的训练缓存。

配置入口为`training/config.yaml`，展开的各阶段配方保存在`training/recipes`。

| 阶段 | 默认预算 | 主要学习率 |
|---|---|---|
| LoRA SSL | 30轮，batch2，rank16/alpha32，mask_ratio参数0.4 | 1e-4 |
| Stage1 | 100轮，batch8，512裁块 | 1.5e-4 |
| joint | 100轮，人工batch4＋无标签batch4 | 5e-5 |
| S-align | 20轮，每轮62更新 | 1e-4 |
| V affinity | 120轮，每轮26更新 | 头1e-4，FPN5e-5 |
| D5a | 60轮，batch4，512裁块，16步/kappa0.03 | 2e-4至2e-6 |
| R1 | 20轮，每轮64更新，人工与SAM两流顺序反向 | affinity 2e-5至2e-6 |
| NAF | 1000更新，batch16，256裁块 | G 1e-3，D 1e-4 |

R1训练采用随机首步恢复增强，部署推理采用零噪声首步恢复。完整损失、冻结范围、采样及学习率日程见[训练配方](training/RECIPE.md)。

训练结果：

```text
runs/train_01/
├─ models/
│  ├─ backend.pt
│  ├─ restorer.pt
│  └─ naf.pt
├─ training_export.json
├─ logs/
└─ 各阶段输出/
```

## 将新训练权重用于加速推理

在目标GPU环境构建引擎：

```bash
python build_engines.py --model-dir runs/train_01/models --output-dir runs/train_01/engines
python infer.py --input-dir /path/to/images --output-dir /path/to/predictions --engine-dir runs/train_01/engines --model-dir runs/train_01/models
```

构建采用FP32，关闭TF32、FP16和INT8。编码器、语义头、affinity头和D5a固定1024输入、batch1；NAF输入为256，batch范围1至16。构建结束保存五个engine及`engines.json`，ONNX作为构建过程的临时文件。

## 校验与记录

训练、推理、引擎构建的保留检查逐项列于[CHECKS.md](CHECKS.md)。运行日志和摘要写入指定输出目录。
