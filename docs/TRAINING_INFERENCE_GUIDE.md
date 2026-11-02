# 训练、推理与引擎构建使用指南

2026-11-01研究工程指南，2026-11-02补充最终源码入口。最终独立训练、推理与引擎构建方法见[提交源码README](../submission/AIC-2026-59388388-代码与数据/README.md)。本文保留研究仓库的使用方法；各次实验的配置、成绩和来源见对应专题文档。下述命令从仓库根目录运行，示例路径替换为实际私有资产路径。

## 当前入口与证据

| 工作 | 入口 | 当前边界 |
|---|---|---|
| 训练计划、检查、分阶段执行 | `train_pipeline.py` → `repro/pipeline.py` | 默认原GT配方13阶段；不等于原C历史最优权重的精确重放 |
| 原C全零、等价精简推理 | `infer.py --backend torch` → `tools/infer_compact.py` | 默认C；保留已评分的原三模型身份及完整算法 |
| 显式使用本轮训练三权重 | `infer.py --training-manifest ...` → `tools/infer_trained.py` | 直接核训练导出清单与三权重，不复制模型；保持全零C算法，目前仅Torch |
| 五引擎与CPU整数读出候选 | `infer.py --backend tensorrt` → `tools/infer_accelerated.py` | 显式提供五引擎清单；150图已完成并取回验收，待独立回分 |
| 构建这五个引擎 | `tools/build_inference_engines.py` | 独立GPU动作；新整理的公开构建入口仍需GPU实跑验收 |
| 已封装、明确三模型身份的便携推理 | `tools/infer_submission.py` | 需自包含配置、`model_identity.json`及源码清单；用于既有封装流程 |
| 旧E10a/G4b历史链 | `repro/train.py` | 仅历史重建，不代替当前完整三模型训练链 |

教师32源训练演习已正常执行完监督链，但导入了SSL/D5a，并非全部冷启动；黑盒80.4，效果淘汰。原人工GT的默认13阶段尚无独立完整fresh GPU实跑记录。历史SSL仍有参数恢复缺口，见[补证说明](SSL_PARAMETER_RECOVERY_20261101.md)。

D5a单独TensorRT候选已回分，无显著退化；全五引擎不能继承该结论，须等自身黑盒。详见[独立D5a](ENGINE_D5A_C150_20261101.md)及[全模块候选](FULL_ACCELERATION_20261101.md)。

## 文件各放在哪里

- `train_pipeline.py`、`infer.py`只作统一入口，不复制训练或推理实现。
- `repro/`负责阶段编排与本轮产物衔接，`config/`保存已有训练、推理配方。
- `tools/`保存具体执行工具，`models/`、`data/`、`utils/`提供其实际依赖。当前不能只拷贝两个顶层文件独立运行。
- 原图、GT、模型、ONNX、engine、逐图结果、比较图和压缩包只在私有忽略目录保存。代码整理不移动或清理现有资产。
- 新训练输出放独立`outputs/`子目录；新推理和引擎也使用独立目录，不覆盖当前封存任务。

## 环境准备

训练及普通推理沿用`sam2_env`和[requirements.txt](../requirements.txt)，均需仓库约定的本地SAM2源码。训练另需官方SAM2底座；完整backend推理直接严格加载三份推理权重，不再另读官方底座checkpoint。正式GPU训练不使用CPU替代结果。

TensorRT是可选依赖。已验证实验环境为Python3.12、PyTorch2.8.0+cu128、TensorRT10.12.0.36、ONNX1.19.0及RTX4080 SUPER。需要隔离加速依赖时，在既有GPU环境创建继承基础包的独立环境，再安装[requirements-tensorrt.txt](../requirements-tensorrt.txt)：

```bash
python -m venv --system-site-packages outputs/trt_env
# Linux下调用独立解释器；Windows改为 outputs/trt_env/Scripts/python.exe
outputs/trt_env/bin/python -m pip install -r requirements-tensorrt.txt
```

这份可选依赖不负责重装PyTorch或选择CUDA wheel。普通Torch路径不需要安装TensorRT。engine是针对构建环境优化的计算文件，部署时保留对应原权重和来源清单；换权重须重新构建，换GPU或TensorRT环境应在目标环境重建并验收。

## 训练的操作顺序

默认配置是[repro/current_pipeline.yaml](../repro/current_pipeline.yaml)。输入包括32源人工标注及r8监督资产、1000无标签图、64源已授权SAM2几何、官方SAM2底座；入口不会替你生成或审核这些监督资产。参数及来源见[训练配方](TRAINING_RECIPE_20261101.md)。

先查看计划和检查代码：

```bash
python train_pipeline.py list
python train_pipeline.py plan --config repro/current_pipeline.yaml --run-dir outputs/train_run_01
python train_pipeline.py run --dry-run --code-only --config repro/current_pipeline.yaml --run-dir outputs/train_run_01
```

这三个命令均不训练；`--code-only`也不核资源内容。在正式训练机器准备好全部输入后，再检查资源和实际执行：

```bash
python train_pipeline.py check --config repro/current_pipeline.yaml --run-dir outputs/train_run_01
python train_pipeline.py run --config repro/current_pipeline.yaml --run-dir outputs/train_run_01
```

默认阶段关系为：

```text
SSL → Stage1 → Stage1边界缓存 → joint
                                 ├→ S-align语义头
                                 └→ V affinity
独立D5a ─────────────────────────────→ R1（只更新affinity）
                                        ↓
                              导出backend/restorer
                                        ↓
                              本轮零噪N1训练源缓存
                                        ↓
                                NAF训练与导出
                                        ↓
                                  三模型身份核验
```

S-align直接使用joint固定教师，默认没有V6。具体依赖由`list/plan`输出，不靠手工拷贝历史checkpoint衔接。`--from-stage`、`--to-stage`可选择阶段区间，但前置回执必须真实完成。`--reuse-completed`只允许复用完全同源、完整核验的回执，不能把半途留下的文件当成完成；源码或配置变化时使用新目录。

真实GPU计算阶段启动前检查CUDA；错误环境会明确提示使用`sam2_env`，不会自动退到CPU长时间训练。只读命令、CPU导出及明确的独立权重导入不受此检查限制。默认Stage1在e0/e1、每10轮与最后一轮保存固定训练样本384缩略图。

主要产物相对本轮目录：

| 文件 | 用途 |
|---|---|
| `context.json`、`identity.json`、`receipts/` | 配置、资产、阶段来源与完成记录 |
| `export_backend/model/backend.pt` | 编码器、语义与affinity模型 |
| `export_backend/model/restorer.pt` | D5a恢复器 |
| `naf_export/model/naf.pt` | NAF生成器推理权重 |
| `verify_export/manifest.json` | 三权重身份及参数上界检查；不代表推理或官方精度验收 |

原C默认推理入口会拒绝不同身份的新权重。训练完成后，用显式训练清单参数直接接入本轮三个导出：

```bash
python infer.py --backend torch --training-manifest outputs/train_run_01/verify_export/manifest.json --input-dir data/test/semifinal --output-dir outputs/infer_trained_01 --dataset semifinal_150
```

新入口核对三文件SHA、实际状态数量、格式、NAF输入合同及参数上界，并记录本轮独立模型身份；不需要另外传`--model-dir`或`--naf-checkpoint`。输出目录须与训练run、输入图目录独立。它保持全零C算法，但不冒称使用旧C权重，也不会自动替换主线。新权重暂不支持该参数与TensorRT组合，不能复用原C引擎。此本地桥不是完整便携提交包，后者格式仍待确定；教师演习的旧封装见[实际推理记录](TEACHER32_C150_INFERENCE_20261101.md)。

## 原C推理的操作顺序

准备包含原`backend.pt`和`restorer.pt`的私有模型目录，以及原NAF checkpoint。固定[area_floor_200.yaml](../config/inference/area_floor_200.yaml)、旧三权重、全零C、两路全图语义、局部1024 affinity、NAF/S0、原图最终票及final200。

```bash
python infer.py --backend torch --input-dir data/test/semifinal --output-dir outputs/infer_C_01 --model-dir outputs/private_models/C --naf-checkpoint outputs/private_models/C/naf.pt --dataset semifinal_150
```

`data/test/semifinal`只是私有路径示例，应换为实际测试源目录。输出目录必须不存在。短测可以加`--only`指定已存在的源名，正式150图运行不传该参数。

输出为`results/`内每图一对`*_inst.png`、`*_class.json`，以及独立`audit/`内的来源、计时和完成记录。实例图为原尺寸单通道uint16、ID≤65535，类别珠光体0、铁素体1。结果黑盒只取`results/`中的规定成员，不能把审计或分析产物混进去。

全五TensorRT候选使用同一组原模型，并额外提供引擎清单：

```bash
python infer.py --backend tensorrt --input-dir data/test/semifinal --output-dir outputs/infer_TRT_01 --model-dir outputs/private_models/C --naf-checkpoint outputs/private_models/C/naf.pt --dataset semifinal_150 --engine-manifest outputs/engines/C_fp32/engines.json
```

引擎清单绑定五个角色的文件SHA与来源模块state SHA。缺引擎、错误权重、错误shape或dtype会失败，不静默改回Torch。NAF sigmoid、D5a输入检查和末端处理保持原路径；编译改变浮点计算方式，因此须独立比较最终结果与回分。

## 引擎重建与batch

新的通用构建工具从指定配置及原权重导出，不依赖私有实验脚本、测试样本或服务器绝对路径。具体参数用以下命令查看：

```bash
python tools/build_inference_engines.py --help
python tools/build_inference_engines.py build --config config/inference/area_floor_200.yaml --model-dir outputs/private_models/C --naf-checkpoint outputs/private_models/C/naf.pt --output-dir outputs/engines/C_fp32
```

`build`是实际GPU构建动作，输出目录必须全新。工具先在相邻的临时目录导出ONNX、构建engine并记录来源，五个角色全部成功后再发布含`engines.json`的目标目录。失败保留诊断目录，不发布可被误认为完整成功的清单；再次构建使用新目录，不覆盖旧引擎。

完整构建需要CUDA及TensorRT；本轮代码整理只执行本地静态、CPU验证，未用该公开builder重跑GPU构建。已完成的150图使用先前已单独验证并封存的引擎，不受本次整理影响。

当前五个`.engine`已包含各模块运行所需的优化后权重，没有启用权重剥离。不过当前加速入口仍会加载原backend/restorer/NAF三份`.pt`来核对来源state、构造原模块并沿用外围计算；因此目前仍需让原权重可访问。`--model-dir`可省略而使用默认`model/`，`--naf-checkpoint`仍须显式提供。专用的纯引擎入口可以去掉原`.pt`加载，但尚未实现；保留PyTorch张量、sigmoid和插值并不一定要求保留原模型权重。

当前四个1024模块固定batch1；NAF的256小块允许batch1–16、通常16。这里没有可以直接把1024改成batch4的现成开关。怎样重建并修改滑窗调度、为何不会必然得到4倍速度，见[batch实用说明](TENSORRT_BATCH_GUIDE.md)。

## 本轮整理的验证

顶层训练、两种推理后端及构建工具的帮助命令通过；默认13阶段代码预检通过且不创建run目录、不加载模型或修改GT。训练编排与推理相关99项CPU测试、新构建工具13项CPU测试均通过，文档链接和Python语法检查通过。

后续[可用性审计](TRAINING_INFERENCE_AUDIT_20261101.md)修复新训练权重推理缺桥、便携收尾假成功、无CUDA意外回退和Stage1过程图遗漏；联合345项CPU测试通过，并以服务器真实三导出通过CPU读取及合同核验。Claude独立审计超时、完成0轮，不冒称获得其通过结论。

新公开构建工具与新训练推理桥尚未GPU实跑；代码审计没有启动训练、重建正式engine或轮询后台150图。随后用户通知推理完成，已[取回原封存全加速150包并完成本地验收](FULL_ACCELERATION_20261101.md)，待独立回分；两类证据分别记录。
