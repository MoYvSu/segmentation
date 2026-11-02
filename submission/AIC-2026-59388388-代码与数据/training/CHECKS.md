# 训练流程保留的检查

这些检查随训练阶段执行；`train.py --check`只检查入口文件、数据和CUDA。位置均相对`training/`，函数范围内的结构、shape、dtype、有限值和索引范围检查归入下表对应功能。

13阶段的执行顺序见[README训练阶段表](../README.md#训练)，各阶段参数与产物见[默认训练配方](RECIPE.md)。

## 入口、文件与写入范围

| 位置 | 检查内容 |
|---|---|
| `runner.py:63`（`check_inputs`） | 执行前核对必要文件、32人工源及标注/原尺寸r8目标、1000无标签源、64无类别SAM2源和mask、P=0/F=1、SAM2源码与CUDA。 |
| `runner.py:122`（`run`） | 输出目录必须新建；每阶段非零退出即停止；只运行13个训练/导出阶段。 |
| `worker.py:39`（`main`） | GPU计算阶段检查CUDA；CPU导出阶段直接加载本轮产物。 |
| `source/repro/current_early.py:67`（`build_stage`） | 生成本轮绝对输入/输出与独立子进程命令；S-align冷重置语义、冻结边界/LoRA；关闭配方外adapter/refiner分支。 |
| `source/repro/current_early.py:270`（`run_stage`） | 阶段输出新建、展开配置新建、子进程成功且所需产物存在。 |

## 原始监督与几何

| 位置 | 检查内容 |
|---|---|
| `source/data/boundary_naf.py:126`（`load_processed_binary`） | r8实例、类别与original_covered/filled/residual_unknown尺寸及0/1域；已知/未知边界、ignore和有效边界域。 |
| `source/train_backend_adaptation.py:61`（`build_datasets`） | 人工32与无类别SAM64源、图像和mask尺寸、无类别合同；不读取历史control或approval。 |
| `source/data/boundary_naf_real.py:58`（`__init__`） | NAF真实输入清单与同一训练源/GT、当前backend/restorer对齐；原尺寸FP32有限概率、逐源shape和内容对应。 |

## 模型初始化、共享特征与严格加载

| 位置 | 检查内容 |
|---|---|
| `source/models/sam2_encoder.py`（`SAM2Encoder.__init__`） | 官方checkpoint仅提取`image_encoder.trunk.`并对Hiera主干strict加载；原生Mask Decoder保持随机构造后丢弃，任务头随机初始化。 |
| `source/repro/current_early.py:230`（`verify_shared_encoder_contract`） | joint/S-align使用同一官方SAM2特征路径与同一完整LoRA状态；冻结LoRA保持；不比较旧模型身份。 |
| `source/train_affinity_geometry_g1.py:201`（`validate_semantic_geometry_contract`） | 组装S-align语义和V affinity时，核对共同joint初始化与共享LoRA兼容。 |
| `source/models/backend_adaptation.py:37`（`build_backend`） | 本轮joint/S-align/V严格组装；语义/affinity头及LoRA结构完整加载。 |
| `source/repro/current_backend.py:292`（`train_fresh_r1`） | FP32 R1只训练affinity；冻结模型和restorer状态保持；模型参数量限制；终态strict重载张量完全相同。 |
| `source/repro/current_backend.py:410`（`run_stage`） | R1实际完成且strict重载后导出；新backend/restorer与本轮训练态张量相同、参数预算有效。 |
| `source/tools/build_semifinal_bundle.py:113`（`export_checkpoint`） | 导出实际模型tensor、转FP32后数值逐tensor一致，输出目录新建。仅调用此导出函数。 |

## 训练数值、抽样与过程图

| 位置 | 检查内容 |
|---|---|
| `source/tools/pretrain_lora_ssl.py:211`（`main`） | SSL有限loss、梯度裁剪及非有限梯度拒绝。 |
| `source/train.py:235`（`train_one_epoch`） | Stage1有限loss/梯度，维持原AMP与FP32更新分支。 |
| `source/train_stage2.py:655`（`train_one_epoch`） | joint/S-align有限loss、梯度裁剪；AMP梯度溢出交由GradScaler跳过并降低缩放值，FP32非有限梯度报错；仅实际更新后更新EMA，冻结模块保持eval。 |
| `source/train_affinity_geometry_g1.py:788`（`main`） | V有限loss、原有效affinity监督、选优及梯度裁剪；AMP梯度溢出跳过并降低缩放值，FP32非有限梯度报错；记录实际更新数，epoch/LR来自配方。 |
| `source/train_rgb_diffusion_restoration.py:361`（`train`） | D5a数据数与all_train分割、模型架构、正训练预算；有限loss/梯度，至多重试失败batch并记录；完成实际更新数。 |
| `source/repro/current_backend.py:255`（`_r1_update`） | 人工/几何两次反向后一次更新；有限loss/梯度、仅affinity梯度。 |
| `source/repro/current_backend.py:232`（`build_fresh_datasets`） | 两流长度匹配；按实际DataLoader随机流生成共同whole/native计划。 |
| `source/utils/training_process_monitor.py:35`（`observation_state`） | 固定训练源小图；保存恢复各模块train/eval和Python/NumPy/Torch RNG，no_grad。 |

## 真实N1、NAF及最终三模型

| 位置 | 检查内容 |
|---|---|
| `source/repro/current_naf.py:221`（`capture_native_boundary`） | 真实局部1024滑窗执行D5a；显式零噪、q_sample一次、初态等于条件图、16步/kappa.03；affinity原尺寸融合。 |
| `source/repro/current_naf.py:280`（`training_config`） | 缓存32源覆盖、同一新backend/restorer和GT、缓存文件位置/内容对应；NAF线程配置有效，旧测试control为空。 |
| `source/train_boundary_rgb.py:299`（`validate_config`） | FP32、有效batch/microbatch/学习率/GAN热身、训练与推理相同二值阈值、线宽归一化、tile/overlap/reflect/概率拼接。 |
| `source/utils/boundary_rgb_training.py:254`（`rgb_gan_update`） | G/D有限loss和梯度、未知域安全的监督与对抗有效域；按配方热身及学习率日程更新。 |
| `source/train_boundary_rgb.py:482`（`monitor`） | 过程图前后G状态保持；仅固定训练pair。 |
| `source/repro/current_naf.py:339`（`run_train`） | 实际完成configured main_updates。 |
| `source/repro/current_naf.py:364`（`run_export`） | complete/实际终态/配方一致；严格加载和重载G，移除D/optimizer/RNG后输出naf.pt。 |
| `source/tools/infer_boundary_rgb.py:34`（`load_naf_checkpoint`） | NAF结构与输入合同严格一致、完整strict state加载、实际模型状态digest一致。 |
| `worker.py:14`（`finish`） | 三导出state结构和有限值、状态元素上界合计<500M；生成training_export.json。 |

## 身份记录和实际对齐

文件SHA写入训练与导出记录，便于定位本轮产物。NAF读取缓存时核对当前模型、GT及逐源缓存内容，防止把另一模型或另一幅图的缓存配给当前目标；这项检查发生在数据封装/加载处。R1冻结模块digest与strict重载检查用于确认实际更新范围和保存完整性。

训练预算由配方给出。循环及完成文件核对实际执行计数。入口按新输出目录运行，预训练只使用SAM2官方底座；训练结束输出本轮`models/backend.pt`、`models/restorer.pt`和`models/naf.pt`。
