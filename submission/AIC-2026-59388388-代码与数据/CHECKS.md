# 运行检查清单

检查用于确认输入、模型接口、训练更新和输出文件符合流程要求。shape表示张量尺寸，dtype表示数据类型，profile表示引擎允许的输入尺寸及批量范围。

## 推理

| 环节 | 保留的检查 | 位置 |
|---|---|---|
| 输入与输出目录 | 输入存在、图片列表非空、文件名stem唯一；输出目录为新目录或空目录 | `infer.py:main`、`inference/pipeline.py:input_files` |
| 图片读取 | 可解码为uint8三通道图像 | `inference/pipeline.py:read_rgb` |
| GPU | NVIDIA CUDA可用 | `infer.py:main` |
| 引擎文件 | 五角色文件存在；使用清单时各角色包含文件名 | `inference/pipeline.py:engine_files` |
| TensorRT加载 | 引擎反序列化成功、执行上下文创建成功；TensorRT明确报告平台、GPU或版本不兼容时，旧进程退出，自动重建并重试一次 | `inference/engine.py:Engine.__init__`、`infer.py:run_with_rebuild` |
| 接口 | 角色输入/输出名称匹配；设备张量、LINEAR布局；FP32或D5a时间索引INT64 | `inference/engine.py:Engine.__init__`、`inference/pipeline.py:Pipeline._check_io` |
| 调用输入 | 名称集合、CUDA设备、dtype、正尺寸、固定维和动态profile范围一致 | `inference/engine.py:Engine.__call__` |
| 执行 | profile选择、输入尺寸设置、地址绑定、输出尺寸推导与GPU执行成功；临时张量保留至同一CUDA流结束 | `inference/engine.py:Engine.__call__` |
| affinity与NAF | 概率有限；NAF输出为本批B×1×256×256；滑窗权重覆盖全部像素 | `inference/pipeline.py:affinity_boundary`、`Pipeline.predict`、`Pipeline._naf` |
| 实例投票 | 实例为原尺寸uint16；语义概率同尺寸、FP32、有限且位于0至1 | `inference/postprocess.py:vote_instances` |
| 分水岭与合并 | marker/标签范围有效；S0按原边界支撑筛种；末端合并保持接触与连通关系 | `inference/postprocess.py:watershed_instances`、`merge_final_instances` |
| 最终文件 | 原尺寸、单通道uint16、ID≤65535；正实例ID与类别键对应、P0/F1；PNG写入成功 | `inference/postprocess.py:verify_output`、`inference/pipeline.py:save_prediction` |

## 训练

训练完整定位表见[training/CHECKS.md](training/CHECKS.md)，参数与产物见[training/RECIPE.md](training/RECIPE.md)。保留检查如下：

1. 新输出目录、必要文件、训练计算阶段CUDA可用，阶段进程正常退出。
2. 32人工源图/LabelMe/早期目标/r8目标配对，1000训练原图，64无类别SAM2几何及对应源图、mask存在。
3. 目标尺寸与原图一致，实例整数格式，P0/F1类别映射；original_covered、filled、residual_unknown等有效域格式一致，未知和padding保持ignore。
4. 官方预训练权重仅载入Hiera主干，任务头随机初始化；checkpoint结构、键、shape及模型架构严格加载；joint、S-align、V的共享编码器与LoRA兼容。
5. 随机初始化、冻结与更新范围符合阶段设置；R1的冻结模块前后状态保持、仅affinity获得梯度。
6. loss与导出张量有限；FP32非有限梯度报错；S-align/V的AMP梯度溢出由GradScaler跳过当次更新并降低缩放值，记录实际更新数。裁剪与更新按配方执行；两条R1数据流共用whole/native视野计划，实际更新计数与循环预算一致。
7. N1由本轮backend/restorer的真实局部零噪推理生成；缓存、目标和模型逐源对齐，尺寸、FP32概率与内容对应。
8. NAF训练与推理共用二值阈值、线宽、tile/overlap/reflect和概率拼接方式；有效batch/microbatch、线程、学习率及GAN热身设置。
9. 过程图保存时保持模型状态并恢复随机流；固定训练样本用于过程观察。
10. 导出为严格可重载的FP32状态，训练态与导出态逐张量相同；三模型参数或状态元素上界合计小于500M。

文件SHA写入本轮记录。NAF加载缓存时将当前模型、GT和各源缓存对应起来；R1冻结摘要与导出重载用于检查更新范围及保存完整性。

## 引擎构建

| 环节 | 保留的检查 | 位置 |
|---|---|---|
| 输入与目录 | 三权重、模型源码及SAM2源码存在；输出目录新建，且与权重及源码目录分开 | `training/engine_export.py:_activate_source`、`load_sources`、`destination` |
| 模型接口 | 严格加载backend/restorer/NAF；共享主干和两个头，D5a16步/kappa.03，NAF单通道边界；FP32状态及参数总量<500M | `training/engine_export.py:load_sources` |
| 导出适配 | 固定1024位置编码来自原bicubic结果；NAF固定原LayerNorm通道；适配前后计算相等、退出恢复原方法 | `training/engine_export.py:fixed_trunk_position`、`fixed_naf_channels` |
| 样本和ONNX | 四级特征尺寸正确；opset17图结构检查通过；主干位置编码已折叠为常量 | `training/engine_export.py:synthetic_inputs`、`export_onnx` |
| 构建设置 | CUDA可用；TF32/FP16/INT8关闭；五角色I/O名称、方向、shape、dtype和LINEAR布局匹配 | `training/engine_export.py:run`、`_check_io`、`build_engine` |
| NAF批量 | min1、opt16、max16；profile读回值有效，索引为0 | `training/engine_export.py:build_engine` |
| 构建结果 | ONNX解析、引擎构建和反序列化成功；五引擎全部完成后发布目录及清单；保留已有输出 | `training/engine_export.py:build_engine`、`run` |

## 本次目录整理移除的运行门槛

- 历史checkpoint固定SHA、固定e9、旧成绩和旧黑盒对照。
- 原实验源码封条、全数据/源码在阶段前后的重复SHA复核。
- 历史control、parity、旧运行回执及approval文件。
- 全局GPU空闲、指定显卡型号、等待截止时间、额外磁盘余量。
- 固定150张及固定测试图名白名单。
- 常规推理加载原三份`.pt`、构造SAM2训练模型和核对源模块状态摘要；平台不兼容时，独立构建进程按需读取三份权重。

## 交付验收

交付验收在独立检查目录执行：源文件与资产复制完整性、依赖与入口导入、训练数据检查、三原尺寸GPU运行、150张最终PNG/JSON逐文件比较、纯引擎路径的原权重加载拦截、文档词语与路径检查。用户运行入口执行上表列出的运行检查。
