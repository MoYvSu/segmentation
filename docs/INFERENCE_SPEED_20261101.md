# 原C推理加速：整数读出与TensorRT

后续评价口径：[独立引擎150图黑盒](ENGINE_D5A_C150_20261101.md)按用户要求转向真实大批量使用；主要比较热运行吞吐和官方分割效果，构建成本可摊薄，只作为部署信息记录。不能把预测变化提前称为精度损耗，也不以150张摊销否决编译。以下三图结果仍是已完成的小样本事实。

2026-11-01。新GT全链候选[用户回报80.4，已淘汰](TEACHER32_C150_INFERENCE_20261101.md)，部署继续使用旧13b3 backend、9315 D5a、20a59 NAF及完整全零C。此前[编译短测](TORCH_COMPILE_TEACHER32_20261101.md)只有Inductor `torch.compile`，没有TensorRT engine。本轮分别测试CPU整数读出和D5a TensorRT，不相互叠加，也不训练、改GT或组正式包。

## CPU整数读出：实际整图完成

新增`utils/fast_instance_readout.py`，只替换原`finish_watershed`的整数计数、过滤和重编号。小整数ID用`bincount`，超大稀疏ID用`unique`；一次查表映射代替逐ID重复全图mask。面积阈值与并列面积按原ID升序的稳定排序保持；超限cap仍收到同样的原ID数组并执行原函数。真实分水岭、两路语义、局部恢复/affinity、NAF、S0、raw浮点均值投票和final200均不变。

独立入口为`tools/infer_fast_readout.py`，复用旧compact参数和默认C，只在本次run作用域替换一个函数，成功或异常后恢复。所有审计JSON明确写入fast模式、两新源码SHA、实际完成计数及无训练标识，另保存`audit/fast_readout_report.json`。旧默认入口不改。

同RTX 4080 SUPER、torch2.8.0+cu128、原FP32/TF32配置，使用test_169（2048×2448）、170（1044×1244）、171（1024×1224）。先普通预热，再普通A→fast A→fast B→普通B，每轮三张原图、同一已加载模型。表中为三图合计秒数：

| 路径 | A轮predict | B轮predict | 两轮均值 | 每图均值 |
|---|---:|---:|---:|---:|
| 原compact C | 22.7190 | 22.9497 | 22.8344 | 7.6115 |
| 仅整数快路径 | 18.4908 | 18.3531 | 18.4219 | 6.1406 |

predict耗时减少**19.3238%**，速度倍率**1.2395**。其中种子/分水岭读出10.8635→6.6547秒，减少38.7423%；NAF2.2480→2.2181秒只是未改算法的计时波动。predict含网络、读出、raw票与final200，不含文件保存/加载；两轮外围墙钟均值25.5142→20.9821秒，包含输出验证，不作为纯GPU时间。

验证范围：

- 相关61项CPU测试通过，覆盖正ID1、同面积排序、阈值相等、稀疏int32、cap链及孤立删除、实际65535输出无溢出、输入不变、真实C/F0分水岭、审计及异常作用域恢复。
- 全5轮15张最终PNG和对应JSON逐字节匹配已评分旧C150包（SHA `d484e9e1b0f30cebc1a0c210d5cc012e22c867dccdba86805a91ce794aee30a5`），uint16/原生尺寸/类别合同通过。
- 同轮间输入、所有模型state、NAF前后边界、两路语义/原图投票概率、种子/高程/WS和final200审计全部exact；参数总数102,583,318。原934份封存文件和新overlay源码前后SHA不变。
- 未跑全150加速推理，不把这三张的耗时分布推广成150图或跨机器固定比例。此处输出没有变化，不需解释成mIoU提升。

本地合成整数字段微基准（Windows、numpy2.3.3，各3次交错）为1024×1224/128区0.2900→0.02973秒，2048×2448/512区4.3208→0.16120秒；仅代表这个整数阶段，不代替上面的真实整图结果。

```text
python tools/infer_fast_readout.py --config config/inference/area_floor_200.yaml --dataset semifinal_150 --input-dir <测试原图目录> --model-dir <旧C模型目录> --naf-checkpoint <旧20a59对应权重> --output-dir <全新私有目录>
```

入口SHA `b901fe1f08845fdc2e2aa460ba2a2589107c9f455c1dfd11e8ae6d43405c044b`；工具SHA `95544768ca76015c7aeeddafff78c907b45550e23d96427b76b49abf8cdc5eee`。旧compact源码封条SHA `7af620dc7cc226e487e1c5947f02b91675ca02ff551ef41f1de3f72377dfe1bd`。服务器根为`/root/seg-local-artifacts/inference_speed_20261101/cpu_integer_v2/`，本地私有回执为`output/inference_speed_20261101/cpu_integer_v2/{complete,verification}.json`；合成基准另在`output/fast_instance_readout_20261101/benchmark.json`。

首版独立worker在主函数前因Python包路由失败，未加载模型/进入推理；保留原worker与日志，改用v2。在封存v2前补齐namespace tools与普通utils包的overlay路由并通过无CUDA导入预检，未修改旧源工程。

## D5a TensorRT：独立试验

仅将真实调用的`D5a.denoise`导出ONNX opset17、再构建TensorRT 10.12.0.36引擎；不是其不用的16步随机`forward`，也不是整条SAM2/语义/affinity/NAF都engine化。使用独立venv继承原环境，固定onnx1.19.0和TensorRT，不改`sam2_env`；原`.pt`及默认入口保留。

保守FP32不启用FP16/INT8，TensorRT关闭TF32；旧主线cuDNN允许TF32、matmul禁TF32，两者的数值开关不完全相同，不声称逐值等价。真实wrapper每次仍执行原输入检查和静态shape/dtype检查，返回未clamp的raw x0，原恢复收尾/零噪声禁令不改。引擎I/O按名字绑定Torch CUDA张量，持有实际连续buffer，当前流执行并同步后返回独立输出；这部分同步开销计入测试。

使用同三张原图的普通A→engine A→engine B→普通B，模型复用、同旧C ZIP对照，单列构建时间、模块数值差异、最终PNG/类别及重复稳定性。实验worker SHA `63f2a426de2ae2b752d2b75055f53a084f4e9e10b68cfce4a910c06d76123f8d`，服务器`/root/seg-local-artifacts/inference_speed_20261101/engine_d5a/`。

2026-11-01 12:45:10（北京时间）完成，无运行失败：

| 三图合计predict秒 | A轮 | B轮 | 两轮均值 | 每图均值 |
|---|---:|---:|---:|---:|
| 普通旧C | 23.1006 | 21.6038 | 22.3522 | 7.4507 |
| 仅D5a换engine | 21.9633 | 20.8294 | 21.3964 | 7.1321 |

整图耗时减少**4.2762%**，速度倍率**1.0447**。相同真实D5a输入的5次同步模块短测90.417→52.345毫秒（**1.727倍**），不能称整网加速1.7倍。ONNX导出0.853秒、构建64.139秒，写盘后由新TensorRT runtime从实际engine文件反序列化加载并执行；整个含预热/构建/五轮整图/校验的实验202.29秒。

引擎19,368,356字节，SHA `bc9b49478cf3cf8a7ea2d24b23da319bbc3a33759c47d424e1cce3fdafc92722`；ONNX13,332,320字节、1003节点，SHA `35ae454bac0051100a08aa81267024d5b93f44e26623d5252ca3aa5e4bdbe916`。I/O为FP32 `xt/condition/x0`各1×3×1024×1024，INT64 `t_index`为1。原始x0相对普通版最大绝对差0.00028512、平均0.000014991，无非有限输出。

| 图 | 原/engine实例数 | F→P / P→F像素 | 前景→0 / 0→前景像素 | 稳定对应翻类 |
|---|---:|---:|---:|---:|
| test_169 | 304 / 304 | 66 / 50 | 307 / 299 | 0 / 303 |
| test_170 | 103 / 103 | 2 / 169 | 149 / 161 | 0 / 102 |
| test_171 | 35 / 35 | 0 / 14 | 39 / 46 | 0 / 35 |

三图F/P像素翻转**301/7,565,616＝0.00397853%**（全部原像素分母），背景变化单列。440个双向覆盖≥90%的稳定对应零翻类，另有2个原/新区域未进入稳定集合，不能把零翻类扩大为所有对象无变化；按新区域最大旧重叠计算的其它共同前景分区像素358/264/22。最终PNG均与普通版不同；169的类别JSON也变，ID重编号会影响此JSON，不等于已发生整粒翻类。

两次engine结果逐文件相同；普通采集轮和A/B两轮逐文件匹配原评分包。全部15份PNG及类别通过输出合同。原图全图raw投票概率和所有模型state在所有轮保持exact；旧934文件封条及engine文件SHA复核通过。实际推理仍显式零噪声，没有重引入随机注入。本地回执`output/inference_speed_20261101/engine_d5a/{complete,build,numeric,verification}.json`及`trial.log`。

本轮未转换编码器、语义头、affinity头或NAF，没有测试全150、独立进程热缓存或跨硬件；同机两次复现不等于跨机逐位保证。没有测试GT或官方engine回分，不能判断301像素变化的好坏。CPU与engine为分别运行的单变量实验，时间百分比不能相加；上一次Inductor试验还是不同的新权重，不能混成统一横评。当前更适合继续验证CPU整数候选，D5a engine保留试验，不替换默认。

[TensorRT10.12算子支持](https://github.com/onnx/onnx-tensorrt/blob/10.12-GA/docs/operators.md)、[官方Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html)与[精度说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)用于核对导出与执行方式；算子支持表不代替本模型实际构建/推理证据。
