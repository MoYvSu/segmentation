# 原C全模块加速整合

2026-11-01。D5a独立引擎候选[用户回分](ENGINE_D5A_C150_20261101.md)为mIoU .7461、面积 .9073，等权参考82.670（旧C82.660），没有明显效果退化。用户授权将其余网络编译/引擎化与CPU整数读出一起整合，再做完整黑盒确认。真实目标为大批量热运行吞吐，构建时间保留记录，不作为150张摊销否决门槛。

## 150图完成、下载与本地验收

正式任务于2026-11-01 **15:24:52（北京时间）**完成；用户通知后取回仅结果黑盒，并在本地独立重算差异、核验全部成员。包为`semifinal_C_full_TRT_FP32_fast150.zip`，8,182,796字节，SHA256：

```text
b71fcb43fbfbe5a9d0deec2863010162c9fe80503fa723d1306e0cd06fc25d7d
```

本地私有路径`output/full_accel_20261101/received/`。150对/300个唯一results成员、ZIP CRC、全部成员SHA、原尺寸16位灰度实例PNG、ID≤65535及类别键集均通过。最终文件逐项匹配服务器完成回执与逐图推理报告，本地对照统计与服务端完整一致。回执为`download_validation.json`及`local_comparison/summary.json`，未自动提交平台。

五引擎来源state核验通过、0回退，实际调用trunk1037、语义300、affinity887、D5a887、NAF732。原权重/零噪声/C算法保持。worker推理、比较及组包共**722.89秒（约12分03秒）**；predict150图645.08秒、均值**4.3005秒/图**，NAF与CPU读出均值分别0.4865/1.6977秒。该150图均值不与下方三图对照直接求速度倍率。

相对已评分D5a包`45d65bb7…c5f698b`：

| 指标 | 全150图结果 |
|---|---:|
| F→P像素 | 10,290 |
| P→F像素 | 10,098 |
| F/P像素变化／全部像素 | 20,388／367,181,856＝**0.00555256%** |
| 稳定对应实例 | 15,598对，其中0个F→P、1个P→F |
| 预测实例数 | 15,662→15,664 |
| 前景→0／0→前景像素 | 37,555／37,308，单独统计 |

稳定对应之外旧64/新66个区域另计，不能把“1个稳定实例改类”理解为全部几何差异。无测试GT，以上是预测变化而非准确率；整合版是否采用仍待本包独立回分。

## 历史小样本结果与150图启动

2026-11-01 **15:10:33（北京时间）**三原生尺寸整合对照完成，使用实际公开整合入口。控制为已评分D5a加速版，其余网络仍普通执行；候选为全五引擎＋CPU整数finish。每轮三张，排除各自完整预热，ABBA两轮均值如下：

| 指标 | D5a已评分控制 | 全五引擎＋整数读出 | 耗时减少 |
|---|---:|---:|---:|
| predict三图合计 | 21.6571秒 | 16.2324秒 | **25.0481%** |
| 每图predict | 7.2190秒 | 5.4108秒 | **速度倍率1.3342** |
| NAF三图合计 | 2.1855秒 | 1.5261秒 | 30.1712% |
| 种子/WS读出三图合计 | 10.9968秒 | 6.5575秒 | 40.3686% |

predict含网络/NAF/CPU读出及投票，不含模型与引擎加载/保存；外围墙钟均值24.1865→20.1738秒，减少16.59%，另含读写及校验。模块同输入诊断倍率trunk1.193、语义1.348、affinity1.489；NAF在B16/15/10/1分别1.709/1.713/1.780/1.884倍，输出均有限。实际整图为主要速度证据，不能把各模块倍率相乘或扩大为全150固定比例。

控制轮逐文件匹配已评分45d65包；候选完整预热、A/B两轮相互逐字节一致，各模块实际引擎调用/来源state核验通过，无回退。但候选与控制不同：F→P396、P→F168，**564/7,565,616＝0.00745478%**（全像素分母）；前景→0为980、0→前景976，另外统计。440个双向90%稳定对应无整粒翻类，其外旧/新各2区域另报。数值差不能统称为全部极小：trunk f2最大绝对差约0.6164、语义logits约0.02634；是否影响精度仍靠整合黑盒回分。

新进程单169实际GPU预检通过，最终PNG/JSON与整合短测逐字节一致；**15:12:46**后台正式150图任务启动，PID **116129**，随后自动对比已评分D5a包并只出`semifinal_C_full_TRT_FP32_fast150.zip`。启动后不代理轮询，完成由用户通知再取包；当前启动记录不代表150图或评分已完成。

实际引擎清单SHA `41af16f71fa0a89e1f686d92c2342566c0709bda72fcaacaadac01edb3438eaf`；正式worker/overlay/来源清单SHA `43bc6fda7e7b3a270fd3482e3b382117a4501d387f90c9be28a74d3843841d1f`。公开入口SHA `aeae8fcf7383376648687c56eb8e63649ed29ef8c7b9f5c453cc0ababae60bb2`，运行桥SHA `0c38b5952f2224b5573dd1f29245de8e9eca46fe4638f14a75eb1daf868d2c22`；正式worker SHA `220c937ab06e1ac27430001d72bd7c058c3d06f227feed0a311c4ca3eaae5631`。本地私有回执在`output/full_accel_20261101/`，包含trial_complete、build_complete、trial_comparison_summary、full_preflight_complete与full_launch。

## 固定项与整合范围

旧13b3 backend、9315 D5a、20a59 NAF、全零C及原所有阈值/融合/投票/面积规则保持；无新训练、GT变更、FP16、INT8或TTA。整合为Hiera trunk、语义头、affinity头、NAF计算体的TensorRT FP32引擎，加原已回分D5a引擎与CPU整数finish读出。NAF只编译logits计算体，原Torch sigmoid和输入检查保持；原两路全图语义、局部1024/25%滑窗、NAF/S0/reflect32/seed50/final200保持。

主要参照是已评分D5a包`45d65bb7f5a990f6671ca14cd7937dc7abca9abf85b1795fea234f56bc5f698b`，本轮新变化不能沿用前期新GT权重的Inductor结果作为证明。先完成D5a＋整数finish与该包的三图逐字节对照，再评估其余引擎带来的数值/实例变化。

公开候选入口`tools/infer_accelerated.py`；旧入口默认保留。`--engine-manifest`明确五个角色、相对路径、文件SHA、来源模块state SHA以及FP32/TF32关闭声明，加载时逐角色核实原模块来源，拒绝错误权重或缺失引擎，不静默回退。原nn.Module对象与state_dict命名保持，只在作用域内替换计算方法；成功或异常退出均恢复。

`utils/tensorrt_runtime.py`按命名Torch CUDA tensor绑定TensorRT I/O；NCHW逻辑形状的channels-last输入先保值转换为连续内存，再绑定，输出每次独立分配。实际连续副本存活至同流同步返回，支持NAF动态batch1–16，调用记录只存元数据。普通入口import无需TensorRT；可选依赖见`requirements-tensorrt.txt`，基础requirements仅增加说明，不改现有环境。

## 导出适配与核验

- Hiera在固定1024输入下，原bicubic位置嵌入以原函数预计算，导出wrapper持有常量；先核逐值相等、退出恢复原方法，不改插值成bilinear、不改原模型state。其余attention沿原实现导出，无手工LoRA合并或attention缩放改写。
- NAF动态batch导出器未能将`LayerNorm(value.shape[-1])`识别为固定通道。逐层在导出前读取原weight长度为Python常量，再调用相同permute/F.layer_norm/参数/eps1e-6/反permute；真实sample16的原/适配logits `torch.equal`通过，原方法恢复后state一致。实际batch1/10/15/16运行另测。
- 第一轮构建得到trunk、semantic、affinity三个引擎，NAF导出失败后保留原worker/失败日志，仅补建NAF。一次profile API检查错误也保留：TRT10.12的`set_shape`实际返回None，改以`get_shape`及profile有效性验证成功；没有因此改网络。
- 62项相关入口/运行桥/fast入口CPU测试通过，覆盖manifest/五角色错误来源、shape/dtype/profile范围、I/O生命周期、独立输出、失败不回退及作用域恢复；运行桥已用实际D5a引擎完成GPU FP32/INT64 I/O smoke。

服务器私有根为`/root/seg-local-artifacts/full_accel_20261101/`，继续留root，不迁移旧数据或改变已评分包。复用独立`trt_env`，torch2.8.0+cu128、TensorRT10.12.0.36、onnx1.19.0、RTX 4080 SUPER；未修改sam2_env。

## 试验与可追溯产物

五个引擎的文件与来源state由`engines.json`（`C_TensorRT_acceleration_v2`）绑定。原权重文件为backend `13b3ab1b…161f04`、restorer `9315da87…17007b`、NAF `f2b8f2db…19ee8f`，NAF旧state算法为20a59；新manifest的来源state统一使用`entry.tensor_digest`，两种摘要算法不能直接比较。

原代码934文件封条SHA为`7af620dc7cc226e487e1c5947f02b91675ca02ff551ef41f1de3f72377dfe1bd`，数据清单SHA `14dd7572aa9e6923bb6160c91d2589af4642b4a09fb45029f0e3657104e5a3c9`。新overlay、worker与engine逐SHA封存，旧源码/模型不改。所有sample tensor、ONNX、engine、预测及压缩包只存私有忽略目录。

| 私有文件 | 用途 |
|---|---|
| plan.json / build_manifest.json | 原C资产、任务范围与初次构建源码 |
| build.log / build_failed.json | 初次构建与NAF动态导出失败记录 |
| resume.log / resume_failed.json / finish.log | NAF原值等价适配与profile API诊断/补建 |
| build_complete.json / builds_v2.json | 四模块成功构建及精度/shape/适配记录 |
| source_digests.json / engines.json | 引擎对应的原权重与运行清单 |
| control_verified.json | D5a＋整数finish三源最终字节匹配已评分D5a包 |
| module_comparison.json | 同真实输入模块计时及数值差异、NAF四种batch |
| trial_rounds.json / trial_complete.json | 三原生尺寸整体ABBA对照，排除控制和候选完整预热 |
| overlay/tools/infer_accelerated.py | 实际使用的公开整合入口快照 |

整图对照使用169/170/171三原生尺寸；控制和候选使用同一非默认CUDA流，先分别完整预热，再control_A→optimized_A→optimized_B→control_B。复用同一已加载模型及引擎context，逐run仅重置调用审计计数，避免把首context/profile开销当稳态吞吐。模型/engine加载不计入predict；模块小测含实际桥接与同步，仅作诊断，以整图对照为主要速度证据。

候选数值变化用F/P、背景进出、稳定实例及未匹配/拆并分别记录，不作无GT准确性结论。小样本通过后使用同一整合入口后台跑150图并只生成结果黑盒；启动后停止代理轮询，用户通知完成再取包，不自动平台/网盘上传。完整代码包模式仍待后续确定。
