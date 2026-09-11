# E10a + G4b 跨头修正对照

## 目的与当前边界

2026-09-11 用户决定停止继续投入直接掩码路线：伪标签、归属约束与 EMA 虽有收益，最终
覆盖与实例几何仍明显低于主线。部署继续使用官方总分 83.94 的 E10a + G4b。本轮检验
**两个任务能否通过读取对方的信息改善最终结果**，先实现小型残差对照，不重新训练 SSL、
LoRA 或原解码器。残差是加在原预测上的小幅修正，初始化时严格为零。

这只是增量实验，不能据此宣称已得到可冷启动复现的新双头架构。若有可重复的部署收益，
再考虑把交流模块纳入短训练链。阶段选择继续使用验证损失，正式晋级仍要求固定部署下的
实例 mIoU、铁素体平均面积项与完整输出比较。

## 唯一对照变量

| 项目 | self 对照组 | cross 实验组 |
| --- | --- | --- |
| 原模型 | 同一 E10a + G4b，全部冻结、eval 模式 | 相同 |
| 修正模块输入 | 本头特征与本头预测概率 | 本头及另一头的特征与预测概率 |
| 两条处理分支 | 独立非线性分支，都读取本头信息 | 其中一条改读另一头 |
| 参数、初始化、采样及更新数 | 87,113 个参数，同一初始化和逐轮随机种子 | 相同 |
| 损失 | 现有语义损失 + 实例连接损失 | 相同 |
| 后处理 | 主线原配置 | 相同 |

原语义 FPN 特征 256 通道、affinity 特征 128 通道，分别连同本任务概率投影到 32 通道。
每头通过两条独立的卷积、GroupNorm、SiLU 分支处理自身和上下文信息，再融合生成修正。
GroupNorm 在单张图像内归一化，不积累跨批统计。两组没有用闲置参数或乘零支路凑参数量。
cross 组的两个任务损失可共同训练这些小型投影，但梯度不会进入原模型。

语义修正从 512 网格放大到 E10a 的原输出网格；原语义 logits 不重采样。8 个 affinity
通道直接在原 512 网格修正，随后仍使用原 gated/short-mean 融合。输出层零初始化，修正
使用 `tanh × 1.0` 限制在 ±1 logit。logit 是转换成概率前的分数；限制它只是在控制改动
幅度，不能保证非零修正后分水岭的实例结构不变。

总参数为 81,754,507（81.754507M），其中原主线 81,667,394。原模型输入仍为
`legacy_none`；FP32 运算设置与原 fused 入口一致（matmul TF32 关闭、cuDNN TF32 开启），
不使用 autocast。两头读取同一个主线 encoder 的兼容特征，不混入 clean60 的 encoder/LoRA。

本轮不加入额外联合损失。尤其不把“同类别”当成“同实例”；同为铁素体的相邻晶粒仍是
负连接。交流只是提供信息，没有人为把语义轮廓强制等同于所有晶界。

## 数据与训练

- 人工新 GT：`outputs/experiments/seeded_label_completion_o1/radius_8/derived_maps`，
  两头从同一实例图派生监督；剩余 unknown 与涉及 unknown 的连接继续 ignore。
- 原 `clean60_split.json` 中 26 张训练图保持不变；验证使用 172、351、686、872、873。
  889 保持排除，也不加入训练。五图用于沿用近期报告口径，并不消除主线及历史实验的选择偏差。
- 每组 20 epoch，每轮有放回抽样 128 张、batch size 1，共 2,560 次更新；两组顺序运行。
  每轮按同一 `seed + epoch` 重建抽样与在线增强序列，含 50% 原尺寸 1024 方窗、50% 整图。
- 只用上述人工监督。固定 240 张伪标签与在线无标签池不进入本轮，以先隔离信息交流效果。
- 学习率 `1e-4 → 1e-5` 余弦下降，AdamW weight decay `1e-4`，梯度裁剪 1.0。
- 现有 E10a 式语义损失（含实例核心/整实例概率项）与类别无关 affinity 损失复用原实现；
  后者跨实例负连接权重 1.5。用冻结主线在 **26 张训练整图** 上的平均两项损失分别定标，
  两组共用且训练中不更新。每轮五图验证损失之和选优，零修正 epoch 0 也参与 best 选择。

## 保存与复核

配置：[mainline_cross_head_ab.yaml](../config/train/mainline_cross_head_ab.yaml)。
训练入口：[train_mainline_cross_head.py](../train_mainline_cross_head.py)；
评估入口：[evaluate_mainline_cross_head.py](../tools/evaluate_mainline_cross_head.py)。

原模型：`outputs/deployment/e10a_g4b_fused.pth`，SHA256：
`53e1b5c6f9bf3973723c7bf4c6ce4ebd5eaeaa45210641df9de8f01e510c790f`。
新 checkpoint 格式为 `mainline_cross_head_v1`，仅保存小型修正层及原模型路径、哈希、
架构、部署配置、损失尺度和划分。加载核对精确原模型哈希并严格检查修正层键；不会用
`strict=False` 掩盖不兼容。推理读取 checkpoint 内的部署配置，不能用外部 YAML 悄悄改融合。

每组只保存 `best_refiner.pth`、`last_refiner.pth` 和逐轮 JSONL 指标。last 另含优化器和
调度器状态。没有逐轮大 checkpoint、特征缓存或逐轮预览；新运行根目录链接已有数据与原模型。

```bash
conda activate sam2_env

# 先做真实 GPU 预检；正式入口也会执行相同检查
python train_mainline_cross_head.py --preflight-only --output-dir outputs/cross_head_preflight

# 两组顺序训练，输出目录不可复用已有 run.json 的运行
python train_mainline_cross_head.py --output-dir outputs/mainline_cross_head_ab

# 训练后分别生成五张验证图的原尺寸最终输出与完整代理指标
python tools/evaluate_mainline_cross_head.py --output-dir outputs/cross_head_evaluation/mainline
python tools/evaluate_mainline_cross_head.py \
  --checkpoint outputs/mainline_cross_head_ab/self/best_refiner.pth \
  --output-dir outputs/cross_head_evaluation/self
python tools/evaluate_mainline_cross_head.py \
  --checkpoint outputs/mainline_cross_head_ab/cross/best_refiner.pth \
  --output-dir outputs/cross_head_evaluation/cross
```

评估增加 `--image-dir data/test` 时只生成无标签推理和输出格式检查，不读取 GT、不评分。
正式判断必须先比较 cross 与同预算 self，再比较原主线；两组一起改善不能归因于交流。
损失下降、预检通过和训练成功都不能代替几何、覆盖率、类别稳定与面积误差的最终验证。

## 本次运行记录

代码分支 `codex/mainline-cross-head`，起点 `f2d0ab6`。服务器根目录：
`/root/autodl-tmp/segmentationv2_cross_head_20260911_1642`。
实际源码清单位于该根目录的 `source_manifest.json`，记录每个运行源文件的 SHA256。
启动前数据盘约 41%，RTX 4090 空闲。

本地已通过 6 项针对性检查，覆盖零输出、self/cross 信息路径、原模型参数及缓冲冻结、
每个修正参数的梯度和更新、严格 checkpoint 读取、旧语义 forward 的兼容性。
真实 GPU 预检也已通过：`train_172` 与 `test_009` 上，self、cross 及保存后重载的 cross
三者，语义 logits、affinity logits、原尺寸融合边界的最大误差全部为 0；最终实例图与
类别映射完全相同。当前原主线生成的两张 PNG 还与上轮保留的 G4b 原尺寸预测字节级一致。
三次真实更新后，两组各28个可训练张量都有非零梯度并发生变化，原模型所有参数及缓冲区
逐张量检查无变化。详见[预检记录](../output/20260911_mainline_cross_head/preflight.json)
与[历史输出核对/启动记录](../output/20260911_mainline_cross_head/launch.json)。

本次定标为 semantic `0.14476349462683386`、affinity `0.2393528910783621`，来源是冻结
主线的26张训练整图；初始五图对应损失为 `0.15314670652151108`、`0.2439535915851593`。
这不是旧 clean60 的定标，不能把两个实验的归一化损失直接横向比较。

正式顺序作业已在上述服务器根目录启动，PID `33809`（仅是启动时记录），输出
`outputs/mainline_cross_head_ab/{self,cross}/`，日志 `outputs/cross_head_train.log`。
`outputs/mainline_cross_head_ab/status.json` 记录当前阶段或最终完成/失败状态。
预检本身的三步更新只用于检查，正式两组重新从相同零修正初始化开始。
训练完成后由用户通知，再执行最终部署比较；没有建立轮询或完成通知任务。
首轮已完成128次更新，训练耗时26.99秒、含验证/保存共29.42秒。两组40轮合计预计约
20–25分钟，实际耗时仍随采样和服务器负载变化。启动状态仅作快照，见
[startup_snapshot.json](../output/20260911_mainline_cross_head/startup_snapshot.json)；
[run.json](../output/20260911_mainline_cross_head/run.json) 保存正式运行配置及共同定标值。
