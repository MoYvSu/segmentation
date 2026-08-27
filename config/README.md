# 配置目录

当前部署基线为 `inference/final_affinity_g4b_high065.yaml`：V6 语义锚点 + G4b 8 通道
affinity，使用 `high=0.65`、seal2、局部重建与受阻分水岭。训练 checkpoint 只能按固定的
完整部署路径验证晋级，Oracle GT 前景重建仅作诊断。完整协议见
`docs/AFFINITY_DEPLOYMENT_EVALUATION.md`；V6/B2 配置仍保留为回退。

当前训练实验为 `train/stage2_semantic_e7b_decoder20.yaml`：从 V6 初始化，只训练 semantic
decoder 20 epoch，冻结 boundary/LoRA/affinity，加入实例等权核心损失、暗边定向增强与高置信
无标签一致性。训练后使用 `experiments/affinity_g4b_high065_semantic_e7b.yaml` 在同一 G4b
实例几何上做严格对照。详见 `docs/SEMANTIC_TRAINING_E7B_20260827.md`。

`train/affinity_geometry_g4_manual_gap.yaml` 是历史断边合并单变量实验：完全复用 G3 的
G2 初始化、数据比例、增强、学习率和 20 epoch，只对人工 LabelMe 样本启用
“实例与未覆盖带之间为负 affinity”；SAM2 未覆盖区和人工 `0-0` 像素对继续 ignore。
设计与判定标准见 `docs/AFFINITY_G4_MANUAL_GAP.md`。G4 完整权重已证实过强；
`train/affinity_geometry_g4b_gap_weight020.yaml` 只把新增人工缺口负边降权至 `0.20`，
其余设置不变，产物 G4b 现作为部署几何基线。

- `default_config.yaml`：可训练、可推理的当前 V6 参考基线；路径跨本机/服务器可移植。
- `inference/`：只改变推理输出与后处理参数，不改变模型架构。
- `train/`：明确区分 Stage 1 与 Stage 2 的可训练参数和输出目录。
- `experiments/`：会改变架构、监督目标或训练策略的实验。
- `stage2_center_heatmap.yaml`：旧命令兼容的完整历史快照；已判废，不作为主线。

配置可用 `_base` 递归继承。`paths.project_root: auto` 默认定位当前仓库；临时覆盖可设置
`SEGMENTATION_PROJECT_ROOT`，无需为 Windows/Linux 分别维护 YAML。

推理会严格比较配置和 checkpoint 的 `boundary_refine`、`center_head`、LoRA 等架构字段。
只有明确进行消融时才使用 `--allow-architecture-mismatch`。

历史 B2 边界主线使用 `train/stage2_refine_v6.yaml`：从 V6 best 初始化，只增加独立高分辨率
refine residual，关闭中心头并保持原后处理不变；当前不再把它描述为唯一主线。

当前建议的下一轮单变量实验是 `train/stage2_refine_v6_physaug.yaml`：继续从 V6 best
初始化 B2，但在 5 个 epoch 内只训练 refine head，关闭无标签一致性，加入显微成像物理增强。
增强每次只抽取 1~2 项（曝光/白平衡、失焦、降采样、低频照明或低对比划痕），不制造
圆形硬遮罩，也不改 GT。

`inference/b2_quality_aware.yaml` 是配套的低复杂度推理实验：固定几何 TTA，并按当前单图的
亮度、对比度、清晰度和偏色分为 `standard`/`weak` 两档。弱档只融合一张确定性增强视图并
应用固定的小幅边界阈值偏移；不读取跨图统计，不按实例数、平均面积或环形拓扑闭环调参。
所有推理配置均要求 `max_instance_id <= 255`。

`train/stage2_refine_v6_stage0_control.yaml` 是物理增强消融之前的 E0 可学习性控制：
固定 seed 42、每 epoch 62 个监督 step、共 5 epoch（310 次更新），关闭无标签流和
物理增强，只训练零初始化 B2 refine。运行指标会额外记录 refine 梯度/残差/权重变化，
并验证 coarse、语义与冻结 LoRA 的最大参数变化严格为 0。

`train/stage2_refine_v6_stage0_long.yaml` 将同一控制实验延长至 20 epoch/1240 次更新，
前段保持 refine LR `5e-5`、末段衰减至 `2e-5`，每 5 epoch 保存一次 checkpoint 和
monitor。验证指标额外记录边界正/背景概率、概率间隔，以及阈值 0.35 下的召回与背景
假阳性率，用于区分真实边界增强和雾状背景同步抬升。

`train/stage2_refine_v6_stage0_continue15.yaml` 从 Long-20 的最佳 checkpoint 初始化，
继续 15 epoch 纯 refine 训练。LR 从 `2e-5` 平滑接续并衰减至 `5e-6`，仍冻结语义、
LoRA 与 coarse boundary；用于确认 Long-20 末端尚未收敛的收益能否继续，同时避免把
联合解冻引入为第二个实验变量。

`train/stage2_refine_v6_e1_physaug15.yaml` 从 Continue-15 best 初始化，在纯 refine 已进入
平台期后进行 15 epoch 物理外观增强实验。只训练 refine head，LR 从 `1e-5` 衰减至
`2.5e-6`；增强保持 40% 干净样本，每张增强图只抽取 1~2 项显微成像退化，不修改 GT
几何，也不使用规则硬遮罩或高斯噪声。

`train/stage2_refine_v6_e2_coarse_unfreeze10.yaml` 从 E1 best 初始化，保持同一增强和损失，
进行 10 epoch 低学习率联合边界训练。refine LR 从 `5e-6` 衰减至 `1.25e-6`，coarse
boundary 始终使用其 5%；语义与 LoRA 继续冻结，用于隔离 coarse 表征适配的收益和风险。

`train/stage2_refine_v6_e3_ridge10.yaml` 回到 E1 best，并继续严格冻结 coarse boundary、
语义与 LoRA。唯一实验变量是局部边界脊线损失：允许 GT 附近 1px 定位误差，
要求核心附近存在高置信峰值，同时抑制 5px 邻域真背景的雾状响应。保持 E1 物理增强，
训练 10 epoch，用于单独验证“窄、亮、连续”边界监督。

`train/stage2_refine_v6_e3b_balanced_ridge10.yaml` 是 E3 的置信度校正实验，仍从
E1 best 独立初始化。正边界峰值目标提高至 logit `2.0`（概率约 0.88）；
背景环只抑制高于 logit `-0.62`（概率约 0.35）的响应，且权重降为 0.25。
其余训练路径与 E3 相同，用于验证能否保留背景误报收益并恢复高置信边界。

`train/stage2_refine_v6_e4_relative_ridge10.yaml` 改用局部相对脊线损失，只要求
GT 附近的边界峰值比 5px 内最强真背景高 `1.5` logit。该损失对全图统一
加减 logit 严格不变，不能像 E3/E3b 一样通过整体变暗或变亮来获利。
仍从 E1 best 开始，其余训练和物理增强保持不变。

`train/gda_mim_g0a.yaml` 使用赛方无标签图进行生成式掩码重建预训练。
冻结 E1 SAM2/LoRA，只训练四尺度 GDA 和临时重建解码器；预训练后丢弃解码器。
`config/monitor/unlabeled_holdout_v1.txt` 中的 24 张图不进入训练，专用于固定无标签 monitor。
