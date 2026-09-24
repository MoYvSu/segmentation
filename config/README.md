# 配置目录

新增`train/rgb_diffusion_d5b.yaml`：继承D5a参数配方，从零训练全1000图、60轮，
仅增加25%批次的两步短链监督，独立随机流、跨步切梯度、辅助权重0.25；不继承历史权重。
入口`tools/run_rgb_d5b.py`，短目录`outputs/d5b/`；GPU配对短测已通过，正式后台队列已启动，
见[D5b约定](../docs/RGB_DIFFUSION_D5B_20260924.md)。

新增`train/rgb_diffusion_d5a.yaml`和`train/rgb_diffusion_d5a_control.yaml`：
两组随机初始化、全1000图、60轮，唯一训练差异为强弱端点平滑模糊；不加载D4权重。
队列入口`tools/run_rgb_d5a.py`，自动保存固定过程图，见[D5a约定](../docs/RGB_DIFFUSION_D5A_20260924.md)。
两组已完成60轮／15000更新、零失败及配对检查；[完成分析](../docs/RGB_DIFFUSION_D5A_ANALYSIS_20260924.md)
已回分control 0.8511／0.8486（84.985）、D5a 0.8469／0.8560（85.145）。
端点配方净+0.160分，仍未超过baseline或旧v4；不更改默认部署配置。

新增`train/backend_d4.yaml`：冻结D4首步与不带D4两组后端微调，均从S-align＋V初始化，
训练LoRA和两个任务头；同在线退化、全32人工与64既有SAM2源、60轮，无代理或验证选优。
入口`tools/run_backend_ablation.py`自动串行完成训练、推理和对比渲染，见[实验约定](../docs/BACKEND_D4_ABLATION_20260924.md)。

输出目录命名：新建运行目录最多四段（按下划线分隔，时间戳也计一段），例如
`rgb_restoration_diffusion_d4` 或 `diffusion_d4_20260923`。增强类型、轮数、monitor等细节
写入配置和运行记录，不拼入目录名。正在运行的旧长目录可提供短名称入口，不为改名重启训练。

扩散正式候选入口：`train/rgb_restoration_diffusion_d1_all60_monitored.yaml`，全1000图、60 epoch、
从零训练，固定过程图同时记录首步与16步。`train/rgb_restoration_diffusion_d1_overfit3200.yaml`
的4图短测已通过工程门槛，见[短测判定](../docs/RGB_DIFFUSION_SHORT_DECISION_20260923.md)；
正式60轮/15000更新已完成、零失败；[完整分析](../docs/RGB_DIFFUSION_ALL60_ANALYSIS_20260923.md)
认为当前16步版本暂不晋级。
当前[D2低噪声对照](../docs/RGB_DIFFUSION_D2_K003_20260923.md)入口为
`train/rgb_restoration_diffusion_d2_k003_all60_monitored.yaml`，只改kappa为0.03。
已按`--stop-after-epoch 20`完成首段，保持原60轮学习率日程，20轮/5000更新后正常退出。
[D2分析](../docs/RGB_DIFFUSION_D2_ANALYSIS_20260923.md)确认输出近乎原图，暂不晋级或继续。
[D3起点强化](../docs/RGB_DIFFUSION_D3_TERMINAL50_20260923.md)入口为
`train/rgb_restoration_diffusion_d3_terminal50_all60_monitored.yaml`，仅新增
`train.timestep_sampling: terminal_half`（t=16占50%，其余15步均分50%）。
已从零完成`--stop-after-epoch 20`，kappa仍为0.03，原60轮学习率日程与48张过程图完整。
[D3结果](../docs/RGB_DIFFUSION_D3_ANALYSIS_20260923.md)显示首步恢复有效，完整16步仍增加梯度误差；
[逐步与空间模糊诊断](../docs/RGB_DIFFUSION_D3_TRAJECTORY_20260923.md)已完成。
[D4空间增强](../docs/RGB_DIFFUSION_D4_SPATIAL_20260923.md)入口为
`train/rgb_restoration_diffusion_d4_spatial_all60_monitored.yaml`，通过`--fork-spatial-from`
从D3 e20完整状态启动到总60轮；原D3也已续到60轮作为同预算对照，两组均已完成。
[结果](../docs/RGB_DIFFUSION_D4_ANALYSIS_20260923.md)：空间合成诊断明显改善，均匀模糊小幅退步，
多步采样问题仍在；D4默认输出短名为`outputs/rgb_restoration_diffusion_d4`。
原配方续训保持原配置，不能把总轮数改成40；新配方分叉严格限制只有空间模糊变化。
短链监督仍未实施。
旧`all60.yaml`保留原800次短测预算；延长必须用`--extend-overfit-from`
并指定新的输出目录，不能绕过严格配置检查。

2026-09-21：[复赛交接](../docs/HANDOFF_SEMIFINAL_20260921.md)。本轮未更改测试数据路径或默认推理配置；
复赛数据目录待下轮核验。以下成绩及68图包均为初赛记录，复赛不得直接沿用固定文件数断言。

2026-09-17：S用户回报官方`0.8393/0.8714/85.535`，为当前总分最佳候选；实际为S语义+L旧GT几何，S+V尚未组合核验。
V已获用户回报官方`0.8430/0.8499/84.645`，接受省去V6独立边界训练，作为后续简化几何基线。
L（E10a/top2，官方`0.8416/0.8538/84.770`）保留成绩回退及当前S语义实验的固定参照。
`train/affinity_geometry_g2_direct_long_newgt.yaml` 继承 L 的120轮配方，只覆盖26张人工训练图的
补缝 GT，原人工验证loss选优不变；见 [N实验记录](../docs/G2_LONG_NEWGT_20260916.md)。
N已完成120轮及固定六图部署比较，用户回报黑盒`0.8415/0.8476`（按上下文归属N），折算84.455。
新GT语义实验已完成，固定L几何的内部代理未获益，但官方总分提高0.765；面积稳定性与GT独立贡献仍待核验，默认推理文件未切换。
S-align已回报`0.8407/0.8620/85.135`，接受正确对齐作为后续语义研究起点；
V-noSAM2回报`0.8362/0.8181/82.715`，不接受取消64源图SAM2监督。
2026-09-19组合入口`experiments/affinity_semantic_aligned_v_top2_20260919.yaml`已完成推理和68图打包：
S-align e13 + V e115，top2/high0.65等保持不变；用户回报官方`0.8416/0.8569/84.925`。
该组合成为后续统一研究对照，原S的85.535仍为最高分回退；未自动修改默认部署入口。
见[统一推理与审计](../docs/UNIFIED_ALIGNMENT_AUDIT_20260919.md)。
用户随后选择先执行[Stage1最终任务替代](../docs/STAGE1_FINAL_TASKS_20260919.md)：
`train/stage1_final_tasks_20260919.yaml`联合适配LoRA，接
`train/stage1_final_geometry120_20260919.yaml`和`train/stage1_final_semantic20_20260919.yaml`。
串行入口为`tools/run_stage1_final_tasks.py`，正式训练已完成6600次实际更新；按loss选中联合e35、几何e57、语义e14，不引入EMA。
部署配置为`experiments/stage1_final_tasks_top2_20260919.yaml`；官方回报`0.8448/0.7699/80.735`，
比统一组合84.925低4.190、比原S低4.800，拒绝晋级。保留该配置供诊断，不修改既有基线。

本轮入口为 `train/stage2_semantic_gt_aligned20.yaml`（仅开启共享空间坐标，20轮/1240次计划更新）与
`train/affinity_geometry_g2_no_sam2.yaml`（取消SAM2、人工重采样补至52次/轮，120轮/3120次计划更新）。
两项完整训练、固定部署与用户回报黑盒均已完成；保留语义修复和V/SAM2几何基线，见[实验记录](../docs/ALIGN_NOSAM2_EXPERIMENT_20260917.md)。
配套 `experiments/affinity_semantic_aligned_l_top2_20260917.yaml` 固定L几何；
`experiments/affinity_no_sam2_top2_20260917.yaml` 固定E10a语义，不合并两个训练变化。

`train/stage2_semantic_gt_new20.yaml`与`train/stage2_semantic_gt_control20.yaml`为S配对实验：
两组继承E10a冷启动、冻结共享特征与固定教师，各20轮，实际25训练/7验证，按共同新GT验证loss选优。
仅训练GT与产物路径不同；共同关闭GT依赖暗边增强、AMP初始scale=256，记录实际更新/跳步。
两组均完成1240次实际更新，loss-best为新GT20/旧GT7。
`experiments/affinity_semantic_newgt_l_top2_20260916.yaml`固定新语义第20轮、L best115及top2，
用于用户授权的68图提交核验；不叠加V几何或改动后处理。
见[S实验记录](../docs/SEMANTIC_NEWGT_20260916.md)。

`train/affinity_geometry_g2_skip_v6.yaml`为V消融：继承L，使用joint-v3参考权重与
`reference_boundary_fpn`初始化，旧GT/预算不变。部署对照入口为
`experiments/affinity_skip_v6_top2_20260916.yaml`；120轮完成，loss-best为115，内部总体接近L且部分指标略好，
珠光体多余预测略增。V官方总分较L低0.125分，接受以此取舍减少独立阶段；见[V实验记录](../docs/G2_SKIP_V6_20260916.md)。

历史部署基线为 `inference/final_affinity_g4b_high065.yaml`：V6 语义锚点 + G4b 8 通道
affinity，使用 `high=0.65`、seal2、局部重建与受阻分水岭。训练 checkpoint 只能按固定的
完整部署路径验证晋级，Oracle GT 前景重建仅作诊断。完整协议见
`docs/AFFINITY_DEPLOYMENT_EVALUATION.md`；V6/B2 配置仍保留为回退。

黑盒确认的 E9 语义实验为 `train/stage2_semantic_e9_highres20.yaml`：以 V6 语义为零漂移锚点，
冻结 semantic FPN/head、boundary、LoRA 与 G4b affinity，只训练 256→512→1024 的高分辨率
residual。训练增加整实例平均概率目标和温和细实例权重，不依赖中心或最高置信像素。部署配置为
`experiments/affinity_g4b_high065_semantic_e9_highres.yaml`，hard-majority 对照在同名 `_hard`
配置；详见 `docs/SEMANTIC_EXPERIMENT_E9_20260828.md`。E7b/E8 保留为历史对照。

`train/stage2_semantic_e10a_cold20.yaml` 是完整语义解码器冷启动实验：保留并冻结 V6 LoRA
特征及 G4b 几何，随机重置 `seg_fpn`、`seg_branch` 和高分辨率语义路径；重置前复制的固定
V6 教师只在高置信无标签像素提供衰减蒸馏。部署配置
`experiments/affinity_g4b_high065_semantic_e10a_cold.yaml` 已获黑盒 mIoU `0.8381`、面积项
`0.8408`、总分 `83.94`，是当前单语义模型主线；E9 保留为历史回退，不执行连续融合。详见
`docs/SEMANTIC_EXPERIMENT_E10A_20260828.md`。

`tools/run_affinity_graph_ab.py` 是 GT-free 历史几何筛查工具：graph-v1
`short=0.40/area200` 黑盒总分为 `83.17`，未超过 E10a watershed；graph-v2 `area150`
出现不自然的笔直边界，已按目检淘汰，不再提交。详见
`docs/AFFINITY_GRAPH_AB_20260828.md`。

`train/affinity_geometry_g7_highres_short.yaml` 现只保留为历史对照：固定协议测试 A/B 显示其
相较 G4b 进一步减少实例、加重欠分割风险，不再作为当前晋级目标。

`train/direct_ssl_semantic_affinity.yaml` 是当前短训练链候选：从 SSL LoRA 同时冷启动 E10a 式
高分辨率 semantic head 与 8 通道 affinity head；先冻结 LoRA 预热，再联合微调。人工样本在
同一增强下监督两头，经人工审核的 SAM2 候选只监督无类别 affinity；完整契约见
`docs/DIRECT_SSL_SEMANTIC_AFFINITY.md`。首轮 Arm A 使用同目录下的 `_no_sam2.yaml`，先隔离检验
最短链路；SAM2 数据审核完成后再运行原配置作为 Arm B。

当前类别纠错候选为 `experiments/affinity_g4b_high065_semantic_dual_e7c_relaxed.yaml`：固定
V6 前景与 G4b 实例几何，仅用 E7b core 分数覆盖部分 V6 hard vote。阈值由缓存置信度扫参
产生，不按实例面积拦截；严格版配置继续保留为反面对照。详见
`docs/SEMANTIC_EXPERIMENT_E7C_20260828.md`。

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
所有推理配置均要求 `max_instance_id <= 65535`，最终实例 PNG 必须以单通道 `uint16` 写出。

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
