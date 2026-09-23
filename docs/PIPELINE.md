# 当前管线与产物约定

本工作区用于扩散研究，当前状态与下一组实验以[扩散安排](RGB_DIFFUSION_ROADMAP_20260923.md)
和[D1全量60轮分析](RGB_DIFFUSION_ALL60_ANALYSIS_20260923.md)为准：1000图/15000次更新已完整完成，
首步优于16步，16步细碎颗粒和梯度误差偏高，暂不晋级。
[D2低噪声对照](RGB_DIFFUSION_D2_ANALYSIS_20260923.md)已完成原60轮日程的前20轮/5000更新，
零失败、48张过程图完整；kappa降到0.03后输出近乎原图，不晋级、不自动续训。
[D3起点强化20轮分析](RGB_DIFFUSION_D3_ANALYSIS_20260923.md)：5000更新零失败，48张过程图完整，
实际起点抽样50.13%；首步RGB/梯度误差比为0.607/0.904，完整16步为0.782/1.140。
修复能力已恢复，但多步退化仍在。[逐步测量](RGB_DIFFUSION_D3_TRAJECTORY_20260923.md)进一步确认
两类固定训练内输入、三采样种子的整图误差均首步最低。
[D4在线空间模糊](RGB_DIFFUSION_D4_SPATIAL_20260923.md)已从D3 e20完整状态启动到总60轮，
原D3同预算续训已排在其后；启动核验新增812更新零失败、17张初始过程图。
短链监督仍未实施，不替换已验证分割链。
D1的112张过程图完整保留。用户回报v6复赛0.8448/0.8506，低于v4的0.8465/0.8678，v4继续作为有分数的修复参照。
下面的主线记录保留分叉点状态，不代表原复赛工作区的最新实验进度。

最新交接：[复赛切换与初赛实验收束（2026-09-21）](HANDOFF_SEMIFINAL_20260921.md)。
复赛数据据用户回报位于23411服务器的autodl-fs盘，准确目录与输入结构待核验；本轮未执行数据分析或替换。
下列黑盒成绩均来自初赛，不能当作复赛基准分数。

## 结论与主线

2026-09-19回分未通过：[Stage1最终任务适配](STAGE1_FINAL_TASKS_20260919.md)。
从Stage1共同初始化，两头预热5轮、联合LoRA30轮，再冻结特征分别精修affinity120轮和语义20轮。
该流程替代旧joint-v3，不读取V6/旧固定教师，不是纯删除消融；6600次实际更新、零AMP跳步。
联合e35、几何e57、语义e14按loss选优。内部原始语义与面积改善，七图实例匹配1000→971、铁素体合并增加；
用户回报官方`0.8448/0.7699/80.735`，较统一组合低4.190分，拒绝替换以下已验证基线。
六张几何诊断图全部在联合训练中出现过，内部面积排序未推广到黑盒。
两份测试预测的对应分析同时发现小实例类别变化与划分变化，不能把掉分全部归为affinity或认定joint-v3不可替代。

2026-09-17更新：S用户回报官方`0.8393/0.8714/85.535`，较L高0.765，作为当前总分最佳候选；
S实际仍搭配L旧GT几何，不能把成绩写成“两头新GT”或“S+V”组合。L/V保留回退。
完整链路、旧GT及固定教师依赖、已确认的一致性坐标问题和后续顺序见[训练统筹](TRAINING_ROADMAP_20260917.md)。

后续简化几何基线采用 V，即 `joint-v3 boundary FPN → 120轮长程G2`，
在此前移除独立G0/G0-long/G1的基础上，进一步省去V6独立边界训练。
固定E10a/top2部署，V用户回报官方`0.8430/0.8499/84.645`；L为`0.8416/0.8538/84.770`，
V以总分低0.125分的代价减少一段训练，作为可接受的简化方案。L保留成绩回退及当前S实验固定参照。
完整证据见[V记录](G2_SKIP_V6_20260916.md)及[L记录](G2_LONG_OLDGT_20260914.md)。
当前 [N实验](G2_LONG_NEWGT_20260916.md) 只替换26张人工训练图的补缝GT；原验证loss选优及
其余L配方保持不变。N best115内部六图小幅改善，用户随后回报黑盒`0.8415/0.8476`（按上下文归属N），
折算84.455、低于L 0.315分，尚无官方增益。后续按单变量顺序开展新GT语义监督对照及独立一致性实验。
[V实验](G2_SKIP_V6_20260916.md)单独跳过V6边界训练，使用joint-v3边界FPN及L的旧GT/120轮配方；已完成120轮，loss-best为115。
固定E10a/top2内部六图匹配777→779、GT惩罚mIoU 0.71033→0.71663、铁素体面积误差34.33%→33.85%，
但珠光体多余预测与分裂/合并略增。官方结果支持上述简化选择，不构成统计等效或显著提升证明。
[S语义新GT配对实验](SEMANTIC_NEWGT_20260916.md)两组均完成20轮、1240次实际更新和零AMP跳步，
新GT按共同验证loss选中第20轮，旧GT为第7轮。固定L部署，语义7图原始像素mIoU为0.95763/0.94102，
新GT收益也存在于原标注覆盖区；最终实例表现却近似，面积误差均高于E10a/L内部基准。
保留实际语义25/7划分及原定L对照；S官方面积项上升而mIoU略降，内部代理未正确预测官方方向。
S-align用户回报`0.8407/0.8620/85.135`，相对S总分−0.400、mIoU略升，接受正确对齐作为后续研究起点；
V-noSAM2为`0.8362/0.8181/82.715`，相对V两项均下降、总分−1.930，保留64源图SAM2掩码监督。
两项分别完成20轮/1240次、120轮/3120次有效更新且0跳步，来源与冻结契约通过。
完整证据见[S-align / V-noSAM2记录](ALIGN_NOSAM2_EXPERIMENT_20260917.md)。
2026-09-19用户回报[S-align+V统一组合](UNIFIED_ALIGNMENT_AUDIT_20260919.md)官方`0.8416/0.8569/84.925`：
较S-align+L低0.210分、较V高0.280分，接受其作为后续统一研究对照，原S仍为最高分回退。
结果支持两项修改组合使用，没有证明统计等效、固定教师项有效或joint-v3可删除。
当前Stage1最终任务替代黑盒未通过，拟先做固定权重的语义/affinity输出交换定位；固定教师净贡献及EMA实验仍待独立检验，
后续安排见[训练统筹](TRAINING_ROADMAP_20260917.md)，不恢复已删掉的独立G阶段。
V6文件在语义冷启动中的载体引用仍待独立替换为已核对相同的joint-v3语义/LoRA并验证。
本轮未修改默认推理文件；以下保留历史部署与实现约定。

历史黑盒基线为“E10a 单语义模型 + G4b 8 通道 affinity → gated boundary → `high=0.65`、
seal2、局部重建 → 受阻分水岭”，得分 `0.8381/0.8408/83.94`。E10a 冻结 V6 特征并
冷启动完整高分辨率语义解码器，现已晋级；E9 的 `0.8421/0.7917/81.69` 保留为历史回退，
不做连续融合。V6/B2 边界路线继续作为几何回退。

当前推荐部署产物为 `outputs/deployment/e10a_g4b_fused.pth`：单个共享 SAM2+LoRA encoder
同时连接 E10a semantic decoder 和 G4b affinity decoder。组合包参数量 `81.667394M`、大小
`326836899` bytes；保存后重新构建的 semantic/affinity logits 最大绝对误差均为 `0.0`，
`test_009` 的实例 PNG 与类别 JSON 也和原三 checkpoint 管线字节级一致。

G3/G4b 尚不能单独证明黑盒竞赛成绩提升，测试目检仍以欠分割为主要风险；GT 前景上的 Oracle
图重建仅作诊断。graph-v1 `area200` 黑盒为 `0.8268/0.8365/83.17`，未超过 E10a watershed；
graph-v2 `area150` 因笔直、失真的归并边界被目检淘汰。G7 在固定协议测试 A/B 中进一步
减少实例并加重欠分割风险，已停止晋级。当前候选检验 `SSL LoRA → semantic/affinity 双头`
短训练链，主线部署仍完全不变。详见
[AFFINITY_GRAPH_AB_20260828.md](AFFINITY_GRAPH_AB_20260828.md)；历史 affinity 审计见
[AFFINITY_DEPLOYMENT_EVALUATION.md](AFFINITY_DEPLOYMENT_EVALUATION.md)，短链实验见
[DIRECT_SSL_SEMANTIC_AFFINITY.md](DIRECT_SSL_SEMANTIC_AFFINITY.md)。

`outputs/stage2_center_heatmap/best_model_stage2.pth` 只保留为负面对照。现有中心 GT 由每个
Labelme polygon 生成一个种子，polygon 与物理晶粒并不等价；同时中心损失和边界损失共享
`boundary_fpn`，实测造成背景雾化、铁素体大块欠分割、珠光体碎裂及薄环嵌套。

当前 B2 架构实验以 V6 权重为语义锚点，并满足：

1. 语义路径冻结或低学习率保护；
2. 边界 refine 路径独立训练，不接收不可靠中心标签的梯度；
3. 若重试辅助任务，使用独立 FPN/stop-gradient，并先验证 GT 与物理实例的一致性；
4. 后处理把高置信边界作为硬障碍，而不是仅改变分水岭可视化灰度。

## 2026-08-21 新发现：颜色先验

已完成 32 张有标签训练图的 ferrite/pearlite 颜色分布分析，详细结果见
[`docs/COLOR_SEPARABILITY.md`](COLOR_SEPARABILITY.md)。结论是 GT 内部存在很强的明度分离：
Lab `L*` 的 pooled 单阈值平衡准确率约 `0.9917`，包含边界混色仍约 `0.9846`；leave-one-image-out
平均约 `0.9912`（含边界约 `0.9845`）。

该结果是颜色先验的重要证据，但不能替代语义头：统计来自 GT 区域，且不同图像的最佳明度阈值
约在 `63.7~81.0` 间变化。后续应采用固定的无标签 holdout monitor，并将 Lab `L*` 先验作为
自适应软辅助或融合信号，不能把固定全局阈值直接写入实例分割主路径。

后续对话进程接续本项目时，必须先查看 `docs/COLOR_SEPARABILITY.md`，特别是其中的限制条件和
“不能替代语义头”的结论；当前 E1/V6 主线、实例 ID `<=65535` 约束和测试集无标签原则保持不变。

## 入口

```bash
conda activate sam2_env

# 当前固定 G4b 部署基线
python tools/run_affinity_submission.py \
  --config config/inference/final_affinity_g4b_high065.yaml

# E10a + G4b：将共享主干和两个分支一次性打包
python tools/build_fused_deployment_checkpoint.py \
  --config config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml \
  --output outputs/deployment/e10a_g4b_fused.pth

# 推荐提交推理入口：只加载一个组合 checkpoint
python tools/run_fused_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml \
  --checkpoint outputs/deployment/e10a_g4b_fused.pth

# 当前 E7b-A 语义专项训练（V6 初始化、decoder-only、20 epoch）
python train_stage2.py \
  --config config/train/stage2_semantic_e7b_decoder20.yaml

# 当前 E9：冻结 V6/G4b，只训高分辨率语义残差（20 epoch）
python train_stage2.py \
  --config config/train/stage2_semantic_e9_highres20.yaml

# E10a：冻结 V6 LoRA/G4b，冷启动完整高分辨率语义解码器（20 epoch）
python train_stage2.py \
  --config config/train/stage2_semantic_e10a_cold20.yaml

# SSL 直达双头：先检查数据门槛，再执行 5 epoch head warm-up + 20 epoch joint LoRA
python train_direct_semantic_affinity.py \
  --config config/train/direct_ssl_semantic_affinity.yaml --check
python train_direct_semantic_affinity.py \
  --config config/train/direct_ssl_semantic_affinity.yaml

# E7b 完成后：同一 G4b 几何，只替换 semantic decoder
python tools/run_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_e7b.yaml

# 历史 V6 参考推理
python inference.py --config config/inference/v6_reference.yaml

# Stage 1 全监督（LoRA 可训练）
python train.py --config config/train/stage1_lora.yaml

# Stage 2 B2；V6 作为冻结语义锚点，落地独立高分辨率 refine head
python train_stage2.py --config config/train/stage2_refine_v6.yaml \
  --phase boundary --tag refine_v6_b2

# B2 单变量重试：V6 初始化 + refine-only + 物理显微增强（5 epoch）
python train_stage2.py --config config/train/stage2_refine_v6_physaug.yaml \
  --phase boundary --tag refine_v6_b2_physaug

# Stage-0 / E0：先验证 B2 在 310 次纯监督更新下确实能够学动
python train_stage2.py --config config/train/stage2_refine_v6_stage0_control.yaml \
  --phase boundary --tag refine_v6_stage0_control

# Stage-0 Long：20 epoch/1240 更新，观察纯 refine 收敛与背景雾化趋势
python train_stage2.py --config config/train/stage2_refine_v6_stage0_long.yaml \
  --phase boundary --tag refine_v6_stage0_long

# 对应的两档质量感知 TTA（训练完成后）
python inference.py --config config/inference/b2_quality_aware.yaml
```

推理常用参数可直接从 CLI 覆盖，不再复制临时 YAML：

```bash
python inference.py --config config/inference/v6_reference.yaml \
  --boundary-threshold 0.35 --min-instance-area 50 --no-center-seeds \
  --output_dir outputs/inference/<name>
```

每个推理目录都会生成 `inference_manifest.json`，记录 checkpoint、架构、实际阈值和三类实例统计。

## 物理增强与质量感知推理边界

- 训练增强只模拟显微成像中可解释的曝光/白平衡变化、轻度失焦、采样分辨率下降、低频照明和
  低对比抛光划痕；单张图只组合 1~2 项，保留足够干净样本。
- 划痕保持原 GT，作为“图像强线条不一定是晶界”的 hard negative；不再使用规则圆形遮罩。
- 推理只分 `standard`/`weak` 两档。弱档保留原图 logits，并融合一张确定性校正视图；所有
  阈值偏移由配置显式给出，便于逐项关闭和复现。
- 分档不以预测实例数、铁素体平均面积、薄环或嵌套现象为目标，也不会跨测试集拟合统计量。
- 实例图使用单通道 `uint16`，每图最多 65535 个非零 ID。只有候选超过格式上限时才按局部
  邻接关系合并最小区域；该保护仅处理输出格式上限，不反向改变分水岭参数，也不再把正常的
  第 256 个及后续实例强制合并。

## Checkpoint 契约

新 checkpoint 格式版本为 2，必须包含：

- `architecture`：encoder、FPN 通道、`boundary_refine`、`center_head`、LoRA；
- `provenance.git_commit`：训练代码版本；
- `config`：完整生效配置；
- decoder/LoRA 权重、epoch 与 best score；
- 中间 checkpoint 额外包含 optimizer/scheduler，用于恢复训练。

检查权重而不构建 SAM2：

```bash
python tools/inspect_checkpoint.py outputs/stage2_v6/best_model_stage2.pth
```

推理和 `--resume` 默认严格核对架构。`--allow-architecture-mismatch` 仅用于明确的消融，不能作为
普通兼容开关。Stage 2 的跨架构初始化仍允许宽松加载，但日志必须检查 missing/unexpected keys。

## 目录职责

```text
config/default_config.yaml       当前 V6 参考基线
config/inference/                同架构推理配置
config/train/                    Stage 1 / Stage 2 训练配置
config/experiments/              改架构/训练目标的实验配置
outputs/stage2_v6/               V6 参考 checkpoint
outputs/stage2_center_heatmap/   中心热图负面对照 checkpoint
outputs/runs/                    配置、指标、环境及 best 权重硬链接
downloads/                       下载到本机的目检结果（不提交）
```

## 服务器保留策略

- 活跃/基准实验：保留 `best_model*.pth`、`metrics.csv`、配置和少量 monitor；
- 负面对照：保留一个 best 权重和一组有代表性的推理可视化；
- 只有计划恢复训练的运行才保留一个中间 epoch（默认最后一个）；
- 删除其余周期 checkpoint、重复 run 权重、smoke 输出和已判废阈值扫描；
- 伪标签缓存只有在配置仍引用且生成代价明显时保留。
