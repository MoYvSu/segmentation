# 当前管线与产物约定

## 结论与主线

当前几何基线为“G4b 8 通道 affinity → gated boundary → `high=0.65` + seal2 + 局部重建
→ 受阻分水岭”。E9 已获黑盒 mIoU `0.8421`、铁素体面积项 `0.7917`；E10a 冻结 V6
特征并冷启动完整高分辨率语义解码器，当前目检优先但尚未黑盒确认。部署决策坚持一个语义
模型：E10a 为候选、E9 为回退，不做连续融合；V6/B2 边界路线继续作为几何回退。

G3/G4b 尚不能证明黑盒竞赛成绩提升，测试目检仍以欠分割为主要风险；GT 前景上的 Oracle
图重建仅作诊断。当前 E9 在冻结 V6/G4b 的前提下学习输入分辨率语义残差，并以整实例概率
聚合处理纤细/非凸实例错配。直接方向 affinity 图切分的 10 图筛查未晋级，详见
[AFFINITY_GRAPH_AB_20260828.md](AFFINITY_GRAPH_AB_20260828.md)；历史 affinity 审计见
[AFFINITY_DEPLOYMENT_EVALUATION.md](AFFINITY_DEPLOYMENT_EVALUATION.md)。

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
“不能替代语义头”的结论；当前 E1/V6 主线、实例 ID `<=255` 约束和测试集无标签原则保持不变。

## 入口

```bash
conda activate sam2_env

# 当前固定 G4b 部署基线
python tools/run_affinity_submission.py \
  --config config/inference/final_affinity_g4b_high065.yaml

# 当前 E7b-A 语义专项训练（V6 初始化、decoder-only、20 epoch）
python train_stage2.py \
  --config config/train/stage2_semantic_e7b_decoder20.yaml

# 当前 E9：冻结 V6/G4b，只训高分辨率语义残差（20 epoch）
python train_stage2.py \
  --config config/train/stage2_semantic_e9_highres20.yaml

# E10a：冻结 V6 LoRA/G4b，冷启动完整高分辨率语义解码器（20 epoch）
python train_stage2.py \
  --config config/train/stage2_semantic_e10a_cold20.yaml

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
- 实例图使用 `uint8`，每图最多 255 个非零 ID。候选超过上限时保留 254 个最大区域，其余
  区域汇入 ID 255，并对全部汇入像素重新进行语义投票；该保护仅处理输出格式上限，不反向
  改变分水岭参数。

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
