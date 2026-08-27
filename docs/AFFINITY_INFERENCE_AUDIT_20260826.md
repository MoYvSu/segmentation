# Affinity 推理、阈值与 LoRA 审计（2026-08-26）

## 结论

当前“边界图肉眼清楚但实例仍欠分割”不是矛盾。G4b 的大部分真实边界响应很强，
但拓扑分割只需一个局部缺口就会合并两个实例；整图缩放后的 8-bit monitor 很难显示
稀疏的低置信泄露点。现阶段优先级应为修正 marker 拓扑和 255 上限处理，再考虑扩大
encoder 可训练参数。

## 阈值分布

工具：`tools/audit_affinity_thresholds.py`。固定六张有标签验证图，使用人工未覆盖带作为
边界诊断，但不参与 checkpoint 选择。

| checkpoint | 融合真边界 mean/q05/q10/q50 | 实例内部 mean/q90 | F1 最优阈值 |
|---|---|---|---:|
| G2 best | 0.771/0.256/0.476/0.850 | 0.177/0.592 | 0.70 |
| G3 best | 0.762/0.215/0.418/0.852 | 0.186/0.596 | 0.65 |
| G4b latest | 0.795/0.350/0.527/0.869 | 0.199/0.625 | 0.70 |

G4b 在 0.55/0.60/0.65/0.70 下的真边界像素漏检率分别为
`11.0%/13.5%/16.8%/20.9%`。同时实例内部最高约 10% 的响应已达到 `0.625` 以上，
所以单纯降低全局阈值会把弱边补回和内部雾状假边一起引入。

原始 short affinity 的同实例/不同实例/人工未覆盖带均值，G4b 为
`0.780/0.077/0.171`。人工缺口比直接不同实例边更难，仍有明显高 affinity 长尾。

## 推理实现问题

### 1. marker 可沿图像边缘绕行

`utils/affinity_deployment.postprocess` 原先没有把 `marker_border_seal_width` 传入
`boundary_watershed_separation`。边界线在画幅边缘终止时，互不相邻的区域可能沿最外圈
连接成同一个 marker。固定 G4b+0.65 的 A/B：

| marker 封边 | 预测实例 | 有效匹配 | 有效 mIoU | GT 惩罚 mIoU | 面积误差 | 代理总分 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 px | 1166 | 690 | 0.8514 | 0.6463 | 0.0903 | 88.0551 |
| 1 px | 1201 | 725 | 0.8513 | 0.6790 | 0.1221 | 86.4609 |
| 2 px | 1229 | 749 | 0.8543 | 0.7039 | 0.1508 | 85.1717 |

封边显著增加有效匹配，证明边缘绕行真实存在；固定阈值下面积误差变差，需要与阈值
联合校准。封 2 px 后，阈值 0.72 得到：总分 `90.7592`、有效匹配 `663`、有效 mIoU
`0.8503`、面积误差 `0.0351`、单图宏平均 `86.7312`。这略高于 G2+0.60 的
`90.7304`，宏平均也高于 G3+0.60 的 `85.7916`，但仍只是六图小样本正信号。

### 2. 255 上限的溢出桶破坏实例连通性

当前候选超过 255 时，最大的 254 个实例保留，其余所有区域统一写成 ID 255。
验证图 `train_172` 中：

- G2+0.55 的 ID 255 包含 24 个不连通组件、20,797 像素；
- G4b+0.65 的 ID 255 包含 40 个不连通组件、42,313 像素。

这会把跨全图的多个小区域当成一个实例，污染类别投票、IoU 和平均面积。前十张测试
monitor 都没有达到 255，因此它不是这些图欠分割的原因，但必须在正式提交前改为基于
区域邻接的逐块合并，不能继续使用全局 overflow bucket。

### 3. 代理评估的限制

`utils.instance_metrics` 按规则文本计算同类、一对一、IoU>=0.5 的有效实例平均 IoU，
实现方向基本正确。但应同时保留 `gt_penalized_miou`、有效匹配数和单图宏平均，因为：

- 有效匹配 mIoU 本身不惩罚未匹配实例；
- 全数据汇总铁素体面积可能发生跨图误差抵消；
- Hungarian 先最大化 IoU 总和、后过滤 0.5，与组委会可能采用的贪心最大 IoU或
  “先最大化有效匹配数再最大化 IoU”仍存在边界行为差异。

## LoRA 评估

系统并非没有 LoRA：V6 语义 checkpoint 已包含 rank-16 Hiera attention LoRA，
affinity 训练读取其特征但把 reference model 和 LoRA 全部冻结，只训练独立 FPN decoder。

不建议直接解冻现有 V6 LoRA。它同时决定已经验证优秀的语义特征，少量几何标签容易使
语义分支遗忘或发生标定漂移。优先方案是：

1. 在四级冻结 Hiera 特征后加入 affinity 专属、零初始化 residual bottleneck adapter；
   语义分支继续读取原特征，单次 encoder 前向即可完成两任务。
2. 若 feature adapter 仍受限，再实现第二套 geometry-only LoRA（rank 4–8），只作用于
   affinity 前向；保留原 V6 LoRA 给语义前向。该方案需要 adapter 切换或第二次 encoder
   前向，计算成本更高。
3. 不采用“共享同一 LoRA、只用 affinity loss 更新”的方案。若必须共享，至少加入冻结
   V6 teacher 的语义 logit/feature 蒸馏约束，并使用远低于 decoder 的 LoRA 学习率。

LoRA/adapter 实验应在固定新的推理协议后进行：`marker_border_seal_width=2`、固定阈值臂、
修复 255 邻接合并，并按完整部署指标选模。否则训练收益会继续被后处理缺陷混淆。

## G4+封边后的推理协议实验

本轮先不更新 checkpoint，固定 G4b 和 2 px marker 封边，拆成三个可归因改动：

1. **255 上限拓扑合并**：候选超过上限时，从最小区域开始按局部接触长度合并；只填被
   选中邻接区域之间的一像素分水岭缝。若区域没有任何局部邻居，则丢为背景，而不是与
   远处区域共享 ID。该修复对未达到上限的图完全不改变输出。
2. **marker-only 局部弱边重建**：`boundary_threshold=0.72` 仍是最终 barrier；以 0.72
   以上响应为强边，只允许它在 `>0.45` 的弱响应中生长最多 8 px。低阈值结果只用于
   marker 连通性，不直接扩大最终 barrier，避免全局降阈值放大雾状背景。
3. **语义概率核心投票**：保留旧 `hard_majority` 基线，实验臂使用实例腐蚀 2 px 后的
   V6 ferrite 概率均值。每张图额外输出 `*_class_confidence.json`，记录 ferrite score、
   分类置信度和实例面积，便于定位接近 0.5 的误判，而不是仅看最终类别颜色。

严格 A/B 配置：

- 基线：`config/experiments/affinity_border_seal2.yaml`，运行时阈值固定 0.72；
- 仅补口：`config/experiments/affinity_g4b_seal2_reconstruct045.yaml`；
- 仅概率投票：`config/experiments/affinity_g4b_seal2_vote_probability.yaml`；
- 组合臂：`config/experiments/affinity_g4b_seal2_reconstruct045_vote_probability.yaml`。

先分别判断几何有效匹配/拆并统计和类别感知 mIoU/铁素体面积，再看组合臂；不能只按
预测实例总数挑选。若局部补口仍引入碎片，下一轮只调 low threshold 与生长步数，不改
训练。只有推理协议稳定后才启动 affinity 专属 feature adapter 或 geometry-only LoRA。

### 实测结果（G4b epoch 20）

必须使用 `outputs/affinity_geometry_g4b_gap_weight020/latest_affinity.pth`：其 epoch 为 20，
SHA-256 为 `1334fdcc8472b4da67386b76b25d9099ad6bcef10b6bd9c77ae6908658b8c619`。
同目录 `best_affinity.pth` 实际是 epoch 0，不能代表用户目检认可的 G4b。所有报告必须同时
记录 checkpoint 路径、epoch 和哈希，不能只写 `best/latest` 别名。

固定 gated fusion、2 px 封边、高阈值 0.72，六张有标签验证图：

| marker 方案 | 预测数 | 有效匹配 | 有效 mIoU | GT 惩罚 mIoU | 铁素体面积误差 | 代理总分 | 单图宏分 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 无弱边重建 | 1121 | 663 | 0.8503 | 0.6202 | 3.51% | 90.759 | 86.731 |
| 无重建 + 概率核心投票 | 1121 | 663 | 0.8503 | 0.6202 | 3.05% | 90.992 | 86.717 |
| low 0.55 / 2 px | 1142 | 673 | 0.8444 | 0.6251 | 5.86% | 89.287 | 86.399 |
| low 0.55 / 4 px | 1162 | 694 | 0.8409 | 0.6420 | 8.66% | 87.714 | 85.901 |
| low 0.50 / 4 px | 1156 | 691 | 0.8385 | 0.6374 | 8.07% | 87.887 | 86.200 |
| low 0.45 / 8 px | 1224 | 741 | 0.8390 | 0.6839 | 16.97% | 83.464 | 83.907 |

弱边重建呈现清楚的代理指标权衡：越强，GT 覆盖和有效匹配越高、合并越少，但铁素体被
拆小，人工验证集上的平均面积误差和有效匹配 IoU 持续恶化。这个结果不能单独否决重建：
六张手工多边形 GT 数量很少，且用户对十张无标签测试图的逐图目检确认 low 0.45/8px
明显缓解欠分割，新增切缝大多落在原图可见边界，仅少量追随划痕。因此该臂提升为当前
测试侧主线候选；无重建输出继续作为保守提交 A/B，而不是唯一默认。

为排除仅凭肉眼的主观偏差，tools/audit_reconstruction_splits.py 对测试图做了不使用 GT
的拓扑审计。相对无重建，重建从 837 增至 951 个实例（+114），新增内部切缝处的局部
对比度为普通实例内部的 2.77x，Sobel 梯度为 3.03x。这支持新增切分具有独立原图
证据。代价是面积不超过 200 px 的实例从 13 增至 34，下一步应针对少量碎片/划痕假边做
简单约束，而不是整体减弱重建。报告位于本地
g4b_inference_protocol_report/gt_free_reconstruct_audit/new_seams_report.png。

概率核心投票只改变少量类别，聚合分提高 0.233，但单图宏平均下降 0.014，且有效匹配
完全不变，属于弱正信号而不是已证实泛化收益。当前仍保留 hard_majority，概率投票仅
作为提交 A/B 候选和置信审计。

### G5 几何专用特征适配器

在推理协议稳定后，G5 从 G4b epoch 20 初始化，在冻结 Hiera/V6 语义 LoRA 与语义 decoder
的前提下，在四级 encoder 特征和 affinity decoder 之间加入 bottleneck ratio 8 的
零门控 residual adapter。四个标量 gate 从 0 开始，因此 epoch 0 与 G4b 几何路径严格
一致；训练参数仅为 affinity decoder 加约 0.3M adapter 参数，不改变语义前向。

训练固定 20 epoch，decoder/adapter/gate 初始学习率分别为
1e-5/5e-5/2e-4，配置为
config/train/affinity_geometry_g5_feature_adapter.yaml。部署 monitor 固定
gated + seal2 + high0.72 + low0.45/8px reconstruction + hard majority。
六张人工验证 GT 的完整部署分仍记录，但不能作为唯一晋级依据；必须同时比较 epoch 0、
latest 的无标签 monitor、gate 开启幅度和测试侧新增切缝质量。

本地可视化报告目录：

`C:/Users/danmo/.codex/visualizations/2026/08/20/01a02173-28e6-7930-873d-c7a5ce1bbf2f/g4b_inference_protocol_report`

其中 `overview_metrics.png` 为指标表，`geometry_comparison.png` 对比补口前后实例，
`semantic_vote_comparison.png` 只展示测试集中发生类别翻转的样本。

## 目检产物

本地目录：

`C:/Users/danmo/.codex/visualizations/2026/08/20/01a02173-28e6-7930-873d-c7a5ce1bbf2f/affinity_visual_audit`

包含 G2+0.55、G2+0.60、G3+0.60、G4b+0.65、G4b+封边2px+0.72 的同十张测试彩图，
以及 G4b 的融合边界概率图。
