# Direct 双头 Manual Target V2 实验方案

## 结论先行

当前 short-chain 的首要问题不是缺少一种更复杂的多任务损失，而是人工监督定义互相冲突：

- 32 张图平均有 `9.58%` 像素未被 LabelMe polygon 覆盖；现有 semantic target 将它们全部写成
  `0=珠光体`，BCE/Dice 又没有 annotation-valid mask。
- 同一批未覆盖像素在 affinity 中通过 `manual_uncovered_as_boundary=true` 成为低权负边。
- semantic 与 instance target 还分别使用坐标截断和四舍五入两套栅格化规则，覆盖区平均约
  `0.46%` 像素的类别定义不一致。

因此下一版固定为 `manual_target_v2`：统一一次栅格化，直接以人工目检通过的 `r=8` 完整结果
替换旧 GT；剩余 unknown 保持 ignore。它不增加模型头，也不增加训练阶段。

## 窄接缝补全 Oracle

实现位于：

- `utils/seeded_label_completion.py`
- `tools/run_seeded_label_completion_oracle.py`

约束如下：

1. 原人工实例区域作为 marker，任何原标像素和实例 ID 均不得修改。
2. elevation 使用轻度平滑后的 Lab 多通道 Scharr 梯度。
3. watershed 只允许进入距人工实例不超过固定 native 半径的 gap；更宽区域继续 unknown。
4. 不覆盖 `data/raw` 或 `data/purified_gt`，只生成可删除的派生候选和 provenance。

全 32 图结果：

| 半径 | 原覆盖率 | 补后覆盖率 | 填入原 gap | component excess 前→后 |
|---:|---:|---:|---:|---:|
| 8 px | 90.42% | 96.68% | 65.38% | 69→51 |
| 16 px | 90.42% | 98.22% | 81.47% | 69→51 |

`r=8` 是首选。三张代表图在 Gaussian `sigma=0.8/2.0` 下的补区实例归属一致率为
`91.3%–92.6%`；人工目检进一步确认紫色新边界能较好贴合图样边界，因此本实验将
`original_covered | filled` 视作一份等置信新 GT。`r=16` 扩张更强，并且 μSAM 自一致性继续
恶化，不进入训练。

固定六图、512 grid 的 μSAM follow-up：

| GT | pred/GT | matches | valid mIoU | symmetric mIoU | 铁素体面积项 |
|---|---:|---:|---:|---:|---:|
| 原 LabelMe support | 989/942 | 899 | 0.85883 | 0.78067 | 0.96075 |
| r=8 补缝 | 995/942 | 913 | 0.84953 | 0.77952 | 0.94068 |
| r=16 补缝 | 1004/942 | 907 | 0.83623 | 0.75544 | 0.92318 |

因此补缝没有挽救 μSAM 三通道表示，独立 μSAM geometry decoder 仍为 No-Go。这个结论不否定
补缝可用于修正 short-chain 的人工监督；两者检验的问题不同。

## `manual_target_v2` 定义

只使用 `load_labelme_instances` 生成 canonical instance map，经 `r=8` watershed 补全后，从同一
instance ID/class LUT 派生 semantic target。provenance 继续保存用于审计，但不再区分训练权重：

- `manual | filled`：新 GT，有效权重 `1.0`；
- `unknown`：其余未覆盖区，权重 `0`，完全 ignore。

语义损失：

```text
L_sem = masked_BCE(new_GT)
      + 0.30 * masked_Dice(new_GT)
      + 0.75 * instance_core_BCE(new_GT instances)
```

三项均只在新 GT 的有效区域计算，剩余 unknown 和 letterbox padding 不产生梯度。

Affinity 对每个 offset pair 使用以下规则：

| pair | target | weight |
|---|---|---:|
| new GT ↔ new GT | 同 ID 为正、异 ID 为负 | 1.0 |
| 任一端 unknown | ignore | 0 |

因此紫色新界面的两侧会直接产生 negative affinity；unknown-ignore 同时由 pixel-valid mask 与
`instance_id>0` 双重保证，不再使用 `manual_uncovered_as_boundary`。

## 最小训练 A/B

Control 使用现有 Normalize Hiera-B+ direct 配方；Candidate 只把人工 target/loss 换成
`manual_target_v2`。以下全部固定：模型、ImageNet normalization、SSL-LoRA 起点、SAM2 无类别几何
数据、增强、epoch、学习率和 `boundary=0.59` 推理点。

训练仍只有：

1. 已有 LoRA SSL（两组共用同一个 loss-best）；
2. 冻结 LoRA 的双头预训练 5 epoch；
3. 开放 LoRA 的联合训练 20 epoch；
4. 本轮不启用可选尾训。

每阶段按预先声明的固定验证损失最小点交接，不按六图 deployment score 选起点。验证损失与训练
使用同一份新 GT：`original_covered | filled` 等权，剩余 unknown ignore；provenance 只作审计，
不再参与损失分权。25 epoch 的 monitor 取每 5 epoch 一组，继续保存 semantic JET 与 affinity HOT
两类缩略图。

比较顺序：固定六图原人工覆盖区 loss/semantic mIoU → 完整 deployment proxy（含 symmetric mIoU、
matches、面积项）→ 固定模糊/低照退化集 → 主线/Control/Candidate 单张横向拼图。只有完整部署输出
稳定改善时才允许晋级；训练 loss 或补缝 Oracle 本身不能替换 E10a+G4b。

## 完成结果（2026-09-07）

训练已完成 5+20 epoch，无 OOM、NaN 或非有限权重。warmup loss-best 为 epoch 5；进入 joint 前
已实际回载该点。joint 验证目标从 epoch 6 的 `0.7671` 持续降至 epoch 25 的 `0.7093`，因此最终
`best_direct_dual.pth` 为 epoch 25。验证集分项从 epoch 1 到 epoch 25：

正式 Candidate checkpoint 为
`outputs/20260907_203850_gtv2/best_direct_dual.pth`，SHA-256
`42dad26726262d8c19a76a67c95dede9b23f00325b02ba32e6d65f58d000db75`。

| 指标 | epoch 1 | epoch 25 |
|---|---:|---:|
| semantic loss | 0.22726 | 0.09955 |
| manual affinity loss | 0.66493 | 0.48372 |
| semantic mIoU（1024 grid） | 0.90028 | 0.94052 |

五组 monitor 已在 epoch `5/10/15/20/25` 完整生成。与 Normalize Control e25 的同图 monitor 相比，
Candidate 的语义概率由大面积边缘过渡和晶粒内渐变变为均匀的高/低置信区域；affinity HOT 图则
变化很小，仍有背景响应偏高、细线中断和前景/边界动态范围不足。因此本轮主要收益可以归因于
新 GT 修正语义监督，不能声称 affinity 问题已经解决。

训练期间写入 `train.log` 的 deployment 数字仍来自旧 LabelMe GT，只保留为历史诊断。训练后已
修复 direct 训练、扫参和退化验证入口：`manual_target_v2` 配置统一使用 r=8 instance/class GT，
并在评分前把预测的 residual-unknown 区域清零，使其真正不进入 IoU、计数或面积统计。旧配置继续
保持 LabelMe 口径。

固定六图、同周期 e25 对 e25 的新口径结果如下：

| checkpoint / boundary | semantic mIoU | valid mIoU | symmetric mIoU | 面积误差 | pred/GT | matches | proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Normalize Control / 0.59 | 0.90376 | 0.80070 | 0.47001 | 13.91% | 1138/942 | 668 | 83.08 |
| Target V2 e25 / 0.59 | **0.94482** | **0.80859** | **0.47551** | **9.95%** | 1090/942 | 641 | **85.46** |
| Normalize Control / 0.63 | 0.90376 | 0.80342 | 0.47908 | **0.33%** | 1028/942 | 613 | **90.00** |
| Target V2 e25 / 0.63 | **0.94482** | **0.80709** | **0.49524** | 2.12% | **999/942** | 613 | 89.29 |

语义 mIoU 相对 Control 提升 `+0.04106`，其中珠光体 IoU 从 `0.86249` 提升至 `0.91895`，铁素体
从 `0.94504` 提升至 `0.97070`。这是明确的大幅改善。实例侧在两个阈值上 valid/symmetric mIoU
也均提高，但幅度远小于语义；`0.59` 保留更多匹配，`0.63` 让预测数更接近 GT，但会减少边界和
有效匹配。Control 在 `0.63` 的 proxy 略高完全来自铁素体平均面积碰巧更贴合，不能据此否定
Candidate 的语义和对称实例质量提升，也不能据此自动把部署阈值从 `0.59` 改为 `0.63`。

旧 LabelMe deployment proxy 选出的 epoch 19，在新 GT、`0.63` 下为 semantic mIoU `0.94255`、
valid mIoU `0.80387`、symmetric mIoU `0.48903`、15 个更少的 matches、proxy `88.34`；均低于
loss-best epoch 25。故“阶段交接与最终 checkpoint 按固定验证损失最小”在本次实验中获得支持。

详细回算结果：

- `output/manual_target_v2_proxy/control_e25_detailed.json`
- `output/manual_target_v2_proxy/e25_loss_best_detailed.json`
- `output/manual_target_v2_proxy/e19_old_proxy_best_detailed.json`

## 后续单变量

下一单变量不再改 GT，而是 `pseudo_stopgrad_v1`：人工 semantic+affinity 继续更新 heads 和 LoRA；
SAM2 pseudo affinity 只更新 affinity decoder，encoder feature 对该分支 stop-gradient，检验低质
cross-mask negative 是否在联合阶段污染共享 LoRA。若仍过分割，再单独修复 pseudo negative 的
term-level 权重；当前 `normalize_edge_weights=true` 下统一 edge weight 会在归一化中抵消。

暂不加入 PCGrad、uncertainty weighting、新 decoder、CRF 正式阶段或额外中间 checkpoint 起点。

## 依据

- marker watershed 的 elevation 与弱梯度平台行为见
  [scikit-image segmentation guide](https://scikit-image.org/docs/stable/user_guide/tutorial_segmentation.html)。
- AffinityNet 对 neutral/unknown pair 采用 ignore，并分组归约正负边：
  [CVPR 2018 paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ahn_Learning_Pixel-Level_Semantic_CVPR_2018_paper.pdf)。
- ScribbleSup 明确把未标区域视为 unknown，而不是背景：
  [CVPR 2016 paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Lin_ScribbleSup_Scribble-Supervised_Convolutional_CVPR_2016_paper.html)。
