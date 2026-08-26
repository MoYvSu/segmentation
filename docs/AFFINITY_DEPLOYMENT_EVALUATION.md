# Affinity 部署闭环与 G2/G3 复核

## 结论

当前候选主线为：冻结 V6 语义锚点 + 8 通道 affinity geometry + G2/G3 SAM2 无类别伪实例监督 → gated-mean affinity boundary → 受阻分水岭 → 语义投票分类。G3 值得保留，但尚不能宣称已提升黑盒竞赛成绩；V6/B2 边界管线继续作为可复现基线和回退。

训练中的 checkpoint 晋级固定使用 `gated + mean + boundary_threshold=0.55` 完整部署路径。Oracle 图重建只诊断 affinity 是否学到局部几何，不得选择 `best_affinity.pth`。

## 为什么旧验证不够

旧验证用 GT `labels > 0` 作为图重建前景，并给所有实例 dummy class。它只回答“真实前景已知时能否还原连通关系”，没有覆盖预测语义、affinity 融合、分水岭、语义投票和类别感知匹配。

## G2/G3 同协议结果

同一组 6 张有标签验证图，路径为“预测 V6 语义 → gated-mean affinity → boundary → watershed → class vote”。总分代理由有效匹配 mIoU 与铁素体平均面积分各占 50 分。

| checkpoint | 阈值 | 总分 | mIoU | 面积分 | 面积误差 | 预测数 | 有效匹配 |
|---|---:|---:|---:|---:|---:|---:|---:|
| G2 best | .50 | 85.6524 | .8477 | 43.2666 | .1347 | 1200 | 700 |
| G2 best | .55 | 87.8559 | .8495 | 45.3826 | .0923 | 1158 | 658 |
| G2 best | .60 | 90.7304 | .8506 | 48.1992 | .0360 | 1098 | 618 |
| G2 latest | .50 | 87.4353 | .8297 | 45.9515 | .0810 | 1221 | 706 |
| G2 latest | .55 | 89.3990 | .8298 | 47.9112 | .0418 | 1177 | 669 |
| G2 latest | .60 | 91.5945 | .8370 | 49.7422 | .0052 | 1096 | 631 |
| G3 best | .50 | 87.5152 | .8389 | 45.5695 | .0886 | 1211 | 702 |
| G3 best | .55 | 89.2824 | .8365 | 47.4552 | .0509 | 1160 | 677 |
| **G3 best** | **.60** | **92.2909** | **.8459** | **49.9947** | **.0001** | **1107** | **629** |
| G3 latest | .50 | 86.7930 | .8389 | 44.8486 | .1030 | 1206 | 714 |
| G3 latest | .55 | 88.6690 | .8427 | 46.5359 | .0693 | 1162 | 672 |
| G3 latest | .60 | 90.9660 | .8432 | 48.8040 | .0239 | 1112 | 635 |

V6 边界基线总分约 `82.40`，mIoU 约 `.85`，面积分约 `40.06`，预测数 `1259`，有效匹配 `731`。

## 阈值解释与契约

G3-best/.60 汇总总分最高，但相比 .55 把有效匹配从 677 降至 629，只在 6 张图中的 1 张取得单图第一。其优势主要来自汇总面积误差接近零，存在跨图正负误差相消。单图总分宏平均为 .50=`87.3797`、.55=`87.1208`、.60=`85.7916`。

因此：

1. 训练期固定 `gated + mean + .55` 选择 checkpoint，不按 epoch 或单图搜索阈值；
2. 训练结束后再扫描 .50/.55/.60；
3. .60 只作为黑盒提交的面积导向候选，.50 作为匹配覆盖率诊断；
4. 不固定每图实例数或平均面积，仅严格保证实例 ID 不超过 255；
5. Oracle geometry 只做诊断，完整 Deployment 指标才允许模型晋级。

## 标准命令

```bash
conda activate sam2_env
python tools/evaluate_affinity_deployment_sweep.py \
  --config config/train/affinity_geometry_g3_native_crop.yaml \
  --split outputs/affinity_geometry_g3_native_crop/split.json \
  --checkpoint g2_best=outputs/affinity_geometry_g2_pseudo/best_affinity.pth \
  --checkpoint g2_latest=outputs/affinity_geometry_g2_pseudo/latest_affinity.pth \
  --checkpoint g3_best=outputs/affinity_geometry_g3_native_crop/best_affinity.pth \
  --checkpoint g3_latest=outputs/affinity_geometry_g3_native_crop/latest_affinity.pth \
  --output-dir outputs/experiments/deployment_selection_g2_g3 \
  --thresholds 0.50,0.55,0.60 \
  --fusion-mode gated --short-reduction mean --monitor-count 0
```

输出包括 `deployment_selection.json`、`deployment_ranking.csv` 和逐图报告。旧 G2/G3 的 best 权重是按 Oracle 指标产生的历史产物，经过部署评估后才能晋级。新 v2 affinity checkpoint 会记录实际选模指标及是否仍需外部晋级。

下一轮实验必须同时报告固定 .55 部署总分、有效匹配 mIoU、铁素体平均面积误差、有效匹配数、预测实例数和单图宏平均。只有 Oracle 指标提高不视为有效进展。