# M0：μSAM AIS 三通道 GT Oracle

## 问题与边界

本实验先回答一个低成本问题：在不训练网络的情况下，μSAM AIS 的
`foreground + center-distance + boundary-distance` 表示能否在当前标注形态和部署分辨率下
近乎无损地恢复实例。若完美真值图都不能可靠恢复，则不进入 geometry decoder 训练。

这不是历史 naive EDT-gradient flow，也不是共享 boundary FPN 的稀疏中心热图。算法定义依据
[μSAM 论文](https://doi.org/10.1038/s41592-024-02580-4)、
[torch-em 距离 target](https://github.com/constantinpape/torch-em/blob/f12a744f18fe3fcf0d2ebc46324f429546e8f095/torch_em/transform/label.py#L454-L633)
和
[官方 seeded watershed](https://github.com/constantinpape/torch-em/blob/f12a744f18fe3fcf0d2ebc46324f429546e8f095/torch_em/util/segmentation.py#L175-L231)：

- foreground 为已覆盖实例区域；未覆盖区域在训练有效性 mask 中保持 ignore；
- center-distance 为到单一实例中心的逐实例归一化距离，中心为 `0`；凹实例质心落在外部时改用
  实例内距边界最远点；
- boundary-distance 为逐实例归一化的反转边界距离，实例深部为 `0`、inner boundary 为 `1`；
- marker 为 `foreground > 0.5 AND center < t_c AND boundary < t_b` 的连通分量；
- seeded watershed 只以 boundary-distance 为 height map，以 foreground 为 mask；
- 与官方默认 `apply_label=True` 一致，同一原始 ID 的不连通分量先分别重标记；marker 使用默认
  全连通；
- 重建不读取 GT 实例 ID、实例数或实例中心列表。类别指标使用 GT semantic majority oracle，
  因此这里只是几何上界，不是完整部署成绩。

watershed 本身没有 ignore 值。为了不把未知区域伪造成可学习的数值，本 Oracle 只在已标支持域内
定义 foreground，重建时等价于使用一张完美 annotated-support mask。它会把未覆盖带作为外部屏障，
是偏有利的表示上界，并不代表部署时能够得到同样的 foreground。

固定验证集与当前 direct 双头训练一致，共六图：`train_172/351/686/872/873/889`，合计
`942` 个 GT 实例。输入按全图等比 letterbox 到 `1024`，默认 geometry grid 为 `512`。

## 512-grid 结果

官方默认平滑和阈值为 foreground sigma `1.0`、distance sigma `1.6`、
`t_fg=t_c=t_b=0.5`。鲁棒性检查只做四个单变量 `±0.1` 和一组固定高斯噪声 `sigma=0.03`，
没有展开组合扫参。

| 条件 | Pred / GT | 有效匹配 | valid mIoU | symmetric mIoU | 铁素体面积项 | 代理总分 |
|---|---:|---:|---:|---:|---:|---:|
| 默认真值 | 989 / 942 | 899 | 0.85883 | 0.78067 | 0.96075 | 90.979 |
| center `0.4` | 970 / 942 | 902 | 0.86257 | 0.80210 | 0.97816 | 92.036 |
| center `0.6` | 1010 / 942 | 898 | 0.85676 | 0.76176 | 0.94652 | 90.164 |
| boundary `0.4` | 959 / 942 | 873 | 0.86244 | 0.78510 | 0.96744 | 91.494 |
| boundary `0.6` | 984 / 942 | 921 | 0.85837 | 0.80342 | 0.97674 | 91.756 |
| 默认阈值 + 噪声 | 991 / 942 | 899 | 0.85584 | 0.77639 | 0.95932 | 90.758 |

官方连通分量重标记先把 `942` 个原始 GT 变成 `960` 个 target component。默认真值下，按原始
GT 统计有 `59` 个实例内部出现多个 marker，共产生 `62` 个额外 marker；另有 `15` 个 GT 没有
marker。没有任何 marker 横跨多个 GT，说明主要失败不是相邻实例 marker 合并，而是断裂 component
定义以及复杂实例内部形成多枚 seed，同时小实例 seed 消失。轻噪声与默认真值几乎相同，表明首要
问题来自表示与标注形态，不是数值扰动。

## 1024-grid 单点归因

为排除“只是 512 grid 太低”的解释，只在 `1024` grid 运行一次官方默认参数：

- `1007 / 942` 个实例，`932` 个有效匹配；
- valid mIoU `0.90992`，symmetric mIoU `0.84215`；
- 铁素体面积项 `0.95725`，代理总分 `93.358`；
- `942` 个原始 GT 被变换为 `959` 个 target component；按原始 GT 统计，`60` 个实例内出现
  多 marker，共 `66` 个额外 marker，仅 `1` 个 GT 无 marker。

提高分辨率改善了边界 IoU 和漏匹配，却没有消除多 marker，反而使细长/凹形实例的多个低距离区域
保留得更完整。因此失败不只是 512-grid 下采样造成的。

## 结论：No-Go

即使使用偏有利的完美 annotated-support mask，该 Oracle 仍未达到“完美距离图近乎无损恢复”的
前置条件，当前不实现、不训练独立 μSAM geometry decoder。原因是当前 LabelMe polygon 中存在
较多非凸、狭长、相互嵌合或经互斥栅格化后断裂的
实例；径向 center-distance 与低 boundary-distance 的交集仍可能在同一 GT 内形成多块 marker。
学习预测只会在这一结构误差之上再增加回归误差。

代理总分看似较高，来自 GT semantic oracle、完美支持域和只平均有效匹配的指标，不能与主线黑盒
分数比较。No-Go 依据是 count、召回和 symmetric mIoU 未达到预先声明的近乎无损门槛。

这项结论只否决“当前 GT + 全图 1024 输入 + 512/1024 grid + μSAM AIS 后处理”作为下一项
训练实验，不否决 μSAM 在细胞数据上的既有成果，也不外推到改变实例定义、原生高分辨率 geometry
或其他拓扑表示。当前正式主线 E10a + G4b 不变。

## 复现

```bash
python tools/run_musam_gt_oracle.py

python tools/run_musam_gt_oracle.py \
  --output-grid 1024 \
  --base-only \
  --output-dir outputs/experiments/musam_gt_oracle_v1_grid1024_control
```

产物：

- `outputs/experiments/musam_gt_oracle_v1/musam_gt_oracle_summary.json`
- `outputs/experiments/musam_gt_oracle_v1/morphology_cases.png`
- `outputs/experiments/musam_gt_oracle_v1/REPORT.md`
- `outputs/experiments/musam_gt_oracle_v1_grid1024_control/`
