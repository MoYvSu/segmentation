# G4 人工未覆盖带负亲和监督实验

## 问题

G3 的人工 LabelMe affinity 目标只监督两个端点都具有正实例 ID 的像素对。若相邻手工多边形之间留有未覆盖带，短程通道看不到明确负边；而部署的 distance-2/4 通道又依赖短程边界门控。这会允许少量高 affinity 泄露穿过可见晶界并在分水岭中造成大实例合并。

## 单变量

配置：`config/train/affinity_geometry_g4_manual_gap.yaml`。

G4 与 G3 使用相同的：

- G2 best 初始化；
- 人工/SAM2 整图/SAM2 native crop 采样质量占比 `0.50/0.25/0.25`；
- 数据增强、优化器、学习率 `2e-5 -> 3e-6`；
- 20 epoch 和 64 samples/epoch；
- 固定 `gated + mean + boundary_threshold=0.55` 部署选模。

唯一训练目标变化：

- 人工 LabelMe 样本中，`实例 ID > 0 ↔ 未覆盖 ID 0` 的像素对作为负 affinity；
- `0 ↔ 0` 仍然 ignore，不把整块未标区域当作同一实例或背景；
- SAM2 样本的任何未覆盖像素仍然 ignore，避免把伪标签漏检写成假边界；
- 已标实例内部正边及不同实例间负边保持 G3 原定义。

## 监督强度审计

在固定六张验证图、512 输出网格上，G4 的负边监督总量约为 G3 定义的 `3.09x`。
八个通道的增幅依次约为：

`5.66x, 5.52x, 4.33x, 4.48x, 3.52x, 3.36x, 2.39x, 2.28x`。

loss 仍对正、负边分别取均值，因此这不会让负损失按像素数直接放大三倍；变化是边界带监督覆盖更完整。

## 判定规则

epoch 0 是同一 G2 checkpoint 的部署基线。新 checkpoint 只有固定 0.55 完整部署总分更高时才晋级。必须同时检查：

- 部署总分、有效匹配 mIoU；
- ferrite 平均面积相对误差；
- 有效匹配数与预测实例数；
- 单图宏平均；
- 无标签 monitor 是否出现碎裂、雾化或明显过分割。

若 Oracle affinity 指标提高而部署有效匹配下降，判定失败。若预测实例明显增加但有效匹配不增，判定为把标注轮廓噪声学成过分割。

## 启动命令

```bash
conda activate sam2_env
python train_affinity_geometry_g1.py \
  --config config/train/affinity_geometry_g4_manual_gap.yaml
```
