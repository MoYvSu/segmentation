# G7 高分辨率短程 affinity 残差

## 假设

E10a + G4b watershed 的主要几何问题不是缺少另一种图切分算法，而是 G4b 在 512 网格输出后，
弱而密集的晶界被插值成模糊、可泄露的边界带。G7 保留已获黑盒 `83.94` 的 E10a-only
部署路径，只提高四个 distance-1 affinity 通道的可学习空间分辨率。

## 冻结与可训练范围

- 冻结 SAM2 encoder、V6 LoRA、E10a semantic decoder、G4b affinity FPN/head；
- G4b 仍在 512 网格输出完整 8 通道 affinity；
- 新残差头读取冻结的 G4b feature、上采样短程 logits 和 1024 RGB/局部对比，输出四通道
  `tanh` 有界修正；最后一层零初始化；
- distance-2/4 通道只做双线性插值，没有可训练参数；
- 不蒸馏 G4b affinity 概率，避免复制其背景雾化。

## 监督可靠性

- 人工实例内/实例间 pixel pair 使用完整 affinity 监督；
- SAM2 同一 mask 内的 pair 是权重 1.0 的 positive affinity；
- 两个互斥 SAM2 mask 之间的 negative affinity 权重为 0.25，降低 SAM2 误拆同一晶粒的伤害；
- SAM2 未覆盖像素以及 covered-uncovered pair 保持 ignore，绝不把漏检写成负边界；
- 沿用 G3 的高覆盖原生 crop 与镜像 letterbox，不引入中心、距离场、裂缝或形状规则。

## 评估与运行

checkpoint 选择运行完整 E10a semantic → gated affinity → `high=0.65` → seal2/局部重建 →
watershed → 类别投票代理。GT 前景 affinity graph 重建只作诊断，不作为主选择指标。

```bash
conda activate sam2_env
python train_affinity_geometry_g1.py \
  --config config/train/affinity_geometry_g7_highres_short.yaml
```

训练后先比较 epoch 0/2/…/20 的固定无标签 monitor，检查弱边变窄、断口减少以及背景雾化；
再用同一 `high=0.65` 生成测试可视化。只有固定阈值下的目检与完整部署代理同时不退化，才生成
单主干融合 checkpoint 或提交黑盒。
