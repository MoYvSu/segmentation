# 配置目录

- `default_config.yaml`：可训练、可推理的当前 V6 参考基线；路径跨本机/服务器可移植。
- `inference/`：只改变推理输出与后处理参数，不改变模型架构。
- `train/`：明确区分 Stage 1 与 Stage 2 的可训练参数和输出目录。
- `experiments/`：会改变架构、监督目标或训练策略的实验。
- `stage2_center_heatmap.yaml`：旧命令兼容的完整历史快照；已判废，不作为主线。

配置可用 `_base` 递归继承。`paths.project_root: auto` 默认定位当前仓库；临时覆盖可设置
`SEGMENTATION_PROJECT_ROOT`，无需为 Windows/Linux 分别维护 YAML。

推理会严格比较配置和 checkpoint 的 `boundary_refine`、`center_head`、LoRA 等架构字段。
只有明确进行消融时才使用 `--allow-architecture-mismatch`。

当前 B2 主线使用 `train/stage2_refine_v6.yaml`：从 V6 best 初始化，只增加独立高分辨率
refine residual，关闭中心头并保持原后处理不变。

当前建议的下一轮单变量实验是 `train/stage2_refine_v6_physaug.yaml`：继续从 V6 best
初始化 B2，但在 5 个 epoch 内只训练 refine head，关闭无标签一致性，加入显微成像物理增强。
增强每次只抽取 1~2 项（曝光/白平衡、失焦、降采样、低频照明或低对比划痕），不制造
圆形硬遮罩，也不改 GT。

`inference/b2_quality_aware.yaml` 是配套的低复杂度推理实验：固定几何 TTA，并按当前单图的
亮度、对比度、清晰度和偏色分为 `standard`/`weak` 两档。弱档只融合一张确定性增强视图并
应用固定的小幅边界阈值偏移；不读取跨图统计，不按实例数、平均面积或环形拓扑闭环调参。
所有推理配置均要求 `max_instance_id <= 255`。
