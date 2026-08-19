# 当前管线与产物约定

## 结论与主线

当前可用参考是 `outputs/stage2_v6/best_model_stage2.pth`：语义完整性与边界连续性明显优于
中心热图多任务模型，但实例结果仍同时存在图样中心碎裂式过分割和外围欠分割，不能视为最终方案。

`outputs/stage2_center_heatmap/best_model_stage2.pth` 只保留为负面对照。现有中心 GT 由每个
Labelme polygon 生成一个种子，polygon 与物理晶粒并不等价；同时中心损失和边界损失共享
`boundary_fpn`，实测造成背景雾化、铁素体大块欠分割、珠光体碎裂及薄环嵌套。

当前 B2 架构实验以 V6 权重为语义锚点，并满足：

1. 语义路径冻结或低学习率保护；
2. 边界 refine 路径独立训练，不接收不可靠中心标签的梯度；
3. 若重试辅助任务，使用独立 FPN/stop-gradient，并先验证 GT 与物理实例的一致性；
4. 后处理把高置信边界作为硬障碍，而不是仅改变分水岭可视化灰度。

## 入口

```bash
conda activate sam2_env

# 当前 V6 参考推理
python inference.py --config config/inference/v6_reference.yaml

# Stage 1 全监督（LoRA 可训练）
python train.py --config config/train/stage1_lora.yaml

# Stage 2 B2；V6 作为冻结语义锚点，落地独立高分辨率 refine head
python train_stage2.py --config config/train/stage2_refine_v6.yaml \
  --phase boundary --tag refine_v6_b2
```

推理常用参数可直接从 CLI 覆盖，不再复制临时 YAML：

```bash
python inference.py --config config/inference/v6_reference.yaml \
  --boundary-threshold 0.35 --min-instance-area 50 --no-center-seeds \
  --output_dir outputs/inference/<name>
```

每个推理目录都会生成 `inference_manifest.json`，记录 checkpoint、架构、实际阈值和三类实例统计。

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
