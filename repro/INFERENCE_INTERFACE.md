# 推理复现接口

本目录当前只实现训练复现。后续推理复现固定接收以下输入，不改变训练产物路径：

```text
checkpoint:
  outputs/stage2_semantic_e10a_cold20/best_model_stage2.pth
affinity_checkpoint:
  outputs/affinity_geometry_g4b_gap_weight020/latest_affinity.pth
config:
  config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml
input_dir: 待推理图像目录
output_dir: 推理复现输出目录
```

计划入口固定为：

```bash
python repro/inference.py \
  --config config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml \
  --semantic-checkpoint outputs/stage2_semantic_e10a_cold20/best_model_stage2.pth \
  --affinity-checkpoint outputs/affinity_geometry_g4b_gap_weight020/latest_affinity.pth \
  --input-dir <images> \
  --output-dir <result>
```

当前不创建 `repro/inference.py`，避免在训练复现尚未验证前维护第二套推理包装。后续实现必须
复用现有 `tools/run_affinity_submission.py` 或组合 checkpoint 入口，并保持实例 PNG、类别 JSON、
实际配置快照和 `max_instance_id <= 255` 输出契约。
