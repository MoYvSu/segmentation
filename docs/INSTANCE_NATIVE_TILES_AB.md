# 官方代理评分与原生分块推理 A/B（2026-08-23）

## 目的

1. 用类别感知的一对一实例匹配和铁素体平均面积，替代语义 mIoU/像素边界 IoU，评估最终实例结果。
2. 在 checkpoint、阈值以外的后处理参数完全相同时，比较当前 1024 Letterbox 与原始分辨率重叠分块产生的 logits。

## 固定条件

- checkpoint：`outputs/stage2_v6/best_model_stage2.pth`
- 验证子集：按项目既有 `seed=42`、`train_ratio=0.8` 得到的 7 张 LabelMe 图像
- 分块：`1024×1024`，重叠 256 像素，正值 raised-cosine 权重融合
- 分块后先融合全图 logits，再运行一次现有分水岭；不拼接 tile 内实例编号
- TTA：关闭
- 中心种子：关闭
- 其余后处理：继承 `config/inference/v6_reference.yaml`

## 汇总结果

| 推理方式 | 边界阈值 | 规则文本代理总分（全数据汇总面积） | 逐图算分再平均 | 有效匹配 mIoU | GT 惩罚 mIoU | 对称惩罚 mIoU | 铁素体面积相对误差 | 预测实例 | 有效匹配 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Letterbox | 0.35 | 88.207 | 86.275 | 0.8650 | 0.7268 | 0.5662 | 0.1008 | 1430 | 936 |
| Native tiles | 0.35 | 78.054 | 78.705 | 0.8891 | 0.8244 | 0.5200 | 0.3280 | 1766 | 1033 |
| Native tiles | 0.40 | 82.085 | 82.598 | 0.8872 | 0.7980 | 0.5154 | 0.2455 | 1725 | 1002 |
| Native tiles | 0.45 | **88.767** | **86.642** | 0.8845 | 0.7558 | 0.5632 | 0.1091 | 1495 | 952 |
| Native tiles | 0.50 | 93.494 | 86.012 | 0.8802 | 0.6977 | 0.5548 | 0.0100 | 1401 | 883 |

## 解释

- 原生分块确实保留了更多可用于实例匹配的边界信息：0.35 时有效匹配由 936 增至 1033，GT 惩罚 mIoU 由 0.727 增至 0.824。
- 1024 Letterbox 下使用的 0.35 阈值不能直接迁移到原生尺度。分块 0.35 的 7 张图中有 6 张达到 255 实例上限，铁素体平均面积被显著压低。
- 0.45 是当前最稳健候选：全局汇总和逐图宏平均两种计分口径均略高于 Letterbox，同时有效匹配和 GT 惩罚 mIoU也更高。
- 0.50 的 93.494 不能直接视为可靠提升。其全局铁素体面积误差只有 1%，但逐图面积误差仍明显，正负误差在数据集汇总后发生抵消；同时有效匹配由 936 降至 883，铁素体匹配召回由 0.849 降至 0.756。
- 在组委会明确面积项是“全测试集汇总后计算”还是“逐图计算后平均”前，不使用 0.50 作为主线。

## 当前决策

- 保留原生分块方向，候选阈值固定为 0.45，不引入按实例数或平均面积自动选阈值。
- 下一次验证应扩大到交叉验证折或新增独立标注图像，确认 0.45 的微小收益不是 7 张验证集波动。
- 若后续训练 affinity/refine 分支，应从原图裁取原生尺度 crop，使训练与分块推理的尺度分布一致。

## 复现入口

```bash
python tools/run_instance_ab.py \
  --config config/inference/v6_reference.yaml \
  --subset val --modes letterbox native_tiles --save-visualization

python tools/run_instance_ab.py \
  --config config/experiments/native_tiles_v6_bt045.yaml \
  --subset val --modes native_tiles \
  --output-dir outputs/experiments/native_tiles_v6_bt045
```

详细报告位于服务器：

```text
/root/autodl-tmp/segmentationv2_main/outputs/experiments/native_tiles_v6*/ab_summary.json
```
