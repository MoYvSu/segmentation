# V6 可复现基线

V6 是当前 B2 refine 实验的固定初始化锚点，而不是待继续训练的中心多任务模型。

## 已核验产物

| 字段 | 值 |
|---|---|
| checkpoint | `outputs/stage2_v6/best_model_stage2.pth` |
| SHA-256 | `c4c9827a18ecdda105056d502e5f93a92fe7fc93ada24d63886b20726c7ec557` |
| checkpoint epoch | 9（日志中的 epoch 10） |
| val mIoU | 0.8254672504 |
| val boundary IoU | 0.5116020453 |
| composite | 0.5743750863 |
| decoder | FPNDecoder, 256 channels, `boundary_refine=false`, `center_head=false` |
| LoRA | enabled, rank 16, alpha 32, frozen during V6 |
| historical Git commit | `2b5d6e4e1854c0ef096265a826a3f684ccbdde8e` |
| environment | Python 3.12.3, PyTorch 2.8.0+cu128, RTX 4090 |

历史运行记录保存在：

```text
outputs/runs/20260808_111823_v6_semantic_label/
```

其中包含完整 `config_snapshot.yaml`、60 epoch `metrics.csv` 和 `run_info.json`。
V6 的语义分支在全部 60 epoch 中保持 `mIoU=0.8254672504`，证明冻结语义路径有效；
最佳边界结果出现在 epoch 10，而不是训练末尾。

## 历史训练入口

原始命令：

```bash
python train_stage2.py --config config/stage2_v6.yaml \
  --init_from_checkpoint outputs/stage2_joint_v3/best_model_stage2.pth \
  --phase v6 --tag semantic_label
```

本地保留的 joint-v3 起点已核验为 epoch 89 历史最优：

```text
SHA-256: dde542325a28666de730a72d6f77f37149c8ac02b44c5434cddad74725f317ce
best_composite_score: 0.5713295122252249
```

当前 B2 不需要重跑上述历史链，直接从已核验的 V6 best checkpoint 初始化。

可随时重新校验权重哈希、架构与训练状态：

```bash
python tools/inspect_checkpoint.py outputs/stage2_v6/best_model_stage2.pth
```

## B2 启动

```bash
cd /root/autodl-tmp/segmentationv2_main
conda activate sam2_env
python train_stage2.py --config config/train/stage2_refine_v6.yaml \
  --phase boundary --tag refine_v6_b2
```

B2 保持后处理不变，关闭中心头，冻结语义/LoRA。前 5 epoch 只训练零初始化 refine head，
第 6 epoch 起以 `0.15 × base_lr` 解冻 V6 coarse boundary 基座。
