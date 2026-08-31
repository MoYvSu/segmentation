# SSL 直达语义 / affinity 双头实验

## 假设

本实验不复用 Stage1、joint-v3、V6 boundary、G0/G1 affinity 课程或 G7。模型只保留：

```text
SAM2 Hiera + SSL LoRA
├─ 随机初始化 E10a 式高分辨率 semantic head
└─ 随机初始化 8 通道 affinity head
```

目的不是立即替换 `83.94` 主线，而是检验最终两项任务能否直接完成任务对齐，避免历史阶段在
LoRA 和 boundary FPN 上反复覆盖知识。当前 E10a+G4b 的部署配置、checkpoint 和输出均不修改。

## 监督边界

人工 LabelMe 样本在同一空间增强后同时产生：

- purified ferrite/pearlite semantic GT；
- 1024 网格实例图，用于 E10a 实例等权 / probability-pool loss；
- 512 网格实例图，用于 8 通道 affinity pair loss。

SAM2 数据只提供无类别几何监督：同一 mask 内为 positive，不同已覆盖 mask 间为 negative；
covered-uncovered 和 uncovered-uncovered pair 保持 ignore。manifest 中出现任何 `class_label` 都会
被拒绝。训练前还必须通过 `approval.json` 的人工审核门槛，未审核候选不会静默回退为可训练数据。

本实验不加入旧 boundary loss、center loss、SSL reconstruction replay 或新的后处理规则。

## 两阶段训练

配置：

```text
config/train/direct_ssl_semantic_affinity.yaml
```

### Phase 1：head warm-up

前 5 epoch 冻结 SAM2 与 SSL LoRA，只训练两个随机解码头。SAM2 geometry 尚不进入训练，避免
随机 head 的噪声和伪几何共同改写 SSL 表示。

### Phase 2：joint LoRA

随后 20 epoch 保持 SAM2 原始参数冻结，以 `2e-6` 解冻 LoRA；semantic 与 affinity head 分别
使用 `2e-5` 和 `1e-5`。每个人工批次同时计算 semantic 与 manual affinity；每步可再加入一个
已审核 SAM2 affinity-only 批次。三项 loss 在训练开始前用固定批次校准一次，校准尺度随后不变。

## 输入门槛

必须存在：

```text
weights/sam2_hiera_base_plus.pt
outputs/lora_pretrain/lora_state_dict.pth
data/raw/
data/purified_gt/
data/sam2_geometry_g2/manifest.jsonl
data/sam2_geometry_g2/masks/
data/sam2_geometry_g2/approval.json
```

当前本地缺少 SSL LoRA 文件和 `data/sam2_geometry_g2/`；历史服务器资产或重新生成候选在进入
训练前仍必须满足上述路径与人工审核契约。

## 命令

优先运行不含 SAM2 geometry 的 Arm A，先检验最短链路本身：

```bash
python train_direct_semantic_affinity.py \
  --config config/train/direct_ssl_semantic_affinity_no_sam2.yaml --check

python train_direct_semantic_affinity.py \
  --config config/train/direct_ssl_semantic_affinity_no_sam2.yaml
```

经审核的 SAM2 geometry 准备完成后，再运行只增加该监督源的 Arm B：

```bash
conda activate sam2_env

python train_direct_semantic_affinity.py \
  --config config/train/direct_ssl_semantic_affinity.yaml --check

python train_direct_semantic_affinity.py \
  --config config/train/direct_ssl_semantic_affinity.yaml
```

断点恢复：

```bash
python train_direct_semantic_affinity.py \
  --config config/train/direct_ssl_semantic_affinity.yaml \
  --resume outputs/direct_ssl_semantic_affinity/latest_direct_dual.pth
```

输出：

```text
outputs/direct_ssl_semantic_affinity/
├─ split.json
├─ latest_direct_dual.pth
└─ best_direct_dual.pth
```

checkpoint 记录 semantic、affinity、LoRA、固定 loss 校准尺度、split、优化器、scheduler 和完整
配置；部署加载不依赖原 SSL 文件。

## 选模与提交

每个 epoch 均走完整 validation 部署路径：

```text
direct semantic + direct affinity
→ gated affinity boundary
→ high=0.65
→ seal2 / 局部重建
→ 受阻分水岭
→ probability-mean 实例分类
```

`best_direct_dual.pth` 只按完整 deployment total score 选择，不使用训练 loss 或 Oracle graph
指标。正式输出：

```bash
python tools/run_direct_semantic_affinity_submission.py \
  --config config/train/direct_ssl_semantic_affinity.yaml \
  --checkpoint outputs/direct_ssl_semantic_affinity/best_direct_dual.pth
```

提交入口逐图检查实例 ID `<=255`，并生成包含 checkpoint SHA-256、epoch、phase、参数量、融合
配置和实例统计的 `submission_manifest.json`。

## 晋级

本实验必须在同一完整部署链上与 E10a+G4b 比较。替换主线至少要求黑盒：

```text
实例 mIoU >= 0.8381
铁素体平均面积项 >= 0.8408
总分 > 83.94
```

本地 smoke、语义 mIoU、affinity loss 和早期 epoch 只能诊断训练是否有效，不能单独晋级。
