# Agent.md — 低碳钢金相图像相区分割开发指南

Python 环境：`conda activate sam2_env`。

## 1. 项目目标

低碳钢金相图像**相区分割 + 晶粒实例分割**：
- 语义：珠光体(0) / 铁素体(1)；
- 边界：独立通道预测晶界；
- 实例：边界骨架化 + 受阻分水岭 + 语义投票。

## 2. 硬约束（不可违反）

1. 总参数量 < 500M（冻结 encoder ~80M + 解码头 ~10.8M，LoRA +1M）。
2. 零预训练解码器：FPN 必须随机初始化，禁止加载 SAM 2 原生 Mask Decoder 权重。
3. 权重只放 `weights/`，禁止全局缓存路径。
4. 一律 Letterbox（等比缩放 + BORDER_REFLECT），禁止挤压。
5. 所有变换/增强在线完成，禁止离线改图。

## 3. 当前架构

当前候选部署路径是：冻结 V6 语义锚点 + 8 通道 affinity geometry + G2/G3 SAM2
无类别伪实例监督 → gated-mean affinity boundary → 受阻分水岭 → V6 语义投票分类。
G3 尚未证明黑盒竞赛成绩提升；V6/B2 边界路线保留为可复现基线和回退。

checkpoint 晋级必须使用固定 `gated + mean + boundary_threshold=0.55` 的完整部署验证。
GT 前景上的 affinity 连通分量重建属于 Oracle 诊断，禁止作为选模指标。详细协议见
`docs/AFFINITY_DEPLOYMENT_EVALUATION.md`。

### V6 语义锚点来源（协议 C：LoRA 全链路）

1. **自监督 LoRA 预训练**（`tools/pretrain_lora_ssl.py`）：1000 无标签图 MAE 掩码重建，只训 LoRA + 重建头 → `outputs/lora_pretrain/lora_state_dict.pth`。
2. **Stage-1 监督**（`train.py`）：冻结 trunk + LoRA，双 FPN 随机初始化，语义 BCE+Dice / 边界 Focal×EDT。
3. **Stage-2 联合微调**（`train_stage2.py`）：双分支联合 + LoRA，有标签监督 + 无标签边界一致性（`stage1_direct` 缓存）。

关键组件：
- 冻结 SAM2 Hiera trunk（4 尺度 112/224/448/896 ch）；
- 独立双 FPN 解码头（seg/boundary 各自 256ch）；
- LoRA 注入注意力 qkv/proj（rank16≈1M 参数，梯度检查点已启用）；
- 后处理：边界头概率 + 补缺式语义梯度融合 → 单阈值 → 桥接（双线合并单线）→ 骨架 → 受阻分水岭。

## 4. 模块地图

| 文件 | 职责 |
|------|------|
| `models/sam2_encoder.py` | 冻结 Hiera trunk，`trainable_lora` 控制梯度 |
| `models/lora.py` | LoRA 注入 / 梯度检查点 / 状态存取 |
| `models/fpn_decoder.py` | 独立双 FPN + 输出头，宽容加载 |
| `data/dataset.py` / `dataset_semi.py` | 有标签流 / 无标签弱强双流 |
| `utils/loss.py` / `loss_semi.py` | BoundaryLoss / 半监督一致性（含峰值 hinge）|
| `utils/post_process.py` | 语义融合、桥接、骨架、受阻分水岭 |
| `tools/pretrain_lora_ssl.py` | 自监督 LoRA 预训练 |
| `tools/precompute_semantic_boundary.py` | 语义梯度单线边界伪标签生成 |
| `tools/preprocess_labels.py` / `precompute_pseudo_labels.py` | GT 净化 / 边界伪标签缓存 |
| `inference.py` | 推理入口（语义投票实例分类）|

## 5. 数据管线

- 有标签：`data/raw/` + `data/purified_gt/*_gt.npz`（semantic+boundary），在线 letterbox 1024、翻转/旋转/crop(512)、EDT 边界权重 [1,4]。
- 无标签：`data/unlabeled/`，弱图仅 letterbox（教师源）、强图加空间/外观增强 + 均值填充 patch mask。
- 划分：train/val = 0.8/0.2，seed 42。

## 6. 配置参考（`config/default_config.yaml`）

- `paths/sam2/decoder/data/train/boundary`：基础结构。
- `inference`：threshold / boundary_threshold（单阈值）/ boundary_logit_scale / sem_edge_merge_weight（补缺式融合 λ）/ sem_edge_boost_alpha / bridge_width / tta / checkpoint。
- `lora`：enabled / rank / alpha / target_layers / lr_ratio / init_from / freeze。
- `semi_supervised`：freeze、boundary_teacher_mode、unsup_weight（边界一致性）/ unsup_seg_weight（语义一致性，当前 0）、boundary_consistency（sobel/tv/bg/margin/pos/peak/rate）、skeleton_filter、patch_mask、monitor。
- `progressive_aug`：学生输入外观增强。

## 7. 训练与推理命令

```bash
# 自监督 LoRA 预训练
python tools/pretrain_lora_ssl.py --config config/default_config.yaml --epochs 30 --batch_size 8

# Stage-1 监督（LoRA 开）
python train.py --config config/stage1_lora.yaml

# Stage-2 联合微调
python train_stage2.py --config config/stage2_joint_v3.yaml \
    --init_from_checkpoint outputs/stage1_lora/best_model.pth --phase joint --tag lora

# 边界精修（语义单线伪标签）
python tools/precompute_semantic_boundary.py --config config/default_config.yaml --checkpoint <best> --check_val
python train_stage2.py --config config/stage2_v6.yaml --init_from_checkpoint <best>

# 推理
python inference.py --config config/default_config.yaml --checkpoint <best>
```

Checkpoint 统一格式：`decoder_state_dict + optimizer + scheduler + lora_state_dict + best_composite_score + config`，读取用 `.get()` 兼容旧键。

## 8. 调试与验证

- `debug_pipeline.py` / `debug_iou.py` / `test_skeleton_watershed.py` / `visualize_instances.py`。
- 训练日志关注：`bnd_output: max/>0.5占比/gap`、test 语义置信度（`>0.8` / 模糊带）。

## 9. 开发规范

- 注释与提交信息用中文（`<type>: 描述`）；`.py` 头保留 `# -*- coding: utf-8 -*-`。
- 新增依赖同步 `requirements.txt`；权重本地化。
- 不破坏既有接口：checkpoint 新键向后兼容；letterbox 约定不可改。

## 10. Git 工作流

- 分支前缀 `codex/`；每次实验单独提交便于回溯。
- `.gitignore` 忽略 outputs/、weights/、config/*（保留 default_config.yaml）。

## 11. 参数方向速查（现象 → 调高/调低）

| 现象 | 方向 |
|------|------|
| 边界热力图雾状（中置信 0.5~0.8 占 18%） | 调高 bg_suppress / TV；调低 pos_weight；提升输出分辨率（受 256 亚像素限制，收益有限） |
| 边界双线/骨架曲折 | 调高 bridge_width（合并双线）；距离变换单脊目标 |
| 欠分割（实例偏少） | 调低 boundary_threshold / 调高 boundary_logit_scale / sem_edge_merge_weight |
| 过分割（实例偏碎） | 调高 boundary_threshold / min_instance_area |
| 边界峰值平台低（>0.7 占比小） | 调高 peak_hinge_weight |
| 边界输出区间压缩（gap<0.4） | 调高 margin_loss_weight / 调低 bg_suppress |
| 圆斑过拟合（黑斑处输出 0） | patch mask 用均值填充（已修复），掩码区权重调低 |
| 语义往均值退化 | seg_lr_ratio 保持 0.1~0.3；关闭 sem_boundary_align_weight |
| 边界分支漏检铁素体内部晶界 | 语义补缺式融合 / 语义单线伪标签训练 |
