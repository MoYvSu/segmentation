# 训练与推理入口

状态更新：2026-11-01。当前部署仍使用旧 **13b3 backend＋9315 D5a＋20a59 NAF**。
训练配方可执行、训练已经完成、候选获得官方分数是不同证据，不能相互替代。

| 用途 | 推荐入口 | 状态与边界 |
|---|---|---|
| 原人工标注＋r8重训配方 | `python train_pipeline.py`，配置`repro/current_pipeline.yaml` | 默认13阶段；尚无这份原GT配方从头完整GPU实跑的记录，不承诺重现原C权重或得分 |
| 教师32源演习 | 同一入口，配置`repro/teacher_pipeline.yaml` | 监督链跨两段run完成；SSL/D5a导入而未重训；150图用户回分约80.4，候选淘汰 |
| 原C Torch推理 | `python infer.py --backend torch`，默认即`torch` | 调用既有`tools/infer_compact.py`；默认C，保留两路全图语义及局部N1/NAF/S0/final200 |
| 全五TensorRT候选 | `python infer.py --backend tensorrt --engine-manifest <清单>` | 调用`tools/infer_accelerated.py`并包含CPU整数读出；150图已完成、下载及本地验收通过，待本包独立回分 |
| 旧E10a/G4b历史链 | `python repro/train.py` | 仅供下方历史记录，不覆盖当前S-align/V/D5a/R1/NAF三模型链 |

`train_pipeline.py`转发给`repro/pipeline.py`，不改变配置、预算或资产检查。只读命令为：

```bash
python train_pipeline.py list
python train_pipeline.py check --code-only
python train_pipeline.py run --dry-run --config <私有资产映射.yaml> --run-dir outputs/repro_current/preview
```

`--code-only`不检查数据是否齐全；正式运行使用新的私有run目录，实际输入需提供原32标注、purified GT、
r8、1000图、原授权SAM64和官方底座，入口不生成或修订GT。默认链已[移除V6及无用缓存](../docs/V6_ELISION_20261101.md)，
S-align直接取joint固定语义教师；旧15阶段仅在显式`semantic_teacher_source: legacy_v6_e9`时启用。
SSL新配方batch2是原15000更新的条件性强推断，mask .4是恢复默认值，完整历史seed未固定；
不能把这些新选择写成已恢复原CLI。详见[当前训练入口](../docs/CURRENT_TRAINING_ENTRY_20261101.md)、
[实际参数及历史缺口](../docs/TRAINING_RECIPE_20261101.md)和[推理接口](INFERENCE_INTERFACE.md)。
完整命令、产物位置与独立引擎构建集中在[训练与推理使用指南](../docs/TRAINING_INFERENCE_GUIDE.md)。

[教师链训练结果](../docs/TRAINING_RESULT_TEACHER32_20261101.md)证明工程执行完成；
[150图回分](../docs/TEACHER32_C150_INFERENCE_20261101.md)为`.7106/.8975`、等权参考80.405，
低于原全零C的82.660，未采用。它不构成原GT配方或原C从零复现的证据。

## 历史记录：E10a/G4b旧训练入口

以下保留旧`repro/train.py`的配置、流程及当时83.94链记录，不作为当前推荐入口。
旧SSL命令仍为batch8，与原checkpoint的15000更新存在冲突，见[SSL补证](../docs/SSL_PARAMETER_RECOVERY_20261101.md)。
旧`--skip-existing`按约定输出文件是否存在跳过，不具备新统一入口的完整身份回执检查。

该历史目录提供从赛方数据和官方 SAM2 backbone 开始、训练当时 `83.94` 链的固定顺序、
历史配置和预检入口。历史 checkpoint 只用于结果校验，不是冷启动输入。

训练目标链如下：

```text
LabelMe 标注 ──> purified GT
无标签图 ──────> LoRA SSL ──> Stage1-LoRA ──> Stage1 边界缓存
                                      │
                                      └──> joint-v3 ──> 语义单线边界缓存 ──> V6
                                                                            ├──> E10a
                                                                            └──> G0 -> G0-long
                                                                                     -> G1
已审核 SAM2 无类别几何数据 ───────────────────────────────────────────────────────────> G2 -> G4b
```

E10a 与 affinity 分支在 V6 之后互相独立；编排入口为便于核对而顺序执行。

### 历史目录内容

```text
repro/
  README.md
  train.py
  configs/
    stage1_lora.yaml
    stage2_joint_v3.yaml
    stage2_v6.yaml
  sam2_geometry_approval.example.json
  INFERENCE_INTERFACE.md
```

三份 YAML 是完整有效配置，不依赖会继续变化的 `config/default_config.yaml`。它们来自旧训练
服务器的历史文件/运行快照，只把绝对 `project_root` 改为 `auto`；joint-v3 和 V6 额外显式
设置了当前配置加载器需要的 `base_checkpoint`，目标 checkpoint 与历史命令保持一致。

### 历史冷启动输入

在仓库根目录准备：

```text
data/raw/                              32 JPG + 32 LabelMe JSON
data/unlabeled/                        1000 张赛方无标签图
weights/sam2_hiera_base_plus.pt        官方 SAM2 Hiera B+ 权重
segment-anything-2/                    仓库约定的 SAM2 子模块
```

环境继续使用项目的 `requirements.txt` 和服务器 `sam2_env`。历史 V6 运行环境为 Python
`3.12.3`、PyTorch `2.8.0+cu128`、RTX 4090；PyTorch/CUDA 应按实际服务器驱动安装，不在本目录
复制第二套依赖文件。

### 旧入口使用方法

在仓库根目录运行：

```bash
conda activate sam2_env

# 只检查源数据、权重、配置和路径，不构建模型
python repro/train.py check

# 查看全部阶段及实际命令
python repro/train.py list

# 打印完整顺序，不训练
python repro/train.py run --dry-run

# 真正从第一阶段开始执行
python repro/train.py run
```

编排器默认拒绝覆盖任何已存在的阶段产物。确认已有产物属于同一次复现时，可以从指定阶段接续：

```bash
python repro/train.py run \
  --from-stage joint_v3 \
  --to-stage v6 \
  --skip-existing
```

`--skip-existing` 只跳过已经生成最终约定产物的完整阶段。某个长训练在阶段内部中断时，先使用
原训练脚本的 `--resume` 恢复，得到该阶段的最终约定产物后，再让编排器继续下一阶段。

### 旧入口固定训练阶段

| 顺序 | 阶段 | 主要产物 |
|---:|---|---|
| 1 | `prepare_labels` | `data/purified_gt/` |
| 2 | `lora_ssl` | `outputs/lora_pretrain/lora_state_dict.pth` |
| 3 | `stage1_lora` | `outputs/stage1_lora/best_model.pth` |
| 4 | `stage1_pseudo` | `outputs/pseudo_labels/stage1_boundary/boundary_probs.npy` |
| 5 | `joint_v3` | `outputs/stage2_joint_v3/best_model_stage2.pth` |
| 6 | `semantic_pseudo` | `outputs/pseudo_labels/semantic_boundary/boundary_probs.npy` |
| 7 | `v6` | `outputs/stage2_v6/best_model_stage2.pth` |
| 8 | `e10a` | `outputs/stage2_semantic_e10a_cold20/best_model_stage2.pth` |
| 9–11 | `affinity_g0`、`affinity_g0_long`、`affinity_g1` | 各阶段 affinity checkpoint |
| 12 | `sam2_geometry_review` | 人工确认门槛，不训练 |
| 13 | `affinity_g2` | `outputs/affinity_geometry_g2_sam2/best_affinity.pth` |
| 14 | `affinity_g4b` | `outputs/affinity_geometry_g4b_gap_weight020/latest_affinity.pth` |

V6 不能直接使用 Stage1 边界缓存。必须先由 joint-v3 运行
`tools/precompute_semantic_boundary.py`，生成 `semantic_boundary` 单线缓存；该步骤已经纳入编排。

### 历史G2数据与人工确认门槛

G2/G4b 使用无类别 SAM2 几何候选。若 `data/sam2_geometry_g2/` 尚不存在，先生成 64 张候选：

```bash
python tools/build_sam2_geometry_dataset.py \
  --input-dir data/unlabeled \
  --output-dir data/sam2_geometry_g2 \
  --count 64 \
  --start-index 0 \
  --exclude-labelme-dir data/raw
```

随后人工检查 `overlays/`、`manifest.jsonl` 和对应 mask，确认：

- 所有源图来自赛方无标签训练集，且不与人工 LabelMe 图重复；
- 候选已完成互斥仲裁，每张最多 255 个实例；
- 候选没有铁素体/珠光体类别，未覆盖区域继续为 `ignore`；
- 明显跨越真实晶界、重复覆盖或失真的候选已经排除/修订。

确认完成后，以 `repro/sam2_geometry_approval.example.json` 为模板创建
`data/sam2_geometry_g2/approval.json`，填写审核人和时间并将 `approved` 改为 `true`。
`repro/train.py` 在该文件通过校验前不会启动 G2/G4b。

### 历史锚点与验收

这些哈希用于比较冷启动结果，不应复制成训练输入：

| 产物 | 历史 SHA-256 | 历史信息 |
|---|---|---|
| LoRA SSL | `4701e770e658a4f152da695c4b3a3dbb3a0b66152d63c03626399cfea2f4c152` | 30 epoch |
| Stage1-LoRA | `62037119de3a3628941c86fd04bf8a51858ce30f8b05e138eb3cd2dc7eb01c2f` | 协议 C Stage1 |
| joint-v3 | `dde542325a28666de730a72d6f77f37149c8ac02b44c5434cddad74725f317ce` | commit `7ff26d6` |
| V6 | `c4c9827a18ecdda105056d502e5f93a92fe7fc93ada24d63886b20726c7ec557` | commit `2b5d6e4`，epoch 9 |

CUDA 算子、库版本和数据加载并行可能使冷启动 checkpoint 无法逐字节复现。验收以配置快照、
数据清单、训练曲线、最终输出契约和完整部署指标为主；历史 V6 指标为 mIoU `0.8254672504`、
boundary IoU `0.5116020453`、composite `0.5743750863`，最终 E10a+G4b 黑盒成绩为
`0.8381 / 0.8408 / 83.94`。

Stage1-LoRA 当时没有独立 `run_info.json`。其训练链对应 2026-08-04 的协议 C 提交序列，最终
准备提交可定位到 `d9c83308`，但这里不把该推断写成精确运行提交。joint-v3 和 V6 均有原始
run snapshot，提交号已经确认。

### 当时的推理接口计划

旧训练入口编写时只固定E10a/G4b的推理产物输入，未创建计划中的`repro/inference.py`。
当前实际推理使用顶层`infer.py`；旧计划和当前接口分别保存在
[`INFERENCE_INTERFACE.md`](INFERENCE_INTERFACE.md)，不能把旧计划命令当成现成入口。
