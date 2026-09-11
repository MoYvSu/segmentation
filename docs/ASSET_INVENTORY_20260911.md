# 服务器资产台账

> 2026-09-11 建立。记录两台训练服务器上的**可复用资产**及其校验值，供后续会话直接查证，
> 不必再靠人工翻找。所有 SHA256 均为本次实际计算，不是从旧文档誊抄。
>
> 机器可读版本：[`asset_inventory_20260911.json`](asset_inventory_20260911.json)。
> **本文件只登记资产位置与校验值，不替代各实验的正式报告。**

## 1. 两台服务器

| | bjb1 | bjb2 |
|---|---|---|
| 连接 | `ssh -p 23411 root@connect.bjb1.seetacloud.com` | `ssh -p 15930 root@connect.bjb2.seetacloud.com` |
| 容器 | `autodl-container-cbfe4794e2-b3f19bec` | `autodl-container-smyulw5z7c-6ebed1b6` |
| GPU | RTX 4090 24G | **无（无卡模式）** |
| CPU / 内存 | — | 128 核 / 1007 GB |
| `/root/autodl-tmp` | 50G，已用 18G，**剩 33G** | 50G，已用 31G，**剩 20G** |
| 角色 | 训练与部署（唯一有卡） | 历史归档 |

**共享盘**：`/autodl-fs/data`（200G，已用 721M）在两台上是**同一个挂载**
（`AutoFS:fsbjbsecond929794`），可写，是现成的中转通道。

## 2. bjb1 — 有卡机

`/root/autodl-tmp/` 下 10 个项目目录。**关键资产集中在 `segmentationv2/`**：

| 资产 | 路径 | 大小 | 说明 |
|---|---|---:|---|
| **部署包** | `segmentationv2_pseudo_20260908/outputs/deployment/e10a_g4b_fused.pth` | 326,836,899 | E10a+G4b 融合包，唯一提交级产物 |
| V6 锚点 | `segmentationv2/outputs/stage2_v6/best_model_stage2.pth` | 100,791,707 | |
| SSL LoRA | `segmentationv2/outputs/lora_pretrain/lora_state_dict.pth` | 4,110,259 | |
| Stage1-LoRA | `segmentationv2/outputs/stage1_lora/best_model.pth` | 157,366,961 | |
| joint-v3 | `segmentationv2/outputs/stage2_joint_v3/best_model_stage2.pth` | 157,371,515 | |
| 语义边界缓存 | `segmentationv2/outputs/pseudo_labels/semantic_boundary/boundary_probs.npy` | 2,097,152,128 | |
| SAM2 权重 | `segmentationv2/weights/sam2_hiera_base_plus.pt` | 323,493,298 | |
| 数据 | `segmentationv2/data/` | | raw 64 / purified_gt 33 / unlabeled 1000 / test 68 |

**bjb1 缺少的**（本次从 bjb2 补齐，见 §4）：
`affinity_geometry_g0/g0_long/g1/g2_sam2/g4b`、`stage2_semantic_e10a_cold20`、
`data/sam2_geometry_g2`。

其余目录：`segmentationv2_main`（**失效 worktree**，`.git` 指向不存在的
`segmentationv2/.git/worktrees/`）、`segmentationv2_repo`（1.7M）、
`segmentationv2_clean60_20260908_133740`（3.6G）、`segmentationv2_cross_head_20260911_1642`（34M）、
`segmentationv2_maskset_*` 四个目录。

## 3. bjb2 — 归档机

| 目录 | 大小 | 内容 |
|---|---:|---|
| `segmentationv2_repo` | 9.3G | **最完整的历史归档**，affinity 全链在此 |
| `segmentationv2` | 13G | |
| `segmentationv2_hiera_l_20260906` | 7.5G | Hiera-L 骨干实验 |
| `segmentationv2_direct_dual_20260831` | 1.4G | |
| `segmentationv2_main` | 80M | |
| `segmentationv2_backup_center` | 164K | 源码片段 |

**`segmentationv2_repo/outputs/` 含 affinity 全链**：
`affinity_geometry_g0`、`_g0_long`、`_g1`、`_g2_sam2`、`_g3_native_crop`、`_g4_manual_gap`、
`_g4b_gap_weight020`、`_g5_feature_adapter`、`_g6_negative_tail`、`_g7_highres_short`、`_oracle`。

另含 `stage2_semantic_e7b/e8/e9/e10a_cold20`、`stage2_v6`、`outputs/experiments/`（A/B 现场）、
`outputs/packages/G4b_high065_submission_20260828.zip`。
`data/sam2_geometry_g2/` 完整（`boundaries/` + `manifest.jsonl` + `masks/` + `overlays/`）。
全机 273 个 `.pth`。

## 4. 校验通过的权重（本次实算）

六个关键权重中有五个有独立锚点，**全部对上**：

| 资产 | SHA256 | 锚点来源 |
|---|---|---|
| `affinity_geometry_g0/latest_affinity.pth` | `8ea618291d6927f2f87222ca14f4bee4aa0ff68c4a51db9c412110c51c88f4d5` | `config/reproduction_mainline_8394_from_v6.yaml` |
| `affinity_geometry_g0_long/latest_affinity.pth` | `d5a7579ddd3b6db93c058467b64a1609fde625d43a6c86a3ff1e9518cb01a845` | 同上 |
| `affinity_geometry_g1/best_affinity.pth` | `635c6b7e34bd0b9603ed55fed3e291c0c944db460f6a6c6c6ef9d5a725db1bd4` | 同上 |
| `affinity_geometry_g2_sam2/best_affinity.pth` | `96f9456e4562e9e0a8fa715160582c8f8aee28deceaf32324ac38ae1570ed445` | — |
| `affinity_geometry_g4b_gap_weight020/latest_affinity.pth` | `1334fdcc8472b4da67386b76b25d9099ad6bcef10b6bd9c77ae6908658b8c619` | 部署包 `sources` |
| `stage2_semantic_e10a_cold20/best_model_stage2.pth` | `1380cf14d63fdbc17dadf2ac73dc47d7c33a7e0f5a61e5edb9d7280ec8b80b91` | 部署包 `sources` |
| **部署包** `e10a_g4b_fused.pth` | `53e1b5c6f9bf3973723c7bf4c6ce4ebd5eaeaa45210641df9de8f01e510c790f` | 交接文件 |
| V6 `stage2_v6/best_model_stage2.pth` | `c4c9827a18ecdda105056d502e5f93a92fe7fc93ada24d63886b20726c7ec557` | 交接文件 + 部署包 |

**结论**：bjb2 上这条 affinity 链是当年的原版训练链，不是重训产物。G4b/E10a 的原始
源路径按部署包 `sources` 记录为
`/root/autodl-tmp/segmentationv2_repo/outputs/...`，与 bjb2 上的实际位置一致。

## 5. 中转与恢复记录（2026-09-11）

**中转区**：`/autodl-fs/data/asset_transfer_20260911/`（1082 文件 + `MANIFEST.sha256`）

**bjb1 恢复落点**：`/root/autodl-tmp/_restored_assets_20260911/`（721M）
**逐文件校验：1082 / 1082 通过，0 失败。**

搬运内容与文件数：

| 目录 | 文件数 |
|---|---:|
| `affinity_geometry_g0` | 3 |
| `affinity_geometry_g0_long` | 9 |
| `affinity_geometry_g1` | 155 |
| `affinity_geometry_g2_sam2` | 192 |
| `affinity_geometry_g4b_gap_weight020` | 282 |
| `stage2_semantic_e10a_cold20`（仅 `best_model_stage2.pth`） | 1 |
| `data/sam2_geometry_g2` | 193 |

**未搬运**：bjb2 的 `purified_gt_uncovered/`（32 项）、`affinity_geometry_g3/g5/g6/g7/oracle`。
需要时再走同一条通道。

## 6. 清理记录

**2026-09-11 删除三份 `segmentationv2.zip`**（各 4,061,095,368 字节）：

| 位置 | 删除前剩余 | 删除后剩余 |
|---|---:|---:|
| bjb1 `/root/autodl-tmp/` | 29G | **33G** |
| bjb2 `/root/autodl-tmp/` | 16G | **20G** |
| 共享盘 `/autodl-fs/data/` | — | 清空 |

**删除前已验证三份逐字节相同**：
`8b1c841f8d2e8ffbd15f0bafd069e8ecd7c7e48b2d8afce8bd39b8a906b37d8b`。
用户确认为数月前的旧快照，两台服务器上均有更新的完整项目目录，Git 仓库为权威来源。

## 7. 待办与已知缺口

- **`data/sam2_geometry_g2/approval.json` 从未存在过** —— 用户确认历来默认通过。
  `repro/train.py` 会在 G2/G4b 前校验该文件，因此直接跑编排器会被拦住。
- `segmentationv2_main` 是失效 worktree，`.git` 指向已不存在的路径。
- `segmentationv2`、`segmentationv2_main`、`segmentationv2_repo` 在 bjb1 上内容重复。
- bjb1 的 `segmentationv2` 数据最全但代码版本较旧（Aug 5）；`segmentationv2_cross_head_20260911_1642`
  代码最新但缺 `purified_gt` 与 `unlabeled`。
- 远程仓库 `https://github.com/MoYvSu/segmentation.git` 为**公开**，服务器无需凭据即可访问。
  但本地分支 `codex/mainline-cross-head`（`67360e4`）**尚未推送**。
