# G1 直接部署候选：删除 G2 的第一步检验

最新回分补记：用户报告第一组mIoU=0.8442、面积项=0.6860，其余C/X尚未提交；按此前提交顺序对应本D候选。
按50/50复算总分76.510，低于历史G2的84.415，当前不满足直接删除G2的条件。
包身份对应依据会话顺序，尚无平台回执中的哈希核对；详细解释见[G1消融结果](G1_ABLATION_RESULTS_20260912.md)。以下保留提交前的历史诊断。

2026-09-12 完成 G1 best(epoch18) 的 68 图推理、提交包验证，以及 G1/G2 同六图新 GT 诊断。
当前结论：候选可提交；是否删除 G2 仍待同一官方评测口径的结果。默认融合部署包未切换，C/X 配对训练尚未开始。
后续更新：用户随后授权提前启动 C/X，原文中的等待安排已由 [配对训练启动记录](G1_ABLATION_TRAINING_20260912.md) 覆盖；本报告的 D/H 诊断结果不变。

## 可直接提交的产物

- 本地：`output/20260912_g1_g2_ablation/submission_g1_e18_high065_20260912.zip`，2,875,221 字节。
- 服务器项目根：`/root/autodl-tmp/segmentationv2_work`。
- 服务器相对路径：`outputs/20260912_g1_g2_ablation/submission_g1_e18_high065_20260912.zip`。
- 包内为 68 张图对应的 136 个文件：`*_inst.png`、`*_class.json`，无嵌套目录。
- `tools/package_submission.py` 已检查原图尺寸、单通道 uint16、实例与类别 ID 严格对应、类别 0/1、ID≤65535，以及 ZIP CRC。

68 图共 4,736 个实例，最大 ID 为 161。历史 G2 输出为 5,897 个实例，最大 ID 为 225。
实例数减少本身不代表效果改善；没有利用这些统计量反推测试 GT 或调阈值。

## 固定协议和权重身份

服务器代码版本：`f710a9be6a4e2556eacb77b462a3523601ab25fb`；本地记录所在 checkout 为 `2dbc9bf`。
本轮未修改训练/推理源码，仅使用已有推理入口、一次性编排/分析脚本和文档。

推理入口为 `tools/run_affinity_submission.py`，显式 `--checkpoint` 指向各臂最终权重。
配置 `outputs/20260912_g1_g2_ablation/inference.yaml` 继承 E10a/0.65 方案，仅关闭可视化保存。
固定 legacy_none、1024 输入/512 输出、E10a challenger_only、gated/mean、high=0.65、seal2、low=0.45、局部重建 8 步、probability_mean、受阻分水岭。

| 角色 | checkpoint 相对路径 | epoch | SHA256 |
| --- | --- | ---: | --- |
| D / G1 | `outputs/affinity_geometry_g1/best_affinity.pth` | 18 | `635c6b7e34bd0b9603ed55fed3e291c0c944db460f6a6c6c6ef9d5a725db1bd4` |
| H / G2 | `outputs/affinity_geometry_g2_sam2/best_affinity.pth` | 21 | `96f9456e4562e9e0a8fa715160582c8f8aee28deceaf32324ac38ae1570ed445` |
| 固定 E10a | `outputs/stage2_semantic_e10a_cold20/best_model_stage2.pth` | 19 | `1380cf14d63fdbc17dadf2ac73dc47d7c33a7e0f5a61e5edb9d7280ec8b80b91` |

V6 reference SHA256：`c4c9827a18ecdda105056d502e5f93a92fe7fc93ada24d63886b20726c7ec557`。
G1/G2 六图与 68 图 manifest 的 fusion、inference、reference SHA、semantic_source、semantic_challenger 元数据逐键一致；G1 实际加载身份为 epoch18/SHA635c6b7e，并未被默认 G4b 覆盖。
历史 H 的 68 图输出直接复用 `outputs/submission_g2_20260911`。H 的官方成绩 84.415 来自既有提交记录，本轮没有重新取得官方成绩。

## 同六图诊断

固定图像：`172、176、349、623、873、888`，原图尺寸输出。
采用 `outputs/experiments/seeded_label_completion_o1/radius_8/derived_maps` 的补缝新 GT，共 909 个已知实例；剩余 unknown 忽略。
使用现有 `load_direct_evaluation_target`、`mask_prediction_to_evaluation_domain`、`evaluate_instance_pair` 与 `summarize_instance_results`。
这些图参加过几何阶段的验证选优，是内部诊断集，不是独立测试集。没有将本轮结果与旧五图分数混比。

| 诊断项 | G1 / D | G2 / H |
| --- | ---: | ---: |
| 有效匹配数 | 561 | 729 |
| 有效匹配实例 mIoU | 0.8287 | 0.8445 |
| GT 惩罚 mIoU | 0.5115 | 0.6773 |
| 预测铁素体实例数（GT=505） | 518 | 715 |
| 铁素体平均面积相对误差，越小越好 | 6.97% | 30.77% |
| 整体汇总代理总分 | 87.95 | 76.84 |
| 逐图代理总分再平均 | 83.27 | 77.30 |

G1 在六张图上的有效 mIoU、GT 惩罚 mIoU 均低于 G2，面积误差则均小于 G2。
因此这次面积改善不仅是整体平均时的正负误差抵消；但 G1 仍有逐图偏差，整体面积误差 6.97% 不能代表每张图。
已知 GT 域内未分配像素比例的逐图均值：G1 1.10%、G2 1.25%；它不是测试集覆盖率，也不能衡量晶粒边界是否分对。

| 图像 | G1 匹配数 | G2 匹配数 | G1 GT惩罚 mIoU | G2 GT惩罚 mIoU |
| --- | ---: | ---: | ---: | ---: |
| 172 | 74 | 125 | 0.3470 | 0.6016 |
| 176 | 77 | 111 | 0.3640 | 0.5366 |
| 349 | 91 | 103 | 0.7071 | 0.8079 |
| 623 | 94 | 110 | 0.6688 | 0.8067 |
| 873 | 114 | 140 | 0.5588 | 0.7000 |
| 888 | 111 | 140 | 0.5393 | 0.6953 |

## 测试输出检查和下一步

68 图的原图尺寸、实例/类别对应关系已完整检查；固定抽查 `test_001、test_024、test_044` 的原图/G2/G1叠图。
`test_001` 的 G1 切分更粗，部分 G2 可见的分界消失；`test_044` 两者都存在跨越可见晶界的大连通区域。
这与“G1 面积项更好、实例匹配更弱”的内部诊断相容，但三图目检不能替代官方评价。

下一步提交 D 的固定候选包，与 H 的同口径官方成绩比较。结果未回时保留 G1+G2 的选择；不提前训练 C/X，也不扫描阈值救回 D。
若 D 不满足删除 G2 条件，再按已审阅方案比较 `G1→G2` 与 `G0-long→G2`；后续开训前仍需核对原 64 图 SAM2 的实际审核记录。

## 本地记录

- `output/20260912_g1_g2_ablation/results.json`：精简逐图诊断、整体/宏平均、各臂 manifest、68 图数量对照。
- `output/20260912_g1_g2_ablation/test_count_comparison.csv`：68 图实例/类别数量差异。
- `output/20260912_g1_g2_ablation/test_comparison.jpg`：固定三图对比，一张图片。
- `output/20260912_g1_g2_ablation/launch.json`、`inference.yaml`、`package_g1.log`：启动参数、已核对权重与打包结果。
- `output/20260912_g1_g2_ablation/complete_deployment.py`、`evaluate_completed.py`：本次一次性编排与 CPU 分析脚本。

未复制原始数据或基础权重，未保存中间 affinity 缓存。默认主线配置和融合权重未改动。
