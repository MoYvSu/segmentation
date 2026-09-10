# 实验与复现索引

部署主线仍为 **E10a + G4b，high=0.65、seal2、局部重建、受阻分水岭**，官方总分
83.94。候选实验的六图代理、训练损失和目检结果使用各自名称记录，不替代官方成绩。
部署命令与产物约定见 [PIPELINE.md](PIPELINE.md)。

## 当前实验

复用同一 SSL LoRA，训练直接输出实例掩码及类别的模型。固定主线生成 249 张伪标签；
人工与伪标签每轮分别抽样 64、192 次，伪标签损失系数 0.5。两组都从相同 SSL 和随机种子
开始，各训练 60 轮解码器预热与 60 轮 LoRA 联训，仅第二组加入权重 0.5 的像素归属损失。
选优使用同一人工验证集的原监督损失，伪标签保留 `mainline_pseudo` 来源。

| 用途 | 入口 | 说明 |
| --- | --- | --- |
| 生成伪标签 | [mainline_pseudo_generation249.yaml](../config/train/mainline_pseudo_generation249.yaml) | 固定主线完整部署输出，排除人工划分与无标签留出图 |
| 仅加入伪标签 | [mask_set_teacher249.yaml](../config/train/mask_set_teacher249.yaml) | 归属损失权重为 0 |
| 加入归属约束 | [mask_set_teacher249_ownership.yaml](../config/train/mask_set_teacher249_ownership.yaml) | 鼓励有效像素由唯一实例负责，unknown 保持 ignore |
| 顺序作业 | [run_mask_set_teacher_ab.py](../tools/run_mask_set_teacher_ab.py) | 生成完成后依次训练两组；当前服务器已有作业，不要重复启动 |
| 完整输出评估 | [evaluate_mask_set.py](../tools/evaluate_mask_set.py) | 使用 checkpoint 固定六图划分，在原尺寸比较最终输出 |

2026-09-10 **22:41 +08:00** 的核验快照：249 张伪标签已完成，第一组完成预热第 47 轮，
第二组尚未开始。这是带时间的快照；实时状态以服务器作业的 `state.json`、`metrics.csv`
和完成标记为准。路径、预览、筛选限制及取回方法见
[主线伪标签实验](MASK_SET_TEACHER_PSEUDO_EXPERIMENT_20260910.md)。

## 已完成对照与数据依据

| 路线 | 结论与适用范围 | 记录 |
| --- | --- | --- |
| 新 GT 补缝 | 分水岭补齐接缝，剩余 unknown 忽略；派生标签与源标注区分保存 | [新 GT 定义](MANUAL_TARGET_V2_EXPERIMENT.md) |
| clean60 双头 | 同一部署方式下未恢复 G4b 几何；方向融合与监督差异仍需分别解释 | [差距诊断](AFFINITY_CLEAN60_DIAGNOSIS_20260910.md) |
| 双头原尺寸局部采样 | e86 损失最优，完整部署比较仍不支持晋级 | [训练方案](AFFINITY_CLEAN60_NATIVE_CROP_20260910.md)、[最终比较](AFFINITY_NATIVE_CROP_COMPARISON_20260910.md) |
| 直接掩码首轮 | e118 最终效果低于主线，发现 BF16 下共享投影无梯度 | [实现](MASK_SET_CLEAN60_EXPERIMENT_20260910.md)、[首轮结果](MASK_SET_RESULTS_20260910.md) |
| 掩码投影修复 | e103 最终六图几何改善但仍低于主线，后续进入伪标签对照 | [修复重跑](MASK_SET_AMPFIX_RERUN_20260910.md)、[结果与决策](MASK_SET_AMPFIX_RESULTS_20260910.md) |
| GT 前景 Oracle | 仅诊断几何可达性，不是部署成绩 | [M0 诊断](MUSAM_GT_ORACLE_M0.md) |

## 文件与版本边界

- `models/`、`data/*.py`、`utils/`、`tools/` 与根目录训练脚本是正式源码；`config/` 保存可移植配置，`tests/` 保存回归检查。
- `docs/` 保存方案、结论及证据链接；[output/README.md](../output/README.md) 索引历史实验现场、脚本、轻量指标和正文关键图。
- 权重、原始数据、完整预测、压缩包和批量预览留在本地或服务器。`output/`、`tmp/` 的忽略规则不删除文件；其中明确纳入版本库的历史脚本保留当时路径和用途。
- [tmp/README.md](../tmp/README.md) 说明临时目录中的历史辅助材料。服务器正在运行的实验目录、数据链接和模型链接不随本地整理迁移。
- 本次源码、配置和检查结果作为候选实验版本保存；不合并或替换已验证的部署主线。
