# clean60 原尺寸局部采样：尺度证据与训练交付

2026-09-10。前置分析见 [差距诊断](AFFINITY_CLEAN60_DIAGNOSIS_20260910.md)。

> 完成更新：训练已跑完 120 轮，损失最优为 e86。固定部署比较显示面积项改善，但六图实例 mIoU 与有效匹配下降，未晋级。完整结果见 [训练后比较分析](AFFINITY_NATIVE_CROP_COMPARISON_20260910.md)。下文保留启动时的方案和交付记录。

已实现并启动一个可回退的训练候选：人工新 GT 训练样本以 50% 概率使用原图中的 1024×1024 随机方窗，另外 50% 保留整图。初始化、模型、损失权重、名单、每轮更新数与 60+60 阶段设计沿用 clean60。主线仍为 E10a+G4b，正式训练结果尚未完成部署评价。

## 为什么现在做这个实验

上一轮已经证明新版在真实跨粒连接上抑制较弱，且换 top2 不能恢复主线几何；但这仍不能区分训练表征不足与输入尺度影响。本轮先固定三个局部，再对同一 checkpoint 做整图输入/原尺寸裁剪输入对照。

模型输入都为 1024，输出都为 512。原始训练图 2584×1936 整图缩小时，每个输出格约对应 5.05 个原图像素；1024 原尺寸裁剪时对应 2 个原图像素。也就是说裁剪保留了更多细节，同时减少了上下文。因此本轮不把两种尺度的同名方向通道当作同一物理连接比较。

两种输入均先在各自输出网格执行主线 gated+mean 融合，再按正式插值方法恢复原尺寸。评分使用**完全相同的原图像素集合**、high=0.65 和固定原图 6px 容差，不改融合或阈值。两模型分别使用各自 encoder/LoRA 和归一化。

在运行前固定的局部为 train_172 的 GT107/168、train_873 的 GT141/142，以及 train_351 的既有正常区域。前两处来自上一轮失败证据，第三处作为正常对照；这些验证图不进入本次训练采样名单。

指定 GT 对的真实共享界面上，e110 的结果如下。该指标表示界面像素附近是否有高响应，不能等同于最终实例已分开。

| 预选 GT 界面 | 像素数 | 原图同位置 B>0.65：整图→裁剪 | 固定 6px 容差覆盖：整图→裁剪 |
|---|---:|---:|---:|
| train_172：107/168 | 351 | 29.63% → 98.29% | 57.26% → 100% |
| train_873：141/142 | 323 | 82.97% → 96.28% | 94.12% → 100% |

改善确实发生在指定的失败分界，不只是方框内其它边界变亮。G4b 在这些分界上原本已有较强响应。由此可以支持“e110 的部分问题对尺度/上下文敏感”，但不能推出“裁剪训练必然改善整图部署”。

副作用也已经出现：

| e110 固定局部 | ROI 内全部界面的同位置高响应：整图→裁剪 | 晶粒深部误响应：整图→裁剪 |
|---|---:|---:|
| train_172 | 85.34% → 98.06% | 2.04% → 4.36% |
| train_873 | 98.70% → 94.95% | 0.029% → 0.053% |
| train_351 正常对照 | 99.84% → 99.17% | 0.082% → 0.197% |

深部排除了 GT 界面、unknown、原图边缘的 21px 邻域。train_873 指定失败分界改善，但其它界面在同位置的响应有所下降；固定 6px 容差下该 ROI 覆盖从 99.55% 到 100%，存在边界位置/宽度变化。train_172 靠原图底部，不能取得对称的上下文。旧 train_873 漏口的部分示例像素原本已超过 high，也说明高响应不自动保证骨架与种子连通关系正确。

因此本轮只启动一次混合采样训练，不把推理改成裁剪拼接，也不追加更强负边损失、SAM2、教师或 recovery 阶段。局部细节能否在整图部署下产生收益，留给训练完成后的固定输出对照检验。

统一色标图：[native_scale_comparison.png](../output/20260910_native_scale_probe/native_scale_comparison.png)。

原始数值：[summary.json](../output/20260910_native_scale_probe/summary.json)、[fixed_pair_probe.json](../output/20260910_native_scale_probe/fixed_pair_probe.json)。尺度探针实现：[probe_affinity_native_scale.py](../tools/probe_affinity_native_scale.py)。后者只做标量边界诊断，不输出正式实例分割。

## 实现与固定项

- 配置：[direct_gtv2_clean_nativecrop_60x60.yaml](../config/train/direct_gtv2_clean_nativecrop_60x60.yaml)。默认配置不开启新行为。
- 人工新 GT 由同一 canonical 实例图派生语义与 affinity；RGB、实例、语义、有效域及标注来源使用同一原尺寸窗口。实例 ID 不重编号，unknown 继续 ignore。
- 语义边界在完整 GT 上生成后再裁剪，避免把裁剪框制造成新边界；随后沿用现有 letterbox 和同步翻转/旋转。
- 窗口位置均匀随机，不按验证失败位置或界面强弱筛选。样本不足 1024 时，方窗边长限于原图短边。记录 `sample_bbox`、`sample_shape`、`is_native_crop` 便于核对。
- 名单：[clean60_split.json](../config/train/clean60_split.json)，与原 clean60 的 26 张训练/6 张验证完全相同。验证永远整图。相同 seed 不再被当作相同 split 的替代证据。
- 每 epoch 64 个样本、batch=1，共 120 epoch / 7680 次计划更新。裁剪不额外增加 step。
- 复用原 SSL 的 `outputs/20260908_133740_ssl60/best_lora.pth`，重新随机初始化双头；不是从 e110 接着训练。输入归一化仍为 `imagenet_v1`。
- 损失固定为原 clean60 标定值：semantic=7.797723293304443，manual_affinity=0.6930767446756363，pseudo_affinity=1.0（本实验关闭伪标签）。避免新裁剪数据同时改变两个任务的相对权重。
- 跳过原 4 batch 损失校准会改变随机数/采样器消耗顺序；本次并非与历史 run 逐 step 相同的随机轨迹。固定的是数据、初始化方案、目标权重和更新预算。若最终增益很小，需要同入口整图控制或重复种子才能区分随机波动。

已逐项对比 e110 checkpoint 内保存的真实训练配置与本次生效配置：SAM2、LoRA、学习率、损失定义、归一化和阶段时长一致；差异限于上述裁剪、显式固定名单/尺度、评估开关与产物保存设置。完整差异见 [config_diff_vs_e110.json](../output/20260910_native_scale_probe/config_diff_vs_e110.json)。

保持四个环节：SSL 初始化（复用同一已完成权重）→ 冻结 LoRA 预热双头 60 epoch → 从预热验证损失最优点恢复并联训 LoRA 60 epoch → 训练后独立固定部署比较。

每轮继续计算六张整图验证损失，阶段按 `val_objective_loss` 最小值选优。关闭每轮完整分水岭代理评分及无标签 monitor；未测的部署分数在 checkpoint 中为 null，不保存 proxy-best。训练完成后 `best_direct_dual.pth` 指向 joint 阶段验证损失最优权重。

## 检查与启动

- 新采样 19 项 CPU 合同测试、5 项既有人工 GT 相关测试通过。
- 新训练入口 13 项 CPU 测试通过，包括两阶段损失选优与关闭部署分支。
- 服务器真实数据预检通过：32 张新 GT、SAM2 关闭、未知继续忽略。
- RTX 4090 / `sam2_env` 上完成真实两阶段 smoke：预热 1 epoch + 联训 1 epoch，每阶段 2 个训练样本，强制裁剪；成功恢复预热最优、保存 joint 最优、严格重新加载并前向。实际模型参数量 81,667,394，语义/affinity 输出分别为 1×1×1024×1024 和 1×8×512×512，均为有限值。
- 这些检查验证运行和数据契约；smoke 损失不是竞赛增益。

服务器：`ssh -p 23411 root@connect.bjb1.seetacloud.com`。

项目根：`/root/autodl-tmp/segmentationv2_clean60_20260908_133740`。

正式训练于北京时间 **2026-09-10 12:38:41** 后台启动，PID **9368**。约 12:40 的一次启动确认显示已完成预热 epoch 9，进程存活、GPU 占用 4954 MiB、验证损失与日志正常写入，部署评估确实关闭。此后不安排追踪，等待用户通知训练结束。启动身份见 [launch.json](../output/20260910_native_scale_probe/launch.json)。

在该项目根目录，运行命令为：

```bash
/root/miniconda3/envs/sam2_env/bin/python -u train_direct_semantic_affinity.py \
  --config config/train/direct_gtv2_clean_nativecrop_60x60.yaml
```

这是已启动任务的命令记录，不应在同一输出目录重复启动。退出 SSH 不会中断训练。

所有正式产物位于 `outputs/20260910_clean60_nativecrop/`：

| 文件 | 用途 |
|---|---|
| `train.log` | 训练日志 |
| `launch.json` | PID、启动命令、SSL 权重及实际源码 SHA-256 |
| `source_snapshot.tar.gz` | 本次训练入口、数据、模型、损失与继承配置的源码快照 |
| `source_before/` | 修改前的两份服务器训练源码，已保留 |
| `best_head_warmup_by_val_loss.pth` | 预热阶段按验证损失选优 |
| `best_joint_lora_by_val_loss.pth` | 联训阶段按验证损失选优 |
| `best_direct_dual.pth` | 联训最优权重的统一入口，进入 joint 后才生成 |
| `latest_direct_dual.pth` | 最近完整 epoch 的权重与优化器状态 |
| `checkpoint_epoch_0060.pth`、`checkpoint_epoch_0120.pth` | 两个阶段末的归档 |

逐轮 CSV 和完整配置快照由现有 RunRecorder 写入 `outputs/runs/20260910_123851_direct_dual_ssl_sem_aff/`。真实 GPU smoke 留在 `outputs/20260910_clean60_nativecrop_smoke/`，不参与正式初始化；验证结果已保存为 [gpu_smoke_verification.json](../output/20260910_native_scale_probe/gpu_smoke_verification.json)。

复用 SSL 权重 SHA-256：`e5b837fce1707a682d4a149645dd3a210556f737b8f5778d9d4c8b275a34cb49`。

## 用户通知结束后

读取 joint 验证损失最优 checkpoint 的实际 epoch、完整日志及阶段恢复记录，再做三方固定比较：G4b、原 clean60/e110、新裁剪候选。首先固定 E10a 语义比较 affinity，各自保留独立 encoder/归一化；然后比较新双头完整输出。均保持 gated+mean、high=0.65、seal2、局部重建与受阻分水岭，汇报最终 mIoU、有效匹配、面积项及同一局部的合并/碎裂变化。

历史 G4b 真实 split 仍未知，六图不能冒充完全公平的未见泛化集；阶段选优不看代理总分。本实验不更改新 GT，不替换部署权重，也尚无新增官方成绩。
