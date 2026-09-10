# 新 GT 的实例集合预测实验

日期：2026-09-10。实验分支：`codex/maskset-clean60`。

完成后更新：120轮训练已结束，最终比较发现BF16下掩码投影无梯度的实现缺陷；已定位并作最小
修复，尚无修复后的正式训练结果。完整指标与证据见 [结果分析](MASK_SET_RESULTS_20260910.md)。
以下启动记录保留当时状态。

## 目标与固定项

将监督单位从像素类别、方向连接改为每个预测实例的完整掩码及类别，检验是否能减少当前
affinity → 边界 → 种子 → 分水岭过程的误差放大。它是架构实验，不是原 affinity 的单分支替换。
主线 E10a+G4b / high0.65 / seal2 / 受阻分水岭继续作为已验证部署。

固定人工确认的新 GT、unknown-ignore、`clean60_split.json` 的 26/6 划分、1024 等比 letterbox
与 reflect 图像 padding、`imagenet_v1` 输入归一化和原 clean60 的整图增强。首版不混入原尺寸
裁剪、SAM2 几何伪标签、历史任务头或蒸馏。不同架构的最终输出都按同一实例评分口径比较；
集合模型直接还原掩码，不进入旧分水岭。

## 直接复用 SSL

- 项目相对路径：`outputs/20260908_133740_ssl60/best_lora.pth`。
- 实测 SHA-256：`e5b837fce1707a682d4a149645dd3a210556f737b8f5778d9d4c8b275a34cb49`。
- 48 个 LoRA 层，96 个权重张量严格匹配；rank16、alpha32，qkv/proj，输入 `imagenet_v1`。
- 本轮不重训 SSL。固定同一资产有助于控制变量，不假设不同次随机重建训练完全等价。

## 模型与监督

配置：[mask_set_clean60.yaml](../config/train/mask_set_clean60.yaml)。

这是 **Mask2Former 式适配**：复用 SAM2 Hiera B+ encoder 与 SSL LoRA；随机初始化普通多尺度
FPN、6 层掩码引导 Transformer 和类别/掩码输出。没有复刻官方可变形注意力 pixel decoder，
也没有加载 SAM2 原生 Mask Decoder。每个 query 同时生成两相类别或 no-object，以及一个掩码。
cross-attention 轮流读取 stride32/16/8，mask features 输出 512 网格。

实际总参数 **79,319,427**，其中新 decoder **9,624,067**，满足 `<500M`。256 个候选槽位
是容量，不是目标实例数。训练每图 112–195 个实例，26 图共 4,184；验证共 942 个实例。
512 目标网格保留了全部 5,126 个 GT 实例 ID。

匹配使用类别、BCE、Dice 成本，权重 2/5/5；损失同样采用类别、BCE、Dice。5 个辅助层独立
匹配后取损失均值，再与主层相加。4096 个采样点先保证每个 GT 至少一个有效正点，再从有效域
补充均匀点；属于实例覆盖优先采样，不宣称是全图均匀积分的无偏估计。验证每轮复用固定采样
种子，并为验证 DataLoader 设置独立随机数发生器。

新 GT unknown 平均 3.32%，最高 9.33%。unknown 和 padding 不进入匹配或掩码损失。
未匹配 query 只有预测正区域至少一半落在已知有效域，或掩码完全为空时，才受 no-object
监督；仅在未知区域预测的候选忽略。此判断不反传梯度，是针对部分标注的明确启发式，不能
保证识别所有未标注实例。当前数据适配器不猜测旋转后 padding 与 unknown 的区别。

最终预测两类均保留独立实例；按类别置信度乘掩码概率分配重叠像素，固定类别阈值0.5、掩码
阈值0.5，不合并同类珠光体，不补洞、不使用面积/数量拟合规则。逐 query 先上采样至输入尺寸，
精确裁除 letterbox padding，再恢复原图，避免一次堆放全部原尺寸概率图。输出单通道 uint16
PNG 与 ID→0/1 类别 JSON。

## 训练与选优

复用 SSL → 冻结 encoder/LoRA、预热 decoder 60 epoch → 恢复预热验证损失最优、联训 LoRA
60 epoch → 独立最终评估。每轮64个增强样本；decoder 学习率预热1e-4、联训2e-5，LoRA2e-6。
BF16、AdamW、余弦衰减、梯度裁剪1.0。只使用确定性验证损失选阶段 checkpoint。
新模型与旧模型损失定义不同，绝对数值不能横向比较。

训练前的两图短拟合仅检查学习能力，正式训练重新随机初始化 decoder，不继承 probe 权重。
训练每轮保存 `last_mask_set.pth`，阶段最优分别为 `best_warmup.pth`、`best_joint.pth`，
每10轮留存 epoch 权重；完成标志为 `COMPLETED.json`，异常写入 `FAILED.json`。

## 入口与环境

独立服务器目录：`/root/autodl-tmp/segmentationv2_maskset_20260910`。
服务器：`ssh -p 23411 root@connect.bjb1.seetacloud.com`，环境 `sam2_env` / RTX 4090。
仅共享既有数据、SAM2 底座、第三方源码与 SSL 资产；实验代码和输出独立保存。

```bash
# 数据、划分和已有SSL资产检查
python train_mask_set.py --config config/train/mask_set_clean60.yaml --check

# 两图短拟合，不使用验证图
python train_mask_set.py --config config/train/mask_set_clean60.yaml \
  --overfit-names train_001 train_007 --overfit-steps 300 \
  --output-dir outputs/mask_set_overfit --num-workers 0

# 本次正式两阶段训练的实际命令（该目录已启动，不重复运行）
python train_mask_set.py --config config/train/mask_set_clean60.yaml \
  --output-dir outputs/20260910_182610_maskset

# 训练完成后，严格加载joint损失最优并输出完整原图结果
python tools/evaluate_mask_set.py --config config/train/mask_set_clean60.yaml \
  --checkpoint outputs/20260910_182610_maskset/best_joint.pth \
  --output-dir outputs/20260910_182610_maskset_evaluation --device cuda
```

## 验证与启动记录

本地模型、损失、推理合同测试37项通过，包含冻结、严格加载、未知域梯度、实例容量、两类
独立编号、重叠竞争、奇数padding和uint16落盘回读。服务器已实测SSL哈希、96张量加载和
79.319427M参数。

真实两图预检曾发现初始化问题：将所有 embedding 初始化为std0.02时，300步后类别损失
已很低，但两图均无IoU>=0.5匹配，Dice损失约0.900。只把query_features/query_positions
改为官方默认尺度std1，其余模块和损失不变，重新冷启动同样两张训练图与300步，结果如下。

| 训练图 / 512网格 | GT实例 | std0.02匹配 | std1匹配 | std1匹配实例mIoU |
|---|---:|---:|---:|---:|
| train_001 | 167 | 0 | 142 | 0.84752 |
| train_007 | 195 | 0 | 156 | 0.84038 |

固定两图目标损失22.4226→3.1229，Dice0.9843→0.2364；用时约64秒。重新加载保存的权重，
两种最终logits最大绝对差均为0。修正不改变权重结构，旧初始化的probe checkpoint仍可严格
加载。结果支持初始化对本次查询区分/收敛的重要性，不能外推为验证集增益或否定新GT。
采样仍是4096有效点，没有因为初次失败额外修改loss或叠加任务。

独立的真实两阶段smoke已完成：预热1轮+联训1轮，每轮2个训练样本，验证仍是固定6图。
日志确认恢复预热最优e1，联训e2正常完成，验证损失15.0887→13.1561。这里只证明两阶段
执行、加载和梯度路径可运行，不作为模型质量结果。

本地预检产物：[output/20260910_mask_set_implementation](../output/20260910_mask_set_implementation)。
训练拟合图：[两图300步对照](../output/20260910_mask_set_implementation/queryinit/training_overfit_300steps_gt_comparison.png)。
图中仍有未分配空隙和漏粒；颜色按GT最大重叠对应，分裂碎片可能同色，分区线单独保留。

进一步核对实际阶段权重：预热最优的全部LoRA张量与原SSL最大绝对差为0；联训后96张量发生
更新，最大绝对差4.0047e-6。probe的原尺寸导出已落盘回读：两图均为1936×2584的uint16，
最大ID分别151、170，类别JSON与图中正ID集合完全一致。原尺寸导出同样只属于训练图检查。

**正式训练已于北京时间2026-09-10 18:26:10启动**，PID `20797`。
实际输出目录：`outputs/20260910_182610_maskset`（位于上述独立服务器项目下）。
从同一SSL重新冷启动任务decoder，不继承300步probe或两阶段smoke权重。
交付时的一次启动核验确认进程正常，已完成预热e6，验证损失7.847879，未产生FAILED标志；
GPU占用约4.7GB。该早期loss仅证明正式流程在推进，不用于宣布模型增益。

| 产物 | 作用 |
|---|---|
| `launch.json` | 启动时间、PID、完整命令、SSL与源码状态 |
| `source_snapshot.tar.gz` / `source_manifest.json` | 136个实际源码/配置文件的快照及SHA-256 |
| `config_resolved.yaml` / `split.json` / `sources.json` | 实际参数、固定26/6划分、GT与SSL来源 |
| `run.log` / `metrics.csv` | 训练日志和逐轮确定性验证损失 |
| `best_warmup.pth` / `best_joint.pth` | 两阶段各自的验证损失最优 |
| `COMPLETED.json` / `FAILED.json` | 完成或异常状态；正式最终比较仍独立执行 |

本轮只确认启动，不设置自动跟踪；训练结束后由用户通知再比较。当前没有新模型的正式验证
结果或官方分数，不替换E10a+G4b部署主线。

## 原始设计来源

- [Mask2Former 论文](https://arxiv.org/html/2112.01527v3)
- [官方解码器](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py)
- [官方匹配与损失](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py)
