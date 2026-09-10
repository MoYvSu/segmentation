# 固定主线伪标签与像素归属约束对照

2026-09-10：用户认可主线伪标签预览质量，已授权继续训练。实现、CPU 测试和真实 GPU 小验证完成；**249 张伪标签于 22:03 生成完成，22:41 核验时第一组已完成预热第 47 轮，第二组等待顺序启动。** 此处为带时间的状态快照；不设置持续跟踪，训练结束后由用户通知分析。E10a+G4b 仍为部署主线。

## 实验问题与固定项

用已经优于学生的 E10a+G4b 为合规无标签训练图生成类别感知实例伪标签，扩大掩码集合模型的任务监督。两组对照使用相同的数据、随机种子、初始化、采样量和训练进度，仅改变像素归属损失权重。

| 项目 | 仅伪标签对照 | 伪标签＋归属约束 |
|---|---|---|
| 配置 | `config/train/mask_set_teacher249.yaml` | `config/train/mask_set_teacher249_ownership.yaml` |
| 人工训练/验证 | 同一新 GT，26/6 划分 | 相同 |
| 主线伪标签目标数 | 249 张 | 同一份数据 |
| 每 epoch 人工/伪标签采样 | 64 / 192，固定 1∶3 | 相同 |
| 人工/伪标签梯度损失系数 | 1.0 / 0.5 | 相同 |
| 初始化 | 相同 SSL LoRA＋seed42 随机解码器 | 相同，不从对照末尾续训 |
| 训练 | 冻结 encoder 的预热 60 轮，恢复本组预热最优，再联训 LoRA 60 轮 | 相同 |
| 新归属项权重 | 0 | 0.5，仅最后一层 |
| 阶段选优 | 原人工验证类别＋BCE＋Dice＋辅助层损失 | 同一口径，不将新增归属项计入选优目标 |
| 输入/掩码网格/查询数 | 1024 / 512 / 256 | 相同 |
| 推理与输出 | 原掩码集合部署，uint16 原尺寸实例 PNG＋类别 JSON | 相同 |

每组 120×256=30720 次 batch1 更新。与原 e103 监督实验相比，无标签样本和总更新数都增加，不能把与 e103 的变化全部归因于伪标签；两组新实验之间则对齐预算。

不重训 SSL，不加载主线任务头或共用主线特征，不启用 EMA，不额外添加覆盖填充后处理。教师的 `legacy_none` 与学生的 `imagenet_v1` 各自保持正确前向。

## 伪标签来源、筛选与预览

入口：[generate_mainline_pseudo.py](../tools/generate_mainline_pseudo.py)。配置：[mainline_pseudo_generation249.yaml](../config/train/mainline_pseudo_generation249.yaml)。

- 来源为 `data/unlabeled` 的赛方训练图；排除人工训练与验证图、固定 `unlabeled_holdout_v1.txt` 和同内容重复图。`data/raw` 在当前服务器只有人工标注子集，不能把它误当完整无标签池。
- 采用 249 张这一保守上限，沿用仓库现有少于 250 张的伪实例约定；主线伪标签不是原 SAM2 无类别候选数据层。
- 将各图统一到最长边 1024 后，用灰度 Laplacian 方差粗排清晰度；保留前 70% 候选，用 seed42 打乱。清晰度仅为筛选启发式，不当作分割质量分数。
- 对候选运行完整、冻结的 E10a+G4b 部署：gated affinity、short mean、high=.65、seal2、局部重建、受阻分水岭、概率均值类别投票。
- 接受主线覆盖率至少 95%、实例数 1..256 的图。超过查询容量的图直接跳过，不删掉实例或强行合并来凑数量。该选择可能偏向实例较少的图，后续评价须注意这一范围。
- 直接保存原尺寸实例 PNG、类别映射及主线置信度记录。id=0 保持 unknown；不伪造 LabelMe JSON、`manual_target_v2` 或人工确认记录。
- 标签格式为 `mainline_pseudo_instances_v1`，`label_source=mainline_pseudo`。保存教师 checkpoint SHA256、部署设置、源图 SHA256、候选顺序、接受/排除原因与预览名单。训练记录明确区分人工与伪标签的样本数、原始损失和加权损失。

主线 fused checkpoint SHA256：`53e1b5c6f9bf3973723c7bf4c6ce4ebd5eaeaa45210641df9de8f01e510c790f`。

用户已看过并认可的首批预览：[636](../output/20260910_maskset_mainline_pseudo/previews/train_636.png)、[499](../output/20260910_maskset_mainline_pseudo/previews/train_499.png)、[597](../output/20260910_maskset_mainline_pseudo/previews/train_597.png)、[746](../output/20260910_maskset_mainline_pseudo/previews/train_746.png)。它们是前四张通过预定筛选的图，并非按目检挑出的最佳示例。生成完成时还会自动保存清晰度低/中/高及实例数最多的预览。

训练可以使用与人工实例相同的硬标签损失，但方法如实记为“固定主线伪标签半监督训练”；模型输出不因保存为 PNG 而变成人工 GT。这里未使用软输出或特征蒸馏损失，不据此否认它存在教师向学生传递知识的性质。

## 像素归属项的实际实现

实现位于 [mask_set_loss.py](../utils/mask_set_loss.py)。默认 `ownership_weight=0`，旧配置和 checkpoint 架构保持兼容。

最后一层按原有匈牙利规则配对 GT/可信伪实例与预测 query。在同一批已知有效采样点上，以该像素所属 GT 对应的 query 为目标，对全部 query 的掩码 logits 计算交叉熵。它提高正确实例的相对分数、压低抢同一像素的其他 query；未匹配 query 也参加竞争，但 unknown/padding 不产生这个损失。

`loss_total = loss_supervised + ownership_weight × loss_ownership`。

继续保留原 BCE/Dice 对绝对掩码概率与形状的约束。单独的归属 softmax 对所有 logits 一起减小不敏感：它可以选对相对赢家，却让所有 sigmoid 掩码都低于部署阈值。因此归属项不能替代正覆盖监督，也不构成硬性无重叠保证。未添加“整图必须填满”的奖励，避免将 unknown 硬填，或由一个巨大掩码钻空子。

训练按源数据固定交替，伪标签系数直接乘整个 batch1 损失，不除以该系数；避免归一化把 0.5 抵消。该系数控制梯度贡献，不声称 Adam 参数更新幅度必然减半。

## 验证证据

- CPU：46 项测试通过，覆盖原模型/推理/损失合同，以及新伪标签来源、留出隔离、容量检查、letterbox＋在线增强、固定采样比例、unknown 零梯度、跨 query 配对、重复候选竞争、原损失兼容。
- 权重测试用无动量 SGD 确认伪标签 0.5 系数作用于真实更新；不把这个线性比例外推给 Adam。
- GPU：`sam2_env`、RTX4090、BF16，真实人工与伪标签各一批，分别经过预热和联训。148 个 decoder 参数均建立优化器状态；联训共 244 个参数（含 96 个 LoRA）均有状态，各阶段 2 步。损失有限，阶段最优恢复正常。仅为通路验证，不是正式增益结果。
- 主线输出兼容：在固定 `train_172` 上，生成器的原尺寸实例 PNG 和类别映射与既有 G4b 部署结果逐项完全相同；本图主线输出 339 个实例，超过学生容量，因此只用于兼容验证，没有纳入伪标签训练。
- 未修改模型架构或最终推理；新增损失的收益留待两组正式最终输出比较。

本地证据目录：[output/20260910_maskset_mainline_pseudo](../output/20260910_maskset_mainline_pseudo)。`manifest_at_handoff.json` 与 `state_at_startup.json` 是启动快照，不是生成/训练完成证明。

工作区整理时取回了[完整生成清单](../output/20260910_maskset_mainline_pseudo/manifest_complete.json)：处理 444 张候选后接受 249 张，状态为 `complete`。另存 [2026-09-10 22:41 作业快照](../output/20260910_maskset_mainline_pseudo/state_at_workspace_organization.json)，记录第一组预热第 47 轮与第二组待运行。后续训练完成须核对服务器各组的完成标记。

## 服务器作业与后续取回

项目根目录：`/root/autodl-tmp/segmentationv2_maskset_teacher_20260910`。

2026-09-10 **20:59:52 +08:00**，顺序作业 PID **28771**，入口：

```bash
/root/miniconda3/envs/sam2_env/bin/python -u tools/run_mask_set_teacher_ab.py \
  --output-dir outputs/20260910_teacher249_ab
```

已作为独立后台进程启动，勿重复运行。此前前台生成被切换为可恢复的后台批处理，已生成的样本按原清单复用，未改标签。作业顺序：

1. 完成 `outputs/datasets/mainline_maskset249_v1`；清单只有达到预定数量后才写 `status=complete`。
2. 训练 `outputs/20260910_teacher249_ab/teacher_only`。
3. 训练 `outputs/20260910_teacher249_ab/teacher_ownership`。

查看 `state.json`、`generation.log`、`teacher_only.log`、`teacher_ownership.log`；两组各有 `best_warmup.pth`、`best_joint.pth`、`last_mask_set.pth`、`COMPLETED.json` 或 `FAILED.json`。阶段按人工验证损失选优，不按伪标签损失、归属项或六图代理分选权重。

任一步失败或可用空间少于 2 GiB，顺序作业停止并写失败状态，不自动进入下一步。作业目录保存源码快照、源码哈希清单、启动命令与用户预览认可记录。数据、SSL 与通用基座通过项目内链接复用，不复制大文件。

收到用户训练结束通知后，先核对两组完成标记、best_joint epoch、实际样本计数与更新，再使用 `tools/evaluate_mask_set.py` 做同六图、新 GT、原尺寸完整比较，同时取回固定局部和几何诊断。当前没有正式训练后的分数。

## 磁盘处理

用户授权清理重复文件后，按文件大小、内容摘要和完整 SHA256 确认重复；31 个历史副本改为指向保留文件的相对链接，原读取路径仍有效，不删除不同 epoch 的独有权重。最终替换步骤实际释放 3,053,252,608 字节（约 2.84 GiB）；启动核验时空余约 7.3 GiB、使用率 86%。这些旧路径作为归档读取，新的训练使用独立输出目录。

最初尝试 XFS 写时复制共享数据块，没有观察到本盘可用空间增加，因此没有据逻辑重复大小报告释放成功；最终以上述链接替换前后的实际可用空间为准。具体保留路径和别名见 [disk_cleanup_links.json](../output/20260910_maskset_mainline_pseudo/disk_cleanup_links.json)。

本次仅每 60 轮额外保存一个阶段快照，同时滚动保留 last 和阶段 best，避免重复累积每十轮权重。项目知识中补充了“归属 softmax 不能替代绝对覆盖损失”的已验证限制，没有修改 AGENTS.md。
