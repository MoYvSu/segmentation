# G1 / G2 逐次消融执行方案

状态：v5 定稿；2026-09-12 经 5 轮 Dualog 审阅通过（MCP `review_status.approved=true`）。本文件不表示已启动实验。
执行更新：随后已完成 D 的 68 图候选包与 D/H 六图诊断，见 [G1 部署记录](G1_DEPLOYMENT_20260912.md)；官方删留结论仍未定，C/X 未开始。
排程更新：同日用户进一步要求利用空闲服务器提前启动 C/X，不再以 D 的官方成绩返回为开训前置条件。该授权覆盖下述原排程中的等待条件；训练配方与晋级标准不变，见 [配对训练启动记录](G1_ABLATION_TRAINING_20260912.md)。
目标：在删除当前末端 G4b 的决策基础上，检验还能否删除 G1 或 G2；证据不足或性能下降时保留 G1+G2。

## 1. 问题与固定项

这里只缩短 affinity 训练支线：`G0 → G0-long → G1 → G2`。
G1 是人工实例与在线增强阶段（20 轮）；G2 仍使用人工实例，并加入 SAM2 无类别几何样本及更强负边损失（30 轮）。
删除整个 G2 阶段不能单独证明 SAM2 数据无用；删除 G1 也不等于删除人工标签，因为 G2 仍使用人工实例。
保留的 G0-long 并非空白初始化：服务器 checkpoint 的 config/step 已核实，它从 G0 接续，在 `train_001、train_002` 两张图上以 batch=1 再训练 1600 步，使用默认 affinity loss、不带 G1 在线增强。因此 X 检验的是“保留既有 G0/G0-long 暖启动后，能否省去独立 G1”，不能外推为无需人工训练或从随机头直接训练 G2。

固定 V6 主干/LoRA、E10a 语义、8 通道 affinity 结构、输入归一化 legacy_none、1024 输入/512 输出。
固定最终推理：gated/mean、high=0.65、seal2、low=0.45、重建 8 步、probability_mean、受阻分水岭、uint16/ID≤65535。
不混入 GT 版本更换、SAM2 重新生成、新损失、阈值搜索、SSL 重训或新架构。

## 2. 四个明确角色

| 角色 | 方案 | 产物/成本 | 回答的问题 |
| --- | --- | --- | --- |
| H 历史部署参照 | G1 → G2 | 现成 G2 best，epoch 21；历史 Oracle 最大值选优 | 现有可部署水平，官方成绩记录 84.415 |
| D 删除 G2 | 直接使用 G1 | 现成 G1 best，epoch 18；同样按历史 Oracle 最大值选优；无需训练 | 已训练链能否直接截在 G1 |
| C 保留 G1 的新对照 | G1 best → 重训 G2 | 新 30 轮；按统一损失选优 | 当前代码/选优协议的完整链对照 |
| X 删除 G1 | G0-long latest → 重训 G2 | 新 30 轮；按相同损失选优 | 保留 G0/G0-long 暖启动时，G2 能否接替独立 G1 阶段 |

`G1 → G4b-from-G1` 的既有 epoch 2 保留在历史报告中，不混入这次 G1/G2 归因矩阵。
G4b 原部署权重保留用于回退，默认部署包的实际切换是后续晋级动作。

## 3. 执行顺序与停止条件

1. 先完成 D/H 的完整固定推理比较。五图结果已经存在，不重复大规模诊断；补齐同一入口、同一 68 图、相同输出格式的候选文件与 manifest。历史 H 输出可在确认内容与推理合同后复用。
2. D/H 都补做 §7 同一六图、新 GT 的完整有标签诊断，并做 68 图无标签目检，再由官方口径决定 D 是否可晋级。旧五图只保留为历史参照，不能代替本次六图对照或决定删 G2；当前 G1 的整体面积误差存在跨图抵消，不能把 89.53 代理分当作官方预期。
3. 若 D 已满足本轮删除 G2 的目标，就先保留 G1、删除 G2，结束本轮。若 D 失败且仍要检验删除 G1，再进入 C/X 两臂训练。若官方结果暂缺，保持“未定”，不把未收到成绩当作 D 失败。
4. 两个新训练串行运行，固定同一 GPU 环境、运行代码、数据、划分与 seed=42。只变初始化和输出目录；每臂 30 轮、52 次样本抽取/轮、batch=2，即每臂 780 次计划更新。G1 的额外历史训练量是被消融内容，不给 X 临时补训练轮数。
5. 完成 C/X 后，用同一部署协议比较其损失最优权重；再与 H 比较决定实际晋级。两臂都完整训练 30 轮，best 出现在不同轮次属于共同选优政策下的结果；它回答整个训练方案是否可简化，不单独证明同轮次参数优劣。另保留 epoch 30 作同更新次数、同学习率调度位置的部署诊断；不额外存 epoch 21，不允许事后按代理/官方分另挑 checkpoint 替代预先声明的 loss-best 候选。
6. 只有接近、相互矛盾且会改变决策时才考虑第二个共同种子；不预设多种子矩阵。一次 seed 只能支持当前配方的操作决策，不证明阶段普遍无用。

## 4. 为什么 C 不能省略

历史 G2 按 Oracle 指标选择 best；当前脚本默认按 V6/0.55 的部署代理分选 best，二者都不是用户要求的损失选优。
只训练 X 再与 H 比较，会把“初始化变化”和“选优变化”混在一起。
C/X 共用新的损失选优与运行环境，H 则独立保留作已验证部署参照。
历史 G2 latest(epoch 30) 可以作辅助回看，不冒充当前 C 的重跑结果。

不采用“先复现前三轮训练曲线，通过后用历史 H 完全替代 C”的捷径：少量训练标量相近不证明全程样本、增强、参数或后续更新一致；H 的 epoch 21 本身是从历史验证指标挑出的最优点，把 X 固定在 epoch 21 并不能构成两臂同一损失选优政策下的比较。现有 checkpoint 又没有完整优化器/RNG 状态，先停在第三轮后无法承诺精确续跑。前三轮可以用于启动检查与计时，但不是免掉 C 的充分证据。

### 已核实的推理入口

服务器 `outputs/submission_g2_20260911/submission_manifest.json` 的格式是 `affinity_submission_v1`，对应 `tools/run_affinity_submission.py`；记录的 checkpoint 是 G2 best(epoch 21)、SHA256 `96f9456e4562e9e0a8fa715160582c8f8aee28deceaf32324ac38ae1570ed445`，语义为 E10a epoch 19 的 `challenger_only`，融合 gated/mean，high=0.65。本地保存原 manifest 于 `output/20260912_g1_g2_plan/historical_H_submission_manifest.json`。

D/H/C/X 统一使用此直接入口，显式 `--checkpoint` 指定各自待测 affinity；沿用 `config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml` 所定义的已核验推理合同或等价固定副本。名称中的 G4b 不代表允许沿用其默认最终权重。每次校验 manifest 中实际 checkpoint 的 SHA/epoch、E10a 来源与推理设置；不只核对 `geometry_init_checkpoint`。
其中原 manifest 的 E10a 身份只有路径/epoch，不含 SHA，V6 的 semantic digest 也不能代替 E10a 身份。执行时另记录当前固定 E10a 文件 SHA，并与本轮审阅报告已经核验的 `1380cf14d63fdbc17dadf2ac73dc47d7c33a7e0f5a61e5edb9d7280ec8b80b91`、epoch 19 对照；这保证新实验共用同一当前 E10a，不反向声称历史提交已记录其 SHA。

实际晋级时才制作融合部署包：必须显式改 `affinity_deployment.checkpoint`，因为打包脚本会用它覆盖初始化 decoder；核对包内 affinity 来源 SHA 和 geometry tensor 与已晋级 checkpoint 一致。`semantic_state_digest` 只能核对语义契约，不能证明选对了 affinity。

## 5. 损失选优的最小实现

当前 `train_affinity_geometry_g1.py` 没有验证 affinity loss，也不支持 loss-min 选优，仅写 YAML 不能实现。
执行时新增显式 `affinity_geometry_g1.selection_metric: val_affinity_loss`，原配置默认行为保持兼容，只有 C/X 启用新模式。同步更新 CSV 的 `fields` 与行内容；现有 `deployment_promotion_required` 在这个模式下自然为 true，保留原含义。

- 在固定人工验证集 `172、176、349、623、873、888` 上计算 `val_affinity_loss`。沿用原人工 GT，1024 输入/512 输出、原八通道，无随机增强；本轮不添加 SAM2 伪标签验证集。
- 直接使用未 sigmoid、未经过 Oracle 插值的 affinity logits，调用 `build_affinity_targets_torch` 和 `balanced_affinity_loss`；正/负边与训练同定义，仅两端已覆盖的有效连接监督，unknown/covered–uncovered ignore。沿用 dataset 的 512 网格标签，`uncovered_as_boundary=None`、`edge_weight=None`，不走 Oracle 的 `validation_grid` 插值。
- 损失参数读取解析后的 `cfg.loss`；本次两臂固定负边权重 1.5、困难负边权重 1.0、gamma=2.0、`normalize_edge_weights=true`，不能在新函数另行硬编码；不加入 G4b 缝隙负边或 tail loss。
- 验证固定逐图 FP32 计算损失，再对图平均；这与训练按 batch 池化的聚合方式不同，不直接比较二者绝对值。每个样本总有效边为零时应报错；对固定六图记录逐图、逐通道的正/负有效边数，并检查整套每个通道都有监督，不能让空通道静默缺席或假零损失占优。
- 该 loss 的困难负边权重和归一化分母依赖当前预测，不是朴素 BCE。相同数据、公式及参数下仍能数值比较，但差值不能解释为部署增益或独立的阶段贡献；本轮只用它在各运行内选 checkpoint，跨臂删除/晋级只看同协议最终部署。非有限 loss 应显式失败，不能由 NaN 比较恒为 false 而静默保留 epoch 0。逐轮记录全部验证损失与最优/次优差距，不因差距小而事后改选。
- 验证 loss 与是否运行部署/Oracle诊断解耦，不受 Oracle 重建上限或阈值影响。验证不改变训练随机数序列或模型状态。
- epoch 0 同口径计算并单独记录，用其 loss 初始化 `best_score` 后保存初始 best，后续严格 `<` 才更新；相等保留先出现者，若 best 是 epoch 0，如实标记未超过起点。
- 新目录只保留 loss-best 和最终 epoch 30。沿用 checkpoint 已有 `epoch` 表示实际保存轮次，补充 `best_selection_epoch`、`best_selection_direction=min`、`validation_protocol_version`，保留 `best_selection_metric`；不能让旧 `best_selection_score` 被误读为越大越好。epoch0 初始保存、best 更新、latest 保存三处均传完整元数据。旧文件没有方向字段时沿用历史 max 语义。
- 保持 geometry 权重结构及推理加载兼容；不回写旧 G1/G2/G4b checkpoint，不额外保存各类指标的 best。

实施新验证函数后、C/X 开训前，先对 G0-long latest、G1 best、历史 G2 best/latest 这四个相关权重各计算固定六图 loss（24 次前向，无训练，无新增大图）。检查可重复性、有限值、shape、监督边计数与函数契约；报告其与已有部署诊断是否同向，但不按代理排序反选损失定义，也不把排序不一致自动当作实现错误。若发现实现错误先修复；若数值正确而与最终表现不同，就如实保留局限。
四个权重统一使用 C 的解析后 `cfg.loss`（1.5/1.0/2.0、边权归一化开启、上述两个 None），不能各用自己 checkpoint 里的历史 loss 参数。G1/G0-long 的数值分别是 C/X 的 epoch0 参照；记录二者差值作起点诊断。历史学习率轨迹不同是初始化来源的一部分；G1 best 实际为 epoch18，不能称为 epoch20 末点，也不能仅凭历史末端学习率推断 X 更有训练余量或检验偏向删除。

这个原始 logits 指标绕开了历史 Oracle 重建上限 255→65535 的变化，是沿用训练目标并遵从损失选优偏好的方式，不是已证明的竞赛分替代指标，也不是唯一可能的诊断标量。历史 H best/latest 的新 loss 可作为参照点；C 曲线经过相近数值不证明参数、采样或训练轨迹已精确复现。

## 6. 数据与实现检查

执行代码应从已核验的服务器工作版本建立独立分支/目录；当前本地 `main=2dbc9bf` 与服务器 `f710a9b` 不同。相关 affinity 训练脚本/损失路径已核对为相同 Git 内容，因而可以本地实现后核对补丁再用于服务器；这不等于整个 checkout、数据和运行时环境已完全相同。
沿用同一人工 26/6 划分、同一历史 SAM2 几何数据及 50% 采样概率、52 次抽取/轮、同一在线增强与 G2 学习率 `3e-5 → 5e-6`。
C/X 解析配置与更新次数必须一致，只有初始化和输出目录不同；禁用 feature adapter、高分辨率 refiner、原生裁剪和人工缝隙负边。C 显式覆盖 `deployment_validation.enabled=false`，X 继承；当前 G1 基配置已新增默认开启的部署验证块，历史 G2 没有。关闭后避免每轮额外完整部署推理，记录解析配置的历史差异。
训练脚本结束后没有自动部署钩子；由独立推理步骤在 §7 六图上评估 `outputs/20260912_g1_g2_ablation/g2_loss_control/{best,latest}_affinity.pth` 与 `outputs/20260912_g1_g2_ablation/g2_skip_g1/{best,latest}_affinity.pth`。两个 best 才是预先声明的正式候选，两个 latest(epoch30) 只作同更新次数诊断；诊断排名反转提示选优与终点表现不同，不能单凭此归因于“选优噪声”或排除初始化影响。
开始前检查 checkpoint 严格加载、reference SHA/语义摘要、冻结参数及随机采样设置。最终检查实例 PNG 与原图同尺寸、单通道 uint16、类别 JSON 对应完整、ID≤65535。
逐项比对 C/X 的解析配置和 checkpoint 内配置；两臂的选择键也必须相同，不能将它列为允许差异。对新的 loss 选优配置断言实际 split.val 恰为上述六图，数据文件变化导致集合改变时失败，不能静默换验证集。默认 num_workers=0，保持采样独立 Generator 与相同全局增强 seed；新增验证走 eval/no_grad、无随机增强，验证前后不改 RNG 或训练参数。

最小验证仅覆盖新增 loss 数值/ignore 行为、FP32、空监督/非有限值、min 选优/epoch0/相等不覆盖、旧模式兼容与 checkpoint 可部署性。一次检查验证前后 RNG 状态与训练参数未变；不消耗随机数做比较、不添加运行中逐轮哈希，不新增泛化测试框架或批量截图。

2026-09-12 只读现场核验：G0-long latest(step1600) 已恢复，39 个 geometry tensor，与 G1/G2 共用 V6 reference；六张新 GT 均存在。SAM2 manifest 共 64 条，没有发现六张人工验证图的源图重叠。
当前 G0-long latest 的 SHA256 为 `d5a7579ddd3b6db93c058467b64a1609fde625d43a6c86a3ff1e9518cb01a845`；与恢复档案内同路径文件及 `MANIFEST.sha256` 对应记录一致。历史 G1 checkpoint 只保存初始化路径，没有初始化 SHA，旧路径文件现已不存在。因此这证明了归档恢复一致性，不能夸大为已经从历史 G1 记录中逐字节证明其当时初始化身份。以该归档资产执行时在结果中保留此证据边界；若另有历史矛盾证据，则停止声称严格历史阶段归因，先核对资产。
但恢复的数据目录当前只有 manifest/masks/boundaries/overlays，**缺少 `approval.json`**。直接训练脚本没有技术拦截，`repro/train.py::validate_approval` 则要求该记录；AGENTS.md 的数据约束也要求 SAM2 候选进入监督训练前经过人工确认。这是审核事实的证据缺口，不能因为直接脚本可以运行就绕过。C/X 开始训练前，先从已有档案核对相同 64 源图的实际审核并恢复真实记录；若已有本次会话授权确实覆盖该批数据，使用既有授权而不重复确认；否则明确需要补齐的审核。不能编造 approved/reviewed_by，也不因历史跑过训练就自动认定通过。这不影响 H/D 现成权重推理或当前方案讨论。

## 7. 最终评估与决策

有标签诊断采用固定历史六图的现成新 GT，训练目标仍用旧 GT。这里的六图参与几何验证选优，不称独立测试集；现成五图报告只作历史参照（三张参加过训练）。
记录有效 mIoU、GT 惩罚 mIoU、匹配数、类别组成、铁素体数量与平均面积误差，保留逐图和整体汇总，避免平均偏差抵消。
68 图无标签比较只看覆盖、连接、碎片、类别稳定性与格式；不推断测试 GT、目标实例数或平均面积，不按这些统计量拟合提交。
官方聚合口径未确认时，两种代理汇总都标为诊断；应核验官方规则/程序，不用官方分数反演隐藏 GT。

先核对实际产物，再按以下规则自上而下决策。`best_selection_epoch=0` 时须核对 geometry 与初始化权重相同；不能把它当作新增训练阶段的收益。C@0 与 D 是同一候选，应合并结果、优先按删除 G2 处理，不能据此宣称保留 G2 有收益。X@0 则实际是 G0-long：只报告为更早截断的观察，不据此宣称直接训练 G2 已成功；本轮 G1/G2 训练阶段贡献保持未定，不临时扩充提交矩阵。

决策规则预先固定：

- D 在同一官方评测口径下总分不低于 H，且无部署契约问题：选择较短的 G1 链，删除 G2；分项取舍和已知几何变化单独报告，不把它称为所有方面都更好。
- D 不满足条件，X 的 best 来自实际 G2 更新且在同一官方评测口径下总分不低于 C、H：在保留 G0/G0-long 暖启动、采用同等 30 轮 G2 重训与相同选优政策的前提下删除独立 G1，保留直接训练的 G2。结论同时报告 C/X 的 epoch0 loss 与差值，但不靠这个差值判定部署收益。这里不把六图代理与官方分交叉比较。
- X 不低于 C，但二者均低于 H：保留历史链和 H，先记录新训练/选优政策的不足，不宣称 G1 已可删除。
- C 优于 X 且不低于 H：保留 G1+G2，C 可按同一晋级规则成为新的完整链权重；“未删阶段”并不意味着重训产物不能有价值。
- 两个删除方案均无充分证据或低于基准：保留 G1+G2。本轮接受“保留”作为有效结果。
- 对微小差值不声称统计显著；不临时发明允许降分的容忍带。若官方版本/数据批次无法对齐，保留“不确定”并核对基准。
- 上述所有 D/H、X/C、X/H、C/H 的“不低于”均指同一官方评测口径，不能用六图代理替代或混比。
- 官方结果由用户提交/返回；agent 准备当前步骤候选包与 manifest。必要结果未到时交付产物、不改默认部署、保持现有 G1+G2 选择，结论写“未定”，不空转等待，也不把未定写成消融失败。

## 8. 最小交付与范围

执行阶段需要：两个配对训练配置、最小 loss 选优支持、精简结果报告、checkpoint/source manifest 与必要提交包。训练日志估时以后续实际前几轮为准，不从旧实验拍定完成时间。

拟修改范围（以下均为待实现，不表示文件/新配置键已生效）：

- `train_affinity_geometry_g1.py`：增加原始 logits 的验证损失、显式 `selection_metric: val_affinity_loss`、min 选优及元数据；复用现有损失实现，不改训练 objective。
- `config/train/affinity_geometry_g2_loss_control.yaml`：继承原 G2 配置，指定损失选优、`deployment_validation.enabled=false` 与独立 C 输出；`config/train/affinity_geometry_g2_skip_g1.yaml`：继承 C，仅覆盖 G0-long 初始化与 X 输出。
- `tests/test_affinity_geometry.py`：增加上述必要选择/验证契约覆盖；最终部署沿用现成命令与已有输出检查。
- 本方案/结果文档：记录最终解析配置、产物、决策；不把实验细节写进 AGENTS.md。

如果某臂最终晋级，交付还包括对应的短链复现入口和实际融合部署包，并同步 README/PIPELINE 的推荐入口；不能只换分离配置的 affinity 路径而让默认融合包仍装着旧权重。该切换以晋级结论为条件，讨论阶段不执行。

当前 geometry checkpoint 不包含完整优化器和随机状态，不能承诺从 latest 精确续训；中断时按该限制处理，不为本轮新增完整续训系统。按用户既有偏好，正式启动后确认启动正常并交付日志/目录，不安排后台持续追踪，训练完成由用户通知。
只新增一个实验输出父目录 `outputs/20260912_g1_g2_ablation/`，下设 C/X 子目录，复用原图/标签/基础权重；不复制完整仓库数据，不保留逐轮大图和全量 checkpoint。两臂显式设 `monitor_interval=30`、`unlabeled_monitor_count=0`，保留现有 epoch0/30 的少量人工验证 monitor；本轮不另加图像监控开关。该配置减少 latest 的中间写入频率，不能因此承诺中断可精确续训。数据盘空间充足不作为增加实验数量的理由。
本轮用户请求是讨论执行方案：当前只写方案并做只读核验，不启动上述训练、不提交官方评测、不替换默认部署。

## 9. Dualog

使用用户配置的 Claude Code / `deepseek-flash[1m]`，共 5 轮；会话 `dialog-1789213575008-6c2c2113`。第 5 轮 verdict 为 APPROVE，MCP 状态亦为 approved=true，讨论已结束。这里是方案通过审阅，不是消融成功或训练完成。

共同结论：D/H 优先；C/X 是按同一损失选优政策检验跳过 G1 的必要对照；保留两阶段是有效结果。
主要采纳项：显式关闭继承的部署验证、统一新 loss 的定义与选优方向、检查 epoch0/空监督/非有限值与固定验证集身份、固定实际推理入口和最终加载权重、补齐最终独立评估步骤、核对历史 H manifest、限制监控与存档。
未采纳的扩展：前三轮标量相近替代完整 C、另存 epoch21、额外 BCE 指标或 D@0.72 测试、无漂移证据时强制重跑 H 的全部 68 图。也不把单一 loss 曲线相交、历史学习率差异或 best/latest 排名反转当作因果证明。C 的起点是历史 Oracle-best、X 是固定末端 G0-long，这一事实已纳入方案身份；它不保证部署比较一定偏向任一方。

完整对话：`output/20260912_g1_g2_plan/dualog_conversation.jsonl`。
历史 H 原始提交记录：`output/20260912_g1_g2_plan/historical_H_submission_manifest.json`。
