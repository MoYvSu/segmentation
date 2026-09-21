# 交接：从已验证主线逐次消融，缩短训练链

更新时间：2026-09-11。用户因本周 Codex 额度将尽而要求归档。本文件记录当前证据和下一步，
不是已完成新实验的报告。**尚未启动本轮消融训练，也未核查服务器当前资产与资源状态。**

## 1. 用户最新意图与约定

- 停止优先手工设计全新架构，改从已验证主线逐次删除训练环节，找到较短、稳定的训练链。
- 用户希望训练阶段尽量精简，最终保持约四阶段是方向，不是强行凑数的硬指标。
- 用户认可补缝新GT质量；失败不能直接归咎于新GT。但本轮历史主线消融应先固定原训练GT，
  不同时换GT、初始化、损失、融合和阈值，以免无法判断删减的影响。
- 阶段选优倾向按损失；部署晋级仍看完整最终输出。历史复现要核对历史选优规则，
  不能把历史规则改成损失选优后，仍称为精确历史复现。
- 长训练如果之后启动，用户会在结束后通知，不需要持续监控或定时追踪。
- 用户重视磁盘空间和额度：避免大规模扫参、重复训练、无必要测试与大量临时产物。
- 用户此前要求自己配置 Git token、自己运行远程提交命令；不要代为写入或索取聊天中的 token。

## 2. 工作区与未提交改动

仓库：`D:\MY_PROGRAMME\segmentationv2`。

交接时实际检查结果：

- 分支：`codex/mainline-cross-head`。
- HEAD：`67360e4 test: 完成GT affinity完整部署检验与融合定位`。
- 未提交修改共4份文档：
  - `README.md`
  - `docs/PIPELINE.md`
  - `docs/EXPERIMENT_INDEX.md`
  - `docs/AFFINITY_GT_DEPLOYMENT_ORACLE_20260911.md`
- 用户未跟踪目录/文件：`.claude/`、`CLAUDE.md`，保留。
- 本交接文件为新增，尚未提交。没有修改生产Python代码、配置、模型或训练数据。
- 用户曾提供未跟踪 `tmp_diagnostics_20260911.md`，本代理没有修改或删除它；
  最新 `git status --short` 已不再列出该文件。不要假定它仍可读取，重要更正已写入正式报告。

4份文档的修改内容：明确真实mean/top2对照已完成，补CPU直线探针复现代码，区分历史源权重
元数据与本次哈希、字节验证与指标相同，更正临时分析稿的统计和因果表述，补逐实例共享状态。
`git diff --check`通过；文档中的CPU探针实际执行通过。未提交、未推送。

## 3. 当前主线与资产

部署仍为：**E10a语义 + G4b八方向affinity + gated/mean + high=0.65 + seal2 + 局部重建 +
受阻分水岭**。历史官方成绩：实例mIoU 0.8381、铁素体面积项0.8408、总分83.94。
这是历史官方记录，不是本轮新提交成绩。

固定推理要点：

- 输入1024等比letterbox、reflect padding，G4b affinity输出512网格。
- 主线归一化 `legacy_none`；clean60/e110使用 `imagenet_v1`。
- 混合不同模型时必须分别运行各自encoder/LoRA，不能直接共用特征。
- 先在512网格融合方向连接，再恢复原尺寸。
- high=0.65、marker low=0.45/重建8步、seal=2、bridge=1、dilate=1、min_area=50。
- 类别采用probability_mean、阈值0.5、不腐蚀；输出原尺寸单通道uint16，ID<=65535。

关键入口：

- `config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml`
- `tools/run_fused_affinity_submission.py`
- `utils/affinity_deployment.py`
- `models/fused_deployment.py`

最近使用过的服务器（**本次交接未重新连接**）：

```text
ssh -p 23411 root@connect.bjb1.seetacloud.com
Python: /root/miniconda3/envs/sam2_env/bin/python
分析项目: /root/autodl-tmp/segmentationv2_cross_head_20260911_1642
主线部署包: /root/autodl-tmp/segmentationv2_pseudo_20260908/outputs/deployment/e10a_g4b_fused.pth
新GT短链项目: /root/autodl-tmp/segmentationv2_clean60_20260908_133740
短链权重: outputs/20260908_133740_clean60/best_direct_dual.pth
```

部署包SHA256：`53e1b5c6f9bf3973723c7bf4c6ce4ebd5eaeaa45210641df9de8f01e510c790f`。
V6历史锚点SHA256：`c4c9827a18ecdda105056d502e5f93a92fe7fc93ada24d63886b20726c7ec557`。
包内构建元数据记录E10a epoch19、G4b epoch20；原始源权重摘要见GT实验报告。
clean60 checkpoint此前确认epoch110、joint_lora阶段。路径、文件存在性和磁盘空闲需重新检查，
尤其G0/G1/G2历史资产可能受之前清理影响，不能假设仍在。

## 4. 已完成的GT affinity检验与正确解释

完整报告：`docs/AFFINITY_GT_DEPLOYMENT_ORACLE_20260911.md`。

五图：172、351、686、872、873，共790个GT实例。GT来自
`outputs/experiments/seeded_label_completion_o1/radius_8/derived_maps`，已知像素占96.04%。
这不是全训练链共同未见的独立测试集；历史训练曝光仍有不确定性。

| 项目 | E10a+G4b | E10a+GT affinity/mean | E10a+GT affinity/top2 |
| --- | ---: | ---: | ---: |
| 有效匹配 | 715 | 1 | 724 |
| 有效匹配mIoU | 0.84750 | 0.79524（仅1对） | 0.91857 |
| 漏检计零IoU | 0.76704 | 0.00101 | 0.84183 |
| 面积相对误差 | 35.39% | 2911.28% | 1.38% |
| 完整代理分 | 74.679 | 39.762 | 95.239 |
| 起始区域marker数 | 1190 | 67 | 843 |
| 共用主导marker的GT数 | 65 | 789 | 105 |

实现：GT同实例连接=1、异实例=0，仅替换两端有GT的连接；unknown/padding/越界保留原G4b。
使用±inf logits，sigmoid后精确0/1；无穷logits不经过插值。目标与实际训练Torch生成器逐值一致。
正常E10a不变，GT不参与前景、marker或类别投票。主试验五图基线PNG/JSON与旧输出实际做过
read_bytes断言，状态记录为通过；完整PNG未入Git，重新逐字节核对需要原始运行目录。

结论：**理想方向连接与当前融合/阈值确有失配；不是证明所有下游都正确，也不是模型泛化成绩。**
直线探针：gated/mean在垂直、对角界面峰值0.5，水平0.642858，均低于0.65；top2约0.785714。
0.5绝不是所有方向和形状的普遍上限。

主线为何仍好：G4b实际输出多个方向共同形成较宽强响应，更适合当前均值/高阈值组合。
历史初始化、旧缝隙负连接监督可能参与形成这种形态，但贡献尚未分别隔离；不能说模型已被证明
主动学会补偿分水岭。理想GT来自新补缝定义，也不等同于G4b当时所有监督目标。

## 5. 不要重复的真实模型对照与分析错误

`docs/AFFINITY_CLEAN60_DIAGNOSIS_20260910.md`及`output/20260910_111210_diag/summary.json`
已有固定E10a、正确独立归一化/前向、固定后处理的六图mean/top2对照（含889，942 GT）：

| 模型 | 匹配mean→top2 | 有效mIoU | 面积误差 |
| --- | ---: | ---: | ---: |
| G4b | 854→856 | .8482→.8412 | 34.44%→36.04% |
| e110 | 792→806 | .8408→.8335 | 18.48%→22.09% |

不支持直接改用top2；也不能反推均值融合完全没损失信息。不要把这项实验再次写成未执行，
不要把六图绝对数字与五图直接比较。当前下一方向已转为训练阶段消融。

已核验的临时分析稿更正：

- 主线−cross逐图匹配差`[3,1,1,2,8]`，穷举3125个配对重采样的总差95%百分位区间为
  `[6,28]`，不是`[-54,80]`。五图方向一致不证明总体优势；配对t检验p约0.083。
- 原0.4/0.45的匹配反例均未过0.5，无效。有效反例和口径说明已在正式报告中；
  当前Hungarian最大总IoU后过滤，与先最大有效匹配数可能不同，但赛方实现未确认，未改评分器。
- μSAM Oracle用了GT语义/完美支持域及不同GT处理，不可据跨实验valid mIoU相近宣布边界到上界。
- D1–D5未转发参数是真实接口风险，但当前配置恰与硬编码相同，不能解释已有成绩或GT实验失败。
- CLI脚本无人import不等于死代码；尺度统计仍未复现，不据此清理或修改训练。
- 相对“G4b+mean”，“GT affinity+top2”新增共享GT86个、解除46个（逐图已复算）；
  同时改了输入和融合，不能将其单独归因top2，更不等同于最终合并数。

## 6. 最新消融方向：先建立可复现对照，再逐段删减

真实训练链（详见`repro/README.md`、`repro/train.py`）：

```text
LabelMe -> purified GT
1000张无标签 -> SSL LoRA -> Stage1 -> joint-v3 -> V6
                             |          |       |-- E10a
                         边界缓存   语义边界缓存 |-- G0 -> G0-long -> G1 -> G2 -> G4b
                                                        已审核SAM2几何 -> G2/G4b
```

**注意：配置继承G3/G4不等于权重必须先训练G3/G4。G4b实际从G2初始化。**
`config/train/affinity_geometry_g4_manual_gap.yaml`显式指向G2，G4b继承该路径。
V6之后E10a与affinity分支可分别训练。

已确认复现入口可列出14个阶段（含缓存/审核等非训练步骤）：

```bash
python repro/train.py check
python repro/train.py list
python repro/train.py run --dry-run
```

本次只实际执行了`list`并读代码/配置；**没有执行完整check、没有完整冷启动训练成功证据**。
`config/reproduction_mainline_8394_from_v6.yaml`是另一个从固定V6开始的阶段清单；
不要把它误认成已被`repro/train.py`读取的编排配置，后者阶段定义在Python的`STAGES`中。

建议下一次从这里开始：

1. 检查checkout和未提交文档，连接服务器核实G2、G1、G0-long、V6、E10a与旧数据/缓存的存留。
   列清“可直接复用”“需要重建”“历史不可确认”，不要凭旧路径直接启动。
2. 固定历史V6、E10a、原训练GT和部署mean/high=.65；先核验并复跑末端G2→G4b建立当前代码基线。
   若历史G2可用，可直接把G2接同一E10a/完整推理，低成本检验最后G4b阶段的必要性。
3. 再测试删G2：从同一G1起点重训G4b，对照G1→G2→G4b。删除阶段后必须重训依赖该阶段的下游，
   不能使用已受该阶段训练影响的旧下游权重冒充消融。
4. 然后考虑G0-long、G0热身等是否可删/合并；每次只动一个环节。
5. 最后才动V6之前的共享特征链。LoRA/encoder变了，E10a/G4b依赖也变了，不默认兼容复用。

评价目标是较短链条下完整分割质量是否保持；训练阶段数之外也记录更新步数、GPU时间。
若删阶段后退步，不立即断定该阶段特有监督不可替代，可能只是总训练量减少；仅在需要区分时
补训练预算相当的对照，不预先展开大矩阵。小差距需考虑随机波动，不凭单次五图排序定性。

## 7. “精确复现每一步”的限制必须明确

- 历史权重推理一致、配方/性能复现、逐梯度/权重逐位相同是三个不同目标。
- `repro/README.md`已明确：CUDA/库版本/数据加载并行可能使权重不逐字节一致。
- Stage1缺少独立run_info，精确历史运行提交未确认；joint-v3/V6有历史快照。
- G0/G1保存函数保留权重、配置、阶段信息，但没有完整优化器/随机状态；可以当阶段起点，
  不保证任意中断step精确续演。阶段间原本重建优化器与中途恢复是不同问题。
- 当前训练脚本可能已演进。当前G1配置开启deployment_validation，`best`按部署代理分选，
  关闭时则按Oracle分选；历史G1/G2实际采用哪一版选优逻辑要核对，不能用当前配置名推断。
- 原始数据/分组、伪标签缓存、SAM2候选、软件版本、随机种子与选优规则都影响结果。
  不是只有复现代码bug才会造成差异，也不能把任何差异都归咎于随机性而跳过代码核查。
- 历史官方83.94不保证重训后精确得到；验收应以锁定协议下的训练/完整输出证据为准。

## 8. Claude Code / Dualog协作已接通

用户使用Claude Code前端连接自己的DeepSeek API，实际模型`deepseek-flash[1m]`、effort `max`。
不是Anthropic Claude模型的独立审查。最初失败原因是PowerShell临时`$env:`没有传给已运行的Codex。
用户随后写入`C:\Users\danmo\.claude\settings.json`的`env`，持久配置实际调用已成功。
只检查设置是否存在与非敏感字段，不打印认证信息，不将用户级认证配置放进Git。

调用`dualog-review-code`已完成两轮：首轮要求文档修正，第二轮正式返回`approved: true / APPROVE`。
会话ID：`review-1789132098791-f9938a0d`，已end_dialog，没有待处理审查进程。
记录：`C:\Users\danmo\.dualog\sessions\review-1789132098791-f9938a0d\conversation.jsonl`。
通过后按其非阻断建议补充了已核验的86/46集合差、五图/六图区分和理想输入检验适用范围。

审查模型也产生过过强结论，部分在讨论中撤回。以正式报告的证据边界为准，不机械执行原始
审查记录里“不是主要瓶颈”“只能来自某种bootstrap”“重跑mean/top2”等多余或矛盾的外推。

## 9. 后续代理最短阅读清单

1. `AGENTS.md`与本文件。
2. `repro/README.md`、`repro/train.py`与上述G0/G1/G2/G4b配置。
3. `docs/AFFINITY_GT_DEPLOYMENT_ORACLE_20260911.md`。
4. 按需要读`docs/AFFINITY_CLEAN60_DIAGNOSIS_20260910.md`，避免重复已做的融合对照。

下一步应优先补齐末端消融所需资产与历史配方核验，不再从“换一个新架构”或“大规模扫参”开始。
