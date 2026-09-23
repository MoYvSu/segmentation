# Candidate observations

This file holds potentially reusable findings that still need validation or
scope clarification. It is not a chronological task log.

## 20260923-diffusion-training-sampling-gap

- status: observation
- last_verified: 2026-09-23
- scope: 条件扩散修复的训练内过拟合检查与长测决策
- finding: 随机时间单次重建误差与实际完整采样误差会不同步；早期多步退化不足以否定可学习性。
  D1只延长800→3200次，完整采样梯度误差由输入的约1.175倍降到0.637倍，三种子均过原门槛；
  但末点首步仍好于16步，不能把“扩散模型可拟合”解释为“多步迭代已经优于回归”。
  后续1000图60轮中16步梯度比值又为1.076，首步0.897；固定权重与初始噪声、只关闭后续加噪声
  可降至0.924。四图过拟合通过不能外推全量配方，反复加噪声是此模型可观测的退化因素。
  D2同预算20轮只将kappa从0.1降到0.03，训练误差更低，实际首步与16步却近乎原图。
  因此降低扩散噪声并非单调改善：既要检查多步颗粒，也要检查是否丢失从模糊输入起步的修复能力。
- evidence: `docs/RGB_DIFFUSION_SHORT_DECISION_20260923.md`；同4训练配对、权重/优化器/随机流连续，
  明确区分随机t训练、首步估计、完整采样与官方分割成绩；过程图独立保存。
  后续证据见`docs/RGB_DIFFUSION_ALL60_ANALYSIS_20260923.md`。v6固定合成配对误差低于v4，
  用户回报复赛总分却低0.945，说明不能用修复误差排序替代分割晋级。
  `docs/RGB_DIFFUSION_D2_ANALYSIS_20260923.md`：三采样seed与FP32复核一致；移除训练中间状态的
  清晰线索后输出接近恒等，支持状态依赖假说，未证明唯一原因。
- reuse_hypothesis: 扩散早期判定应同时看训练/采样误差及充分拟合趋势；通过工程门槛后，
  用同checkpoint的首步和完整输出检查多步贡献，保持正式模型晋级按真实部署/黑盒证据。
- verification_gap: 全量训练仍是单训练seed；定量修复对照仅4个训练配对，真实图目检4张，
  D1/D2无黑盒成绩，D2只到20轮。未证明全量绝对收敛或多步退化的唯一原因；
  增加推理起点训练比例的有效性尚待验证；v6回分缺平台包哈希回执。
- limits: 三个采样种子不是三次独立训练；本条不调整门槛、不允许从过拟合或目检宣称测试补边正确。

## 20260912-validation-area-ranking-transfer

- status: observation
- last_verified: 2026-09-19
- scope: 小规模有标签验证代理对官方实例/平均面积成绩的外推
- finding: 代理在一个候选对上排序正确，不足以建立稳定排序能力；面积项在六图逐图同向改善，也可能无法推广到官方测试。
  合并平均面积与仅有效匹配的IoU还可能掩盖逐图退步和漏配，应同时查看逐图结果与GT惩罚指标。
- evidence: `docs/G1_ABLATION_RESULTS_20260912.md` 后续黑盒节。按会话提交顺序归属D/G1，六图D面积误差整体及逐图优于H/G2，用户回报官方面积项却为D 0.6860、H历史0.8460，总分方向也反转。
  `docs/TOP2_G0_ABLATION_20260914.md` 的X/Y固定mean对照：Y合并代理77.97→79.67，逐图平均78.27→69.01，有效匹配720→411；此例尚无Y官方成绩。
  同文2026-09-14用户回分补充：Z六图面积误差逐图全部改善，官方面积项却由X的0.8464降至0.8136；X-top2六图合并面积误差变差，官方面积项反而升至0.8526、总分比X-mean高0.195。
  `docs/G2_LONG_OLDGT_20260914.md`的2026-09-15回分：L六图面积误差五差一好、逐图平均代理比X-top2低2.1677；用户回报官方面积项0.8538高于X的0.8526，总分84.770高于84.705。
  `docs/G2_LONG_NEWGT_20260916.md`的2026-09-16回分（按上下文归属N）：N六图面积误差34.33%→33.30%，用户回报官方面积项却由L的0.8538降至0.8476，总分84.770→84.455。
  `docs/SEMANTIC_NEWGT_20260916.md`的2026-09-17回分：S内部语义7图面积误差23.29%→27.54%、历史6图34.33%→40.51%，官方面积项却由L 0.8538升至0.8714，总分85.535；官方收益稳定性和来源仍未知。
  `docs/ALIGN_NOSAM2_EXPERIMENT_20260917.md`：去SAM2后6/6图面积误差改善，但6/6实例mIoU、GT惩罚mIoU和最终类别像素mIoU下降；总匹配779→690，合并代理74.885→78.095。用户明确回报V-noSAM2黑盒0.8362/0.8181，总分82.715，比V低1.930；官方面积项反而由0.8499降至0.8181，决定保留SAM2监督。
  `docs/STAGE1_FINAL_TASKS_20260919.md`：Stage1最终任务替代的两组内部面积误差均改善，逐图为5/7、4/6改善，但用户回报官方面积0.8569→0.7699、总分84.925→80.735；7图匹配1000→971，6图又全部参与过联合适配训练。再次反对以内部面积方向预测官方晋级。
- reuse_hypothesis: 保留代理诊断与固定loss选优，阶段删留仍按实际最终部署/官方证据；不要用代理优势幅度或同类候选实例总数接近替代正式比较。
- verification_gap: D/N身份由会话上下文解读，Z/X-top2按用户明确组名归属，均缺平台包哈希回执；反转的具体数据/评价/泛化原因未分离。Y例只验证内部聚合口径可能给出相反排序，不外推官方成绩。
- limits: 不否定新GT质量或所有本地指标，也不由绝对面积项推断隐藏GT、过/欠分割方向或调整目标实例数。

## 20260919-instance-count-shift-includes-class-voting

- status: observation
- last_verified: 2026-09-19
- scope: 铁素体计数/平均面积变化的语义与几何归因
- finding: 铁素体像素总量接近而实例数减少，不足以将变化全归为合并；小实例改判为珠光体也会显著改变均面积分母，却只改变少量像素。
- evidence: `docs/STAGE1_FINAL_TASKS_20260919.md`回分后诊断。旧新68图按形状IoU≥0.5对应4966对，F→P 308、P→F 17，净类别计数差−291；未对应部分净F差−278，总F差−569。F→P旧预测面积中位1575像素，F→F为14950.5像素，前者仅占旧F总像素0.90%。
- reuse_hypothesis: 面积退化时同时检查类别组成与预测划分；以各自encoder/LoRA独立前向的输出交换进一步定位，不把旧头直接挂到新共享特征。
- verification_gap: 无测试GT，无法判定改判正确性；预测间对应仅为计数分解，不能按−291/−278给语义/几何作因果分摊，也不能量化官方面积损失来源。
- limits: 不据此恢复错误类别、强制铁素体数量或用黑盒分数反推目标平均面积；一次替代配方失败不证明旧joint-v3不可替代。

## 20260914-stage-count-is-not-training-budget

- status: observation
- last_verified: 2026-09-15
- scope: 删除几何暖启动阶段的归因与训练流程简化
- finding: 固定后续微调配方再删除前置训练，检验的是整段训练过程是否可以直接省略；它同时减少更新/样本曝光并改变初始化，不能证明前置样本或独立阶段不可替代。简化独立运行环节可以允许总训练量增加。
- evidence: `docs/TOP2_G0_ABLATION_20260914.md` 的预算补充；G0/G0-long共2000次batch1更新、LR1e-4/5e-5，G2共780次batch2更新、LR3e-5→5e-6。X/Y总更新2780/780，总样本抽取3560/1560；Y上采样/输出层随机初始化，Z则已有400步G0。
  `docs/G2_LONG_OLDGT_20260914.md`：同原GT、无两图暖启动的L改为120轮和分组LR后，loss-best为0.20555，接近X的0.20188；固定top2有效匹配X/L/Y为733/777/413。L内部仍有更多未匹配预测与更大六图面积误差，但2026-09-15用户回报官方0.8416/0.8538/84.770，略高于X-top2的84.705，支持本配方下移除独立两图预训练环节。
- reuse_hypothesis: 区分直接删阶段、预算重分配和监督替换；同时核对更新数、样本抽取、来源比例与学习率过程，为未训练输出层设计独立于旧微调配方的训练预算。
- verification_gap: L已在本次用户回报的官方测量保持成绩，但未取得平台包哈希回执、没有多随机种子重复，也未分离初始LR、分组LR、衰减时程与预算各自贡献；不证明全局收敛或普遍等效，不能把优化不足指定为此前退步的唯一原因。
- limits: 不宣称提高LR/延长训练必然有效，不以学习率之和替代优化量，也不把同样更新数或样本抽取量称为所有条件等价。

## 20260912-scalar-reproduction-is-not-trajectory-identity

- status: observation
- last_verified: 2026-09-12
- scope: 历史多阶段训练重跑、选优政策与初始化消融的归因
- finding: 对应轮次验证损失极接近，仍不足以证明权重或随机训练轨迹相同；不能据此把部署差异唯一归因于checkpoint选优。
- evidence: `docs/G1_ABLATION_RESULTS_20260912.md`；`output/20260912_g1_g2_ablation/historical_latest_tensor_comparison.json`。C与历史G2末轮loss差约5.5e-5，但39/39 geometry张量不同，最大绝对差0.00115811。
- reuse_hypothesis: 将“重跑表现接近”“参数相同”“历史逐步复现”分开记录，避免把少量相近标量误当作消除混杂因素的证据。
- verification_gap: 尚未分离参数差异与选优轮次差异各自的部署影响，也没有不同seed的方差估计。
- limits: 本例不说明重跑失败或差异具有竞赛意义；低成本CPU张量核验适用于出现严格复现主张时，不要求每次例行评估重复比对。

## 20260916-completed-target-edge-cohorts

- status: observation
- last_verified: 2026-09-16
- scope: 补缝GT替换后的监督变化诊断
- finding: 相同旧GT验证loss接近，仍可能伴随新增监督位置与旧已标位置的不同取舍。保持预定选优口径，同时在同一验证图上分别诊断原已标边、新增正边和新增负边，比仅看整体affinity均值更有解释力。
- evidence: `docs/G2_LONG_NEWGT_20260916.md`；N/L旧验证loss 0.20703/0.20555，补缝目标loss 0.24613/0.25291；N新增短程负边误连率降低，但旧已标跨界平均affinity升高、新增正边平均affinity略降。固定部署有效匹配777→788，不能据任一边子集指定唯一机制。
- reuse_hypothesis: 标签补齐或ignore域改变时，先验证旧有效边是新有效边的子集且目标不变，再比较各子集；训练loss更换目标后不可直接视作同一指标。
- verification_gap: 本次单seed、小验证集；用户随后回报黑盒0.8415/0.8476（按上下文归属N），未取得平台包哈希回执。分层诊断不是因果分解或边界召回率，也未证明官方增益。
- limits: 不以这次结果改为新GT验证loss选轮次，不推断补缝GT普遍优于旧GT。

## 20260916-checkpoint-carrier-versus-stage-contribution

- status: observation
- last_verified: 2026-09-16
- scope: V6边界阶段消融、joint-v3来源及共享encoder/语义兼容性
- finding: 下游引用某阶段checkpoint，不等于依赖该阶段训练过的所有内容；应逐模块核对实际差异。V6与其joint-v3前身的LoRA和语义权重完全相同，V6特有的边界FPN可以作为独立初始化变量检验。任务头随机重置也不等于上游知识清空，还应核对共享LoRA与固定教师这两条传递路径。
- evidence: `docs/G2_SKIP_V6_20260916.md`及受忽略的`output/20260916_g2_skip_v6/training_preflight.json`、`lora_config_audit.json`。joint-v3 e88与V6 e9的96个LoRA、36个语义张量相同，LoRA rank/alpha/target_layers相同，28个边界FPN张量不同；保留L几何与E10a时，更换参考来源后的既定验证图最终实例/类别完全一致。
  同目录`joint_v3_role_audit.json`确认Stage1→joint-v3的96个LoRA张量全部改变，joint-v3→E10a全部相同；joint-v3保存配方包含Stage1缓存边界一致性，E10a冷启动则另启固定教师语义蒸馏。常规`unsup_seg_weight=0`不能单独证明没有无标签监督。
- reuse_hypothesis: 跳过冻结阶段前，先区分checkpoint作为权重载体与该阶段实际贡献，复用完全相同的模块，避免不必要地回退整个encoder或重训语义分支。
- verification_gap: V已完成同预算120轮、完整部署及用户回报的官方对照，工程删留结论见`docs/G2_SKIP_V6_20260916.md`；仍缺重复训练及完整冷启动复现。接受删去独立边界阶段，不等于所有读取V6文件的载体引用已被替换，也不证明统计等效。
- limits: 不说明V6边界预训练无用，也不把语义/LoRA相同外推到其他checkpoint或Stage1/SSL特征。

## 20260916-semantic-gt-pairing-and-amp-budget

- status: observation
- last_verified: 2026-09-16
- scope: E10a冷启动、GT替换与训练预算的成对解释
- finding: GT派生的数据增强也会使换标签变成换输入；旧E10a暗边增强依赖boundary target，隔离GT比较时需固定其输入或两组共同关闭。AMP的scaler.step调用次数也不等于实际更新次数；溢出会跳步，应另记attempted/applied/skipped。
- evidence: `docs/SEMANTIC_NEWGT_20260916.md`、`tests/test_semantic_completed_gt.py`。两组首轮真实GPU短跑默认scale65536均出现inf梯度，优化器state为空且decoder完全相同，但旧日志计数为1；受忽略的`output/20260916_semantic_newgt/`保存预检证据。
- reuse_hypothesis: 解释监督或预算消融前，同时核对GT依赖增强、具体划分与实际更新，避免把相同seed或名义轮数当作充分的实验隔离证据。
- verification_gap: 两组20轮各1240次实际更新、零AMP跳步及冻结契约已核验；S官方总分比L高0.765，但S-old尚无回分，不能把全部增益单独归因新GT。不把单步溢出解释为历史E10a训练失败。
- limits: 只针对当前E10a实现，不外推其他训练器；语义25/7与affinity26/6不同，六图代理也不是语义未见样本。

## 20260917-online-semantic-coordinate-alignment

- status: observation
- last_verified: 2026-09-19
- scope: `UnlabeledDataset`到`compute_unsupervised_loss`的在线语义教师监督
- finding: 教师弱视图保持原位、学生强视图独立翻转/旋转时，直接逐像素一致性会惩罚不同位置。离线边界缓存有显式同变换，不代表在线语义目标也已对齐；固定教师换成EMA不会自动修复。
- evidence: `docs/TRAINING_ROADMAP_20260917.md`、`docs/ALIGN_NOSAM2_EXPERIMENT_20260917.md`。本地与服务器S运行目录的dataset/trainer/loss源码按LF内容相同；实际损失CPU合成复现：同视图0、仅学生翻转0.4646745、共享翻转0。修复开关已覆盖翻转/旋转、缓存仅变换一次、学生随机行为不变；服务器短跑3次有效更新、0跳步，LoRA及非语义模块逐张量不变。
- independent_check: `docs/UNIFIED_ALIGNMENT_AUDIT_20260919.md`重新读取两次实际训练源码、配置与日志；4张真实无标签图的冻结checkpoint交叉诊断，S在旧/新目标下MSE为0.249557/0.008784，S-align为0.253180/0.005957。共享变换输入与变换教师预测仍有1.68%～3.13%原图域类别分歧，不能宣称完全等价。相同输入也不代表固定教师蒸馏为零：师生权重不同、目标锐化都可产生非零梯度，不应凭外观增强未触发就判定监督失效。
- reuse_hypothesis: 增加任务一致性前，用非对称输入验证像素坐标及有效域；方向affinity还需验证通道随几何变换的对应关系。优先共享几何、只差外观增强，避免错位污染监督。
- verification_gap: 20轮1240次有效更新的正式对照及固定L部署已完成；一致性损失末5轮下降96.7%、置信覆盖基本不变，原始语义略降、最终实例近似持平。用户回报S-align黑盒0.8407/0.8620/85.135，相对S总分−0.400、mIoU+0.0014；接受正确实现继续研究，但不把面积变化认定为噪声或等效证明，也不能替代与“关闭该一致性项”的无标签收益对照。
- limits: 不作废已得到的S官方分数，不把此缺陷归于新GT，不外推已对齐的离线边界缓存或整个joint-v3无标签训练无效。

Before adding an entry, search active and retired references for the same
lesson. Record only candidates that pass the admission filter in `SKILL.md`.

Use this shape:

```markdown
## YYYYMMDD-topic

- status: observation
- last_verified: YYYY-MM-DD
- scope: affected workflow, module, environment, or experiment family
- finding: concise statement
- evidence: current reproducible evidence or pointer
- reuse_hypothesis: why this may matter in a future task
- verification_gap: what remains uncertain
- limits: known exceptions or counterevidence
```

## 20260831-fused-deployment-contract

- status: observation
- last_verified: 2026-08-31
- scope: 83.94 E10a + G4b fused checkpoint 推理、提交校验与打包
- finding: 正式提交推理应读取 fused checkpoint 内冻结的 fusion/inference 参数；外部 YAML
  只负责路径和模型构建环境。源三 checkpoint 入口保留作实验与权重对照，不应作为正式提交入口。
- evidence: `models/fused_deployment.py::frozen_deployment_contract`、
  `tools/run_fused_affinity_submission.py` 与 `tests/test_submission_package.py`；本地已验证契约复制、
  格式校验及确定性 ZIP，`docs/REPRODUCTION.md` 记录当前边界。
- reuse_hypothesis: 可避免配置后续编辑使同一组合权重产生不同后处理结果，也能防止复现说明再次
  混淆部署复现、源权重追溯和完整重训练。
- verification_gap: 本地缺少正式 fused/source checkpoint；需在服务器用正式权重完成一次全测试集
  推理、自动校验和打包，确认旧 v1 bundle 的输入尺寸回退与最终预测保持一致。
- limits: 新 bundle 会显式保存输入尺寸和构建配置哈希；旧 v1 bundle 没有输入尺寸字段时仍从构建
  YAML 回退读取，因此在重建组合包前尚未完全消除这一项外部配置依赖。
