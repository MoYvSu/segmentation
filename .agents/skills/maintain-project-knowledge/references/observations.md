# Candidate observations

This file holds potentially reusable findings that still need validation or
scope clarification. It is not a chronological task log.

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

## 20260911-cross-head-refinement-readout

- status: observation
- last_verified: 2026-09-11
- scope: 冻结E10a+G4b、等参数self/cross修正、新GT原任务损失、固定完整部署
- finding: 信息交流的收益与共同修正训练的收益需要分开。cross相对self多10个有效匹配，
  但两组均低于原主线；原始语义像素更准也没有自动改善实例投票及铁素体平均面积。
- evidence: 同五图主线/self/cross匹配715/690/700，漏检计零IoU .76704/.73703/.74844。
  只关闭affinity修正，两组恢复715匹配，但铁素体数699→726、面积误差35.39%→37.68%。
  完整权重e20、部署和源文件已核验，见docs/MAINLINE_CROSS_HEAD_RESULTS_20260911.md。
- reuse_hypothesis: 新增双头交流时保留等预算自身信息对照，并在原部署链中拆分两项输出；
  不能把所有像素损失改善归因于交流，也不能只看原始语义mIoU便复用候选语义预测。
- limits: 一个种子、五图及固定旧主线的增量实验，不否定新GT或其他联合训练方法。cross跨界
  affinity均值略差于self而最终几何略好，均值并非最终分界质量的充分指标。保护负边修正
  只是后续待验证方向，不是已证方法，也不据此增加AGENTS.md规则。

## 20260908-clean60-affinity-calibration

- status: observation
- last_verified: 2026-09-10
- scope: radius-8 新 GT 的 clean60/e110、50% 原尺寸裁剪/e86与 E10a+G4b 的固定部署比较
- finding: 局部裁剪推理的正面响应不等于裁剪训练能改善整图部署。固定 E10a 与完整主线后处理，
  e110→裁剪e86 的六图 mIoU 全部下降，原六对漏界GT的核心仍共用种子和最终实例；本候选不晋级。
  新 GT 保留，采样增强不足以解释或解决主线差距。mean/top2实测也不支持把融合视为已证首因。
- evidence: G4b/e110/e86 的有效匹配854/792/771，mIoU .8482/.8408/.8220；跨粒affinity均值
  .0740/.1093/.1281。e110→e86 的>.5极强误连比例2.128%→1.873%，跨界起点B>.65
  89.18%→92.34%，但共用主要预测区域的GT177→210，明显切成多块的GT162→202。
  面积误差18.48%→11.78%，代理分数82.80→85.21；不能把面积收益直接当成几何提升。
  此前e110输入改为1024原生裁剪时，两处指定共享界面B>.65曾由29.63%→98.29%、
  82.97%→96.28%；训练加入裁剪后再做整图部署未恢复这两对的分离。训练完整120轮，
  loss-best e86；最佳固定整图val loss .39228，原e110 .38162。
  详见 docs/AFFINITY_NATIVE_CROP_COMPARISON_20260910.md、此前诊断及尺度探针报告。
- reuse_hypothesis: 同时看方向有效负连接、标注来源、真正深部误响应、种子及最终实例关系。
  均值、尾部误连、阈值覆盖可能反向变化，不能单独代替分界连续性与最终输出。
  原始方向offset跨输入尺度对应不同物理连接，不能直接比较同名通道作为同一连接。
  固定旧ROI/GT对，保留各自encoder/归一化；更换读出需给两个模型都做对照。
- verification_gap: 历史G4b实际split未找到；当前G4b/Direct算法不同不能当作已证历史泄漏。
  e110/e86同26/6名单和loss尺度，但跳过校准改变随机轨迹，无重复种子；验证与测试缩放不同，
  六图结果不证明所有测试图必退步。理想GT连接经过完整后处理的相容性、初始化等因果尚未隔离。
- limits: 不否定新GT，不以无标签实例数量评价精度，不因代理总分上涨替换官方83.94主线。
  保留损失选优和SSL→双头预热→联合→固定部署验证短链。特定方向直线的.50合成峰值不是
  普遍上限，已退休该强解释，见retired.md的20260910-affinity-mean-ceiling条目。

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

## 20260903-direct-neg2-shared-lora

- status: observation
- last_verified: 2026-09-03
- scope: SSL 直达语义/affinity 双头 ArmB，人工 affinity 负样本权重单变量实验
- finding: 将 `affinity_loss.negative_weight` 从 `1.5` 提到 `2.0` 没有带来稳定实例收益；
  在共享 LoRA 联合训练下，测试目检反而出现铁素体向珠光体的语义漂移，因此不能把更多实例数
  直接解释为更好的跨实例分离。
- evidence: 相同 split 哈希下，原 ArmB epoch 24 在重扫后的 `boundary=0.59` 为
  `score=89.4684, mIoU=0.7913, area_error=0.0020, pred=1110, matches=642`；T1 best
  epoch 16 为 `89.3088/0.7863/0.0001/1125/631`。T1 latest epoch 25 在窄扫中最高仅
  `89.3522`，没有稳定超过原 ArmB。`test_009/011/018` 上 T1 总实例数由 ArmB 的 361
  增至 369，但铁素体实例从 219 降至 182；对比图为
  `mainline_armb059_t1neg2_overview.png`，训练配置为
  `config/train/direct_ssl_semantic_affinity_neg2.yaml`。
- reuse_hypothesis: 后续若增强 affinity 约束，应同时检查共享 LoRA 对语义分类的干扰；优先考虑
  保持原负样本权重、改进模糊/降采样增强，或另做梯度隔离实验，而不是继续放大同一损失权重。
- verification_gap: 只有一个随机种子和六图验证划分，尚无官方黑盒结果；尚未用冻结 LoRA 的对照
  严格证明语义漂移由共享编码器梯度竞争导致。
- limits: 六图面积项对阈值很敏感，千分位阈值造成的小分差不能作为模型优劣证据；本结论主要由
  mIoU、有效匹配数和固定三图类别/实例对比共同支持。

## 20260906-hiera-l-direct-dual

- status: observation
- last_verified: 2026-09-06
- scope: SSL 直达语义/affinity 双头 ArmB，仅将 SAM2.1 Hiera-B+ 底座替换为 Hiera-L
- finding: Hiera-L 在完整 30 轮 SSL 后，`5+20` 轮双头训练明显不足；延长到 `5+95` 后，最佳
  checkpoint 出现在 epoch 31，验证集预测实例与有效匹配显著增加，68 张测试图的实例总数也从
  `4992` 恢复到 `6303`。继续训练到 epoch 100 后，语义头又显著纠回：验证语义 mIoU 升至
  `0.8294`，68 张测试图与主线的逐像素语义一致率达 `96.84%`，同时保持较强实例分离。剩余偏差
  主要是珠光体实例偏碎、铁素体实例略粗，而不是全局语义失真。因此 Hiera-L 已证明有可利用潜力，
  但尚未证明能替代 ArmB 或当前主线。
- evidence: 相同 split SHA-256
  `11e7f14cc3d44317257da1de4fa9db75f670210278569abbd9e48fe3f95f0451` 下，Hiera-L
  best epoch 24 为 `score=86.3230, instance_mIoU=0.7922, area_error=0.0657,
  pred=1049, matches=606, semantic_mIoU=0.8209`；ArmB epoch 24 在重扫后的 `0.59` 为
  `89.4684/0.7913/0.0020/1110/642`，训练日志语义 mIoU 为 `0.8235`。Hiera-L 的
  `0.65` 单点进一步降至 `72.4004/0.7911/0.3431/903/539`。配置为
  `config/train/direct_ssl_semantic_affinity_hiera_l.yaml`，服务器日志位于
  `outputs/direct_ssl_semantic_affinity_hiera_l/train.log`。68 张无标签测试图上，主线、ArmB、
  Hiera-L 分别输出 `6178/6250/4992` 个实例；六图固定目检中，模糊 `test_018` 为
  `138/130/61`，较清晰 `test_045` 为 `171/158/148`，且未见不自然的笔直切割边界。报告为
  `downloads/hiera_l_visual_audit_20260906/VISUAL_AUDIT_REPORT.md`。
  同一 split、固定 `boundary=0.59` 的长程 best epoch 31 为
  `score=89.2553, instance_mIoU=0.7861, area_error=0.0010, pred=1177, matches=717,
  semantic_mIoU=0.8111`；ArmB epoch 24 为
  `89.4684/0.7913/0.0020/1110/642/0.8235`。长程 epoch 100 为
  `78.2465/0.7926/0.2276/1367/809`，说明后期匹配增加但面积项恶化。68 张测试图长程 best 为
  `6303` 个实例，其中铁素体/珠光体 `2572/3731`；模糊 `test_018` 为 `106`，清晰细粒
  `test_045` 为 `180`。长程报告为
  `downloads/hiera_l_long100_visual_audit_20260906/VISUAL_AUDIT_REPORT.md`。
  epoch 100 完整测试输出为 `7305` 个实例，铁素体/珠光体实例 `3946/3359`；像素级铁素体占比
  从 e31 的 `81.67%` 回升到 `86.53%`，主线为 `88.03%`。Hiera-L e100 与主线的语义像素一致率
  为 `96.84%`；铁素体平均/中位实例面积为 `31731/12133`，接近主线 `29540/10213`，珠光体
  则为 `5802/1354`，小于主线 `9305/2009`。完整对比报告为
  `downloads/hiera_l_e100_vs_mainline_visual_audit_20260906/VISUAL_AUDIT_REPORT.md`。
- reuse_hypothesis: 后续评估更大 SAM/SAM2 底座时，应同时要求足够长但可观测的训练日程和同划分的
  完整部署指标；不能以参数量、SSL reconstruction loss、训练 loss 或单一总分推断收益。应按实例
  mIoU、语义 mIoU、面积项和匹配数联合选 checkpoint。类别实例数不能替代像素级语义统计；当两者
  冲突时应进一步检查各类别实例面积。当前 B+ 仍是更稳妥的默认起点，Hiera-L 可作为重点候选。
- verification_gap: 只有一个随机种子和六图有标签验证划分，测试集目检无 GT，且没有官方黑盒；
  没有检查 Hiera-L 在 `0.59` 附近的细粒度阈值。epoch 80 虽有最高实例 mIoU `0.7997`，但因未保存
  中间 checkpoint 无法做对应部署目检。
- limits: 结论只适用于当前 LoRA rank、训练日程、随机双头和后处理；不能外推为 Hiera-L 对所有
  分割任务或完整课程训练链均无价值。

## 20260906-training-monitor-cadence

- status: observation
- last_verified: 2026-09-07
- scope: `train_direct_semantic_affinity.py` 的长短程 GPU 实验启动与可解释性检查
- finding: 100-epoch Hiera-L 实验沿用只保存 `best/latest` checkpoint 和标量日志的 direct trainer，
  启动前没有配置定期可视化 monitor 或中间 checkpoint；长程实验因此无法回溯标量曲线中的关键
  非 best epoch，属于会直接削弱实验可解释性的启动检查遗漏。反过来，10-epoch affinity tail 若仍
  使用 `monitor.interval=10`，也只会留下最终一组图，无法观察短程内的转折。因此 monitor 间隔应随
  总训练轮数缩放，而不能把“每 10 轮”机械套用到所有实验。
- evidence: `outputs/direct_ssl_semantic_affinity_hiera_l_long100/` 仅有 `train.log`、`split.json`、
  `best_direct_dual.pth`（epoch 31）和被最终轮覆盖的 `latest_direct_dual.pth`（epoch 100）。日志显示
  epoch 80 的实例 mIoU 达 `0.7997`，但该状态没有 checkpoint，无法补做对应目检。用户在训练完成后
  明确指出 monitor 缺失是严重错误。后续 Hiera-L semantic recovery 实跑已在 epoch 10/20 分别
  生成周期 checkpoint 和固定 10 图的语义 JET / affinity HOT monitor；两轮 boundary monitor
  逐图 SHA-256 相同，符合 affinity 冻结预期。10-epoch affinity tail 的部署最优出现在 epoch 1
  (`85.7012`)，随后持续退化到 epoch 10 (`82.5579`)，但原配置只在 epoch 10 生成 monitor；这使
  最关键的早期变化只能依靠 best checkpoint 事后补图，无法恢复 e2/e4/e6/e8 的完整视觉轨迹。
- reuse_hypothesis: 以后启动超过短程验证长度的训练前，应显式核对三项可达产物：周期性标量、固定
  样本可视化 monitor、足以覆盖关键曲线转折的中间 checkpoint；当前约定长程 direct 训练每 10
  epoch 保存 checkpoint，并沿用项目已有的两种可视 monitor：JET 语义概率热力图和 HOT affinity
  边界概率热力图。显式配置 monitor 间隔时，以约 5 个时间点为短程下限，使用
  `min(10, max(1, ceil(total_epochs / 5)))`：10 轮取 2、20 轮取 4、100 轮取 10。checkpoint
  周期可独立设置，不必因可视化增密而保存同样多的大权重。无标签 monitor 不输出或宣称无法验证的
  精度数字。
- verification_gap: affinity tail 配置已改为 epoch 2/4/6/8/10 生成 monitor，但尚需在下一次实际
  短程运行核对五组产物；普通 joint long100 分支虽共用 monitor/checkpoint 实现，仍需在下次实际
  长程启动后核对首个 epoch 10 产物。
- limits: 此遗漏不否定现有标量指标或 epoch 31/100 checkpoint；当前仍可比较 best 与 latest，
  但无法准确重建 epoch 80 或 affinity tail e2/e4/e6/e8 等未保存状态。monitor 只负责可解释性，
  不能替代部署指标或 checkpoint 保存策略。

## 20260907-hiera-l-semantic-recovery-selection

- status: observation
- last_verified: 2026-09-07
- scope: Hiera-L direct semantic/affinity 双头从长程 epoch 31 出发的 semantic-only recovery
- finding: 锁定 encoder、LoRA 与 affinity decoder 后，20 轮 semantic tail 可把验证 semantic mIoU
  从 `0.8111` 恢复到 `0.8290`，几乎达到原 100 轮联合训练的 `0.8294`；但按部署总分选出的
  `best_direct_dual.pth` 是 semantic mIoU 仅 `0.8100` 的 epoch 9。语义恢复类实验不能用单一
  deployment-best 同时代表语义目标，应并行保存 deployment-best、semantic-best、latest 和周期
  checkpoint。这里锁定的是 affinity+LoRA 参数，不是最终实例图的严格几何锁定，因为 semantic
  logits 仍参与 watershed terrain。
- evidence: 相同六图 split、固定 `boundary=0.59` 下，source epoch 31 为
  `score=89.2553, instance_mIoU=0.7861, area_error=0.0010, pred=1177, matches=717,
  semantic_mIoU=0.8111`；recovery epoch 20 为
  `85.1205/0.7855/0.0831/1177/722/0.8290`，semantic 最高 epoch 17 为 `0.8291` 但未保存。
  source 到 epoch 20 的 affinity 39 个张量与 LoRA 192 个张量逐张量 `torch.equal`，最大差异为
  `0.0`；六图最终前景 XOR 为 `0.1772%`、前景 IoU 为 `0.9982`、原始实例 ID 像素一致率为
  `95.13%`。铁素体预测实例由 `606` 增至 `674`，平均面积由 `36124` 降至 `33155`，解释了
  面积误差上升。epoch 20 将 boundary threshold 单点提高到 `0.65` 仅得
  `85.7198/0.7890/0.0746/1033/622`，不能靠该阈值变化解决问题。配置为
  `config/train/direct_ssl_semantic_affinity_hiera_l_e31_semantic_recovery.yaml`，远端紧凑证据为
  `outputs/runs/20260906_220734_direct_dual_semantic_recovery/metrics.csv` 与同目录 `run_info.json`。
- reuse_hypothesis: 后续阶段式多任务训练应按阶段目的保存独立最佳 checkpoint，并在宣称冻结几何时
  验证最终部署实例图，而不只核对 affinity 权重或实例总数。当前结果支持把 semantic tail 作为
  联合训练后的低成本纠偏手段，但后续应在语义锁定条件下单独修复 affinity/实例面积，而不是让
  语义头重新偏移来换取面积项。
- verification_gap: 只有一个种子和六张有标签验证图，epoch 17 checkpoint 已丢失，尚无正式测试集
  目检或官方黑盒；source epoch 31 的近零面积误差究竟有多少来自类别误判与几何误差抵消，仍需
  更大验证证据确认。
- limits: 不能把 semantic mIoU 提升直接解释为竞赛总分提升；也不能因总实例数相同就声称实例图
  不变。当前正式主线仍是 E10a+G4b，不由本实验替换。

## 20260907-direct-pseudo-negative-weight-normalization

- status: observation
- last_verified: 2026-09-07
- scope: direct semantic/affinity 训练中的纯 SAM2 pseudo batch 与 `balanced_affinity_loss`
- finding: 当前 `normalize_edge_weights=true` 时，`pseudo_negative_weight` 对一个纯 pseudo batch 内
  所有负边施加相同的任意非零系数，该系数会在负边加权均值的分子与分母中完全抵消。因此把它从
  `1.0` 改为 `0.25` 等非零值不会实现伪负边衰减；只有置零、关闭归一化，或把负项系数放到正负
  项聚合层才会改变 loss。
- evidence: `train_direct_semantic_affinity.py::compute_affinity_loss` 先把 pseudo 负边统一写为同一
  edge weight，`utils/affinity_loss.py::balanced_affinity_loss` 随后以这些权重之和归一化每个负边项。
  固定 logits/target 的最小数值复验中，weight `1.0/0.25/0.01` 的 loss 均为
  `0.657836437225`，置零后才变为 `0.288037955761`。当前 direct pseudo loader 每步提供独立的
  SAM2 batch，符合统一缩放被抵消的条件。
- reuse_hypothesis: 后续设计 SAM2 cross-mask 负边退火或置信度权重前，应先修正权重语义并加单元
  测试；否则会消耗完整训练但实际配置没有改变优化目标。
- verification_gap: 尚未比较修正后的伪负边调度对验证和黑盒指标的效果。
- limits: 当负边权重在边内不均匀、同一聚合中混合不同来源，或
  `normalize_edge_weights=false` 时，统一抵消结论不一定成立；这不是否定 edge weighting 本身。

## 20260907-hiera-l-affinity-tail-control

- status: observation
- last_verified: 2026-09-07
- scope: Hiera-L semantic recovery epoch 20 之后冻结 encoder/LoRA/semantic 的 affinity-only tail
- finding: 当前 manual + SAM2 affinity 目标只在第 1 个 tail epoch 带来小幅部署改善；继续优化到
  epoch 10 时训练 loss 仍下降，但强边界进一步锐化、预测实例持续增多并使铁素体面积项恶化。
  因此该阶段若保留，应视为极短 affinity 校准而不是新的长程课程阶段；延长同一损失没有收益。
- evidence: 固定六图 split 与 `boundary=0.59` 下，epoch 0/1/10 分别为
  `score=85.1205/85.7012/82.5579`、`instance_mIoU=0.7855/0.7922/0.7857`、
  `ferrite_area_error=0.0831/0.0782/0.1345`、`pred=1177/1188/1246`。epoch 1 是唯一超过起点的
  tail 点；到 epoch 10，manual/pseudo loss 从 `0.4360/0.4236` 降至 `0.4248/0.4088`，但部署
  指标反向退化。source 到 e1/e10 的 semantic 65 个张量与 LoRA 192 个张量逐张量相同，affinity
  39 个张量全部变化。固定 10 张无标签图上 e1/e10 实例数为 `2009/2120`，新增 111 个实例中
  91 个为铁素体，铁素体平均面积下降约 7.3%；语义 monitor 逐图相同，affinity 高分位响应增强。
  配置为 `config/train/direct_ssl_semantic_affinity_hiera_l_affinity_tail_control.yaml`，远端紧凑证据为
  `outputs/runs/20260907_100623_direct_dual_affinity_recovery/metrics.csv`。
- reuse_hypothesis: 后续 affinity-only 尾训应优先使用 1--2 epoch/少量 step、部署指标早停和密集
  monitor；若希望继续训练更久，应先改变 affinity 的目标聚合、置信度校准或伪监督构成，而不是只
  降低学习率或延长当前 loss。
- verification_gap: 只有一个随机种子、六张有标签验证图和十张无标签目检图，尚无官方黑盒；原运行
  未保存 e2/e4/e6/e8 checkpoint，无法完整重建短程视觉轨迹，也未拆分 manual 与 SAM2 pseudo 各自
  对过分割趋势的因果贡献。
- limits: e1 只是 Hiera-L 短链内部候选，不能据此替换 E10a+G4b 正式主线；无标签实例数和面积统计
  仅作目检诊断，不能当作精度。

## 20260907-four-stage-loss-selected-training

- status: observation
- last_verified: 2026-09-07
- scope: direct semantic/affinity 系列的正式复现训练链与阶段间 checkpoint 选择
- finding: 正式复现链应限制为四个可解释的宏阶段：LoRA SSL、冻结 LoRA 的双头预训练、开放
  LoRA 的双头联合训练，以及至多一次冻结 LoRA 的双头分项校准。阶段间允许选优，但必须使用该
  阶段预先声明的优化损失最小 checkpoint；六图部署指标、面积项或事后目检不用于决定下一阶段
  起点。固定周期 checkpoint 只负责恢复与诊断。
- evidence: 当前六图指标已多次表现出面积项和阈值敏感性，不能稳定代表泛化；本轮 normalize
  对照采用 e25 对 e25 才能隔离固定训练长度效应。现有历史主线包含多个不同目标、缓存和模型节点，
  用户已确认其依赖关系难以准确讲述和冷启动复现。本轮修订报告为
  `output/degraded_val_fixed_20260907/VISUAL_AUDIT_NORMALIZE_VS_LEGACY_20260907.md`。
- reuse_hypothesis: 后续设计 direct 训练和最终报告时，把实验探索图与正式权重依赖链分开；为每个
  宏阶段固定 loss 名称、方向和权重，并保存 `loss_best`、`latest` 与恢复 checkpoint。这样既保留
  合理的阶段选优，也避免把偶然的中间模型扩张成新的课程阶段。
- verification_gap: 尚未把四阶段约束和 `loss_best` 选择写入统一编排器，也未确认当前各 trainer
  记录的 total loss 是否都能在跨 epoch 时按完全一致的采样和权重口径比较。
- limits: loss 最小只决定下一阶段的可复现起点，不等价于最终竞赛性能最优；最终模型仍需按完整
  部署口径验证。不同阶段的 loss 定义和量纲不同，禁止跨阶段直接比较数值。

## 20260907-monitor-preview-and-visual-board

- status: observation
- last_verified: 2026-09-07
- scope: direct 双头训练 monitor 与测试集无标签横向目检交付
- finding: 原尺寸 JET/HOT monitor 为 `2584x1936`，服务器界面难以整图浏览；训练应保留原图，
  同时在每个 epoch 的 `preview/` 中写最长边 512 的等比缩略图。模型横向目检不应只散列若干独立
  图片或文字链接，应额外生成一张每行“原图 / 基线 / 候选”的高分辨率拼接图，并在标题中注明
  checkpoint 身份与 F/P 实例数。
- evidence: `train_direct_semantic_affinity.py::save_direct_monitor` 已支持 `preview_max_side`，基础 direct
  配置设为 512，`tests/test_direct_semantic_affinity.py` 验证保留 12x8 原图并生成 6x4 preview；服务器
  已为 normalize e5/e10/e15/e20/e25 回填 100 张 512x384 preview。横向大图及报告为
  `output/mainline_vs_norm_e25_20260907/board/overview.png` 和同目录 `VISUAL_AUDIT_REPORT.md`。
- reuse_hypothesis: 后续训练可先看 preview 判断趋势，需要局部细节时再打开原图；所有主线晋级目检
  沿用单张拼接大图，减少因逐图切换遗漏跨模型一致性和类别组成偏移。
- verification_gap: preview 已通过 CPU 测试和服务器现有产物回填，但尚未由下一次正式训练自动生成
  首组产物；拼接图仍是无标签诊断，不能验证精度。
- limits: 热力图缩略图只用于趋势浏览，细边界判断必须回看原尺寸文件；F/P 实例数和像素占比不能
  替代有标签或官方黑盒指标。

## 20260907-musam-ais-gt-oracle

- status: observation
- last_verified: 2026-09-07
- scope: 当前 LabelMe 实例 GT 上的 μSAM AIS 三通道距离表示与 seeded-watershed 后处理
- finding: 原版 μSAM `foreground + center-distance + inverted-boundary-distance` 即使输入完美
  GT 距离图并使用偏有利的完美 annotated-support mask，也不能在当前复杂 polygon 形态和
  512/1024 geometry grid 上近乎无损恢复实例；主要失败是官方连通分量定义及同一凹形、狭长、
  互嵌或断裂 GT 内产生多枚 marker，而不是轻度回归噪声。因此当前不进入独立 μSAM geometry
  decoder 训练。
- evidence: 当前 direct 固定六图共 `942` 个 GT。512-grid 官方默认阈值下得到
  `pred=989, matches=899, valid_mIoU=0.85883, symmetric_mIoU=0.78067,
  ferrite_area_term=0.96075`；官方 `apply_label=True` 先产生 960 个 target component，按原始 GT
  统计有 59 个实例含多 marker，共 62 个额外 marker，15 个 GT 无 marker。
  center 阈值单变量降至 0.4 是小扫中代理总分最好条件，但 symmetric mIoU 仅 `0.80210`；距离图加固定
  `sigma=0.03` 噪声仍为 `0.77639`，不是首要退化源。1024-grid 默认单点虽把 valid mIoU 提到
  `0.90992`，仍输出 `1007/942` 个实例，60 个 GT 内产生 66 个额外 marker。完整证据与复现命令
  见 `docs/MUSAM_GT_ORACLE_M0.md` 和
  `outputs/experiments/musam_gt_oracle_v1*/musam_gt_oracle_summary.json`。
- reuse_hypothesis: 以后引入新的实例几何表示前，先在当前固定 split、目标输出分辨率和最终后处理上
  做不读取 GT 数量/ID 的 Oracle；若完美表示不能高保真恢复，就不要用神经训练或大规模扫参补救。
  μSAM 相关实验若重启，必须先改变实例定义、全局拓扑表示或原生高分辨率条件，而不是只换 decoder
  或损失权重。
- verification_gap: 没有测试原生 1936x2584 输出、组合阈值大扫或学习预测；这是有意的 No-Go
  边界，因为 512/1024 完美真值均未满足近乎无损门槛。
- limits: 未覆盖像素只在训练有效性 mask 中保持 ignore；因 watershed 不支持 ignore，Oracle 重建
  使用完美已标支持域 mask，这比真实部署更有利。结论只适用于当前赛题标注、互斥栅格化、全图
  1024 输入以及 μSAM AIS marker/watershed；不否定 μSAM 在细胞实例数据上的论文结果，也不替换
  E10a + G4b 正式主线。

## 20260907-labelme-gap-supervision-conflict

- status: observation
- last_verified: 2026-09-07
- scope: 32 张 LabelMe 人工图在 direct semantic/affinity 双头中的 target 构建与窄接缝补全
- finding: 当前 direct 人工监督没有把 polygon 未覆盖区当 unknown：语义 target 把它们全部写为
  `0=珠光体`，全图 BCE/Dice 又没有 annotation-valid mask；affinity 同时把 covered-uncovered pair
  当低权负边。因此同一标注接缝会同时推动珠光体偏误和过分割。semantic/purified GT 还使用坐标
  截断栅格化，instance affinity 使用四舍五入栅格化，造成覆盖区 target 也不完全一致。修复顺序应
  先统一 canonical instance/class target 和 ignore/provenance，再讨论动态多任务权重或新模型头。
- evidence: 32 图平均未覆盖 `9.58%`；覆盖区 semantic 与 instance-class 平均冲突约 `0.46%`。
  原标注包含 5127 个有效 shape，canonical rasterizer 接受 5126 个；43 个实例 ID 不连通，累计
  多出 69 个 component。以原实例作 marker、Lab Scharr gradient 作 elevation 的 native `r=8`
  窄带 watershed 将覆盖率从 `90.42%` 提至 `96.68%`，填入 `65.38%` gap，且原标注像素零修改；
  `r=16` 提至 `98.22%`，但风险更高。固定六图 μSAM 512 Oracle 的 symmetric mIoU 从原始
  `0.78067` 变为 r8 `0.77952`、r16 `0.75544`，故补缝未挽救 μSAM 表示。证据见
  `outputs/experiments/seeded_label_completion_o1/summary.json`、目检拼图和
  `docs/MANUAL_TARGET_V2_EXPERIMENT.md`。Normalize B+ 同周期 e25 对照在 r8 六图、unknown-ignore
  新口径下，semantic mIoU 从 Control `0.90376` 提至 Target V2 `0.94482`；用户目检确认语义概率
  区域更均匀、过渡区显著减少。`boundary=0.63` 时 valid/symmetric mIoU 从
  `0.80342/0.47908` 提至 `0.80709/0.49524`，但 affinity monitor 仍表现为背景偏亮和细线中断。
- reuse_hypothesis: 本轮目检已确认 native `r=8` 紫色新边界总体贴合图样且相对旧 GT 显著改善、
  至少不下降，因此 `manual_target_v2` 直接使用补全后的单一 instance map/class LUT 同源派生
  semantic 与 affinity：`original_covered | filled` 均为权重 1，同 ID pair 为正、异 ID pair 为负；
  仅剩余 ID=0 unknown 与 letterbox padding ignore。三态 provenance 只作审计，不再改变损失权重。
  不要全图洪泛，也不要用补全后的 union 重跑旧 Canny boundary purification。
- verification_gap: r8 已通过用户目检并完成 Normalize Hiera-B+ 5+20 epoch 的同周期 Control A/B；
  当前仍缺独立标注集/正式黑盒验证，且 affinity 动态范围与连续性未明显改善，所以不能凭六图
  派生 GT 代理替换 E10a+G4b 主线。
- limits: 在 watershed 派生 GT 上重跑 μSAM 具有构造性，只能验证表示自一致；补区不可用于宣称
  新 GT 精度或替换正式主线。6 个 LabelMe `linestrip` 当前仍被闭合填充，未获人工确认前不得静默
  删除或改写。

## 20260907-output-directory-naming

- status: observation
- last_verified: 2026-09-07
- scope: 训练与分析产出目录命名
- finding: 用完整实验配置拼接目录名会快速失去辨识度。新产出目录采用实际启动时间
  `YYYYMMDD_HHMMSS`，只有确有必要时追加一个极短标签，例如 `_gtv2`；模型、损失和数据细节留在
  配置快照、`run_info.json` 与实验文档中。
- evidence: 已完成的 Target V2 训练目录由
  `outputs/direct_ssl_semantic_affinity_imagenet_norm_target_v2` 迁移为
  `outputs/20260907_203850_gtv2`，时间戳取该次运行的真实启动时间；迁移后 loss-best checkpoint
  SHA-256 保持 `42dad267...0db75`。
- reuse_hypothesis: 后续每次独立训练在启动时生成时间戳目录，短标签只用于区分同一时刻附近的少量
  并行实验；不要继续把 base、增强、loss 和 target 名全部堆入路径。
- verification_gap: 目前只迁移本次已完成实验，未批量改名历史产物，也尚未把自动生成规则接入所有
  训练入口。
- limits: 不在训练进行中改名；历史日志、checkpoint 内嵌配置和运行快照保留当时路径，避免篡改运行
  证据。稳定数据集目录、配置文件名和需要长期引用的发布别名不受此约定约束。

## 20260910-maskset-query-initialization

- status: observation
- last_verified: 2026-09-10
- scope: Hiera B+ / 固定SSL LoRA / 新GT的Mask2Former式随机查询解码器
- finding: 将所有Embedding统一初始化为std0.02会弱化不同query的初始区分，可能在类别损失
  迅速下降时仍产生近乎空的实例掩码。先验证预测实例是否学到，再判断预训练/GT或加新损失。
- evidence: 同一合成特征探针中，只有query_features/query_positions改为std1后，第六层
  query余弦相似度从0.988降至0.848。真实train001/007同配置300步训练，原初始化均0个
  IoU>=0.5匹配；修正后142/167、156/195，匹配mIoU0.8475/0.8404，仍有漏粒和空隙。
  同期保持4096有效点和BCE/Dice/类别损失；见docs/MASK_SET_CLEAN60_EXPERIMENT_20260910.md
  与output/20260910_mask_set_implementation。旧checkpoint仍严格加载，不重置已保存的query。
- reuse_hypothesis: 从官方架构改写实例查询模块时，核对query与位置向量初始化尺度，避免
  不加区分地套用统一Transformer初始化；少量真实训练图的实例匹配能揭示单看loss遗漏的问题。
- verification_gap: 两图拟合仅证明学习门槛恢复，尚无完整训练验证集与正式部署增益证据；不能
  将此尺度作为所有查询架构的通用最优，也不能认定它解释此前affinity路线的退步。

## 20260910-maskset-amp-shared-projection-gradient

- status: observation
- last_verified: 2026-09-10
- scope: 当前 PyTorch 2.8.0 / BF16 / 共享 mask_embedding 的实例集合解码器
- finding: 同一 autocast 上下文中先在 no_grad 里调用共享投影，再在有梯度路径复用，会使后续
  调用使用无梯度的低精度权重缓存。参数 requires_grad=True、在优化器组中以及总loss下降均
  不保证该模块真正训练；FP32短拟合通过也不能覆盖BF16正式训练的梯度行为。
- evidence: 首轮120epoch的6个mask MLP参数无optimizer state，e10/e57/e60/e118/e120逐值相同；
  GPU BF16 mask-only反传时6个参数grad=None。只将首次投影移出no_grad并detach其结果后，
  同一权重推理logits/masks最大差0，6个参数均恢复非零有限梯度。见
  docs/MASK_SET_RESULTS_20260910.md、output/20260910_mask_set_results/*gpu_gradients.json
  与tests/test_mask_set_model.py的FP32/BF16 mask-only梯度回归。
- reuse_hypothesis: 任务头共享模块跨越no_grad与AMP时，用实际精度分别验证各任务损失到关键模块
  的非零梯度及真实参数更新；不要仅验证总loss.backward或LoRA更新。仅对注意力阈值结果detach，
  避免在同一AMP上下文提前缓存应训练模块的无梯度权重。
- verification_gap: 修复后同条件120轮已完成，6个投影参数持续更新。e103最终六图相比旧e118
  多匹配30个实例，匹配IoU提升1.65个百分点，仍低于主线；缺陷仅解释部分差距。见
  docs/MASK_SET_AMPFIX_RESULTS_20260910.md。不能将梯度修复本身等同于最终几何问题已解决。
- limits: 原版query/attention/像素特征仍有mask梯度，不能夸大为整个掩码分支冻结。新版PyTorch
  的缓存行为可能变化，需核对运行版本；本次不升级框架、不修改既有权重与训练归档。

## 20260910-maskset-ownership-versus-coverage

- status: observation
- last_verified: 2026-09-11
- scope: 独立sigmoid掩码、阈值部署与新增跨query归属交叉熵的组合
- finding: 归属softmax只比较query之间的相对分数，不能单独保证绝对掩码覆盖率；所有logits
  同减20时归属损失不变，但掩码可以全部低于推理阈值。应保留BCE/Dice正负监督与unknown-ignore。
- evidence: tests/test_mask_set_loss.py中的test_ownership_does_not_replace_absolute_coverage_loss，
  及docs/MASK_SET_TEACHER_PSEUDO_EXPERIMENT_20260910.md；GPU混合来源两阶段小验证通过。
- reuse_hypothesis: 给掩码集合增加互斥/归属项时，同时检查相对分配与实际sigmoid阈值输出；
  不凭“softmax和为1”宣布覆盖或几何已修复。
- evidence_update: 249张主线伪标签正式同预算A/B已完成。在排除已见图889后的五图上，归属0.5
  相对0减少碎片，GT惩罚IoU由.6342降至.6285，原尺寸已知区未分配率由7.60%升至9.97%；512网格原始重叠
  在六图仅略减，在四张无标签留出图反而更多。见docs/MASK_SET_TEACHER_RESULTS_20260911.md。
- verification_gap: 单种子结果支持同时检查覆盖的必要性，未证明整体logits平移是空缺增加原因，
  未验证其他归属权重。2026-09-11同预算续训/EMA对照使未覆盖10.29%→8.91%，但原始候选平均
  重叠次数1.441→1.469，漏检计零IoU仅.6398→.6479；一致性部分改善覆盖，未解决结构差距。
  见docs/MASK_SET_CONSISTENCY_RESULTS_20260911.md。保持候选观察，不晋升为通用训练规则。

## 20260911-cross-directory-image-aliases

- status: observation
- last_verified: 2026-09-11
- scope: 人工data/raw与赛方data/unlabeled分开组织的实例伪标签与在线一致性数据
- finding: 两目录的同名图像不保证同内容；人工图又可能以别名存在于无标签目录。只在无标签
  目录内按排除名称建立内容摘要，仍会漏掉人工图别名，必须读取人工目录的真实排除图像。
- evidence: 32张人工图在无标签池有不同名称的同内容图；旧249张固定伪标签包含9个别名，
  其中train_589对应验证图train_889。见docs/MASK_SET_CONSISTENCY_EXPERIMENT_20260911.md及
  output/20260911_091212_consistency_launch/content_alias_audit.json。生成器和读取器加入跨目录
  内容检查，相应回归测试通过。
- reuse_hypothesis: 跨目录隔离同时使用图像名称和实际文件内容，保留别名关系；发现亲本已见过
  某验证图时，不能仅删除新阶段样本便称该图独立，应从选优与独立评价中显式排除。
- limits: 文件SHA256发现的是字节相同图像，不保证发现重编码或裁剪副本；本次没有推断这些
  变体。旧训练和选择历史保留，新阶段两组共同去除9个伪标签别名并改用五图损失选优。
