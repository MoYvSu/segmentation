# 配置目录

当前使用方法见[训练、推理与引擎构建指南](../docs/TRAINING_INFERENCE_GUIDE.md)：
训练入口`train_pipeline.py`默认`repro/current_pipeline.yaml`；原C推理入口`infer.py`
默认Torch及`config/inference/area_floor_200.yaml`，实际全零执行由入口明确覆盖历史协议标签。
教师GT候选已完成训练与推理但80.4回分淘汰；全五TensorRT候选已完成150及下载验收，待本包独立回分。
下方按日期保留历史过程，不能将其“刚启动”等措辞当作当前状态。

2026-11-01后续：`repro/current_pipeline.yaml`与`teacher_pipeline.yaml`默认`semantic_teacher_source: joint`，13阶段，省去V6及语义边界缓存；显式`legacy_v6_e9`才保留旧链。已完成joint可用严格`completed_joint_continuation`在新run接续，必须同GT/资产及完整来源封条。76 CPU及真实checkpoint CUDA确定性桥接通过，[当前接续记录](../docs/V6_ELISION_20261101.md)区分等待joint完成与后半段训练。

2026-11-01：`repro/teacher_pipeline.yaml`是[C教师32源训练演习](../docs/TEACHER_C32_REHEARSAL_20261101.md)的独立配方；使用显式教师资产、`explicit_separators_v1`边界域、原SSL/D5a导入和重新建立的监督链。240 CPU及真实GPU预检通过，统一入口已在后台开始Stage1；原默认推理/训练配置和旧GT保留。

2026-11-01：[当前统一训练入口](../docs/CURRENT_TRAINING_ENTRY_20261101.md)使用`repro/current_pipeline.yaml`与私有资产映射；`repro/pipeline.py`覆盖当前三模型全链。完成配置/CPU及现有32/1000/64源校验，未启动实际训练/GT优化。GT、SAM64和未来新权重分别绑定真实版本，不放宽旧消融合同。

2026-11-01：[当前实际训练流程与参数总表](../docs/TRAINING_RECIPE_20261101.md)汇总三权重采用链及预算、初始化、数据尺度、优化器/LR/loss/精度/冻结，历史恢复配方与已核日志分列；旧repro尚未覆盖当前后续阶段。新`tools/infer_compact.py`继续使用`inference/area_floor_200.yaml`继承合同，默认`--geometry-mode C`，可独立指定F0；没有改变推理/训练默认配置。52项CPU与12图C最终成员byte exact已通过，[测速与验收](../docs/INFERENCE_COMPACT_20261101.md)。未来[GT重训计划](../docs/GT_RETRAIN_PLAN_20261101.md)未启动。

2026-10-31：`train/semantic_native_random.yaml`的20轮/2560步已完成，[终态与同12图对照](../docs/SEMANTIC_NATIVE_RANDOM_ANALYSIS_20261031.md)验收通过；独立`tools/infer_semantic_native_random.py`严格接受本随机训练完整终态。另`tools/infer_flat_semantic.py`使用原13b3/旧20a59，仅把WS语义底图置128、保持原raw票，[全150黑盒](../docs/FLAT_SEMANTIC_PACKAGE_20261031.md)已完成并下载，300成员独立验真。两任务权重与身份独立，无新推荐默认或官方分。

2026-10-31：`train/semantic_native_random.yaml`继承原生零噪配方，新增完整语义头随机初始化seed1101、LR1e-4→1e-5；20轮/2560步、同32源GT与裁窗/增强/损失、冻结其它模块。入口`tools/train_semantic_native_random.py`，独立checkpoint版本，不从上一组e20接续。[单组正式训练已启动](../docs/SEMANTIC_NATIVE_RANDOM_20261031.md)，不追加第二组对照；启动后停止代理监督，等用户通知结束。目前无本轮最终指标。

2026-10-31：`train/semantic_native_zero.yaml`的20轮/2560已完成，[终态与两路对照](../docs/SEMANTIC_NATIVE_ZERO_ANALYSIS_20261031.md)核验6302010语义参数实际更新、GT/冻结/源码保持；全图raw→D5a新头票翻转0.954522%，原生U0→U20为0.757242%。独立`tools/infer_semantic_native_zero.py --semantic-checkpoint`只接受正式2560步增量，并显式记录基座＋新头身份；未改推理默认或晋级，不复跑旧带噪全图训练器。原配方full warm-start/LR3e-5→3e-6/batch1、D4与损失均保留。

2026-10-31：[原图语义双用候选](../docs/RAW_SHARED_SEMANTIC_20261031.md)新增`tools/infer_raw_shared.py`，继续核验N1＋final200配置，实际审计另标`raw_shared`：global_restoration不计算，原图全图logit/prob同时供WS与末端票；局部恢复仍zero。未改默认YAML或训练配置。四源无新capture重放用C原图概率的阈值等价辅助logit，严格核实际WS输入hash与终态exact，不冒充网络logit重建。

2026-10-31：[恢复图末端投票候选](../docs/RESTORED_GLOBAL_VOTE_20261031.md)使用独立`tools/infer_restored_vote.py`，仍接受N1＋final200配置，运行时显式`restored_vote/global_restored_shared_probability`；不再计算raw语义，WS与最终票复用原尺寸全图恢复语义。仅12图与四源缓存桥接，几何exact但类别改变；未改默认配置或训练YAML，不将legacy捕获键`raw_probability`误认作raw RGB来源。

2026-10-31：[统一滑窗候选](../docs/UNIFIED_SLIDING1024_20261031.md)新增`tools/infer_unified.py`，继续接受既有N1＋final200配置，实际审计另标`unified_local1024`：全图恢复不计算，语义和affinity同用局部零噪恢复特征；旧`raw_probability`捕获键明确标为局部恢复语义，不能认作raw RGB。仅12图候选与四源交叉完成，未替换全零默认或新增训练YAML。

2026-10-31：[全链零噪声推理](../docs/GLOBAL_ZERO_20261031.md)使用独立`tools/infer_zero.py --global-noise zero`，固定旧20a59/S0/native1024/final200。入口接受既有N1＋final200源配置核验，审计中的实际配置/运行协议另显式记为`explicit_zero_v1`，全图与局部均为zero；原配置和runtime另存为baseline。旧`infer_semifinal.py`及其固定Philox校验仅用于历史对照，不可把它当作全零入口。12源两遍完成，无新训练或推理YAML修改。

2026-10-31：[高阈值缓存2×2](../docs/NAF_THRESHOLD_20261031.md)完成，只在私有冻结配置副本改`inference.boundary_threshold .65→.75`，没有修改任何推荐推理YAML。双输入部分去环伴明显区域合并，旧模型无明确通用收益；旧.65部署保持，两组.75不晋级。

2026-10-31：`train/naf_dual.yaml`对应1000更新已完成，[终态分析](../docs/NAF_DUAL_ANALYSIS_20261031.md)显示原标补缺与负区抑制进步，但完整线覆盖和实际微环风险退步。新state776f仅作候选，旧20a59推理配置/S0白名单不改；建议的单高阈值缓存2×2尚未执行，不预写为推荐部署。

2026-10-31：`train/naf_dual.yaml`继承`boundary_width`，显式双输入版本`boundary_continuous`／`t8_continuous_v1`。仅新增原N1连续float32输入，公共随机初态、原1000更新/16000计划、旧GAN/loss保持；monitor不自动WS或150图。32源只读缓存共享、连续缓存上限1GiB、磁盘保留10GiB；实际快盘展开配置与守卫入口见[执行记录](../docs/NAF_DUAL_20261031.md)。旧推理配置不变，新格式不冒认旧S0权重。

2026-10-31：`train/naf_perturb.yaml`对应s1000已完成，但固定同输入配对不支持普遍收益，不写入推荐推理配置。新state695ca只作候选留档；S0白名单、旧20a59部署保持。见[终态分析](../docs/NAF_PERTURB_ANALYSIS_20261031.md)，资源异常来自独立诊断脚本，不属于训练配置变更。

2026-10-30：`train/naf_perturb.yaml`继承旧`boundary_width`，只新增冻结受扰热图manifest及独立50/50选择流；不继承已训练NAF，不改loss/模型/1000预算。正式运行的实际路径与SHA保存在私有run配置，公开YAML保持相对路径；`monitor.full_readout_steps=[]`，训练期间仅过程热图，不自动WS/150图。见[合同及GPU预检](../docs/NAF_PERTURB_20261030.md)。

2026-10-30：[无网络P连通组合并](../docs/PEARLITE_MERGE_20261030.md)已按用户要求撤掉共享入口开关，不新增配置开关。独立`tools/infer_pearlite.py`固定原C的S0/20a59/final200；末端只执行一次P连通合并，保持F最终ID及类别精确不变。150图包已生成，配方源码留档；新增参数0。

2026-10-30：[实际关系合并](../docs/BOUNDARY_RELATION_READOUT_20261030.md)已实现显式CLI入口，但0.1全范围读出未通过效果晋级，不新增默认启用配置。原命令无`--relation-checkpoint`时保持旧链；启用时要求N1＋旧20a59＋`--seed-support original-zero`＋final200，检查checkpoint来源和新增211865参数。支持`--relation-threshold`／`--relation-expected-model-sha`，不对未知GT或测试实例数自动校准。13源最终输出及90项测试完成，完整SAM2主链本轮未重算。

2026-10-30：`train/boundary_relation_long.yaml`两组均完成1000更新，[分析](../docs/BOUNDARY_RELATION_LONG_ANALYSIS_20261030.md)支持RGB在已见F/F参照上的排序增量。固定0.1仅作此前预设的候选诊断，不写入全局部署；不拿GT相别当部署门控。未改loss、推理配置、实例或原C，尚无新正式黑盒；完整协议／源哈希保留，当前无需重复控制组或追加训练。

2026-10-30：新增`train/boundary_relation_long.yaml`，继承五轮配方，显式`max_updates=schedule_updates=1000`，仅按0／215／500／750／1000更新保存monitor和完整续训权重。maps／RGB均从原共同随机`initial.pt`重新开始，不加载e5；batch64与完整轮尾批58不变，最终是23整轮＋第24轮11批。标准队列`tools/train_boundary_relation_pair.py`显式接收数据／初态／快盘输出路径并逐步核共同输入；旧epoch配置行为保留，旧源哈希checkpoint不绕过严格身份校验。见[执行合同](../docs/BOUNDARY_RELATION_LONG_20261030.md)。

2026-10-30：[H0官方回分](../docs/STRONG_READOUT_20261030.md#官方回分不晋级保留同源c) .7452／.9063，较同末端200的C两项均略降。
本轮只记录回分，不将取消marker弱生长写入默认配置；关系分类实验仍用原C监督输入，配方不变。

2026-10-30：新增`train/boundary_relation.yaml`，新关系分类任务各五轮、batch64、FP32、随机初始化。
maps／rgb通过CLI只改变RGB可用性，共用完整初态和冻结数据，不加载主线模型权重。
配置默认不应用合并；实际FS manifest用`--data-manifest`覆盖，不把机器绝对路径写入配置。
2746例每轮43次、各共215次更新，两组五轮已完成；[合同与完成记录](../docs/BOUNDARY_RELATION_20261030.md)包含source／checkpoint身份和monitor限制。
过程图／日志／权重已复制到快盘`/root/autodl-tmp/seg/supervised/runs/boundary_relation/`，原FS归档保留，不重跑。
服务器新实验统一使用相应路线`autodl-tmp/seg/<路线>/runs/<短名>/`，显式传`--output`；配置中的相对outputs只是可移植默认值，不替代服务器组织约定。
本地两个旧一次性FS启动器现已停用、原件封存；[清理与归位结果](../docs/STORAGE_MAINTENANCE_20261030.md)包括标准代码入口及实际数据路径。
当前relation的完整数据已归回快盘；FS保留原始执行冷备，不篡改旧checkpoint的路径身份。恢复训练须另核路径兼容，不以strict=False绕过。

2026-10-30：[封存无监督语义B e5对照](../docs/SEMANTIC_UNSUP_COMPARE_20261030.md)已完成，
无监督原配置仍在其工作树`train/semantic_distill_band.yaml`，使用配套LoRA/归一化，非只换监督decoder。
本次仅私有独立推理入口，150＋4例固定几何，未新增训练或公共配置；不整体替换当前语义。

2026-10-30：[旧最高＋200交付与5%缓存诊断](../docs/SUPPORT_RATIO_20261030.md)：
旧最高包仅末端200合并，来自旧20a59结果ZIP，不是新NoGAN的area_floor200包。
S0比例仅私有脚本启用0/5%，公共helper/配置默认未改，比例不叠200，发现小P→F风险暂不推广。
本轮无新训练配置或权重变更，NoGAN继续保留独立候选。

2026-10-30：[area_floor三档已获官方回分](../docs/FINAL_AREA_FILTER_20261029.md)，
100/200/500等权81.305/81.620/81.840，均不及旧最高82.625；保持候选，不修改部署缺省。
后续建议对旧最高单独验证200、再研究S0支撑比例，不把当前NoGAN配置当新主线；本轮未新增配置或任务。

2026-10-29：[独立面积参数与三档候选](../docs/FINAL_AREA_FILTER_20261029.md)已实现：
`seed_min_area`控制起点，`final_min_instance_area`控制终态；缺省分别回落旧`min_instance_area`。
`inference/area_floor_100.yaml`及200/500继承固定N1，统一seed50、显式`neighbor_merge_v1`，
仅终态门槛不同，实际NoGAN＋S0还须显式绑定8eb权重与S0参数。旧配置不自动改变动作。
50/50旧路径、显式拆分及合并路径在真实169最终PNG/classes完全相同；源码冻结后顺序CPU出三包。

2026-10-29：[NoGAN无S0已回.7489/.8609](../docs/NAF_NOGAN_20261029.md#官方回分无s0包整体退化)，
整体劣于旧naf-support，不晋级；新S0仍待独立回分。无配置/阈值变更，暂停新增实验。

2026-10-29：新增`train/boundary_color.yaml`，继承`boundary_nogan.yaml`，
只添加冻结D5a零噪恢复RGB条件与相应输入格式/清单，GAN0、ridge0.1、bilinear及其余配方固定。
[真实配对、初态与存载预检](../docs/NAF_RGB_RETRY_20261029.md)已过，GPU128门槛全部通过；
首轮因旧B1绘图入口退出；train2补齐正确RGB monitor，实际s0/s1验收通过，并修正私有验收模式名误拦。
原s1权重接续至总1000，最后检查已到285，当前日志`naf_color/train2/resume/train.log`；训练配方/已有控制保持。
训练monitor仍是B1/none，最终S0另比较，不自动150或改全局默认。

2026-10-29：[新NoGAN＋S0](../docs/NAF_NOGAN_S0_20261029.md)使用同N1配置，显式
`--seed-support original-zero --expected-model-sha 8eb7c8da...`（实际必须完整64位SHA）。
新入口有限允许旧20a59和已桥接8eb，缺省仍旧20a59，不接受任意新SHA；85项CPU/169实际桥接通过，正式150已启动。
[RGB重试](../docs/NAF_RGB_RETRY_20261029.md)的新配置与执行进展见本页上方。

2026-10-29：[已验证最高分](../docs/NAF_SUPPORT_PACKAGE_20261028.md)为N1＋旧20a59 NAF＋
`--seed-support original-zero`（S0），.7467/.9058、平台82.62。配置继续N1，S0是显式CLI读出，
缺省none没有自动改。新NoGAN/ridge候选目前用none；本成绩不属于新8eb。
S0当前只允许已桥接的20a59/8eb，需按完整冻结入口复现；移植其他新G要单独验收，不仅改开关。

2026-10-29：`train/boundary_plain.yaml`已1000更新终态G `989a8954…`，
[固定四格消融](../docs/NAF_PLAIN_20261029.md)支持关闭ridge导致外围多轨/最终小碎块回升，
原loss更低不能代替结构判断。保留`boundary_nogan.yaml`的ridge0.1为优先候选，不扩训或替换部署。

2026-10-29：`train/boundary_plain.yaml`继承nogan，只关闭ridge并独立输出；
[准备回执](../docs/NAF_PLAIN_20261029.md)确认实际同服务器完整初态/GT/计划/原损失及CPU预检配对，
有效批16/1000预算、GAN0、原.65/.45不变；NoGAN包已交付后，本组队列PID318803已启动，
先128 GPU短测过门槛再fresh1000，自验首步、保留过程图；不宣称正式训练已经完成。

2026-10-29：`train/boundary_nogan.yaml`继承ridge，仅`training.adversarial_weight .01→0`和独立输出。
[预检与缓存阈值小测](../docs/NAF_NOGAN_20261029.md)保留ridge0.1/完整随机初态/GT/批16/1000预算及原.65/.45读出。
显式GAN0短测严格验证D不前向/更新、参数缓冲和优化器不变，G真实更新；原补边保线门槛保持。
128短测与独立正式1000已完成（G `8eb7c8da…`），全±8三轨151→0，四测试小F800→34、116点对分开；
原强支持也有下降，保留候选而非部署晋级。.65/.55仅独立读出候选，本轮仍原.65/.45；无新训练或150图。

2026-10-29：`train/boundary_ridge.yaml`已1000更新结束，G `8ae39216…`。
[终态诊断](../docs/NAF_RIDGE_20261029.md#终态分析中央填谷有效外围多轨仍在)支持中央填谷有效，但外围多轨未解决，
四测试小F958→800；原生四人工分隔保留，另有局部粘连/内部响应退化，不晋级或原样延训/出包。
下一仅建议独立GAN消融，尚无新配置/训练；原默认推理及旧控制保持。

2026-10-29：`train/boundary_ridge.yaml`继承resize，仅加显式`ridge_supervision.normal_hull_v1`权重0.1。
[配方与门槛](../docs/NAF_RIDGE_20261029.md)保留完整随机初态/模型/原监督/采样/读出；
只在处理后GT实例走廊恰好一次界面切换处填中央±2暗谷，外侧锚点±4停止梯度。
CPU关闭完整更新兼容、真实缓存资格和128固定批短测通过，独立1000队列已进入正式阶段，真实首步及s0/s1已验。
不自动全150、出包或换部署，默认旧配置不启该项；产物仅私有`runs/naf_ridge/train2/`。

2026-10-29：`train/boundary_resize.yaml`已1000更新完整结束，G `70b7852c…`。
[终态对照](../docs/NAF_RESIZE_20261029.md#终态分析周期伪影下降平行多轨和最终碎分割加重)显示周期/位移敏感下降，
但平滑平行多轨仍在，自然缺口2→0/161、固定四测试小F30→958。不晋级/延训/自动150或出包。
源码/配置合同不变，旧默认checkpoint保持严格兼容；下方“尚未改up/启动”属于当时历史状态。

2026-10-29：`train/boundary_resize.yaml`继承原T8擦断，只加`model.upsampling: project_bilinear_v1`及独立输出。
[实际启动验收](../docs/NAF_RESIZE_20261029.md)已核99/29项CPU检查、同随机公共骨干及D、新up独立随机、128短测门槛；
正式1000更新已有真实首步、initial/latest与s0/s1图。无RGB/shiftloss，训练器/数据/损失/monitor/读出使用原封存代码。
旧默认`pixelshuffle`不新增架构键，仍严格兼容；错误up形状拒绝，未自动150/出包或换部署。

2026-10-29：`train/boundary_shift.yaml`已完成1000更新；[终态诊断](../docs/NAF_SHIFT_20261029.md#终态分析局部位移稳定性改善多轨与最终微块仍在)
显示人工4/4保留，但自然缺口和四缓存微块未改善，不原样续训/自动出包或晋级。
上采样替换仅为下一步方案，当前未新增结构模式或修改旧checkpoint合同。

2026-10-29：`train/boundary_shift.yaml`继承单通道擦断组，仅加`shift_consistency`权重0.1和两张位移monitor。
[实际启动验收](../docs/NAF_SHIFT_20261029.md)已核数据/初态/关闭兼容、256固定批全部门槛，
独立随机初始化正式1000更新已实际首步，s0/s1图和initial/latest保存正常。无RGB，原loss/GAN/LR/批16/读出保持。
当前不自动150/出包/部署替换；快盘约5.15GiB并保留3GiB下限。

2026-10-29：`train/boundary_rgb_erasure.yaml`已完成1000更新，终态G `943503b8…`。
[恢复RGB终态诊断](../docs/NAF_RGB_ERASURE_20261028.md#2026-10-29终态rgb参与学习完整分隔和微块未改善)显示自然缺口仅2→5/161，
人工四例完整分隔4/4→1/4、四测试小F30→163，未晋级/续训/自动出包；固定原B1与部署均不改。
源码/配置沿用启动SHA，权重和缓存仍只在私有忽略目录。

2026-10-28：[恢复RGB条件擦断](../docs/NAF_RGB_ERASURE_20261028.md)使用`train/boundary_rgb_erasure.yaml`，
继承单通道T8擦断，只增加冻结D5a/N1zero恢复RGB；train/monitor清单隔离，显式`float32_unit`不再/255。
NAF随机共同骨干＋RGB入口0，复用afc1bbaa控制；128短测与正式1000首步/权重/s0/s1已验收。
GT/擦断/目标/GAN/LR/原B1与后端不改；不自动150/出包/晋级。权重和缓存只在私有忽略目录。

2026-10-28：[显式擦断](../docs/NAF_ERASURE_20261028.md)使用`train/boundary_erasure.yaml`继承8核控制，
只在线损伤已有支持短段，目标/valid、采样计划、损失与原B1不变；七个共同monitor显式补回历史draw9921。
已完成1000更新/892.01秒，G状态`afc1bbaa…`；人工四例恢复完整分隔，自然共同缺口仅0→2/161、116未解。
raw周期和bridge/thin成环仍在，不原样扩训；终态/共同初态和热数据保留，低频中间/短测已SHA冷备。
复用20a59控制；不加载旧训练权重、不叠S0、不自动150图或晋级。

2026-10-28：[S0独立出包](../docs/NAF_SUPPORT_PACKAGE_20261028.md)已完成150图及一次性手动下载后的本地全量核验。
配置继续固定`inference/semifinal_n1.yaml`；仅入口显式`--seed-support original-zero`选择S0，缺省none保持旧8核读出。
私有包`output/naf_support/submission/semifinal_naf_support.zip`，SHA前缀`939cccb59175`，150对/300文件；
737上游窗口及NAF/语义/raw保持exact，共撤507个seed，平均预测7.873秒/图（NAF0.710秒、读出4.849秒）。
没有修改默认推理配置或替换部署，尚无新回分；原本机自动拉包器已停止，不再后台轮询。

2026-10-28：固定`train/boundary_width.yaml`对应20a59旧G的[读出对照](../docs/NAF_READOUT_20261028.md)
支持保留S0局部撤种子候选；B0/C1不直接采用。原部署配置未改，没有新增默认开关、150图包或官方成绩。
S0沿用原low0.45，不新增面积/类别/比例阈值；后继8核入口/队列见上条，不直接混用旧Region入口。

2026-10-28：`train/boundary_kernel6.yaml`继承`boundary_width.yaml`，只改独立输出和训练/推理共同的ellipse核8→6。
实际使用旧width冻结源码，pointwise/r2/网络/GT/计划/LR/读出不变，公共随机初态1000更新、控制不重训。
已完成1000更新及全部monitor，终态SHA前缀`a9c360ab3b77`；四缓存部分环/周期起伏减轻，
共同缺口仍0/161、小F29→35、小P197→263，尚不支持补边或精度晋级。实际合同、共同分母及限定见
[核6实验记录](../docs/NAF_KERNEL6_20261028.md)。不原样扩训/继续缩核/自动150图或出包，原8核部署入口保持。

2026-10-28（终态）：`train/boundary_coherent.yaml`的1000更新已完成，点阵减弱但带内多轨/小F增加，
当前不原样续训、出包或晋级；只做固定热图的marker桥接独立对照，未改默认配置。
原文读出差异、终态模型SHA与证据见[完整分析](../docs/NAF_COHERENT_20261028.md#终态分析点阵减弱带内多轨和小环仍有退化)。

2026-10-28：`train/boundary_coherent.yaml`继承标准化输入控制，仅将覆盖目标改为
`rgb_loss.coverage_mode: coherent3`并使用独立输出；缺省`pointwise`保持旧目标。
三个可信相邻GT点共用法向偏移，非可靠短段保持原逐点覆盖，全部原positive及分母保留。
不改网络/GAN/负集/学习率/后处理，不继承旧权重，直接复用已完成控制。
90项服务器CPU检查通过，真实预检和G/D初态/配方核验通过；正式启动状态及monitor见
[三点覆盖实验](../docs/NAF_COHERENT_20261028.md)。

2026-10-27（作者式输入试验）：`train/boundary_width.yaml`继承已完成单通道控制，
仅增加训练/推理共同的`input_normalization: {mode: skeleton_dilate, kernel_size: 8}`及独立输出。
整图先处理再crop/D4或pad/tile；缺省none保持旧路径。模型/loss/GT/计划/LR/预算保持，
106项CPU与真实none兼容、初始化/配方核对和128步短测通过；不重训控制、不继承旧权重。
raw T共同缺失指标和标准化自身指标另列，路径及状态见[专题](../docs/NAF_WIDTH_TRIAL_20261027.md)。

2026-10-27（新独立配方）：`train/boundary_rgb.yaml`和`train/boundary_rgb_control.yaml`
分别为RGB＋真实N1二值T与T-only，只有输入模式/输出路径不同，公共骨干从同随机初态建立。
两组各1000更新，以容差截面覆盖＋内部抑制＋GAN0.01（前64步关闭）恢复新GT，G/D LR独立。
新入口`train_boundary_rgb.py`、队列`tools/run_boundary_rgb.py`；旧骨架/Region配置不被修改或续训。
两组128步真实固定批门槛已通过；RGB正式1000更新已结束。队列完成汇总重复checkpoint键导致控制未启动，
已修外层调度并单独恢复原配方控制；不改已封存的训练源码/损失，避免破坏RGB消融。
监督自由度、真实终态及恢复路径见[专题](../docs/BOUNDARY_RGB_NAF_20261027.md)。

2026-10-27（原affinity预检）：[缓存及实际终态门槛](../docs/AFFINITY_GAP_20261027.md)未支持启动C/A。
未完成的`train/affinity_gap.yaml`草稿已移至被忽略私有归档，当前没有可启动的正式训练配置；旧配置/部署保持。
浅盆地候选仅独立诊断，没有自动启用或出包；上下文seed可靠性网络仍是未实施候选。

2026-10-27（后续归因）：[缓存归因检查](../docs/NAF_LIMITS_20261027.md)后，骨架输入NAF实验配置转为封存对照，
不继续扫权重或扩大负掩码；保留原配置/权重，不改默认推理。后续优先连续affinity信息及邻接几何，不启动新训练。

2026-10-27：三份`naf_sweep_h05/l2/s2.yaml`所声明的独立1000步训练均已完成并归档，
单变量及相同初态/采样检查通过；[终态分析](../docs/NAF_TRAIN_SWEEP_20261026.md)暂不支持出包或替换部署。
没有根据monitor调整推理阈值或叠加种子筛选，没有新训练/全150。下方排队状态保留为10月26日启动记录。

2026-10-26：[NAF三权重短训](../docs/NAF_TRAIN_SWEEP_20261026.md)使用
`train/naf_sweep_h05.yaml`、`train/naf_sweep_l2.yaml`、`train/naf_sweep_s2.yaml`，
都以`boundary_naf_region.yaml`为基底，除输出目录外只改一个损失权重。
服务器从旧封存完整配置展开路径，复用原控制manifest/ROI；H05已完成并归档，L2训练中、S2排队。
不改变结构、输入、语义、选割、学习率或读出，不继承旧权重，不自动150图推理或出包。

2026-10-26：[种子支撑四臂对照](../docs/MARKER_SUPPORT_20261026.md)沿用相同N1配置及NAF s1000，
没有新增默认配置或自动启用过滤。独立函数仅使用既有low=.45，零内部支撑撤种子；
8源小样本完成、35项CPU检查通过。下一候选需显式读出接入，不能把未改的旧NAF包当过滤包。

2026-10-26：`tools/infer_boundary_naf.py`复用`inference/semifinal_n1.yaml`，
显式`--naf-checkpoint`接s1000，仅在完整边界拼接后加NAF，不修改原后处理。
完整test169桥接及150图出包已通过，结果已下载；80项CPU回归通过，总参数102.583M。
推理输出与训练输出分开，核验JSON不入ZIP；见[运行与小空腔诊断](../docs/BOUNDARY_REGION_20261026.md#用户授权的150图黑盒包及边界小空腔诊断)。

2026-10-26：`train/boundary_naf_region.yaml`的1000步已正常结束，但未晋级。
保住线幅值却主要加粗旧骨架，116最终缺口仍未分开；采样均衡，不能归为漏用训练源。
固定过程的缺失线响应后期退步，旧可疑线未删；不原样续训/出包，参见[终态分析](../docs/BOUNDARY_REGION_20261026.md#完整训练分析)。

2026-10-26：`train/boundary_naf_region.yaml`继承真实N1纯BCE配置的网络/数据/前1000步LR，
只将监督改为可信内部保持、已有可信线保持及完整8邻接分隔，精确BCE/GAN/PH关闭。
已随机正式开始1000步，旧同预算控制复用，固定后端不变；不是加载旧NAF续训。
81项CPU检查和GPU128步可行性门槛通过，过程图0/1/100/250/500/1000，最终仅固定四图读出，
不自动150图推理/出包。私有正式配置`runs/naf_region/config_run.json`，
详见[监督及实际启动记录](../docs/BOUNDARY_REGION_20261026.md)。

2026-10-26：`train/boundary_naf_real_bce.yaml`对应的6000步完整结束，但未通过最终输出验收。
固定四图原接法均严重粘连，32源强线几乎全灭；五档全图单调校准未补上116既定缺口。
配置与权重保留，不作为部署候选晋级，不原样续训；详见[终态分析](../docs/BOUNDARY_NAF_REAL_20261026.md)。

2026-10-26：`train/boundary_naf_real_bce.yaml`只覆盖真实输入组的`adversarial_weight: 0.0`。
同随机初态512步固定批通过原门槛，64次换批稳定，正式5000步BCE＋1000步BCE/PH已实际更新；
主/拓扑LR、网络、32处理后GT与采样不变。随机重启，不继承短测或旧NAF；四图N1控制直接复用。
旧GAN1真实输入配置保留失败对应。详见[纯BCE排查与启动](../docs/BOUNDARY_NAF_REAL_20261026.md)。

2026-10-26：`train/boundary_naf_real.yaml`继承旧NAF结构/损失/预算，仅改真实N1输入任务及相同N1监控。
必须先整图细化再裁窗；禁止额外人工擦断、按GT遮掉输入未知区，或继承旧NAF权重。
全部32源、同公共随机初态。缓存及重载通过，但512步固定批质量门槛未过，正式训练未启动；
旧6000步权重只在相同N1上重评，不重复控制训练，正式部署未改。
详见[真实输入短测](../docs/BOUNDARY_NAF_REAL_20261026.md)，模板路径需绑定私有缓存回执。

2026-10-26：[NAF6000完整结束](../docs/BOUNDARY_NAF_20261026.md#第6000步终态与固定输入细化对照)。
原`train/boundary_naf.yaml`未中途改变；细化输入仅为独立固定四图研究包装层，不写入正式部署配置。
该包装层能恢复边界/116最终分隔，但新增切分和F均面积漂移尚不支持晋级；无新训练或150图候选推理。

2026-10-26：[N0/N1/N2用户回报官方成绩](../docs/SEMIFINAL_NOISE_SCORES_20261026.md)支持后续优先以N1进行局部零噪声研究。
这不消融全图噪声，也不更改本轮NAF的N0控制；三个推理配置及包均保持，N2微小总分优势不作自动晋级。

2026-10-26：`train/boundary_naf.yaml`是独立NAF恢复分支，不继承旧补边训练配置或其权重。
生成器width48／encoder1,1,2,4／middle2／decoder各1；随机初始化17.632M。
处理后GT32、原生256、有效batch16，5000主训＋1000有限H1微调是阶段检查，
不是原文53,000更新的完整训练；固定旧N0作参照，明确输入0.5二值化并单报adapter-only。
新增GUDHI3.11.0；推理单次、无修复后回退，正式部署配置不变。
CPU／真实GPU短测已过，正式随机初始化训练已实际更新，详见[运行记录](../docs/BOUNDARY_NAF_20261026.md)。

2026-10-25：[尺度检查](../docs/SCALE_SHAPE_20261025.md)只使用封存N0配置和私有研究包装层，
没有修改正式尺度、融合、噪声、模型或训练配置。768／1024／1280是真实原生视野，
均映到1024网络，不是三个网络输入尺寸；裁不出的档明确缺失，不用填充补造上下文。
局部响应有尺度敏感性但未做最终读出；夜间扫参搁置，不将该诊断配置当作新部署。

2026-10-25：`train/boundary_mass.yaml`继承`boundary_structure.yaml`，只覆盖输出目录和
`structure.mass_weight: 0.005`。实际五轮从同一公共随机e0开始，复用0.05结构e5控制，
不重复控制或继承训练后权重；其余条件固定。CPU、实际GPU和启动前独立审查通过，
五轮320更新及e0/e1/e3/e5过程图已完成。人工gap改善、结构项竞争解除，但天然弱界未稳定改善，
116实际仍未分开；保留研究候选，不晋级、原样续训或自动150图推理。
旧`boundary_structure.yaml`仍保留0.05以对应控制，详见[单变量实验](../docs/BOUNDARY_MASS_20261025.md)。

2026-10-25：`train/boundary_structure.yaml`继承RGB控制配置，仅新增结构监督配方。
原网络／公共随机e0／32源采样／五轮320更新／20轮LR日程固定，复用已有RGB e5控制。
在±12原生像素合法走廊内选择连续路径，原N0<.65的自由区释放自复制约束，
增加路径正响应和防整片抬亮项；不训练上游，不把0.9精确GT细线作教师。
73项CPU、实际32源旧配对exact及CUDA梯度／重载／续训、七图e0最终输出通过。
正式五轮／320更新已完整结束，未继承smoke权重。冻结梯度诊断发现mass/path存在竞争，
人工断口改善未转化为116天然漏界分隔；不晋级或原样续训。后续0.005采用上列独立配置和run，
本0.05配置继续绑定原实验。e0/e1/e3/e5缩略图留服务器，不自动150图推理，
详见[结构短训分析](../docs/BOUNDARY_STRUCTURE_20261025.md#五轮完成后的分析)。

2026-10-25后续：原RGB队列在CUDA第三步SHA核验失败，原五轮未开始。
修复只对该续训核验临时确定化，不改正式数值模式；29本地检查和实际CUDA门禁通过，
新封存`boundary_context_fix`已从共同随机e0实际启动五轮。旧控制与失败现场保留，
没有续用两更新smoke权重。详见[实际重试记录](../docs/BOUNDARY_CONTEXT_20261025.md#实际cuda通过与正式启动)。

2026-10-25：`train/boundary_context.yaml`继承`boundary_real.yaml`，仅增加对齐原RGB输入。
公共参数绑定旧随机e0，新增RGB通道权重为零，不继承e5；严格独立checkpoint版本。
原真实热图配对、32源×8窗、batch4、五轮320更新及原损失不变，旧控制直接复用。
运行配置另绑定新四图只读monitor，保留旧三图及e0/e1/e3/e5缩略图。
CPU预检/独立审计已过，CUDA smoke及正式五轮已接入等待N0结束的单GPU队列，尚非已完成训练。
见[RGB输入消融](../docs/BOUNDARY_CONTEXT_20261025.md)及[新150图登记](../docs/SEMIFINAL_DATA_20261025.md)。

2026-10-25：`train/boundary_real.yaml`继承补边短训配置，显式选择`real_n0_heatmaps`。
先在32源原图上缓存完整冻结N0热图，目标保持实际幅值、宽度与底部，再做局部断口/假线重建。
沿用相同随机初始化、采样计划、五轮预算和固定三图读出；不加载旧补全e5，不切换N1/C5。
`real_recipe`明示损伤参数，`target_sigma: null`表示不再渲染GT软线。
模板私有manifest路径需在运行时绑定实际文件；来源、噪声协议及SHA不匹配会拒绝启动。
详见[真实热图配对](../docs/BOUNDARY_REAL_20261025.md)。

2026-10-25：`train/boundary_restore.yaml`为原生512单通道独立补边短训，
全部32源处理后新GT、每源8窗、有效batch4、每轮64更新，先五轮，保留20轮LR日程。
`train_boundary_restore.py --preflight/--smoke/--run`分别执行CPU预检、独立两更新与正式训练。
e0/e1/e3/e5固定过程图接入原N0冻结缓存；仅在连续边界拼接后增加补全，不训练上游或更改后处理。
真实运行在私有配置中绑定源回执与冻结控制SHA，模板中的相对路径不代表数据已包含在仓库中。
详见[补边试验](../docs/BOUNDARY_RESTORE_20261025.md)。

2026-10-25：`train/semantic_dino.yaml`为冻结DINOv3 ViT-B/16＋随机末端语义头候选，
使用32源处理后新GT，每轮整图/原生1024各128视图、64更新、有效batch4，先五轮，保留20轮LR日程。
原几何及部署控制不训练，e0/e1/e3/e5固定monitor已接入。`train_dino_semantic.py --preflight`
只核数据/采样/monitor输入；`--smoke`要求真实授权骨干并测试2更新；`--run`才正式五轮。
模板权重SHA为空会拒绝训练，需使用实际官方权重及准备好的运行配置，不能用随机或任务分割权重替代。
当前真实数据预检及CPU检查通过、正式训练待授权，详见[本轮说明](../docs/SEMANTIC_DINO_20261025.md)。

2026-10-24：半决赛新增显式150图推理入口，使用`tools/infer_semifinal.py`。
旧`inference/submission_reflect.yaml`省略`submission.dataset`时仍为
`quarter_final_100`（`test_069`至`test_168`），不会根据文件数量自动切换阶段。
以下三组通过`_base`继承，均明确采用`semifinal_150`（`test_169`至`test_318`）：

| 组别 | 配置 | 局部 affinity 恢复 | 固定项 |
| --- | --- | --- | --- |
| N0 | `inference/semifinal_n0.yaml` | 原D5a，首步固定Philox噪声：`single/restorer` | 全图恢复、全图及末端语义、窗口、融合和后处理 |
| N1 | `inference/semifinal_n1.yaml` | 原D5a，仅局部零噪声：`zero/restorer` | 与N0相同 |
| N2 | `inference/semifinal_n2.yaml` | 局部换C5 e5并保持零噪声：`zero/local_restorer` | 与N1相同，全图仍用原D5a |

`local_restorer.pt`独立保存并核对SHA，不覆盖`restorer.pt`。旧图小测必须显式切回旧阶段，
例如（原图目录和模型目录替换为实际位置，输出使用新目录）：

```bash
python tools/infer_semifinal.py --config config/inference/semifinal_n2.yaml --dataset quarter_final_100 --input-dir /path/to/old_test_images --model-dir /path/to/model --output-dir outputs/n2_smoke --only test_101
```

子集仍按所选阶段的完整编号计算固定索引，不按本次`--only`顺序重新编号。
新150图正式推理使用对应配置默认阶段，不加上述旧阶段覆盖；输出须有150对原尺寸单通道uint16 PNG及0/1类别JSON。
本轮不实现本地代理黑盒或新评分器。固定101/116/166三图两遍GPU验收已通过，新N0与旧输出逐文件相同，
固定全图路径及实际零噪声状态核验通过；详见[入口验收](../docs/SEMIFINAL_ENTRY_20261024.md)。真实150图尚未获得，也没有新官方收益结论。
赛程及取舍见[半决赛规则与行动计划](../docs/SEMIFINAL_RULES_20261024.md)。

2026-10-07：`train/restore_zero.yaml`继承共同首步配方，C实际注入噪声为零，仍保留双抽样／双forward，
与已有A相同有效batch4和5轮1250更新。`tools/run_restore_zero.py`只运行C并逐步核对旧A，
`tools/restore_zero_monitor.py`保留双推理模式图；58项CPU与实际四更新GPU短测通过。
正式C现已完成5轮1250更新，逐步配对与完整状态严格重载通过；四已见源同零噪声原生1024恢复误差略降，
但不能沿用旧带噪推理。固定后端与旧部署保持，见[阶段结果](../docs/RESTORATION_ZERO_20261007.md)。
A/C已累计完成e10／2500配对更新，`tools/continue_restore_zero.py`保留各自原冻结配置与60轮LR日程，
e7/e10过程图及完整续训状态齐全。C10同零噪声恢复误差仍略低于A10，但相对C5的原生空间模糊
MSE+2.876%，六图固定后端无明确额外增益，暂不原配方扩轮或修改默认部署。
见[e10阶段结果](../docs/RESTORATION_ZERO_E10_20261007.md)；无自动100图推理。

2026-10-07：`train/restore_consistency.yaml`固定部署D5a e60导出，全部1000原图、
后端不加载／训练，两组共同进行T16双噪声首步适配，仅候选附加权重0.1的一致性损失。
各5轮1250更新，沿用60轮LR日程并保存完整续训状态；源数据epoch从60开始，
不重启局部损伤预热。正式两臂均完成5轮／1250更新，逐步配对与e0/e5严格重载通过；
监控e0/e1/e3/e5齐全。独立恢复及固定后端探针显示连续波动降低，但最终面积稳定性没有明确优于旧部署，
见[阶段分析](../docs/RESTORATION_CONSISTENCY_20261007.md)。不自动扩轮、全100图／黑盒或替换部署。

2026-10-04：`train/semantic_grain_view.yaml`预检已完成32源64窗，整对象类别无错，21个有效像素信号全为P。
默认入口`tools/probe_semantic_grain_view.py`只列计划，`--preflight --all-sources`显式执行。
独立`train/semantic_grain_fit.yaml`的一次有限64更新机制短测已完成：从原R1开始，whole／完整GT裁窗共同目标等权；
旧A64只读复用。`tools/fit_semantic_grain_view.py`默认仅列计划，先独立`--smoke`2更新，再fresh `--go64`。
smoke只评固定8整图／8训练裁窗，正式末态固定40整图／64训练裁窗及严格重载均通过，无100图推理或自动晋级。
裁窗局部P错误明显下降，但整图P回退、测试只改变小片且小F组成增加；不原样扩训，默认部署不变。
用户随后授权单独一版固定s64＋reflect原100PNG的评分对照；验收结果以专题追加记录为准，不改训练配方或选epoch。
`inference/semantic_grain_submission.yaml`及`tools/package_semantic_grain.py`已通过固定八图GPU精确验收；
默认只列计划，`--run --limit 8`验收，`--run`才完整100图。该工具只产出内部结果及兼容ZIP；
按10月4日新格式，正式复赛包还必须封装`results/`及完整`code_model/`。当前完整100图和双目录包已完成，
独立工程两图精确复现通过；未重新安装环境或独立重算100图。`inference/submission_reflect.yaml`为便携模板，
由`tools/build_semifinal_bundle.py`填入实际模型SHA；`tools/infer_semifinal.py`为随包推理入口。
包328.67 MiB，平台实际限额未确认，见[提交演习](../docs/SUBMISSION_REHEARSAL_20261004.md)。
已完成目录不fresh覆盖；源位、裁窗配方、编码缓存及监督来源绑定，结果和局限见[本轮说明](../docs/SEMANTIC_GRAIN_20261004.md)。

2026-10-04：`train/semantic_object_fit.yaml`的raw末端A64/B64现均已完成；默认CLI只查计划，不fresh覆盖已完成目录，`--resume`仅用于未完成优化状态。32源各两遍、相同初始化／特征／共同GT，B增区域均值BCE0.5；固定PNG和40源strict重载／BN／上游／源码输入通过。182区域GT分歧64/49/25，固定4918匹配错类1/1/3；B小片GT一致改善不等于真实物相或比赛提升。八测试F数532→636、聚合均面积−16.19%，暂无新官方分数。暂不扩训／全量／黑盒或晋级；后续`inference/semantic_native.yaml`完成冻结309前向，只改类小片；继承它的`inference/semantic_components.yaml`完成9前向与原输出精确回放，当前修正减少34.3031%记录GT像素分歧，保留支路。两工具默认仅列计划、显式`--run`才执行，要求fresh忽略目录；当前不训练、不改最佳。结果见[尺度与分支检查](../docs/SEMANTIC_NATIVE_20261004.md)，前次见[终态分析、标签粒度与权重SHA](../docs/SEMANTIC_OBJECT_FIT_20261004.md)。

2026-10-04后续：`train/affinity_retention_guard.yaml`继承原interior_fit，新增独立负关系KL，原connect控制／损失不改。`tools/probe_affinity_retention_guard.py --go32`32更新完成，一例外围恢复但固定689F／135面积整体仍退步，不扩训或出包。93项CPU、实际梯度／冻结／strict通过，续训状态与s32头保留。`inference/semantic_raw_stress.yaml`与继承它的`semantic_acquisition.yaml`分别完成只读原图S-align光度／原生局部模糊检查，各四固定源21前向，原标整对象均未错相；强模糊仅局部像素错误增加，不据此声称测试饱和或开同项训练。详见[外围保护](../docs/AFFINITY_RETENTION_GUARD_20261004.md)、[光度压力](../docs/SEMANTIC_RAW_STRESS_20261004.md)与[采集模糊](../docs/SEMANTIC_ACQUISITION_20261004.md)。

2026-10-04：`train/teacher_seams.yaml`继承`affinity_gt_source.yaml`，只恢复原teacher窄零缝支持，绑定派生manifest与旧B320步／状态SHA。旧common继续决定crop、原合法答案保持；同R1起点5轮已完成，不重复控制。配对／冻结／strict通过，e0/e1/e5齐全；六源最终收益不明确，按用户取舍搁置伪GT，不扩训／全量／出包。只读`tools/probe_teacher_seams.py`／`tools/probe_teacher_seams_extra.py`分别核对四固定与两预随机源。已返回独立新GT外围保护方向，最新状态见上方guard条目。见[末态分析与取舍](../docs/AFFINITY_TEACHER_SEAMS_20261004.md)。

2026-10-04：`train/affinity_gt_source.yaml`继承`affinity_balance_mixed.yaml`，用于相同32训练源的已处理人工GT／固定部署伪几何来源对照。用户已确认本批候选；两组各e5／320更新已完成，冻结上游、语义与D5a，共同ignore与实际输入配对。A100图完成，B按短跑分析需要停在75；未来队列`tools/run_affinity_gt_source_queue.py`默认仅训练，显式`--evaluate`才追加全量。已有目录不能重复fresh运行；不修改最佳部署，不凭暗热图续训。配方、结果与限制见[本轮说明](../docs/AFFINITY_GT_SOURCE_20261004.md)。

2026-10-03：`inference/affinity_interior_trace.yaml`和`inference/affinity_support_trace.yaml`仅用于冻结末态定位，入口分别`tools/probe_interior_trace.py --run`／`tools/probe_interior_support.py --run`。封存输入、私有目标和旧头按SHA固定，重算实际完整窗口；第二项还需核对control缓存与封存ZIP清单。两项已完成，不训练、不改GT，不替换正式推理配置；见[完整结果与下一步](../docs/AFFINITY_INTERIOR_TRACE_20261003.md)。

2026-10-03：`train/affinity_interior_fit.yaml`是独立三源32更新诊断，继承`affinity_balance_mixed.yaml`，不替换正式部署。私有目标计划按SHA固定；入口`tools/probe_affinity_interior_fit.py --go32`才执行有限训练，默认仅查配方。两组均已完成并出现七源退步，不原样再跑、扩轮或出包；详见[结果与冻结定位方向](../docs/AFFINITY_INTERIOR_FIT_20261003.md)。产物仅在被忽略目录。

2026-10-03最新：[镜像marker完整候选](../docs/AFFINITY_MARKER_TRIAL_20261003.md)复用R1 mixed balance e20及原49部署合同，独立入口`tools/run_marker_reflect.py`固定reflect32，仅改变WS种子，不叠加seams；不新增默认训练／推理配置。
六源及100图完成，原控制全部exact；已交付`output/marker_trial/reflect1024.zip`（SHA前缀`c812fbf3`）。2026-10-03用户更正回分为0.8645／0.9006，等权88.255，刷新已记录最高；后续研究以reflect作已评分参照，默认配置未自动改动。47项CPU检查及远端／本地包契约通过。下方为历史记录。

七源逐阶段CPU重放及独立汇总见`docs/AFFINITY_MARKER_STAGES_20261003.md`，入口`tools/probe_marker_stages.py`／`tools/analyze_marker_stages.py`。只复用已有连续图；原及reflect终态exact，不细化无新增完整目标修复且有安全副作用，未新增训练配置或修改部署默认。

2026-10-03最新：[marker读出定位与镜像边框](../docs/AFFINITY_MARKER_PATH_20261003.md)完成CPU短测；独立入口只换WS种子，原R1及U／R32的32／64完整输出一致。R原匹配名单恢复，固定21面积不变，但原R1全图F数／平均面积有变化；先扩大多源检查，不新增正式训练／部署配置。31项CPU测试通过，下方为历史记录。

2026-10-03最新：[弱界保留32步](../docs/AFFINITY_WEAK_RETAIN_20261003.md)两组完成：相同完整新GT＋局部监督，只改保留项0／1。复用R1 e20及原推理参数，没有新增正式训练／部署配置。
目标弱界与上轮F-P退步修好，但均有另一已知FF相界损失，额外保留独立精度差很小；暂不扩20轮、黑盒或换默认。真实冻结／重载、14项CPU及完整终态检查通过。下方为历史记录。

2026-10-03最新：[复制头32步短测](../docs/AFFINITY_WEAK_FIT_20261003.md)完成。复用R1 mixed balance e20，A原完整有效GT／B局部合法short，各32步；冻结D5a、encoder、LoRA和语义，最终重算全部12窗。
本例两组均拆粘，B深部归属更完整但整图最终F→P增加；只支持现有特征在此例可学，不作为新部署或官方指标。无新正式配置／扩轮／黑盒，默认保持。下方为历史记录。

2026-10-03最新：[弱晶界定位](../docs/AFFINITY_WEAK_PATH_20261003.md)完成，只使用原R1／coverage e20冻结配置，未新增或切换训练／部署配置。
实际目标保留，原关系预测已弱；有标签输出纠正能分开主要区域，但带来侵入／小片风险。后续固定特征复制头32步检查已完成，见上方新记录。下方为历史记录。

2026-10-03最新：`train/affinity_coverage.yaml`的独立20轮／1280更新及100图分析完成，旧R1未重训。
必要资源已迁至`connect.bjb2.seetacloud.com:15930`的临时盘独立项目；配置仍使用原相对路径，未写入服务器绝对地址。
共同3051个已见GT铁素体平均IoU0.900239→0.900120，466共同安全面积MAE2.930%→2.953%，固定弱界未分开；不足以支持扩轮或晋级。
默认训练／部署配置不切换；本轮无新训练、黑盒包或平台提交。详见[覆盖结果](../docs/AFFINITY_COVERAGE_20261002.md)与[迁移记录](../docs/SERVER_MIGRATION_20261003.md)。下方为历史记录。

2026-10-02回分：`inference/instance_seams.yaml`对应seams1024，用户转报0.8625／0.8786、平台总分87.05。
候选ZIP SHA `a2a1c760…`与本地复核一致；两项显示值均分87.055，较原R1约＋0.040分，记录为最高单次成绩，不自动更换默认配置或裁窗实验对照。

2026-10-02最新：新增`train/affinity_coverage.yaml`，继承`affinity_balance_mixed.yaml`。
唯一训练变量为manual native选窗：每轮16旧随机／16合法九点；来源、新GT、增强、原初始化／损失与上游冻结保持。
全部32源实际512格目标及真实8步GPU门禁通过；独立20轮／1280更新已用`tools/run_affinity_coverage.py --go20`启动，旧控制不重训。
输出`outputs/affinity_coverage/candidate/`；保留固定缩略图和完整epoch边界状态，结束自动推理100图native1024。
固定R1对照及默认配置不更换；详见[覆盖实验](../docs/AFFINITY_COVERAGE_20261002.md)。以下为历史记录。

2026-10-02最新：[同相细零缝归属](../docs/AFFINITY_SEAMS_20261002.md)新增候选配置`inference/instance_seams.yaml`，
继承`train/affinity_balance_mixed.yaml`并锁定R1 e20 SHA `43da103c…`及最高原ZIP `ac7533b5…`。
`legacy_none`显式记录原归一化；只有`instance_seam_assignment`启用，无训练或默认配置更换。
`tools/package_instance_seams.py`从原ZIP直接转换；`tools/probe_instance_seams.py`只CPU分析原32训练缓存。

| 分析方案 | 结果 | 结论 |
| --- | --- | --- |
| 固定原467面积与完整匹配 | MAE2.928%→2.283%，共同F IoU＋0.004731 | 已见GT支持小机制验证，不重新筛分母或开启训练 |
| 唯一预设规则／100图格式 | F5984固定、逐图均面积中位＋0.5950%；200文件位深／尺寸／类别／CRC／SHA通过 | 已回分见顶部；不按无标签均面积或小粒尾部调阈值 |

原正像素／ID集合／类别文件保持，未证明拓扑全不变或语义学习；113粒面积增幅＞10%，尤其留意小粒。
候选`output/affinity_seams/test/seams1024.zip`（SHA `a2a1c760…`）已获用户转报87.05；默认配置及原R1训练对照保持。
36项CPU、父114／来源148／新GT／runtime与原467名单验真通过，弱界监督覆盖仅建议尚未执行。以下为历史记录。

2026-10-02最新：[留缝／固定语义blur32](../docs/AFFINITY_FERRITE_BLUR_20261002.md)完整完成，没有新训练／推理配置。
诊断只在局部D5a＋affinity入口施加固定原生σ1–4场，全图两路语义、模型、阈值／融合及后处理保持。
六层clean重放和来源／新GT／冻结通过，454共同安全F面积MAE2.965%→3.098%，未支持直接开新空间模糊20轮。
下一项仅建议可信GT界面监督分配审计，不重复旧界面增力、不压GT遗漏横线；当前尚未开新审计或训练。
最高mixed balance87.015及所有已验证配置不变，无新包／黑盒；原32来源诊断详见专题及私有图册。

2026-10-02最新：[正式1024铁素体检查](../docs/AFFINITY_FERRITE_20261002.md)完成全部32训练原图。
没有新训练／推理配置，最高mixed balance e20的D5a、encoder／LoRA、两路语义、融合／阈值及后处理均未改。
可靠一对一面积仅467粒、配对汇总收缩1.249%；确定一例完整FF粘连，固定多切例有GT漏分歧义。
先拆ID0来源与正式尺度模糊连通性，不依据训练内类别良好或安全小子集推出黑盒病因，不直接加内部负权重。
本轮不训练／打包／晋级，后续诊断仅建议；详见专题覆盖口径和回执。

2026-10-02最新：[划分机制定位](../docs/AFFINITY_PARTITION_20261002.md)完成20组关系干预及两原图正式native1024补查。
已有合法短关系能改变真实最终拆粘，但五预设中只有珠光体109误切在正式原图两组重现。
未新增训练／推理配置，不叠加TAIL、不增权或扩轮，最高mixed balance87.015仍保持；后续监督候选先核实空间覆盖。

2026-10-02最新：[TAIL单变量采样](../docs/AFFINITY_TAIL_20261002.md)：`train/affinity_tail.yaml`继承旧FUSED，
仅新增`affinity_tail_ranking_v1`与`tail_rank_stratified_v1`，改输出目录为`outputs/affinity_tail/`。
完整10%尾池内按预设等距响应秩取最多256点，其余lambda／margin、新GT、图序、初始化与gated不变。
入口`tools/run_affinity_tail.py --queue`；不重训旧控制或旧FUSED，不继承旧fused权重。
41项CPU及8步GPU门禁通过，148源封存、正式e20／1280及100图已完成；续训状态和108固定过程图齐全。
完整新GT四draw匹配393不变、共同389平均IoU−0.003775，保守误切7→8；100图分布较旧FUSED收敛，但未证明更准。
新增8项CPU及只读GPU最终划分诊断通过，旧源码／冻结／控制输出保持；暂缓TAIL黑盒、续训与增权，不改训练或部署配置。
short1024回分0.8623／0.8691（86.570），不晋级；默认最佳仍mixed balance87.015。以下为历史阶段。

2026-10-02：[新GT融合排序](../docs/AFFINITY_FUSION_TARGET_20261002.md)e20／1280及完整100图分析完成，暂缓黑盒。
新增`train/affinity_fused.yaml`继承mixed balance；仅登记人工源附加排序0.1，其余初始化、来源、
混合视野和gated部署不变。使用处理后的radius8完整新GT已知区域，包含补缝，未知区忽略。
入口`tools/run_affinity_fused.py --queue`，8步门禁后原初态独立20轮；原控制不重训，默认最佳配置不改。
四draw完整匹配393不变、保守误切7→10；全100汇总铁素体均面积−4.117%。
实际内部top256仅约0.163%尾端，不因动态loss仍活跃扩轮或加权重；下一项仅建议代表性尾部取样。

2026-10-02最新：[short全100图对照](../docs/AFFINITY_SHORT_20261002.md)已完成，无新训练或默认配置改动。
独立`tools/run_affinity_short.py`读取已评分mixed balance e20的实际config，只临时将
`affinity_deployment.fusion_mode`由gated改short，逐键断言其余不变。两源控制最终输出／语义门禁通过。
138来源与模型不变，全100候选通过输出验收，建议单独黑盒，不改最高分配置或开启Cshort。
分析后单独候选包`output/affinity_short/short1024.zip`，未提交平台或晋级；详见专题身份／分布风险。

2026-10-02最新：[距离／融合诊断](../docs/AFFINITY_DISTANCE_20261002.md)没有新增训练或部署配置。
独立`tools/run_affinity_distance_diagnostic.py`只读复用mixed balance e20，160融合／128梯度／4图终态完成；
共享梯度不支持现在开Cshort。四图仅临时改变fusion mode为short，其余全固定，旧gated终态精确复现。
共同IoU改善但铁素体匹配未增加、预测略增，优先建议固定short全100图风险检查，本轮尚未执行。
最高分配置与134封存来源未改，不自动训练／提交／晋级。

2026-10-02最新：`train/affinity_source_p100.yaml`继承mixed balance，仅β0.5→1，
实际人工／SAM2 0.75／0.75。独立`train_affinity_p100.py`复用封存逻辑，不放宽旧P025白名单。
73项新CPU、独立复核、服务器self-test及八步GPU放行通过；旧控制只读复现／回执复用。
正式e20／1280更新、100图与完整分析均完成，来源／配对／冻结／重载通过；
等召回误边界增加、GT新增误切，P100暂缓黑盒，停止β密扫、不自动扩轮或出包。
新根`outputs/affinity_p100/`，最高mixed balance保持不变。详见[P100结果](../docs/AFFINITY_P100_20261002.md)。

2026-10-02最新：`train/affinity_source_p025.yaml`对应的中断恢复已经完成e20／1280更新、
100图及完整分析，原20轮日程、122来源、冻结与抽样不变，控制没有重训。
专用恢复入口`tools/resume_affinity_source.py`的CPU7项与GPU单步短测通过，独立输出
`outputs/affinity_source_resume/p025/`；小幅等召回改善未形成充分新增纠错，暂不打包／晋级。
P100 β1、实际人工／SAM2 0.75／0.75仅规划，现入口白名单不接受β1，后续须独立实现并短测；
未启动新组或密扫β。详见[来源实验](../docs/AFFINITY_SOURCE_20261001.md)。

2026-10-01最新：`train/affinity_source_p025.yaml`继承mixed balance，唯一来源比β0.5→0.25。
新`train_affinity_source.py`按α=1.5/(1+β)缩放反传，实际人工／SAM2为1.2／0.3，原BCE日志不缩放。
r1／h1／gamma2、原初态、抽样、冻结及最终部署保持；36项CPU、GPU各8步、控制最终输出与有界兼容通过。
`tools/run_affinity_source.py`绑定122份来源，保留旧119份；正式P025 20轮／1280更新已启动PID46561。
β0.5仅8步技术兼容，禁止完整控制重训；结束自动100图1024对照，不自动打包／提交。
H0训练／推理／分析已完成，未见充分新增纠错，暂不占黑盒；融合正式86.850低于最高87.015，
N075／N125继续暂缓、最佳不变。见[来源比例实验](../docs/AFFINITY_SOURCE_20261001.md)。

2026-10-01最新：`train/affinity_sweep_n075.yaml`／`train/affinity_sweep_n125.yaml`继承mixed balance，
唯一训练变量r=0.75／1.25；新`train_affinity_sweep.py`严格白名单核验控制和共同初态。
`tools/run_affinity_sweep.py --smoke`两组各8更新、已有完整输出精确复现及保存／重载／过程图检查通过，
正式同入口各20轮／1280更新及各100图原生1024推理均完成；旧控制不重训。
输出`outputs/affinity_sweep/`；116份实际服务器代码配置、第三方SAM2代码／配置／环境与smoke绑定。
冻结／配对／重载／monitor通过；没有新官方成绩，不自动展开h／β或提交。
默认部署与暂缓融合包保持；见[执行与分析](../docs/AFFINITY_SWEEP_20261001.md)。

2026-10-01最新：[生效参数与监督扫参](../docs/AFFINITY_SWEEP_20261001.md)明确实际读取的loss节与继承但未启用字段。
用户暂缓融合包，当前配置与最高分部署不改。先建议r=0.75／1.25，后续条件h=0／2、SAM2来源比β=0.25／1；
β组需新归一化入口，`pseudo_negative_weight`正数常量被现有分母抵消，不作连续扫参。
现balance入口仅支持r=1复现；首批候选使用专用sweep入口，不直接套旧入口。

2026-10-01最新：`inference/affinity_balance_fusion.yaml`继承当前mixed balance完整部署，
显式注册native／mixed两e20 SHA，仅将同1024视野的连续边界按50/50融合。
`tools/probe_affinity_balance_fusion.py --mode probe`完成12源精确端点核验，14项CPU测试通过；
两头参数分别计入，总90,621,069。内部`--mode full --full-ok`的100图已完成，端点／冻结／源码核验通过。
唯一包`output/balance_fusion/balance_fusion.zip`（SHA `0f88f206…`）完整验收通过，尚无正式回分。
固定D5a／语义／encoder／LoRA／融合阈值／后处理，通用默认未改，无新训练或比例扫描。
身份、局部取舍与正式状态见[融合记录](../docs/AFFINITY_BALANCE_FUSION_20261001.md)。

2026-10-01最新：`train/affinity_pair.yaml`继承mixed balance，仅新增同源互补视野0.25辅助监督。
正式20轮／1280更新与e20原生1024全100图输出完成，输出`outputs/affinity_pair/`。
主序列、共同原初态、优化器、D5a、冻结SAM2／LoRA、语义和后处理固定，已有20轮控制不重训。
36张过程图及完整性通过，主损失仅降0.255%，四张已见GT的589个配对保持同一批，完整输出近似mixed。
用户回分0.8629／0.8760（86.945），比mixed balance低0.070；不续训或晋级。
下一项仅建议固定新native／mixed balance同视野连续边界50/50融合，小诊断通过再出包，尚未实现。
具体身份、数值兼容及结束分析见
[实验记录](../docs/AFFINITY_PAIR_20261001.md)，当前通用部署默认不变。

2026-10-01最新：`train/affinity_balance_mixed.yaml`对应mixed e20＋原生1024官方用户回分
**0.8629／0.8774（87.015）**，成为已记录综合及面积项最高分包；native balance仍保留mIoU最高。
用户提醒+0.170分未必可靠，两组为近邻对照，不因这点差距认定混合采样稳定优于纯局部。
显式使用`outputs/affinity_balance/mixed/final.pt`（SHA256 `43da103c…`）和`--views 1024`，
旧D5a／语义／LoRA／完整后处理固定；通用默认配置不变。交付ZIP `ac7533b5…`重新核验一致。
以下保留早期状态，最新完整身份以[实验记录](../docs/AFFINITY_BALANCE_20260930.md)文末为准。

`train/affinity_balance_native.yaml`／`train/affinity_balance_mixed.yaml`分别继承原纯1024／混合配置，
唯一loss改动`negative_weight: 1.0`（控制1.5），完整候选配置保存在checkpoint中。
入口`tools/run_affinity_balance.py --smoke`后正式执行同入口；两组独立同初态，各20轮／1280更新，
已有控制不重训。本地19项及GPU两组各8更新短测通过，两组正式20轮与全量推理完成；输出`outputs/affinity_balance/{native,mixed}/`。
实际输入源、裁窗、增强及视野逐步对齐各自旧控制，冻结／重载与216张过程图通过核验。
分析`tools/analyze_affinity_balance.py`显示内部连接提高但真分界下降；native原生候选随后正式回分
0.8639／0.8730（86.845），比旧p1024+1.105分、比旧全图control+0.610分，晋级当前研究对照。
显式用本native配置＋`outputs/affinity_balance/native/final.pt`固定e20＋原生1024推理，
旧D5a、语义与完整后处理固定；mixed balance尚无回分，旧全图control留作回退。
mixed balance e20＋原生1024包已交付`output/balance_analysis/mixedbalance1024.zip`，
ZIP SHA256 `ac7533b529a35ce98a94ac8b2822d9f7543ea015762c15343b62dd3207fa3c77`。
它是上述mixed配置的1.0权重候选，与原loss mixed包不同，不直接替换native最佳。
详见[实验记录](../docs/AFFINITY_BALANCE_20260930.md)，不从旧e20续训、不改默认部署。

`train/affinity_mixed.yaml`继承`affinity_native.yaml`，唯一训练变化为全图／原生1024各50%的视野配方。
入口`tools/run_affinity_mixed.py`，复用原control初始化与已完成控制输出，只训练affinity头20轮／1280更新。
每轮精确32次全图和32次局部，同步两种监督来源的视野，源图和增强回执继续与control核对。
损失、旧D5a、语义、LoRA及后处理不变；正式20轮及两套100图已完成，冻结、配对、源码和重载通过。
输出`outputs/affinity_mixed/`，108张过程图完整；分析入口`tools/analyze_affinity_mixed.py`。
混合原生比p1024预测铁素体数+2.76%、均面积变化中位−2.02%，4张已见GT仅多1个有效匹配，未证明净收益。
2026-10-01原loss mixed1024回分0.8582／0.8536（85.590），比p1024综合低0.150分；不晋级，勿与惩罚组混淆。
详见[实验记录](../docs/AFFINITY_MIXED_20260930.md)，不自动出包或晋级。

`train/restore_native.yaml`继承旧D5a配方，只训练新恢复器；A原生1024，B原生1024／512各50%，
网络输入均1024。同一份随机初态，1000源无留出、有效batch4、各20轮／5000更新，保留60轮学习率日程。
原生模糊／降采样／局部任务先于配对放大，明确记录两个像素尺度；不继承旧恢复器权重。
入口`tools/run_restore_native.py`，输出`outputs/restore_native/`，两组已续至40轮／10000更新及100图对照，隔离通过。
A的1024合成去模糊、B的512保真改善，真实输入作用偏弱，尚无最终分割增益；各160张过程图保留。
用户授权的20→40续训已完成：`tools/continue_restore_native.py --stop 40`，原配置／60轮日程不变，
e20备份424文件及e40状态通过复核；下一轮预设lr=0.0000515，未启动60轮或新含噪配方。
只替换局部affinity路径恢复器，主检查固定p1024 e20、全图语义仍使用旧D5a；不改默认部署。
实施、续训命令与结果边界见[实验记录](../docs/RESTORE_NATIVE_20260930.md)。

`train/affinity_native.yaml`继承`affinity_connectivity.yaml`，只新增原生裁块选项和输出根目录。
入口`tools/run_affinity_native.py`先`--smoke`，通过后顺序训练1024／512，每组20轮／1280更新，
分别从旧control的原初始化出发；旧全图control直接复用。两组已完成，配对／冻结／重载通过。
仅affinity decoder训练，原32人工／64 SAM2、退化、D5a、语义与后处理不变；
过程图和各自全图／本尺度局部及control局部的6套100图对照全部完成。
分析入口`tools/analyze_affinity_native.py`，私有图册`output/native_analysis/index.html`。
用户回报1024权重配原生1024为0.8580／0.8568（85.740）、512配原生512为0.8534／0.8130（83.320）；
综合均未超过control的86.235，不替换旧全图入口权重、不改默认配置。
旧control原生1024随后回分0.8551／0.8217（83.840），p1024同入口高1.900，支持当前尺度适配训练收益；
仍保留旧control全图综合最佳。[完整对照与独立复核](../docs/AFFINITY_NATIVE_REVIEW_20260930.md)。
输出`outputs/affinity_native/`，详见[实验记录](../docs/AFFINITY_NATIVE_20260928.md)。

`experiments/affinity_patch.yaml`固定最高分control e20、D5a e60，做原生1024／512、raw／D5a
及逐张／batch检查。入口`tools/probe_affinity_patch.py`，完成8源×6方案完整分区及32对恢复。
追加`experiments/patch_structure.yaml`重放相同32对，检查已知铁素体内部无清晰参照支持的边界响应；
入口`tools/probe_patch_structure.py`。汇总`tools/analyze_affinity_patch.py`，私有`outputs/affinity_patch/`。
两配置均只读冻结诊断，不启动训练或修改默认推理；后续原生裁块训练完成记录见上条。冻结结果见
[专题记录](../docs/AFFINITY_PATCH_20260928.md)。

`experiments/affinity_budget.yaml`为最高分control e20上固定4源8视图的力度机制短测，
入口`tools/probe_affinity_budget.py`，输出`outputs/affinity_budget/`。两组各64步，
仅比较旧输出梯度上限与15%参数梯度预算；使用缓存特征与固定初始候选分区，不是正式续训协议。
2026-09-28完成；新预算粘连修复仅多1个、新增粘连更多，未通过20轮门槛。
配置不会自动启动正式训练，未改变旧训练入口或默认部署；见[记录](../docs/AFFINITY_BUDGET_20260928.md)。

`train/affinity_interface.yaml`继承`affinity_connectivity.yaml`，仅更换附加监督为完整短程界面选位。
输出`outputs/affinity_interface/`，从相同原始初始化只训候选20轮／1280更新，复用既有control。
共同配置、初始化、GT/样本来源和逐步增强回执必须与control一致；原后处理和梯度预算不改。
入口`tools/run_affinity_interface.py`要求64视图检查及同源码smoke通过；2026-09-28完成20轮及100图。
候选额外作用仍小，98.64%控制实例匹配IoU≥0.95，当前保留control；分析入口`tools/analyze_affinity_interface.py`。
详细作用与新增错误风险见[专题记录](../docs/AFFINITY_INTERFACE_20260928.md)，无新黑盒、不晋级。

`experiments/affinity_preserve.yaml`只在最高分control e20上开启既有分隔恢复，
以`marker_partition_restore.area_filter_enabled: false`独立关闭附加log-IQR面积过滤。
原最小面积50和其它部署项不变；旧恢复配置默认仍启用附加过滤，control配置仍关闭全部恢复。
入口`tools/run_affinity_preserve.py --mode train/test`分别运行冻结的标注源诊断／测试推理，均不训练。
64标注视图及100测试图已完成；候选用户回分0.8499／0.8559（85.290），比同权重control低0.945分，不晋级。
详见[实验记录](../docs/AFFINITY_PRESERVE_20260928.md)及[回分分析](../docs/AFFINITY_REVIEW_20260928.md)；未改配置或默认推理。

`train/affinity_connectivity.yaml`保持D5a／joint-v3／S-align／原部署，唯一A/B差异是affinity双向困难关系损失。
两组各20轮、1280更新及100图部署已完成，产物`outputs/affinity_connectivity/{control,candidate}/final.pt`。
control用户回分0.8486／0.8761（86.235），是当前综合最佳研究对照；candidate尚无黑盒。
新增监督实际参数梯度偏弱、完整输出几乎相同；再次只改附加损失时应从相同初始权重训练新候选并复用control。
精确复现使用`train_affinity_connectivity.py --arm control --infer`及本组final，详见专题命令；不沿用旧通用入口推断权重。
分析入口`tools/analyze_affinity_connectivity.py`，详见[完成分析](../docs/AFFINITY_CONNECTIVITY_20260928.md)。

`experiments/marker_restore.yaml`在固定D5a／语义LoRA／V上开启保守种子恢复和统一面积离群过滤。
面积阈值取每图首次真实分割的log面积箱线图下界，系数1.5；一次估计后冻结，对所有实例同等处理，
不读取训练GT。入口`tools/run_marker_restore.py`，私有`outputs/marker_restore/`；其余配置默认关闭。
实现、检查和全量结果见[记录](../docs/MARKER_RESTORE_20260927.md)。

`experiments/semantic_split.yaml`用于只读比较simple／gray4／语义LoRA，固定D5a输入及LoRA最终轮廓。
入口`tools/analyze_semantic_split.py`；最终有效统计`outputs/semantic_split_checked/`。
同配置供`tools/probe_semantic_channels.py`追踪原有种子连通性，以及`tools/probe_marker_anchor.py`
生成只改变种子的反事实输出。前置封边仅存在于诊断helper，没有更改默认部署配置或重训控制。
复现门槛、统计范围与整体方案见[诊断记录](../docs/SEMANTIC_SPLIT_20260927.md)。

`train/semantic_coverage.yaml`目前用于`tools/probe_semantic_coverage.py`的只读预检查。
在256网格新增192／步长64窗口，仅补旧128窗口零支持且全图一致的位置；旧目标保持不变。
完整1000源／1280视图仅14源增加监督，新增目标与已完成LoRA零类别分歧，仅2格未达0.9。
预检查未满足新增信号条件，未接通正式训练协调器或启动20轮；见[实验记录](../docs/SEMANTIC_COVERAGE_20260927.md)。

新增`train/semantic_scale.yaml`：在私有LoRA原配方上，仅把D5a之后的语义输入改为50%全图／50%有效域512方窗放大。
原图先验先生成再同步裁切，保持原核心／概率池化损失及全图部署；复用LoRA e20控制，不重训控制。
入口`tools/run_semantic_scale.py --smoke`后`tools/run_semantic_scale.py`，20轮、输出`outputs/semantic_scale/`；
保留原monitor，并增加固定训练源的同ROI全图／局部过程缩略图。详见[尺度实验](../docs/SEMANTIC_SCALE_20260927.md)。

新增`train/semantic_lora.yaml`：从gray4继承全部数据、灰度监督、梯度预算及20轮学习率，只开放独立语义LoRA。
SAM2基础权重共用、原affinity LoRA冻结；候选重放gray4实际选中的增强视图，不重新选样。
新入口`tools/run_semantic_lora.py --config config/train/semantic_lora.yaml`，首次需加`--smoke`。
控制直接复用`outputs/semantic_gray4/prior`；父配置的旧`reuse_control_dir`不用于本入口。
99项本地检查和GPU短测通过，冻结6步逐值复现gray4；正式20轮／1280更新、100图部署与96张过程图已完成。
相对gray4改判65实例，主要在分类阈值附近；affinity冻结及严格重载通过，末轮`candidate/epoch_020.pt`。
输出`outputs/semantic_lora/`，无新官方分数、不提交黑盒或更改默认模型；结论见[完成分析](../docs/SEMANTIC_LORA_20260927.md)。

新增`train/semantic_gray4.yaml`：继承gray3，增加候选无标签光度困难视图选择及逐步附加梯度预算。
全局／局部类型交替，20%保留原视图，其余从原图增强与两个候选中选较难者；清晰图先验不改写。
名义系数仍最高0.15，实际系数限制使附加梯度范数不超过GT。共同GT、初始化、冻结上游及最终部署不变。
复用`outputs/semantic_gray/control`，76项CPU检查与GPU短测通过；正式20轮／1280更新、100图推理和96张过程图已完成。
对控制102处改判，训练源固定光照视图的区域冲突120／60→1／1；后期同时存在低学习率与当前合成任务趋于满足。
未验证官方收益，未自动续训；建议学习率实验保持视图与监督不变，详见完成分析。
入口`tools/run_semantic_gray.py --config config/train/semantic_gray4.yaml`，输出`outputs/semantic_gray4/`。
训练前诊断入口`tools/probe_semantic_hard.py`，仅抽取训练源；见[实验记录](../docs/SEMANTIC_GRAY4_20260927.md)。

新增`train/semantic_gray3.yaml`：继承gray2，只有灰度损失平均上限改为20，最高权重仍为0.15。
`semantic_consistency.reuse_control_dir`显式引用`outputs/semantic_gray/control`，正式只训练／推理prior。
复用前检查共同训练条件、依赖、旧权重与100图部署，并用新代码6步短测复现旧控制输入及loss。
62项CPU检查及GPU短测／8图部署通过；正式20轮候选、100图部署和48张过程图已完成。
旧控制全1280步配对通过，gray3对控制35／8668实例改判，训练先验分歧较gray2减少31.17%。
这是训练规则拟合与预测变化，不是准确率；输出`outputs/semantic_gray3/`，未晋级或继续训练。
入口`tools/run_semantic_gray.py --config config/train/semantic_gray3.yaml`（首次必须先加`--smoke`）。
过程缩略图和完整100图部署已保存，无新黑盒成绩。见[完成分析](../docs/SEMANTIC_GRAY3_20260927.md)。

新增`train/semantic_gray2.yaml`：原simple e60重新接续的20轮A/B；候选改为单向灰度约束，
可靠亮／暗区达到0.9／0.1后梯度归零，最高系数由0.10提高至0.15，前5轮渐增。
目标接受域、数据、随机种子、冻结模块和最终部署不变；短目录`outputs/semantic_gray2/`。
入口`tools/run_semantic_gray.py --config config/train/semantic_gray2.yaml --smoke`，
本地38项检查、GPU三组短测及完整8图推理通过，正式两组各20轮、100图部署与48张过程图已完成。
实际活动比例0.06146%，对控制仅6／8668实例改判，未晋级或自动续训。
两控制全部模型张量及100图输出一致，后续必须复用旧控制；本轮重复控制权重已清除，
现存控制为`outputs/semantic_gray/control/epoch_020.pt`。显式复用入口已在上方gray3落实；
没有复用字段的历史配置仍会按原流程运行。见[完整分析](../docs/SEMANTIC_GRAY2_20260927.md)。

新增`train/semantic_gray.yaml`：原simple e60接续的20轮A/B；A为GT＋现有增强，
B额外接受清晰训练原图的独立Lab明度软监督。全部1000源循环抽样，已有GT覆盖排除先验，
保留Stage1/joint-v3、D5a、LoRA及affinity冻结。入口`tools/run_semantic_gray.py --smoke`，
短测后去掉`--smoke`，输出`outputs/semantic_gray/`；保存8图过程缩略图及完成后的完整推理。
见[实验约定](../docs/SEMANTIC_GRAY_20260927.md)，45项CPU检查及GPU三组短测通过，
确定性零权重重放完全一致；正式各20轮、1280更新、100图部署及48张过程图均已完成。
B对A为43／8668个实例改判，接受域类别分歧仅0.0455%，多数实例分数向0.5靠近；
无新黑盒或晋级，不自动续训。此旧配置仍为软目标BCE，新单向约束另见上方`semantic_gray2.yaml`。

新增只读诊断`experiments/semantic_teacher_probe.yaml`：原图S-align教师对D5a/simple e60，
检查全部32人工源及按内容排除后固定抽取的64张其他训练图，各两次独立退化。
入口`tools/probe_semantic_teacher.py`，7项CPU检查、三图GPU短测及正式96图均已完成；
64图共同可靠域214万格点的两教师类别完全一致，暂不启动仅换教师的训练。
不创建留出、不读取测试图、不更新模型。口径及结果见
[教师可用性检查](../docs/SEMANTIC_TEACHER_PROBE_20260926.md)。

新增`train/semantic_consistency.yaml`：原simple e60接续的20轮A/B，各1280次更新。
两组共同使用现有模糊／噪声及D5a前新增的正斜率外观模拟，仅B附加1000训练源的可靠语义一致性；
弱教师固定、共享几何、全源循环抽取，无留出。保留Stage1/joint-v3依赖，冻结D5a、编码器／LoRA和affinity。
入口`tools/run_semantic_consistency.py --smoke`，通过后去掉`--smoke`；保存8图过程缩略图，
完成后自动推理／渲染并停止，不自动续训。见[分布依据](../docs/SEMANTIC_DOMAIN_20260926.md)
与[实验约定和完成分析](../docs/SEMANTIC_CONSISTENCY_20260926.md)。两组已各完成20轮及100图推理，
B对A仅7／8668个对应实例改判，额外一致性作用很小；不原样延长到60轮，尚无新黑盒成绩。

新增`train/illumination_calibration.yaml`：独立小型校光模块，全部1000训练源、60轮；
冻结指定V e115＋原simple e60及其joint-v3特征，同一校光权重分别放D5a前后对照。
入口`tools/run_illumination_calibration.py`先`--preflight`、再`--short`，通过后正式训练；
保存五组过程缩略图和四图完整预测，不生成提交包。见`docs/ILLUMINATION_CALIBRATION_20260926.md`。

新增`train/backend_cold.yaml`：固定D5a首步与SAM2基座，纯SSL LoRA＋随机simple/affinity。
60轮两头预热＋60轮LoRA联合，全32人工及既有64 SAM2，无留出，不接额外无标签流或新增光照。
入口`tools/run_backend_coldstart.py`，输出`outputs/backend_cold/`；短测通过，但用户已于e93后停止正式训练与后续推理。
该纯SSL短链绕开尚未消融的Stage1/joint-v3，仅保留探索记录，不作为后续默认初始化。
权重及诊断已保存，详见[实验记录](../docs/BACKEND_COLD_20260926.md)。

新增`train/semantic_light.yaml`：继承simple既有60轮配方，只覆盖输出目录并增加光照配置。
D5a后50%概率增强，整图／平滑局部偏移各半、幅度最多±0.10，限制有效内容新增截断不超过1%。
入口`tools/run_semantic_light.py`只训练simple，复用历史无光照组；短测独立强制光照覆盖，
正式seed仍为`20260925`。60轮已完成并产出100图提交包；受截断保护影响，实际非零增强37.97%。
黑盒回分`0.8474/0.8445`（84.595），比原simple仅+0.055分，不晋级，未修改默认推理。
详见[实验与完成分析](../docs/SEMANTIC_LIGHT_20260926.md)。

2026-09-25：扩散修复与配套后端／语义对照配置已随研究分支归入`main`。
本次合并保留既有推理入口和权重选择；下列实验配置、黑盒结果及采用结论继续有效。

新增`train/semantic_d5a.yaml`：固定D5a首步与SAM2/LoRA/affinity，只训练语义头。
A现有完整头适配（3e-5），B随机简化FPN＋分类层（1e-4）、无直接RGB残差；
全32新GT，60轮、3840更新／组，不留代理，包含D5a端点模糊与配对噪声。
入口`tools/run_semantic_d5a.py`，短目录`outputs/semantic_d5a/`，见[实验约定](../docs/SEMANTIC_D5A_20260925.md)。
已完成60轮／3840更新、零失败及全量打包，两组预测较接近；
[完成分析与亮度探针](../docs/SEMANTIC_D5A_ANALYSIS_20260925.md)确认部分实例对亮度敏感，
现已回分full `0.8473/0.8461`（84.670）、simple `0.8471/0.8437`（84.540），
差0.130分；两组面积项下降、总分低于原D5a与baseline，不晋级。未改默认模型或启用亮度校正。
后续[全32有标签光照诊断](../docs/LABELLED_LIGHT_20260925.md)未发现明显的实例分类脆弱性：
全局±0.10零新增错误、A/B局部调光每条件最多新增1错；这是训练内GT区域诊断，
不能排除测试工况上的光照作用。本轮未新增启用光照增强的训练配置。

新增`train/rgb_diffusion_d5b.yaml`：继承D5a参数配方，从零训练全1000图、60轮，
仅增加25%批次的两步短链监督，独立随机流、跨步切梯度、辅助权重0.25；不继承历史权重。
入口`tools/run_rgb_d5b.py`，短目录`outputs/d5b/`；已完成60轮／15000更新、零失败，
见[D5b分析](../docs/RGB_DIFFUSION_D5B_ANALYSIS_20260925.md)：第2/3步相对D5a改善，但自身首步仍最好，
首步未改善，不晋级。保留[D5b约定](../docs/RGB_DIFFUSION_D5B_20260924.md)。

新增`train/rgb_diffusion_d5a.yaml`和`train/rgb_diffusion_d5a_control.yaml`：
两组随机初始化、全1000图、60轮，唯一训练差异为强弱端点平滑模糊；不加载D4权重。
队列入口`tools/run_rgb_d5a.py`，自动保存固定过程图，见[D5a约定](../docs/RGB_DIFFUSION_D5A_20260924.md)。
两组已完成60轮／15000更新、零失败及配对检查；[完成分析](../docs/RGB_DIFFUSION_D5A_ANALYSIS_20260924.md)
已回分control 0.8511／0.8486（84.985）、D5a 0.8469／0.8560（85.145）。
端点配方净+0.160分，仍未超过baseline或旧v4；不更改默认部署配置。

新增`train/backend_d4.yaml`：冻结D4首步与不带D4两组后端微调，均从S-align＋V初始化，
训练LoRA和两个任务头；同在线退化、全32人工与64既有SAM2源、60轮，无代理或验证选优。
入口`tools/run_backend_ablation.py`自动串行完成训练、推理和对比渲染，见[实验约定](../docs/BACKEND_D4_ABLATION_20260924.md)。

输出目录命名：新建运行目录最多四段（按下划线分隔，时间戳也计一段），例如
`rgb_restoration_diffusion_d4` 或 `diffusion_d4_20260923`。增强类型、轮数、monitor等细节
写入配置和运行记录，不拼入目录名。正在运行的旧长目录可提供短名称入口，不为改名重启训练。

扩散正式候选入口：`train/rgb_restoration_diffusion_d1_all60_monitored.yaml`，全1000图、60 epoch、
从零训练，固定过程图同时记录首步与16步。`train/rgb_restoration_diffusion_d1_overfit3200.yaml`
的4图短测已通过工程门槛，见[短测判定](../docs/RGB_DIFFUSION_SHORT_DECISION_20260923.md)；
正式60轮/15000更新已完成、零失败；[完整分析](../docs/RGB_DIFFUSION_ALL60_ANALYSIS_20260923.md)
认为当前16步版本暂不晋级。
当前[D2低噪声对照](../docs/RGB_DIFFUSION_D2_K003_20260923.md)入口为
`train/rgb_restoration_diffusion_d2_k003_all60_monitored.yaml`，只改kappa为0.03。
已按`--stop-after-epoch 20`完成首段，保持原60轮学习率日程，20轮/5000更新后正常退出。
[D2分析](../docs/RGB_DIFFUSION_D2_ANALYSIS_20260923.md)确认输出近乎原图，暂不晋级或继续。
[D3起点强化](../docs/RGB_DIFFUSION_D3_TERMINAL50_20260923.md)入口为
`train/rgb_restoration_diffusion_d3_terminal50_all60_monitored.yaml`，仅新增
`train.timestep_sampling: terminal_half`（t=16占50%，其余15步均分50%）。
已从零完成`--stop-after-epoch 20`，kappa仍为0.03，原60轮学习率日程与48张过程图完整。
[D3结果](../docs/RGB_DIFFUSION_D3_ANALYSIS_20260923.md)显示首步恢复有效，完整16步仍增加梯度误差；
[逐步与空间模糊诊断](../docs/RGB_DIFFUSION_D3_TRAJECTORY_20260923.md)已完成。
[D4空间增强](../docs/RGB_DIFFUSION_D4_SPATIAL_20260923.md)入口为
`train/rgb_restoration_diffusion_d4_spatial_all60_monitored.yaml`，通过`--fork-spatial-from`
从D3 e20完整状态启动到总60轮；原D3也已续到60轮作为同预算对照，两组均已完成。
[结果](../docs/RGB_DIFFUSION_D4_ANALYSIS_20260923.md)：空间合成诊断明显改善，均匀模糊小幅退步，
多步采样问题仍在；D4默认输出短名为`outputs/rgb_restoration_diffusion_d4`。
原配方续训保持原配置，不能把总轮数改成40；新配方分叉严格限制只有空间模糊变化。
短链监督仍未实施。
旧`all60.yaml`保留原800次短测预算；延长必须用`--extend-overfit-from`
并指定新的输出目录，不能绕过严格配置检查。

2026-09-21：[复赛交接](../docs/HANDOFF_SEMIFINAL_20260921.md)。本轮未更改测试数据路径或默认推理配置；
复赛数据目录待下轮核验。以下成绩及68图包均为初赛记录，复赛不得直接沿用固定文件数断言。

2026-09-17：S用户回报官方`0.8393/0.8714/85.535`，为当前总分最佳候选；实际为S语义+L旧GT几何，S+V尚未组合核验。
V已获用户回报官方`0.8430/0.8499/84.645`，接受省去V6独立边界训练，作为后续简化几何基线。
L（E10a/top2，官方`0.8416/0.8538/84.770`）保留成绩回退及当前S语义实验的固定参照。
`train/affinity_geometry_g2_direct_long_newgt.yaml` 继承 L 的120轮配方，只覆盖26张人工训练图的
补缝 GT，原人工验证loss选优不变；见 [N实验记录](../docs/G2_LONG_NEWGT_20260916.md)。
N已完成120轮及固定六图部署比较，用户回报黑盒`0.8415/0.8476`（按上下文归属N），折算84.455。
新GT语义实验已完成，固定L几何的内部代理未获益，但官方总分提高0.765；面积稳定性与GT独立贡献仍待核验，默认推理文件未切换。
S-align已回报`0.8407/0.8620/85.135`，接受正确对齐作为后续语义研究起点；
V-noSAM2回报`0.8362/0.8181/82.715`，不接受取消64源图SAM2监督。
2026-09-19组合入口`experiments/affinity_semantic_aligned_v_top2_20260919.yaml`已完成推理和68图打包：
S-align e13 + V e115，top2/high0.65等保持不变；用户回报官方`0.8416/0.8569/84.925`。
该组合成为后续统一研究对照，原S的85.535仍为最高分回退；未自动修改默认部署入口。
见[统一推理与审计](../docs/UNIFIED_ALIGNMENT_AUDIT_20260919.md)。
用户随后选择先执行[Stage1最终任务替代](../docs/STAGE1_FINAL_TASKS_20260919.md)：
`train/stage1_final_tasks_20260919.yaml`联合适配LoRA，接
`train/stage1_final_geometry120_20260919.yaml`和`train/stage1_final_semantic20_20260919.yaml`。
串行入口为`tools/run_stage1_final_tasks.py`，正式训练已完成6600次实际更新；按loss选中联合e35、几何e57、语义e14，不引入EMA。
部署配置为`experiments/stage1_final_tasks_top2_20260919.yaml`；官方回报`0.8448/0.7699/80.735`，
比统一组合84.925低4.190、比原S低4.800，拒绝晋级。保留该配置供诊断，不修改既有基线。

本轮入口为 `train/stage2_semantic_gt_aligned20.yaml`（仅开启共享空间坐标，20轮/1240次计划更新）与
`train/affinity_geometry_g2_no_sam2.yaml`（取消SAM2、人工重采样补至52次/轮，120轮/3120次计划更新）。
两项完整训练、固定部署与用户回报黑盒均已完成；保留语义修复和V/SAM2几何基线，见[实验记录](../docs/ALIGN_NOSAM2_EXPERIMENT_20260917.md)。
配套 `experiments/affinity_semantic_aligned_l_top2_20260917.yaml` 固定L几何；
`experiments/affinity_no_sam2_top2_20260917.yaml` 固定E10a语义，不合并两个训练变化。

`train/stage2_semantic_gt_new20.yaml`与`train/stage2_semantic_gt_control20.yaml`为S配对实验：
两组继承E10a冷启动、冻结共享特征与固定教师，各20轮，实际25训练/7验证，按共同新GT验证loss选优。
仅训练GT与产物路径不同；共同关闭GT依赖暗边增强、AMP初始scale=256，记录实际更新/跳步。
两组均完成1240次实际更新，loss-best为新GT20/旧GT7。
`experiments/affinity_semantic_newgt_l_top2_20260916.yaml`固定新语义第20轮、L best115及top2，
用于用户授权的68图提交核验；不叠加V几何或改动后处理。
见[S实验记录](../docs/SEMANTIC_NEWGT_20260916.md)。

`train/affinity_geometry_g2_skip_v6.yaml`为V消融：继承L，使用joint-v3参考权重与
`reference_boundary_fpn`初始化，旧GT/预算不变。部署对照入口为
`experiments/affinity_skip_v6_top2_20260916.yaml`；120轮完成，loss-best为115，内部总体接近L且部分指标略好，
珠光体多余预测略增。V官方总分较L低0.125分，接受以此取舍减少独立阶段；见[V实验记录](../docs/G2_SKIP_V6_20260916.md)。

历史部署基线为 `inference/final_affinity_g4b_high065.yaml`：V6 语义锚点 + G4b 8 通道
affinity，使用 `high=0.65`、seal2、局部重建与受阻分水岭。训练 checkpoint 只能按固定的
完整部署路径验证晋级，Oracle GT 前景重建仅作诊断。完整协议见
`docs/AFFINITY_DEPLOYMENT_EVALUATION.md`；V6/B2 配置仍保留为回退。

黑盒确认的 E9 语义实验为 `train/stage2_semantic_e9_highres20.yaml`：以 V6 语义为零漂移锚点，
冻结 semantic FPN/head、boundary、LoRA 与 G4b affinity，只训练 256→512→1024 的高分辨率
residual。训练增加整实例平均概率目标和温和细实例权重，不依赖中心或最高置信像素。部署配置为
`experiments/affinity_g4b_high065_semantic_e9_highres.yaml`，hard-majority 对照在同名 `_hard`
配置；详见 `docs/SEMANTIC_EXPERIMENT_E9_20260828.md`。E7b/E8 保留为历史对照。

`train/stage2_semantic_e10a_cold20.yaml` 是完整语义解码器冷启动实验：保留并冻结 V6 LoRA
特征及 G4b 几何，随机重置 `seg_fpn`、`seg_branch` 和高分辨率语义路径；重置前复制的固定
V6 教师只在高置信无标签像素提供衰减蒸馏。部署配置
`experiments/affinity_g4b_high065_semantic_e10a_cold.yaml` 已获黑盒 mIoU `0.8381`、面积项
`0.8408`、总分 `83.94`，是当前单语义模型主线；E9 保留为历史回退，不执行连续融合。详见
`docs/SEMANTIC_EXPERIMENT_E10A_20260828.md`。

`tools/run_affinity_graph_ab.py` 是 GT-free 历史几何筛查工具：graph-v1
`short=0.40/area200` 黑盒总分为 `83.17`，未超过 E10a watershed；graph-v2 `area150`
出现不自然的笔直边界，已按目检淘汰，不再提交。详见
`docs/AFFINITY_GRAPH_AB_20260828.md`。

`train/affinity_geometry_g7_highres_short.yaml` 现只保留为历史对照：固定协议测试 A/B 显示其
相较 G4b 进一步减少实例、加重欠分割风险，不再作为当前晋级目标。

`train/direct_ssl_semantic_affinity.yaml` 是当前短训练链候选：从 SSL LoRA 同时冷启动 E10a 式
高分辨率 semantic head 与 8 通道 affinity head；先冻结 LoRA 预热，再联合微调。人工样本在
同一增强下监督两头，经人工审核的 SAM2 候选只监督无类别 affinity；完整契约见
`docs/DIRECT_SSL_SEMANTIC_AFFINITY.md`。首轮 Arm A 使用同目录下的 `_no_sam2.yaml`，先隔离检验
最短链路；SAM2 数据审核完成后再运行原配置作为 Arm B。

当前类别纠错候选为 `experiments/affinity_g4b_high065_semantic_dual_e7c_relaxed.yaml`：固定
V6 前景与 G4b 实例几何，仅用 E7b core 分数覆盖部分 V6 hard vote。阈值由缓存置信度扫参
产生，不按实例面积拦截；严格版配置继续保留为反面对照。详见
`docs/SEMANTIC_EXPERIMENT_E7C_20260828.md`。

`train/affinity_geometry_g4_manual_gap.yaml` 是历史断边合并单变量实验：完全复用 G3 的
G2 初始化、数据比例、增强、学习率和 20 epoch，只对人工 LabelMe 样本启用
“实例与未覆盖带之间为负 affinity”；SAM2 未覆盖区和人工 `0-0` 像素对继续 ignore。
设计与判定标准见 `docs/AFFINITY_G4_MANUAL_GAP.md`。G4 完整权重已证实过强；
`train/affinity_geometry_g4b_gap_weight020.yaml` 只把新增人工缺口负边降权至 `0.20`，
其余设置不变，产物 G4b 现作为部署几何基线。

- `default_config.yaml`：可训练、可推理的当前 V6 参考基线；路径跨本机/服务器可移植。
- `inference/`：只改变推理输出与后处理参数，不改变模型架构。
- `train/`：明确区分 Stage 1 与 Stage 2 的可训练参数和输出目录。
- `experiments/`：会改变架构、监督目标或训练策略的实验。
- `stage2_center_heatmap.yaml`：旧命令兼容的完整历史快照；已判废，不作为主线。

配置可用 `_base` 递归继承。`paths.project_root: auto` 默认定位当前仓库；临时覆盖可设置
`SEGMENTATION_PROJECT_ROOT`，无需为 Windows/Linux 分别维护 YAML。

推理会严格比较配置和 checkpoint 的 `boundary_refine`、`center_head`、LoRA 等架构字段。
只有明确进行消融时才使用 `--allow-architecture-mismatch`。

历史 B2 边界主线使用 `train/stage2_refine_v6.yaml`：从 V6 best 初始化，只增加独立高分辨率
refine residual，关闭中心头并保持原后处理不变；当前不再把它描述为唯一主线。

当前建议的下一轮单变量实验是 `train/stage2_refine_v6_physaug.yaml`：继续从 V6 best
初始化 B2，但在 5 个 epoch 内只训练 refine head，关闭无标签一致性，加入显微成像物理增强。
增强每次只抽取 1~2 项（曝光/白平衡、失焦、降采样、低频照明或低对比划痕），不制造
圆形硬遮罩，也不改 GT。

`inference/b2_quality_aware.yaml` 是配套的低复杂度推理实验：固定几何 TTA，并按当前单图的
亮度、对比度、清晰度和偏色分为 `standard`/`weak` 两档。弱档只融合一张确定性增强视图并
应用固定的小幅边界阈值偏移；不读取跨图统计，不按实例数、平均面积或环形拓扑闭环调参。
所有推理配置均要求 `max_instance_id <= 65535`，最终实例 PNG 必须以单通道 `uint16` 写出。

`train/stage2_refine_v6_stage0_control.yaml` 是物理增强消融之前的 E0 可学习性控制：
固定 seed 42、每 epoch 62 个监督 step、共 5 epoch（310 次更新），关闭无标签流和
物理增强，只训练零初始化 B2 refine。运行指标会额外记录 refine 梯度/残差/权重变化，
并验证 coarse、语义与冻结 LoRA 的最大参数变化严格为 0。

`train/stage2_refine_v6_stage0_long.yaml` 将同一控制实验延长至 20 epoch/1240 次更新，
前段保持 refine LR `5e-5`、末段衰减至 `2e-5`，每 5 epoch 保存一次 checkpoint 和
monitor。验证指标额外记录边界正/背景概率、概率间隔，以及阈值 0.35 下的召回与背景
假阳性率，用于区分真实边界增强和雾状背景同步抬升。

`train/stage2_refine_v6_stage0_continue15.yaml` 从 Long-20 的最佳 checkpoint 初始化，
继续 15 epoch 纯 refine 训练。LR 从 `2e-5` 平滑接续并衰减至 `5e-6`，仍冻结语义、
LoRA 与 coarse boundary；用于确认 Long-20 末端尚未收敛的收益能否继续，同时避免把
联合解冻引入为第二个实验变量。

`train/stage2_refine_v6_e1_physaug15.yaml` 从 Continue-15 best 初始化，在纯 refine 已进入
平台期后进行 15 epoch 物理外观增强实验。只训练 refine head，LR 从 `1e-5` 衰减至
`2.5e-6`；增强保持 40% 干净样本，每张增强图只抽取 1~2 项显微成像退化，不修改 GT
几何，也不使用规则硬遮罩或高斯噪声。

`train/stage2_refine_v6_e2_coarse_unfreeze10.yaml` 从 E1 best 初始化，保持同一增强和损失，
进行 10 epoch 低学习率联合边界训练。refine LR 从 `5e-6` 衰减至 `1.25e-6`，coarse
boundary 始终使用其 5%；语义与 LoRA 继续冻结，用于隔离 coarse 表征适配的收益和风险。

`train/stage2_refine_v6_e3_ridge10.yaml` 回到 E1 best，并继续严格冻结 coarse boundary、
语义与 LoRA。唯一实验变量是局部边界脊线损失：允许 GT 附近 1px 定位误差，
要求核心附近存在高置信峰值，同时抑制 5px 邻域真背景的雾状响应。保持 E1 物理增强，
训练 10 epoch，用于单独验证“窄、亮、连续”边界监督。

`train/stage2_refine_v6_e3b_balanced_ridge10.yaml` 是 E3 的置信度校正实验，仍从
E1 best 独立初始化。正边界峰值目标提高至 logit `2.0`（概率约 0.88）；
背景环只抑制高于 logit `-0.62`（概率约 0.35）的响应，且权重降为 0.25。
其余训练路径与 E3 相同，用于验证能否保留背景误报收益并恢复高置信边界。

`train/stage2_refine_v6_e4_relative_ridge10.yaml` 改用局部相对脊线损失，只要求
GT 附近的边界峰值比 5px 内最强真背景高 `1.5` logit。该损失对全图统一
加减 logit 严格不变，不能像 E3/E3b 一样通过整体变暗或变亮来获利。
仍从 E1 best 开始，其余训练和物理增强保持不变。

`train/gda_mim_g0a.yaml` 使用赛方无标签图进行生成式掩码重建预训练。
冻结 E1 SAM2/LoRA，只训练四尺度 GDA 和临时重建解码器；预训练后丢弃解码器。
`config/monitor/unlabeled_holdout_v1.txt` 中的 24 张图不进入训练，专用于固定无标签 monitor。
