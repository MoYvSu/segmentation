# Candidate observations

This file holds potentially reusable findings that still need validation or
scope clarification. It is not a chronological task log.

## 20261031-boolean-mask-before-large-array-indexing

- status: candidate; last_verified: 2026-10-31.
- scope: 私有GT来源分层统计；NPZ的original_covered/filled/residual_unknown为uint8。
- finding: bool核心与uint8按位与仍为uint8，`gt[mask]`变成整数行索引；1936×2584掩码展开到1936×2584×2584，uint16约24.08GiB，再astype uint32约48.16GiB，足以压满62GiB容器。本轮train001统计停滞、内存98%，恢复后NPY头与3×4微型例确认索引错误；9保护文件SHA一致，训练和独立配对不受影响。
- evidence: `docs/NAF_PERTURB_ANALYSIS_20261031.md`及私有incident的原脚本、微型复现、恢复header/cgroup/哈希回执；未取得现场PID、RSS峰值或内核OOM日志。
- reuse: 索引前显式np.bool_并核同shape二维，原标/填补/未知字段不做含义不明的fallback。CPU大图统计与GPU推理解耦、限制工作进程内存并留PID；容器按cgroup而非宿主机/proc/meminfo判预算。本轮只验证类型修复，未恢复大图统计。
- verification_gap: 不将所有高RAM都归因此错误，不把重启当实验增益，不因一次事故晋升AGENTS。
- followup_bounded_GPU_diagnostics_20261031: 固定batch warm后的GPU worker VMS约12.79GiB而RSS峰约1.30GiB，不能沿用CPU统计8GiB RLIMIT_AS。实际采用warm后额外2GiB地址增长＋外部RSS4GiB/cgroup12GiB/180秒守卫，4源8ROI流式完成；CPU复票13源峰162.4MiB、无torch。退出瞬间/proc可先失去RSS而waitpid尚未回收，v1曾把exit0误报监视失效；v2有界重读并再次poll，仍保留原限额，独立8项Linux微测通过，未重跑GPU。证据`docs/NAF_CHECKS_20261031.md`及私有guard回执；这是诊断资源合同，不是模型准确率或所有CUDA环境的统一内存值。
- followup_training_warmup_20261031: 同环境训练不能直接套仅前向的GPU地址预算。双输入NAF在batch16前向后加8GiB RLIMIT_AS仍于首个反向报CUDA OOM，当时显存空闲22.4GiB、RSS约4.28GiB，VMS接近该上限。原失败/已通过169最终桥接保留；在RSS8GiB/cgroup16GiB守卫不变下，完整G/D前后向和Adam预热66步后才设VMS增长限，80步短测及两路下一更新/优化器exact通过，容器峰约9.36GiB。源缓存只读共享，避免train/fixed重复驻留。证据`docs/NAF_DUAL_20261031.md`与私有`naf_dual/{preflight_bridge,preflight}`；只支持本配方的阶段性预算，不能把任意CUDA OOM都归于地址上限或推广固定GiB值，不晋升AGENTS。

## 20261030-same-predicted-P-is-not-instance-sameness

- status: candidate; last_verified: 2026-10-30.
- scope: 同源C末端只将相邻预测P连成整组，F像素/ID/类别及面积完全不变。
- finding: 150图P5422→3813、FP净少1488，但TP净少121，mIoU .7461→.7436，面积仍.9070。precision约+7个百分点不能当作新实例评分增益；按同类匹配，此次变化应定位于P匹配。
- evidence: `docs/PEARLITE_MERGE_20261030.md`官方回分节，包SHA588aa7f3…62d7；用户报分与总预测14062、GT13068核对吻合。
- reuse: 区分相邻预测同类与同一GT实例；把全连通组合并与一次关系配对区分。净TP/FP差不能逐对标成合并对错，尤其不能由1488/1609估计正确率。
- verification_gap: 无逐图GT、类别匹配清单或重复回分，无法确定误判桥接/链式合并/GT粒度的分别贡献；不否定所有P去碎方法，不作统计显著声明，不晋升AGENTS。

## 20261030-frozen-RGB-perturbation-is-not-line-erasure

- status: candidate; last_verified: 2026-10-30.
- scope: 原生RGB大范围平滑模糊，经固定旧D5a/N1后，作为单通道T8 NAF训练输入。
- finding: 增大RGB模糊不只会删弱线。32源原标区域523023个可信GT正点中新缺口3838、旧缺口恢复2350，内部T8输入像素约+7.60%；损伤与新增响应并存。统计需分列原缺口、新缺口、恢复旧缺口和内部输入，不用净缺口数或名义sigma判断难例是否充分。
- evidence: `docs/NAF_PERTURB_20261030.md`、私有preflight/diagnostic32；6源清晰输入经真实冻结前端逐窗口及整图B exact。共同初态/16000采样计划/GT和首个clean批损失也exact旧控制，保留原loss即可直接监督新增缺口。
- reuse: 区分RGB退化宽度与后端响应宽度；保持整图扰动再裁窗，防止重叠窗出现人为增强接缝。训练访问日志的valid-GT计数与coverage进一步腐蚀后的trusted分母不能混用。
- verification_gap: 32源参加训练且GT不完整，内部响应并不都是真假边界；80步固定批可学性不是泛化。正式候选刚启动，无官方增益，不晋升AGENTS。

## 20261030-single-zero-merge-is-not-global-idempotence

- status: candidate; last_verified: 2026-10-30.
- scope: 基于原实例图一次性填一个零像素的P邻接连通组合并。
- finding: 每次调用只填原始一零桥，不等于连续调用仍保持同一邻接范围。第一次填出的P像素可令第二次形成新的桥；循环到稳定会改变实验定义。
- evidence: `tests/test_pearlite_merge.py::test_single_pass_does_not_iterate_newly_filled_zero_into_another_bridge`，7×9零图中令[3,2]=1、[2,3]=2、[3,5]=3且均P，首次只合1/2，第二次又连3。见`docs/PEARLITE_MERGE_20261030.md`；无测试GT。
- reuse: 缓存回放与部署入口均只调用一次。若未来研究迭代，应明确算另一项逻辑变更，而不是无害的重复调用。
- verification_gap: 不涉及准确率或官方收益，不泛化成所有形态合并都非幂等，不晋升AGENTS。

## 20261026-ph-matching-and-valid-origin

- status: observation
- last_verified: 2026-10-26
- scope: 处理后GT可靠域上的二维cubical有限H1监督；不涉及测试GT或最终实例数约束。
- finding: 自由持久图Wasserstein配对可能把真实弱环送对角线并接受漏GT的代价，
  得到压低缺口的反向梯度；必须用能分辨“保留真实结构／删除假结构”的合成概率场测试，
  不能只测试有限loss或Betti计数。当前GT数量约束匹配在gap0.2／背景0.05弱环上给出
  抬高缺口的实际梯度；这是受限适配，不保证空间逐实例对应。
- followup_valid_origins: valid有4px裁边忽略时，从0起的stride网格会让224窗全部被拒绝，
  尽管[4:252]内实际有合法窗。用积分图核所有合法起点，再按合法域边界／中心取有限代表点，
  可恢复覆盖且不把unknown当背景。大窗、分离合法域、非方形索引与步数轮换已测。
- evidence: `utils/boundary_topology.py`、`tests/test_boundary_topology.py`及
  `docs/BOUNDARY_NAF_20261026.md`；真实32配对预检与合成梯度测试分别报告，不混作泛化收益。
- limits: 只检查有限H1，平坦预测没有有限持久点时不能伪造补环梯度；需记录缺失目标。
  合法候选分母与网格版本不同，不直接比较通过率；不晋升AGENTS或据此晋级部署。

## 20261024-cache-inventory-not-dependency-closure

- status: observation
- last_verified: 2026-10-24
- scope: 双路线工作区的历史缓存冷存、控制组复用与路径身份。
- finding: 一份已核验文件清单不等于完整运行依赖闭包；清单外的缓存仍可能被后续步骤读取。
  同样，history导航入口不证明整个旧输出目录都需要常驻快盘。冷存应结合封存源码、实际输入协议、
  缓存别名与文件清单；不要仅按大小、日期、未出现在清单或无绝对路径字符串引用来删除。
- evidence: `docs/STORAGE_MAINTENANCE_20261024.md`及私有
  `output/storage_cleanup_20261024/dependency_cleanup_audit.json`。
  proto/fast的688MB fit_pool不在原3402项保护集，但phase重建会读取，故保留完整fast。
  原Affinity控制封存源码在线从RGB生成监督，不依赖旧CV v1；R2/full显式绑定独立v2缓存。
- reuse_conditions: 下一次空间清理先核对当前依赖；闲置载荷可在逐文件校验冷备后离开快盘，
  需要历史重跑时先恢复到原路径，不以指向fs的链接静默改变Path.resolve身份。
- limits: 本次结论仅覆盖选定缓存和当前协议；不是全部历史实验均可即刻重跑的证明，
  也不是自动清理规则。不修改AGENTS.md或旧不可变回执。
- followup_archive_namespace_20261030: FS暂存清重中13个旧checkpoint的manifest成员名省略真实tar的`tree/`前缀；先保留疑项，限定读取约590MiB解压前缀才找到全部实际成员并核SHA，未据文件大小或旧清单删除。归档容器存在／清单记录相同SHA不代替实际成员可读回证据；成员命名需按真实header绑定。证据`docs/STORAGE_MAINTENANCE_20261030.md`及私有FS追加核验回执。567项TMP冷存和4442保护项前后复核只覆盖本轮范围，不晋升自动清理规则。

## 20261005-cuda-randn-device-layout

- status: observation
- last_verified: 2026-10-26
- scope: PyTorch2.8／CUDA12.8、FP32连续1024 RGB、新建CUDA generator且offset0的D5a首步推理。
- finding: 同种子不保证跨GPU得到同一大张量噪声。CUDA normal4写入grid受SM数量影响，
  会改变恢复器实际输入；不应将跨卡差异默认全部归为舍入或模型降精度。
  阈值与分水岭可将连续边界差异放大为区域变化，进而改变相同raw语义概率的实例内投票。
- evidence: `docs/INFERENCE_REPRO_20261005.md`及私有`output/final_delivery/runtime_probe/`。
  五图各三次独立前向的实例／概率／噪声逐值一致；关闭TF32和数学注意力未消除差异。
  本机G156随机内核在三种seed、两个编译目标下逐值复现原生torch.randn；模拟4090的G768后，
  五图类别像素差从0.197%–1.564%降至0.002%–0.026%，F数量相同、预测F均面积最大差约0.003%。
  固定实例图交换原／本地raw语义概率时test101类别变化0，支持几何范围驱动该例翻转。
  正式接入后，虚拟布局与实际线程配置解耦，21组shape/seed、105种执行网格逐值一致；
  compute70/75/89/120在同一5060上的三seed检查一致。五图两进程十次正式输出逐字节重复，
  且复现诊断原型最终结果；已有训练默认保持legacy，正式入口显式启用版本化策略。
- followup_linux: 随包compute70 PTX现已在Linux RTX4090实测；三个完整1024 RGB噪声张量
  （seed314159、314191、−1）与Windows RTX5060 Laptop的SHA完全相同，4090上还与原生
  fresh CUDA generator逐值一致。固定协议解决的是输入噪声复现，不等于网络对不同噪声图案鲁棒。
  证据见`docs/RESTORATION_NOISE_20261007.md`及私有`output/noise_probe/kernel_*.json`。
- followup_noise_sensitivity: 六源同卡八种子隔离检查中，局部恢复平均差仅约0.35–0.73灰度级，
  四源仍改变F数量；固定末端raw概率也能因分区变化造成实例投票翻类。六源control与原最高分
  PNG数组／类别完全相同，模型／源码冻结通过。重建误差、噪声复现、终态稳定性需分别验收；
  不把图像近似不变或相同类别计数当作完整几何稳定，也不把相对控制的变化当作准确率。
- followup_consistency_bias_variance: 同e60起点、1250更新且逐步数据／噪声完全配对的恢复器短适配中，
  额外双噪声一致性使四个已见训练源的原生1024模糊输出像素std比控制低约25%–27%，
  但弱图像梯度方向保留少约0.7–1.1个百分点。八种子经验MSE拆分中，全部模糊条件的
  均值偏差项都略升；有的总MSE微降只来自方差下降。稳定性、偏差与细节需分别验收，
  更低随机误差不能自动称为更准重建。见`docs/RESTORATION_CONSISTENCY_20261007.md`及私有
  `output/noise_consistency/quality8/`；四源均参加过训练、非独立尺度因果或官方准确率证明。
  六测试源八种子仅替换局部恢复器后，B比共同适配A的连续边界std低15.7%，但最终F均面积
  相对波动旧／A／B为0.885%／1.000%／0.905%，且预测均值也漂移。连续稳定不保证
  离散分区更稳；需隔离平均几何变化、方差及固定语义概率下的实例投票影响，不据此晋级。
- followup_zero_mode_cross: 相同e60起点、1250配对更新的零噪声短适配C，在双方都用零噪声首步时，
  四个已见源的原生1024均匀／空间MSE比带噪适配A低3.95%／2.06%，弱图像梯度保留略升；
  但给C恢复旧噪声推理时，模糊输出std变为A的约2.25–2.87倍。训练噪声与推理噪声应交叉比较，
  仅换权重而保留旧随机输入可能得到相反结论。零输入的确定性是构造性质，不是质量证明；
  零输入输出也不等于多噪声输出平均。此为带噪预训练后的短适配，不能证明从零训练的噪声无用。
  证据见`docs/RESTORATION_ZERO_20261007.md`与私有`output/noise_zero/quality/`；非黑盒晋级依据。
- followup_zero_extension: 同配方由1250接续到2500更新后，同零噪声C仍优于A，但C5→C10的512均匀
  MSE低1.132%，原生1024空间MSE却高2.876%且四源全差，弱图像梯度保留少0.469个百分点；
  A原生空间MSE也高2.776%。训练视野改善不能代替实际部署视野检查，也不能把共同回升归因于
  零噪声独有缺陷。两套视野裁块／退化不同、均为已见源，非纯尺度因果或独立泛化验证。
  六图固定后端未呈现明确额外收益；证据见`docs/RESTORATION_ZERO_E10_20261007.md`。
- followup_zero_acceptance: C5局部零噪声在4090／5060各两独立进程、三源终态全部同卡精确，
  实际逐窗噪声／state核验及跨卡预设工程容差通过；旧固定Philox同样通过，部分最终指标残差仍略小于C。
  去除种子自由度不会消除GPU浮点残差，也不自动证明质量更好。两已见GT源的可信边界仅7.71%／2.91%，
  C模糊法线误差比共同适配A略降但弱点半参考响应覆盖不变、清晰漂移略增；GT锚定梯度误差小降
  不等于补回更多边界。全图语义缓存固定，未覆盖完整包重算或任意GPU，见`docs/RESTORATION_ACCEPT_20261007.md`。
- followup_third_gpu: RTX4080 SUPER 32GB的原D5a固定Philox三源两独立进程同卡全精确，20窗实际噪声、模型状态、首窗输入及核心精度与旧4090一致；跨卡恢复MAE仍约5e-6至1.26e-5，连续边界MAE约4.7e-5至5.9e-5，最终三态像素差0.261%至0.494%，其中一源F数208→209、预测F均面积变动−0.4405%。这不是新噪声布局不一致；同输入噪声不保证恢复/后端浮点计算与离散分区跨卡一致。未将全部差异单独归因D5a，未测C5第三卡、不据预测间差异判准确率；见`docs/VGPU_CHECK_20261024.md`和私有`output/vgpu_check_20261024/output_comparison.json`。完整portable三图两进程同卡文件精确，不推广150图或任意设备。
- followup_official_local_zero: 用户回报半决赛150图N0/N1为0.7433/0.9025与0.7439/0.9032，
  后者仅取消局部推理噪声，原权重不适配，等权+0.065分，未见保留局部噪声的得分必要性。
  C5局部零噪声N2为0.7412/0.9062，较N1总分仅+0.015、净TP−33/FP+489；
  更低已见源重建误差或更稳定输入不能自动推出更佳实例匹配。N2对N1净增456预测中428为P，
  因此总FP不能全归F碎块，P也会影响实例项。分数来自用户，ZIP SHA和TP+FP总数本轮直接核验；
  见`docs/SEMIFINAL_NOISE_SCORES_20261026.md`。仅支持这次局部消融，无全图去噪或显著等效证明；
  当前绑定N0的NAF不换控制，最高分包保留与后续研究控制选择分开。
- reuse_hypothesis: 跨设备复现先比较实际随机张量、恢复输出、连续概率与最终空间配对，
  再检查计算精度。需要复现旧输入时记录参考随机布局或随机张量，不直接换CPU RNG／零噪声。
  维持原已评分精度设置；更严格FP32可能改变微小实例组成，不能假定更接近原链。
- verification_gap: 旧五源跨卡与六源多种子仍仅为机制诊断；新增同赛段150图局部消融回分见上条，
  不能补足这些跨卡检查的完整性或推成全链去噪结论。
  已补Linux4090及第三款4080 SUPER的固定噪声/有限终态检查，仍无跨卡完整100/150图检查，第三卡未达到此前严格终态工程参考线；依赖支持CUDA12.8的驱动及架构兼容的Torch。
  不承诺跨任意GPU逐位一致，不推广到任意offset／dtype／随机算子。
  不晋升AGENTS全局规则，不改GT或已验证部署权重。

## 20261004-semantic-region-label-granularity

- status: observation
- last_verified: 2026-10-04
- scope: 当前raw／S-align末端、冻结几何与特征，同32已见源的原目标／附加预测区域均值监督，各64更新。
- finding: 父GT类别与覆盖都一致，并不保证其内部细分预测区域的物理相被单独标注。
  两处亮岛完全位于原始P多边形，原始F多边形重叠为0，处理后original覆盖逐值复现；更符合父P类别不能直接称物相纠错。
  同初态／输入的新增目标确实可学，但只改善小片时能同时牵动F平均面积，不能因GT分歧减少就扩大训练。
- evidence: `docs/SEMANTIC_OBJECT_FIT_20261004.md`；182区域GT分歧A49／B25，B多26符合GT、另退2；
  B原目标同源第二遍升6.18%、区域项降42.45%，固定4918匹配错1→3；8测试仅小片改类，F数532→636、聚合均面积−16.19%。
  私有原标逐多边形查证、相同PNG及完整复核在`output/semantic_object_analysis/`。
  后续`docs/SEMANTIC_NATIVE_20261004.md`冻结原生尺度与分支贡献检查均精确回放当前部署：裁块只改小片，
  266测试大对象整区类不变；当前高分辨修正减少34.3031%已见原标域像素分歧，不能因整块类不变就称支路无效。
  32源同父GT溯源找到小片判P遮蔽额外F计数的局部抵消，统计改正严格64片后预测聚合均面积仅−0.5091%。
  后续完整父GT裁窗的64窗预检中，21个有效像素学习信号全为P，锚点原标梯度中位4.89%、
  裁断邻居72.88%，整对象及核心类别仍全对。门槛通过只证明局部任务可学；
  即使裁窗完整保留锚点，共同损失仍可能主要学习邻居，不能直接升为整粒错相的针对性证据。
  少数P内部亮片层的核心概率确实降低，亦不能把信号全部称边缘误差；见`docs/SEMANTIC_GRAIN_20261004.md`。
  随后同源64更新混合视野使裁窗P错误相对旧A64从1.8144%降至0.6629%，却使整图P错误相对R1
  从0.5674%升至1.4126%；固定过程支持裁窗仍学习、整图漂移未消失，局部任务拟合不能替代部署视野验证。
  同一实例PNG上八测试F数532→693、像素只增0.570%、聚合均面积减22.80%，主要是既有小片改相，
  不得误称新产生碎分割，也不是官方面积得分；全部改类小于10000px，不能据此证明大块语义改善。
- reuse_hypothesis: 区分GT一致性、完整实例语义与真实局部物相；检查父标注粒度和原处理来源，保护旧正确对象，
  同时报告小F数量及面积组成。不要把源coverage或记录类纯度当作局部物相确定性，不用测试外观补标签。
  聚合面积可能含几何与类别的共同偏差；样本多不能排除系统性抵消，局部抵消存在也不能估计隐藏分贡献。
  固定PNG上的大块类别不变与像素／核心概率改善可并存，应分开核算，不能仅凭前者删除修正分支。
- verification_gap: 当前仅已见图、短预算、两个原标例和八无标签测试源，无官方增益或泛化证据；
  不能断言亮岛一定为F、GT普遍错误、训练已饱和或所有区域监督无效。不晋升全局规则，不改GT。

## 20261001-cuda-repeatability-gate

- status: observation
- last_verified: 2026-10-01
- scope: 当前FP32 affinity短测，同GPU／源图／种子／初始化，未启用确定性反向算法
- finding: 原mixed路径自身复放8步也不能逐位复现历史权重；首步loss及监督一致，梯度范数差一个
  FP32末位，后续loss最大偏差约2.10e-6、梯度范数约5.67e-5。零辅助新入口的偏差同量级。
  应区分输入／冻结模块精确相等与反向更新数值兼容，不能把历史bitwise失败报告为逐位通过。
- evidence: `docs/AFFINITY_PAIR_20261001.md`；私有`outputs/pair_setup/legacy8`、
  `outputs/affinity_balance_smoke/mixed`及首份零辅助短测的实际8步回执。
- reuse_hypothesis: 若零变化训练轨迹未逐位一致，先复放旧实现测其重复性，再用明确且有限的数值
  门槛检查；来源、监督数量、初始状态、冻结状态和已有权重的最终部署输出仍保持精确门槛。
- verification_gap: 尚未逐算子定位波动；CUDA双线性插值反向是候选，不能认定唯一原因。
  本次8步波动范围不外推长训练或其他环境，也不构成候选泛化／竞赛收益证据。

## 20260928-native-patch-structure

- status: observation
- last_verified: 2026-10-01
- scope: 当前D5a＋affinity的冻结尺度诊断与两组原生裁块适配
- finding: 原图裁窗再送网络与缩小整图后裁窗是不同实验。原全图control的20轮适配先整图letterbox，
  历史阶段有native crop不等于当前恢复输入已适配原生尺度。1024局部可改善已见可信域的匹配，
  512放大同时大幅增加小片；裁窗结果需拼连续图后完整分区，不能以局部实例代替全图输出。
- evidence: `docs/AFFINITY_PATCH_20260928.md`固定4训练＋4测试的48次分区；4张control完全复现。
  追加32对发现D5a平均RGB／梯度误差下降时，局部模糊＋0.5倍降采样的铁素体内部新增强边界
  仍可3777→7422；单纯0.5倍降采样却613→221。不同退化下的结构作用不能混为一个平均数。
- reuse_hypothesis: 恢复与尺度试验同时查看已知内部的错误边界响应、完整最终分区和分类别统计；
  只铁素体进入面积目标，珠光体碎片数量不直接当面积损失。针对裁块训练可复用已有同初始化／
  同预算整图control，另做全图与局部推理交叉对照，避免把训练时长与视野改变混为一项。
- verification_gap: 机制诊断仅小样本、已见训练GT和合成退化；旧control原生1024及两组官方回分已补齐，
  支持同推理入口的训练收益，但尚不能定位各模块独立贡献。内部强响应不是错误实例数，
  不能把512测试分片全部归因于D5a，完成证据见`docs/AFFINITY_NATIVE_20260928.md`。
- followup_native_training: 两组独立20轮／同初始化同预算，100图同尺度512铁素体小片6307→354，
  但四张已见GT的铁素体匹配361→357；1024同尺度365→360。训练可抑制大量碎片而同时牺牲
  部分真分界，不能凭去碎片／轮廓更细认定全面增益。训练后退回全图时测试实例减少、已见GT匹配下降，
  所以必须同时比较旧全图、旧权重原生、新权重全图、新权重原生，分开推理视野与训练收益。
- followup_official: `docs/AFFINITY_NATIVE_REVIEW_20260930.md`核对用户在伙伴会话中的回分，
  p1024为0.8580／0.8568、p512为0.8534／0.8130；1024提高mIoU却未胜control综合分。
  三份100图实际提交包重检后，1024有44图预测均面积上升、56图下降；22张大图全部下降，
  47张另一尺寸组中位反而上升。合并均面积／净实例数会掩盖尺寸分组取舍，不能反推测试GT或合并目标。
  当前512失败也不证明1024以下视野存在理论上限：放大不创造信息，但会改变网络格点映射与上下文。
- followup_control_official: 旧control原生1024补回0.8551／0.8217（83.840）。同权重切换裁块
  推理比全图−2.395分，p1024同入口比旧权重+1.900，支持局部训练挽回部分损失；已见4图上
  旧权重的365个铁素体匹配高于p1024的360，却未预测官方排序，应保持训练内诊断的使用边界。
  只改推理铁素体净增762，其中<200档增164；尺度训练后净减604，小档减56，不能将全部
  面积变化归给极小碎屑。三组均用旧D5a，尚不隔离恢复器、上下文与拼接的独立贡献。
  证据见`docs/AFFINITY_NATIVE_REVIEW_20260930.md`文末及私有`output/native_review/control_score.json`。
- followup_native_restorer: `docs/RESTORE_NATIVE_20260930.md`只替换局部恢复器、固定p1024后端。
  A/B随机初态各20轮配对通过；新增12源96对中，混合视野B在512放大的36个模糊配对RGB／梯度
  误差全部低于旧版，且12个清晰放大配对均减少不必要改动，但弱图像边缘支持下降；固定后端
  已见GT匹配无增益、100图小铁素体区域增加。应拆开插值输入保真、弱边缘恢复、真实输入修复量
  与最终划分，不能用单一重建误差判定尺度适配可部署。Canny小闭合区包含晶内纹理，不能称作
  真实晶粒保留率。新模型20轮仍改善、旧版60轮且共同退化不同，尚不归因尺度单项或断言饱和。
- followup_restorer_e40: 同96配对A的1024弱图像边缘支持已超过旧版，但清晰512放大的额外改动
  比e20增至4.31倍；混合视野B保住插值目标。合成改善没有形成明确最终分割增益，真实输入仍偏弱。
  固定12源模糊后只加2/255独立噪声，A/B相对各自无噪声的原校正方向保留中位56%／58%，旧91%；
  新模型RGB误差仍略低于旧版，不能将校正减弱等同于精度排序。该结果支持检查高频扰动敏感性，
  不证明测试差距全部由噪声造成，也不说明只要加噪声训练就能提分。证据与定义见上述恢复器专题文档。
- followup_native_texture_calibration: 48个预固定原生1024窗中，带限频谱加局部归一未通过已知
  0.5／2倍缩放的方向自检，不可用来估倍率。去线性趋势的全谱代理通过两窗约半／双倍检查，
  测试组中位粗纹理尺度偏大23%–37%，但只加sigma8模糊也能移动约24%，且清晰度／暗占比
  联合匹配为0/16。尺度代理应先做已知缩放和模糊混淆校准；不能把统计域差解释为物理倍率，
  也不能用不合格代理的区间重叠认定尺度一致。范围仅48中心窗、无真实晶粒标签，
  证据见`docs/ROBUSTNESS_SCALE_20261007.md`与私有`output/scale_domain/`；保留候选、不升AGENTS。
- followup_semifinal_fov_20261025: 新半决赛固定原生窗中亮度差仍明显，纹理波长区间重叠且受模糊
  混淆；CV边缘曲折度混入清晰珠光体内部纹理，不能用来验证真实晶粒形状错配。固定N0同中心
  原生768／1024／1280映至1024，在exact复现原窗口／拼接并配对网络噪声后，768通常线带更细、
  中心峰更强；逐像素原强响应跌阈会将线带收窄误作断边，需另查同原图坐标邻域峰和最终分隔。
  116两处弱口的尺度偏好相反；更细更强不足以证明统一倍率或真实补界。本轮只查九源24档局部，
  视野、上下文、网格间距和原图点噪声对应关系同时变化；没有完整实例读出或官方成绩。
  证据见`docs/SCALE_SHAPE_20261025.md`及私有`output/scale_shape/`，保留候选，不升AGENTS。
- followup_mixed_budget: 固定1280更新的全图／1024混合头，各视野只有640更新；拆分相同输入后，
  两边末5轮人工loss仍比对应专用控制高1.26%／2.80%。这既不等于混合无效，也不足以证明梯度冲突，
  因为专项更新预算同时减半。报告混合实验时同时给出总预算和各视野预算，并固定推理入口比较。
  `docs/AFFINITY_MIXED_20260930.md`的同入口全100图及已见GT未证明双尺度共同增益；
  2026-10-01用户回分0.8582／0.8536，较p1024的0.8580／0.8568综合低0.150分。
  数值mIoU微升不等于混合产生附加收益；固定总预算下各视野更新减半仍限制因果归因。
  小于200像素铁素体数下降与相对较小区域净增可共存，不能只用极小片计数认定面积风险消失。
- followup_balance: `docs/AFFINITY_BALANCE_20260930.md`两种采样各独立同预算、同输入重训，
  只改类间权重后，内部连接率提高同时真分界识别下降；去碎片不等于边界辨识全面改善。
  mixed的100图总实例恰好不变，却掩盖铁素体−312／珠光体+312；须分类别看几何与面积变化。
  私有预测重叠检查确认新实例可跨越旧两类实例，由冻结语义统一投票后大块变类。
  所以冻结语义网络不保证最终类别图不变，matched-IoU上升也须伴报有效匹配数。
  同日用户随后回报native balance为0.8639／0.8730（86.845），比同入口p1024两项都高、
  综合+1.105分，比旧最佳全图综合+0.610；四张已见GT未预测收益幅度，不能否定官方改善。
  该结果支持当前配方／尺度的类间权重调整，不推广为更强连接监督必然更好；
  `docs/AFFINITY_EQUAL_RECALL_20261001.md`随后完成32源clean／blur等召回，关系层与严格内部
  融合层的真假边界排序近乎重合，固定阈值下内部假响应降低而真分界保留也下降。
  这支持工作点取舍更合适，不证明训练与改阈值完全等价，也不否定已取得的正式收益。
  mixed balance随后回分87.015，比native balance86.845高0.170；用户提醒微小优势不可靠。
  原loss与balance下采样排序相反，足以要求限定旧结论的适用配方，不足以证明统计交互或混合稳定优越。
  先区分最高单次记录、机制诊断与跨种子稳定性；随后补齐native／mixed的同输入等召回，亦未证明mixed稳定优越。
- followup_pair: `docs/AFFINITY_PAIR_20261001.md`同源互补视野保持主序列和50/50期望贡献，额外曝光后
  20轮主loss仅−0.255%，四张已见GT仍匹配同一批589个、IoU仅微升。重新组织同一监督不是新增目标，
  不能预设配对曝光翻倍就等于新增纠错；也不由该例判定整个affinity饱和。用户随后回分
  0.8629／0.8760（86.945），比mixed balance低0.070，未显示额外收益；单次结果不证明统计等价。
  100图预测铁素体5984→5986、像素+0.0221%，逐图预测均面积中位几乎不变，却有−8.35%～+4.53%尾部，
  来自单图少数实例划分变化。这支持同时检查逐图尾部，不能用净总数相近当推理等价或正式面积稳定。
- followup_fixed_fusion: `docs/AFFINITY_BALANCE_FUSION_20261001.md`同视野两新balance头的固定连续边界
  平均可保住部分相反的轮廓优势，却不能保证保留两端有效实例匹配的并集。四已见原标可信域中
  融合匹配IoU略高于两端、总匹配数相同，但丢掉一个单端独有匹配，共同未匹配仍未解决。
  因此集成检查应记录具体匹配集合、事先固定的优势位置、修复与新增关系，不用平均IoU或净实例数
  代替互补证明。边界平均也不能外推为新增监督；即使后续正式涨分，仍未分离普通集成和视野历史作用。
  用户随后回报固定50/50正式0.8631／0.8739，综合86.850，比mixed 87.015低0.165分；
  已见轮廓互补没有转为总体收益，停止本次比例扩展。测试无GT且没有重复种子，不能外推所有集成无效
  或指定唯一错误来源。回分与已验收包身份见上述专题最新状态。

## 20260928-affinity-target-location

- status: observation
- last_verified: 2026-10-02
- scope: 多距离端点affinity转换为像素边界的解释与诊断
- finding: 长offset负关系只表明两个端点不同实例，不表示source像素恰好位于晶界。理想distance4可有4格负响应，但实际gated-top2融合及阈值可能抑制拖尾，因此不能由通道响应宽直接断言最终边界必然宽。
- evidence: `docs/AFFINITY_NEXT_20260928.md`；实际配置下11种CPU合成ID图，水平／垂直单界的融合高阈值带只有1格，间隔2–8格双界仍分开；两个斜向为1／2格。
- reuse_hypothesis: 先用已知几何经过实际目标构造和融合检查表示含义，再区分网络预测、融合及后处理的误差；不要同时强迫所有距离通道变成同一窄边缘图。
- verification_gap: 仅128网格理想logits与0.65阈值，未走真实GT、原尺寸恢复或完整后处理，未证明当前部署主瓶颈或新融合方案收益。
- followup_flip: `docs/AFFINITY_PAIR_20261001.md`回分后，用规范理想关系检查实际gated／top2融合。
  垂直界经水平翻转再还原标量边界时高响应错开一格，等权平均后均低于0.65；水平界不变。
  所以source锚定的affinity标量不能默认按普通像素分割做朴素翻转平均；应先验证边位置定义及变换对齐。
  私有`output/tta_plan/ideal_flip.json`保留配方与七种合成结果；未运行真实模型／完整后处理，
  不等于TTA普遍无效或证明原部署有错，也不据此修改训练或部署阈值。
- followup_training_target: `docs/AFFINITY_FUSION_TARGET_20261002.md`的五种48格虚构几何CPU检查说明，
  诊断用的“任一短距跨界”为真不能直接成为融合图硬目标1：仅一个跨界关系时理想top2为0.5，
  垂直界另一侧理想gated约0.285714。硬推到1可能与正确关系监督取舍，不能把这一侧低值等同于整界漏分。
  增加融合监督前须检查理想输入自洽与可信池化支持域；后续已做排序的真实梯度及20轮训练，
  不能将其结果外推到尚未训练的融合软目标。
- followup_newgt_support: 用户明确采用处理后的新GT后，核验`manual_target_v2/v1`、radius8的32源
  NPZ／类别表本地服务器字节一致。当前mixed主监督已用此新实例图；本轮排序及空间诊断也从
  增强后的同一实例图与valid_content派生支持，包含补缝，剩余unknown忽略。
  `original_covered`是来源审计，不再默认作为新目标的监督mask；旧保守可信域的GT计数不能直接
  对比完整新GT支持域，须在同口径重算控制诊断。86项CPU检查验证理想输入、unknown保护及
  主BCE＋排序梯度／三步更新；不作为官方收益。证据见上述专题与`tools/verify_affinity_newgt.py`。
- followup_tail_sampling: 同源、同初始化、同1280draw的融合排序训练已完整验收。虽然配置为最难10%
  且最多256点，实际两侧每步都触顶：中位只选择真边界2.655%、内部0.163%的最极端点，
  不等于在10%尾部中代表性取样；whole／native实际尾宽还不同。每轮首步共20次、仅首共享
  卷积上，加权排序对人工BCE梯度比中位59.8%，分母未含SAM2；全部配对仍活跃。
  四已见draw缺口像素减少40%，内部响应却增加25%。
  支持将“候选尾部范围”与“计算采样预算”分开核验，不能仅由动态极端loss未满足推断应延长训练。
  可能存在异常／重采样／GT缺陷影响，但当前数据尚未确定来源，也未证明代表性采样能提高官方分。
  证据：上述专题完成分析、私有`output/affinity_fused/completed/analysis/training.json`与空间前后回执。
- followup_final_partition: 同一新GT四draw经共享冻结语义、完整原生裁窗后处理补查后，
  匹配393→393，共同386的平均IoU几乎不变；排除unknown／图框对象的共同95则−0.006302，
  保守误切7→10。一个珠光体条带GT42响应组件／闭环数量均不变，却新增内部切割，IoU
  0.9185→0.5303；因此缺口减少、响应组件数量或尾端排序不能替代实际划分核验。
  本组暂缓黑盒及扩轮；仅已见人工退化视野，不外推所有排序方案或官方泛化。
  证据：上述专题完整GT补查与私有`completed/analysis/four_gt/`。
- followup_representative_tail: `docs/AFFINITY_TAIL_20261002.md`保留原初始化、1280配对draw及权重，
  将极端top256改为完整10%尾池等距响应秩采样。首共享卷积上20次抽查排序／人工BCE梯度比中位32.81%、
  余弦全部为正；四已见新GT缺口−6.85%、深内部响应−11.33%，100图预测面积大幅变化从45张降到12张。
  但完整划分匹配393不变、共同389平均IoU−0.003775，保守误切7→8；跨相混合预测25→31。
  旧FUSED误切条带GT42被恢复，不代表超过控制。排序覆盖、抽查梯度方向和更小分布偏移均不能
  替代最终划分增益；响应排序的代表性也不等于空间／类别／实例均衡。暂缓本组黑盒及续训增权。
  仅四已见人工退化视图、测试无GT；不外推所有排序无效、模型饱和或语义必为病因。
  原始新增跨相GT对均触unknown／图框、保守粘连对身份不变，不能将原始混合预测计数直接记为新增可信错误。
  后续先在完整可信对象上定位具体错误，冻结输出干预验证正确关系经融合／划分能否实际纠错，避免无门槛加权。
  私有证据`output/affinity_tail/completed/analysis/training.json`、`four_gt/report.json`、`deployment.json`。
- followup_local_partition_intervention: `docs/AFFINITY_PARTITION_20261002.md`在同四已见新GT视图的
  真实语义／8通道关系／GPU融合图上精确重放基线，20个局部关系干预中，短关系68条修复FF21／22粘连，
  7条修复TAIL珠光体109误切，均无新增raw拆粘；另154的all纠正退出显著误切计数但固定片段仍异ID。
  因此显著拆粘统计减少不能直接当完整修复，须核对固定片段、面积和真实种子／最终区域；
  补查154第二预测ID的GT贡献116→100像素跌出10%门槛，全预测区仍约1.49万像素，
  本质是邻区贡献／边界偏位未消失，不能把每个GT内贡献ID当新晶粒或独立误切。
  46的all纠正虽改变2460个连续边界像素，GT内低阈值域完全未变；须核整个支持带和实际阈值翻转位置，
  不由logits修正数或连续响应变化量判定有效选位。
  阶段连通性可反复变化，“首次分开”不能独断后处理原因。
  同两源原图经正式1024滑窗后五预设只有109在两头均重现，说明应先核实训练退化视图与部署原图的错误存在性，
  不能将所有whole诊断直接记为正式病因，原图／人工退化差异也不能全部归因尺度。
  合法局部关系能改变最终划分，支持继续核实实例／空间覆盖，但未证明新监督泛化收益；
  不因所改关系数多或输出干预成功就晋级模型，不拟合测试GT，不叠加TAIL或直接增权。
  证据：私有`output/affinity_tail/partition/report.json`及`partition_native/report.json`，148来源／冻结／父回执保护通过。
- followup_ferrite_native: `docs/AFFINITY_FERRITE_20261002.md`冻结最高分R1正式native1024，
  补齐全32已见训练原图。完整同类一对一F面积样本仅占F对象14.63%／像素8.71%；
  其小粒收缩明显且ID0量接近配对净亏，但邻区流入／流出会抵消，不能以净量接近推唯一原因或正式均面积方向。
  固定两例same-parent多切沿RGB可见而GT未分的横线，GT自身单连通不证明物理单晶粒；
  将相对GT的多片直接作为假边界惩罚可能压制真实弱界。unknown／图框删失和候选歧义须分别报告。
  复用条件：错误分型始终核完整ID范围、原图、实际种子及类别专属面积；小安全子集只定位机制，
  同一训练源类别较好不排除真实测试域错误。局部模糊定位若固定全图语义只改变affinity输入，
  应明确其诊断用干预范围，不冒称全链鲁棒性或单独affinity的独立贡献。
  验证缺口：真实F内部假切尚缺完整无歧义案例；正式测试无GT，未做新训练或黑盒。
- followup_short_blackbox: `docs/AFFINITY_SHORT_20261002.md`只切gated→short，四张已见GT的
  588个共同匹配平均IoU＋0.005087、铁素体匹配仍358；用户随后转报0.8623／0.8691（86.570），
  相对固定gated最佳87.015两项均下降、综合−0.445。已见GT轮廓收益不能替代完整官方两项检验；
  本次不晋级short，也不据此扩展未满足梯度冲突门槛的Cshort。包身份／用户转报回执为私有
  `output/affinity_short/package.json`与`official_score.json`，包SHA 3ef060c4…已核验。
  未独立核验平台、无重复种子；不称统计显著，不反推测试GT目标或推广为所有短程方案无效。
- followup_seam_blur: `docs/AFFINITY_FERRITE_BLUR_20261002.md`完成32已见原图，固定两路whole语义，
  仅局部D5a＋affinity施加原生σ1–4平滑场。final0距正ID一格，几乎全部为真实WS−1，后正区域删除0；
  距离一格不等于严格一像素线宽或连续响应带窄，也不排除前marker过滤改变归属。固定467面积名单454留存，
  失效13的原误差反而更低，不能预设删失总会制造改善；新增safe17不得混进共同名单。
  同一固定概率也能因分区改变而翻转末端实例类别，不能直接归为语义输入漂移。
  659原完整FF粘连在blur中主要分开，小粒仍10.72%侵入大粒；残留仅占大预测区1.662%，
  因双向10%门槛退出FF名单，单向侵入仍显著。不能由候选数消失推完整修复，
  也不能由dominant-GT着色相近的缩略图判断最终ID相同，必须核逐ID贡献。
  当前没有强端粘连单调恶化，暂不开全图模糊后裁块；下一项需先查真实GT界面监督覆盖，
  不重复已试旧界面增力或将GT遗漏线强压。仅合成机制诊断，不证明真实测试鲁棒性／饱和。
- followup_seam_assignment: `docs/AFFINITY_SEAMS_20261002.md`在最高分原输出上执行一次预设同类细零缝归属，
  只读取原ID0／8邻类别与最近原正像素；32已见图的原467面积对MAE2.928%→2.283%，338好／125差。
  3新censor仍留分母，原预测增量约18.1%落在自己GT以外；所以净面积回收不等于逐对象正确扩张。
  100图F5984／类别JSON字节保持，逐图预测均面积中位+0.595%，但113个F相对增量超过10%。
  固定细像素层对小粒更敏感，不以全图平均变化小断言局部无风险，也不能由面积增量方向推官方误差改善。
  原正像素／ID集合／类别固定不保证组件连通或ID接触完全不变；本规则未修复原安全拆粘或训练语义。
  本地SciPy1.15.3与server1.18.0在预设005／008完整数组SHA一致，但只证明这两图，复现仍保留版本回执。
  候选通过100图最终契约且用户放宽黑盒门槛，支持一次机制验证，不推断测试GT或强制数量。
  用户随后转报seams1024为0.8625／0.8786、平台总分87.05；显示两项均分87.055，比原R1 87.015＋0.040分。
  mIoU−0.0004、面积＋0.0012，支持细缝面积归属有小幅正式收益；规则未改变类别或学习权重，
  不归为语义学习或弱界拆粘改善，不由最高单次近邻推稳定优越。平台总分与显示值计算分单独保存，未独立平台核验。
  当前覆盖训练仍固定不加细缝的原R1控制；以后组合时控制与候选两侧应使用相同归属规则，避免叠加变量。
- limits: 不依据合成检查选择测试阈值；不把处理后新GT仍为residual_unknown的区域当GT，
  也不把本条推广为长程affinity应移除。

## 20260928-affinity-gradient-budget

- status: observation
- last_verified: 2026-09-28
- scope: 困难关系附加损失与稠密affinity BCE的联合微调
- finding: 输出logits上的梯度比例不能直接解释为可训练参数的更新比例。稀疏采样的输出梯度集中，输出端范数上限可能压低其传播到头部参数后的相对影响；必须说明预算测量位置与启用频率。
- evidence: `docs/AFFINITY_CONNECTIVITY_20260928.md`；两组20轮的320次附加步骤实际系数中位0.001301，名义0.1。e20固定8个人工视图无优化复测，输出梯度比均25%，affinity参数比却为1.93%–3.38%（中位2.20%），每4步才启用。最终100图99.25%控制实例与候选匹配IoU≥0.95；共同适配收窄响应带，不能归功于附加项。
- reuse_hypothesis: 解释附加目标是否无效前，在真正训练的参数上测量梯度、方向与频率；仅修改候选预算时复用已完成控制，不凭非零loss或名义系数断言监督足够强。
- verification_gap: 初始诊断仅8个末轮固定视图；后续参数级小批64步对照见下，尚未实施完整20轮预算消融。control随后用户回分0.8486／0.8761，同链综合+0.615；candidate尚无黑盒，该收益不能记给附加项。不能将弱力度定为唯一原因或宣称增权必有收益。
- followup_trace: `docs/AFFINITY_TRACE_20260928.md`在control e20的32源／64固定输入只纠正实际所选输出关系，共7826条边，融合图累计4992格变化超过0.01，但只修复固定主要错误中的3个粘连、0个显著过分割点对；可信域全关系替换则影响大得多。选边是否作用于最终划分也需检查，不能只按梯度范数增权。这里是固定视图输出干预；训练的共享参数、跨epoch选边会影响其它位置，不能宣称实际训练必定无效。
- followup_continuous: `docs/AFFINITY_INTERFACE_20260928.md`连续界面选位完成同初始化20轮，1280步复用配对通过。320次启用全部受25%输出梯度上限约束，实际系数中位0.003689；每轮首步参数梯度抽查中位4.185%，100图98.64%旧实例仍有IoU≥0.95对应，主loss几乎重合。单纯增大名义weight在本次轨迹上不会解除现有限制；应区别输出端预算、参数端梯度和AdamW最终更新。仍未做参数预算因果对照，不将弱作用认定为唯一病因，也不将不同视图抽样的旧2.20%与新4.185%当作严格力度消融。
- followup_actual_effect: 同一32训练源／64视图的真实e20预测仅修复15个旧粘连、2个旧误切点对，并新增5个误切、9个核心粘连；不能相减作净精度。必须把冻结输出强注入验证与真实训练后复测分开，前者显示监督位置能够影响分区，不能替代后者的学习结果。完整control输出复现及语义一致核验通过，无测试GT或新黑盒得分。
- followup_parameter_probe: `docs/AFFINITY_BUDGET_20260928.md`从control e20复制两头、同4源8固定视图各64步，附加参数梯度中位6.30%→15%，同历史本步额外AdamW更新比1.65%→3.41%；粘连修复26→27、误切修复15→15，新增粘连27→32，未通过正式训练门槛。力度变大不能作为有效纠错的替代指标。此处选位初始分区固定、小批过拟合且新优化器，不是原20轮的完整预算消融，不外推方向无效或模型饱和。
- limits: 预测响应带宽不是GT边界精度，实例增加不是自动纠错；本条不推广为所有稀疏损失都需放大。

## 20260926-maskset-history-coverage

- status: observation
- last_verified: 2026-09-26
- scope: 实例联合预测结构的历史复用与覆盖诊断
- finding: 项目已以`mask_set`名称实现Mask2Former式掩码／类别联合预测；不能只因main没有
  对应模型名，就将其列为未尝试方向。修复梯度缺陷、主线伪标签、归属约束与EMA后仍有覆盖缺口。
- evidence: `docs/MASK_SET_HISTORY_20260926.md`；分支`codex/maskset-mainline-pseudo`保留原文，
  最终e144五图790 GT中匹配615，主线715；已知区域未覆盖8.91%对1.27%。原局部图本地仍在。
- reuse_hypothesis: 重访联合mask/class结构前先说明针对未覆盖与互斥分区的具体变化；
  覆盖、有效匹配与漏检计零指标要一起看，不能用已覆盖像素分类率或匹配IoU替代完整输出质量。
- verification_gap: 历史同图诊断，非官方黑盒；主线划分独立性未认证；不是官方完整Mask2Former。
  本轮核对报告、代码归档及图，没有重新训练或复算所有指标。
- limits: 不证明整个架构家族无效，不强制填满未知区，也不恢复历史留出代理作为当前训练要求。

## 20260926-clipping-limited-illumination-coverage

- status: observation
- last_verified: 2026-09-26
- scope: 带饱和像素保护的在线光照增强及单变量实验解释
- finding: 对称的请求幅度和名义抽样率不代表实际增强分布对称；新增截断上限会把部分幅度
  缩到零，并对增亮与变暗产生不同限制。应同时统计选中率、非零率、实际幅度与方向。
- evidence: `docs/SEMANTIC_LIGHT_20260926.md`；配置`config/train/semantic_light.yaml`在3840更新
  中选中49.51%、实际非零37.97%；整图增亮选中后归零33.83%，变暗12.10%，
  两方向有效平均幅度分别0.02457与0.04568。来源为私有逐步日志，代码基于`main@cafd18e`。
- reuse_hypothesis: 分析带截断约束的增强无明显收益时，先核实实际覆盖，避免按名义参数
  断言目标因素无效；局部场系数也不能当作平均逐像素变化量。
- verification_gap: 单次训练；用户回报光照组0.8474/0.8445，相对原simple仅+0.055综合分。
  没有分离饱和保护、实际强度与准确率的因果关系，不由此否定光照因素。
- limits: 不据此自动放宽截断保护或针对测试图选择增强值，不将翻转稳定性等同于正确率。

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
  D3保持kappa0.03，只将起点训练比例增至50%，同20轮首步RGB/梯度比值恢复到0.607/0.904，
  完整16步却为0.782/1.140。起点监督能改善当前低噪声配方，但不同时证明后续迭代有收益；
  不能把D2的失败单独解释成噪声太弱。
  完整前缀测量中，两类训练内输入的整图误差均首步最低；合成空间模糊弱区即使最优前缀
  也可能比输入更差。整图平均改善可以掩盖局部破坏，应同时查看已知合成强弱区域。
  D4与原D3同预算60轮中，32张训练源图的新空间模糊配对均改善，首步RGB／梯度误差相对
  降低19.49%／5.38%，均匀模糊却小幅退步。增强收益应覆盖其目标退化，原单一过程配对可能漏报；
  输入适应改善也不代表多步有效，此时16步平均误差仍不及首步。
- evidence: `docs/RGB_DIFFUSION_SHORT_DECISION_20260923.md`；同4训练配对、权重/优化器/随机流连续，
  明确区分随机t训练、首步估计、完整采样与官方分割成绩；过程图独立保存。
  后续证据见`docs/RGB_DIFFUSION_ALL60_ANALYSIS_20260923.md`。v6固定合成配对误差低于v4，
  用户回报复赛总分却低0.945，说明不能用修复误差排序替代分割晋级。
  `docs/RGB_DIFFUSION_D2_ANALYSIS_20260923.md`：三采样seed与FP32复核一致；移除训练中间状态的
  清晰线索后输出接近恒等，支持状态依赖假说，未证明唯一原因。
  `docs/RGB_DIFFUSION_D3_ANALYSIS_20260923.md`：仅起点抽样分布变化，三采样seed与FP32复核一致；
  关闭后续新噪声仍不及首步，迭代退化的归因需同时考虑状态差距。
  `docs/RGB_DIFFUSION_D3_TRAJECTORY_20260923.md`：三采样seed、全部16个清晰图预测及区域误差；
  前缀末项与部署forward逐值相同。合成空间探针缺同内容均匀弱糊对照，不归因唯一机制。
  `docs/RGB_DIFFUSION_D4_ANALYSIS_20260923.md`：D3/D4起点状态、每次时间步直方图和LR一致；
  相同32源图的均匀／空间配对，FP32实际采样比较；真实图仍未证明正确补边。
- reuse_hypothesis: 扩散早期判定应同时看训练/采样误差及充分拟合趋势；通过工程门槛后，
  用同checkpoint的首步和完整输出检查多步贡献，保持正式模型晋级按真实部署/黑盒证据。
- verification_gap: 全量训练仍是单训练seed；早期定量对照4个训练配对，D4扩展至32个已见源图，
  真实图目检4张。D1/D2/D3/D4无黑盒成绩；D2到20轮，D3/D4已60轮。未证明绝对收敛或退化唯一原因；
  起点强化的跨训练seed稳定性、泛化与黑盒收益尚待验证；v6回分缺平台包哈希回执。
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
- last_verified: 2026-10-30
- scope: 铁素体计数/平均面积变化的语义与几何归因
- finding: 铁素体像素总量接近而实例数减少，不足以将变化全归为合并；小实例改判为珠光体也会显著改变均面积分母，却只改变少量像素。
- evidence: `docs/STAGE1_FINAL_TASKS_20260919.md`回分后诊断。旧新68图按形状IoU≥0.5对应4966对，F→P 308、P→F 17，净类别计数差−291；未对应部分净F差−278，总F差−569。F→P旧预测面积中位1575像素，F→F为14950.5像素，前者仅占旧F总像素0.90%。
  `docs/SEMANTIC_CROSS_20260925.md`补充100图最终PNG逐字节固定、仅交换分类JSON的用户黑盒回分：
  raw和D5a两种划分改用D5a分类，mIoU分别+0.0021/+0.0024，面积项却−0.0207/−0.0119。
  因此边界完全不变时也可发生面积项退化，且不要求mIoU同步变差；应分开报告两项，
  不能把“分类对面积不利”写成“语义全面退化”。
  `docs/SEMANTIC_D5A_ANALYSIS_20260925.md`后续full/simple适配相对原D5a也呈mIoU微升、
  面积项下降，训练loss及GT区域近满分未兑现成官方收益；此轮最终划分亦变化，
  不能直接把全部退步归于分类，或套用前次固定PNG交叉实验的效果量。
  `docs/SEMANTIC_FIXED_20261001.md`进一步固定当时native balance实例PNG字节，八张测试图的私有语义LoRA
  使预测铁素体数520→560，总像素只增8819（0.0583%），逐图预测均面积中位变化−6.49%。
  55处珠→铁中45块小于1000像素；这验证小块改判对计数的敏感性，不代表改判错误或官方面积项下降。
  四张已见GT的589个可信、几何匹配实例在四路下全对，无法据此排序测试语义路线。
- followup_priority: 当前reflect十二源诊断已完成，见`docs/SEMANTIC_REFLECT_20261003.md`。四路590原标域固定配对无错相，完整未删失子集为0；另查纯原标非图框23对象，raw错9而LoRA全部改对，但均是现有GT未匹配小片，仅占完整GT面积0.17%–7.40%。匹配子集全对可掩盖真实局部相别错误；修类亦未必恢复有效完整实例。测试LoRA134翻转中113小于1000px、115形状不变，294个大F逐项不变；F数+19.55%而像素仅+0.204%，不能归为reflect新增碎片或据预测均面积认定官方退化。保持候选，不扩大语义黑盒；修几何后再做类别交叉。官方有效同类匹配条件均值不单独区分错相／漏匹配，当前area驱动收益亦不证明下一次几何训练边际收益一定更高；不晋升全局规则。
- reuse_hypothesis: 面积退化时同时检查类别组成与预测划分；以各自encoder/LoRA独立前向的输出交换进一步定位，不把旧头直接挂到新共享特征。
- verification_gap: 无测试GT，无法判定具体改判正确性；旧替代实验的预测对应仅为计数分解，不能按−291/−278给语义/几何作因果分摊。新交叉可量化固定PNG后的分类来源影响，但PNG生成已含上游语义作用，不能推成纯affinity因果效果或LoRA失配的唯一原因；回分来自用户，暂无逐图官方分数与重复实验。
- limits: 不据此恢复错误类别、强制铁素体数量或用黑盒分数反推目标平均面积；一次替代配方失败不证明旧joint-v3不可替代。
- followup_WS_semantic_input_20261030: 同20a59/S0/200、同13源固定NAF及分类概率，仅把WS的128/200语义底图改为统一128，当前9图F计数逐图不变，F均面积却全部下降0.1106%–0.3428%；该改动同时改变两类区域边界带的灰度对比，不能只归因消除了语义界面。独立种子内缩1px则使九图F总像素全部增加，但总F733不变来自两图+1/−1抵消。固定head和raw投票不等于冻结几何、逐图面积或逐实例类别。两组均无稳定原标核／接口定位收益，不据此调目标面积或解释旧高分为错误抵消。证据`docs/SEED_CORE_20261030.md`、私有`output/readout_core/received/`；九图不是全150，GT已见不完整，保持观察项，不晋升AGENTS。
- followup_readout_identity_20261030: 分层检查不能停在种子身份或最终面积档。K1虽保留种子ID，仍能把原F面积推过末端200门槛，间接触发合并；H0重新生成ID后，须按空间覆盖及双方类别核对，才确认237的两个非超小F预测区合并，而非超小碎块去噪或语义翻转。逐实例精确面积保留私有risk记录，不复制进公开知识条目。种子计数、最终总数、小F面积档、实例身份和类别变化应分别记录；预封ROI未包含该变化位置，补图必须标为后验差异定位，不能混成预封样本或测试真值证据。证据`docs/STRONG_READOUT_20261030.md`，无新增官方收益结论，保留观察项。
- followup_H0_official_20261030: H0同末端200正式回分两项均略降，等权参考相对C低.080分，FP少94伴TP少11。已见接口覆盖微升、全150实例净少105和precision提高，均未保证官方收益；不能把旧support50到H0的变化全归因取消弱生长。证据见同报告的回分节，保留C、不晋级；只更新该候选观察，不推断弱生长普遍必要或统计显著性。

## 20260925-brightness-change-versus-sensitivity

- status: observation
- last_verified: 2026-09-25
- scope: 前端修复与后端语义模型的外观归因
- finding: 前端前后平均亮度差小，不能推出后端对亮度变化不敏感；需固定修复结果和最终实例，
  直接扰动送入分类模型的亮度，并先复现零扰动输出。亮度触发少数类别翻转也不能证明它解释
  高置信大区域的共同失败，或翻转即纠错。
- evidence: `docs/D5A_APPEARANCE_20260925.md`记录修复前后平均L*差约−0.196；
  `docs/SEMANTIC_D5A_ANALYSIS_20260925.md`完成4图／330固定实例／3头／9条件，12组零偏移
  全部复现。正RGB偏移没有截断仍触发翻转，上部两大浅色区域全部条件保持高概率原判定。
  后续`docs/LABELLED_LIGHT_20260925.md`在32训练源／5126固定GT实例上，全局±0.10零新增
  实例错误，A/B局部调光每条件最多新增1错，核心全部正确且未排除实例；不足支持追加训练。
- reuse_hypothesis: 将“前端实际改变了什么”“后端对什么敏感”“改变是否提高任务准确率”
  分开检验，避免由平均外观差或少量翻转直接指定增强方向。GT理想区域上的近满分不能替代
  实际预测实例的投票诊断；核心未截断也不能排除模型参考其它截断区域的影响。
- verification_gap: 无测试GT；训练内数字全局／局部偏移和gamma不等价于真实照明恢复，
  未证明亮度增强训练有效。训练图正偏移有明显截断，需单列整图零截断子集；gamma也改变对比。
- limits: 不根据测试图分类翻转选择逐图亮度值，不将诊断预测制作成训练标签。

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
- followup_20260926: 用户要求的头部冷启动不能自动解释为删除Stage1/joint-v3。纯SSL随机双头短链同时切断共享LoRA、几何FPN与固定教师依赖；用户纠正后于e93停止，见`docs/BACKEND_COLD_20260926.md`。保留这些阶段可通过复用其匹配权重实现，并不必然要求重跑上游；若要改变D5a进入上游的时点，须另列实验问题。该短链没有正式回分，不能用停止决定证明任一阶段的必要性。
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

## 20260926-identity-input-layout

- status: observation
- last_verified: 2026-09-28
- scope: 在既有FP32双头部署前串联RGB校光模块的恒等检查
- finding: 像素值完全相同仍不足以保证部署输出精确复现。此次单图RGB的batch stride由3变为3145728（其余stride相同）后，GPU语义概率最大差约.03487；新输出按输入完整stride分配复制后，恒等概率与原simple完整分割复现通过。未单独定位底层CUDA kernel机制，不能外推所有布局变化都有该幅度。
- evidence: `tools/run_illumination_calibration.py::calibrated`、`tests/test_illumination_calibration.py`、`docs/ILLUMINATION_CALIBRATION_20260926.md`；私有`outputs/light_calibration/preflight.json`。
- reuse_hypothesis: 插入前处理时，将输入数值/布局与实际语义、affinity和最终实例输出一起核验；不得只检查校正张量为零。
- verification_gap: 已在指定4090、sam2_env和原simple/V模型上核验；尚未跨设备或PyTorch版本复现。
- followup_budget: `docs/AFFINITY_BUDGET_20260928.md`的新缓存入口将非连续单图stack成连续batch，D5a输出像素完全相同，但encoder特征变化，原图最终实例218→219；同一恢复图改回原channels_last布局后各级特征逐值一致。短测在任何更新前由最终输出复现断言拦截，随后保留原单图布局。对逐张／batch或缓存重构应同时检查像素、噪声、布局与最终分区；未进一步定位底层CUDA算子。
- limits: 本次仅修正新增校光入口的输出布局，不追溯宣称历史模型或既有实验无效。

## 20260926-monitor-before-loader-fork

- status: observation
- last_verified: 2026-09-26
- scope: GPU模型及多线程CPU预览已运行后，再启动校光DataLoader workers
- finding: 默认fork的4个数据worker停在futex，GPU空闲且首个更新未完成；改为spawn后同一数据配方完成首个更新。与继承已初始化线程状态一致，但未用栈采样单独定位锁来源。
- evidence: `train_illumination_calibration.py::train`、`docs/ILLUMINATION_CALIBRATION_20260926.md`；私有`outputs/light_calibration/startup/`保留失败启动与e0，当前`train.log`记录首个更新。
- reuse_hypothesis: 先运行GPU/CPU预览再创建数据worker时，显式使用spawn并在启动验收确认首个有效更新，不能只凭进程存活宣称训练正常。
- verification_gap: 当前4090/sam2_env已验证启动恢复，正式训练尚未结束。
- limits: 不要求为历史已稳定训练的所有DataLoader统一改造。

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

## 20260926-manual-pool-source-identity

- status: observation
- last_verified: 2026-09-26
- scope: 当前32张人工原图与1000张允许训练原图的来源对应关系
- finding: 两个目录曾采用不同编号；同名文件并不代表同一图。32张人工图均可按文件SHA在1000图中找到唯一原图。现有监督数据集按人工目录自身文件名读取对应GT，不能据此误判标签错配。
- evidence: `tools/analyze_semantic_domain.py::verify_cohorts`及其改名来源测试；完整对应关系仅存私有`outputs/semantic_domain/manifest.json`。
- reuse_hypothesis: 跨目录统计有标签覆盖、去重或排除已标注样本时，先核验来源身份，不按文件名直接关联。
- verification_gap: 已核验当前服务器32/32一一对应，尚不推广到其他历史数据副本。
- limits: 不改变原图、GT或文件名；无标签训练仍可完整遍历1000源。
- followup_20261030: 封存无监督语义对照再次遇到raw32与1000池train001/557同名不同像素，GPU前hash门槛截住；正确原标raw32与冻结诊断cache RGB完全一致，改用无损同像素输入后执行，不放宽hash或改旧资料。仅凭同名不能推断模型是否见过某图，跨池需原SHA映射；正式150另逐图绑定原file及RGB hash。证据`docs/SEMANTIC_UNSUP_COMPARE_20261030.md`、私有`output/unsup_semcheck/received/diagnostic_rgb_difference.json`；旧32/32映射未在本轮重新全部核验，不扩张其适用范围。

## 20260926-consistency-coverage-and-signal

- status: observation
- last_verified: 2026-09-26
- scope: 固定语义教师、共享冻结编码器、可靠内部像素的无标签一致性实验
- finding: 1000源全部产生监督、可靠覆盖75.88%，仍不代表获得强学习信号；教师／学生在接受区域平均概率差仅0.000386，B对配对A全量输出仅7／8668实例改判。20个有标签epoch对应无标签池1.28遍，不能据此宣称扩池无效或饱和。
- evidence: `docs/SEMANTIC_CONSISTENCY_20260926.md`完成分析；私有`outputs/semantic_consistency/analysis/summary.json`，两组各1280更新、完整100图部署、冻结及输入配对核验通过。
- followup_20260926: `docs/SEMANTIC_TEACHER_PROBE_20260926.md`进一步检查原图S-align与D5a/simple自身教师。全32人工＋固定64其他源中，两教师共同可靠域均零类别分歧；64图原图教师独有覆盖仅0.65%，人工图独有GT格点两次退化合计32次正确纠错、182次错误方向。仅换输入域和教师身份未必提供新增正确目标，应分别核对共同与独有接受域。新诊断弱视图口径不同，不能将其概率差与旧训练日志直接横向比较。
- followup_20260927: `docs/SEMANTIC_GRAY_20260927.md`中的独立灰度目标仍在接受域与学生高度同类（分歧0.0455%），尽管每轮首步实测的加权先验／GT梯度范数比均值35.04%。A/B各20轮后43／8668实例改判、83.44%实例分数向0.5靠近，提示应区分类别新信息与固定软目标的置信度牵引；未分解各部分实际梯度，不能声称唯一机制。三个较大改判占变化实例面积72.27%，少量改判也不能推出官方影响必然很小；无新黑盒，不宣称灰度监督无效或已饱和。
- followup_gray2_20260927: 单向约束去掉已满足置信门槛的处罚后，活动格点仅占接受量0.06146%，47.42%步骤额外loss为零。仍以全部接受位置作分母，系数0.10→0.15但逐轮首步梯度比均值35.04%→0.368%，全量仅6／8668实例额外改判。需区分名义系数、活动量与归一化分母；记录为保守配方失去实际信号的观察，不推广为所有稀疏约束应无上限放大。证据与下一步未执行建议见`docs/SEMANTIC_GRAY2_20260927.md`。
- followup_gray3_20260927: 保持gray2数据流与接受位置，仅改为上限20的活动平均后，逐轮首步梯度比均值增至3.876%，同输入上的先验类别分歧减少31.17%；完整部署改判35／8668且34个向铁素体。说明恢复梯度可加强规则拟合，但不自动解决覆盖与泛化；50.86%步骤附加损失仍为零，不能用整体抬权重替代有效信息供给。无新黑盒，不把先验分歧减少等同真实错误减少。证据见`docs/SEMANTIC_GRAY3_20260927.md`。
- followup_gray4_20260927: 同一64张非人工训练源、每张两次退化中，保持清晰图灰度目标不变，原增强的5058个重复统计区域仅2次整体分歧，附加强全局／平滑局部光度变化后为120／60次。20轮后相同视图分别剩1／1次，固定难例在接近峰值学习率的e5已无区域类别分歧；说明应区分当前合成任务满足与低学习率，不能由后期停滞推断整个模型饱和。区域均值选样比像素损失更早满足：后5轮267次比较中240次所有区域评分为零，其中115次未选视图仍有更多逐点分歧，但差额中位数仅7格、无大块，不能夸大为真实测试大块错误的主因。较强视图的初次短测又令附加梯度超过GT，说明稀疏信号放大会随视图难度改变实际强度。完整部署相对旧控制改判102／8668，强于gray3的35但不是正确率证据。详见`docs/SEMANTIC_GRAY4_20260927.md`；未证明真实测试错误的光照因果、学习率因果占比或黑盒收益。
- reuse_hypothesis: 扩池实验同时报告来源遍历数、接受比例、学生与教师实际分歧、完整部署变化；避免把“effective_sources”当作新增正确知识。损失数量级不等于梯度贡献。
- followup_prior_windows_20260927: `docs/SEMANTIC_LORA_20260927.md`的只读排查确认：局部明暗双峰要求少数模式至少5%，会使近单相窗口失去支持，规则网格上留下矩形忽略区。该区mask=False、灰度损失梯度为0，热图却把0.5占位画成统一颜色。解释先验热图时必须结合接受mask；区域被忽略与被赋错标签不同。尚未证明这一覆盖机制是测试误判主因，不能据此直接放宽全部阈值。
- followup_private_lora_20260927: 同一1280步输入与gray4实际选中视图下，仅增加语义专用LoRA学习；GT／先验末5轮loss下降1.30%／37.52%，相同64训练源的合成全局／局部区域冲突1／1→0／0。但完整测试相对gray4仅65／8668实例改判，62处原分数距0.5不超过0.1，后5轮52.81%步骤额外梯度为零。提示增加特征适配自由度可改善既有规则拟合，却不能绕过监督覆盖和真实域泛化的问题；不能宣称冻结特征是主因或整个模型饱和。平均概率0.4994→0.5010即可造成整实例改判，进一步说明变化数量／面积不等同纠错。证据与范围见`docs/SEMANTIC_LORA_20260927.md`及私有`output/semantic_lora_analysis/`，无新官方分数。
- verification_gap: 无新黑盒成绩，也未验证过滤掉的困难区域是否是失效主因。
- followup_coverage_20260927: 128→192窗口仅补零局部支持区的预检查完成1000源／1280次固定视图；仅14源新增48666格点，已完成独立LoRA与新增目标零类别分歧，仅2格未达0.9。新增可靠GT内部2962格点无冲突，但不具备足够新增学习信号，按预定条件未启动20轮。可见的接受mask缺口不等于模型不会该区域，扩覆盖须同时验证目标可靠性与未满足信号。见`docs/SEMANTIC_COVERAGE_20260927.md`；此结论限于零支持补充及固定末轮诊断，不外推所有忽略域或从同一初始化训练的效果。
- followup_scale_20260927: D5a之后混合全图与512局部放大的20轮实验改变了输入任务，混合loss不能直接对比旧全图loss。拆分同输入全图步骤后，末5轮loss比旧LoRA高4.6%；固定2张训练ROI的严格末态对照却全图错像素−2.72%、局部−9.95%，两者统计域不同，不矛盾。局部错误仍多于全图，不能把放大预测无条件当教师。100图105／8668实例改判不足以证明泛化，少数大面积变化仍可能影响官方指标；尺度候选无黑盒。旧LoRA原后处理已补回分0.8445／0.8557，此前无分状态仅是历史记录。证据见`docs/SEMANTIC_SCALE_20260927.md`，保留训练ROI与真实测试的边界，不自动续训。
- followup_raw_stress_20261004: 当前最高reflect末端是S-align/raw，而近期强先验／尺度候选多在D5a/simple；弱候选不能否决不同实际入口的训练机会。raw四固定已见源21次前向清晰SHA与类别精确复现后，强全局／局部低对比下616对象／463大对象原标汇聚均无错相，但像素错误仍变化；只说明该具体纯光度任务未提供大对象纠错启动理由，不证明测试语义饱和、光照无因果或模型没有记忆。下一候选检验采集模糊破坏纹理，未训练。证据`docs/SEMANTIC_RAW_STRESS_20261004.md`，不晋升全局规则。
- followup_acquisition_stress: 随后同四源原生sigma3／8及平滑空间混合、叠加固定光度，共21次实际raw前向通过清晰exact与冻结；sigma8使原覆盖像素错误0.6886%→1.4261%，但616／463大对象原标和核心均值仍0错相。像素敏感、GT粒汇聚和部署预测对象投票是不同层次，GT粒近满不能排除碎片误相或未见域错误。预先固定强度未按输出追加搜索，不把名义二阶宽度说成测得测试PSF；见`docs/SEMANTIC_ACQUISITION_20261004.md`。仅否决同项纯增强短训的充分启动依据，未声明语义饱和，不晋升全局规则。
- limits: 当前保守固定教师配方的单次受控观察，不推广为所有无标签方法无效，不凭测试目检构造标签。

## 20260927-reuse-unchanged-control

- status: observation
- last_verified: 2026-09-27
- scope: 只修改候选附加损失的连续短周期对照
- finding: 相同控制配方不因新增候选而需要再次完整训练。用户明确要求复用已运行控制；现成协调器默认同时重跑A/B属于实现便利，不构成新增实验的理由。
- evidence: `docs/SEMANTIC_GRAY2_20260927.md`；gray与gray2控制的初始化／冻结权重标识一致，1280步输入、增强、随机序列及学习率逐步相同，监督loss最大差0。完成后已核验末态全部模型张量、100图实例掩码、类别及实例分数完全一致；重复权重已清除，旧控制及两份日志保留。差异配置均为输出路径或控制组未启用的灰度分支。
- reuse_hypothesis: 下一候选启动前优先复用匹配控制的checkpoint、日志与完整部署结果；缺少复用调度入口时先补入口，不默认以重复GPU训练绕过。
- followup_gray3_20260927: 已落地显式复用入口，新GPU控制6步精确重现旧正式控制前6步的输入、增强、学习率与监督loss；实际初始化／冻结摘要、数据记录、末轮权重与完整部署检查通过。正式仅调度候选训练／推理，完成后全部1280步共同监督输入配对通过；实现与证据见`docs/SEMANTIC_GRAY3_20260927.md`。
- verification_gap: 当前证据覆盖相同配方下本轮全程，不保证任意未来代码或数据改动都能复用。继续只忽略明确不影响控制计算的字段，不能仅凭相同seed认定可复用。
- limits: 若共同训练条件改变、旧产物不完整，或用户要求不同seed的独立重复实验，可重新训练对照；不据此取消验证新计算路径所必需的短测。

## 20260927-marker-frame-contact

- status: observation
- last_verified: 2026-09-28
- scope: 原affinity边界经骨架化、反相连通域种子、分水岭和整实例语义投票的部署链
- finding: 语义像素已分成两类，也可能因两侧共用同一marker而在最终投票中整片统一。骨架化保留边界自身连通性，不保证保留边界与图框接触及两侧背景分隔；贴边粗线末端回缩后，后置窄封边可能接不回。
- evidence: `docs/SEMANTIC_SPLIT_20260927.md`；固定100图8668实例，原语义投票与affinity复现差0。466例内部混合预测中50例固定代表点在桥接后分开、骨架后相通，全部贴边；实际服务器使用skimage回退。原生／回退合成测试及实际分水岭marker截获验证可复现。
- reuse_hypothesis: 排查成片语义误判时先拆开逐点概率、种子形成、分水岭和区域投票；连通域在整图求解，再裁切显示，不能用裁块关闭图外通路。测试骨架时同时验证背景分隔与画框约束。
- verification_gap: 条件筛选的50例不是全测试真实漏分率；无标签目检和恢复原有分隔不等于准确率／官方得分改善。
- followup: 全100仅替换marker的前置封边恢复50/50分隔，但实例8668→10575，50–199像素小块410→1136，550个拆分旧实例全部贴边；实际marker与分水岭障碍逐图核验通过。机制成立不代表粗放修复可部署；候选不晋级，下一步只接受原骨架前独立区域支持的保守种子设计。
- followup_conservative: `docs/MARKER_RESTORE_20260927.md`改为原粗边界全图独立区域支持的种子恢复，并按用户最终要求用每图实际分割log面积下界统一过滤全部实例。100图仍恢复50/50，实例9062、小块363；无标签几何／类别变化不能当作正确率。只有8图29个seed低于自适应阈值，说明“统计小侧离群”不等于固定小面积删除；分布异质时下界可很低。不以不完整训练GT推断测试最小晶粒，不按期望平均面积反调阈值。
- followup_crop_scope: 保守修复后的局部展示仍可能只露出一个跨区域实例的小部分，使其看起来已独立分割。新的冻结前向重现部署投票差0，发现局部偏铁概率被图外相连的大面积偏珠预测平均；不能仅凭裁图最终颜色认定语义核心判反。检查最终实例的完整ID范围、裁图内外各自概率后再定因；同一D5a张量裁块放大能改变局部判断，但邻近区域也可能反向变化，不足以直接替换整图推理。私有复现材料见`output/semantic_region_probe/research.md`与`report.json`。
- followup_blackbox: 保守恢复候选用户回分0.8474/0.8354，未胜出、不晋级。修复已知分隔且小块数量减少，仍可使铁素体平均面积系统性下降；实际预测91/100图下降，中位−4.396%。同权重独立LoRA原后处理随后回分0.8445/0.8557，补齐配对：恢复＋过滤组合mIoU＋0.0029、面积项−0.0203、综合−0.870分。此时不得用“碎片减少”替代面积分布检查；也不能从两组合分数分配恢复与过滤各自贡献、由均值下降反推真值方向或认为每处新增分隔均错误。历史simple/light不是严格A/B。证据与边界见`docs/MARKER_RESTORE_20260927.md`。
- followup_control_trace: `docs/AFFINITY_TRACE_20260928.md`固定新最高分control e20，在32人工源原图／模糊各32输入的已知粘连点对中，238/358、304/554在bridge后分开、细化后重连；69例最终通路可完全处于原标可信域。三处全图通路重放分别距图框3/8/8像素、未知像素0，说明问题不只来自测试未知标签。点对不独立且是已训练图，不能当测试错误率；多数其它通路不能排除未知区影响。真实marker捕获、GPU/CPU部署等价和全部冻结核验通过，但不据此重新启用旧恢复＋过滤组合。
- followup_restore_only: `docs/AFFINITY_PRESERVE_20260928.md`在同一最高分control上独立恢复分隔、关闭附加面积过滤，64视图的542/542个已定位细化重连点对恢复，按现有GT同时新增58个显著误切、7个粘连点对。100测试图实例9201→9600、小块504→466，平均铁素体面积93图下降、中位−4.788%；说明恢复／种子重生长本身能改变面积分布，无须附加面积过滤。实际分水岭图和全部模型固定，但新旧实例存在周围轮廓重分配。用户随后回分0.8499／0.8559，相对control的0.8486／0.8761综合−0.945分；不晋级，也不把历史组合回退全部归给过滤。点对不独立且新增／修复口径不同，不相减充当净精度。
- followup_area_bands: `docs/AFFINITY_REVIEW_20260928.md`读取两组实际100图提交包，铁素体数＋5.009%、总像素＋0.0216%；按每图control铁素体面积中位数统一分档，小于25%档净增141／299，至少25%档净增158，最小1%档反而−3。应同时检查类别专属及相对尺度分布；全类别50–199像素计数下降不足以排除小侧区域增多，但分档净增包含跨档和投票变化，不等于错误碎片数。官方有效匹配mIoU与铁素体平均面积对未匹配区域的计入方式不同，局部轮廓改善与面积项退步可以共存；不能据此设期望均值或按黑盒反调删除阈值。
- followup_interface: `docs/AFFINITY_INTERFACE_20260928.md`在同一32源64视图中，以完整可信短程界面替代每组16条孤立困难边，涵盖预测零缝附近、但仍忽略未知GT。实际有梯度的位置注入正确关系后，修复33粘连／60误切（旧选位3／0），同时新增26误切／38核心粘连；这只支持一次有限训练，不能相减作净准确率或推断训练必然有效。选中142494关系相对旧7826，不能单独归因于零缝处理；后续区分监督位置、覆盖量与梯度预算。检查既有错误修复时必须同时记录新错误和变成背景的代表点。
- limits: 预封边会产生新小腔，合成25像素粗线出现99/110像素额外种子，超过现有面积门槛；不能全局加粗封边后直接晋级。原本就没有分隔的内部断口、同相晶粒和语义自身误判需另行处理。

## 20261001-normalized-weight-cancellation

- status: observation
- last_verified: 2026-10-01
- scope: 类别分别归一化的affinity监督权重扫描，限当前native／mixed／balance计算路径。
- finding: 配置可修改不代表梯度会变化。某类所有有效关系乘同一正数，再用该类权重和作分母，会把常数抵消；当前`pseudo_negative_weight`从1改0.5或0.25即如此。将它设0则删除该类梯度，是突变机制，不是连续减权。困难度位置重加权和整路来源loss系数与此不同。
- evidence: `docs/AFFINITY_SWEEP_20261001.md`的执行路径／公式检查，及忽略目录`output/affinity_sweep/audit_loss.py`的虚构8×8 CPU独立梯度检查；三个正数loss与logits梯度差0，r扫描改变两类梯度，未知位置梯度0。不读赛题图／权重，不作准确率判断。
- reuse_hypothesis: GPU扫参前先核对实际调用分支与归一化分母，用固定虚构输入确认候选数值改变有效梯度；同时区分名义总系数、来源比例和实际裁剪后的更新。
- limits: 仅常数且同一归一化类别内一致、未触及分母clamp时成立。非均匀权重、未归一化模式或未来改动须重核；本条不能推广为来源加权无效，未证明任何新候选的正式收益。

## 20261003-graph-readout-gap

- status: observation
- last_verified: 2026-10-03
- scope: 冻结mixed balance e20的512格四短关系、原gated-top2融合及原生marker／分水岭；完整七源8轮干预与另两源4轮实际融合选位诊断。
- finding: 在合法affinity图中修改瓶颈关系，不保证修改实际部署标量边界。七源正向96条关系在4源native boundary逐值不变、7目标均未修复；另两源改用实际融合节点容量及活跃短通道，边界确有变化但最终实例／类别全部精确不变。图通路、连续屏障、种子和完整粒归属须分开验收，活跃通道只是必要检查。
- evidence: `docs/AFFINITY_BOTTLENECK_20261003.md`；私有`outputs/bottleneck_summary/final.json`及`outputs/readout_summary/report.json`、两套完整冻结回执，真实空mask／原缓存／模型／源GT前后核对；106项虚构CPU测试。未启动原样128步或20轮训练，无官方增益证据。
- reuse_hypothesis: 采用maximin／瓶颈监督前，先验证所选合法关系经真实融合、插值、拼接和读出能影响最终目标；原生未知／细条不可被网格邻接跳过，同GT负路径容量1的边应整段同GT。局部修正只给窗口内路径见证，不冒充原固定端点的全图路径。
- limits: 稀疏8轮不能否定所有连通监督或模型可学性；两源4轮与原8轮预算及关系数量不同，不能据此直接排名或通过多源训练门禁。原生GT删失、结构歧义和mirror已有解单列；不晋升AGENTS或自动开启长训。

## 20261002-shared-gradient-numerics

- status: observation
- last_verified: 2026-10-02
- scope: FP32张量、cuDNN允许TF32的512格affinity共享卷积梯度分组诊断。
- finding: 数学上等价的“各任务分别反传再相加”和“一次反传组合目标”，可能因GPU卷积数值精度不能通过严格重构门禁。不能直接把该差异解释成目标公式错误或任务冲突；需要固定真实输入检查计算精度及误差，而不是先放宽容差。
- evidence: `docs/AFFINITY_DISTANCE_20261002.md`及私有`output/affinity_distance/numerical_probe.json`：同一首个真实梯度样本，在cuDNN TF32下L2相对误差3.157e-5，严格FP32前向／反向为4.518e-7；方向余弦变化不到0.000081，旧全通道BCE重构门槛保持不变。新协调器局部恢复精度／模式／requires-grad／buffer／随机状态，48项CPU含正常和异常退出检查通过。
- reuse_hypothesis: 多任务梯度诊断同时报告重构误差、各任务范数、方向及实际计算精度；用共享层，不用各任务独立输出行。高精度诊断须局部控制和恢复设置，不能悄悄更改已评分部署的推理数值合同。
- followup: 完整128个真实条件的严格FP32重构均通过原门禁，最大L2相对误差1.269e-6，参数／已累积梯度／来源未变化。同精度train／eval logits完全相同；严格FP32相对原部署前向最大差0.005616，融合和最终推理保留原设置。未从浮点门禁问题推出任务冲突或训练收益。
- limits: TF32原因仍来自单个失败样本的受控复现，不能推广所有GPU误差都来自TF32；近零或抵消后的梯度方向可能不可靠。128条件只覆盖16源、两个时点及一个共享层，不据此设全局训练规则或晋级。

## 20261002-native-spatial-exposure

- status: observation
- last_verified: 2026-10-04
- scope: mixed balance e20的32人工源、1280步实际回执，其中640次原生1024裁块；完整处理后新GT。
- finding: 图像遍历均匀不等于局部界面的原生曝光均匀。每源均出现40次，但5635对铁素体接触中839对未进入任何原生裁块；逐源原生次数归一化后，256像素边缘带接触边平均曝光5.66%，内部31.77%，32源边缘均低。均匀合法起点会产生位置覆盖差异；本轮640次人工原生裁块均首窗通过，不能归因为标注覆盖门槛的重选。
- evidence: `docs/AFFINITY_SIGNAL_20261002.md`；私有`output/affinity_crop_audit/analysis/report.json`与完整接触CSV，SHA、源表、逐接触与分区求和独立闭合。旧源码、权重、GT及1280步回执前后不变。
- reuse_hypothesis: 原生视野训练同时核来源频次、裁块位置及可信界面的曝光次数，分别列出全图与原生视野；改变采样前，用同一批界面和原生次数比较新获见、新遗漏及中部损失，不按测试输出挑位置。
- verification_gap: 全32源实际网格、真实参数更新、完整20轮／100图及32张已见新GT最终输出已检查；候选无官方回分，未把训练图机制诊断当未见图泛化结论。
- followup: 同预算／单预设种子的50%旧随机、50%合法3×3起点反事实，FF原生未曝光14.89%→5.09%，补649对而新增97对零曝光；边缘曝光5.66%→10.90%、内部31.77%→28.15%，整体21.79%→21.55%。这是覆盖机会重分配的实证，不是额外监督总量或训练收益；窗口／元数据CPU检查通过，但真实数据路由、512网格和参数短测尚未完成。
- followup_grid: 后续32源实际512格审计中，649个原生获救对均保留short，其中2对仅靠对角短关系；FF short零曝光843→289，但新增99对零曝光、内部短关系−10.80%，总量−0.296%。GT ID消失均为裁下片段，没有完整GT支持消失；新锚点全偶数的抽样相位风险仍在。whole合并后FF short零曝光仅4→1，不能将native零曝光理解成多数界面从未监督。真实8步路线、参数梯度、冻结和重载门禁通过，只支持单变量20轮尝试。证据见`docs/AFFINITY_COVERAGE_20261002.md`及完整私有grid／smoke回执。
- followup_trained: 同源同预算20轮完成，只改320人工native坐标，其余输入、原初始化／BCE、冻结与状态合同保持。全32已见GT的共同3051个F平均IoU0.900239→0.900120，同466安全面积MAE2.930%→2.953%；原467名单新增1个失效仍列出，固定弱界244像素中超过阈值的87→87且仍粘连。该单种子配方改善监督机会但没有显示最终分割改善，提醒先区分“原生曝光不足”和“真正未监督”，并追踪实际边界闭合而非响应小幅上升。无官方收益证据，不扩轮或晋级；不能推广为空间采样普遍无效。证据见同一实验文档及私有`output/affinity_coverage/analysis/gt/`。
- followup_weak_path: 冻结两臂的正式原49 GPU／8组CPU完整终态定位中，固定原标界面全部保留在实际short网格2原生像素内，关键缺口只有一窗参与，原short约0.36／0.44且融合略高。合法short纠正能拆主要区域却保留邻粒侵入；all显露689／125像素内部片，直接native参考仍有141片。50px／10%显著规则漏计部分有深部贡献的小片；监督曝光、概率提高、两点分离及候选数下降都不能替代整粒所有贡献／内部连通检查。该单已见训练源反事实本身不能证明特征足够或训练可学；后续复制头短测另列，未晋级。证据`docs/AFFINITY_WEAK_PATH_20261003.md`及私有`output/affinity_weak/capture/`。
- followup_weak_fit: 同一已见源两窗、共同R1 e20复制头各32步，冻结真实上游特征；原完整损失A和局部short界面／内部正关系B均分开主要粒，B保住小粒全部深部，证明此例在现有特征下可学但不外推泛化。B固定21安全F面积改善，却使已知F→P增加，主要为原匹配F与P同预测实例；原语义概率逐值冻结，几何变化会经最终实例投票改变相别。新增split／merge均为P，不能直接归为F面积代价，3个原匹配F失配均删失且保留，删失不抹去已知F像素误分类。局部目标改善、全图保留和最终类别须并列；B改变边数／活跃通道／每边力度，不能锁定为只改位置的因果。下一步可靠旧关系保留仍是假说，不固化为项目规则。证据`docs/AFFINITY_WEAK_FIT_20261003.md`及私有`output/weak_fit/capture/`。
- followup_weak_retain: 同一实际输入、共同R1起点两组32步，完整GT＋局部学习U与额外高置信GT一致KL保留R均保住目标深部并避免旧B的F-P大块退步；补完整GT自身已有作用，不能归因全部给保留。保留降低约13%整图响应漂移，但固定21面积差仅0.0058个百分点，均丢原匹配F的全部已知内部到邻F，说明概率保留不保证相界拓扑。删失限制完整真面积，不抹去已知区域损失；新增候选相别及原固定匹配名单须同时检查。原同输入单步U-U存在2.98e-8参数差，配置单变量不等于CUDA逐位稳定，不能解释微小诊断差为稳健增益。仍为单已见源观察，无官方证据，不扩训或晋级；证据`docs/AFFINITY_WEAK_RETAIN_20261003.md`与私有`output/weak_retain/capture/`。
- followup_marker_path: 原环境实际全图重放发现新增FF粘连时连续响应仍能分隔，失效出在骨架端点退离图框；强响应／可靠概率保留不能保证marker背景分隔。镜像延续后细化在单已见源恢复原名单、32／64完整输出一致，原固定安全面积逐项不变，但全图F数量与平均面积仍变，不能据此晋级或取消多源安全门禁。固定面积子集、GT覆盖诊断行都不能替代正式类别JSON口径的全图F分布；同源不同状态不当独立样本。候选只换marker，真实WS障碍逐像素冻结；与旧种子恢复机制不同但仍需双向拆并／贴框碎片检查。候选观察，不修改AGENTS；证据`docs/AFFINITY_MARKER_PATH_20261003.md`及私有`output/weak_retain/marker_path/`、`marker_reflect/`。
- followup_marker_trial: 六固定已见源确认原R1加镜像后557匹配F无新增丢失、114固定面积逐项不变，而R32单源短训头多源仍丢6匹配，说明修复一个读出问题不能反向证明局部训练可扩。100图只换marker仍造成23图预测均面积绝对变化>5%，即使F像素仅+0.0313%、相别像素翻转0.0582%，数量尾部也能显著影响面积；不能据低像素漂移称几乎无副作用。2026-10-03用户更正reflect回分为0.8645／0.9006（88.255），同链原R1→reflect综合+1.240分，其中面积贡献1.160分，支持真实读出修正确有收益；未独立登录平台核验，不将剩余误差全部归因细化，也不反向证明局部训练可扩。原部署逐图及ZIP字节复现、实际地图冻结和双向关系用于独立候选身份控制。证据`docs/AFFINITY_MARKER_TRIAL_20261003.md`，保持候选知识，不晋升全局规则或自动改默认配置。
- followup_marker_stages: 七源实际阶段图显示“曾在骨架阶段重开”不等于最终粘连：reflect修好的2FP由后续dilate重新隔开；剩余6FF全程相通，2例早期端点0则归因未知。7个同粒目标纯骨架一度相通而dilate又切开，需分流完整界面与内部通道，不能全局取消dilate。nothin无新增完整目标修复，固定reflect实际689匹配F虽无丢失，仍新增安全F内部拆分、4项固定面积删失；保住匹配不等于拓扑安全。继承旧688名单的均值不能与新689名单直接相减，需固定参照GT逐项算；安全与保守拆分口径的同GT错误不得相加。见`docs/AFFINITY_MARKER_STAGES_20261003.md`及私有`output/marker_stages/`，已见处理后GT／结构歧义限制保留，仍为候选知识，不晋升全局规则。
- followup_structure: 七源完整标量屏障反事实表明，659的790px邻粒泄漏转成591px独立片后，虽退出GT面积10%显著门槛，仍有251深部像素，阈值通过不等于整粒修复。原21安全面积精确不能覆盖新658/143px F实例；须同时保留所有贡献及新对象谱系。内部清除在4/6原覆盖目标使已知深部100%归一，但874因外界缺口而吞入邻GT，丢原匹配，002邻粒面积变差；全135固定面积MAE3.1354%→3.1812%。完整内部连通必须与外部纯度／原匹配和邻粒面积共同检查。实际匹配相别翻转0；None失配不可叫语义翻转。native标量反事实未验证实际通道可表达或参数可学习，GT归一不自动证明物理边界正确，均不能据此部署或自动扩大训练。见`docs/AFFINITY_STRUCTURE_20261003.md`及私有`output/geometry_structure/analysis_v2.json`；仍为候选观察，不更新AGENTS。
- followup_structure_representation: 随后347实际12窗short4／all8同GT原覆盖内部关系修正均经原CUDA融合／reflect读出生效，104原匹配／28安全面积保持；四短程已能表达该例归一，不能再把其失败直接归为融合不可表达。all的IoU稍高但新合资格目标面积误差更大，不作优势排名。未选logits精确不等于标量边界全域精确，gated支撑及插值会越出选择域，必须另测真界面／未知传播。原多边形本已将可见暗弧两侧记录同粒，原覆盖目标不依靠填补桥接，但标签归一不能裁决物理边界；此反事实不证明参数可学习或测试收益。证据同文档及私有`output/geometry_interior/`；不升级全局规则。
- followup_interior_fit: 排除已知歧义后的三源、共同R1起点和真实冻结特征，各32更新×累积3，full／outer／高置信GT一致KL共同配方仍丢27原匹配、6固定面积资格失效；local0.25又加重为51／13，固定689平均IoU相对control再降0.036129。局部深部归一可学，但另一目标由control已正确匹配变为邻F吞并；聚合正概率提高的目标仍保留分隔。原覆盖／非空外围损失和旧概率保留都不保证外界闭合，deep100%不保证对象独立；None失配仍不能叫相别翻转。局部加权梯度并不弱、初态KL梯度为0，不以系数判断保护力度。下一步应冻结末态定位剩余屏障覆盖及逐通道尾部和外界泄漏，不凭局部观感继续加权或延训。只是一轮已见源机制反例，不否定所有局部监督／GT优化，GT扩充暂缓；见`docs/AFFINITY_INTERIOR_FIT_20261003.md`和私有`output/interior_fit/`。不晋升全局规则。
- followup_interior_trace: 冻结末态的原三头完整实际窗精确重放发现，一例粒内响应虽下降，在marker膨胀后仍重闭；只移除35原覆盖marker像素、或只强化足迹内99条既有可信short关系(360原生位置)均消除主要深部残片。聚合local概率约0.93不能替代困难足迹内0.766及其8条<0.5尾部，四短程本例足够表达不证明参数可学。另一例只恢复额外695原覆盖位置失败，恢复706filled旧响应则恢复两F独立；原覆盖全路径限制可以使额外保护对完整界面断续，而原full GT监督仍存在。选择域的来源、概率尾部、真实marker阶段及最小支持干预须分开核实；不据一例将全部filled当可信GT或全局取消dilate。两源反事实不抹去前轮七源回归；优先检查完整known外围可靠断开保留，再考虑内部困难尾部。仍为候选机制观察，不升级AGENTS或改部署。证据`docs/AFFINITY_INTERIOR_TRACE_20261003.md`与私有`output/interior_trace/`、`output/interior_support/`，92项CPU／实际GPU及输入冻结核验通过；未新训或改GT。
- followup_retention_support_gate: 对原外围泄漏病例的真实冻结八通道logits、固定band及原监督做只读交集检查，original-covered→processed-known的负关系保留扩展在原生足迹空间上有1886条候选，其中1583条通过GT一致／教师置信>0.9及全known路径条件（包含重叠窗口重复）。旧mask逐值复现且为新mask子集，unknown／padding／第三GT仍排除。必须量实际band与置信交集，空间上限和“KL存在”均不能替代真正命中；命中后仍须检验实际梯度、完整外界隔离及多源回归。证据`docs/AFFINITY_GT_SOURCE_20261004.md`和私有`output/retention_actual/report.json`、`output/gt_source/retention_cache/`；只完成覆盖诊断，未训练该保护变体，不能宣称其有效或将所有filled视为可靠真值，不升级AGENTS或改部署。
- followup_retention_normalization: 外围保护四个真实缓存窗新增约2.8万～3.1万负边，而旧负保留仅约0.4万～0.6万；直接合并mask后每通道负均值使旧负边单条权重只剩约9.3%～20.2%。集合子集保持不等于保护梯度预算保持。为验证“新增保护”而非稀释旧保护，独立配方保留原KL不动、另加同表达式的added-only负KL，显式新增项系数与版本；空新增必须回退旧目标和梯度。见`docs/AFFINITY_RETENTION_GUARD_20261004.md`；这是本轮启动前风险与配方选择，尚不证明额外项有效，不晋升全局规则。
- followup_retention_trained: 独立新增项32更新／96次同输入训练完成，旧控制复用、真实梯度与终态严格重载核验。新增项恢复351两邻粒独立匹配，证明原保护域支持缺口的局部因果；但对R1固定689F仍−0.037165／丢33匹配、135面积10失格，658深部残片仅少1像素。s32新增参数梯度达到旧共同项42%～71%，且与local方向余弦为负，不能把失败解释成项未生效，不能因local loss下降推断内部尾部修复。证据同专题及私有`output/retention_guard_analysis/report.json`；已见不完整GT与单次短测仅支持机制，不晋级／扩训或推翻所有保留项方案。
- limits: 每源另有14–27次全图视野；原生未曝光不等于从未监督。原裁窗普查量增强／缩放前的覆盖，后续512格核验另列；标注接触不自动等于真实可见弱晶界。同一接触跨分区的569对中，边缘相对内部曝光差中位仅−1.93个百分点，不把聚合约六倍差解释为单界面或学习因果。

## 20261004-hard-instance-teacher-support

- status: observation
- last_verified: 2026-10-04
- scope: 同32训练源、R1起点、冻结上游的manual／teacher五轮几何来源对照；相同有效域、输入和部署。
- finding: 最终实例ID作为硬教师，会把终态同实例内未造成分开的旧连续边界响应改为连接目标；教师零缝若共同ignore，还可能使已有分隔监督退出。较暗热图、较低自身loss或更多负标签均不足证明分隔变好：负关系的位置可以同时发生合并与新增切分。
- evidence: `docs/AFFINITY_GT_SOURCE_20261004.md`；320步实际输入配对，四固定图B平均边界概率较A低8.1%–10.6%，原GT深部及分隔带同时减弱；已有75图B既更多拆分也更多合并。32whole512目标哈希复现后，teacher0共同删去18.60%原GT短程断开；排除filled的FF断开关系中另36.33%被教师改连接，同时新增原FF同粒断开，负标签总数反而+69.24%。
- reuse_hypothesis: 教师监督实验分别检查ignore造成的支持缺失、关系答案变化、连续概率与最终拓扑；恢复零缝支持和增加可靠断开保留作为独立变量，不把总负样本数量或显示线宽当作用力度。先复用短跑固定小样本，正式候选再完整推理。
- followup_support_preflight: 固定1原生像素最近归属、两种教师ID邻接的全32窄缝预检保留8,165个已有实例，恢复87.83%零缝像素、此前忽略原GT断开的71.91%，但同时恢复241,287条原同粒／教师断开关系。恢复合法域不会自动提高标签正确性，需分别报告原GT一致恢复与分歧；旧合法答案虽不变，per-channel正负归一化分母仍改变。见`docs/AFFINITY_TEACHER_SEAMS_20261004.md`；不晋级经验或全局规则。
- followup_support_e5: 同R1／旧B实际输入的320更新已完成，旧答案、冻结及strict重载通过；共同SAM2训练断开specificity回拨0.3984百分点，但四源冻结分隔带未一致恢复、固定188F最终IoU几乎不变且测试小F增加，另两预随机非monitor无一致收益。预检恢复大量合法关系不保证最终结构改善；日志恢复比例不能分摊剩余误差的因果占比，也不能据5轮判饱和。用户据阶段收益／成本搁置伪GT并返回此前外围保护方向；这不是伪GT普遍无效的结论。不晋级经验或全局规则。
- verification_gap: 支持修补5轮已完成，无独立验证集或本轮官方回分；GT端点关系不等于重新确认的物理边界，原生窗口计数变化与whole不同，不能外推为所有尺度同方向；未训练断开冲突保留变体。
- limits: 保持既有允许数据、人工确认和未知ignore边界；不从测试图推断GT，不凭诊断替换最佳部署，不晋升AGENTS全局规则。
## 20261008-unsupervised-provenance-evaluation

- status: observation
- last_verified: 2026-10-08
- scope: 纯无监督路线的初始化、教师、评价和模型选择来源审计。
- finding: 脚本名含SSL或论文称无监督，并不保证默认执行链完全不读人工标签。
  与表征梯度隔离的监督探针，不等于表征训练被GT回传；但若据其选权重、阈值或配方，
  仍可能引入选择依赖。需分别核对梯度路径、数据读取和模型选择，不能只查loss。
- evidence: 本地`tools/pretrain_lora_ssl.py`默认eval_every=5，探针加载监督decoder与人工验证集；
  no_grad且best按重建loss、末态固定导出。STEGO官方训练代码还含标签读取、detach的监督probe、
  GT簇匹配及GT cluster mIoU checkpoint选择。详见`docs/UNSUPERVISED_RESEARCH_20261008.md`及其原始来源链接。
- reuse_hypothesis: 新严格路线关闭GT探针并使用独立允许清单；旧权重同时审计训练和人为选择历史。
  只证明checkpoint哈希相同不足以证明纯净来源；不因发现GT评估就断言其进入SSL梯度。
- verification_gap: 组委尚未澄清冻结后GT评估、自动mask或人工质量筛选的认定；本观察不代替正式资格判定，
  不晋升AGENTS规则，也不改变有监督主线的训练／部署要求。

## 20261023-linked-workspace-source-preservation

- status: observation
- last_verified: 2026-10-23
- scope: Windows实际源码快照导入Linux新Git worktree；原实验实体原位保留。
- finding: 尾斜杠目录ignore不一定匹配新建的目录符号链接，应在新仓库以准确、无尾斜杠路径补充本地exclude并逐项check-ignore。原字节保留CRLF时，Git 2.34.1即使正常status或update-index --refresh也可能多报M；实际diff与filtered hash应先区分内容改动和索引stat差异。
- evidence: `docs/SERVER_WORKSPACE_RESULT_20261023.md`；私有`output/workspace_setup/reference_audit/stat_refresh_receipt.json`与`new_entries.json`。仅对filtered hash等于index blob的准确tracked名单刷新stat后，前后ls-files --stage完全一致、869/1025源码SHA不变、真实17/1修改与187未跟踪源码保留，stage空；106新链接和两个实际保密hook通过。
- reuse_hypothesis: 保留历史字节和路径优先于让目录看起来整齐；新入口既要验证resolve目标，也要验证Git是否忽略链接。CRLF导入出现大量M时，先核内容而非批量转换换行或暂存真实改动。
- verification_gap: 本轮Git版本及两工作树已验证，不外推到所有Git/filter配置；不是允许宽泛git add的规则。只处理已证明内容与索引一致的明确名单，不触碰原工作树或真实未提交改动，不晋升AGENTS。

## 20261025-gap-origin-through-frozen-window-trace

- status: observation
- last_verified: 2026-10-25
- scope: 固定N0、三张旧测试图20个原生窗口；只捕获响应并重放实际融合、插值、拼接和最终输出，不使用测试GT。
- finding: 近闭合强响应缺口不能仅由最终热图归因于拼接或marker。此次主缺口在各覆盖窗口的短程top2中已经低于现有弱阈值，长程门控接近全开；原网格邻点也偏弱，实际融合与拼接只是进一步降低。另一个位置确有融合稀释，但最终仍超过弱阈值，不能据其局部降幅认定整环失效原因。该结论只定位到恢复器＋编码器／affinity共同链路，未隔离某个头或某个训练目标。
- evidence: `docs/SEMANTIC_DINO_20261025.md`；私有`output/gap_trace/summary.json`、`roi_statistics.json`及远端逐窗数组。八方向SAME显式转CUT，融合网格、原生插值、BlendMap、边界缓存、最终实例／类别、全图分支及模型摘要均与原N0逐值兼容。
- reuse_hypothesis: 按原始方向→短程归约→门控融合→插值→重叠窗权重逐层追踪，在实际部署输出兼容后判断哪个环节丢失证据；避免用单方向高值、各窗极值图接缝或单个阈值跨越代替完整连通分析。
- verification_gap: 三张图只作机制和工程诊断，没有GT准确率、官方收益或不同尺度／输入／骨干的归因；不能据此全局降低阈值、改最大融合、强制补环或认定融合普遍无害。不晋升AGENTS。

## 20261025-synthetic-boundary-restoration-domain

- status: observation
- last_verified: 2026-10-26
- scope: 随机单通道残差补边五轮；处理后32源GT合成配对，固定N0三旧图原生连续热图与原reflect终态。
- finding: 合成断口/假线误差下降与真实分割改善可以分离。此次固定和64新随机配对均有效，真实关注缺口也实际分开，但另一真实图在F像素近似不变时F小片大增。只看热图强度、净实例数或合成损失会掩盖修复与新增碎分割并存。
- evidence: `docs/BOUNDARY_RESTORE_20261025.md`完成分析；私有`output/boundary_restore/analysis/`的训练审计、单调映射、响应域与最终关系统计。只按像素原强度的单调映射解释63%–73%输出变化，模型仍有超出映射的空间作用；.45响应带宽中位数合成6.00px、真实12.65px，低于.01像素比例86.41%/0.0369%。真实e5生成沿宽边界的双线/小环，不能把全部归因为统一增亮。
- followup_naf_width_intervention: 独立随机17.632M NAF的s5000在四固定图将147/403/124/363实例降至2/1/1/1，二值适配单独控制仍175/423/126/390。固定权重CPU三训练源同图干预中，GT原细线正位置输出均值0.996–0.999，只有膨胀半径2后原线中心低于3e-8，半径2邻域最大均值降为0.370–0.429；真实N0响应带面积/骨架长度约10px而GT约2px。N0仅细化后中心线强响应恢复，两个测试中心窗同向；直接连续输入未修复。说明容量和人工缺口过拟合不覆盖真实输入表示验收，域差不能只归测试来源。尚未证明细化后完整划分更准；细化也改变线位/支路，需同预处理无NAF控制。固定六配对gap仅有效域0.1406%、s5000缺口响应0.296而全域MAE0.00197，须分开稀疏目标预算与原线保持。证据`docs/BOUNDARY_NAF_20261026.md#第5000步弱响应诊断线宽失配已得到直接支持`及私有`output/boundary_naf/analysis5000/`；只有33个CPU窗口变体、非完整GPU/官方收益，无新训练或当前拓扑干预，不晋升AGENTS。
- followup_naf_final_readout: 原配方6000完整结束后，四图二值接法仍2/1/1/1；64已见源配对按源gap MAE0.629→0.529。额外冻结GPU对照将整图二值N0先统一细化再分窗，原N0终态与八条原NAF二值输出SHA全部exact、权重不变。仅细化与细化后3×3膨胀都未使116既定内外点最终分开，NAF5000/6000却分开；强0.65原始背景仍连通，必须区分概率屏障与marker膨胀/最终分隔。细化控制四图175/456/142/436，NAF6000为234/583/161/509；预测F均面积降4.1%–19.9%，不能把恢复强线/分开一例当完整收益。恒等细线保持、稀疏gap恢复、真实输入表示与新增线可靠性是不同门槛。见同文档6000末节及私有`output/boundary_naf/analysis6000/`；无新训练、测试GT、150图候选或官方分数，不晋级或把阶段收益单归PH。
- followup_naf_gt_and_samples: 全32训练源同GT审计中，s6000原标域新增骨架6.84%距记录边界<=4px、82.79%>8px深部，填补域83.99%近界；来源落点与支持界面OO/含填补必须分开。仅深部响应不等于切分，须在完整同GT-safe域看子块，再用移除新增深部屏障的诊断反事实排除共现；395个双原标支持旧分量（375个GT）严格成立，但原标仍可遗漏，非物理错误率或最终WS成绩。与此同时精确重建96k样本访问、计划/日志匹配，可靠边界目标曝光99.975%、实际擦断87.327%，不能把失败首先归为没有采到；裁窗/D4不等于独立源，32源不保证泛化/收敛。gap像素访问83.86%落filled也不是错误率或参数梯度比。真实N0假支路/粗细/底噪未出现在GT擦断输入，需优先检查任务域覆盖。证据`docs/BOUNDARY_NAF_GT_20261026.md`及私有`output/boundary_naf/gt_audit/`；12构造测试通过，GT来源遮罩不另造连通域，未知忽略。没有新训练/测试图/官方评分，不晋升AGENTS。
- reuse_hypothesis: 热图恢复首先检查真实输入线宽/底噪/幅值与合成配方，并用不看空间的强度对照、训练源新随机配对、固定终态及分类别拆分/合并诊断分解作用。优先试真实热图自损伤恢复其自身幅值和宽度，可作为隔离输入失配的实验，但不能默认纠正教师原有错误。
- verification_gap: 没有本轮真实测试GT或官方成绩；域差与副作用同时观察到，未隔离哪项为唯一原因。64新配对源图参加过训练，单调映射只是输出变化分解，不是准确率/候选提交；后续真实热图短训见下方followup。不晋升AGENTS或替换最佳。
- followup_same_source_domain: 后续80源同图/跨域对照确认两层失配同时存在：32GT同图native512的GT/真实N0 .45带宽为6.00/10.10px，原始标注层仍成立；GT采样插值控制为6.32，单窗/实际拼接带宽配对差中位为0。其余24训练与分层24旧测试真实N0又为10.00/13.11px，不能把真实域差全归合成。原标注4–8px距离带响应GT/控制/N0为.0347/.0544/.3737，显示宽软尾部；峰值则相近。详见`docs/BOUNDARY_DOMAIN_20261025.md`。这些观察不隔离采集倍率、模糊、组织/曝光或模型各环节，不提供e5副作用的因果贡献率；真实热图配对训练仍未启动。
- followup_real_self_corruption: 五轮真实N0自损伤后，同目标64配对断口MAE旧/新为.1911/.1022，完整输入误改.1061/.0133；反向合成配对则旧更好，支持表示适配、不是新模型普遍更强。固定166的原单峰剖面多峰606/1753降到6/1753，F<500为65→1；但101为4→14，116原缺口点.324→.566后环内外仍同marker/最终ID。保存教师原幅值能抑制大幅误改，却不给教师既有漏界新的答案。固定假线中心去除还可伴原样邻域新增亮斑：须用input==target排除低于mask门槛的腐坏尾巴，再报邻域而非全域均值。实际noise单像素附加损失/普通valid中位1120倍提示稀疏区域独立分母的风险，不是参数梯度倍率或已证实因果。证据`docs/BOUNDARY_REAL_20261025.md`、私有同目标配对/clean64/原固定读出；无测试GT或官方收益，不晋级AGENTS。未来应区分局部恢复、既有响应保持和真实遗漏纠正，不以删小块或加epoch替代目标设计。
- followup_rgb_induced_response: 8源/24ROI的原生sigma2/4均匀RGB模糊→固定N0，零档16窗与旧缓存exact、实际注入噪声同字节；新变化包含原弱线增强、错位及弱化，不能把所有扰动当单向退化或把原图教师视真值。54对原闭合区域在两档分别仅1对、2对合并，sigma4另12对不确定；真实补边e5的合并变0却13对不确定、配对MAE反升，说明删去受扰锚点后分母缩小可造成伪修复。所测一维幅值对照仅解释约3%变化能量，不支持主要归全局重标定，但个别粘连仍可由幅值映射触发；也未证明新增RGB一定有效。证据`docs/BOUNDARY_RGB_20261025.md`、私有capture/analysis及48次严格旧模型配对前向；没有GT/测试准确率或新训练，不能以八中心ROI否决所有模糊配方。下一候选为RGB输入单变量，MALIS须先核实际部署对应，不晋级AGENTS。
- followup_rgb_context: 固定目标/损伤/共同随机层/五轮预算，仅加原RGB的实际对照中，模型确在使用RGB，完整输入到N0的偏移.01328→.00897，但人工断口误差.10219→.10871、22/29有效源退步，116关注环e1/e3/e5均未分隔。额外信息与训练loss下降不等于纠正教师原有错误；原目标漏线没有新增正答案，应先检验独立结构监督，再考虑扩输入或轮数。此结论只适用于当前自损伤目标，不否定RGB、其它配对或所有补边方法。证据`docs/BOUNDARY_CONTEXT_20261025.md`同配对/完整输入/原读出诊断；32源参加训练，CPU native与GPU计数相符但热图SHA不同，无官方准确率，不晋级AGENTS。
- followup_dualcgb_fidelity: 核对原文后确认当前补边器仅借鉴恢复思路，并非Dual-CGB复现。
  原文包含前端EGD/NormSkip抑噪、约17.63M恢复器、GT破损→完整GT的BCE+GAN训练、
  后续持久同调微调、迭代修复与未连接扩展回退；4340的五轮约53,000更新，不能与当前320更新
  按epoch等量比较。原文消融也有单加恢复器显著退步，且组合并非所有指标都升。
  此次跨设计对照不能归因失败于单一缺失模块，也不能仅加轮数冒充原文验证；N0自复制目标与
  GT纠错目标的差别需显式保留。证据为原文doi:10.1016/j.mtcomm.2026.115965的2.1–2.5节及
  表4/5/8/11，对照见`docs/BOUNDARY_RESTORE_20261025.md`原论文核对节；仅候选知识，不升AGENTS。
- followup_core_relations: 全32已见源、52异GT可信内核对的冻结反事实显示，处理后共享界面位置参照使39对消除预设显著共享、原同主体32对中25对主体分开，但严格零共享仅24对；将任意小共享对应的整marker画紫，会把54px／0.261%残留放大成主体仍粘的观感。严格零共享、主体分隔、实际泄漏量及登记原标内核副作用必须并列，不因像素级门禁未过而否定关系监督。六例扩已知外围使修改量3.38倍，严格残留未消、内核伤害增加，不支持仅扩大修改域；这些例的残留在两GT已知并域内已存在，不能先归第三GT／unknown绕行。原子稳定只覆盖登记内核，旧0变正的新跨GT共享需额外核对，内核外小种子不在保证内。证据`docs/BOUNDARY_CORE_20261025.md`、私有`output/boundary_core/analysis/`与`neighborhood/`；25项CPU、封存test116最终读出exact桥接及32源marker重放。属于机制观察，尚未训练可微关系目标、未验证泛化或官方增益，不将派生0.9界面当可靠新真值，不晋升AGENTS。

## 20261025-source-namespace-content-alias

- status: observation
- last_verified: 2026-10-25
- scope: 已处理32GT原图目录与允许1000训练池之间的来源绑定、抽样排除与固定索引。
- finding: 两目录沿用相同train编号样式，但同basename可能是不同图片，编号对应也不是固定偏移。此次32GT原图各自通过已有SHA，并在1000池按文件SHA唯一匹配；按名字做“同图”或排除已标注来源会错配。原图文件SHA绑定正确的既有训练不因此失效。
- evidence: `docs/BOUNDARY_DOMAIN_20261025.md`；私有`output/boundary_domain/gt_pool_identity.json`的1000内容索引、32唯一对应与按内容排除集合，以及`selection_audit.json`。同图GT与实际N0中心RGB逐值相同，三桥接旧N0完整ROI相同。
- reuse_hypothesis: 跨目录同图比较、人工源排除与seed/index复用先绑定内容身份，再记录各目录展示名和完整源池index；不同命名空间的派生产物用显式别名区分，不能靠重命名真实数据补救。
- verification_gap: 本轮32源均有文件SHA完全一致对应，没有研究重新压缩但解码等价的图片；不允许把近似视觉匹配默认为精确身份，不修改AGENTS或原数据。

## 20261025-gt-boundary-support-density

- status: observation
- last_verified: 2026-10-25
- scope: 已处理32源GT、完整真实N0原生热图，正边界纠错候选；不涉及测试GT或新的人工标注。
- finding: 原标像素覆盖率不能替代晶界监督覆盖率。此次原标覆盖90.42%图像像素，但79.66%处理后GT边界中心落在filled；侵蚀原标联合域8像素后只剩6.26%骨架。仅严格域零候选不足以否决GT正监督。允许最近原标同实例且≤4像素的薄filled后仅4段/4源，放开来源约束的观察对照也仅8段/6源，多数被邻近偏移响应检查拒绝；已见训练源的GT半径4强响应匹配98.90%，难以给冻结后端新增漏界答案。
- evidence: `docs/BOUNDARY_CORRECT_20261025.md`，私有`output/boundary_correct/analysis/`严格、thin_fill与processed_upper分立回执；构造器86554d2a…07ad3f55，21项CPU通过及独立正例/反例复核。观察域必须包含known-safe中未获GT正监督资格的旧响应，否则会漏检旁边已有线；仅看缺段中点也会漏掉端部偏移线。
- reuse_hypothesis: 构造GT纠错时，分别统计像素/边界/实际修复/有效分隔分母，并用不改变其它条件的来源对照判断是支持域排空还是已拟合监督缺新信息。不要用重复裁少数已见短段充当多源新监督，或关闭防双线检查凑数据。连通检查应绑定补线核心而非任意提亮尾部；核心与两锚同网络仍可能只是单端附接原网络，要先报两锚原本是否已连、再看两侧区域是否真正分开，不能把同网络算新增双端补通。
- verification_gap: 当前计数依赖短段长度、锚点与阈值，不能推出GT或所有补边方案无效；processed-known对照会改变路径分段，不是候选数量的数学上限。三探针仍连通也不能证明局部增补全错。没有新训练、测试准确率或官方收益；不晋升AGENTS，不替换N0。

## 20261025-cuda-resume-exact-probe

- status: observation
- last_verified: 2026-10-25
- scope: Torch2.8 CUDA、RGB补边器、FP32及旧TF32设置下的两更新后第三步续训检查。
- finding: 相同初态/输入的前向与loss逐值相同，CUDA反向更新自身重复仍可能使参数SHA不同；先比较自身重复本底，再判断是否恢复错误。本次继续/重载权重maxdiff1.669e-6，与两个重载分支自身重复1.738e-6相近，梯度分别7.02e-8/7.49e-8。仅在第三步核验临时启用确定性算法后两路梯度/权重逐值相同，允许保留严格SHA门禁而不改变正式训练数值模式。
- evidence: `docs/BOUNDARY_CONTEXT_20261025.md`；私有`output/boundary_context/diagnosis/resume_probe.json`、`deterministic_probe.json`及原失败smoke现场。实际48参数/优化器状态完整、更新前SHA及前向exact；29项CPU回归及独立AST审计证明修复只作用于短测第三步、正常/异常均恢复enabled与warn_only，不改TF32或正式训练。
- reuse_hypothesis: 严格续训验收须分清状态加载、前向、反向和优化器更新。未经重复基线诊断，不直接放宽容差或单凭SHA不同指控checkpoint损坏；若仅核验需要确定性，scope退出必须恢复设置，避免污染控制实验。
- verification_gap: 当前证据没有定位唯一非确定算子，不保证任意设备/算子均支持确定性模式，也不证明正常正式训练逐位复现。这不是D5a推理噪声的消融或解决证据；不晋升AGENTS。

## 20261025-auxiliary-gradient-conflict

- status: observation
- last_verified: 2026-10-25
- scope: 随机初始化RGB补边器的连续路径附加监督，五轮；固定16个已见源、两种输入、八批e5冻结参数梯度，不涉及测试GT。
- finding: 附加目标非空、总loss下降和人工损伤恢复改善，不足以证明新增纠错目标有效。此次约98.5%的已满足路径位置仍进分母；防泛亮项的loss系数虽只有0.05，其实际参数梯度在八批均更强且与补线路径反向，合计附加梯度也反向。全局LR、同乘的梯度裁剪或整体放大附加权重不能修复内部相对方向。应先分解真实参数梯度，再选择单变量配比，而非仅看loss标量大小或继续加epoch。
- evidence: `docs/BOUNDARY_STRUCTURE_20261025.md#五轮完成后的分析`；私有`output/boundary_structure/mechanism_report.json`和`mass_calibration.json`。31,334位置中477个未达0.70；八批合计附加/path余弦-0.467至-0.205，mass系数0.005的已存梯度校准为0.942至0.996。64同目标配对人工gap误差按源等权0.10909→0.09730，但116自然缺口完整GPU读出仍未分开；七图热图与原monitor逐值兼容。
- reuse_hypothesis: 辅助损失没有明显改变目标行为时，分别记录有效纠错分母、各项参数梯度范数及方向、损伤任务与天然错误的变化；只改被证实竞争的相对系数，保留原控制，不将局部梯度校准当作新模型收益。
- verification_gap: 冻结位置的普通梯度一阶分析不是Adam真实更新重放，不证明所有训练批次均竞争，也未由该校准验证0.005的训练收益。未隔离分母稀释的最终因果份额，开放链也不保证完整分隔。没有独立准确率或官方增益，不晋升AGENTS或替换部署。
- followup_mass005: 同随机e0／同320步的0.005单变量五轮确使新e5八批结构/path恢复同向（0.920–0.998），64配对人工gap约降12.9%，七图双峰／低值抬亮和F小片减少；但原样训练图359弱位置的e1改善到e5回落，116实读出仍未分开。新e5原样输入四批base/path反向且base范数大得多，说明修复一个辅助项内部竞争不足以保证原始纠错任务学会；后续须同时审查主任务共享参数竞争、有效监督分母及真正分隔。训练clean强线MAE增加而测试原强线掉阈反减（18→4），也说明到教师的无符号偏移不能代替方向／阈值／最终读出。证据`docs/BOUNDARY_MASS_20261025.md#五轮完成后的分析`与私有paired/mechanism/native_gpu回执；未作全过程Adam因果重放，没有GT或官方收益，不能把未迁移唯一归为测试域差，不晋升AGENTS。
- followup_naf_gan20261026: 真实N1骨架→处理后GT的NAF固定16批512步中，BCE＋GAN在已有线/漏线上退化并出现点阵；失败权重的瞬时全局BCE/GAN梯度余弦仍为+0.169，合计BCE投影为纯BCE的1.126倍。保持同初态、同批次、LR及预算，仅关GAN后原三类门槛全过，GT中心响应约0.999996、背景约7.1e-8、深部输入线r1响应约3.1e-8；64次正常换批也数值稳定。说明单时刻梯度同向不能排除对抗训练耦合动态造成拟合失败，固定批GAN失败不能直接否定容量/可学性；需实际单变量干预，而非仅数学缩放梯度。邻域max也不能单独证明线连贯，应同步检查精确GT像素和图样。固定批只训练draw0–15，monitor里更大索引不能当同批拟合图。证据`docs/BOUNDARY_NAF_REAL_20261026.md`及私有`output/boundary_naf_real/{g0_smoke,warm_complete,gradient_probe_complete}.json`；只支持本种子/批次，不证明GAN普遍有害或测试收益，不晋升AGENTS。
- followup_naf_real_final: 随后同GAN0配置的6000步正式训练数值稳定，却使四图最终全部1实例；32源GT-r4峰从旧同N1的0.977降为0.285，精确GT像素BCE却从0.355降为0.0469，31源>.65全空。固定16批拟合与换批64步稳定不是最终幅值/分隔验收，精确薄线loss更低也可能伴宽弱平均响应。GT正线虽仅1.76%，正loss对全域贡献仍约2/3，不能由像素比例直接推断无有效正梯度。预定两源的GT4px附近输入骨架中75%–79%落于精确背景目标，支持位置差异解释但未证明因果；五档单调校准恢复部分test116分区而仍未补既定缺口，排除“简单调亮即已可用”。证据同专题完整训练分析、私有`analysis/{geometry_summary,probability_summary,calibration116,positional_probe}.json`。不据此否定所有真实输入纠错，不晋升AGENTS；下一步先检验位置容差监督候选，而非盲目增加epoch、正类权重或全测试推理。

## 20261025-boundary-training-timing

- status: observation
- last_verified: 2026-10-25
- scope: RGB结构补边小网络、FP32、有效batch4／microbatch4；只读已有320步日志与过程图文件时间，不改变正在运行的mass消融。
- finding: GPU显存闲置不能直接推出扩大有效batch会等价提速。此循环的step_seconds在同步构造batch之后才开始，漏掉逐窗界面／骨架／候选几何；每条路径还重复校验整张预测／目标，normalizer与loss重复检查固定几何，预测依赖的选路需CPU/GPU同步。monitor保存缩略图但实际仍推理七张整图并做CPU读出，不能用JPEG尺寸判断其成本。
- evidence: `docs/BOUNDARY_MASS_20261025.md#下一轮的速度优化候选`；`train_boundary_context.py`的`_batches`与计时起点、`models/boundary_structure_loss.py`的`_validated_map/select_path/_support`、`tools/boundary_context_monitor.py`的逐tile推理。已完成控制总1224.29秒，已计step合计371.55秒，四次monitor时间差合计322.27秒，剩余530.47秒未细分。
- reuse_hypothesis: 先覆盖数据准备／CPU动态选路／前后传／monitor／保存的完整计时，再尝试静态校验复用、确定性预取和monitor批量tile；保留随机流、有效batch、更新预算与数值模式，验证输入／选路／更新和最终读出，不先用更大batch或AMP改变对照。
- verification_gap: 文件时间和已有日志不是精确profiler，未证明各个候选的独立耗时或加速收益；未实施任何提速或额外GPU基准，不把全部剩余时间归给几何，不保证并行batch推理逐位一致。不晋升AGENTS。

## 20261026-complete-separator-feasibility

- status: observation
- last_verified: 2026-10-26
- scope: 真实N1输入、随机NAF、原标可信内核的完整8邻接分隔选路；固定16视图128步短测。
- finding: 内部平均响应很低仍可能有少数高点挡住静态种子，使完整分隔对被拒绝；只看均值会漏掉有效监督停止。保留静态种子和严格拒绝条件，将内部保持损失集中到未满足低响应条件的位置后，短测后段种子阻挡减少，60对固定关系分开17对。不能通过随预测删掉/换掉种子把失败改算成功。
- evidence: `docs/BOUNDARY_REGION_20261026.md`；私有`runs/naf_region/{smoke,smoke_tail}`、固定关系明细及选路原因。另在32视图中，完整可编辑域都挡不住的固定绕行可先精确拒绝；10组落点不变，选路从73.69秒降至9.36秒。
- reuse_hypothesis: 关系监督需同时记录静态身份、被拒绝原因、实际有效梯度和稀有冲突位置；保证完整分隔时不得删除未知节点或拿裁框封口。优化选路应先证明位置不变，再报速度。
- verification_gap: 这是固定批可行性，不是测试准确率或完整晶粒分隔收益；短测test116关注缺口仍未闭合。全图marker/WS读出还需正式阶段检查，不能据热图和关系计数晋级，不晋升AGENTS。
- followup_formal1000: 1000步/32源均衡访问结束，未塌陷但固定116仍未分开；新输出骨架为输入99.9826%、强线面积4.6687倍，普通3×3膨胀已解释大部分GT邻域覆盖增量。四图实例1027→1191→1203分别对应N1/仅骨架/网络，不能将适配变化全部归给模型；raw8诊断的数万微片也被纯加粗复现。先隔离输入表示、线宽与最终marker/WS，再讨论补结构收益。固定训练缺失559位置过阈率s250的64.22%到s1000的24.33%，409个记录GT深部输入位置仍全过阈；静态种子/完整割可行性不能保证足够补边监督或清除旧假线。详见同专题完整分析及私有geometry/width_control、ferrite_splits、target116回执；GT已见、可漏标，没有官方新分数，不晋升规则或部署。
- followup_seed_ancestry: 用户指出101强边界旁小块。一个seed的anchor到膨胀阶段才落入封闭背景，不能证明整颗空腔由该阶段创造；需全种子映射各阶段全图背景祖先。169的55个原N1零低支撑seed中，52在强阈值时已有封闭祖先，细化时却均并入触边共享祖先，再膨胀/封边恢复；“此前未闭合＋零支撑”只选3且偏向图边，不宜直接作硬门槛。NAF为全部428个seed生成了低谷，不能用其自身低谷证明可靠；零原支撑与较大原内核应区分，101的2030px定位块有1175px原低核心，不能靠同一去微洞规则删。作用于内核的50px即便复用已有数值也是新语义门槛，需独立验证。证据同专题出包与整种子诊断、私有`output/naf_submit/receipts/`。没有测试GT、过滤后得分或通用安全证明，不晋升规则。
- followup_zero_support_filter: 8源四臂（N1/NAF×滤前后）在真实reflect seed上撤销原N1≤.45像素为0者，WS高度/语义/raw分别固定，基线最终全exact。六测试小F为N1 40→38、NAF 62→38；无整片旧实例变0，新增0半径≤1，但101大F2030→2058仍保留。已见GT两处新共享仅15/10px与7/15404px，不能夸成主体误并，也不保证细通路安全。说明读出须与重建表示一起做隔离对照，撤种子与末端删实例不同；小块减少不等于准确率、相同总数不等于相同对象。证据`docs/MARKER_SUPPORT_20261026.md`及私有`output/seed_filter/review/`；35项CPU、实际32次小样本读出，未做官方验证，不改默认或晋升规则。
- followup_official_unfiltered: 用户报告同赛段未滤Region s1000 mIoU/面积0.7403/0.8821，比N1总分低1.235，净TP−74/FP+2750；平台预测总数20455与原ZIP吻合，未核平台提交哈希。150图重读显示F像素仅−0.204%、F数+3.35%，127图预测F均面积下降，符合微分区风险；不能把净增FP逐个归给新碎片，也不能用池化均值反推逐图面积分。正式结果不支持把响应增强/变细当有效补边，先等独立filtered回分、降低本版NAF密扫优先级。只胜过较差原NAF不足晋级，须同时对N1；filtered尚无官方分，不否定所有补边方法或升级为全局规则。证据`docs/SEMIFINAL_NOISE_SCORES_20261026.md`及私有`output/naf_submit/score_20261026.json`，原包不变。
- followup_positive_gate_audit: 2026-10-27复核最后S2及旧/H05/L2实际回执，四组1000步均关闭精确GT BCE/GAN/PH；保线只覆盖原输入骨架附近的GT支持点，天然缺口只有被动态选入活动分隔才获直接正监督。GT允许走廊不是完整正标签，合法raw8分隔也不证明最终WS原先未分开。后100步每16样本仅平均选中1.27–1.81对，不能将此比例当缺口覆盖或梯度占比；损失独立归一且全程裁剪，未测逐损失梯度归因。RGB增加信息不能自动弥补监督入口限制；改输入表示/监督后须单列归因，再做同目标RGB对照。证据`docs/NAF_TRAIN_SWEEP_20261026.md`审计节及私有`output/naf_sweep/rgb_preflight_audit.json`；未新训/前向、未证明各因素因果，不晋升全局规则。
- followup_tolerant_degeneracy: 2026-10-27新RGB截面目标的纯合成反例显示，单直线、带内附着两环和64个横向交替离散点，均58/58覆盖且重建loss同为4.1223e-9（不含GAN）；逐截面max及带外负项不保证峰间连接或带内单线。正式RGB末100步全部正点覆盖97.333%，天然缺失点加权覆盖仅201/15329=1.311%；缺失点只占共同正分母0.9623%。提醒区分监督入口、困难子任务预算、覆盖与结构可辨识性；不能由标量覆盖或固定批过拟合宣称闭合。证据`docs/BOUNDARY_RGB_NAF_20261027.md`末节及私有`output/naf_rgb/analysis/{cached_audit,tolerant_degeneracy}.json`；未证实际所有伪影唯一原因，控制仍待完成、无官方分，不晋升全局规则。
- followup_rgb_pair_raw: 随后同配方T-only1000完成，逐步data/LR/目标及源码均配对。4缓存×2NAF重算热图SHA/实例数exact，原始数值2×2相位均值极差RGB0.6085–0.6402、T0.2003–0.2483，N1仅0.00056–0.00179；点阵不只是JPEG。T的0.65原生微暗腔更多，最终小F却44而RGB143（N1为26），说明数字孔洞、周期性、最终实例必须分开评估。T更保守但两组116预登记内外仍同ID3，均无稳定补缺证据；据此选择T作作者式输入表示试验控制，不推广为RGB普遍无效。证据同专题完整横向节、私有output/naf_rgb/pair_analysis/{comparison,raw_complete}.json；无测试GT/官方分，未隔离PixelShuffle/GAN因果，不晋升全局规则。
- followup_width_final: 作者式输入T1000的4缓存终态exact，小F44→29但共同raw缺口仍0/161、116未补开。总实例1085→1011中P少65、F只少9，提醒按目标类别拆分后才能讨论面积风险。新2×2相位均值极差更大、同格局部棋盘对比更低，周期伪影的不同指标可反向变化，不能挑一项判全面改善。分区叠加也见较大区域合并/拆分与改类，少分区不自动等于少误切。证据docs/NAF_WIDTH_TRIAL_20261027.md终态节与私有output/naf_width/evaluation各回执；固定4图无GT/官方分，仅保留整形研究候选，不晋升规则或部署。

## 20261026-runtime-config-parser-identity

- status: observation
- last_verified: 2026-10-26
- scope: 使用冻结`utils.config.load_config`读取完整JSON运行配置、再与训练identity核对的独立NAF队列。
- finding: JSON文件经该YAML解析器读取时，`5e-05`与`1e-08`在此环境变成字符串，`json.loads`则保留float。直接将原始JSON对象与训练器实际解析后的配置逐值比较，会在训练正常完成后误报配方改变。H05仅两处有此差异；实际run/config与冻结解析器输出、identity.recipe完全一致，Adam eps消费者显式float转换且与控制相同。
- evidence: 私有`runs/naf_sweep/failed.json`保留原核验错误；`recovery_formal_check.json`通过修正后的解析身份及其余全部旧门槛，含1000步采样/LR、严格权重重载、六阶段monitor和四图基线exact；专题`docs/NAF_TRAIN_SWEEP_20261026.md`。
- reuse_hypothesis: 同时保留声明配置的单变量检查与“用训练器同一冻结解析器解析后的完整配置/identity”检查，不全局放宽字符串/数值比较，也不改旧训练器来迁就核验。完成的训练经全部原门槛复核后再接续，不因元数据误报重训控制。
- verification_gap: 这里只核实该解析器版本和两字段；其他数值消费者不保证自动转换，不能把任意字符串差异视为数值等价。不晋升AGENTS或改解析规则。

## 20261027-seed-core-final-filter

- status: observation
- last_verified: 2026-10-27
- scope: 冻结N1原生reflect种子、真实OpenCV分水岭及最终面积50的8源缓存候选。
- finding: 降低种子核心门槛但保持最终面积门槛不变，并不自动保护输出。25/50候选纳入23个小核心，21个终态仅52–123px；另两个先竞争分走像素，再因面积45/47被最终过滤，可能留下0而非让邻居接管。两张已见GT的可信内核新增18个0像素和1次碎分，未恢复原0内核。种子数量变化不能直接解释为有效实例召回。
- evidence: docs/SEED_CORE_20261027.md；私有output/seed_core/complete.json及integrity.json，原50/50种子及最终PNG/classes在8源exact，实际WS高程哈希相同，16份终态均通过最终面积50及uint16契约。
- reuse_hypothesis: 改种子生成时追踪过滤前watershed面积和过滤后核心去向，检查临时种子是否先侵占旧实例后被删除；不能只看最终仍有最小面积过滤或仅在post_process普通分支改参数，遗漏生产reflect替换。
- verification_gap: 仅6张无标签测试及2张已见不完整GT；未运行官方评分、训练或全量推理，不证明每个微块都假、不否定其他种子策略，不晋升AGENTS。
- followup_dense_marker_constraint: 只读同8源既有N1 core_audit与原尺寸，内部正marker覆盖全图93.11%–97.59%，像素合计31324915/32566016=96.19%，占最终正区域97.43%。原CV分水岭只把未标记像素入队；因此保留整片marker而只更换高程，大部分像素不会获得重分配自由。本批正marker已避开被CV强制清空的外框。后续应区分seed数量／身份、占地与读出方案；同数量收缩时至少保留最终50门槛所需的核心，否则会混入成品过滤。证据docs/MARKER_SIMPLE_20261027.md“后续定位”、私有output/seed_core/complete.json与既有capture manifest；此为几何自由度诊断，不是准确率，也不保证少量边界变化不能翻转整块类别。新的2x2方案尚未运行，不晋升AGENTS。
- followup_fixed_final_area_preview_20261029: 现有共享min_area同时筛核心和成品，不能为去微块直接整体调高；最终PNG又按面积重排ID，不可直接当WS marker ID。仅有终态PNG时，原0分界会让简单同ID重标留下孤岛；本次固定200的局部邻接合并显式回填同root接缝并保留其它大区正像素，8项合成和18真实候选契约通过，但无GT/官方增益。旧cap合并器会按数量停止、孤立区置0，不能直接拿cap替代面积下限。最外行不触边也不排除bbox距边1px的残片。证据`docs/FINAL_AREA_FILTER_20261029.md`及私有`tiny_filter/`；类别继承与同高程WS重票是两种读出，阈值分离与处理策略变更应分别说明，不据微块清理假说晋升AGENTS。
- followup_deferred_cleanup_contract_20261029: 若末端候选要合并并继承目标类别，就不能先在WS把这些小区删成0，也不能在合并后无声明地重新作raw投票。需保留旧drop缺省、显式延期到最终语义票之后，并验证marker/reflect/S0真实调用链，而非只看普通API张量。现有最终ZIP可用于同上游终态变换，不能借此声称重算种子或恢复源流程已删除的小区。真实169旧/拆分50/50/延期合并50三种最终ID与类别exact，18张T200与旧预览exact，证据同专题及私有`output/area_floor/bridge/compatibility.json`；只证明兼容与范围，尚无新官方分数。
- followup_official_area_tradeoff_20261030: 用户报告同8eb/S0的100→200净TP−6/FP−1513，200→500净TP−137/FP−2248；500虽为新族最高81.840，仍被旧20a59/S0的.7467/.9058两项同时超过。按半决赛专项公告FP不直接扣实例项、面积为逐图全部F均值误差，不能因P碎片骤减/Precision上升晋级。500全局F10383近旧10345、池化均面积仅低.258%，但逐图F数69多/55少相互抵消，官方面积仍低.0097；这验证池化接近不能替代逐图官方分。只支持本配方统一门槛的匹配代价和诊断边界，不证明所有小区是假或物理倍率不同。证据同专题及私有`score_20261030/`；旧部署保持，不把新旧上采样/擦断/ridge/GAN多变量比较归因单一GAN，支撑比例候选尚未执行，不晋升AGENTS。
- followup_support_ratio_risk_20261030: 旧20a59五例完整缓存的5%支撑门槛撤17核心；169同一函数比例0最终对原包exact，所检r4高覆盖GT内核对无新共占、预测F连通复核无新割断，却仍有train557小P289px在撤seed211px后部分被F接管：151个GT P像素中38原标/113filled。说明‘内核无新合并’或‘大主体不翻类’不能排除小结构相别损伤；r4侵蚀亦会排除细通路，需另查未侵蚀known区/通路及标签来源。train001的30个thin F→P全filled、29为填补P，不能仅凭亮色或填补GT认定全改对/全误伤。支撑少是原预测冲突而非假实例真值，撤种后的归属/投票同样重要；比例与面积不等价，5%不直接推广。证据`docs/SUPPORT_RATIO_20261030.md`与私有`output/support_ratio/`，两已见不完整训练GT/一个本赛段无标签及两历史图范围有限，不晋升AGENTS。

## 20261027-skeleton-connectivity-contract

- status: observation
- last_verified: 2026-10-30
- scope: 将连续边界B经0.5阈值、整图细化S后，接完整8邻接分隔监督和原冻结marker/WS读出的NAF支线。
- finding: 细化保持某种数字拓扑不等于保持另一种背景分隔。32源8前景组件数和4背景闭环数保持，但8背景闭环数均改变；闭环要排除触框组件，总背景组件数未必相同。完全不经过网络，B/二值T/S在冻结最终读出已产生不同分区（116为391/431/451、169为365/391/429）。因此先分开幅值阈值化、线宽/位置变化、原生连通和实际读出，不把适配效果全部归给恢复网络。
- evidence: `docs/NAF_LIMITS_20261027.md`及私有`output/naf_limits/{weak,adapter}/`；B参照与S既有monitor均exact，原语义/raw投票/reflect固定，无模型前向或训练。原标524169个GT线点中阈值缺失2.590%、细化位置偏差7.229%，后者不是原幅值本来弱。
- reuse_hypothesis: 接入骨架或拓扑损失时，显式记录前景/背景4/8邻接及边界处理，验证原生图与完整marker/WS链；先保留连续信息，再评估是否值得扩大负标签或新增恢复模块。
- verification_gap: raw闭环计数不是实际watershed或准确率，未证明数字邻接差异占用了多少梯度/正式失分；训练GT已见且可漏标，两测试图不能代表全150。10月27日该NAF支线封存不等于该网络架构普遍无效，不晋升AGENTS。
- followup_original_representation: 2026-10-27已核对Dual-CGB公开官方代码（n-okt/Dual_CGB的readme.md、config/train_res/316L_Grains.yaml、test.py、utils/image_processing.py）。恢复器输入确为1通道；作者训练/推理有细化后再恢复受控线宽，并非直接1px骨架。RGB进入前端检测器，用户提出RGB＋二值T→NAF→GT属于新条件扩展，不是恢复器原样复现；作者tau为二值票比例而非直接概率阈值。输入线宽、目标、极性和最终读出必须分别核对，不能凭发现差异断言旧失败主因。细节与新候选保存在docs/BOUNDARY_NAF_20261026.md末节，未训练，不改变旧封存状态或晋升全局规则。
- followup_width_cohorts: 2026-10-27核官方推理ellipse核316L=8、TBM=3、4340=4/指定5图11；训练按GT前景EDT70分位估厚，不可称作者统一1px或训练恒8px。同N1合同32训练/4固定测试中央原生1024的T>=.5典型2EDT宽10/11.83、P90为12.65/16.25、>16骨架点2.06%/11.18%；.65典型均8.25但尾部仍不同。提示实际二值入口阈值、肩部与宽尾需分开检查，骨架剔叉口后另查交汇尾部；核边长不等于宽度代理，ellipse8在轴向/斜向为8/10。N0历史24额外训练/24旧测试及4新测试另层同向，不与N1混池。证据docs/NAF_WIDTH_20261027.md及私有output/naf_width/各回执；固定测试不是150随机总体，未作模型线宽因果干预、无新推理/训练/官方分，不晋升全局规则。
- followup_actual_cut_gate: 回原affinity的同64视图预检中，静态seed从dS>4换为固定原B<=.45后，合法割2→16、raw8新增分隔1→11，说明dS距离不能冒充连续B低响应资格；旧政策85次被挡中两seed墙像素比例中位仅0.1999%。但按顺序预封的171实际原WS已分开，27px理想cut没有目标收益、却新增252px P片（232px落记录core），触发保持门槛。现有处理已修好的raw8漏路不能直接作为新的待补边目标；先以真实最终实例错误取得资格，再看理想干预和head梯度。此例不证明物理假边或官方扣分，原GT可漏标；完整理想割证书也不能直接授予开训资格。证据`docs/AFFINITY_GAP_20261027.md`、私有`output/affinity_gap/ideal_v2/final_gate/`；按规则早停而非异常，没有C/A或过滤网络训练。
- followup_iterative_readout_20261030: 两step1000恢复器各五例1/2/3轮、固定语义及原N1 S0，十例一轮概率与缓存exact且169终态各对正式包exact。直接递归使旧组小F<200由7升至2130，NoGAN16升至188；纯T8空循环不幂等，但三张测试图闭孔略减，不能独立解释碎区膨胀。169第三轮旧高阈值背景8连通无闭孔、4连通却有密集微孔，经bridge及细化后扩宽产生大量可存活小腔，重建S0 markers与保存结果exact。提示固定读出并不意味着它不会放大新输出的微结构；必须分别核输入规范化、数字邻接、读出阶段和最终实例。新增小F均有实际原N1低响应种子支撑，S0支撑不能验证围边正确。证据`docs/NAF_ITERATION_20261030.md`、私有`output/naf_iter/{verified,trace}.json`；仅五例且GT已见/不完整，无官方新分，169第三轮不能推广为单轮全150根因，不晋升AGENTS。
- followup_marker_dilation_20261030: 同旧20a59单轮、S0原N1与末端200的五例实际干预中，只将种子细化后扩宽1→0，真实WS障碍哈希与控制相同，过滤前仅7区、过滤后每图各1区；不能把扩宽当无关修饰直接删除。D0同时得到“无小F、无GT核多分、薄F覆盖保持”，却显然全图粘连，说明这些代理须与实际分隔和粘连风险联合解释。扩宽2仅有较小变化，.45／.55输入阈值会交换新增碎分和丢失分隔，未修开116登记缺口。证据`docs/READOUT_SCAN_20261030.md`、私有`output/readout_scan/received/`；只覆盖五例、GT已见且不完整，不证明所有模型都必须使用同一扩宽值，不晋升全局规则。
- followup_finite_weak_growth_20261030: 固定20a59/S0/200，仅marker生长8→4步在当前9图重建mask完全相同，历史101只少4px，最终13例pre/final ID与类别均逐SHA完全相同；应先验干预是否真的改变信号，不能把无有效扰动解释成“训练或纠错无收益”。低阈值.45→.50去掉当前图额外重建弱像素的25.25%，只小幅移边，未稳定去环；新8图F<500少1来自跨500档而非删除实例，需按空间身份跟踪，不能把分档计数下降当去碎。证据`docs/WEAK_READOUT_20261030.md`与私有`output/readout_weak/received/`；已见不完整GT仅诊断，九图不代表全150，不将该饱和范围推广为所有恢复器的规律。后续保留原生ID连通性检查，116仅作历史案例，不设为普遍晋级门槛。

## 20261027-naf-phase-mechanism

- status: observation
- last_verified: 2026-10-29
- scope: 作者式线宽T-only NAF step1000，固定权重、严格二值合成输入、CPU单窗诊断；不涉及训练或测试GT。
- finding: 初态至训练后，8px竖线2×2局部交替幅度0.01865→0.36871；输入平移1px的逆对齐线带MAE0.407–0.546，而16px为0.00168。临时平均最后/倒数第二/全部四级PixelShuffle的四相位激活，交替幅度降为0.11609/0.09054/0.01032，定位多级相位自由度被训练放大；输入标准化、拼接、JPEG并非必要原因。现行容差重建目标给连续48点线与48个孤立侧移点完全相同loss4.1223e-9，覆盖不保证连续。
- evidence: `docs/NAF_CHECKER_20261027.md`及私有`output/naf_checker/{cpu_complete,loss_counterexample}.json`；同seed初态、原始float概率、单次前向、位移及激活干预，模型state与checkpoint文件SHA前后相同，CUDA未初始化。无损合成示意`synthetic_phase.png`，脚本哈希记录于专题文档。
- reuse_hypothesis: 遇到周期伪影时，先以原始float、单窗、初态对照、平移协变和局部激活干预定位，再选单变量训练消融；不得只因含PixelShuffle就断言唯一原因，也不能用逐截面覆盖率证明边界连通。
- verification_gap: 激活平均也改变内容/幅值并丢失子像素信息，不是正式修复；原PixelShuffle分支GAN贡献未独立消融，后继bilinear/ridge配方的GAN对照见下；GPU精度或官方增益未由本探针验证，合成输入不是实际准确率验证，不晋升AGENTS或替换部署。
- followup_real_RGB_phase_20261029: 冻结D5a恢复RGB＋T8擦断组的两块真实干净256窗口，原输出SHA与固定monitor相同；RGB/T同步移位后只比较中心valid，GT±8带内移1px逆对齐概率MAE0.163–0.230，移16px仅0.0014–0.0036。该模型的2×2棋盘项下降，规则竖向多轨仍在，空间均值RGB也未消除网格；因此单个棋盘统计改善不等于相位稳定或线条更规整。证据`docs/NAF_RGB_ERASURE_20261028.md`终态与私有`output/naf_rgb_erasure/analysis/natural_probe/{complete,phase}.json`。只定位现象，不唯一归因PixelShuffle，也不把均值RGB下缺口峰值增加当真实补边；不晋升AGENTS。
- followup_width_occupancy_and_multistage_phase_20261029: 同GT法向剖面应拆开高响应占地、外包跨度和分离段数。无RGB平移约束终态的固定train173窗口中，N1占地/跨度均5.96离散采样格；NAF占地2.51、跨度6.41，297/359高度相邻位置为多轨。量化对角法向步长sqrt(2)，不能把采样格称欧氏像素宽。单窗绑平均最后up相位几乎无效，绑全部四级多轨降至86/359，但自然缺口强点3→0；参数SHA不变、移除hook后原前向exact。支持多级相位参与，不支持直接平均部署或唯一归因PixelShuffle；完整B1又把多轨组织成微环，四缓存小F30→81。复用时同步核原浮点输出、位置/方向、GT可信域、共同平移内域及最终读出，不能只比较线宽或棋盘项。证据`docs/NAF_SHIFT_20261029.md`终态、私有analysis/{phase_probe,phase_profile_supplement,stages_shift,complete}.json；已见不完整GT和固定无标签测试不代替官方成绩，不晋升AGENTS。

- followup_bilinear_control_20261029: 同原T8擦断1000更新、数据/GT/LR逐步配对，仅四级up换投影＋bilinear，两个真实原生窗的1px逆对齐MAE0.312→0.052、0.112→0.013，棋盘项同步下降；但359同截面多轨308→302，平滑平行轨经原B1变成狭长碎实例，四固定测试小F30→958。自然缺口2→0/161，原强点支持基本保持，四个指定人工分隔仍保住但另一批176点强响应176→165。冻结实际重建项的合成核对中，单线/中间留黑双轨/实心带同loss4.1223e-9；仅证明重建项不约束单连通，不代表GAN总目标等价。上采样能改善相位却不保证唯一主脊及最终拓扑，应分开验收相位、多轨、弱线和完整实例。证据`docs/NAF_RESIZE_20261029.md`及私有`output/naf_resize/analysis/{complete,training_audit,phase_probe,stages_resize,loss_counterexample}.json`；GT不完整/已见、测试无GT且阶段探针仅101，无官方新分，不晋升AGENTS或部署。

- followup_ridge_supervision_preflight_20261029: 原pointwise容差不删任何正点，额外用处理后GT实例走廊筛单A/B界面，再在±4预测剖面构造停止梯度的区间上包络，只监督中央±2。深暗谷采用稳定KL值和融合BCE解析p-q反向，避免概率平方饱和或log-sigmoid微小舍入反向推峰；单线偏移/弱单峰/实心带零新项，双/三轨只抬谷不压峰。轴向3/斜向5条完整整数射线排除A-B-A、A-B-C和交叉，额外17×17全known，仅限制新项；32源前128 draw资格83.74%，两已见实际多轨缓存有有限独立填谷梯度，固定128短测保住174/174人工缺口及约99.9%原支持。证据`docs/NAF_RIDGE_20261029.md`、`tests/test_boundary_ridge_loss.py`及私有cache_check/audit/smoke回执；这是设计及可学性证据，正式泛化/最终B1未知，不晋级或晋升AGENTS。未标出的界面仍无保护保证，外围错轨仍依赖原negative。

- followup_ridge_supervision_final_20261029: 同1000更新配对终态中，GT资格中央±2多轨64→2，但全±8三轨及以上136→151，.45弱阈值同样有外轨残留；只验中央目标会漏掉外围结构。原生四人工分隔保留、强点165→176/176，四测试小F958→800但仍明显碎分，已见GT内核出现新显著共用实例及内部响应增加。外轨大多已属原negative，不能简单当无监督灰区或扩大填亮半径。两已见固定窗、D eval且无更新的完整G梯度显示GAN/ridge余弦−.9305，直接谷上多数GAN压暗、新项填亮，全目标仍多数填谷；支持单独GAN消融，不足以断言GAN唯一致因。证据`docs/NAF_RIDGE_20261029.md`和私有analysis/{complete,phase_scope_supplement,weak_threshold_probe,gradient_competition,core_summary}.json；高度相邻剖面/已见不完整GT/无标签测试均非官方精度，不晋级或晋升AGENTS。

- followup_readout_threshold_20261029: 固定ridge概率只升高阈值.65→.75或低阈值.45→.55，两个测试的附着细轨/小环确有减少，但不是只去外围弱轨：小F424→263/280，大F377→332/333，原N1弱marker相对原已覆盖点额外丢125/42；116登记断口仍未分开。较强并行响应在.75下仍残留。提高读出阈值应同时查弱边支持、粘连与类别投票，不能只以碎块减少判好；低阈值单独提高仅保留为较温和候选。证据`docs/NAF_NOGAN_20261029.md`和私有`output/naf_threshold/complete.json`，N1参考非GT、已见GT不独立，无官方新分，不替换部署或晋升AGENTS。

- followup_zero_GAN_and_input_edges_20261029: 同bilinear＋ridge、完整随机初态/GT/16000draw/LR及B1不变，仅GAN.01→0，1000步后同379剖面全±8多轨253→2、三轨151→0；高响应占地反增而外包跨度/暗缝下降，是分离细轨变实带，不是整体压暗。四测试最终小F800→34（冻结N1为26），116登记点对实际分开；原强支持8375→8277/8428，仍有局部偏移/弱边代价，无官方增益证据。末端分解定位外峰已在learned ending中；20次固定相位横/竖直线宽度反事实显示旧外轨随二值带两侧跳变移动（竖线宽6–12的外峰始终在边缘外1.5px），去GAN在.65下全单段。支持该配方GAN与输入边缘偏置参与，不能归罪读出、只靠填谷中央统计验收，或推断所有GAN/所有相位都无效。微孔增多与最终碎块减少同时发生，概率孔不可当最终种子/FP。证据`docs/NAF_NOGAN_20261029.md`、私有analysis/{training_audit,gan_onset}.json、phase/{complete,summary,width_probe}.json及固定读出；仅2已见窗/4无标签测试/有限合成线，不晋升AGENTS，不证明填谷可删或替换部署。

- followup_plain_factorial_20261029: 补齐同初态1000预算的GAN×ridge四格，CPU同379剖面全±8多轨(G1R0/G1R1/G0R0/G0R1)=306/253/120/2，四测试小F958/800/189/34。GAN不是多轨必要条件，去GAN和填谷在此配方互补；去填谷原coverage/interior更低、部分弱支持略好，却恢复外围轨/碎块，并让已见GT69/115主要内核重新显著共用实例。116与四人工两组仍分开，不能把填谷收益泛化成所有断口都更强。同379中央±2均无多轨，外围变化不能全解释成这些中央位置被直接填平；更完整合格目标的诊断loss/梯度和共享参数作用应区分。raw微孔少也可能对应更多可成种子的较大暗域，不能以微孔总数替代最终拓扑。证据`docs/NAF_PLAIN_20261029.md`及私有analysis/{complete,training_audit,core_review}.json、phase/{complete,summary,width_probe}.json；当前保留G0R1，不证明0.1最优、任意预算下R0都无效或官方增益，不晋升AGENTS。
- followup_official_and_cohort_20261029: 新NoGAN无S0官方.7489/.8609，虽相对病态GAN＋ridge四图大幅去碎，却相对真正最高旧20a59＋S0净多2487FP、面积−.0449。同none仍退化，不能只归因缺S0；新旧还含上采样/擦断/ridge差异，不能单归GAN。全150 F覆盖近似不变而数量+7.67%、预测汇总均面积−7.28%；微块档只解释部分计数净增，正常尺寸分区/投票也会改变面积分布。更关键的是固定101/116/166/169只有169属于当前169–318评分集。机制对照应同时锚定真实部署，并核诊断样本与当前评分集的交集；出包后的全量类别/面积分布可作风险检查，不能代替GT或按基线数量校准答案。证据`docs/NAF_NOGAN_20261029.md`官方节及私有`score_20261029/`；新S0待分，无新训/推理，不退休原多轨机制结论或晋升AGENTS。
- followup_area_histogram_20261029: 进一步拆分逐F面积后，268仅新增87/150px两个小实例，就令全均值/中位数低11.23%/27.81%，而>=500档数量和大面积加权CDF近乎不变；312的残余变化也主要落在500–10000档。不能把“>=500档均值下降”自动命名为大晶粒主体普遍缩小。输出诊断宜并列绝对数量直方图、按实例计数CDF与按F像素面积加权CDF，并按图像尺寸分层；均值/中位/分档不是同对象面积追踪，也不是GT误差。原说明中的“正常尺寸”仅指超过500px分档，不具有物理或标注正确性含义。证据同专题面积直方图节及私有`area_hist/{areas,summary,tail_decomposition}.json`；没有应用面积过滤或新增模型实验，不晋升AGENTS。

## 20261028-naf-stage-to-instance

- status: candidate; last_verified: 2026-10-30.
- scope: 冻结N1后接作者式输入T-only NAF s1000，再使用既有bridge/reflect/core50的完整读出。
- finding: 高阈值热图的海量碎组件未必进入最终实例。四缓存NAF的313939个高阈值8组件在bridge后为126，178个微核心被已有核心50门槛排除，最终1011种子与缓存最终实例数相同。与此同时，全150图总实例较N1净少409，F却净多290、P净少699；全类别去碎不等于面积目标改善。
- evidence: `docs/SEMIFINAL_NOISE_SCORES_20261026.md`的2026-10-28回分节、`docs/NAF_COHERENT_20261028.md`；私有`output/naf_followup/{scores,readout_report}.json`。用户回分mIoU提高0.0032但面积下降0.0076，综合尚未晋级；17370个输出与TP+FP一致，未核平台提交ID。
- reuse: 设计新过滤前先沿实际threshold→bridge→thin→seed→final追踪变化，并按目标类别拆分计数；不能用像素孔洞或全类别实例数替代FP/官方面积。
- verification_gap: 四缓存无GT，未证明被过滤者全部为物理噪声，也未识别全部净TP变化对象；保ID种子收缩只有计数未实际读出，不把其像素释放当准确率增益。本候选不晋升AGENTS，也不支持按基线F数量/均面积拟合阈值。
- followup_coherent3: 后继同初态/同1000计划的三点共同偏移目标减弱局部点阵，却在四缓存将小F29→236；原始概率已出现平行轨和横杆。合成证明单线、带内宽带、双峰/梯形环仍可同样低loss，短段共同offset不是整条唯一主脊约束。实心宽带的细化并未凭空生环，不能只因宽就归罪thin。101/116固定新热图、真实WS高程和raw投票，仅marker bridge1→0减少部分附着小环，但伴随其它拆合且116目标未补通；这支持读出适配有空间，不证明准确率。证据`docs/NAF_COHERENT_20261028.md`终态节及私有`output/naf_coherent/evaluation/`；不按环数/基线均面积选参数，不直接晋级、扩训或扩大三点为五点。
- followup_same_ID_cooccupancy: `compare_marker_atoms`的a/b_shared_pixels分别对全部共同新ID求和，
  不代表同一对象内双方主体混合。实际A在两个新ID内8666/6px、B为2/6885px，
  并集量8672/6887看似很大，主体仍分别占据不同ID，仅有6/2px交叉。
  后续量化主体误并需逐新ID并列双方像素/比例，同时保留微接触与旧正core分裂统计；
  不能把严格零接触门槛产生的pair数或并集量写成整粒合并数。
  证据`docs/NAF_READOUT_20261028.md`及私有`output/naf_readout/seen_risk/refined/complete.json`。
  此为诊断量解释，未修改公共函数历史口径，也不将已见不完整GT当独立准确率，不晋升AGENTS。
- followup_local_marker_change: 固定20a59和真实WS高程取消前bridge可清理登记小环，却会重塑其它seed占地，
  两已见源之一出现48796px旧原标core的4914px切出；同高程不等于仅局部去块。
  改为保持原B1种子，仅撤销零原N1低响应内部支撑的seed，四测试14登记小块全部接管，
  两源旧正core无碎分/归零，仍有最多15px次要支微接触。这只支持局部候选，不证明所有撤销者错误或官方增益。
  证据`docs/NAF_READOUT_20261028.md`、私有`output/naf_readout/{readout,seen_risk,support}/`；
  helper以哈希绑定的私有副本加载，不补写缺失模块到旧R8快照，不将本例晋升全局筛选规则。
- followup_marker_threshold_20261030: 固定同一20a59／真实WS场，仅marker强起点.65→.60仍可造成非单调的空间重组；当前9图种子与pre200都净增1，final却净增21，实际由末端200少合并20放大，F／P需再拆开。阈值后的细化、反射、连通、面积与S0使“降低阈值只会多切”不成立；种子、pre200与最终数量相等也不是数组别名。应追踪空间身份及各阶段候选／成功合并数，不能把净增实例当新增TP，或将未能单父归属的种子叫新增晶粒。证据`docs/MARKER_HIGH_20261030.md`及私有26份终态／seed详情；只支持机制解释，已见GT方向不一致，无P060官方增益，不晋升AGENTS。
- followup_S0_official_20261029: 用户补报完整20a59＋S0包.7467/.9058、平台82.62，ZIP SHA939cccb59175…及16863预测=TP+FP一致；相对同G无S0，mIoU−.0004/面积+.0102/综合+.490、TP−5/FP−502。支持该固定模型读出适配的面积/综合收益，而非额外净检出，也不证明所有撤销seed为假。新NoGAN＋ridge/双线性/擦断包读出仍none，不能继承本分数或把跨读出变化全算给训练；应按完整旧S0方案留基线，新G按同读出另验。证据`docs/NAF_SUPPORT_PACKAGE_20261028.md`、`docs/SEMIFINAL_NOISE_SCORES_20261026.md`和私有score_20261029.json；未核平台提交ID，无新实验，不晋升全局筛种规则。
- followup_RGB_peak_vs_partition_20261029: 同GT/擦断任务加入恢复RGB后，四个人工案例176个缺口点仍全>.65，原B1无显著两GT原标core共享却从4/4降为1/4（两例主体合并、一例B的72.9%与A主体共ID）；其中两例clean输入已混合，不能把退化全部算给人工擦断。101/166骨架孔数下降，但后继belt<500px孔332→2153/337→2377，四测试最终小F30→163。需要同时保留输入clean/damaged、局部峰值、同一新ID内双方core像素及最终形态；峰值或骨架阶段单项好转不能当完整分隔增益。证据同`docs/NAF_RGB_ERASURE_20261028.md`与私有analysis/{complete,stages_rgb,belt_hole_bins}.json；已见不完整GT和无标签固定测试均不替代官方评分，不将孔数当误检数，不据此否定所有RGB方案。

## 20261028-input-kernel-anchor-and-connectivity

- status: observation; last_verified: 2026-10-28.
- scope: NAF整图细化→OpenCV离散椭圆膨胀的输入表示；本次CPU合成环境OpenCV4.12.0。
- finding: k8/6/4的实际离散核不是连续等比缩放；缩核既可分开相邻线，也可撤销输入级补口。偶核anchor相对外接框中心的+.5元数据不等于整体预测平移(.5,.5)：单点k8实际质心约(.226,.075)，k6约(.310,.103)。因此保持偶数/默认anchor只能减少混杂，不能声称几何偏置完全相同。
- evidence: `docs/NAF_WIDTH_TRIAL_20261027.md`末节；私有`output/naf_coherent/evaluation/input_kernel_cpu/{analysis.md,complete.json}`。真实历史raw T→骨架+ellipse8支持一定去碎但闭环/栅格转角混合，不能把新核越小越好当结论。
- reuse: 核尺寸、NAF输出形态、GT容差分别消融，训练/推理共同改变输入核；若研究更准定位，不把收紧GT容差与输入缩核混为一项。
- limits: 初始证据仅CPU形态与历史4缓存；后续k6证据见下条，k4仍未训练。实际精度需按原生像素/源码版本复核，不由质心或闭环计数代替。不晋升AGENTS。
- followup_kernel6: 同初态/源码/1000步数据计划、训练推理共同8→6后，四缓存局部交替和部分附着环减少，
  但共同raw缺口终态仍0/161、116未分开；F585→586、小F29→35、小P197→263。
  原N1弱响应共同骨架17559点的r2强响应保留16887→16499，说明局部去环可伴随弱响应代价和其它位置碎分增加。
  不能用变化后的normalized缺口分母、局部环或总数代替完整分区；也不能以s500短暂16/161择优代替统一终态。
  证据`docs/NAF_KERNEL6_20261028.md`及私有`output/naf_kernel6/evaluation/`；四测试图无GT、七训练pair已见，
  没有k6官方分或架构普遍失效结论，不直接晋级/扩训或推广为缩核定律。

## 20261028-pythonw-child-console-flash

- status: observation; last_verified: 2026-10-28.
- scope: Windows本机以pythonw运行的服务器队列等待/下载器；不涉及模型或服务器任务变更。
- finding: 隐藏Python主程序不保证SSH/SCP子进程也隐藏。19:27:21直接捕获本机等待器启动SSH状态查询及其conhost；同秒启动Windows Terminal，随后确认窗口可见。脚本的两处subprocess.run仅capture_output，未指定禁止创建窗口的creationflags或隐藏startupinfo，成功等待间隔为60秒。
- evidence: 私有`output/diagnostics/console_flash_20261028_192721.json`保留进程父子关系、可见窗口时间及脚本SHA256；源为`output/naf_support/deploy/pull_when_complete.py`。采用无需管理员权限的短时进程轮询和窗口枚举；Terminal直接父进程为系统服务，因此其与SSH的联系按时间关联记录，不伪装成直接父子链。
- reuse_hypothesis: Windows后台等待器应在SSH/SCP等每个外部命令处设置适用的无窗口参数，并实际观察子进程启动时是否出现可见窗口；只验证主进程pythonw存活或日志quiet字段不足以证明静默。
- verification_gap: 本次只定位，未修改/重启下载器、未终止服务器任务，也未实测修复后的无窗口行为；不能据此排除其它时刻存在其它闪窗来源。不晋升AGENTS。

## 20261028-naf-gap-cohort-identity

- status: observation; last_verified: 2026-10-28.
- scope: BoundaryRGBNAF跨旧真实输入monitor、新erasure cohort和原生整图机制评估。
- finding: “缺口”同名不等于同分母。旧`real_pair_metrics`按raw T到GT欧氏距离>4选161点，新erasure cohort按r2方形邻域缺失选185点；两者输出都取r2峰值，不能因此混比。另固定七pair人工缺口与四个原生人工断口恰好都为176点，但坐标集合不同，总数相同也不能当配对证据。
- evidence: `docs/NAF_ERASURE_20261028.md`终态口径节；私有`output/naf_erasure/evaluation/{monitor_audit,clean161,complete}.json`。首次严格断言在GPU前向前拦下混比，保存现场后，仅重建旧161点所在两块clean输入；source/box/D4/GT/valid与cohort SHA绑定，得到旧0/161、新2/161。训练和冻结代码未改。
- reuse_hypothesis: 对缺口/保留/假线指标同时记录选择器距离定义与半径、评估响应邻域、输入参考版本、有效域与坐标集合身份；同总数仍须核mask/坐标哈希。新添人工破坏与天然缺口、局部峰值与最终分隔分别报告。
- limits: 这些是已见不完整GT的机制诊断，不是独立准确率或测试答案；本观察不改变原历史指标、训练配方或AGENTS，不将161/185推广为其它配置固定常数。

## 20261029-frozen-monitor-entry-coverage

- status: observation; last_verified: 2026-10-29.
- scope: 从旧控制运行导出冻结代码、再叠加少量训练修改的RGB条件NAF实验。
- finding: 本地测试和冻结快照的GPU固定批短测都通过，仍可在正式monitor0失败；两者不能证明最终训练入口完成了首次更新。此次导出只叠加训练兼容文件，遗漏工作树已有的RGB monitor支持；旧monitor将B1 clean边界传入B4模型，并且尚不支持恢复float32 RGB缓存。
- evidence: `docs/NAF_RGB_RETRY_20261029.md`的启动故障与恢复节；私有`naf_color/train/train.log`在`save_monitor`报`Expected nonempty floating B4HW input`，正式steps/initial/latest均未生成，旧GPU128回执通过。工作树与冻结monitor源码SHA不同。
- reuse_hypothesis: 按最终冻结快照核验实现与测试版本，并执行正式monitor0和首步保存路径；收到子进程PID与模型可学性短测，应分别标为“已启动”和“训练推进已确认”。新增输入模式还需检查clean/erased与整图monitor是否使用同一输入合同。比较展示元数据时，先声明输入模式迁移，只归一已明确核对的字段，避免把预期的模式名变化误判为训练差异。
- limits: 这是部署与观测入口覆盖缺口，不是模型学习或RGB方案无效的证据；旧数据/初态/短测可在明确绑定源码差异后复用。恢复验证结果另记实验文档，不直接晋升AGENTS。

## 20261030-random-initialization-portability

- status: observation; last_verified: 2026-10-30.
- scope: 新关系分类CNN，Windows／Linux同Torch2.8.0+cu128、seed、float32及完整架构的初态与训练权重加载。
- finding: 原始uniform流SHA相同不保证最终初始化tensor逐位相同；22个tensor中9个字节不同，抽样首值最大差约7.45e-9，同服重建则exact。若推理loader用本平台重新随机初始化的SHA验证保存权重，会误拒架构与实际state均合法的跨平台checkpoint。
- evidence: `docs/BOUNDARY_RELATION_20261030.md`；两组真实GPU重载／下一AdamW更新exact，新增模拟跨平台CPU回归验证合法权重及预测exact、坏架构／state SHA／冲突来源仍拒绝；冻结v2服务器13项测试通过。共同初态文件保持未改，两组正式读取同一文件。
- reuse_hypothesis: 成对训练复用完整初态文件；推理校验实际保存state、架构和来源元数据自洽，将当前构造器随机初态作为审计信息而非跨平台加载必要条件。训练初态／续训防混用合同单独保留，不能为移植盲目strict=False。
- limits: 只证明此CNN与这对平台存在末位差及旧检查误拒，不证明所有初始化／硬件均如此，也未将具体CPU指令归为唯一原因；不改AGENTS，不等同扩散GPU噪声问题或官方增益。

## 20261030-reliable-pair-selection-bias

- status: observation; last_verified: 2026-10-30.
- scope: 不完整GT上原生相邻预测区域的keep／merge可靠监督筛选。
- finding: 可靠性过滤也会制造目标捷径。两源pilot中仅凭“两侧都大”规则的平衡召回达92.83%；不改筛选阈值扩到32源后降至72.35%，说明早期少源总体拟合可能主要反映筛选与尺寸相关，而非RGB边界识别能力。原标像素覆盖与可标邻接对覆盖也是不同分母。
- evidence: `docs/BOUNDARY_RELATION_20261030.md`；全32为21446候选→2746可用，2225 keep／521 merge，两类均32源；共同大面积FF及PP有两类支持，但小F正例极少。人工GT核只产生label／审计元数据，不进入输入；独立NPZ头及monitor valid检查通过。
- reuse_hypothesis: 同源序、同初态的无RGB控制及面积／相别／响应强度共同支持组应一并保留；不能凭总体训练准确率判断上下文增量，也不为凑数降低可信监督门槛。固定monitor的采样覆盖应明确，不能用少量靠边样本替代全量分组分析。
- limits: 事后面积规则仅审计标签偏差，不是新后处理或竞赛代理分数；本轮训练尚无泛化或官方增益证据，不晋升AGENTS。
- followup_fixed_scores_20261030: 固定e5已见参照上RGB总体净少28错，却在大F/F的未校准.5工作点把keep误合43增到133；全F/F AUC近同、大F/F排序略好，等7个keep参照受损的保守预算下merge命中70→88，而等35错误预算为134→132。跨相可分性、组内排序与分数校准应分开；同阈值不等于同风险，总体拟合也不能认证物理假边。两已见细线窗标签相反仅提示监督歧义，不据图重标。证据`docs/BOUNDARY_RELATION_ANALYSIS_20261030.md`及私有fixed_eval；无实际合并或官方增益，不晋升AGENTS。
- followup_long1000_20261030: 同初态／数据的1000更新配对显示短训单阈值退化不应被当作RGB内在有害：固定.1大F/F maps误合6／命中78，RGB6／110，AUC .96536→.98088；四个预定阶段均有组内排序增量，但固定阈值的错误不随训练单调变化。净增32中前三源占26，小侧200–499仍误合4／11 KEEP；同错误数也会交换错误对象并涉及更大区域。源级自举支持AUC增量，极低错误预算检出差的区间仍跨0；只能支持小样本最终读出诊断，不能认证安全合并或泛化。旧五轮与新长训学习率日程不同，不归因于单独多训785步。证据`docs/BOUNDARY_RELATION_LONG_ANALYSIS_20261030.md`及私有scores／ranking／risk_notes；无实际合并或官方分数，不晋升AGENTS。
- followup_actual_readout_20261030: 相同.1固定13源实际合并后，当前9图F733→maps565／RGB564，F总像素变化<.05%，聚合均面积约+30%主要来自少实例，不能当赛题面积项改善。已接受GT仅支持两训练图中3／4个MERGE，余者IGNORE／无标签，故训练排序增量不覆盖实际应用的大量未知关系。全缝未见仅各1次，观测范围不足不是这轮大幅合并的主要解释；不应把合理架构风险直接当实测归因。真实patch tensor／零操作／ID类别／safe-fill工程合同通过仍不构成净收益认证；保留显式推理候选而默认不晋级。证据`docs/BOUNDARY_RELATION_READOUT_20261030.md`及私有actual_summary／context_risk／全缝图库；无官方增益证据，不晋升AGENTS。
