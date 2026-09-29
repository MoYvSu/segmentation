# 配置目录

`train/affinity_native.yaml`继承`affinity_connectivity.yaml`，只新增原生裁块选项和输出根目录。
入口`tools/run_affinity_native.py`先`--smoke`，通过后顺序训练1024／512，每组20轮／1280更新，
分别从旧control的原初始化出发；旧全图control直接复用。两组已完成，配对／冻结／重载通过。
仅affinity decoder训练，原32人工／64 SAM2、退化、D5a、语义与后处理不变；
过程图和各自全图／本尺度局部及control局部的6套100图对照全部完成。
分析入口`tools/analyze_affinity_native.py`，私有图册`output/native_analysis/index.html`。
优先建议1024权重配1024原生推理作黑盒候选；不替换旧全图入口权重、不改默认配置。
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
