# 当前管线与产物约定

2026-09-29：[原生1024／512裁块适配](AFFINITY_NATIVE_20260928.md)训练、全量推理及完成分析均结束。
`config/train/affinity_native.yaml`、入口`tools/run_affinity_native.py`；先1024再512，各20轮／1280更新，
两组独立复用原control当时初始化，只训练affinity，不重跑control或删Stage1/joint-v3。
35项本地检查、双组8更新GPU短测、6份完整输出复现及全部5120裁块监督检查通过。
快盘`outputs/affinity_native/{p1024,p512}/`保留逐步回执、e0/e1/每5轮缩略图和末态权重。
6套新增100图输出与每组108张过程PNG完整；700份最终输出及2560配对回执检查通过。
同尺度512铁素体小片6307→354，1024原生比旧全图铁素体数+2.65%、平均面积变化中位−2.48%。
4张已见GT图说明局部视野有收益，但同尺度适配的铁素体匹配略降；优先建议1024原生黑盒，
不晋级／扩轮／自动提交。分析入口`tools/analyze_affinity_native.py`，私有`output/native_analysis/index.html`。

2026-09-28：[原生patch冻结检查](AFFINITY_PATCH_20260928.md)完成；入口`tools/probe_affinity_patch.py`，
配置`config/experiments/affinity_patch.yaml`。4训练＋4测试、6种affinity输入，语义全图固定、连续边界拼接后仅一次全局分区。
主检查及追加32对`tools/probe_patch_structure.py`均冻结核验通过，4张全图control完整输出复现。
1024原生窗口在已见训练可信域有改善，512放大有明显小片风险；D5a的RGB／梯度改善并不排除
局部模糊＋降采样下的内部假边响应。全部汇总及图册`output/affinity_patch/index.html`，
本次冻结检查未训练、未推理全100图、未改默认部署；后续双尺度训练见上条，控制直接复用。

2026-09-28：[参数梯度预算短测](AFFINITY_BUDGET_20260928.md)完成，入口`tools/probe_affinity_budget.py`。
配置`config/experiments/affinity_budget.yaml`，从最高分control e20复制两个affinity头，固定4源8视图各64步，
只比较旧25%输出梯度上限与15%参数梯度上限；原20轮control未重跑，固定项／配对／严格重载通过。
新预算产生更大真实更新，但纠错未显著增加、新粘连更多；不放行20轮、不替换部署、不生成黑盒。
末轮小头`outputs/affinity_budget/{logit,parameter}/head_064.pt`需与原完整control组合，不是通用模型包。
图册`output/affinity_budget/index.html`；[原生局部输入计划](AFFINITY_PATCH_PLAN_20260928.md)已执行，见上方完成记录。

2026-09-28：[连续界面监督](AFFINITY_INTERFACE_20260928.md)已完成20轮／100图，配置`config/train/affinity_interface.yaml`。
入口`tools/run_affinity_interface.py`只训练一组20轮／1280更新候选，严格复用旧control，
保持原始初始化及Stage1/joint-v3、D5a、LoRA、语义和原后处理，唯一变化为附加监督的完整短程界面选位。
全32源64视图冻结检查及8更新同配置GPU短测通过；新增误切26／粘连38仍是风险，不当作晋级依据。
全量检查、随机分档小侧区域图册见专题；不叠加分隔恢复或过滤。
快盘输出`outputs/affinity_interface/`，状态`pipeline_status.json`、日志`candidate.log`，
固定过程图`candidate/monitor/epoch_*/`齐全，严格重载及100图完成。98.64%旧实例匹配IoU≥0.95，
实例9201→9216、铁素体5969→5965，平均面积变化中位−0.00165%；附加作用小，主loss几乎重合。
逐轮首步参数梯度比中位4.185%，320次启用全受输出梯度上限约束；无新黑盒、不晋级或原样续训。

2026-09-28：[纯分隔恢复消融](AFFINITY_PRESERVE_20260928.md)完成，入口`tools/run_affinity_preserve.py`。
配置`config/experiments/affinity_preserve.yaml`固定control e20，只恢复原粗边界已有分隔；
新增`area_filter_enabled: false`独立关闭附加面积离群过滤，原有50像素最小面积不变。
64标注视图的542个已定位细化重连点对全部恢复，按现有GT新增58个显著误切、7个粘连点对；
100测试图对照逐像素／类别复现，候选9600实例，平均铁素体面积变化中位−4.788%。
冻结、实际障碍图及输出契约核验通过；本地42项、服务器5项断言及GPU预检通过。
私有`outputs/affinity_preserve/`及本地`output/affinity_preserve/`保留独立候选包、对比图和新增错误例子；
用户随后回分0.8499／0.8559（85.290），比同权重control低0.945分；面积项回退抵消mIoU小幅上升。
没有新训练、代用户上传或默认晋级，当前官方最佳仍为control e20。
完整面积分层统计、Jev分布及下一步见[回分分析](AFFINITY_REVIEW_20260928.md)。

2026-09-28：[完整划分链定位](AFFINITY_TRACE_20260928.md)完成，入口`tools/probe_affinity_chain.py`。
固定control e20及D5a、语义、LoRA和全部后处理，32人工训练源各原图／在线模糊，共64输入、四种输出干预。
细化前分开后重连238/358、304/554个粘连点对，严格原标路径子集69例；selected-only只修复3个粘连、0个显著过分割。
实际marker捕获、冻结状态、64输入CPU/GPU数值部署等价及两个既有测试输出复现通过。
不把反事实计数当精度／上界；先检查保留已有分隔及有效选边，15%参数预算暂缓，无新训练或默认部署改动。
私有完整产物`outputs/affinity_trace/`，本地`output/affinity_trace/`，含64整图缩略图和三处实际通路细图。

2026-09-28：[affinity连通性对照](AFFINITY_CONNECTIVITY_20260928.md)两组各20轮／1280更新、100图推理完成。
入口`tools/run_affinity_connectivity.py`，配置`config/train/affinity_connectivity.yaml`，私有`outputs/affinity_connectivity/`。
只适配V affinity，固定D5a、joint-v3编码器/LoRA、S-align与原后处理；恢复图语义仍参与几何，raw只末端复投票。
两组final均保留；control用户回分0.8486／0.8761（86.235），同链提高0.615分，成为当前综合最佳研究对照。
candidate仍无黑盒，不随control晋级；仅改附加梯度预算时复用此次control，从相同原始初始化训练，不重跑对照。
共同微调收窄响应带约9%，新附加项效果很小；不自动续训或改默认推理。下一步见[AFFINITY_NEXT_20260928.md](AFFINITY_NEXT_20260928.md)。
本地完整分析与图册`output/affinity_analysis/report/`，删除清单`outputs/cleanup_affinity/`；快盘已释放约19 GiB。

2026-09-27：[尺度语义实验](SEMANTIC_SCALE_20260927.md)完成20轮／1280更新，控制复用私有LoRA e20。
配置`config/train/semantic_scale.yaml`，入口`tools/run_semantic_scale.py`；D5a后50%全图／50%局部2倍，
原图先验先生成再同步裁切，不修改损失池化、几何或全图部署。GPU禁用6步精确回放、候选6步及8图输出通过；
冻结、逐步输入配对、严格重载及100图部署通过；105／8668实例改判，45珠→铁、60铁→珠。
局部任务有所学习，但全图末5轮同输入loss比控制高4.6%；逐图铁素体平均面积变化中位−0.00344%。
`outputs/semantic_scale/`保存过程图及最终结果，本地`output/semantic_scale_analysis/`含完整分析；
尚无本候选黑盒，不自动续训或晋级。直接控制用户回分0.8445／0.8557（85.010）。

2026-09-27：[保守种子恢复与统一面积过滤](MARKER_RESTORE_20260927.md)接入正常推理入口。
`utils/marker_restoration.py`恢复已有分隔；`marker_partition_restore`默认关闭，仅候选配置开启。
面积下限由每图首次watershed所有实例计算，冻结后统一清理旧／新小种子并重新生长，不读取GT。
62项CPU检查和两图GPU完整输出检查通过；私有`outputs/marker_restore/`，无新训练或默认晋级。
全100完成：已知50例分隔全部恢复，实例9062，小块363（原8668／410）；统一过滤8图29个seed。
每图阈值0–228、中位27.5，规则偏温和；完整轮廓重分配及类别变化见报告，不等同官方收益。
用户随后回报`marker_restore.zip`为`0.8474/0.8354`（84.140），不晋级；默认保持关闭。
实际预测铁素体实例6101→6390、总像素面积−0.181%，91图平均面积下降，中位−4.396%。
独立语义LoRA e20＋原后处理随后回分0.8445／0.8557（85.010），配对后处理净差
mIoU＋0.0029、面积项−0.0203、综合−0.870；恢复与过滤两步各自贡献仍未拆分，不晋级。

2026-09-27：[固定轮廓／种子诊断](SEMANTIC_SPLIT_20260927.md)全100图完成。
复用simple、gray4、独立LoRA权重，固定LoRA轮廓；以`outputs/semantic_split_checked/`为有效统计。
全466例内部混合对象追踪发现50例骨架化前分开、之后相通，全部贴边；服务器实际走skimage回退。
`tools/probe_marker_anchor.py`只在分水岭入口替换前置seal2的marker，逐图核验原marker／实际障碍图，
保留默认推理和权重。`outputs/marker_anchor/`是诊断候选，不能作为已验证黑盒方案。
全100候选已完成，50例恢复分隔但小块410→1136，总实例8668→10575；不晋级。
后续保守恢复已按上方实验落实；未启动affinity训练。

2026-09-27：[大窗口补监督预检查](SEMANTIC_COVERAGE_20260927.md)已完成全1000源、1280次固定视图重放。
仅14源新增48666格点，已完成LoRA与新增目标零类别分歧，仅2格未达0.9置信门槛。
41项CPU检查通过；按新增信号不足的预定条件不启动20轮，不重训控制。
入口`tools/probe_semantic_coverage.py --config config/train/semantic_coverage.yaml`，
私有`outputs/semantic_coverage_probe/`，`decision.json`记录未启动；当前配置仅用于预检查。

2026-09-27：[语义专用LoRA](SEMANTIC_LORA_20260927.md)已完成20轮／1280更新、100图部署及96张过程图。
保留SAM2底座、原affinity LoRA和D5a冻结，只开放复制的语义LoRA及原simple头。
重放gray4逐步选中视图，直接复用完整gray4作为控制；99项本地检查、GPU冻结回放与候选短测通过。
入口`tools/run_semantic_lora.py --config config/train/semantic_lora.yaml`，输出`outputs/semantic_lora/`。
完成后全1280步回放、affinity冻结及严格重载核验通过，末轮SHA256前缀`23b23d1f0cec`。
相对gray4的65处改判中62处原分数距阈值不超过0.1，test101仅新增一处小区域变化。
固定64训练源的全局／局部光度区域冲突1／1→0／0，后5轮52.81%步骤无附加梯度。
支持当前合成目标进一步满足，未证实真实测试泛化突破；不自动续训、提交黑盒或晋级。

2026-09-27：[gray4困难视图](SEMANTIC_GRAY4_20260927.md)完成20轮／1280更新、100图部署，过程图96张，控制复用与冻结核验通过。
只在候选无标签分支增加全局／平滑局部光度候选，按独立灰度区域的困难度选视图；每步附加梯度范数限制在GT范数内。
保持原simple e60初始化、GT流、冻结D5a／SAM2／LoRA／affinity及推理契约；复用`outputs/semantic_gray/control`。
入口`tools/run_semantic_gray.py --config config/train/semantic_gray4.yaml`，输出`outputs/semantic_gray4/`。
原8图monitor之外，另保存4个固定训练难例的原增强／困难光照对照；这些图继续训练，不作验证选择。
全量对控制102处改判，gray3为35；固定训练视图复测显示大块合成光照失败基本消除。
后5轮平均学习率1.446e-6，同时区域选样判据大多满足；仍有非零像素监督，不能称模型整体饱和。
末轮`prior/epoch_020.pt`的SHA256前缀`ebbf255ebea6`；无新黑盒、不晋级或继续训练。

2026-09-27：[gray3活动位置平均](SEMANTIC_GRAY3_20260927.md)已完成20轮prior、100图部署和48张过程图。
仅新增灰度平均增益上限20，权重0.15和共同配方不变；旧控制末轮、日志、部署复用，1280步配对通过。
相同训练输入上先验类别分歧较gray2减少31.17%；逐轮首步额外／GT梯度比均值3.876%。
对控制35／8668实例改判，34珠→铁、1铁→珠，test101两处边缘小区域变化；尚无准确率结论。
入口`tools/run_semantic_gray.py --config config/train/semantic_gray3.yaml`，输出`outputs/semantic_gray3/`。
控制引用在`control_reference.json`；候选`prior/epoch_020.pt`，SHA256前缀`574e42576b04`。
私有分析`output/semantic_gray3_analysis/`含10图、训练曲线及过程对照；未晋级或自动续训。

2026-09-27：[单向灰度约束](SEMANTIC_GRAY2_20260927.md)本地38项检查、GPU三组短测及8图部署通过。
配置`config/train/semantic_gray2.yaml`继承前一轮，仅改变灰度损失、最高系数0.15及输出目录。
原simple e60重新接续20轮A/B，32人工GT／1000训练源、D5a与匹配上游冻结约定不变。
入口`tools/run_semantic_gray.py --config config/train/semantic_gray2.yaml`；正式各20轮、1280更新、
100图部署及48张过程图完成。活动位置占接受量0.06146%；逐轮首步额外／GT梯度比均值0.368%，
对同一控制仅6／8668实例改判，test101零额外改判。不晋级或原样扩轮，后续候选见上方gray3。
两控制末态全部模型张量及100图输出相同，复用`outputs/semantic_gray/control/epoch_020.pt`；
新控制重复权重已清除，历史日志／部署仍保留。清理63个临时及冗余文件释放8.83 GiB，
清单在`outputs/cleanup_gray2/`，未删除正式依赖、候选末轮、过程图或原始数据。

2026-09-27：[独立灰度先验对照](SEMANTIC_GRAY_20260927.md)已完成原simple e60接续的20轮A/B。
保留全32人工GT且不留出，B额外循环读取1000训练源，以清晰原图明度生成软目标，
监督其退化经D5a的输入；已有GT覆盖排除灰度监督。仅语义头训练，D5a、匹配的joint-v3/LoRA及affinity冻结。
入口`tools/run_semantic_gray.py`，配置`config/train/semantic_gray.yaml`，输出`outputs/semantic_gray/`。
45项CPU检查及GPU短测通过，确定性零权重重放逐值一致；正式各1280更新、零失败，
各100图推理及48张过程图齐全。先验接受988／1000源、60.00%内容，类别分歧仅0.0455%；
B对A为43／8668实例改判，test101两处均在左边缘，三个较大变化占改判实例面积72.27%。
多数分数变化表现为向0.5靠近，尚无准确率结论；不原样扩轮或晋级，后续改为上方单向约束对照。

2026-09-26：[原图语义教师检查](SEMANTIC_TEACHER_PROBE_20260926.md)完成32＋64训练源、每源两次退化。
入口`tools/probe_semantic_teacher.py`，配置`config/experiments/semantic_teacher_probe.yaml`；
冻结原图S-align、原simple e60、D5a及匹配的joint-v3编码器，未启动训练或测试图推理。
64图共同可靠区两教师零类别分歧，原图教师独有覆盖仅0.65%；不直接投入仅换教师的训练。
私有`outputs/semantic_teacher/`保留完整报告及96张缩略图，本地分析在`output/semantic_teacher_analysis/`。
未创建留出集；已见GT接近满分不代表泛化。后续affinity宽带检查未在本轮开展。

2026-09-26：[训练／测试外观检查](SEMANTIC_DOMAIN_20260926.md)已完成全量一次性统计。
随后实施[20轮语义一致性对照](SEMANTIC_CONSISTENCY_20260926.md)：A/B完整继承原simple e60，
共同模拟D5a前的亮度、对比及轻微色偏，仅B增加1000训练源的可靠预测一致性。
全32新GT无留出，每组1280更新，B先完整遍历1000源再抽280张；affinity及其匹配编码器冻结。
入口`tools/run_semantic_consistency.py`，输出`outputs/semantic_consistency/`；A/B各20轮、零失败，
各100图完整推理与48张过程缩略图已完成。B比A仅7／8668个对应实例改判；全1000源均贡献可靠像素，
实际遍历1.28次，可靠区教师／学生概率均差0.000386。暂不原样延长到60轮，无新黑盒、不晋级。
私有分析在`outputs/semantic_consistency/analysis/`；affinity宽带仅列为后续定位候选。
训练结束后的快盘清理已完成：25个临时权重、释放8.15 GiB，正式权重／数据／日志／图册保留。

2026-09-26：[D5a前后校光](ILLUMINATION_CALIBRATION_20260926.md)优先检查语义与外观。
固定原simple e60＋V e115及joint-v3配套特征，校光从零训练，同一权重分别前置/后置D5a。
全1000原图、60轮、固定四图五组monitor，不留代理、不打黑盒包；启动状态见专题记录。

2026-09-26：[D5a双头冷启动](BACKEND_COLD_20260926.md)经用户确认于14:19停止，最后完整e93／5952更新，零失败。
原计划纯SSL起点、随机simple/affinity、60轮头预热＋60轮联合，同时跳过尚未消融的Stage1/joint-v3，
不能作既有路径的受控冷启动。e90四图完整输出出现碎分割风险，训练loss仍缓降，不宣称已饱和或黑盒退步。
训练及队列均退出；保留`outputs/backend_cold/`与`outputs/backend_cold_audit/`中的e60/e90/e93、过程图和停止回执。
后续自动全量推理已取消，未生成本轮正式提交包；新无标签对照未启动，默认部署不变。

2026-09-26：[固定simple光照对照](SEMANTIC_LIGHT_20260926.md)只在D5a输出后加入在线光照增强，
其余训练、初始化与推理保持原simple配方；全32图、60轮／3840更新，历史simple作直接对照。
入口`tools/run_semantic_light.py`，私有目录`outputs/semantic_light/`，保存过程图并在结束后自动推理／渲染。
52项CPU检查及GPU短测通过；正式60轮、100图推理与包校验现已完成，56张过程图齐全。
配对与冻结核验通过；实际非零调光37.97%，71/8668个对应实例改判，四图复测无一致鲁棒性增益。
本地包`output/semantic_light_analysis/semantic_light.zip`回分`0.8474/0.8445`（84.595），
比原simple仅+0.055分，无明显实用增益；不晋级，默认模型不变。
后续仅讨论D5a后端冷启动／实例级联合建模，没有启动新训练。
用户提醒后的[历史核查](MASK_SET_HISTORY_20260926.md)确认Mask2Former式`mask_set`已多轮尝试，
最终覆盖仍落后当时主线，撤回将其列为未尝试优先新方向的建议。

2026-09-25：按用户授权，将扩散修复、后端／语义对照及诊断工具归入`main`。
来源分支为`codex/diffusion-restoration-20260923`（`c5fffe0`），保留其完整历史与分支供回溯。
本次仅合并代码、配置与汇总记录；各候选是否采用仍按下述黑盒结论，推理配置与模型权重选择不变。

2026-09-25：[有标签光照诊断](LABELLED_LIGHT_20260925.md)完成32图／5126固定GT实例，
无留出。全局±0.10没有新增实例错误，A/B局部调光每条件最多新增1错，核心全部正确；
未达到启用新增强训练的证据门槛，训练及推理配方维持原样。
入口`tools/labelled_light_probe.py`；私有结果`output/labelled_light/`。
该训练内诊断不排除测试分布的光照作用，不涉及实际预测实例的划分错误。

2026-09-25：[D5a后语义适配／简化头](SEMANTIC_D5A_20260925.md)采用全32张新GT、每组60轮。
A继续训练现有完整语义头，B全新随机FPN＋分类层、无直接RGB修正；D5a、SAM2/LoRA和affinity
全部冻结。同在线退化与噪声，保留过程图，不划代理。入口`tools/run_semantic_d5a.py`、
输出`outputs/semantic_d5a/{full,simple}`，结束后自动两组全量推理、打包与渲染。
初始化、结构、学习率同时不同，不作为纯结构消融。现已回分full
`0.8473/0.8461`（84.670）、simple `0.8471/0.8437`（84.540），相差0.130分；
两组mIoU略升但面积项退步，不晋级权重，简化结构可继续作研究底座。
现已完成，见[训练结果与亮度响应](SEMANTIC_D5A_ANALYSIS_20260925.md)：各3840更新零失败、
各56张过程图，A/B形状配对后267/8668实例类别不同。全量包与图已下载并校验。
4图固定330实例、三语义头的9条件亮度探针完成，零偏移精确复现；test101加亮0.10时A/B
有5/7实例翻转，上部大区未普遍改变类别。只确认模型亮度敏感，不证明翻转正确或应统一提亮提交。
没有启动新训练，默认模型和阈值不变。

2026-09-25：[D5a外观诊断](D5A_APPEARANCE_20260925.md)完成100图／8668实例／10种模型响应。
颜色与平滑明暗变化较小；分类转向主要随明暗细节／局部对比变化，已出现在编码特征→粗语义路径。
仅换高分辨支路RGB不能消除大部分转向，残差平均在抵消偏移。
本轮未训练、未调阈值、未产生新提交；下一步先看下方固定实例类别交叉包回分。

2026-09-25：[固定实例划分×分类来源](SEMANTIC_CROSS_20260925.md)四格全100图已完成，
baseline和D5a两原对角逐ID精确复现；两个新包保持源PNG字节不变，仅交换最终分类。
私有本地`output/semantic_cross/`；用户回报`grestored_sraw`为0.8445／0.8679（85.620），
`graw_srestored`为0.8449／0.8440（84.445）。固定PNG时D5a分类在两种划分上均小幅提高mIoU、
降低面积项与总分，支持分类变化抵消修复收益，不支持语义全面退化或唯一归因LoRA。
D5a划分＋原图分类比baseline高0.245分、比旧v4低0.095分，作为语义训练的额外参照；
最终划分已包含上游语义作用，不是纯affinity与语义头的完全分离。未切换默认部署。

2026-09-25：[D5b完成分析](RGB_DIFFUSION_D5B_ANALYSIS_20260925.md)：从零、全1000图、
60轮／15000更新、零失败，350张过程图与配对核验完整。短链监督改善后续步相对D5a的误差，
但32张新端点模糊两模型均首步最好；D5b第3步比自身首步RGB／梯度误差高6.78%／4.30%。
首步未改善，不晋级、不单独出黑盒包。训练代码`4834157`；
入口`tools/run_rgb_d5b.py`、输出`outputs/d5b/`，分析私有目录`output/d5b_analysis/`。

2026-09-24：[D5a完成分析](RGB_DIFFUSION_D5A_ANALYSIS_20260924.md)：候选/原空间配方对照
均从零完成全1000图、60轮／15000更新、零失败，不继承D4，四类过程图与配对检查完整。
新端点退化强区RGB／梯度误差比对照降低7.06%／1.92%；真实补边未获证明，16步仍差于首步。
沿用原分割后端的候选与对照首步黑盒包均完成全量推理、双端校验与分割渲染，本地`output/d5a_analysis/`。
用户回报control 0.8511／0.8486（84.985）、D5a 0.8469／0.8560（85.145）；
相对control净+0.160分，但相对baseline−0.230、旧v4−0.570，不晋级。
D5b维持原计划，直接参照为本次D5a；两包未混入后端微调。
入口`tools/run_rgb_d5a.py`，不包含短链监督。后端适配两包已回分，结论见下。

2026-09-24：D4首步复赛回分0.8490/0.8532，折算85.110，低于旧v4。
已完成[冻结D4开关＋后端配对微调](BACKEND_D4_ANALYSIS_20260924.md)：两组各60轮／3840更新、零失败，
同增强/全人工标注/64图既有SAM2监督，只比较是否经过D4；短链监督暂缓。
入口`tools/run_backend_ablation.py`，短输出`outputs/d4_adapt/`；两包全量推理、双端校验与渲染均完成。
用户回报D4适配0.8476/0.8563（85.195）、raw适配0.8391/0.8010（82.005）。
D4适配相对原D4仅+0.085分，仍低于baseline和旧v4；raw面积项明显退步，相对baseline总分−3.370。
本轮适配不晋级；两组差3.190分不能当作相对稳定部署的收益。上方D5a为独立修复实验。

扩散研究已纳入`main`，历史训练路线见[扩散安排](RGB_DIFFUSION_ROADMAP_20260923.md)
和[D1全量60轮分析](RGB_DIFFUSION_ALL60_ANALYSIS_20260923.md)：1000图/15000次更新已完整完成，
首步优于16步，16步细碎颗粒和梯度误差偏高，暂不晋级。
[D2低噪声对照](RGB_DIFFUSION_D2_ANALYSIS_20260923.md)已完成原60轮日程的前20轮/5000更新，
零失败、48张过程图完整；kappa降到0.03后输出近乎原图，不晋级、不自动续训。
[D3起点强化20轮分析](RGB_DIFFUSION_D3_ANALYSIS_20260923.md)：5000更新零失败，48张过程图完整，
实际起点抽样50.13%；首步RGB/梯度误差比为0.607/0.904，完整16步为0.782/1.140。
修复能力已恢复，但多步退化仍在。[逐步测量](RGB_DIFFUSION_D3_TRAJECTORY_20260923.md)进一步确认
两类固定训练内输入、三采样种子的整图误差均首步最低。
[D4与D3对照](RGB_DIFFUSION_D4_ANALYSIS_20260923.md)现均完成60轮／15000累计更新，零失败。
32张训练源图的新空间配对上，D4首步RGB／梯度误差相对降低19.49%／5.38%，均匀配对小幅退步；
16步仍未优于首步。建议保留空间增强并独立测试短链监督，尚未实施，不替换已验证分割链。
D1的112张过程图完整保留。用户回报v6复赛0.8448/0.8506，低于v4的0.8465/0.8678，v4继续作为有分数的修复参照。
下面保留扩散分叉前的初赛主线记录；复赛实验状态以本页上方记录为准。

最新交接：[复赛切换与初赛实验收束（2026-09-21）](HANDOFF_SEMIFINAL_20260921.md)。
复赛数据据用户回报位于23411服务器的autodl-fs盘，准确目录与输入结构待核验；本轮未执行数据分析或替换。
下列黑盒成绩均来自初赛，不能当作复赛基准分数。

## 结论与主线

2026-09-19回分未通过：[Stage1最终任务适配](STAGE1_FINAL_TASKS_20260919.md)。
从Stage1共同初始化，两头预热5轮、联合LoRA30轮，再冻结特征分别精修affinity120轮和语义20轮。
该流程替代旧joint-v3，不读取V6/旧固定教师，不是纯删除消融；6600次实际更新、零AMP跳步。
联合e35、几何e57、语义e14按loss选优。内部原始语义与面积改善，七图实例匹配1000→971、铁素体合并增加；
用户回报官方`0.8448/0.7699/80.735`，较统一组合低4.190分，拒绝替换以下已验证基线。
六张几何诊断图全部在联合训练中出现过，内部面积排序未推广到黑盒。
两份测试预测的对应分析同时发现小实例类别变化与划分变化，不能把掉分全部归为affinity或认定joint-v3不可替代。

2026-09-17更新：S用户回报官方`0.8393/0.8714/85.535`，较L高0.765，作为当前总分最佳候选；
S实际仍搭配L旧GT几何，不能把成绩写成“两头新GT”或“S+V”组合。L/V保留回退。
完整链路、旧GT及固定教师依赖、已确认的一致性坐标问题和后续顺序见[训练统筹](TRAINING_ROADMAP_20260917.md)。

后续简化几何基线采用 V，即 `joint-v3 boundary FPN → 120轮长程G2`，
在此前移除独立G0/G0-long/G1的基础上，进一步省去V6独立边界训练。
固定E10a/top2部署，V用户回报官方`0.8430/0.8499/84.645`；L为`0.8416/0.8538/84.770`，
V以总分低0.125分的代价减少一段训练，作为可接受的简化方案。L保留成绩回退及当前S实验固定参照。
完整证据见[V记录](G2_SKIP_V6_20260916.md)及[L记录](G2_LONG_OLDGT_20260914.md)。
当前 [N实验](G2_LONG_NEWGT_20260916.md) 只替换26张人工训练图的补缝GT；原验证loss选优及
其余L配方保持不变。N best115内部六图小幅改善，用户随后回报黑盒`0.8415/0.8476`（按上下文归属N），
折算84.455、低于L 0.315分，尚无官方增益。后续按单变量顺序开展新GT语义监督对照及独立一致性实验。
[V实验](G2_SKIP_V6_20260916.md)单独跳过V6边界训练，使用joint-v3边界FPN及L的旧GT/120轮配方；已完成120轮，loss-best为115。
固定E10a/top2内部六图匹配777→779、GT惩罚mIoU 0.71033→0.71663、铁素体面积误差34.33%→33.85%，
但珠光体多余预测与分裂/合并略增。官方结果支持上述简化选择，不构成统计等效或显著提升证明。
[S语义新GT配对实验](SEMANTIC_NEWGT_20260916.md)两组均完成20轮、1240次实际更新和零AMP跳步，
新GT按共同验证loss选中第20轮，旧GT为第7轮。固定L部署，语义7图原始像素mIoU为0.95763/0.94102，
新GT收益也存在于原标注覆盖区；最终实例表现却近似，面积误差均高于E10a/L内部基准。
保留实际语义25/7划分及原定L对照；S官方面积项上升而mIoU略降，内部代理未正确预测官方方向。
S-align用户回报`0.8407/0.8620/85.135`，相对S总分−0.400、mIoU略升，接受正确对齐作为后续研究起点；
V-noSAM2为`0.8362/0.8181/82.715`，相对V两项均下降、总分−1.930，保留64源图SAM2掩码监督。
两项分别完成20轮/1240次、120轮/3120次有效更新且0跳步，来源与冻结契约通过。
完整证据见[S-align / V-noSAM2记录](ALIGN_NOSAM2_EXPERIMENT_20260917.md)。
2026-09-19用户回报[S-align+V统一组合](UNIFIED_ALIGNMENT_AUDIT_20260919.md)官方`0.8416/0.8569/84.925`：
较S-align+L低0.210分、较V高0.280分，接受其作为后续统一研究对照，原S仍为最高分回退。
结果支持两项修改组合使用，没有证明统计等效、固定教师项有效或joint-v3可删除。
当前Stage1最终任务替代黑盒未通过，拟先做固定权重的语义/affinity输出交换定位；固定教师净贡献及EMA实验仍待独立检验，
后续安排见[训练统筹](TRAINING_ROADMAP_20260917.md)，不恢复已删掉的独立G阶段。
V6文件在语义冷启动中的载体引用仍待独立替换为已核对相同的joint-v3语义/LoRA并验证。
本轮未修改默认推理文件；以下保留历史部署与实现约定。

历史黑盒基线为“E10a 单语义模型 + G4b 8 通道 affinity → gated boundary → `high=0.65`、
seal2、局部重建 → 受阻分水岭”，得分 `0.8381/0.8408/83.94`。E10a 冻结 V6 特征并
冷启动完整高分辨率语义解码器，现已晋级；E9 的 `0.8421/0.7917/81.69` 保留为历史回退，
不做连续融合。V6/B2 边界路线继续作为几何回退。

当前推荐部署产物为 `outputs/deployment/e10a_g4b_fused.pth`：单个共享 SAM2+LoRA encoder
同时连接 E10a semantic decoder 和 G4b affinity decoder。组合包参数量 `81.667394M`、大小
`326836899` bytes；保存后重新构建的 semantic/affinity logits 最大绝对误差均为 `0.0`，
`test_009` 的实例 PNG 与类别 JSON 也和原三 checkpoint 管线字节级一致。

G3/G4b 尚不能单独证明黑盒竞赛成绩提升，测试目检仍以欠分割为主要风险；GT 前景上的 Oracle
图重建仅作诊断。graph-v1 `area200` 黑盒为 `0.8268/0.8365/83.17`，未超过 E10a watershed；
graph-v2 `area150` 因笔直、失真的归并边界被目检淘汰。G7 在固定协议测试 A/B 中进一步
减少实例并加重欠分割风险，已停止晋级。当前候选检验 `SSL LoRA → semantic/affinity 双头`
短训练链，主线部署仍完全不变。详见
[AFFINITY_GRAPH_AB_20260828.md](AFFINITY_GRAPH_AB_20260828.md)；历史 affinity 审计见
[AFFINITY_DEPLOYMENT_EVALUATION.md](AFFINITY_DEPLOYMENT_EVALUATION.md)，短链实验见
[DIRECT_SSL_SEMANTIC_AFFINITY.md](DIRECT_SSL_SEMANTIC_AFFINITY.md)。

`outputs/stage2_center_heatmap/best_model_stage2.pth` 只保留为负面对照。现有中心 GT 由每个
Labelme polygon 生成一个种子，polygon 与物理晶粒并不等价；同时中心损失和边界损失共享
`boundary_fpn`，实测造成背景雾化、铁素体大块欠分割、珠光体碎裂及薄环嵌套。

当前 B2 架构实验以 V6 权重为语义锚点，并满足：

1. 语义路径冻结或低学习率保护；
2. 边界 refine 路径独立训练，不接收不可靠中心标签的梯度；
3. 若重试辅助任务，使用独立 FPN/stop-gradient，并先验证 GT 与物理实例的一致性；
4. 后处理把高置信边界作为硬障碍，而不是仅改变分水岭可视化灰度。

## 2026-08-21 新发现：颜色先验

已完成 32 张有标签训练图的 ferrite/pearlite 颜色分布分析，详细结果见
[`docs/COLOR_SEPARABILITY.md`](COLOR_SEPARABILITY.md)。结论是 GT 内部存在很强的明度分离：
Lab `L*` 的 pooled 单阈值平衡准确率约 `0.9917`，包含边界混色仍约 `0.9846`；leave-one-image-out
平均约 `0.9912`（含边界约 `0.9845`）。

该结果是颜色先验的重要证据，但不能替代语义头：统计来自 GT 区域，且不同图像的最佳明度阈值
约在 `63.7~81.0` 间变化。后续应采用固定的无标签 holdout monitor，并将 Lab `L*` 先验作为
自适应软辅助或融合信号，不能把固定全局阈值直接写入实例分割主路径。

后续对话进程接续本项目时，必须先查看 `docs/COLOR_SEPARABILITY.md`，特别是其中的限制条件和
“不能替代语义头”的结论；当前 E1/V6 主线、实例 ID `<=65535` 约束和测试集无标签原则保持不变。

## 入口

```bash
conda activate sam2_env

# 当前固定 G4b 部署基线
python tools/run_affinity_submission.py \
  --config config/inference/final_affinity_g4b_high065.yaml

# E10a + G4b：将共享主干和两个分支一次性打包
python tools/build_fused_deployment_checkpoint.py \
  --config config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml \
  --output outputs/deployment/e10a_g4b_fused.pth

# 推荐提交推理入口：只加载一个组合 checkpoint
python tools/run_fused_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml \
  --checkpoint outputs/deployment/e10a_g4b_fused.pth

# 当前 E7b-A 语义专项训练（V6 初始化、decoder-only、20 epoch）
python train_stage2.py \
  --config config/train/stage2_semantic_e7b_decoder20.yaml

# 当前 E9：冻结 V6/G4b，只训高分辨率语义残差（20 epoch）
python train_stage2.py \
  --config config/train/stage2_semantic_e9_highres20.yaml

# E10a：冻结 V6 LoRA/G4b，冷启动完整高分辨率语义解码器（20 epoch）
python train_stage2.py \
  --config config/train/stage2_semantic_e10a_cold20.yaml

# SSL 直达双头：先检查数据门槛，再执行 5 epoch head warm-up + 20 epoch joint LoRA
python train_direct_semantic_affinity.py \
  --config config/train/direct_ssl_semantic_affinity.yaml --check
python train_direct_semantic_affinity.py \
  --config config/train/direct_ssl_semantic_affinity.yaml

# E7b 完成后：同一 G4b 几何，只替换 semantic decoder
python tools/run_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_e7b.yaml

# 历史 V6 参考推理
python inference.py --config config/inference/v6_reference.yaml

# Stage 1 全监督（LoRA 可训练）
python train.py --config config/train/stage1_lora.yaml

# Stage 2 B2；V6 作为冻结语义锚点，落地独立高分辨率 refine head
python train_stage2.py --config config/train/stage2_refine_v6.yaml \
  --phase boundary --tag refine_v6_b2

# B2 单变量重试：V6 初始化 + refine-only + 物理显微增强（5 epoch）
python train_stage2.py --config config/train/stage2_refine_v6_physaug.yaml \
  --phase boundary --tag refine_v6_b2_physaug

# Stage-0 / E0：先验证 B2 在 310 次纯监督更新下确实能够学动
python train_stage2.py --config config/train/stage2_refine_v6_stage0_control.yaml \
  --phase boundary --tag refine_v6_stage0_control

# Stage-0 Long：20 epoch/1240 更新，观察纯 refine 收敛与背景雾化趋势
python train_stage2.py --config config/train/stage2_refine_v6_stage0_long.yaml \
  --phase boundary --tag refine_v6_stage0_long

# 对应的两档质量感知 TTA（训练完成后）
python inference.py --config config/inference/b2_quality_aware.yaml
```

推理常用参数可直接从 CLI 覆盖，不再复制临时 YAML：

```bash
python inference.py --config config/inference/v6_reference.yaml \
  --boundary-threshold 0.35 --min-instance-area 50 --no-center-seeds \
  --output_dir outputs/inference/<name>
```

每个推理目录都会生成 `inference_manifest.json`，记录 checkpoint、架构、实际阈值和三类实例统计。

## 物理增强与质量感知推理边界

- 训练增强只模拟显微成像中可解释的曝光/白平衡变化、轻度失焦、采样分辨率下降、低频照明和
  低对比抛光划痕；单张图只组合 1~2 项，保留足够干净样本。
- 划痕保持原 GT，作为“图像强线条不一定是晶界”的 hard negative；不再使用规则圆形遮罩。
- 推理只分 `standard`/`weak` 两档。弱档保留原图 logits，并融合一张确定性校正视图；所有
  阈值偏移由配置显式给出，便于逐项关闭和复现。
- 分档不以预测实例数、铁素体平均面积、薄环或嵌套现象为目标，也不会跨测试集拟合统计量。
- 实例图使用单通道 `uint16`，每图最多 65535 个非零 ID。只有候选超过格式上限时才按局部
  邻接关系合并最小区域；该保护仅处理输出格式上限，不反向改变分水岭参数，也不再把正常的
  第 256 个及后续实例强制合并。

## Checkpoint 契约

新 checkpoint 格式版本为 2，必须包含：

- `architecture`：encoder、FPN 通道、`boundary_refine`、`center_head`、LoRA；
- `provenance.git_commit`：训练代码版本；
- `config`：完整生效配置；
- decoder/LoRA 权重、epoch 与 best score；
- 中间 checkpoint 额外包含 optimizer/scheduler，用于恢复训练。

检查权重而不构建 SAM2：

```bash
python tools/inspect_checkpoint.py outputs/stage2_v6/best_model_stage2.pth
```

推理和 `--resume` 默认严格核对架构。`--allow-architecture-mismatch` 仅用于明确的消融，不能作为
普通兼容开关。Stage 2 的跨架构初始化仍允许宽松加载，但日志必须检查 missing/unexpected keys。

## 目录职责

```text
config/default_config.yaml       当前 V6 参考基线
config/inference/                同架构推理配置
config/train/                    Stage 1 / Stage 2 训练配置
config/experiments/              改架构/训练目标的实验配置
outputs/stage2_v6/               V6 参考 checkpoint
outputs/stage2_center_heatmap/   中心热图负面对照 checkpoint
outputs/runs/                    配置、指标、环境及 best 权重硬链接
downloads/                       下载到本机的目检结果（不提交）
```

## 服务器保留策略

- 活跃/基准实验：保留 `best_model*.pth`、`metrics.csv`、配置和少量 monitor；
- 负面对照：保留一个 best 权重和一组有代表性的推理可视化；
- 只有计划恢复训练的运行才保留一个中间 epoch（默认最后一个）；
- 删除其余周期 checkpoint、重复 run 权重、smoke 输出和已判废阈值扫描；
- 伪标签缓存只有在配置仍引用且生成代价明显时保留。
