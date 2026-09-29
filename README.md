# 低碳钢金相图像相区分割

> 2026-09-29：[原生裁块affinity训练与分析](docs/AFFINITY_NATIVE_20260928.md)完成：1024／512独立各20轮／1280更新，同初始化与旧control配对、冻结与严格重载通过，6套新增100图输出齐全。512同尺度适配使铁素体<200像素小片6307→354；1024原生结果相对旧全图铁素体数+2.65%、逐图平均面积变化中位−2.48%，512为+12.95%／−9.86%。四张已见GT图相对旧全图有匹配改善，但同尺度训练后铁素体有效匹配略降，存在去碎片与保留真边界的取舍。按用户要求，两组e20原生推理提交包已下载到`output/native_analysis/native1024.zip`和`native512.zip`，各100图、格式与传输哈希通过，等待回分；未晋级或扩轮。入口`tools/analyze_affinity_native.py`，私有图册`output/native_analysis/index.html`；主线官方仍0.8486／0.8761。

> 2026-09-28：[原生patch与D5a检查](docs/AFFINITY_PATCH_20260928.md)完成，固定control e20，4训练＋4测试源、48次完整分区。4张基线输出逐像素复现；训练原标可信域的铁素体匹配349→365／375，匹配IoU 0.9033→0.9442，但不是独立泛化精度。测试原生1024的铁素体平均面积变化中位−10.21%、小于200像素实例5→19；512放大为−59.05%、385，存在明显碎分割风险。追加32对恢复结构检查发现：D5a在局部模糊叠加降采样时可生成被affinity识别的内部网纹，像素误差改善不能替代结构验证。私有图册`output/affinity_patch/index.html`；后续双尺度训练见上条，不改默认部署。

> 2026-09-28：[参数梯度预算短测](docs/AFFINITY_BUDGET_20260928.md)完成。最高分control e20复制两头，同4源8视图各64步；旧预算分支用于隔离共同继续训练，未重跑20轮control。附加参数梯度中位6.30%→15%，真实单步额外更新影响1.65%→3.41%；粘连修复26→27、误切修复15→15，新增粘连27→32，未通过正式20轮门槛。不晋级、不继续增权；完整过程与8视图图册在私有`output/affinity_budget/`。[后续原生patch检查计划](docs/AFFINITY_PATCH_PLAN_20260928.md)已执行，结果见上条。

> 2026-09-28：[连续界面监督20轮完成分析](docs/AFFINITY_INTERFACE_20260928.md)：候选完成1280更新及100图推理，复用control、全部输入配对与冻结／严格重载通过。新增作用仍小：98.64%旧实例匹配IoU≥0.95，实例9201→9216、铁素体5969→5965，平均面积变化中位−0.00165%。320次附加步骤全被25%输出梯度上限约束，逐轮首步参数梯度比中位4.185%，主loss几乎重合；不能只增名义系数或据此认定模型饱和。固定36张过程图与完整测试图册保留在私有`output/interface_analysis/`。保留最高分control，不原样续训或默认晋级，无新黑盒。

> 2026-09-28：[纯分隔恢复消融](docs/AFFINITY_PRESERVE_20260928.md)用户回分 **0.8499／0.8559（85.290）**，相对同权重control **−0.945分**，不晋级。mIoU＋0.0013、面积项−0.0202；铁素体实例＋5.009%、总像素仅＋0.0216%。[回分分析与下一步](docs/AFFINITY_REVIEW_20260928.md)重新读取两组完整100图包：相对各图控制铁素体中位面积，小于25%档净增141／总净增299，其余158在更大档；不能全部归因于极小碎屑。保留control e20，先检验小侧分隔及连续界面监督，不直接加阈值或放大旧稀疏监督。新统计和Jev完整概率在私有`output/affinity_review/`，本轮无新训练。

> 2026-09-28：[完整划分链定位](docs/AFFINITY_TRACE_20260928.md)完成。固定最高分control e20，32人工训练源的原图＋固定模糊共64输入，原有权重及最终输出复现核验通过。原图238/358、模糊304/554个已知粘连点对在细化阶段重新连通，其中69例通路完全处于原标可信域；现有附加项所选输出关系即使强制正确，也只修复3个粘连、0个显著过分割点对。优先检查分隔保留和有效选边，暂缓直接增权训练；这些是训练图机制诊断，不是测试精度或晋级依据。私有图及记录`output/affinity_trace/`。

> 2026-09-28 黑盒回分：普通 affinity 适配 **control e20** 为 **0.8486／0.8761（86.235）**，同链训练前85.620，旧综合最佳85.715。当前研究对照更新为该组，保留旧方案回退；candidate尚无黑盒，不随之晋级。用户所见榜首89.8，差3.565分、无两项拆分。新增[归因与下一步](docs/AFFINITY_NEXT_20260928.md)：先定位监督覆盖及完整划分链，再考虑参数梯度预算单变量20轮、复用control；尚未启动训练或改通用推理默认值。已完成CPU理想affinity合成检查与Jev辅助判断，原始概率分布存于私有`output/affinity_next/`。

> 2026-09-28：[affinity双向连通性对照](docs/AFFINITY_CONNECTIVITY_20260928.md)完成两组各20轮／1280更新及100图部署，配对、冻结、严格重载通过。共同适配使固定10图预测响应带宽中位约缩小9%，实例8668→9201／9203、小块409→504／510；新增监督额外作用很小，99.25%控制实例与候选匹配IoU≥0.95。末轮8视图复测附加参数梯度仅人工BCE的1.93%–3.38%，输出25%上限不能当作参数预算。该分析早于黑盒回分；control随后成为上条研究对照，candidate不晋级；未自动续训。私有分析`output/affinity_analysis/report/`；按授权清理约19.1 GiB，快盘可用约23.6 GiB，正式依赖与本轮final保留。

> 2026-09-27：[全图／局部尺度语义训练](docs/SEMANTIC_SCALE_20260927.md)完成20轮／1280更新及100图部署，零失败，冻结、逐步输入配对和严格重载通过。相对原LoRA控制，105／8668实例改判（45珠→铁、60铁→珠），实例数不变；逐图铁素体平均面积变化中位−0.00344%。局部任务有所学习，但全图末5轮同输入loss仍比控制高4.6%，test101未出现成片纠正；无尺度黑盒，不自动续训或晋级。入口`tools/run_semantic_scale.py`，私有`outputs/semantic_scale/`及`output/semantic_scale_analysis/`。

> 2026-09-27：[LoRA原后处理控制回分](docs/MARKER_RESTORE_20260927.md)为`0.8445/0.8557`（85.010），补齐与恢复＋统一过滤`0.8474/0.8354`（84.140）的同权重比较：后处理组合mIoU＋0.0029、面积项−0.0203、综合−0.870分，保持关闭。恢复已知50处分隔、小块410→363仍不保证综合收益；铁素体实例6101→6390、91图平均面积下降。恢复与过滤的独立贡献仍未拆分；LoRA控制也未超过历史最佳，不晋级。入口`tools/run_marker_restore.py`，私有产物见专题报告。

> 2026-09-27：[固定轮廓与种子阶段排查](docs/SEMANTIC_SPLIT_20260927.md)完成100图、8668实例的三语义版本复现；原投票及affinity最大差0。最新LoRA有466例内部混合预测，其中50例在骨架化前分开、之后相通，全部贴边。只改marker的全100前置封边使50例全部恢复分隔，但总实例8668→10575、50–199像素小块410→1136；机制验证成立，粗放修复不晋级。建议先实现有原始分隔支持的保守种子恢复，再定位剩余affinity缺边。三个诊断入口及私有对照见文档；未启动训练、不改默认部署。

> 2026-09-27：[大窗口补监督预检查](docs/SEMANTIC_COVERAGE_20260927.md)完成：全1000源、1280次原增强重放，仅14源新增48666格点，覆盖60.00000%→60.07776%；已完成LoRA对新增位置零类别分歧，仅2格未达0.9门槛。固定GT规则检查新增2962内部格点无冲突，但未提供足够新增学习信号，按预定条件不启动20轮。规则、私有缩略图及41项检查已完成，控制不重训；入口`tools/probe_semantic_coverage.py`，私有`output/semantic_coverage_probe/`。

> 2026-09-27：[语义专用LoRA完成分析](docs/SEMANTIC_LORA_20260927.md)：20轮／1280更新、100图部署与96张过程图完成；重放gray4输入、直接复用控制，affinity冻结及严格重载通过。相对gray4改判65／8668实例（62珠→铁、3铁→珠），62处原分数距阈值不超过0.1；test101仅新增一处小区域改判。固定合成光照任务进一步满足，后5轮52.81%步骤无额外梯度；不能把规则拟合改善当作真实泛化突破。末轮`outputs/semantic_lora/candidate/epoch_020.pt`，私有12图及曲线`output/semantic_lora_analysis/`；无新黑盒、不晋级或原样续训。

> 2026-09-27：[gray4完成分析](docs/SEMANTIC_GRAY4_20260927.md)：20轮／1280更新、100图部署、两套共96张过程图完成，控制复用与冻结检查通过。对控制102／8668实例改判（99珠→铁、3铁→珠），gray3为35；test101额外改判2→10，仍以局部变化为主。固定64训练源的同视图复测中，强全局／局部区域冲突120／60→1／1；固定难例e5已消除区域类别冲突，后5轮同时存在低学习率和当前合成目标接近满足，不能只归因于学习率或宣称模型整体饱和。末轮`outputs/semantic_gray4/prior/epoch_020.pt`，私有分析`output/semantic_gray4_analysis/`；无新黑盒、不晋级或自动续训。

> 2026-09-27：[gray3完成分析](docs/SEMANTIC_GRAY3_20260927.md)：仅候选完成20轮／1280更新、100图部署及48张过程图，旧控制复用成功且全1280步配对通过。相同无标签数据和先验接受位置上，灰度规则分歧比gray2减少31.17%；逐轮首步附加／GT梯度比均值0.368%→3.876%。对控制改判6→35／8668（34珠→铁、1铁→珠），类别变化像素0.03953%，test101新增两处边缘小区域改判，未解决大块语义问题。已渲染10图含3张随机非monitor图；私有分析`output/semantic_gray3_analysis/`。保留候选，建议下一步检查并增加可靠困难区域的训练机会，不继续盲目放大系数；无新黑盒、未晋级或启动下一轮。

> 2026-09-27：[单向灰度约束完成](docs/SEMANTIC_GRAY2_20260927.md)：各20轮、1280更新、100图部署及48张过程图齐全。仅0.06146%接受位置仍受罚，47.42%步骤额外loss为零；逐轮首步加权先验／GT梯度比均值由上轮35.04%降至0.368%，名义增权未形成实际增强。对控制仅6／8668实例改判，test101零额外改判；不晋级或原样扩轮，建议先限制幅度地修正活动位置的损失平均。两控制末态全部模型张量及100图输出完全一致，后续复用旧控制；重复权重已清除。按要求清理63个临时／冗余文件，释放8.83 GiB，正式依赖、候选及过程记录保留。私有分析`output/semantic_gray2_analysis/`；后续见上方gray3，无新黑盒。

> 2026-09-27：[灰度先验20轮对照完成](docs/SEMANTIC_GRAY_20260927.md)：A/B各1280更新、100图完整推理及48张过程图，冻结与配对核验通过。独立先验遍历1000源、接受988源，但接受域类别分歧仅0.0455%；B对A为43／8668实例改判，约83.44%实例的分数向0.5靠近。test101仅左边缘两处阈值附近改判；test103等三处较大珠→铁变化占改判实例面积72.27%，不能按改判数量推断黑盒无收益。暂不晋级或原样延长，建议下一轮只改为满足灰度证据后停止处罚的先验目标。入口`tools/run_semantic_gray.py`，私有分析`output/semantic_gray_analysis/`，无新黑盒成绩。

> 2026-09-26：[原图语义教师检查完成](docs/SEMANTIC_TEACHER_PROBE_20260926.md)：全32人工源＋固定64其他训练源、每源两次退化，参数冻结核验及96张缩略图齐全。64图共同可靠域2,144,130格点中，原图S-align与simple自身教师零类别分歧、概率均差0.000143；原图教师独有覆盖仅0.65%，人工图独有区域的错误方向又多于纠错方向。暂不启动仅换教师的20轮训练；训练内5126个固定GT实例接近满分不代表测试泛化，affinity宽带仍待独立定位。

> 2026-09-26：[语义一致性对照完成](docs/SEMANTIC_CONSISTENCY_20260926.md)：A/B各20轮、1280更新、零失败，各100图完整推理与48张过程图齐全。原simple→A/B分别280／287个对应实例改判，但B相对A仅7／8668个（0.0808%），test101零额外改判。全1000无标签源均贡献可靠像素，实际仅遍历约1.28次，可靠区教师／学生概率均差0.000386；当前配方额外作用很小，不直接续到60轮或晋级。affinity保持冻结，512网格及融合后宽带问题列为后续定位候选，未启动改造。此前[一次性外观检查](docs/SEMANTIC_DOMAIN_20260926.md)仍有效；无新黑盒成绩。按用户要求清理25个临时权重，已释放8.15 GiB，正式产物保留。

> 2026-09-26：[D5a前后校光实验](docs/ILLUMINATION_CALIBRATION_20260926.md)已完成全1000源、无留出的60轮训练，固定V e115 affinity、原simple e60语义及joint-v3配套编码器。排除此前10图后随机另抽8图，完成e15/e20/e60五路径完整输出对照；全100图提亮幅度没有整体衰减，但随机样本的类别翻转仅约1.4%–2.0%，尚无准确率增益结论。私有图册位置及统计见专题记录，暂不制作黑盒包。

> 2026-09-26：[D5a双头冷启动](docs/BACKEND_COLD_20260926.md)经用户确认于14:19停止，最后完整第93轮／5952更新，零失败；保存e60/e90/e93权重与过程诊断，后续自动全量推理已取消。该短链同时绕开尚未消融的Stage1/joint-v3并重置两头，不能作为沿既有路径的冷启动对照；四图完整输出出现碎分割风险，不能解释为已证明训练无效或饱和。默认部署不变，后续保留既有上游依赖。

> 2026-09-26：[Mask2Former历史核查](docs/MASK_SET_HISTORY_20260926.md)确认`codex/maskset-mainline-pseudo`保留完整`mask_set`多轮实验。最终EMA e144在五图已知区未覆盖8.91%，当时主线1.27%；615/790对715/790匹配。核心实例联合预测机制已尝试，撤回将其列为未尝试新方向的建议；这些是历史同图诊断，非复赛黑盒。

> 2026-09-26：[固定simple的光照增强回分](docs/SEMANTIC_LIGHT_20260926.md)为`0.8474/0.8445`（84.595），比原simple仅+0.055分，无明显实用增益，不晋级。60轮／3840更新及56张过程图完整，配对与冻结核验通过；实际非零调光37.97%，71/8668个对应实例改判，四图复测无一致鲁棒性收益。后续讨论转向D5a后端冷启动与实例级联合建模，尚未启动新训练，默认部署不变。

> 2026-09-25：扩散修复及配套语义对照、光照诊断代码按用户授权归入`main`；集成来源为`codex/diffusion-restoration-20260923`（`c5fffe0`），研究分支保留回溯。后续从主线继续开发，各候选的黑盒成绩与采用结论见下方记录；本次合并不更改推理配置或模型权重选择。

> 2026-09-25：[有标签光照诊断](docs/LABELLED_LIGHT_20260925.md)完成全32图／5126固定GT实例、清晰与模糊两输入条件。三语义头在全局±0.10下均无新增实例错误，A/B局部调光每条件最多新增1错、核心投票全部正确；未发现足以支撑追加光照训练的短板。训练源接近满分不代表测试鲁棒，且正偏移有明显截断。保留诊断工具与私有`output/labelled_light/`结果，训练配方和默认部署不变；建议下一步检查实际预测实例的边缘／混类投票以及低对比与模糊。

> 2026-09-25：[D5a语义两方案回分](docs/SEMANTIC_D5A_ANALYSIS_20260925.md)：full `0.8473/0.8461`（84.670）、simple `0.8471/0.8437`（84.540），只差0.130分，支持保留简化结构研究；相对原D5a的mIoU微升、面积项下降，两权重均不晋级。初始化与学习率亦不同，不是严格结构消融。已有训练内光照阴性结果不能排除真实工况差异，后续建议固定simple做光照增强单变量黑盒。两组各60轮／3840更新、56张过程图；包在`output/semantic_d5a_analysis/`。本次未启动新训练或更改默认推理。

> 2026-09-25：[D5a外观与语义定位](docs/D5A_APPEARANCE_20260925.md)完成全100图、8668个固定实例与10种诊断输出。平均L*仅−0.196，细纹理强度2.51倍；低频和色度互换表明分类转向主要随明暗细节／局部对比变化。特征/RGB交叉进一步定位到编码特征→粗语义路径，高分辨残差总体抵消部分偏移。尚不能判断翻转真伪或哪一层失配；等待既有交叉包回分，未启动新训练。私有图与统计`output/d5a_appearance/`。

> 2026-09-25：[分类来源交叉](docs/SEMANTIC_CROSS_20260925.md)已回分：D5a划分＋原图分类`0.8445/0.8679`（85.620），原图划分＋D5a分类`0.8449/0.8440`（84.445）。两种分类来源下D5a最终划分都小幅改善两项；固定PNG改用D5a分类则mIoU微升、面积项下降，支持分类变化抵消部分修复收益，不支持语义整体变差或唯一归因LoRA。85.620较baseline高0.245，仍低于旧v4 0.095，作为语义训练的额外参照。私有包`output/semantic_cross/`。

> 2026-09-25：[D5b完成分析](docs/RGB_DIFFUSION_D5B_ANALYSIS_20260925.md)：从零、全1000图、60轮／15000更新、零失败，350张过程图与配对核验完整。短链监督使32张新端点模糊的第3步RGB／梯度误差比D5a同期降低3.76%／2.77%，但仍比自身首步高6.78%／4.30%；首步未改善，暂不晋级、不单独出黑盒包。入口`tools/run_rgb_d5b.py`，短目录`outputs/d5b/`。

> 2026-09-24：[D5a回分分析](docs/RGB_DIFFUSION_D5A_ANALYSIS_20260924.md)：用户回报control为0.8511／0.8486（84.985），D5a为0.8469／0.8560（85.145）。端点配方使mIoU−0.0042、面积项+0.0074，净+0.160分；相对历史D4仅+0.035，仍低于baseline 0.230分、旧v4 0.570分，不晋级。两组均从零、全1000图、60轮／15000更新、零失败；沿用原后端。合成恢复改善尚未转化为总分突破；D5b维持原计划，其直接参照为本次D5a。私有包与图在`output/d5a_analysis/`。

> 2026-09-24：[后端适配配对回分](docs/BACKEND_D4_ANALYSIS_20260924.md)：用户回报D4组0.8476/0.8563（85.195），raw组0.8391/0.8010（82.005）。D4适配相对原D4仅+0.085分，raw相对baseline −3.370分；D4适配仍低于baseline的85.375及旧v4的85.715，不晋级。两组各60轮／3840更新、0失败；私有短目录`output/d4_adapt/`。

> 扩散研究来源分支：`codex/diffusion-restoration-20260923`，现已归入`main`。[D4与D3同预算分析](docs/RGB_DIFFUSION_D4_ANALYSIS_20260923.md)：两组均完成60轮／15000累计更新，零失败。32张训练源图的新空间模糊配对上，D4首步RGB／梯度误差比D3降低19.49%／5.38%，均匀模糊小幅退步；16步仍差于首步，真实图补边未获证明。后续D5a与D5b均已完成。短入口为`outputs/rgb_restoration_diffusion_d4/`与`outputs/rgb_restoration_diffusion_d3/`。v4继续作为已验证修复基准，上述后端适配回分未证明不适配是主要瓶颈。

> 2026-09-21：[复赛交接入口](docs/HANDOFF_SEMIFINAL_20260921.md)。用户已把复赛数据放到23411服务器的autodl-fs盘下，准确目录待下轮核验；本轮仅整理工作区，未分析或替换数据。以下分数均属初赛。

> 2026-09-17：S（新GT语义e20 + L旧GT几何/top2）用户回报黑盒`0.8393/0.8714/85.535`，为当前总分最高候选，L/V保留回退。
> 后续简化几何基线采用 V（joint-v3直接进入长程G2，固定E10a/top2），用户回报黑盒
> `mIoU=0.8430 / 面积项=0.8499 / 总分=84.645`；支持省去V6独立边界训练，见 [V记录](docs/G2_SKIP_V6_20260916.md)。
> L保留成绩回退与当前S实验的固定参照：`0.8416/0.8538/84.770`，较V高0.125分，见 [L记录](docs/G2_LONG_OLDGT_20260914.md)。
> [N：仅替换人工训练GT](docs/G2_LONG_NEWGT_20260916.md) 用户回报黑盒`0.8415/0.8476`（按上下文归属N），折算84.455；L仍作成绩基准。
> [S记录](docs/SEMANTIC_NEWGT_20260916.md)：官方总分较L高0.765，mIoU略降、面积项上升；收益稳定性及GT独立贡献未证明。S+V尚未组合核验。
> [S-align / V-noSAM2](docs/ALIGN_NOSAM2_EXPERIMENT_20260917.md) 已回分：S-align `0.8407/0.8620/85.135`，接受正确对齐作为研究起点，原S保留最高分回退；V-noSAM2 `0.8362/0.8181/82.715`，比V低1.930分，保留64源图SAM2掩码监督。
> [训练流程与下一步](docs/TRAINING_ROADMAP_20260917.md)：现有基线复用SSL→Stage1→joint-v3。2026-09-19用户回报[S-align+V统一组合](docs/UNIFIED_ALIGNMENT_AUDIT_20260919.md)黑盒`0.8416/0.8569/84.925`，作为后续统一研究对照；原S保留最高分回退。[Stage1最终任务联合适配→两头精修](docs/STAGE1_FINAL_TASKS_20260919.md)回分`0.8448/0.7699/80.735`，面积退化、总分低4.190，拒绝晋级。该整套替代失败不证明joint-v3不可替代；下一步拟做固定权重的两头输出交换定位，尚未运行；EMA尚未加入。
> 下列进度与默认部署文件保留历史 G4b 方案；本轮未切换默认推理入口。
> 当前状态、入口与产物约定以 [docs/PIPELINE.md](docs/PIPELINE.md) 为准。

颜色先验分析已记录在 [docs/COLOR_SEPARABILITY.md](docs/COLOR_SEPARABILITY.md)。后续对话进程在
讨论语义阈值、颜色辅助或数据增强前应先阅读该文档。

## 当前进度

- **语义锚点与参考基线**：`outputs/stage2_v6/best_model_stage2.pth`。其语义分支仍是当前
  可复现锚点，原边界分支作为回退基线。
- **当前几何基线**：G4b affinity + `high=0.65` + 封边/受阻分水岭。相较 G3，G4b 是当前
  更稳定的部署基线；测试集仍以欠分割和局部合并为主要风险，尚未证明黑盒竞赛分数提升。
  详见 [docs/AFFINITY_DEPLOYMENT_EVALUATION.md](docs/AFFINITY_DEPLOYMENT_EVALUATION.md)。
- **当前语义选择**：E10a 黑盒结果为 mIoU `0.8381`、铁素体平均面积项 `0.8408`、总分
  `83.94`，已取代 E9 成为单模型主线；E9 的 `0.8421/0.7917/81.69` 仅作稳定历史回退。
  详见 [docs/SEMANTIC_EXPERIMENT_E10A_20260828.md](docs/SEMANTIC_EXPERIMENT_E10A_20260828.md)。
- **统一部署模型**：`outputs/deployment/e10a_g4b_fused.pth` 把共享 SAM2+LoRA encoder、
  E10a semantic decoder 与 G4b affinity decoder 打包为一个 81.67M 参数模型；加载时不再依赖
  三份源 checkpoint，`test_009` 与原管线最终实例图/类别 JSON 字节级一致。
- **方向图切分结论**：graph-v1 `area200` 黑盒为 `0.8268/0.8365/83.17`，未超过 E10a
  watershed；graph-v2 `area150` 目检出现不自然的笔直边界，已停止晋级，不再提交。详见
  [docs/AFFINITY_GRAPH_AB_20260828.md](docs/AFFINITY_GRAPH_AB_20260828.md)。
- **中心热图实验已降级为负面对照**：共享 `boundary_fpn` 的中心辅助任务破坏了边界表征；
  `outputs/stage2_center_heatmap/best_model_stage2.pth` 不作为后续初始化主线。
- **G7 已停止**：固定协议测试 A/B 显示其相较 G4b 进一步减少实例，欠分割风险加重；不再作为
  当前晋级目标。配置与说明仅保留为历史对照。
- **当前训练猜想**：检验 `SSL LoRA → 随机 semantic/affinity 双头` 的短链路；先冻结 LoRA
  预热双头，再以小学习率联合训练，并可加入经人工审核的无类别 SAM2 geometry affinity 监督。
  详见 [docs/DIRECT_SSL_SEMANTIC_AFFINITY.md](docs/DIRECT_SSL_SEMANTIC_AFFINITY.md)。
- **审计结论（2026-08-27）**：G3 的验证提升尚不能等同于竞赛增益；G3 相比 G2 同时改变了
  native crop、采样、训练时长、学习率和外观增强，需在统一部署评估链上做单变量复验。详见
  [docs/AFFINITY_DEPLOYMENT_EVALUATION.md](docs/AFFINITY_DEPLOYMENT_EVALUATION.md)。

## 环境配置

### Python 环境

```bash
conda create -n sam2_env python=3.11 -y
conda activate sam2_env
pip install -r requirements.txt
```

GPU 版 PyTorch 请按 [pytorch.org](https://pytorch.org/) 的 CUDA 匹配命令安装（如 `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`）。

### SAM 2 本地权重

将 `sam2_hiera_base_plus.pt` 放入 `weights/` 目录，`segment-anything-2/` 为本地 SAM 2 源码仓库：

```
weights/
└── sam2_hiera_base_plus.pt
```

> **约束**：严禁依赖 `~/.cache/` 等全局隐式路径，所有第三方权重必须存放在项目 `weights/` 目录。

## 数据准备

| 目录 | 内容 | 用途 |
|------|------|------|
| `data/raw/` | 有标注图像 + 同名 Labelme `.json`（label 为 `ferrite` / `pearlite`） | Stage 1 / Stage 2 有标签流 |
| `data/purified_gt/` | 离线净化 GT（`.npz`：semantic + boundary） | 训练真值 |
| `data/unlabeled/` | 无标注图像（约 1000 张） | Stage 2 半监督无标签流 |
| `data/test/` | 测试图像（68 张） | 推理 / 训练监控 |
| `data/smoketest/` | 冒烟测试图像（10 张） | 快速验证 |

有标签数据准备（标注变更后需重新生成净化 GT）：

```bash
python tools/preprocess_labels.py            # 生成 data/purified_gt/*_gt.npz
python tools/preprocess_labels.py --visualize  # 可视化净化结果
```

净化流程：CLAHE 增强 → Canny 边缘检测 → 晶粒内部掩码腐蚀剪裁 → 边界带膨胀，得到纯净、无划痕的晶界真值（二值语义掩码 + 二值边界掩码）。

## 项目结构

```
segmentationv2/
├── config/
│   ├── default_config.yaml      # 当前 V6 参考基线
│   ├── inference/               # 同架构推理配置
│   └── experiments/             # 改架构/训练目标的实验配置
├── data/
│   ├── raw/                     # 有标注图像 + Labelme .json
│   ├── purified_gt/             # 离线净化 GT（_gt.npz）
│   ├── unlabeled/               # 无标注图像（半监督）
│   ├── test/  smoketest/        # 测试 / 冒烟测试图像
│   ├── dataset.py               # 在线数据管道（letterbox / 边界权重 / BoundaryDataset）
│   ├── dataset_semi.py          # 半监督双流数据集（Labeled / Unlabeled）
├── models/
│   ├── sam2_encoder.py          # 冻结 SAM 2 Hiera trunk（4 尺度特征）
│   └── fpn_decoder.py           # 独立双 FPN 解码头（seg_fpn + boundary_fpn）
├── utils/
│   ├── loss.py                  # BoundaryLoss（语义/实例核心 + 边界 Focal×EDT）
│   ├── loss_semi.py             # 半监督一致性损失 + EMA 更新 + 骨架过滤
│   ├── semantic_training.py     # E7b 实例等权核心损失 + target-aware 暗边增强
│   ├── metrics.py               # SegMetrics 评估（mIoU / mDice / Boundary IoU）
│   ├── post_process.py          # 骨架化 / 受阻分水岭 / 实例 ID
│   └── progressive_aug.py       # 渐进式外观增强（学生输入专用）
├── tools/
│   ├── preprocess_labels.py     # 离线边界净化 GT 生成
│   ├── precompute_pseudo_labels.py  # Stage-1 边界伪标签离线预计算（TTA + 质量报告）
│   └── tmp_color_separability.py # 临时 GT 颜色可分性分析脚本（未接入训练）
├── weights/                     # 本地权重（sam2_hiera_base_plus.pt）
├── segment-anything-2/          # SAM 2 源码（本地仓库）
├── train.py                     # Stage 1 训练入口
├── train_stage2.py              # Stage 2 半监督训练入口
├── inference.py                 # 推理入口（语义投票实例分类）
├── debug_pipeline.py            # 数据管线诊断（letterbox / 边界权重 / 受阻分水岭）
├── test_skeleton_watershed.py   # 骨架 + 分水岭纯图像验证
└── visualize_instances.py       # 实例图着色可视化
```

## 当前训练与基线

G4b high0.65 + V6 语义的黑盒结果为总分 `80.67`、实例 mIoU `0.8441`、铁素体平均面积项
`0.7693`。E9 进一步得到 `0.8421/0.7917/81.69`；E10a 最终得到
`0.8381/0.8408/83.94`，以显著面积收益成为当前黑盒最佳。E7b-A 已完成验证但不晋级：它从 V6
初始化只更新语义 decoder，训练期 semantic loss 虽下降；严格同部署口径下，代理分数由
当前 hard 基线的 `79.1800` 小幅降至 `78.8379`（阈值 `0.65`）。详见
[docs/SEMANTIC_TRAINING_E7B_20260827.md](docs/SEMANTIC_TRAINING_E7B_20260827.md)。

E7c 不丢弃 E7b 的局部纠错能力，而是让 V6 固定实例几何与默认类别，仅允许 E7b
通过门控修正低置信/黑边实例。缓存扫参已选择 E7c-R 进入目检：六图验证仅翻转 1 个且
新增一个正确铁素体匹配，68 张测试图翻转 67/6178 个实例；不设置实例面积门槛。详见
[docs/SEMANTIC_EXPERIMENT_E7C_20260828.md](docs/SEMANTIC_EXPERIMENT_E7C_20260828.md)。

E8 已证明低分辨率残差能够产生局部纠错，但 `+/-2` logit 很快饱和且验证 mIoU 未提升，故不
晋级。E9 保留 V6 零漂移锚点，把可学习决策提高到输入分辨率，并增加与实例投票一致的
整实例概率损失；epoch 20 probability-mean 已通过黑盒验证并保留为稳定回退。详见
[docs/SEMANTIC_EXPERIMENT_E9_20260828.md](docs/SEMANTIC_EXPERIMENT_E9_20260828.md)。

当前 E10a 冻结 V6 LoRA、边界和 G4b affinity，随机重置完整 semantic FPN/head 与高分辨率
解码路径；固定 V6 只在无标签高置信区域提供衰减蒸馏。E10a 已获黑盒总分 `83.94`，不与
E9 连续融合；标准几何仍固定为 G4b watershed。详见
[docs/SEMANTIC_EXPERIMENT_E10A_20260828.md](docs/SEMANTIC_EXPERIMENT_E10A_20260828.md)。

如需复现实验，按既定约定可直接在训练服务器运行：

```bash
conda activate sam2_env
python train_stage2.py --config config/train/stage2_semantic_e7b_decoder20.yaml
```

E9 当前训练命令：

```bash
python train_stage2.py --config config/train/stage2_semantic_e9_highres20.yaml
```

E10a 冷启动完整语义解码器：

```bash
python train_stage2.py --config config/train/stage2_semantic_e10a_cold20.yaml
```

SSL 直达 semantic/affinity 双头候选：

```bash
python train_direct_semantic_affinity.py \
  --config config/train/direct_ssl_semantic_affinity.yaml --check

python train_direct_semantic_affinity.py \
  --config config/train/direct_ssl_semantic_affinity.yaml
```

E9 输出目录为 `outputs/stage2_semantic_e9_highres20/`。训练固定 G4b geometry、V6 decoder
与 LoRA，仅更新零初始化 high-resolution semantic residual；checkpoint 按验证 semantic mIoU 选择，无标签 holdout monitor 只检查
泛化稳定性。历史 Stage 1、B2、affinity 训练命令见 [docs/PIPELINE.md](docs/PIPELINE.md) 和
`config/README.md`。

## 推理

```bash
conda activate sam2_env
python tools/run_affinity_submission.py \
  --config config/inference/final_affinity_g4b_high065.yaml
```

E7b 训练完成后的固定几何对照：

```bash
python tools/run_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_e7b.yaml
```

当前 E7c-R 类别纠错目检候选（实例几何仍与 G4b 完全一致）：

```bash
python tools/run_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_dual_e7c_relaxed.yaml
```

E9 训练完成后的整实例概率聚合：

```bash
python tools/run_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_e9_highres.yaml
```

E10a 训练完成后的固定 V6/G4b 几何 challenger：

```bash
python tools/run_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml
```

推荐的单主干部署方式：

```bash
# 只需在源 checkpoint 更新后重新生成一次组合包
python tools/build_fused_deployment_checkpoint.py

# 日常推理仅加载一个组合 checkpoint
python tools/run_fused_affinity_submission.py \
  --checkpoint outputs/deployment/e10a_g4b_fused.pth
```

组合包保存完整 encoder 权重及两个任务分支，约 327 MB；源 checkpoint 路径与 SHA256 仍写入
包内供审计，但不是推理依赖。禁止用参数平均代替分支拼接。

E7b 整体替换配置只替换 semantic decoder；E7c-R 则只允许 E7b 改实例类别。两者都会验证
E7b 与 V6 的 LoRA 张量逐字节一致；若 LoRA 发生变化，将拒绝复用 G4b affinity checkpoint。V6 双头历史推理仍可使用
`python inference.py --config config/inference/v6_reference.yaml`。

## 实例级分类器（已废弃）

> v4.0 实例分类器推理管线（对照实验曾证实优于语义投票）已在协议 C（LoRA）落地后废弃：
> 当前语义头在 LoRA 特征上已接近直接可用，实例类别由分水岭 + 语义投票产生。
> 历史实现见 git 标签 `v4.0-instance-clf`。

## 输出文件

推理后在输出目录生成：

- `{basename}_inst.png` : 单通道 uint16 实例图（1~65535，按面积降序编号）
- `{basename}_class.json` : `{"实例ID": 类别标签}` 映射（0=珠光体，1=铁素体）
- `{basename}_mask.png` : 语义掩码可视化（`post_process.save_visualization=true` 时）
- `{basename}_boundary.png` : 边界概率热力图

## 技术约束

| 约束 | 说明 |
|------|------|
| 参数量 < 500M | 冻结 encoder 约 80M + 解码头约 10.8M，满足约束 |
| 零预训练解码器 | FPN 全随机初始化，禁止加载 SAM 2 原生 Mask Decoder 权重 |
| 本地化隔离 | 权重存放于 `weights/`，禁止全局缓存 |
| 长宽比保真 | Letterbox 等比缩放 + BORDER_REFLECT 镜像填充，禁止挤压变形 |

## 类别定义

| 通道 | ID | 名称 | 说明 |
|------|----|------|------|
| 语义 | 0 | pearlite | 珠光体 |
| 语义 | 1 | ferrite | 铁素体 |
| 边界 | 1 | grain_boundary | 晶界（独立通道二值预测） |
