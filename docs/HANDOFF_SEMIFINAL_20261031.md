# 半决赛研究交接：双输入NAF已分析，下一候选为768视野

> 最新反向对照：[原图全图语义双用](RAW_SHARED_SEMANTIC_20261031.md)同12图完成，删除全图D5a，原图语义一次兼供WS与末端票；局部N1/NAF/S0/raw票exact，F/P翻转0.044765%、F814→815，非稳定小区另列，未取得新官方分。当前头沿S-align/R1冻结，未针对D5a恢复输入再训练；历史full/simple适配未采用。仍保留已评分全零C，GT/训练未启动。

> 新对照：[恢复图全图语义兼供末端票](RESTORED_GLOBAL_VOTE_20261031.md)已完成12图，真正省raw语义前向，全部实例PNG与全零C字节一致；55F→P/2P→F、像素翻转0.82629%、F814→761。无候选新官方分、不改主线；原全零C仍为`.7462/.9070`。该候选保留全图恢复WS语义，不是统一局部滑窗或F0。

> 最新回分：全零150图包`d484e9e1…aee30a5`按用户去噪上下文及15666实例归属，回分`.7462/.9070`、TP/FP/FN10773/4893/2295；较旧C几乎不变，取消推理噪声得到评分支持。详见[全零包回分节](GLOBAL_ZERO_PACKAGE_20261031.md)。统一滑窗仍只做12图；F0仍无官方分，不能混用。

> 后续研究更新：[统一1024滑窗对照](UNIFIED_SLIDING1024_20261031.md)已完成12测试＋四源交叉，局部恢复/N1/NAF/S0 exact，语义统一后F/P翻转0.60494%、F814→807，末端分类来源是主要变化位置；无新GT/官方分，不判统一架构失败或晋级。旧全零150包保持，768仍暂停；GT整理与随机语义头原生1024训练尚未启动。以新报告和当前源码为准。

> 最新交付：[全零150图黑盒包](GLOBAL_ZERO_PACKAGE_20261031.md)已完成，私有`output/zero_full/submission/semifinal_zero150.zip`，SHA `d484e9e1…aee30a5`。12张复用＋138张新算、无TTA、全部零噪声；300文件本地/服务器独立核验通过。未自动提交，后续用户回分见上。以下早期交接不代表新的推理建议。

> **后续更正（2026-10-31）：用户已明确要求取消剩余全图噪声。当前后续推理入口为`tools/infer_zero.py`，全图与局部D5a均显式零噪声。最新150中固定12张的两遍独立推理已完成；旧C→全零最终F/P像素翻转率0.00393887%，全零重复翻转0%，详见[执行报告](GLOBAL_ZERO_20261031.md)。下文“全图仍保留固定噪声”等内容仅说明本交接时点的旧已评分控制，不再是当前推荐。768四源前向已在用户暂停前结束，收尾guard原失败保留，后续分析、扩量与GT替换训练均暂停。不得机械沿用下文下一步。**

交接日期：2026-10-31，北京时间。用户准备换会话；本次只整理交接，不启动实验。
新会话可以批判性复核下面的判断；已验证事实、当前建议和未执行计划分开，不必机械沿用下一方向。

## 新会话先看这里

1. 工作区：`D:\MY_PROGRAMME\segmentationv2`。先读本文件，再读`AGENTS.md`、README最新条目和下列两份终态报告。
2. **保留旧最高分控制：N1局部零噪声＋旧20a59 NAF＋S0＋原高阈值.65＋final200。**
3. 双输入NAF已完成1000更新并分析；补缺/内部抑制进步，保线和微环风险退步，未晋级。
4. 旧/双输入×高阈值.65/.75缓存检查已完成；.75不是当前推荐，不再密扫此阈值。
5. **下一候选是原生768视野；尚未实现、运行或出包。**用户刚确认其含义：原图局部窗口768，内部1024，NAF256不变。
6. 上一次实际资源核验：GPU无计算任务，容器约6.28／62GiB，快盘余约10.61GiB；新会话必须再读当前状态。
7. 本轮没有新黑盒包、没有消耗黑盒次数，也没有替换主线。黑盒由用户亲自提交。

优先阅读：

- [双输入NAF终态](NAF_DUAL_ANALYSIS_20261031.md)
- [高阈值2×2结论](NAF_THRESHOLD_20261031.md)
- [双输入实现与启动合同](NAF_DUAL_20261031.md)
- [窗口/方向与复票检查](NAF_CHECKS_20261031.md)
- [端到端研究清单](PIPELINE_RESEARCH_20261031.md)：是较早研究时点，里面部分“未执行”已被上列报告更新，不能当最新进度。
- `docs/REPORT_SOURCE_INDEX.md`是技术报告资料入口，用户另有材料整理工作，勿覆盖其改动。

## 当前checkout与未提交内容

本次核验：分支`main`，HEAD `5de4f37`（`chore: 同步半决赛实验代码与技术资料`）。
双输入模型/数据/训练等代码已存在于当前checkout；不要从旧会话片段重新实现一遍。

交接前已有未提交项：

- 修改：`.agents/skills/maintain-project-knowledge/references/observations.md`、`README.md`、
  `config/README.md`、`docs/PIPELINE.md`。
- 未跟踪：`docs/NAF_DUAL_ANALYSIS_20261031.md`、`docs/NAF_THRESHOLD_20261031.md`、
  `submmit_requirement/`；后者是用户材料，勿擅自删改/暂存。
- 本交接文件为新增；实际`git status`以新会话读取为准。

本次交接不commit/push，不建立新分支。需要新分支时遵循`codex/`前缀。
与本次交接一起新增的只有本文及README入口；上列是写本文前的状态快照。
赛题原图、标签、预测、逐ID类别JSON、热图、对比图、ZIP、权重等全部保密，
**不得提交Git/GitHub，包括私有仓库或改名/base64变体**。私有产物在忽略目录`output/`。
提交前按AGENTS检查暂存区及完整待推历史，不宽泛暂存、不用`add -f`或`--no-verify`。

## 比赛时间与目标

此前已核官方通知并存入`docs/SEMIFINAL_RULES_20261024.md`：

- 结果截止：11月1日20:00；材料截止：11月2日20:00。
- 每天5次，成绩留榜、取本阶段最高；不要假设未用次数结转，也不重复交成功控制包“保底”。
- 用户在本轮安排时说明剩两天、10次；本轮未提交，新会话核最新余额/回分，不能将旧余额当实时状态。
- 当前优先有监督主线；纯无监督仍未成熟，暂不重启该路线或混入其权重。
- DINOv3授权被拒，不再把获取DINO/重建大型分支列为眼前依赖。

评分细节见`COMPETITION_RULES.md`：实例项按同类一对一匹配、IoU>=.5、每图全部GT分母、图片等权；
未匹配FP不直接扣实例项，切碎/粘连/错类仍影响GT匹配；平均面积项只统计F。
**数量更少、precision更高、均面积接近旧值、训练loss更低、热图更顺均不是官方增益。**
不建本地代理黑盒，不强制实例数/均面积，不制作或修改测试GT。
基础视觉模型限制已开放，但LLM/VLM不得参与实际预测生成或修改结果；总参数暂按<500M等现行约束执行。

## 服务器、环境与资源规范

当前服务器：

```text
ssh <本地SSH服务器别名>
身份文件：使用本地SSH配置指定的密钥
训练Python：/root/miniconda3/envs/sam2_env/bin/python
本地Python：D:/Anaconda/python.exe
```

卡为vGPU 32GB，`nvidia-smi`显示RTX4080 SUPER。容器实际内存上限62GiB；
`/proc/meminfo`显示宿主机约503GiB，不能据此判预算。
活动工作区：`/root/autodl-tmp/seg/supervised/runs/<短名>/`。
FS `/root/autodl-fs`只作冷备，不能把训练/日志/checkpoint/过程图转到FS来绕过快盘不足。
FS免费20GB不是硬上限；不为省少量费用删除唯一资产。

本地无窗口SSH/SCP辅助器：`output/storage_cleanup_20261030/server.py`，支持：

```text
D:/Anaconda/python.exe output/storage_cleanup_20261030/server.py exec <本地Python脚本> [参数]
D:/Anaconda/python.exe output/storage_cleanup_20261030/server.py put <文件...> <服务器目录>
D:/Anaconda/python.exe output/storage_cleanup_20261030/server.py get <服务器文件> <本地文件>
```

该工具使用CREATE_NO_WINDOW，不开可见黑框/监听器。不要另起后台SSH循环轮询。
`exec`通过stdin执行，远端`__file__`为`<stdin>`；需要脚本SHA/守卫时先上传成真实文件再运行。
在PowerShell中保持原生文件操作，不拼跨shell删除命令。

最近清理：8份封存权重冷备并逐文件核SHA后移除快盘副本，释放约1.15GiB；
备份在`/root/autodl-fs/seg_archive/naf_checkpoints_20261031/`，回执在本地
`output/naf_dual/archive_receipt.json`。旧20a59、NoGAN及控制资产未改。

### 内存事故与必须保留的保护

曾有私有GT统计忘记将uint8的filled转bool，`gt[mask]`被当成行号索引，
1936×2584展开成1936×2584×2584，申请24GiB再转型申请48GiB，卡死62GiB容器。
用户已重启，权重均核验无损。**不要恢复该旧全图GT统计。**

- NPZ先查头/shape/dtype；选择mask显式`np.bool_`、二维同shape，索引前断言。
- 逐图/逐源处理，unknown严格忽略，不常驻多份全图数组。
- CPU检查可用AS8GiB、RSS2.5GiB/cgroup16GiB；OpenBLAS/OMP/MKL线程变量在导入NumPy前设1。
- GPU不能套CPU低地址限。训练反向与Adam映射明显大于仅前向；要完整预热后再设VMS增长限，
  并始终有外部RSS/cgroup守卫。
- 旧守卫`output/inference_checks/window/guard_run_v2.py`可复用，但退出RSS竞争仍可能误报。
  若guard failed而worker0/complete齐全，保留原失败，严格核终态后另写reconciliation，不能静默改成功。
- 任何保护触发先排查，不能直接解除限制重跑。快盘保留10GiB，若不足先核SHA冷备确实不用的资产。

## 当前部署链与关键身份

```text
全图旧D5a固定噪声 → 全图恢复语义（参与几何底图）
原图局部原生1024 → letterbox1024 → 旧D5a局部零噪声 → SAM2/LoRA affinity
→ 原图尺度融合N1边界 → 整图T8（阈值.5、骨架、ellipse8）
→ 旧NAF256滑窗概率拼接 → 原强弱阈值/reflect32种子＋原N1的S0支撑
→ 分水岭 → 原图全图1024语义概率按实例投票 → final200接管并继承类别
```

**N1只把局部恢复噪声置零，全图恢复仍保留原固定Philox噪声协议。**
不要把所有路径都改为零噪声，也不要替换为普通CUDA randn；现有跨卡输入噪声协议需保留。

固定后处理：high .65、low .45、weak growth8、bridge1、dilate1、marker_border_seal_width2、
reflect32、legacy seed50、raw probability_mean投票.5、final200 neighbor_merge_v1。
S0只移除原N1不支撑的种子，不换参考图；final200继承接管目标类，不复票、不迭代合并。

旧NAF不是NoGAN：它是T8单通道、PixelShuffle、原GAN .01，state为20a59。

| 资产 | 身份 |
| --- | --- |
| 后端backend文件SHA | `13b3ab1bc509fa45848e2eb8a40ee458b178fe747fc83eb65333c527a8161f04` |
| 旧D5a restorer文件SHA | `9315da872ead66735f799e27f85eb97f8c2c265d44d64c6f8a271be50417007b` |
| 旧NAF状态SHA | `20a59c3971b2112c48a3b744c31f8051f5687772456cd35b5d38f0ae41b158e8` |
| 旧NAF文件SHA | `f2b8f2dbbf445461f8e1b54b869d4a36551cf45dc90a62b13f8d8bc7c019ee8f` |
| 旧NAF checkpoint | `/root/autodl-tmp/seg/supervised/runs/naf_width/train/candidate/latest.pt` |

实际旧NAF loader：`tools/infer_boundary_rgb.py`，不是旧Region版`infer_boundary_naf.py`。
冻结生产代码/任务：`runs/naf_support_submit/{code,job.py,source.json}`；
部署准备资料在`runs/naf_support_deploy/`。需要model-dir/入口参数时读原job/source，不猜路径。
原冻结配置：`runs/naf_iter/frozen_config.json`，project_root指旧naf_support_submit/code。

### 官方参考及本地包

| 方案 | mIoU / 面积 | 说明 |
| --- | --- | --- |
| N1 | .7439 / .9032 | 局部零噪声，无NAF |
| 旧NAF＋S0＋final50 | .7467 / .9058 | 用户报官方总分82.62 |
| 旧NAF＋S0＋final200（控制C） | .7461 / .9070 | 按四位小数折算82.655；平台正式总分未单独给，不夸大约.03分差 |
| NoGAN＋S0＋final200 | .7479 / .8845 | 面积显著退，不能因图更好看就重新当未试候选 |
| 无条件P合并 | .7436 / .9070 | FP少1488但TP少121，已封存，不叠加 |

最高分控制包本地：`output/support200/submission/semifinal_support200.zip`。
本次重新核文件存在且SHA为
`cc346f8c4ab7f510fdc6be712baa7c45aa32f394672cda8adbe34266761567c8`。
旧final50包也保留，二者差异很小，不据此宣称统计显著突破。

## 刚完成的双输入与阈值实验

双输入含义：同一NAF接收`[T8, 原同源连续float32 N1]`。
公共G/D从同一随机初态重建，不继承旧训练权重；新增零初始化、无bias 3×3输入分支，增加432参数。
32份处理后新GT，1000更新、batch16、16000 draw与原GAN/loss固定，复用旧控制不重训。

实现：`models/boundary_dual.py`、`data/boundary_dual.py`、`tools/train_naf_dual.py`、
`tools/boundary_dual_monitor.py`、`config/train/naf_dual.yaml`。
旧raw_input_boundary字段其实是bool，新增连续通道另从封存NPZ float32读取。
新模型input_mode为boundary_continuous，checkpoint为naf_dual_training_v1，不能冒充旧格式。
训练与fixed monitor共享只读缓存；missing诊断仍以T8为准，避免误用第二路连续B。

候选：`runs/naf_dual/candidate/latest.pt`，step1000。

- G state：`776f4d0028507e8ccfc99fd48dfa1242520d882b9fe402628ae896b76538ec7c`。
- 文件SHA：`a7c5dec568d536ed484d8e0637de141fcd7d491dc8b67dfa10ba7fe0658142b7`。
- 77张过程图入口：`runs/naf_dual/candidate/monitor/index.html`。
- 130项CPU合同检查、真实GPU batch16零/关闭分支、169完整旧C输出桥接及80步学习性/严格续训通过。
- 正式训练完整；guard收尾误报已保留原回执，`runs/naf_dual/training_reconciliation.json`另行核对。

固定终态256裁块都是已见训练图，不是验证集：

| 原标指标 | 旧 → 双输入 |
| --- | --- |
| 自然T8缺口过.65，3206点 | 4.62%→12.38% |
| 记录内部负区过.65 | 2.082%→1.096%，32/32源下降 |
| 已有T8支持边界过.65 | 98.884%→97.908% |
| 输入完整626个GT骨架组件，保持r2全覆盖 | 626→616 |
| 输入不完整55组件，达到r2全覆盖 | 3→15 |

补缺与内部抑制确有进步，但保线有代价；r2覆盖不等于真正闭环/实例分开。
原标GT可能漏标，内部响应减少不能全称物理假边减少；unknown始终忽略。

六源同旧C读出：当前169/226/302/313，历史101/116。没有新黑盒分。
原始点阵图的小孔统计会失真，必须比较实际骨架障碍/marker带和最终实例。
当前四图实际marker带50–199孔64→442，pre200小实例43→130；不能称双输入普遍去环。

随后仅改global high .65→.75、复用两套.65缓存，结果：

| 当前4源 | 旧.65 | 旧.75 | 双输入.65 | 双输入.75 |
| --- | ---: | ---: | ---: | ---: |
| marker带50–199孔 | 64 | 69 | 442 | 325 |
| pre200小于200实例 | 43 | 48 | 130 | 75 |
| 最终F | 436 | 434 | 424 | 385 |
| 最终>=500像素F | 423 | 424 | 407 | 374 |

双输入.75 F数少9.2%，F像素仅少.78%，四图均面积升4.2%–17.0%；226/302可见分隔消失。
169主细通路保留，116登记断口四格都未切开。旧.75没有明确收益。
故保留旧.65，不为两套.75出包；不是证明任意其它阈值都无效。
新模型是否适配旧读出不能由幅值/小孔单指标决定。

本地私有证据：

- `output/naf_dual/analysis_audit/`：日志、配方、固定终态和结构审计。
- `output/naf_dual/analysis/analysis_paired/summary.json`：256固定配对。
- `output/naf_dual/analysis/analysis_readout/summary.json`：六源完整读出。
- `output/naf_dual/analysis/analysis_structure_retry/summary.json`：实际骨架/marker小孔。
- `output/naf_dual/threshold/`：四格JSON、全部对比图、解释与index.html。
- `output/inference_checks/`：更早的窗口/方向及复票诊断。

服务器对应资料全部在`runs/naf_dual/`，图库：`analysis.html`及`threshold/index.html`。

## 下一步：768视野的精确定义与实现注意

**尚未启动。**这是建议的下一单变量推理实验，不是新的训练任务。

现有源码路径已核：

```text
tools/affinity_native_views.py::predict_views
tile_boxes(size) → 原图局部裁块
→ tools/probe_affinity_patch.py::tensor_from_rgb（letterbox到1024，允许放大）
→ 旧D5a局部零噪声恢复 → boundary_grid（encoder/affinity）
→ crop_letterbox_output(...1024,...原生裁块shape)
→ BlendMap拼成原图尺度的N1边界
→ 原NAF/原读出
```

- 从原图裁768×768，适配成1024后送D5a和SAM2，方形完整块约放大4/3。
- 边缘不足方形的裁块仍等比letterbox＋reflect，不挤压。
- overlap比例.25固定，通常stride由768变576；窗口数/格点变化属于视野实验。
- 全图恢复语义与raw语义仍各自固定全图1024，不随局部视野改。
- NAF仍对拼回原图的N1整图做T8/kernel8，然后256窗、overlap64、reflect32；不把NAF改768。
- S0规则固定，但768组必须参考**自己未经NAF的N1_768**，不能借用N1_1024支撑缓存。
- 视野改变同时影响D5a及encoder所见纹理/模糊尺度；收益不能独归SAM2，不训练D5a或后端。
- 使用旧20a59，不叠加双输入、.75、P合并或其它候选。

### 现有入口不是改一个YAML就能跑

- `tools/infer_semifinal.py::validate_config`锁submission.native_sizes==[1024]。
- `predict`硬编码sizes=[1024]并取predictions['patch1024']。
- `tools/infer_boundary_rgb.py`会调用旧验证，并要求每源恰一次NAF解码/raw投票。
- 不把[1024,768]塞进原入口，不放松旧合同来假装兼容。采用独立研究入口/显式版本，
  768单独使用patch768和独立缓存身份。
- 先通过新入口的1024控制分支复现实际旧C最终PNG/类别，再试768。

建议实施顺序：

1. 读当前Git、资源与原job/source；确定独立短目录，建议`runs/scale768/`。
2. 实现最小单尺度入口并做CPU shape/配置/关闭路径检查，不动原部署默认。
3. 固定一张当前图做真实1024兼容核验；原NAF/S0/投票/final200最终输出必须一致。
4. 再做当前169/226/302/313等少量同图1024/768对照，覆盖复杂交错、细通路、完整界和小晶粒；
   历史116只能作补充，不只围绕这一张。
5. 保留各分支N1、NAF、最终轮廓和F分布；时间与内存均留回执，先不自动150图。
6. 若有值得验证的实质差异，再决定全量包。模型/最终读出/输入清单/ZIP SHA保持一一绑定。

## 其它已做检查与避免重复投入

- final200合并后复票：13源113个接管目标0翻类，暂不扩大，不为它出包。
- 旧NAF窗口：4源8ROI原概率exact；重叠窗有约4.1%的单窗强点被平均压低，
  但登记116断口最高单窗也未接通，不能把主要问题都归拼接。
- D4方向：主要是宽带内亮点/多轨换位，概率平均明显压幅，不能直接用旧阈值替换。
- 受扰热图训练NAF state695ca：1000已完成、固定配对无普遍收益，不直接续训。
- RGB、NoGAN、双线性上采样、相位一致性等有大量历史尝试；先查专题docs，勿重新包装成未试方案。
- 更复杂的关系合并/RGB头虽有训练排序进步，实际大范围F合并风险已验证，不叠加。

## 用户协作约定

用中文、简明解释术语和决策取舍。每项完成后给“分析方法→观察结果→结论”的概括表。
冻结已验证主线、单变量、复用控制，不重训相同控制。不凭少数图/训练loss晋级。
训练必须留固定缩略monitor，启动验收真实更新后不持续盯训练，用户会通知完成。
不默认跑150图；本轮只有小样本诊断。活动文件名/目录名保持短，不超过四个词。
没有必要自动创建新聊天、定时任务或后台SSH监听；本次仅交接，用户自行切换会话。
