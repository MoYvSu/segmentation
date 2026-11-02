# 原图全图语义兼供分水岭与实例投票

2026-10-31。用户授权反向测试：删除全图D5a恢复语义路径，原图全图语义只计算一次，兼供分水岭及末端实例票。固定12图GPU、四源CPU严格桥接和本地验真已完成；无训练、无新150包、无候选官方分。

## 方案与固定项

控制C为已回分`.7462/.9070`的全零150图包`d484e9e1…aee30a5`，本轮沿用其中同一12图的既有控制结果。

```text
原图整图 → 等比letterbox1024 → encoder/语义头一次 → 映回原尺寸
                                              ├→ WS语义底图
                                              └→ 原图概率均值投票
局部原生1024 → 零噪D5a → affinity/N1 → T8/旧20a59 NAF → S0/分水岭
→ 原图语义票 → final200一次
```

新增`tools/infer_raw_shared.py`，真正跳过全图恢复、全图恢复encoder/语义及未消费的全图affinity。局部零噪D5a、affinity融合/拼接、NAF、S0、reflect32、.65/.45、seed50/pre_vote0/final200全部保留。唯一预测变化是WS底图的语义来源由恢复全图改为原图；末端原图概率逐SHA保持C，不是重新训练或调分类阈值。

实际计数对应12图、60局部窗：

| 网络调用 | 原C | 本候选 |
|---|---:|---:|
| encoder | 84 | 72：原图全图12＋局部60 |
| 语义头 | 24 | 12：原图全图每图一次 |
| D5a | 72 | 60：全图0、局部60 |
| affinity头 | 72 | 60：仅局部 |
| 全图恢复model调用 | 12 | 0 |
| final200 | 12 | 12 |

当前仍有原图全图语义和局部恢复边界两种视野，不等同此前的统一局部双头候选。全图D5a与原图语义均值投票是不同计算，本次删前者并共享后者。

## 样本与冻结验收

沿用169、172、174、207、211、226、237、258、277、280、302、313，三尺寸各4张，共30,262,464像素。没有按新输出换样本、制作测试GT或把测试图用于训练。

所有12图原图文件/原尺寸、局部输入和D5a恢复、整图N1、NAF概率、S0实际种子、末端原图概率及模型state与C exact。几何语义改为原图logit，故最终轮廓允许改变，并逐源记录。

四源169/226/302/313不新存浮点缓存：从C原有raw概率构造同阈值mask的±1辅助logit。它不是网络logit重建，也不冒充真实网络输出。当前WS几何只消费语义`.5`二值mask，额外语义边缘融合、中心种子和partition restore均关闭；临时类别随即被真实C raw概率票覆盖。

逐源硬门：C旧logit/原raw票先exact复现C；辅助mask SHA须匹配候选实际raw mask；CPU重放的实际WS输入图SHA须匹配候选GPU中真实`cv2.watershed`调用前图像；再要求最终ID/类别exact候选。两个分支均在final200前投票，各只final200一次。四源全部通过，没有用概率容差掩盖差异。

## 结果与解释

| 分析方法 | 结果 | 解释边界 |
|---|---|---|
| 全12 F/P像素翻转 | 13547/30262464＝0.04476503%；图等权0.05567433%；共同前景0.04538118% | 预测差异，不是错误率 |
| 方向与背景分开 | F→P 7276、P→F 6271；前景→0为46799、0→前景46821 | 不把缝隙移动混为改相 |
| 全部实例组成 | 1221→1222；F814→815，P407→407；F<500均22 | 总计数接近不等于逐实例不变 |
| 双向覆盖≥90%稳定对应 | 1198对，无整区F/P翻类 | 不能扩大成全部对象零改类 |
| 稳定名单外 | 旧23/新24，旧14个主覆盖归较大对象 | 全部去向单列，不能因稳定集合零翻类忽略接管 |
| 名单外改归异相 | 3个旧小区，面积203/204/230像素 | 其中两个主覆盖≥90%被异相较大区域接管；是预测变化，不判真值对错 |
| 最终边界 | XOR142551像素，占边界并集13.17145%、全图0.471049%；12/12实例图均非exactC | 不是数值等价的代码精简 |

空间重分配诊断记录4个旧区拆分、10个新区含多源重叠，其阈值为交集至少32像素且至少相应来源5%；可能包含邻界重分配，不能当真实晶粒拆并错误数量或用其净值预测得分。

固定中心和事后最大变化窗口显示，本组主要是边缘带及小区变化，未见上一组恢复语义投票的大片整区F→P现象。此前恢复全图兼投票的同12图F/P翻转0.82629%、55F→P/2P→F；统一局部双头为0.60494%。本候选更接近C，但变化小不证明准确率更高，不能把候选主线晋级或官方分数预先写实。

| 同12源计时，秒 | C | 本候选 |
|---|---:|---:|
| predict合计 | 98.7157 | 97.5101 |
| predict均值/图 | 8.2263 | 8.1258 |
| NAF均值/图 | 0.9588 | 0.9592 |
| 种子/WS读出均值/图 | 4.8248 | 4.8750 |

predict包含前向、NAF、读出与投票/清理，排除保存打包。单次顺序运行约快1.22%，不足证明稳定提速；C保存过4份capture、本轮0，不用wall差异夸大前向收益。保留已评分全零C，候选尚无官方回分。

## 当前语义头是否接受过D5a特征训练

**限定当前实际采用的语义头：没有针对D5a恢复图经编码器后的特征再训练。**D5a输出的是恢复RGB，语义头接收的是后续encoder特征，不是D5a内部中间特征。

来源链：当前导出backend文件`13b3ab1b…` ← R1 mixed-balance epoch20 `43da103c…` ← S-align e13（checkpoint epoch字段12）`ea905a94…`语义头。R1只优化affinity，encoder/LoRA/语义及其统计buffer保持eval和冻结，实际冻结回执通过。继承配置中出现D5a或semantic学习率，不表示实际优化器训练了语义头。

证据：

- `output/delivery_training/source_manifest.json:6`：导出与R1绑定。
- `config/experiments/affinity_semantic_aligned_v_top2_20260919.yaml:5`、`docs/UNIFIED_ALIGNMENT_AUDIT_20260919.md:46`：S-align来源。
- `models/backend_adaptation.py:53`：装配S-align语义头；`train_affinity_connectivity.py:33`及`train_affinity_native.py:134`：仅affinity可训练。
- `output/balance_analysis/training.json:97`：R1实际冻结/严格重载通过。
- `output/20260919_unified_alignment/audit_sources/S_align/train_stage2.py:818`及配套dataset：原图/在线增强直接进入模型，无D5a恢复；人工监督是整图缩至1024后裁512，无标签一致性用全图1024。

**项目历史上训练过D5a语义适配，但没有采用为当前权重。**`semantic_d5a/full`与`simple`各完成60轮/3840更新，输入为退化图→D5a→冻结encoder→训练语义头；结果和未采用结论见`docs/SEMANTIC_D5A_ANALYSIS_20260925.md`。不能把“当前头没适配”和“整个项目没做过实验”混用。

本次查证依据是已保存来源、实际源码和冻结回执，没有重新逐tensor比对当前头与原S-align或重做训练复现。未适配是当前输入选择的合理动机，不单独证明之前所有恢复图预测都错误，也不证明GT是唯一根因。

## 身份、资源与产物

- 基点`5de4f37e2715505da989378b399fc2b9713a61e3`，当前`codex/global-zero`；已有改动保留，独立新增入口与测试，未提交推送。
- backend `model/backend.pt`：R1 e20，文件SHA `13b3ab1bc509fa45848e2eb8a40ee458b178fe747fc83eb65333c527a8161f04`；D5a `model/restorer.pt`：e60，SHA `9315da872ead66735f799e27f85eb97f8c2c265d44d64c6f8a271be50417007b`；NAF `runs/naf_width/train/candidate/latest.pt`：step1000，state `20a59c3971b2112c48a3b744c31f8051f5687772456cd35b5d38f0ae41b158e8`。总参数102,583,318，未更新。
- 入口SHA `274a1849a4018d900e4d5c660f9093d25c41f19607dda5fd04b70f49b97ea280`；配置SHA `7d97829d130c5335716454b8fa8681011c52a6ca7d47b8473a0e4fb419019db6`；源码/worker seal `fd64ea5ad074632ccac345a2d755f5896d57a3606af4975d69deff12fdcd5036`。
- 分析器SHA `1778e75de67d541794597989d4f5f5e68b1a9572987255ce9f5e7f803cf520db`；最新150 manifest `14dd7572…a3c9`，固定selection `59588df6…adad`。
- 新11项及相关CPU合同共95项通过；根进程独立执行新/zero/NAF集合85项和真实CPU微型重放脚本，编译检查通过。
- GPU worker/guard exit0，worker约103.15秒、RSS峰3.01GiB、容器峰8.96GiB；CPU分析/guard exit0，约72.56秒、RSS峰1.73GiB、容器峰7.86GiB。RSS8/2.5、容器16、CPU AS8和快盘保留10GiB守卫不变。
- 服务器`/root/autodl-tmp/seg/supervised/runs/raw_shared/`；本地`output/raw_shared/received/analysis/index.html`为图册，`summary.json`及逐源`report.json`包含稳定/非稳定对应和4份精确桥接，`candidate/audit/raw_shared_report.json`记录实际调用次数。
- 回执/预览包87成员（86内容＋manifest）、3,989,905字节，SHA `786dcdcecea465b3c0450d870388f2b385098d30730f53a815602e48b7bda3f8`，逐成员SHA与12对PNG/JSON原尺寸uint16/ID类别契约验真；1222实例，最大ID306。没有下载模型或浮点缓存。
- 收尾生产seal和控制引用全部不变，GPU无计算任务，快盘余10.061GiB、容器约6.44GiB。产物均被Git忽略，没有GT整理、训练、全150新推理或正式候选提交包。

入口参数与全零入口兼容：`python tools/infer_raw_shared.py --input-dir <最新150目录> --output-dir <新目录> --config <N1/final200配置> --model-dir <原模型目录> --naf-checkpoint <旧20a59文件> --dataset semifinal_150 --global-noise zero --only <固定图名>`。
