# 同源32图几何监督来源对照（2026-10-04）

## 问题与实验边界

本轮检验：在相同训练源、输入、可监督区域和优化预算下，用当前已评分部署产生的
实例划分替换 processed GT 的几何关系，是否能改善 affinity 后端的最终输出。
这是**同源自蒸馏**：模型重新学习自己的几何结果，不能把教师结果称为正确GT或新增独立事实。
本轮不修改旧GT，不训练语义头，不训练恢复器或编码器，也不自动替换最佳部署。

| 项目 | A：manual | B：teacher |
| --- | --- | --- |
| 人工源图 | 相同32张已处理GT训练源 | 相同32张训练源，全部重新推理生成候选 |
| 几何答案 | 原 processed GT 的实例关系 | 固定 R1＋旧D5a＋reflect32 的实例关系 |
| 共同支持 | `processed_manual > 0 & teacher > 0` | 完全相同 |
| 教师零缝、原GT未知区 | ignore | ignore |
| 既有SAM2几何源 | 原64源、原manifest，权重0.5 | 同样64源、同样权重0.5 |
| 语义监督 | 无新增语义GT | 教师类别JSON仅作推理诊断，不进入训练 |
| 初始化与输入 | 严格载入同一完整 R1 e20；相同视野、增强、恢复与源顺序 | 完全相同 |

`processed known` 包含原人工覆盖与已批准的补缝区 `filled`，共同支持不是只取
`original_covered`。两类区域仍分别统计，但完整几何损失沿用当前等权合同。
教师零缝造成的像素／关系覆盖损失必须记录，不能把剩余区域收益归因于全域监督改善。
原生薄零缝可能在 nearest 缩小时消失，因此原生像素损失与实际512监督网格边损失分开报告。

仅使用允许的训练数据；测试图不参加训练，不制作、推断或修改测试GT。
32张人工源和原64张SAM2源的唯一源图联合数量按图像SHA核对，不凭训练次数计数。
新的硬实例候选保存在独立目录，不并入旧SAM2已确认manifest。

## 固定身份与训练预算

| 固定项 | 身份或设置 |
| --- | --- |
| R1完整后端 | `outputs/affinity_balance/mixed/final.pt`，e20 |
| R1 SHA256 | `43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434` |
| 旧D5a SHA256 | `048954795dc634f82fecf96d5e02bb2f2505afe5eb76298dff5a1b0713b3c80a` |
| 初始化方式 | 从同一完整R1继续；Stage1、joint-v3及既有LoRA状态均继承，未省略既有阶段 |
| 唯一训练模块 | affinity decoder；冻结D5a、encoder／LoRA和语义头 |
| 每臂预算 | 5 epoch × 64更新 = 320更新 |
| 共同采样 | 每轮32次whole与32次native1024；两个源流在同一更新使用相同视野 |
| 每次更新 | 一次manual源及一次既有SAM2源；两臂实际输入、crop、退化与合法边回执逐项比较 |
| 优化器 | AdamW，初始学习率 `2e-5`，`eps=1e-4`；沿用原weight decay与梯度裁剪 |
| 学习率日程 | 原20轮日程，第5轮停止，不另压缩为5轮降到最低学习率 |
| 原几何损失 | 原八通道关系，negative weight 1、hard negative weight 1、gamma 2；两臂相同 |
| 模块模式 | affinity `.train()`；真实头使用GroupNorm，无BN／dropout；其余模块 `.eval()` |
| 过程monitor | e0、e1、e5固定缩略图；仅观察，不选择测试效果最好的epoch |

裁窗先由共同覆盖决定，再分别取两份实例答案；不能让教师实例形状改变裁窗抽样。
正式阶段固定比较e5。5轮只是预算有限的阶段检查，不预设收敛或允许自动延长。
权重保存包含优化器、调度器、随机状态、完整输入合同和实际320步回执，支持严格续训核验。

## 三个程序入口

配置：`config/train/affinity_gt_source.yaml`，仍使用现有 `_base` 继承。
以下命令在GPU服务器的项目根目录、`sam2_env`环境执行；不是本地CPU训练命令。

```bash
# 1. 固定原部署在32张训练源生成教师几何与配对图库
python tools/run_affinity_gt_source.py --config config/train/affinity_gt_source.yaml

# 2. A臂：原processed GT；须先有完整且核验通过的32源manifest
python train_affinity_gt_source.py --config config/train/affinity_gt_source.yaml \
  --arm manual --output outputs/gt_source/manual

# B臂仅在用户完成本次32候选质量确认后执行
python train_affinity_gt_source.py --config config/train/affinity_gt_source.yaml \
  --arm teacher --output outputs/gt_source/teacher \
  --paired-steps outputs/gt_source/manual/steps.jsonl

# 3. 每臂训练完成后，固定e5完整100图推理及无GT比较
python tools/evaluate_affinity_gt_source.py --config config/train/affinity_gt_source.yaml \
  --arm manual --checkpoint outputs/gt_source/manual/final.pt
python tools/evaluate_affinity_gt_source.py --config config/train/affinity_gt_source.yaml \
  --arm teacher --checkpoint outputs/gt_source/teacher/final.pt
```

生成器严格载入R1／D5a及已验证runtime，使用封存的原生1024 views：先捕获实际CUDA
输出，再核对原marker、真实watershed elevation和原语义rawvote，最后只采用固定reflect32。
封存源码、模型、图像、processed GT和原SAM2 manifest身份，末尾复查未变。
额外保留的冻结关系缓存必须与历史窗索引集合、实际boundary与监督audit精确一致；
它供另一个保护诊断使用，不偷偷加入本轮A/B损失。

## B臂质量确认

生成候选不是人工确认。先查看全部32张原图／processed GT边缘／教师边缘对比及局部图，
再由用户明确确认本次候选是否允许作为训练几何监督。2026-10-04用户已明确确认本批32候选，
独立确认回执已写入服务器；候选仍不是独立真值，原GT保持不变。

`targets/manifest.json`保持不可变，生成器中的 `quality_confirmation`不会被改写来冒充确认。
真正确认后使用独立的 `outputs/gt_source/quality_review.json`，训练器要求：

| 字段 | 约束 |
| --- | --- |
| `status` | 只有用户明确完成质量确认后才可记录 `explicit_user_confirmed` |
| `manifest_sha256` | 必须绑定本次完整、不可变manifest的实际SHA256 |
| `review_checked_target_count` | 必须为32 |

无文件、状态pending、数量不足或manifest SHA不符均拒绝B臂。不得为通过程序自行生成确认。
A臂使用的答案仍是原已处理GT，仅借教师正ID建立两臂共同ignore域，完成32源身份核验后
可以独立开始；这不代表B臂候选已经获得确认，也不需要重复训练相同A控制。

## 完整部署评估合同

每臂e5权重只在全量100图上forward一次，随后一次读取已评分R1
`outputs/marker_trial/reflect1024.zip`比较，不重新推理R1。测试图按原完整排序，
seed固定为 `314159 + full100_index`，不得重新编号或按图效果调整seed。

| 项目 | 固定与核验 |
| --- | --- |
| 推理链 | 旧D5a首步、原生1024／overlap 0.25、原融合、连续boundary拼接、reflect32 marker、实际WS elevation、原rawvote |
| 权重 | 完整fused候选必须e5／320更新，arm与完整contract一致；严格加载及冻结源digest核验 |
| 上游输出 | 每图semantic及raw probability SHA与R1报告完全相同，防止恢复／语义输入悄悄改变 |
| 参考产物 | R1 ZIP必须恰好100对uint16实例PNG／类别JSON，逐文件SHA与已评分报告一致 |
| 最终输出 | 原图尺寸、单通道uint16、最大ID≤65535、类别JSON与实际正ID一一对应 |
| 运行环境 | 实际FP32、legacy_none；精度、runtime、源码和输入前后未变 |
| 对比诊断 | 相别像素变化、共同正域的编号无关成对划分变化、零覆盖变化、F实例／面积、双向显著split／merge |
| 图库 | 固定完整排序前6图；原图、边缘与相别叠加，附中心局部；不含测试GT |

“成对划分变化”检查两像素是否仍属于同一实例，换实例编号不会产生差异。
split／merge仅相对R1教师预测描述拓扑变化，**不是正确率**。平均面积变化也不是官方面积得分，
不能用这些数量反向拟合测试答案。

私有产物位置：

```text
outputs/gt_source/
  targets/manifest.json           # 32源、配对覆盖、关系冲突与身份
  targets/teacher/                # uint16教师PNG，类别JSON仅诊断
  targets/pairs/                  # 共同支持上的manual／teacher几何
  targets/gallery/                # 全32图确认图库
  quality_review.json            # 用户明确确认后才存在的独立回执
  manual/ 或 teacher/
    status.json, steps.jsonl, last_head.pt, final.pt
    monitor/e00、e01、e05/          # 固定过程观察
    deployment/
      patch1024/                 # 恰好200个最终预测文件
      gallery6/
      report.json, status.json, source/
```

当前代码不创建黑盒压缩包、不上传平台、不自动晋级最佳部署。
所有赛题数据、预测、图库、回执、权重及代码快照留在被忽略的私有目录，不进入Git。
可公开本说明及经核验的汇总文字；不修改共享README或项目全局规则。

## 已恢复执行与启动核验（2026-10-04）

新会话的主执行器和SSH均恢复正常，已核验原后台任务完整结果，没有重复生成候选。
以下是本次实测快照，**不代表尚未结束的训练或推理已经完成**：

| 分析／执行方案 | 已核实结果 | 结论与限制 |
| --- | --- | --- |
| 历史目录逐项归档、成员与内容SHA校验后清理 | 11／11组完成；快盘空闲3.36→33.59 GiB，随后因下载归档约为33.5 GiB | 回收约30.23 GiB；当前活动项目、环境、依赖代码与权重保留，运行不从慢盘读取 |
| 固定R1部署生成同源候选 | 32／32完成；私有review ZIP及167成员SHA全验通过 | 不改原GT；候选只用于本次已获确认的几何来源对照 |
| 全32原生覆盖与512格关系比较 | 共同原生覆盖95.41%；教师零ID使两臂共同忽略1.32%的processed-known像素 | 相同ignore域确保A/B只改答案，不把教师边界缝当负样本 |
| 原八通道同／异实例关系分层 | 整体冲突3.89%，原覆盖双方1.61%，涉及filled的关系29.10% | 差异集中在补全区域；不能据此判定教师更准确 |
| 冻结实际缓存与原监督保护交集 | 另一路外围保护诊断的1886条空间候选中，1583条通过原固定band、教师置信和known路径条件 | 计数含重叠窗口重复；只证明mask命中，不证明梯度有效、拓扑修复或黑盒增益；不加入本轮A/B损失 |
| CPU合同检查与CUDA启动 | 90项CPU检查通过；4张e0完整PNG／JSON字节复现；实际首步logits复制一致、梯度有限非零 | A已开始优化；仅affinity训练，总参数84,951,045；不能将启动检查写成训练完成 |

完整32候选manifest SHA256为
`1796105aed0911f78d5f3b0752a645483482120e4a33bccfcda84016052ba52d`。
用户确认回执绑定该SHA和32数量；manifest保留生成时的pending字段，没有反向改写历史。

训练器SHA256为`6ac0305a5fd28ca293a89b711a7ae41b722f5834314757716ebe8c7ca2f4838e`，
评估器为`44946b861e92205a7a3666e4d3bd702d4f570a0750d8b15e1b8b59c991f93989`，
配方文件为`7f5b56c71f6c177db91acce9af88083a1317cc75bb4520d642bfb8df258eb857`。
具体输入、源码、冻结状态与首步回执保留在私有`outputs/gt_source/startup_receipt.json`，
本地镜像在`output/gt_source/`。

本次原队列已在15930服务器执行，PID启动快照为`51449`，原顺序是A训练→B训练→A全100图→B全100图。
队列在任一阶段失败时停止后续阶段，不自动重复控制组、延训、打包、提交或晋级。
两臂5轮均已结束，A100图完成；用户指出短跑全量推理过早后，B在已完成75图时主动停止，
保留所有已输出文件。`queue/status.json`的failed是子进程主动SIGTERM带来的机械退出状态，
训练没有失败；停止原因及原状态另存私有`short_run_stop.json`。
未来短跑队列已修正为默认只训练，只有显式`--evaluate`才追加全100图；当前目录已有结果，
**不要再次执行fresh队列**。入口如下：

```bash
python tools/run_affinity_gt_source_queue.py --config config/train/affinity_gt_source.yaml
# 仅决定做完整部署评估时，在新的fresh队列显式选择：--evaluate
```

私有`outputs/gt_source/queue/status.json`记录阶段，四个阶段日志分别存于同目录。
e0／e1／e5过程缩略图已保存。尚无本轮黑盒成绩或提交包。

## 五轮阶段复核与下一步（2026-10-04）

两臂均e5／320更新，frozen、D5a、strict完整重载通过；B对A的320步实际输入、裁窗、增强、
恢复图和合法边逐项匹配。真实affinity decoder没有BN或dropout，旧状态记录中的泛化措辞不准确；
历史回执保持原样，不改写来掩盖。固定热图使用0–1的INFERNO色阶，随后缩略，没有逐图min/max归一。
只看缩略图亮度不能区分概率下降和亮边宽度／面积变小。

| 分析方案 | 实测结果 | 结论／限制 |
| --- | --- | --- |
| 相同已有SAM2答案及输入上的训练日志配对 | 64／64源B连接识别升、断开识别降；e5连接recall A90.221%／B92.428%，断开specificity A95.571%／B94.497% | 存在系统性连接偏移；SAM2仍候选几何，这不是独立准确率 |
| 原八通道loss和训练模式审查 | 正／负项分别归一；A/B模式与显示相同；没有实际BN／dropout | 不支持简单的“负样本被数量淹没”或BN模式归因；答案变化会改变每边归一权重 |
| teacher最终实例ID与连续概率的区别 | 同实例两端全部给same=1，未造成终态分开的原连续纹理也被推向连接 | 硬实例自蒸馏不等于保留旧概率；低loss包含自身答案更容易拟合的成分 |
| 首步尚未更新时人工源loss | A0.220956／B0.201467；同期SAM2 loss两臂相同0.287347 | B低loss本来就存在，不能据此证明新增监督更正确 |
| 保存结果的同批75图CPU比较 | 所有225份实例图原尺寸、16位灰度uint16与类别对应通过；R1 ZIP及300份候选文件SHA通过 | 复用原已完成输出，不再运行GPU全量；B75没有完整100图终态GPU回执 |

| 相同75源，仅预测分布 | 原reflect | A：manual e5 | B：teacher e5 |
| --- | ---: | ---: | ---: |
| 铁素体实例 | 4397 | 4480 | 4540 |
| 小铁素体＜200原生像素 | 57 | 59 | 73 |
| 相对reflect显著F拆分／合并 | — | 131／6 | 167／48 |
| 逐图F均面积相对reflect变化中位 | — | −1.304% | −1.469% |
| F均面积绝对变化＞5%的图 | — | 13 | 23 |

显著拆／并沿现有max(50px,来源实例面积10%)定义，只描述预测间交叉，不称正确或错误。
更暗的边界不保证最终实例减少；经过种子提取和watershed后，响应变化可能同时影响内部误切及邻粒隔离。
本轮五轮尚不能称收敛或饱和；目前先保留e5，不原样延长或仅凭热图占用黑盒名额。

另一路几何保护机制实验沿已有外界泄漏证据，只扩展GT一致高置信负关系的保留支持到processed-known。
先确认命中位置与实际泄漏足迹相交，再从相同R1起点做32更新的唯一mask变体，复用已有connect s32
作控制，不重复旧控制、不从旧s32再训练后与旧s32作等预算比较。
full／outer／local权重、原KL表达式、输入窗表、eval模式及优化器均保持。
检查新增保留的实际梯度和外围闭合，以及原七源689固定匹配／135固定面积对象；未匹配或资格失效不缩小分母。
通过后先做固定小样本完整划分，决定是否正式训练或全量提交检查；不自动追加100图。

私有分析：`output/gt_source/short_analysis/training_analysis.json`、
`output/gt_source/paired75/analysis.json`与固定101／116图库；均不进入Git。

### 固定四图原概率核验

为区分显示差异、细线面积与实际数值，只对原monitor的两训练图和两无标签测试图诊断，
没有新训练或全100推理。每图一次透明捕获原R1实际CUDA encoder特征，再重放R1、A、B三个头，
原窗口、融合、裁回和拼接不变；12份native float boundary SHA均与原monitor精确一致。
三个真实头各11个GroupNorm模块、BN／dropout均0；候选完整bundle非head state与R1逐值相同。
输入、上游、D5a与源码前后哈希一致，未新运行watershed。报告用时189.59秒。

| 固定源类别 | R1平均边界概率 | A e5 | B e5 |
| --- | ---: | ---: | ---: |
| 训练源1 | 0.180884 | 0.181693 | 0.164638 |
| 训练源2 | 0.174531 | 0.173641 | 0.155279 |
| 无标签测试源1 | 0.261657 | 0.268299 | 0.246661 |
| 无标签测试源2 | 0.217382 | 0.221317 | 0.203191 |

B比A均值低8.1%–10.6%，变暗确有数值依据。按同一原GT原覆盖mask分层，
训练源1深部同粒区域boundary≥0.65占比R1 3.378%→B2.734%，而异GT边带87.061%→76.611%；
源2对应2.550%→1.977%及69.604%→59.648%。因此内部多余响应被压低时，原GT分隔带也一起变弱。
局部深部用5×5同ID原覆盖区域，边带用原覆盖双端异ID邻接的原生radius2；
这些都是已有标签机制诊断，不保证物理边界正确，也未验证整段native路径或完整闭合。

辅助每通道ROC诊断将候选同粒recall对齐R1的0.5工作点后，原覆盖异粒specificity仍
99.528%→98.892%、98.461%→97.802%，不支持把问题全部解释成统一阈值平移。
只用FF同粒边重新对齐的FF异粒specificity为99.224%→99.059%、97.958%→97.581%，
下降较温和。FP没有同粒正样本，不能称FP独立匹配recall；它仅沿用全原覆盖校准阈值。
阈值仅用于已见GT诊断，不改部署或用测试标签选参。

工具`tools/probe_gt_source_boundary.py`及统计测试新增；25项统计／特征CPU检查通过。
私有报告`output/gt_source/boundary_probe/report.json`、共享0–1色标图库与384宽float32概率缩略图
已下载，122成员SHA验真。全native统计先计算再缩略，没有保存大feature缓存；测试没有GT。

### 教师细边界／零缝是否削弱监督

用户提出恢复伪GT边界信号的假说后，只读核验32份processed GT、pair NPZ、教师实例PNG和
历史类别lookup的SHA，全部匹配；重建32份whole512合法域／目标与manifest完全一致。
另按回执的翻转→旋转重建5个实际训练draw，合法域哈希及A/B正负数量精确复现。
没有修改GT、生成修复候选或启动训练。

affinity训练的是双端same／different，细线的显示宽度本身不决定监督力度。
teacher0两臂共同ignore，确实删去原processed GT短程断开关系18.603%、长程9.661%。
但320次实际人工源输入上，B总断开标签仍比A多3.392%；whole为+8.155%，native为−5.996%。
每通道正负项分别归一，计数变化不等于同比例的负项梯度变化。

为排除filled，进一步只看512网格双端均original-covered的短程关系：

| 原标注关系 | 原断开数 | teacher0共同忽略 | B改为连接 | B保留断开 |
| --- | ---: | ---: | ---: | ---: |
| FF：两端铁素体 | 119,632 | 22,314／18.65% | 43,461／36.33% | 53,857／45.02% |
| FP：铁素体／珠光体 | 200,813 | 37,823／18.83% | 56,839／28.30% | 106,151／52.86% |

FF中又有110,848条原同粒关系被教师改为断开，使B总负标签反而比共同域A多69.24%。
因此有真实的监督支持缺失，也有负监督位置重排；只补零缝不能修复已被教师合为同粒的关系。
这些是原有标签的端点关系，不是重新确认的物理边界，不能把全部差异都称为教师错误，
也不保证完整原生路径已知或分隔闭合。

GT来源方向的下一顺序：先作无需训练的零缝归属预检，仅在processed-known训练区域内
将教师零缝互斥地分配给相邻实例，保留原非零ID归属，未知和padding继续ignore。
重建同／异关系，记录原断开恢复、既有关系变化和小实例保留；不能以热图更亮为通过标准。
若支持恢复成立，再从原B相同R1起点、相同320步输入做唯一GT处理变体的5轮短跑，
复用已有B作为控制，不重跑A，不自动接100图。
若主要缺失仍来自teacher合并，另设可信原标断开保留或边界辅助监督实验，
明确教师连接目标与原标断开目标的冲突；不把它和零缝修补一次叠加。
已有外围保留1583条实际命中的另一实验保持独立，不能拿其32更新控制比较本轮320更新。

后续落实：[教师窄零缝支持恢复](AFFINITY_TEACHER_SEAMS_20261004.md)的全32预检已完成，
87.83%待补像素获就近归属，全部实例保留，恢复此前忽略原GT断开的71.91%。
独立同B起点／输入的5轮候选已完成，旧B直接复用；没有同时加入辅助边界项或自动100图。
末态e5／320更新配对、冻结及严格重载通过；固定四源分隔带未一致恢复、最终F划分近乎不变，
两monitor测试新增小F，两预随机非monitor也无一致收益。补零缝确实回拨部分训练响应，但不是充分条件。
按用户最新取舍搁置伪GT路线，不再追加断开冲突保留、扩训或黑盒；回到此前独立的原处理新GT
外围保护与粒内通路问题。独立[新增外围保护项短测](AFFINITY_RETENTION_GUARD_20261004.md)已落实，
最新门禁／训练状态见其专题，伪GT对照保持结项。

私有报告`output/gt_source/short_analysis/target_support_analysis.json`，SHA256
`2732ebf29e45c9e6c86745d1739cae6c23b3ee97d05c380b2351e864170dffbe`。

## 剩余黑盒安排与判定

用户最新限制：未来24小时仅5次提交机会。预留1～2次给纠错和复现，GT来源A/B至多占2次，
且只有在两臂完整e5／100图身份、冻结和配对检查通过，并观察到值得验证的完整输出差异后
才考虑提交。不得在仅有loss下降、单图改善或纯编号变化时自动占用名额。

另一个几何保护方向先检验**实际训练mask是否命中待纠正／待保留关系**，通过后只做有界短测；
不能以“保护项存在”代替真实监督覆盖与梯度作用证据，也不与本轮GT来源变量叠加。

| 分析方案 | 能得到的结论 | 不能得到的结论 |
| --- | --- | --- |
| 原生与网格覆盖／八通道关系冲突 | 两臂是否公平、教师主要分开还是合并现有GT | 哪份GT物理上更正确 |
| 同源e5、相同输入／曝光的A/B | 本轮几何监督来源造成的实际输出差异 | 对独立样本的准确率、长期收敛或“GT已修好” |
| 冻结后端完整100图＋R1比较 | 稳定部署中的分布／拓扑变化，是否值得耗用黑盒 | 只凭F数量／平均面积预言官方分数 |
| 用户授权后的官方黑盒 | 当前赛题口径的实际得分变化 | 小幅变化自动证明某机制或替换最佳主线 |

CPU测试结果以独立测试代理最终回执为准；CPU合同通过不等于本轮GPU训练已启动、已完成
或已获得竞赛增益。
