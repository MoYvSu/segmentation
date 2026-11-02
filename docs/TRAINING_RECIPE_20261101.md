# 当前实际训练流程与参数：权重谱系及重训准备

状态更新：2026-11-01。默认统一入口已[删除V6与未使用的语义边界缓存](V6_ELISION_20261101.md)，
改为joint直接提供S-align固定教师；历史已评分权重的V6 e9来源不追溯改写。
教师32源监督链已跨两段run完成，SSL/D5a导入而未重训；随后150图用户回分`.7106/.8975`、
等权参考80.405，候选淘汰，见[训练终态](TRAINING_RESULT_TEACHER32_20261101.md)及[回分](TEACHER32_C150_INFERENCE_20261101.md)。
**原人工标注＋r8的默认13阶段配方尚无从头完整GPU实跑证据，教师演习不等于原C冷启动复现。**

核对日期：2026-11-01。本文整理已经执行过的训练，不启动新训练，也不把未来GT优化方案写成已完成事实。
当前已评分控制C采用 **13b3 backend + 9315 D5a + 20a59 NAF**；近期native-zero和随机语义头均为候选，尚未替换该控制。
全零C用户回报为`.7462/.9070`，来源归属及限制见[全零包报告](GLOBAL_ZERO_PACKAGE_20261031.md)。
F0是另一个推理读出候选，没有重新训练上述三份权重，不能把其成绩或状态混入训练记录。
仅D5a换TensorRT后用户回分`.7461/.9073`、等权参考82.670，仍用上述三份原权重。
全五引擎＋CPU整数读出是另一个[推理加速候选](FULL_ACCELERATION_20261101.md)，三源短测通过、150图已启动，尚未核完成或回分。

## 1. 如何使用这份记录

“初始化”指训练开始时从哪份权重接续；“冻结”指参数不参与更新；epoch是一轮预定抽样，update是一次优化器更新。
SHA256用于辨认具体文件或张量状态。**文件SHA和模型state SHA不是同一件事**：删除优化器、改导出格式会改变文件SHA，模型张量仍可能相同。

证据按以下顺序取用：实际完成回执/逐步日志与checkpoint元数据 → 当时配置快照/封存源码 → 当前恢复配置 → 专题说明。
当前yaml中有继承而未使用的字段，不能据它认定某模块实际训练过。本文分别注明“已核实际记录”和“恢复配方/待补”。
本轮没有重新加载全部历史大checkpoint做张量比较，也没有重新生成32张GT；历史审计结论保留其原有边界。

只读证据清单和本轮重加结果私存`output/training_recipe_20261101/evidence.json`。
另读取了当前服务器既有NAF的config/identity/complete小JSON，原样存`naf_remote_receipts.json`；没有模型前向、GPU查询或训练。
这些私有文件、原图、GT、缓存和权重均不进入Git。本文仅保存汇总配方、来源路径及身份摘要。

### 核心预算速览

下面是历史采用链的索引，不是要求重跑所有旁支；详细损失、冻结、精度和来源见后文。
标“恢复配方”的值尚无完整原CLI证明，不与已核实际步数混用。

| 阶段 | 主要训练项 | 数据/视野 | 预算与学习率 |
|---|---|---|---|
| LoRA SSL | LoRA＋临时RGB重建头 | 1000无标签图，1024 | 原meta30轮；原优化器15000更新、LR1e-4；恒定1000源/批量下推得batch2，文档batch8冲突 |
| Stage1 | LoRA＋语义/边界FPN | 人工32源划分，1024后裁512 | 原checkpoint配置100轮预算/best字段80、batch8、头LR1.5e-4 |
| joint-v3 | LoRA＋双FPN | 标注/无标签各batch4，1024后裁512 | 100轮/best字段88；主LR5e-5，模块比例另列 |
| V6 | 边界分支 | 固定语义边界缓存 | 60轮/历史采用e9；主LR5e-5 |
| S-align | 完整语义头 | r8人工25/7、人工1024后裁512；无标签1000图whole1024 | 20轮/1240更新，采用e13；LR1e-4 |
| V affinity | affinity decoder | 人工26/6＋SAM64，1024→512关系网格 | 120轮/3120计划更新，采用e115；FPN5e-5/新层1e-4 |
| D5a | RGB恢复器 | 全1000图，1024后裁512 | 60轮/15000更新、batch4；LR2e-4→2e-6 |
| R1 mixed balance | affinity decoder | 全32人工＋SAM64，whole/native混合 | 20轮/1280更新；LR2e-5→2e-6 |
| NAF width | 边界G＋训练用D | 32源真实N1＋r8目标，原生256、batch16 | 1000更新；G起始1e-3沿53000步日程，D恒1e-4 |

## 2. 当前三份实际权重

| 模块 | 已采用训练产物/导出 | 身份与实际作用 |
|---|---|---|
| backend | `outputs/affinity_balance/mixed/final.pt`，固定20轮；导出`model/backend.pt` | 原训练文件SHA `43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434`；导出文件SHA `13b3ab1bc509fa45848e2eb8a40ee458b178fe747fc83eb65333c527a8161f04`；导出state SHA `6ae2866ff8b76137dba8e3282f1f54de7a485e975105a453521ed7133906a3f4` |
| RGB恢复D5a | `outputs/d5a/candidate/epoch_060.pt`；导出`model/restorer.pt` | 原训练文件SHA `048954795dc634f82fecf96d5e02bb2f2505afe5eb76298dff5a1b0713b3c80a`；导出文件SHA `9315da872ead66735f799e27f85eb97f8c2c265d44d64c6f8a271be50417007b`；state SHA `e6ab2b2665b1dceb39d0fa6b1dbf9de621d63f9ec9cd97e876badcabdc64df89` |
| 边界恢复NAF | 当前服务器`supervised/runs/naf_width/train/candidate/latest.pt`，step1000 | **G模型state SHA** `20a59c3971b2112c48a3b744c31f8051f5687772456cd35b5d38f0ae41b158e8`；checkpoint文件SHA `f2b8f2dbbf445461f8e1b54b869d4a36551cf45dc90a62b13f8d8bc7c019ee8f`（245,200,199字节，含训练状态，已部署推理回执核对）。actual complete回执SHA `89a93ba9f38c243b056dd5e37f68abeeffaf676e2a57a3331856130779dc6a82` |

部署backend为81,667,394参数，D5a为3,283,651参数，NAF G为17,632,273参数，共**102,583,318**。
NAF训练判别器D不参加推理；同一D5a在不同输入路径复用，不重复计作多份模型。
backend结构为SAM2 Hiera B+编码器及LoRA、6,302,010参数语义头、八通道affinity头；无额外geometry adapter/refiner。
旧模型谱系元数据见`output/compliance_review/lineage/local_evidence.json`，后期部署验收见`output/zero_full/`及`output/naf_width/`。

## 3. 实际继承关系

```text
官方SAM2 Hiera B+（静态骨干）
  └─ LoRA SSL掩码RGB重建，30轮
      └─ Stage1-LoRA人工监督，100轮预算，best原字段80
          └─ Stage1训练图边界概率缓存
              └─ joint-v3，100轮，best原字段88
                  ├─ V几何：joint-v3边界FPN + 随机新上采样/输出层
                  │   └─ 120轮，人工26/6划分 + SAM64，采用e115
                  └─ 语义单线边界缓存 → V6（60轮，历史采用e9）
                      └─ S-align：完整语义头随机重置，20轮，loss-best e13

1000张赛方训练原图 → 独立随机初始化D5a，自监督恢复60轮

joint-v3编码器/LoRA + S-align语义 + V affinity + 冻结D5a
  └─ R1 mixed balance：只更新affinity，20轮/1280更新 → 导出13b3

冻结13b3/9315生成训练32源的真实局部零噪N1边界缓存
  + 32源r8已处理GT → 独立随机初始化NAF G/D，1000更新 → G state20a59
```

**control、native、mixed、R1不是顺序继承四个末态。**它们从同一组joint-v3/S-align/V初态独立开始。
现有训练器会读取旧control/native/mixed日志和参考预测，核对同源抽样、初始化和完整输出；这些属于复现审计依赖，不能写成累积权重继承。
V选择的是跳过G0/G1/G4b及V6边界预训的分支；V6仍通过固定语义教师进入S-align来源链，不能据“V skip V6”说整个最终模型没有V6依赖。

“N1”另有同名历史文档：`docs/N1_NATIVE_MULTISCALE_TRAINING.md`是8月V6原生多尺度候选。
当前部署N1是**仍执行D5a、局部显式零噪**的affinity边界输入，不等于该8月训练候选，也不等于省去D5a。

## 4. 数据与GT：三类监督不能混写

| 来源 | 实际规模与处理 | 在采用链中的用途/边界 |
|---|---|---|
| 赛方训练原图 | 1000张无标签训练图；历史按内容SHA核实来源，不能仅按文件名认定同源 | SSL、joint半监督、S-align固定教师分支、D5a在线退化重建；测试图不进入训练GT或梯度 |
| 原始人工标注 | 32张LabelMe及原图，类别P=0/F=1 | 早期Stage1/joint/V使用当时原始/净化目标；早期32图名称与1000图编号存在重命名，已核32/32内容匹配 |
| r8完成实例GT | `manual_target_v2`；LabelMe种子＋窄Lab梯度watershed；最大原生填补距离8px、blur sigma1.2 | S-align、R1人工监督和NAF目标来源；原标覆盖、filled与residual_unknown分开保存，unknown保持ignore。manifest SHA `5f4af78b746b22e7bffcd41fcb7d995ef9ad7f2a1f1591e3c2942926ba73d753` |
| SAM2无类别几何 | 固定64源`data/sam2_geometry_g2/manifest.jsonl`，SHA `4ed3e2e1fcfd2fa369dd647fc5ce7c869c5bcb7f472e29398ee889b368e40749` | 自动掩码经互斥处理，仅监督几何；`class_label:null`，不当作相区语义GT；未覆盖及covered-uncovered关系ignore |

历史内容审计确认SAM64均来自1000训练图，与人工32零内容重叠，因此是96个不同几何来源。
r8 manifest记录原覆盖像素改动0、补入10,029,722、剩余未知5,310,858；这是已保存manifest统计，本轮未全图重算。
`data/purified_gt`的早期语义/净化边界，不等于r8完整实例图；不能把二者路径互换。

SAM2自动几何候选生成曾使用官方mask生成能力；**最终任务FPN/语义/affinity decoder由项目自行构造，不能加载SAM2原生Mask Decoder作为这些任务头初始化**。
旧SAM64的历史`approval.json`未恢复，不能虚构审核人/时间；已有复用授权与原manifest必须保留。新生成或修订候选按[伪实例规范](sam2_pseudo_instance_dataset.md)单独留人工确认记录。
它与近期另行审核的32源teacher-GT候选不是同一数据版本；后者没有替换当前R1/NAF的目标来源。

NAF本轮实际目标manifest是`boundary_restore/assets/processed_gt32.json`（SHA `0b0c46558f31028c2951a06321dba78ffe19d2e3b7c8281ca3ca9fdf76442a8f`），
是同32源r8数组的训练封装，不应把它的封装SHA错当r8源manifest SHA。NAF目标取两个已知不同正ID接触处的双侧边界，不用相区类别造线；未知、图框、裁窗边缘及padding按valid规则排除。

## 5. 早期共享特征链：已知参数及尚缺原始凭据

### 5.1 官方SAM2与LoRA SSL

- 官方基础文件`weights/sam2_hiera_base_plus.pt` SHA `d0bb7f236400a49669ffdd1be617959a8b1d1065081789d7bbff88eded3a8071`。
  先前审计对正式backend的298个静态编码器状态逐张量比对，均与该官方底座相同；本轮读取该既有比较回执，没有重新比较全部大张量。
- SSL开放LoRA和临时RGB重建头，结束后丢弃重建头。原`meta.json`确认rank16、alpha32、48个qkv/proj注入层、1,019,648个LoRA参数、30轮。
- **2026-11-01补查Git早期历史和fs归档后，SSL的学习率及优化器参数已由原checkpoint直接确认。**
  目标导出`outputs/lora_pretrain/lora_state_dict.pth` SHA `4701e770e658a4f152da695c4b3a3dbb3a0b66152d63c03626399cfea2f4c152`；
  `best_lora.pth` SHA `650a77ed1ee99c0c36d60072b1b8b6a167d5df9b8a483c42538f644ac866e368`，epoch字段29（第30轮），96个LoRA张量与该导出逐张量完全相同。
  两组AdamW原状态均为LR1e-4、betas(.9,.999)、wd1e-4、eps1e-4；16个有优化器状态的参数张量step均为15000。
- **batch2为条件性强推断，batch8不能继续写作已确认实跑配方。**
  源码每批更新一次、drop_last=true；按30轮、每轮固定1000源计算，15000/30=500批/轮，对应batch2；batch8仅有3750次更新。
  2026-08-05 README及8月31日旧复现入口明确写batch8，但没有目标run的原始启动命令，且10月3日数据manifest的1000源快照不能证明8月4日数据量从未变化。
  还需原运行日志排除续训或中途换批量/数据的情况，不静默修改旧入口以伪装冲突已解决。
- 入口`tools/pretrain_lora_ssl.py`的历史源码支持letterbox1024、32–64px块遮挡、仅遮挡区域L1、梯度裁剪5、FP32、无学习率日程；
  `mask_ratio=.4`仍只是默认值，实际CLI覆盖值没有恢复。源码未固定完整SSL训练随机种子，不能把下游划分seed42写作SSL训练seed。
- LoRA组注册96个参数张量，但原优化器仅8个LoRA张量有状态，重建头8个全部有状态。
  因此“注入48层”不等于48层都在SSL实际获得更新；恢复时须保留历史重建头所取特征层，不擅自改成多尺度重建。
  细节、搜索范围和余缺见[SSL历史参数补证](SSL_PARAMETER_RECOVERY_20261101.md)。

### 5.2 Stage1-LoRA

- 从官方底座＋上述SSL LoRA开始，任务FPN随机初始化；只开放LoRA与自建语义/边界FPN，静态骨干冻结。
- **2026-11-01补查更正：原Stage1 checkpoint内嵌完整config，不能把主要参数仅列为恢复配方。**
  `output/compliance_review/lineage/early_times.jsonl`第4条直接绑定SHA `62037119…01c2f`及该config。
  它确认32人工源按seed42/.8切分、1024 letterbox后训练裁512、batch8、100轮预算、
  任务头LR1.5e-4、LoRA比例.5、wd1e-4、clip1、25轮warmup后cosine、FP32（amp=false）和SSL初态路径。
  AdamW及config未显式记录的eps1e-4来自训练源码默认值，证据级别与内嵌显式参数分列；
  可移植的`repro/configs/stage1_lora.yaml`用于恢复执行，不能拿邻近run的batch4配置替换该原checkpoint的batch8。
  语义BCE/Dice（Dice .3）＋边界Focal/EDT（gamma2、alpha.75、boundary alpha3）；checkpoint选择复合指标权重语义.2/边界.8。
- 产物`outputs/stage1_lora/best_model.pth` SHA `62037119de3a3628941c86fd04bf8a51858ce30f8b05e138eb3cd2dc7eb01c2f`，归档元数据best原字段80。
- Stage1仍没有独立原run_info/完整启动命令、精确每步更新总数及完整耗时；这些属于执行记录缺口，
  不再据此把已由原checkpoint确认的batch/LR/日程/AMP列作未知。

### 5.3 joint-v3与V6

| 项目 | joint-v3（原运行快照） | V6（原运行快照） |
|---|---|---|
| 初始化 | Stage1-LoRA best；原run_info明确`--phase joint --init_from_checkpoint outputs/stage1_lora/best_model.pth` | joint-v3；语义生成的单线边界缓存，不重新标测试图 |
| 数据/尺度 | 32人工划分＋训练1000图；image1024/crop512，标注batch4/无标签batch4，每轮抽250个无标签样本 | 同共享数据入口及1024/512；采用semantic_boundary缓存 |
| 更新/冻结 | 自建双FPN＋LoRA；静态SAM2冻结，LoRA ratio2 | 语义/LoRA冻结，训练边界分支；该冻结安排需保留原phase实现，不能只看yaml的继承值 |
| 预算/选择 | 100轮曲线，best原字段88 | 60轮曲线，历史采用epoch字段9 |
| 优化 | 主LR5e-5，seg比例.3/边界1/LoRA2，AdamW wd1e-4、eps1e-4、clip1；warmup5、flat50、线性衰减到.4倍 | 主LR5e-5，seg .3/边界1；warmup5/flat50/end .4沿原快照；冻结项对应组无更新 |
| 半监督/损失 | 固定Stage1边界缓存，unsup .4、ramp10；语义unsup0；patch mask .2、块160/8块；边界Sobel.5/TV.05/背景抑制.2、正例3、margin项1/margin.4，rate .1→.6/40轮，peak hinge .5/阈值.8 | 固定语义边界缓存，unsup .4/ramp10；rate固定.2；余项按原快照；语义无梯度 |
| 精度 | 原train.amp=false：FP32 | 原train.amp=false：FP32 |
| 输出SHA | `dde542325a28666de730a72d6f77f37149c8ac02b44c5434cddad74725f317ce` | `c4c9827a18ecdda105056d502e5f93a92fe7fc93ada24d63886b20726c7ec557`（历史恢复锚点） |

joint-v3原run为`20260805_141030_joint_lora_v3`，V6为`20260808_111823_v6_semantic_label`；原snapshot、CSV和run_info保存在`output/compliance_review/lineage/early/`。
joint-v3与V6精确逐步优化次数/早期AMP字段之外的执行记录未重新核全；表中轮数来自曲线，不能自行用当前数据加载器推算为历史实际步数。
这些时期还保留了若干失败/旁支run，不能按目录时间把它们一概串入当前权重谱系。

## 6. 当前backend的语义与几何训练

### 6.1 S-align：当前语义头来源

- 入口：`train_stage2.py --config config/train/stage2_semantic_gt_aligned20.yaml --phase semantic`。
  V6固定教师在学生重置前保存；完整语义FPN、分类层和highres_v1 RGB残差路径随机初始化，共6,302,010参数。
  编码器、LoRA及非语义分支冻结。没有D5a恢复输入，不能称该已评分语义头经过D5a域适应训练。
- 人工r8 GT：25训练/7验证；有标签图等比letterbox1024后随机512裁窗，验证整幅1024。
  无标签训练图来自1000图，教师/学生使用whole1024，每轮248次/标注与无标签batch各4，共62更新/轮；教师/学生共享flip/90°空间坐标，外观增强仅作用学生。
- 实际20轮**1240更新、AMP跳步0**；以r8验证loss最小选择e13（checkpoint原epoch12），不把未采用e20当当前语义。
- AdamW LR1e-4、wd1e-4、eps1e-4、clip1；warmup2（起始倍率.25）、flat12，随后线性衰减至.25倍。
  **FP16 autocast＋GradScaler初值256**，不是近期native训练的BF16。固定教师锐化温度.7、置信度≥.9、蒸馏权重.12→.03/12轮；EMA实际上关闭。
- 主监督为语义BCE/Dice＋实例core/pool/thin项：Dice .30、instance .75、core radius3/min12/boundary .20、类平衡、F权重1、hard gamma.5/floor.5、pool .5、thin1.5；Tversky关闭。
  物理外观增强从第1轮渐进4轮到概率.6，dark_rim关闭，enable_patch_mask=false；不能因父配置含mask参数就说本阶段做patch masking。
- 输出：`outputs/20260917_align_nosam2/semantic_training/best_model_stage2.pth`，SHA `ea905a94355117b750feeb73a99f3aca3200ec2219d1ae5f61a85e54d90dff44`。
  原semantic.log明确cold reset、25/7、batch4、固定教师及优化器分组；20行CSV重加1240更新/0跳步。
  原best/final回执确认96个LoRA张量及36个非语义decoder张量未变。

### 6.2 V affinity：R1之前的几何初始化

- 入口：`train_affinity_geometry_g1.py --config config/train/affinity_geometry_g2_skip_v6.yaml`。
  原日志明确从joint-v3的`decoder.boundary_fpn`严格载入FPN；上采样/八通道输出随机初始化，**不加载G0/G1/G4b或V6边界权重**。
  冻结共享编码器/LoRA/语义，只更新geometry decoder；输入1024、affinity网格512，FPN256/up128。
- 原人工GT26训练/6验证（seed42），SAM64无类别整图；每轮有放回抽52样本，人工/SAM采样概率各.5，batch2。
  120轮、26批/轮，即3120次计划更新；已核120行原曲线及完成best e115，**该旧CSV没有optimizer_steps/AMP skip列，不能把3120计划数标成逐步核验且零跳步**。
- AdamW：FPN5e-5、新层1e-4，CosineAnnealingLR到5e-6，wd1e-4、默认betas(.9,.999)/eps1e-8、clip1；AMP=true，实际入口默认FP16 autocast/GradScaler。
  正负边平衡BCE，negative1.5、hard-negative1/gamma2、edge权重归一；unknown关系忽略。以6图val_affinity_loss最小选择。
- 增强：flip/90°；brightness/contrast/gamma .12，噪声概率.25/std.01、模糊概率.1、随机crop概率.5/scale .65–.9。
  恢复配置完整值见私有evidence；原log直接支持两组LR及采样量，未把当前yaml当唯一执行证据。
- 输出：`outputs/20260916_g2_skip_v6/training/best_affinity.pth`，epoch115，SHA `9bfacdad4d82d148e950f5fe36d08af44290244835bd70b35a93d94cb7c8a739`。

### 6.3 D5a输入下的control、native、mixed与R1

共同起点是joint-v3/S-align/V严格组装，初始化state SHA `2142ea364eea8c53abd02019c4ac1e26d0a92346a55b85d16beda4f842fbd1c2`，
冻结D5a e60。32人工r8每源重复2次＋64 SAM2源，每轮64更新、无留出。每更新顺序处理1个人工样本和1个SAM样本，
分别回传后再做一次optimizer step：有效一人工＋一伪样本，**不是同源batch2**。总loss为人工affinity项＋.5×SAM affinity项。

| 独立实验 | 唯一主要配方差异 | 末态SHA及采用状态 |
|---|---|---|
| 全图control | whole等比1024；原negative1.5，附加connectivity loss关闭 | `outputs/affinity_connectivity/control/final.pt`；作为历史审计控制，非当前backend |
| native p1024 | 原图先裁1024再进入同一退化/D5a；negative1.5 | `b750a3da2e93195bfa1f220aa1873ac7e85fe6ab34302afc614959fb0e583fca`；非当前backend |
| mixed | 每轮固定32 whole＋32 native1024，独立view seed20260930；negative1.5 | `f047f70823df16ddd361231fd3499e5a703e9300e049bf66c8e7b82f1fc220f7`；审计参照 |
| **R1 mixed balance** | 同mixed图序/视野/增强；negative_weight **1.0** | **`43da103c…e80e434`，当前backend来源** |

上述适配每组20轮/1280更新，seed20260928；只训练affinity decoder，**S-align语义、编码器/LoRA及D5a全冻结，包括非语义buffer**。
AdamW LR2e-5、wd1e-4、eps1e-4、clip1；20轮余弦到.1倍/2e-6，无额外warmup；实际FP32，不能引用父配置`amp_dtype:bfloat16`冒称BF16。
继承的semantic LR1e-5和LoRA LR1e-6不在实际optimizer中。

R1关系BCE参数：negative1.0、hard-negative1/gamma2、pseudo_negative1、normalize_edge_weights=true；
manual_uncovered_as_boundary=false，父配置保留的uncovered weight .20不启用。connectivity/interface附加项在本采用训练中为0；
本轮重加1280条steps确认whole/native各640，人工32/SAM64、所有auxiliary_weight=0，冻结/严格重载回执通过。

输入先做对应whole/native视野，再在线D4/退化并通过冻结D5a first_x0。
退化沿D5a配方：清晰/模糊混合、sigma .4–5.2、重采样概率.5/scale .65–1、gamma .8–1.25、曝光 .8–1.15、通道gain .95–1.05、illumination .15；
另有probability .5/sigma0–.006的训练噪声。**历史训练使用`restore_first`默认随机恢复状态，不是2026-10-31才固定的全零推理合同**。
全零推理已获得评分支持，不能追溯修改过去的训练工况。

实际入口顺序（审计引用准备好后）：

```text
train_affinity_connectivity.py --config config/train/affinity_connectivity.yaml --arm control --output outputs/affinity_connectivity/control
train_affinity_native.py --config config/train/affinity_native.yaml --mode train --size 1024 --output outputs/affinity_native/p1024
train_affinity_mixed.py --config config/train/affinity_mixed.yaml --output outputs/affinity_mixed/candidate
train_affinity_balance.py --config config/train/affinity_balance_mixed.yaml --output outputs/affinity_balance/mixed
```

命令是已核入口索引，不是当前可无前置条件一键冷启动的承诺。各级历史SHA、参考输出及逐步对照依赖见第10节。

## 7. D5a：独立的RGB恢复训练

- 入口`train_rgb_diffusion_restoration.py`，候选配置`config/train/rgb_diffusion_d5a.yaml`；从随机初始化开始、seed42，不继承D4或其他恢复权重。
- 全1000赛方训练图，无留出；先等比letterbox1024、在线退化，再取512训练crop及空间增强。目标来自同图清晰RGB，不需要实例/相区GT。
- PixelDiffusionRGBRestorer/NAF骨架：width32，encoder[1,2,4]、middle4、decoder[2,2,1]、time_dim128；16步，kappa.03、eta_start.001、schedule power.3；3,283,651可训练参数。
- 60轮、batch4、250更新/轮，共**15000实际更新，failed_updates=0**。AdamW LR2e-4→2e-6余弦、wd1e-4、默认eps1e-8/betas(.9,.999)、clip1；RTX4090实际BF16 AMP，余弦按epoch推进。
- 时间步t=16概率.5，其他15个步共享剩余.5；训练对带扩散噪声的中间状态预测清晰图。loss=`RGB L1 + .2×梯度L1`。
  没有短链反向展开监督；16步时间表不表示每次训练都反传完整16次恢复。
- 清晰/均匀模糊/原平滑随机场/强弱端点平滑场期望20/40/20/20%；端点weak sigma .6–1.5、strong4–5.2、transition .2–.5，两端有效域至少5%。
  其余外观与重采样同第6.3节。局部损伤前6轮渐进，之后模糊样本中以.25继续；不是用测试图当清晰目标。
- 固定e60，不依真实无标签过程图挑epoch；原summary计时3238.42秒。自动保存清晰/退化/首步/多步及固定无标签真实图缩略图。
- 输出/导出身份见第2节；当前推理只调用D5a的first_x0，2026-10-31取消推理注噪属于部署变更，不是重训D5a。

证据：原`raw_logs/2_run_config.json`、`3_summary.json`和15000更新记录；原配置记录training_noise=null不表示不做扩散q_sample噪声，实际算术由该版本训练器决定。

## 8. 当前NAF width1000：只恢复边界的独立训练

本轮直接只读核服务器既有`naf_width/train/config.json`、`candidate/identity.json`、`candidate/complete.json`与preflight。
实际config SHA `7454699d756ca922900e8ef83d276532d9b587b61181204b53b1b9d88e2fc906`；identity文件SHA `31b93360e0470f4941634128488f89ba2f04b256969ddc07046e6608007465b3`。
当前yaml继承中的合成擦断、RGB条件及PH阶段有大量旁支遗留，以下以actual identity为准。

| 项目 | 实际执行配方 |
|---|---|
| 初始化/模型 | G/D独立随机初始化，不继承旧NAF或短测末态；G seed20261026，D seed20261027；G width48、enc[1,1,2,4]/middle2/dec[1,1,1,1]，17,632,273参数；D width64，2,762,689参数 |
| 数据/输入 | 32源r8目标；固定13b3/9315生成真实N1缓存（局部zero）；整张原生N1先`B>=.5`、skeletonize、ellipse8 dilation，再crop/D4；单通道boundary，**不输入RGB或连续N1** |
| 目标/有效域 | 原生不同已知ID双侧边界；known/filled参与，unknown/frame/crop rim/padding ignore；GT不遮挡输入T，T预测不充当target |
| 抽样尺度 | 原生crop256、stride50、D4共8种；16000抽样，32源各500 draw；候选增强窗430080，非穷尽；实际训练1个预定抽样epoch |
| 预算/batch | **1000更新，effective/micro batch均16**；topology_updates=0；不是继承父yaml的5000主训＋1000 PH，也不是论文53000步完整训练 |
| 优化器/LR | G/D均Adam，betas(.9,.999)、eps1e-8、clip1；G起始1e-3，cosine日程分母**53000−1**、最低倍率.01，但只跑1000步；D恒1e-4。不能把它写成1000步余弦降到1e-5 |
| loss | 法向容差覆盖1＋可信内部抑制1，tolerance2、默认orientation radius3、pointwise覆盖；GAN .01，step0–63关闭，从step64启用；exact像素BCE=0、PH=0，无coherent3/相位/shift额外项 |
| 精度/冻结 | FP32；TF32关闭、cuDNN benchmark=false/deterministic=true；训练只加载缓存，不加载/更新backend或D5a；NAF训练D不部署 |
| 监控/终态 | s0/1/100/250/500/750/1000固定过程图，固定test缓存只作无标签展示；1000步complete、计时870.12秒，采用G state20a59，不按早期过程图选轮 |

父配置中的identity_probability .2、gap数量/长度/半宽用于旧合成输入任务；actual identity明确`artificial_corruption=false`，这些设置在real_n1任务被忽略。
实际输入不依GT重新选缺口；NAF目标没有语义相区监督；SAM64不直接进入这组NAF训练。
NAF当前训练目标与R1同为已见32源，不能把训练覆盖率或monitor当独立验证准确率。

入口为`train_boundary_rgb.py --config <本轮展开并封存的boundary_width配置> --output <新私有目录> --mode train --device cuda`，
此前需同版本preflight/smoke、源缓存与目标manifest校验。当前推理tile256/overlap64/reflect32/batch16、一次概率拼接，是部署合同，不是额外训练epoch。
详情见[width实验](NAF_WIDTH_TRIAL_20261027.md)。

## 9. 近期语义候选：已经训练，尚未采用

| 项目 | native-zero接续 | native-random单组重训 |
|---|---|---|
| 初态 | 从13b3当前S-align语义权重接续 | 从同13b3构建后**完整重置**语义FPN/分类/RGB highres三组，seed1101；未加载上一组e20 |
| 可训练/冻结 | 仅完整6,302,010参数/65张量语义头；编码器/LoRA、affinity、D5a全冻结，训练不载NAF | 同左，结构不变；norm初始化1/0，RGB残差末端按结构约定归零 |
| 数据/视野 | 同32源r8、无留出；native1024部署网格，每源12窗，4窗/源/轮，D4在zero-D5a前；unknown/padding忽略 | 完全相同的数据/GT/采样seed20261031；前三步实际输入/恢复哈希与接续组相同 |
| 预算 | 20轮×128更新=2560；batch1，workers0 | 同左 |
| 优化/LR | AdamW 3e-5，warmup2，后余弦到.1倍；wd/eps1e-4、clip1 | AdamW **1e-4**，同日程到1e-5；其它相同 |
| loss/精度 | BCE/Dice/instance core/pool/thin保持第6.1节系数；D5a/编码器FP32、decoder BF16、loss FP32；TF32关闭/deterministic algorithms开启 | 同左 |
| 版本/权重 | `semantic_native_zero_delta_v1`；final文件SHA `d03ef974334576bd4f9d139d4f7c5b637159ffe89795116540242eab533abbe4` | `semantic_native_random_delta_v1`；final文件SHA `7735fc899ccb12c0c3bb29f38e19eb22d6e093ea9f8837a5245fb326f912df63` |
| 终态head SHA | `c693c549b9dece01bb840add323f5cbce3bf4fbe623cbf224f73d05e286d8ff3` | `8fb1205bf52dea2c4bd3a7a703f1b190b9fffa954207ff2fb4e03e49e6886037` |

两组均已有真实GPU断点接续逐值核验、2560步CPU终态验收和同12图部署分析；e0/e1/5/10/15/20各四训练源缩略图保存。
random格式分列原始head `463c50de…0b23a`、随机初态 `3a236cec…7d16`和训练终态；不能让随机初态冒充不可变基座。
接续组e0复现的是旧头的**统一原生U0**，不是原C大图；random e0有意不与U0相等，因此使用初始化/affinity桥接，不能补写e0 exactC。
两组输入是D5a恢复后的RGB经共享编码器得到的特征，不是直接读取D5a内部中间特征。
训练数值环境比原部署更严格，不能说冻结特征在两种运行工况下逐bit一致；实际部署exact另有GPU证据。

这些候选同时存在全图letterbox及native1024输入域差异；未获得候选官方分，不替换13b3。
random相对接续组同时改变**初始化和学习率**，不能把差异全归因随机初始化。
也不能把2026-09-25已训练但未采用的semantic_d5a full/simple当作当前已评分语义头来源。
详见[接续训练](SEMANTIC_NATIVE_ZERO_20261031.md)、[随机训练](SEMANTIC_NATIVE_RANDOM_20261031.md)及各自终态分析。

## 10. 复现入口当前覆盖范围与必须补齐的接口

顶层`python train_pipeline.py`转发给`repro/pipeline.py`，配方为`repro/current_pipeline.yaml`。
默认13阶段覆盖SSL、Stage1/cache、joint、S-align/V、D5a、fresh R1、三模型导出、真实零噪N1缓存及NAF。
S-align默认取joint固定教师，显式`semantic_teacher_source: legacy_v6_e9`才保留旧15阶段。
**原GT配方从头完整GPU实跑仍未验证。** 已完成的教师32源演习更换了监督，并导入SSL/D5a，
其工程完成及较低官方回分单独记录，不作为原GT训练或原C得分复现证据。
操作和限制见[当前统一入口](CURRENT_TRAINING_ENTRY_20261101.md)，旧入口和历史事实保留。

| 入口 | 实际用途 | 验证边界 |
|---|---|---|
| `train_pipeline.py`＋`repro/current_pipeline.yaml` | 原人工标注/purified GT及后期r8的默认13阶段重训配方 | 可检查/分段执行；尚无原GT整链fresh实跑与原C得分重现记录 |
| 同入口＋`repro/teacher_pipeline.yaml` | teacher格式32源监督演习，SSL/D5a显式导入 | 训练及150图推理已完成；用户回分约80.4，候选淘汰 |
| `repro/train.py` | 旧E10a/G4b历史编排 | 不覆盖当前三模型链，SSL batch8与原optimizer计数存在冲突 |
| `infer.py --backend torch`（默认） | 调用原C compact等价精简推理 | 固定旧三权重；不会自动采用新训练导出 |
| `infer.py --backend tensorrt` | 全五引擎＋CPU整数读出候选 | 显式engine清单及逐角色来源state核验；不等于已有整合版官方分数 |

`repro/train.py`是旧编排器：prepare_labels → LoRA SSL → Stage1 → Stage1缓存 → joint-v3 → semantic缓存 → V6 → E10a → G0/G0-long/G1/SAM审核/G2/G4b。
其`prepare_labels`输出早期`data/purified_gt`，**不生成当前r8实例GT**。它止于旧E10a/G4b模型，不覆盖当前S-align/V/D5a/R1/NAF，也不导出当前三权重。
故“可dry-run/配置可加载/源码齐全”不等于“当前最高分方案已从零完整复现”。原`output/delivery_training/training_audit.json`也明确from_scratch_training_verified=false。

后续完整复现需区分已补接口与尚待实际训练验收的事项：

1. 固定数据版本：原32 LabelMe、r8或其未来新版本、SAM64及内容SHA、split和ignore规则；不得把测试图或隐藏GT加入。GT优化后的版本不得沿用旧manifest SHA。
2. 当前分支编排已补：joint→V及joint固定教师→S-align；独立D5a；共同初始化→R1；导出backend/restorer；冻结真实N1缓存→NAF。旧joint→V6→S-align仅作显式legacy；原GT默认配方仍待完整fresh GPU及部署验收，教师演习已有独立结果且未晋级。
3. 统一入口使用独立fresh R1和本轮输入/冻结/重载回执，避免依赖历史control/mixed日志或旧参考ZIP；原消融训练器的旧合同不放宽。新上游身份动态记录，不通过改旧SHA常量伪造对照。
4. 区分“重现历史权重”和“GT优化后全量训练”：后者会改变监督与所有下游输入分布，应保留旧C回退，重新确认语义/affinity初始化与冻结策略；不能只换NAF目标而无视上游已改变。未来是否复用无标签SSL/D5a，须随实际GT变化与重训范围说明，不能把不受GT训练的分支强行写成已重训。
5. NAF重建依赖新的上游真实N1缓存。缓存必须绑定新的backend/restorer、尺度、噪声和融合合同；旧13b3/9315缓存不能冒充新模型输入。真实目标仍由新GT独立生成，不能用本模型预测替代GT并以自一致性当正确性。
6. 为每阶段保存展开配置、源码/数据seal、实际update计数、优化器/LR/RNG、冻结摘要、初态/终态及小过程图。候选必须完成固定推理链与官方目标比较后才讨论晋级；训练loss、已见GT或无标签翻转率不代替官方分。

原SAM64旧审核回执、SSL原启动命令/实际mask ratio与batch冲突、Stage1启动/逐步计数记录及完整训练时间仍有凭据缺口；
Stage1主要显式训练参数已有原checkpoint config，SSL的LR/优化器/15000更新也已有实物记录，不能继续将它们笼统列为未核验。
SSL源码没有固定完整训练随机种子；这属于历史可重复性限制，不能靠补一个猜测的seed解决。
原C历史链的直接计时四阶段约2h12m，加入joint/V6文件mtime旁证约5h06m，仍缺SSL/Stage1/缓存准备等；
教师监督链的5小时46分23秒不含重训SSL/D5a，**不能据此写原C从零完整链已实证≤72小时**。
当前3模型参数数低于500M与完整训练耗时核验是不同问题，均依[当前比赛规则](../COMPETITION_RULES.md)分别说明。

推理分流不改变训练谱系。`infer.py`默认Torch；TensorRT候选要求五角色engine文件SHA及其来源模块state SHA严格一致，
保留原NAF sigmoid、全零噪声和C几何/投票/final200。编码器、语义、affinity、D5a为静态B1×1024，
只有NAF的256窗口支持batch1–16；批量处理150张图不等于网络batch150。
命令与结果合同见[推理接口](../repro/INFERENCE_INTERFACE.md)，加速数值与评分证据见各自专题报告。

## 11. 证据索引与本轮核验边界

| 记录 | 用途 |
|---|---|
| `output/compliance_review/lineage/early/`、`early_logs.jsonl` | SSL原meta、joint/V6原config_snapshot/run_info/曲线及归档SHA |
| `output/ssl_parameter_search_20261101/`及[SSL补证](SSL_PARAMETER_RECOVERY_20261101.md) | 全358个可达Git提交/4版SSL源码、15份fs manifest；原SSL best与目标LoRA逐张量绑定、优化器及batch冲突 |
| `output/compliance_review/lineage/early_times.jsonl`第4条 | 原Stage1 checkpoint的真实SHA、epoch字段和完整内嵌config；补查纠正其参数证据级别 |
| `output/compliance_review/lineage/raw_logs/10_semantic.log`、`11_metrics.csv` | S-align实际初始化、split、1240更新/0跳步、选e13 |
| `output/compliance_review/lineage/raw_logs/13_training.log`、`14_training_status.json`、`15_metrics.csv` | V实际FPN初态、26/6＋SAM64、分组LR、120轮/e115；旧表无逐步计数列 |
| `output/compliance_review/lineage/raw_logs/2_run_config.json`、`3_summary.json` | D5a实际展开配置、1000来源、15000更新、BF16、固定e60 |
| `output/compliance_review/lineage/raw_logs/5_status.json`、`6_steps.jsonl` | R1实际完整config、三祖先SHA、冻结/FP32、1280步及whole/native640/640 |
| `output/compliance_review/lineage/local_evidence.json`、`encoder_base_comparison.json` | 正式导出与原训练checkpoint绑定、官方静态编码器既有比对 |
| `output/delivery_training/README.md`、`training_audit.json`、`code/` | 原随包训练源码/入口及资源缺口；不是一次完成的从零训练回执 |
| `output/training_recipe_20261101/naf_remote_receipts.json` | 本轮只读取得NAF actual recipe/identity/step1000完成记录 |
| `output/semantic_native_{zero,random}/final_verified.json`及`received/` | 近期两个候选完整训练终态与部署比较，和当前C分开 |

本文最初参数整理只重加S-align更新数、R1逐步来源/视野/附加项，并读取既有权重元数据与NAF训练小JSON；
SSL补查另以CPU读取同SHA原best和fs内LoRA导出，比较全部96个张量，并核对原优化器、小文本及历史数据manifest。
上述整理与补证当时未生成GT、训练或运行GPU；后续教师生成、训练、150图推理及回分均见独立报告，不能把两次工作的范围混写。
本次文档状态整理没有重新执行这些任务，也没有修改原数据、权重或提交材料目录。
复现交付仍应携带实际允许的数据/资源与严格版本记录；本文提供当前配方与缺口说明，不把缺失凭据包装成已通过的完整复现。
