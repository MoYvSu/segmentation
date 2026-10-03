# 固定新最佳实例图的语义末端诊断（2026-10-01）

入口 `python tools/probe_semantic_fixed.py`，默认配置 `config/train/affinity_balance_native.yaml`。只读新最佳native1024 balance e20的实例PNG：测试来自 `outputs/affinity_balance/native/deployment/patch1024/`；四张已见训练源来自 `outputs/affinity_balance/analysis/train/<source>/native_local/`。不运行geometry，不改变分水岭，不训练，不扩100图，不制作新的语义提交包。

## 路线与身份

| 路线 | 语义输入 | checkpoint及SHA256 |
|---|---|---|
| 原S-align | 原图 | balance e20 `2bbae197f4716ced694309b9c5d49b5e24a4154f74755d4652d2beb0e1318d5b` |
| 相同S-align | 旧D5a e60首步 | 同上 |
| 语义专用LoRA | 相同D5a张量 | `outputs/semantic_lora/candidate/epoch_020.pt`，`23b23d1f0cecb268091943610ceddffd1dffc4371ca548ce725639e2dfc4068b` |
| 完整语义适配 | 相同D5a张量 | `outputs/semantic_d5a/full/epoch_060.pt`，`cce91a6643b919e69f6cd1f6f4b331265a33c80842e1587a241386d964f7d2a7` |

D5a固定为 `048954795dc634f82fecf96d5e02bb2f2505afe5eb76298dff5a1b0713b3c80a`。每图只恢复一次，三条恢复输入路线共用同值张量。保留原部署letterbox/reflect和tensor stride，不强制转换NCHW连续布局；FP32，全图logits先恢复原尺寸，再CPU sigmoid，以float32实例均值严格`>0.5`投票。

12图（既有四张GT、test101/test116、种子20261001随机六张非monitor图）全部通过：原S-align/raw classes逐实例精确复现；所有候选PNG直接复制原文件字节，SHA相同；三个后端共同encoder/geometry LoRA摘要相等，全部状态运行前后不变。语义LoRA经 `models.semantic_lora.semantic_features` 调用私有LoRA，实际前向12次，私有权重与geometry LoRA不同。

三个完整bundle及D5a实际诊断参数249,041,872。把新最佳完整bundle与单个候选完整bundle一起计数的保守部署上界：LoRA167,374,478，full166,618,439；原部署84,951,045。均低于500M。此处不把共享参数复用后的更小值冒充已经实现的部署参数。

## 固定配对GT诊断

只使用原标覆盖域，未知域同时从GT/预测删除。按类无关几何作一次一对一IoU≥0.5匹配，再将完全相同的配对集合交给四条语义路线。未匹配的几何错误不进入语义混淆。

四图共有589个匹配实例（铁素体359、珠光体230）。四条路线均零分类错误。改类只发生在配对之外的实例，因此这些训练内结果不能支持某条语义路线更优，也不能证明测试语义已经正确。四图参加过训练，且GT不完整。

## 八张无标签测试图

| 路线 | P→F实例 | F→P实例 | 预测F实例总数 | 预测F总像素 |
|---|---:|---:|---:|---:|
| 原S-align/raw | 0 | 0 | 520 | 15,130,181 |
| 相同S-align/D5a | 0 | 58 | 462 | 14,912,708 |
| 语义LoRA/D5a | 55 | 15 | 560 | 15,139,000 |
| 完整适配e60/D5a | 25 | 51 | 494 | 14,987,989 |

LoRA的55次P→F中，22块小于200px，45块小于1000px，面积中位268px。F实例增加40，但F像素仅增加8,819；逐图预测铁素体平均面积变化中位数为−6.49%。这属于预测分布和面积项敏感性风险，不是已知错误或官方面积成绩。

S-align直接改用恢复输入产生58次单向F→P，改变217,473px。full e60改变76实例、其中51次F→P；这两个恢复输入路径没有展现比原图末端分类更稳健的机制信号。LoRA在test101的较大片类别分布基本沿用raw，较多变化发生在小实例。

每个实例保存全区域、形状自适应核心、外环概率、margin及核心/全域分歧。核心只作诊断，没有用于改投票。低置信预先定义为原模型实例均值距0.5小于0.1；三候选改类中原模型高置信者分别42/54/57，不能把模型翻转直接视为纠错。

原S-align/raw的八张测试图772实例中，核心与全域均值类别只有13处不同，test101占4处。该小诊断不足以把“范围错配”确诊为主要原因，也不足以支持改用核心投票；本轮保留原均值票。

## 结论、验证及私有产物

没有足够证据在截止期内立即切换语义路线或追加语义训练。私有LoRA路径可以保留为独立备选，但当前信号主要来自小块改类，需官方验证才能确认价值；本诊断不建议抢占mixed balance回分前的黑盒机会。语义仍有提升空间，本检查只是现有三个替代路径缺少晋级证据。

5项合成测试通过（配对类无关/未知域、合并一对一、空覆盖、严格0.5、字节复制）；GPU实跑exit 0。私有产物在服务器 `outputs/semantic_fixed/`、本地 `output/semantic_fixed/`，包含24张图册、逐实例概率JSON、固定PNG哈希、权重和架构回执、固定配对及混淆。没有平台提交。

下载图册SHA256：`9bb4e242e4327c18600c43071b0a35ca0d21e277bebd56e2a0804082408e34dc`，CRC通过。工具源码SHA为 `5b2fd9178319e75c63b0e43edab12d41d48b58e85a12d9e49bf47256d914823f`。四个源文件本地/服务器字节哈希相等；`tools/semantic_crossover.py`仅换行不同，规范化文本逐字符相等，服务器原源码快照保留。下载回执为 `output/semantic_fixed/download_audit.json`。所有图像、逐图/逐实例JSON、掩码和压缩包均在被忽略目录。
