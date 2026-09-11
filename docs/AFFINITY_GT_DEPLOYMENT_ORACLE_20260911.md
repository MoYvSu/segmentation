# GT affinity 经过完整部署链的检验

## 结论

**当前 `gated + mean + high=0.65` 没有通过理想 affinity 检验。** 从新 GT 实例编号直接生成
正确的八方向连接，保留正常 E10a 预测和全部主线后处理，五图有效匹配由 715 降至 1。
仅把短程方向融合从 mean 改为现有 top2，有效匹配恢复到 724，漏检计零 IoU 达到 0.84183。
这给出了监督表示与下游读出不相容的直接证据；不能再默认“affinity 越接近 GT，固定部署越好”。

本实验使用 GT 参与预测，是理想输入诊断，**不是模型泛化成绩，也不是官方提交分数**。
部署主线仍保留 E10a+G4b 的官方 83.94；没有改变主线配置或训练任何权重。

## 对照定义与输入

| 对照 | affinity | 语义 | 短程融合 | 其他后处理 |
| --- | --- | --- | --- | --- |
| 主线 | G4b 正常预测 | E10a 正常预测 | mean | 主线固定 |
| GT affinity | GT 直接导出 | 同一份 E10a 正常预测 | mean | 主线固定 |
| GT affinity + top2 | 与上一组相同 | E10a 正常预测 | top2 | 主线固定 |

第三组在主试验失败后增加，仅用于定位融合环节。它不检验真实 G4b/e110 预测改用 top2 的效果。

- 图像：`train_172/351/686/872/873`，共 790 个 GT 实例，其中铁素体 459、珠光体 331。
  与最近五图报告保持相同口径，排除 889。这不是新建的独立测试集。
- GT：`outputs/experiments/seeded_label_completion_o1/radius_8/derived_maps`，新 GT 补缝版本。
  全图共 25,013,120 像素，已知标注 24,022,944 像素，占 96.04%；其余保持 unknown。
- 1024 输入，`legacy_none`，等比 letterbox + reflect padding。GT 按训练使用的
  `letterbox_instance_geometry` 最近邻映射到 512 网格；790 个实例均仍出现在该网格。
- 八个 `(dy,dx)`：`(0,1),(1,0),(1,1),(1,-1),(0,2),(2,0),(0,4),(4,0)`。
- 两端均有有效 GT 时，同实例 affinity=1，异实例 affinity=0；unknown、填充及越界连接
  保留 G4b 原始 logits。因此准确称谓是“已知连接上的理想 affinity”，不是完整无未知 GT。
- 使用正负无穷 logits，让 sigmoid 后有效连接**精确为 0/1**；先融合再恢复原图尺寸，
  不对这些无穷 logits 做插值。边界概率转回 logits 时仍使用主线原有截断函数。
- 正常 E10a logits 继续进入分水岭地形和实例平均概率分类。GT 不参与前景、种子或类别投票；
  GT 在 affinity 生成后只用于计分、诊断和配色。
- 完整后处理固定：high=0.65，marker low=0.45/8步重建，seal=2，bridge=1，dilate=1，
  min_area=50，probability_mean/阈值0.5/不腐蚀，原尺寸 uint16/最大ID<=65535。

## 完整最终输出

| 指标 | 主线 E10a+G4b | GT affinity + mean | GT affinity + top2 |
| --- | ---: | ---: | ---: |
| 有效 GT 区内预测实例数 | 1183 | 48 | 818 |
| 类别感知有效匹配 | 715 / 790 | 1 / 790 | 724 / 790 |
| 有效匹配实例 mIoU | 0.84750 | 0.79524 | 0.91857 |
| 漏检计零实例 IoU | 0.76704 | 0.00101 | 0.84183 |
| 铁素体预测实例数 | 699 | 17 | 462 |
| 铁素体平均面积相对误差 | 35.39% | 2911.28% | 1.38% |
| 完整代理总分 | 74.679 | 39.762 | 95.239 |
| 单图代理总分均值 | 75.289 | 7.970 | 94.439 |
| 已知区域未分配比例 | 1.275% | 0.158% | 1.137% |
| 忽略类别后的几何有效匹配 | 715 | 1 | 725 |

`GT affinity + mean` 的有效匹配 mIoU 0.79524 只来自唯一成功匹配，不能表示整体尚可。
同理，该组虽然覆盖率超过 99.8%，但几乎把所有晶粒合并。需结合漏检计零 IoU 和匹配数量。
GT 辅助 top2 的 95.239 也不能与官方 83.94 比较：输入使用了 GT，评估数据与官方口径亦不同。

| 图像 | 主线匹配 | GT mean 匹配 | GT top2 匹配 | 主线漏检计零IoU | GT top2漏检计零IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_172 | 141 | 0 | 141 | 0.67604 | 0.73263 |
| train_351 | 123 | 0 | 121 | 0.83876 | 0.89249 |
| train_686 | 147 | 0 | 152 | 0.82839 | 0.92630 |
| train_872 | 156 | 0 | 150 | 0.78249 | 0.82184 |
| train_873 | 148 | 1 | 160 | 0.73511 | 0.85801 |

![相同 E10a 和完整下游的对照](../output/20260911_gt_affinity_top2_control/readout_control.png)

配色按与 GT 的最大重叠统一，黑线保留真实预测分区；深灰为未分配、浅灰为 GT unknown。

## 失败发生在哪里

| 诊断量 | 主线 | GT mean | GT top2 |
| --- | ---: | ---: | ---: |
| 面积过滤后分水岭起始区域数 | 1190 | 67 | 843 |
| 与其他 GT 共用主导起始区域的 GT 数 | 65 | 789 | 105 |
| 已知 GT 没有任何起始区域的实例数 | 0 | 0 | 0 |
| 短程跨界起点 B>0.65 的比例 | 94.27% | 22.44% | 60.38% |
| 远离 unknown/填充至少8网格像素的起点 B>0.65 | 94.49% | 20.39% | 58.77% |

主导起始区域按每个 GT 内覆盖像素最多的 marker 定义；两个及以上 GT 共享时一起计入。
这里的 GT 只用于观察，不进入 marker 生成。marker 数与最终有效区内预测实例数不同，原因包括
后续面积过滤和 unknown 区域不计分。
共用数量的统计对象始终是同一批790个GT，可在组间比较；它不随marker总数必然单调变化，
也不能单独代替最终匹配或轮廓质量。聚合数量相同亦不能证明逐实例没有新增合并。

利用归档中的每图`markers.shared_groups`，按GT编号集合复核两组的共享状态：

| 图像 | 主线共享GT | GT affinity + top2共享GT | 后者新增共享 | 后者解除共享 |
| --- | ---: | ---: | ---: | ---: |
| train_172 | 20 | 47 | 41 | 14 |
| train_351 | 6 | 10 | 9 | 5 |
| train_686 | 10 | 4 | 3 | 9 |
| train_872 | 4 | 25 | 23 | 2 |
| train_873 | 25 | 19 | 10 | 16 |
| 合计 | 65 | 105 | 86 | 46 |

两组共享集合互不包含，既有解除，也有新增。这里比较的是“G4b + mean”与“GT affinity + top2”，
输入和融合均不同，不能把86/46的变化单独归因于top2，也不能等同于最终实例合并数。

跨界起点统计使用“至少一个有效短程连接为负”的源像素，共 86,740 个；远离 unknown 的子集
为 55,335 个。**这不是正式边界召回率**：一个物理边界可能有多个方向、多个源像素，不要求
每个源像素都成为最终障碍。它与几乎全部共用 marker、最终匹配崩溃共同支持断界解释。

图中 G4b 的多个方向通道形成较宽且相似的强响应；GT 正确连接只在相应方向跨界时响应，
更细且方向差异明显。mean 混合后，许多必要边段达不到 0.65。局部重建只在强边附近有限步
扩张，无法补齐大量缺少强响应的长边段，分水岭开始之前多数实例已经落入同一连通起始区域。

![相同0到1色标的方向通道与融合边界](../output/20260911_gt_affinity_oracle/train_172_readout.png)

为避免把特定方向的数值错误概括为普遍上限，用实际融合函数补了三个无 unknown 的两实例
128网格直线输入探针，在远离图像外边缘的区域读取峰值：

| 界面 | 四短方向 mean 峰值 | gated+mean峰值 | gated+top2峰值 |
| --- | ---: | ---: | ---: |
| 垂直 | 0.50000 | 0.50000 | 0.78571 |
| 水平 | 0.75000 | 0.64286 | 0.78571 |
| `x=y` 对角线 | 0.50000 | 0.50000 | 0.78571 |

水平界面的短程响应本可到0.75，但距离2/4方向的平均值约0.5，gated加权后约0.64286，
仍低于0.65。这是具体输入的实测结果；曲线、阶梯和多晶粒交汇处可以更高，**0.50不是普遍上限**。
探针结果见 `output/20260911_gt_affinity_oracle/straight_interface_probe.json`。

以下CPU代码调用实际目标生成器和融合函数，复现归档中的峰值与超阈值像素数；从仓库根目录
运行，不需要模型权重，也不生成新文件。统计区域为128网格中央的64×64窗口，远离外边缘。

```python
import json
from pathlib import Path
import numpy as np
import torch
from utils.affinity_graph import build_affinity_targets
from utils.affinity_fusion import affinity_boundary_probability

torch.set_num_threads(1)
y, x = np.indices((128, 128))
results = {}
for name, side in {"vertical": x >= 64, "horizontal": y >= 64,
                   "diagonal": x >= y}.items():
    labels = side.astype(np.int32) + 1
    target, valid = build_affinity_targets(labels, np.ones_like(side))
    assert valid[:, 32:96, 32:96].all()
    logits = torch.where(torch.from_numpy(target)[None] > .5,
                         float("inf"), float("-inf"))
    results[name] = {}
    for key, mode, reduction in [("short_mean", "short", "mean"),
                                  ("gated_mean", "gated", "mean"),
                                  ("gated_top2", "gated", "top2")]:
        p = affinity_boundary_probability(logits, mode=mode,
                                           short_reduction=reduction)[0, 0, 32:96, 32:96]
        results[name][key] = {"maximum": float(p.max()),
                              "pixels_above_065": int((p > .65).sum())}
expected = json.loads(Path("output/20260911_gt_affinity_oracle/straight_interface_probe.json").read_text(encoding="utf-8"))["results"]
for name, variants in results.items():
    for key, measured in variants.items():
        assert abs(measured["maximum"] - expected[name][key]["maximum"]) < 1e-6
        assert measured["pixels_above_065"] == expected[name][key]["pixels_above_065"]
print(json.dumps(results, indent=2))
```

## 对后续工作的影响

1. 当前原始 affinity 监督目标与部署读出存在明确失配。即使优化把有效连接推向正确0/1，
   最终实例也可能变差。对依赖正确0/1连接恢复实例的新设计，应先检查目标到边界/种子的相容性，
   再继续加训练阶段或损失；理想输入检验并非所有学习模型的通用性能门槛。
2. 固定 E10a 和后续分水岭，在换一个融合方式后能恢复绝大多数实例，说明主要断点至少包含
   融合与固定阈值的配合；不能把原失败归因于 E10a 语义头不可用。
3. top2仍只匹配724/790，65个GT连忽略类别的几何匹配也未通过，并有105个GT共用主导marker。
   unknown、网格离散化、方向边到像素边的定位和后续形态处理仍可能影响残余错误；本轮未分别
   隔离这些因素。不能宣称所有下游步骤已经无问题。
4. 本结果不证明真实e110或cross-head退步主要由融合引起，也不推翻此前真实预测top2增益有限
   的对照。[已完成的真实模型六图实验](AFFINITY_CLEAN60_DIAGNOSIS_20260910.md)中，G4b的mean→top2
   匹配854→856、有效匹配mIoU .8482→.8412；e110为792→806、.8408→.8335。
   理想输入与训练模型输出分布不同；不重复排期既有mean/top2对照，不直接把主线改成top2，
   也不据此宣布均值融合完全没有损失信息或否定新GT。新的监督/读出设计需另行验证。
   该历史六图含889，与本轮五图不同，上述数字仅用于各自实验内的配对比较。

## 实现、核对与复现

- 入口：`tools/run_affinity_gt_deployment_oracle.py`，仅新增诊断；调用现有模型、预处理、融合、
  后处理与实例评价器，没有修改生产推理实现。
- 主线五图最终 PNG 和类别 JSON 与上一轮归档**全部逐字节相同**。
  依据为主脚本`--reference-dir`路径上的实际字节断言及归档`baseline_reference_checked=true`；
  两侧完整PNG没有纳入Git，需原始运行目录才能再次核对。指标一致不能替代该字节检验。
- top2控制组在同一checkpoint、配置和精度设置下重新进行eval前向，没有随机数据增强；
  该控制脚本未另存或逐值断言两次前向的原始logits，不额外声称张量级逐位相同。
- GT连接由NumPy生成，并与训练使用的Torch目标生成器逐值一致；逐图检查所有有效连接为精确0/1，
  unknown/填充连接与G4b logits完全一致。CPU探针覆盖同实例、跨实例、unknown和全unknown。
- 原尺寸、单通道uint16、返回值与保存PNG一致、ID<=65535均已检查；两个脚本语法检查通过。
- 配置：`config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml`。
- 部署包格式：`phase_affinity_fused_v1`，参数量81,667,394；SHA256：
  `53e1b5c6f9bf3973723c7bf4c6ce4ebd5eaeaa45210641df9de8f01e510c790f`。
- 部署包构建期元数据记录的E10a epoch19源文件摘要：`1380cf14d63fdbc17dadf2ac73dc47d7c33a7e0f5a61e5edb9d7280ec8b80b91`。
- 部署包构建期元数据记录的G4b epoch20源文件摘要：`1334fdcc8472b4da67386b76b25d9099ad6bcef10b6bd9c77ae6908658b8c619`。
  本次重算部署包整体摘要，未重新读取上述两份原始训练checkpoint。
- 代码基线：`4874017`；新增脚本和运行时关键源码摘要记录在`protocol.json`及控制组`summary.json`。
- 环境：服务器`sam2_env`，RTX4090，PyTorch2.8.0+cu128、OpenCV5.0.0；FP32、无autocast，
  matmul TF32=false、cuDNN TF32=true，与基线一致。
- 运行项目根：`/root/autodl-tmp/segmentationv2_cross_head_20260911_1642`。
  部署包链接指向`/root/autodl-tmp/segmentationv2_pseudo_20260908/outputs/deployment/e10a_g4b_fused.pth`。
- 本地证据：`output/20260911_gt_affinity_oracle/`、`output/20260911_gt_affinity_top2_control/`；
  服务器对应目录位于项目根`outputs/`。两组完整分析产物约12MiB，没有保存稠密张量缓存。

在同一服务器项目根中运行；为保留产物，输出目录必须不存在：

```bash
python tools/run_affinity_gt_deployment_oracle.py \
  --output-dir outputs/20260911_gt_affinity_oracle \
  --reference-dir outputs/20260911_cross_head_results/mainline

python tools/run_affinity_gt_readout_control.py \
  --primary-dir outputs/20260911_gt_affinity_oracle \
  --output-dir outputs/20260911_gt_affinity_top2_control
```

新机器若没有旧主线输出，可省略`--reference-dir`；仍执行真实基线推理，但无法声称完成历史
逐字节核对。GT与部署包需按项目约定路径准备，不需要加载或训练其他checkpoint。

## 对临时分析稿的核验补充

`tmp_diagnostics_20260911.md`是讨论材料，保留原文；以下更正用于后续决策，不把其建议自动
转成训练、清理或部署任务。

- **配对统计**：主线相对cross的逐图匹配差为`[3,1,1,2,8]`。穷举全部`5^5=3125`个等概率
  有放回图像配对重采样，总差值的百分位95%区间是`[6,28]`，不是`[-54,80]`；
  后者与独立重采样两组图像的结果接近，但缺少原始计算代码，不能断言其具体错因。
  五图领先方向一致不等于已证明总体优势；配对t检验p约0.083，bootstrap百分位区间无需与
  该检验给出相同结论，也不要求区间中点等于点估计。历史训练曝光和五图代表性限制仍在。
- **匹配口径**：原稿中的0.4/0.45都未达到0.5阈值，反例不成立。不过“先最大化总IoU再过滤”
  与“先最大化有效配对数”的确可能不同：交集矩阵`[[5,2],[3,0]]`可由互斥实例实现，
  IoU矩阵为`[[0.5,2/7],[3/8,0]]`。前者选反对角后无有效对，后者保留0.5得到一对。
  这只证明两个定义不同，不能证明赛方实现采用其中哪一种，不据此改写当前评分器。
- **Oracle可比性**：μSAM使用GT语义和完美标注支持域，且GT处理条件不同，不能据跨实验的
  有效匹配mIoU相近就宣布“边界已到上界、损失全在数量”。已匹配实例类别全对也不能排除
  未匹配实例的语义影响，更不能消除支持域差异。
- **当前配置**：部署路径确实未转发D1–D5，但本次生效值为边界缩放1.0、两个语义边项0.0、
  不使用中心种子、未启用面积分辨率缩放，与现实现一致；这是修改这些配置时的接口风险，
  不能解释已有实验数字。不在本次审阅中顺带改变生产路径。
- **脚本与尺度**：命令行入口无Python引用不等于死代码；标记为“未复现”的图像尺度统计
  保持未复现，单次裁剪标定有压缩也不能推出所有图像的尺度估计都是严格下界。

本次由Codex与Claude Code中的用户指定`deepseek-flash[1m] / max`完成两轮审阅，
会话`review-1789132098791-f9938a0d`返回`APPROVE`。通过后按第二轮非阻断建议核验并补充了
上述共享集合差、六图/五图标注及理想输入检验的适用范围；报告采用经源码或产物核验的结论。
