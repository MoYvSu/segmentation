# 固定最终实例划分、交换分类来源

2026-09-25。已完成100张复赛图的四格对照，原baseline和D5a两格逐实例类别精确复现。
两份新交叉包已生成并下载、本地校验、固定四图渲染；尚未提交平台、无交叉组官方成绩。

## 所检验的问题

用户怀疑修复改变输入分布，后端分类没有充分适配，或更细的实例划分放大类别错误。
本轮先隔离最后的实例类别投票：固定已有最终实例PNG，仅交换模型从原图／D5a首步图得到的分类概率。
采用已回分D5a，避免把D5b训练变化混入分类来源变化。

这里G表示完整最终实例划分，S表示最终分类概率来源。G已经包含语义参与分水岭、过滤、重编号等影响，
不是纯几何头输出。本实验不能完全拆开几何与语义路径，更不能直接证明LoRA初始化是原因。

| 固定实例划分 | 原图分类 S_raw | D5a输入分类 S_d5a |
|---|---|---|
| G_raw | 已有baseline：0.8428／0.8647 | 新交叉包A |
| G_d5a | 新交叉包B，优先测试 | 已有D5a：0.8469／0.8560 |

四格共享原S-align e13＋V e115后端、joint-v3 e88参考编码器及既有LoRA。
无后端训练、无重新SSL、无测试GT、无人工调整类别或数量。
所有阈值固定，不根据本轮输出数量／面积调参。

## 精确复现与输出约束

GPU缓存复用原`predict_maps_with_challenger`路径及完整权重，核验两原提交的配置、代码、权重和100张输入哈希。
原图保持1024等比letterbox；D5a为FP32、t=16首个x0、kappa0.03、strength1，
按原文件排序使用`314159 + index`的噪声种子，并保持原提交`condition + (prediction-condition)`算序。
restorer e60 SHA256：`048954795dc634f82fecf96d5e02bb2f2505afe5eb76298dff5a1b0713b3c80a`。

语义logits先裁去padding、插值回原尺寸，再按原后处理在CPU做sigmoid；
逐图保存float32的P(铁素体)。每个最终实例内取概率均值，`probability_mean`、erode0、严格`>0.5`归铁素体。
完整部署去重参数97,027,847，小于500M。

新增工具`tools/semantic_crossover.py`：

1. 两原包与缓存100图集合和原尺寸一致；单通道16-bit PNG、类别0/1、ID一一覆盖。
2. baseline原图分类与D5a自身分类两格必须全部逐ID复现，否则拒绝生成交叉包。
3. 交叉只改JSON；源PNG原始字节复制，不重新编码、重编号、过滤、合并或分水岭。
4. 两包各100图／200文件，本地CRC、SHA、原尺寸uint16、最大ID≤65535、类别ID覆盖和PNG字节一致再次通过。

11项CPU专项检查通过，覆盖不连续ID、严格0.5平局、完整区域与裁块投票一致、对角复现门槛及无效缓存拒绝。
真实100图对角复现均通过，两个交叉包mask bytes与对应源包完全相同。

## 描述性结果

| 实例来源／分类来源 | 实例总数 | 铁素体 | 珠光体 | 相对同划分原类别翻转 |
|---|---:|---:|---:|---:|
| raw／raw | 8601 | 5840 | 2761 | 0 |
| raw／D5a（A） | 8601 | 5344 | 3257 | 520，6.05% |
| D5a／raw（B） | 8668 | 5964 | 2704 | 572，6.60% |
| D5a／D5a | 8668 | 5414 | 3254 | 0 |

- A：85/100图有类别翻转；508个铁素体→珠光体、12个反向；涉及1.34%已覆盖像素。
  逐图预测铁素体平均面积相对raw/raw的比值中位数1.03743（+3.74%）。
- B：79/100图有类别翻转；561个珠光体→铁素体、11个反向；涉及1.06%已覆盖像素。
  逐图预测铁素体平均面积相对D5a/D5a的比值中位数0.96099（−3.90%）。
- 翻转后的概率均值距0.5阈值的绝对差中位数分别0.10697／0.16084；翻转不全是极近阈值的数值扰动。

可确认：修复输入会系统性改变现有后端的最终分类倾向；在两种固定划分上，D5a输入都更倾向将实例判为珠光体。
只改变分类、轮廓完全不动，也足以改变预测铁素体平均面积。因此解释面积项变化时必须考虑类别归属变化。
这是输出机制证据，不是分类准确率证据：可能是原图丢失纹理造成漏判，也可能是修复纹理使后端误判。
没有真实GT或交叉回分，不能选定其中一种解释，更不能判定哪些具体测试实例错分。

## 回分顺序与判读

优先提交B（D5a划分＋原图分类），与既有D5a同轮廓比较，直接检验恢复原图分类能否改善最终指标。
其次A（原图划分＋D5a分类）与既有baseline同轮廓比较，用反方向验证分类来源的影响。

若B改善、A退步，将支持D5a输入的最后分类阶段拖累部署；
若B退步、A改善，则更支持D5a类别信息有益，应关注实例划分／两条路径的相互作用。
若两边结果混合，分类效果可能依赖实例划分，需要按实际两项指标保留结论。
即使出现前一种结果，也仍不能将原因唯一归于LoRA的原图SSL初始化。
固定PNG前已存在上游语义作用，因此不能把四格差分当作完整的几何／语义因果分解。

## 私有产物与复现

本地`output/semantic_cross/`，服务器`outputs/semantic_cross/`；全部为被忽略的保密目录。

- A：`submission_graw_srestored.zip`，4,456,250字节；SHA256
  `00945eec7aa6007cf0b08f9e750ff19931ef8e862ff23d1d3a47911f90534233`。
- B：`submission_grestored_sraw.zip`，4,476,478字节；SHA256
  `7887d2eb35848c2412093973f210454621201f943175145066027438e0ff5b93`。
- `report.json`：四格逐图／聚合、逐实例投票分数和翻转统计；`validation.json`：本地二次校验。
- `cache_manifest.json`：准确后端checkpoint路径、epoch、SHA、完整resolved config与D5a参数。
- `cache.py`：本轮GPU概率生成入口；`raw/`、`restored/`缓存仅留服务器。
- `validate_render.py`和`comparison/`：固定四图、左右ROI、金色铁素体／蓝色珠光体／紫色边界。

缓存完成后运行：

```bash
python tools/semantic_crossover.py \
  --raw-zip outputs/20260922_semifinal_submission/baseline/submission_semifinal_baseline_Salign_V_20260922.zip \
  --restored-zip outputs/d5a_candidate/submission_d5a.zip \
  --raw-prob-dir outputs/semantic_cross/raw \
  --restored-prob-dir outputs/semantic_cross/restored \
  --output-dir outputs/semantic_cross \
  --provenance outputs/semantic_cross/cache_manifest.json
```

入口拒绝覆盖已存在report／ZIP；重复运行应使用新的短输出目录。
目前交叉包完整部署输出已核验，官方mIoU／面积得分仍未知；未修改任何默认部署配置。
