# 交接：G2/G4b 官方对照、G4b 阶段零收益，与末端消融现状

更新时间：2026-09-12。承接 `docs/HANDOFF_MAINLINE_ABLATION_20260911.md`。
本文件记录本轮服务器工作、官方黑盒结果与离线代理的首次正面对照，以及一个被证据推翻的消融前提。
**不是**已验证的新主线报告。

---

## 0. 三句话结论

1. **把主线部署的 affinity 从 G4b 换成 G2，官方 +0.47 分（83.945 → 84.415）。**
   离线代理预测 +5.31 分：**方向全对，幅度高估 11.3 倍**。代理能排序，不能定价。
   这个对照是**干净的**：权重确实不同、部署路径同源、评测规则变化对两次提交都无影响。
2. **`G4b` 的选优权重就是 `G2` 本身**——`best_affinity.pth` 的 `epoch=0`，39/39 个 tensor
   与 `G2 best` **逐位相同**。G4b 训练 20 个 epoch，**没有一个 epoch 在该选优指标上超过起点**。
3. **因此"能否删掉 G2 阶段"这个提问方向错了。该删的是 G4b 阶段**：
   从 `G2 best` 再训 20 epoch 就是 `G4b latest`，官方分从 84.415 掉到 83.945。

---

## 1. 官方黑盒结果（用户提供，唯一权威口径）

| 项目 | G4b（历史记录） | G2（本轮提交） | 差值 |
| --- | ---: | ---: | ---: |
| 实例 mIoU | 0.8381 | 0.8423 | **+0.0042** |
| 铁素体面积项 | 0.8408 | 0.8460 | **+0.0052** |
| 总分 | 83.945 | 84.415 | **+0.470** |

按 `50 × mIoU + 50 × 面积项` 复核，两行都自洽（41.905+42.040；42.115+42.300）。
**分解**：mIoU 项 +0.210，面积项 +0.260——面积项贡献略大。

G4b 的 83.945 来自 20260911 交接文件 §3 记录的历史官方成绩，**不是本轮新提交**。

### 1.1 这个对照是干净的（已逐项核实）

两个分数对应的权重**确实不同**，不是同一个东西测了两遍：

| 提交 | affinity 权重 | SHA256 | epoch |
| --- | --- | --- | ---: |
| 历史 G4b | `affinity_geometry_g4b_gap_weight020/latest_affinity.pth` | `1334fdcc…` | 20 |
| 本轮 G2 | `affinity_geometry_g2_sam2/best_affinity.pth` | `96f9456e…` | 21 |

- 权重差异：39/39 个 tensor 不同，`max_abs = 6.62e-03`（最差项 `affinity_head.3.weight`）。
- 融合部署包 `e10a_g4b_fused.pth`（`53e1b5c6…`）内记录的
  `deployment.fusion.checkpoint` 正是 `.../latest_affinity.pth`，
  `sources.reference` 的 SHA `c4c9827a…` 与 G2 提交的 reference 一致。
  **两边的 V6 与 E10a 是同一份权重，部署路径也同源。**
- 两条链的 `split.json` 验证集划分完全相同。

---

## 2. 代理 vs 官方：首次同题对照

| 目录 | checkpoint | epoch | 图数 |
| --- | --- | ---: | ---: |
| `ablation_g2`（五图代理） | G2 best `96f9456e` | 21 | 5 |
| `submission_g2_20260911`（68 图提交） | 同上 | 21 | 68 |
| `ablation_g4b`（五图代理） | G4b latest `1334fdcc` | 20 | 5 |

**代理与提交同源，对照结构成立。**

| 指标 | G4b (ep20) | G2 (ep21) | 代理预测差 | 官方实际差 | 失真 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 有效匹配 | 715 | 668 | −47 | —— | —— |
| 有效 mIoU | 0.84750 | 0.84862 | +0.00112 | +0.0042 | 3.75×（代理**低估**） |
| 漏检计零 mIoU | 0.76704 | 0.71757 | −0.0495 | —— | —— |
| 面积相对误差 | 35.39% | 24.89% | −10.50pp | −0.52pp | **20.2×（代理高估）** |
| 完整代理分 | 74.679 | 79.988 | **+5.309** | **+0.470** | **11.3×（代理高估）** |

### 结论

1. **四个符号方向全部正确**。代理的**排序能力**在本例中被验证。
2. **两个分项的失真方向相反**：mIoU 项代理低估 3.75×，面积项代理高估 20.2×，
   互相抵消后总分仍高估 11.3×。
3. **绝对值也严重失真**：代理说 G4b 面积误差 35.39%，官方只有 15.92%。代理把两模型的
   **差值**放大了 20 倍，同时把**绝对水平**抬高了 1.6–2.2 倍。
   任何"代理 74.7，还差 25 分"式的推算都是错的。
4. **有效匹配数不是官方 mIoU 的代理**：G2 匹配**少 47 个**，官方 mIoU 反而**更高**。
   官方口径 `mIoU = ΣIoU/N`，N 就是匹配对数，匹配少不直接受罚。

### 对上一轮判断的修正

上一轮我写过"整个 +5.31 增益来自面积项……代理无法裁决"。**前半句被证实，后半句要改**：
代理在本例中的方向判断是对的，它裁决不了的是**幅度**。正确说法是
**"代理能排序，不能定价"**，而不是"代理没有信息"。

---

## 3. 本轮最重要的发现：G4b 阶段是零收益

### 3.1 证据

直接对比两个 checkpoint 的 `geometry_state_dict`（39 个 tensor）：

```
G2_best  vs  G4b_best     差异 tensor = 0/39    max_abs = 0.0000e+00   ← 逐位相同
G2_best  vs  G4b_latest   差异 tensor = 39/39   max_abs = 6.6246e-03
```

元数据交叉印证：

| 文件 | epoch | best_selection_metric | best_selection_score | best_val_gt_pen |
| --- | ---: | --- | ---: | ---: |
| `g4b_gap_weight020/best` | **0** | `deployment_score_total` | 87.85586220012725 | 0.588192103988734 |
| `g2_sam2/best` | 21 | （无此字段，旧版脚本） | —— | **0.588192103988734** |
| `g4b_gap_weight020/latest` | 20 | `deployment_score_total` | 87.85586220012725 | 0.7117782527448716 |

- `G4b best` 的 `epoch = 0`，`geometry_init_checkpoint` 指向 `G2 best`——
  它是**训练开始前的初始权重**，一次梯度都没走。
- 它与 `G2 best` 的 `best_val_gt_penalized_miou` 到 **15 位有效数字完全相同**。
- `G4b latest` 里保存的 `best_selection_score` 与 `best` 文件**相同**（87.85586…），
  说明该值就是全程峰值，且峰值出现在 **epoch 0**。

**即：G4b 的部署代理选优指标认为，它的初始化点（= G2）是整条 20-epoch 曲线上的最优点。**

### 3.2 链条重写

```
G1 ──(30 ep)──> G2 ──┬── best(ep21) ──────────────> 官方 84.415   ← 用户本轮提交
                     └── latest(ep30)                  (未部署)
                          │
                          └── 作为初始化
                              ↓
                     G2best ──(20 ep)──> G4b ──┬── best(ep0)  = G2best 本身（零收益）
                                               └── latest(ep20) ──> 官方 83.945  ← 历史部署
```

**"从 G2 出发训练 G4b"这件事，官方口径的净效果是 −0.470 分。**

这与 20260911 交接文件 §6 的提问（"能否删掉 G2 阶段"）正好相反：
**G2 阶段不该删，G4b 阶段才该删。** 部署 `G2 best` 即可，链条可缩短一段。

### 3.3 为什么训练会让分数下降

部署代理分与实例 mIoU 在 G4b 训练中**方向相反**：

```
val_instance_miou_valid:  ep1 = 0.8619 → ep20 = 0.8714   （上升）
val_deployment_score_total: ep1 = 83.3  → ep20 = 82.0    （下降）
best_val_gt_penalized_miou: ep0 = 0.5882 → ep20 = 0.7118 （大幅上升）
```

拆开部署代理分，退化**全部来自面积项**（G4b-from-G1 日志同型）：

| epoch | miou | area_err | 部署代理分 |
| ---: | ---: | ---: | ---: |
| 9 | 0.8396 | 18.64% | 82.66 |
| 20 | 0.8483 | 21.30% | 81.76 |

mIoU 项涨、面积项跌。**训练越久，实例分割越好、铁素体平均面积越偏。**
这与官方口径下"面积项贡献 +0.260 > mIoU 项 +0.210"是同一件事的两面，
也与 20260911 交接文件 §4 的 GT-affinity 结论（高阈值 0.65 与锐利边界失配）互相呼应。

### 3.4 ⚠️ 一处需要核查的内部不一致

`metrics.csv` 里 `val_deployment_score_total` 的最大值是 **84.5482（epoch 2）**，
而 checkpoint 记录的 `best_selection_score` 是 **87.8559（epoch 0）**。
**两个数字既不等值也不等 epoch。** 说明保存 checkpoint 时用的分数**不是** `metrics.csv`
的那一列（可能来自 `unlabeled_monitor`，或另一套图集）。

这不影响 §3.1 的结论（权重逐位相同是硬证据），但**"到底是哪个量在选优"必须查清**，
否则任何基于该指标的训练决策都不可靠。

---

## 4. 混杂因素清单

官方两次测量之间，主办方黑盒口径发生过一次变化：**实例数上限 255 → 30000+**。

| # | 混杂 | 状态 | 证据 |
| --- | --- | --- | --- |
| A | 评测规则 255 → 30000+ | **已排除** | 两次提交实测 max = 254 / 225，均 ≤255（§4.1） |
| B | checkpoint 选择（latest vs best） | **已转化为真实差异**，见 §5 | 两者权重确实不同 |
| C | 部署路径（融合包 vs 分离配置） | **已排除** | 融合包内源权重哈希与 G2 的 reference 一致 |
| D | 选优指标不一致 | **仍存在** | G2 的 metrics 没有 deployment 列 |

### 4.1 规则变化（混杂 A）——已排除

两份 68 图预测的实例数上限，本轮**都做了实测**：

| 预测 | checkpoint | min | median | max | >255 的张数 | 总实例 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ablation_g4b_full68_latest` | G4b latest (ep20)，**= 历史提交权重** | 31 | 90 | **254** | **0** | 6178 |
| `submission_g2_20260911` | G2 best (ep21)，= 本轮提交 | 30 | 86 | 225 | 0 | 5897 |

**两次提交在旧口径（≤255）下都合规，规则变化（255 → 30000+）对二者都没有影响。**
G4b 历史提交的 max = 254，**距上限只差 1**，没有任何余量——但它没有越界。

因此 **+0.470 可以整体归因于 affinity 权重**，不需要为规则变化打折。

顺带确认这两份预测确实是不同的东西：**逐字节相同 0/68**；
G4b latest 平均每张图比 G2 多 4.1 个实例（51/68 张更多，最大差 31）。

> ⚠️ **不要用五图代理目录的分布外推 68 图。** 五图里 G4b `max=339`、G2 `max=286`,
> 各有一张 >255；五图实例密度 ≈158/图，而 68 图测试集只有 ≈87/图。若只看五图，
> 会得出"G4b 违反 255 上限"的**错误**结论。

### 4.2 选优指标（混杂 D）

- G4b 走 `latest`（ep20），G2 走 `best`（ep21）——两者**确实不是同一选择规则**。
- **G2 的 metrics.csv 只有 21 列，到 `learning_rate` 为止，没有 `val_deployment_score_total`。**
  说明 G2 训练脚本是旧版，其 best 按 Oracle 类指标选。
- 但本轮发现 `G4b best` = `G2 best`（§3.1），所以"G4b 用 best"与"G4b 用 latest"
  的差别，**恰恰就是 G2 vs G4b 的差别**——混杂 B/D 已经和主效应合并，不再是独立的混杂。

---

## 5. 末端消融：G1 → G4b

### 5.1 设计与执行

按 20260911 交接文件 §6 步骤 3：固定历史 V6 / E10a / 原训练 GT / 部署 mean+high=.65，
唯一变量是 **G4b 的初始化从 G2 改为 G1**。

配置 `config/train/affinity_geometry_g4b_from_g1.yaml`
（`_base: affinity_geometry_g4b_gap_weight020.yaml`，只改 `geometry_init_checkpoint` 与 `output_dir`）。
执行：2026-09-11 23:32 起，20 epoch，2026-09-12 **00:12:44 完成**（每 epoch 约 1:25）。

### 5.2 训练曲线（6 图内部验证集，仅诊断用）

`val_deployment_score_total`：

| epoch | 1 | **2** | 3 | 5 | 10 | 15 | 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G4b（从 G2 初始化） | 83.3 | **84.5** | 82.9 | 84.0 | 82.1 | 81.6 | 82.0 |
| G4b-from-G1 | 88.9 | **89.1** | 85.7 | 85.4 | 82.3 | 81.8 | 81.8 |

**两条曲线都是 epoch 2 达峰、之后 18 个 epoch 单调退化**，终点几乎重合
（82.00 vs 81.76）。从 G1 出发能训到与从 G2 出发相同的终点。

### 5.3 但这个消融的价值有限

因为 §3.1 已经证明：**G4b 的 `best` 权重落在初始化点上**。所以
"G1→G4b 的峰值 89.1 高于 G2→G4b 的峰值 84.5"只说明**G1 这个初始化点本身更好**，
不说明 G4b 阶段有贡献。**两个方向的 G4b 训练都是负收益。**

**本轮没有跑 G4b-from-G1 的推理，所以它没有部署口径的结论。**

---

## 6. 下一步

### P0：G4b 阶段可删——结论已成立，无需新提交

三条证据已经闭合，**不需要再花任何提交额度**：

1. 官方：`G2 best` 84.415 > `G4b latest` 83.945（+0.470）
2. 选优指标：`G4b best` = `G2 best`，20 个 epoch 无一超过起点（§3.1）
3. 规则变化对两次提交均无影响（§4.1）

**行动：把部署权重的默认值从 `g4b_gap_weight020/latest_affinity.pth`
改为 `g2_sam2/best_affinity.pth`。** 链条因此缩短一段，权重还能省一次训练。

> 若用户仍想在**同一规则下**复测 G4b 以双保险，
> `outputs/ablation_g4b_full68_latest` 已是一份可直接打包提交的 G4b 全量预测
> （与历史提交同权重，但走的是当前分离配置路径）。**这不是必需项。**

### P1：查清 §3.4 的选优不一致

- `best_selection_score` 87.8559 到底取自哪个量、哪套图？
- 为什么它与 `metrics.csv` 的 `val_deployment_score_total` 差 3.3 分且 epoch 不同？
- **在查清之前，不要再依据 `deployment_score_total` 做任何晋级决策。**

### P2：解释"训练越久面积越偏"

- 部署代理分的面积项为何随训练单调恶化，而 mIoU 项单调改善。
- 是否与 high=0.65 阈值、marker 重建步数、seal 宽度有关。
  20260911 交接文件 §4 的直线探针（gated/mean 在垂直界面峰值 0.5 < 0.65）是现成线索。
- 这可能比"删哪一段"更有价值：**若能把面积项按住，mIoU 的持续改善就能真正兑现。**

### P3：G2 自身还有空间

`G2 latest`（ep30，`66b5ceb0…`）从未部署过，而它的 `val_instance_miou_valid`(0.8645)
高于被部署的 `G2 best`（ep21, 0.8601）。值得用代理先看一眼——**注意只能看排序，不能定价**。

---

## 7. 资产与复现命令

服务器 `ssh -p 23411 root@connect.bjb1.seetacloud.com`，
工作区 `/root/autodl-tmp/segmentationv2_work`，解释器
`/root/miniconda3/envs/sam2_env/bin/python`。

```bash
# 本轮两次 G4b 全量推理（best 实为 G2 权重，latest 是历史部署权重）
python tools/run_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml \
  --checkpoint outputs/affinity_geometry_g4b_gap_weight020/{best,latest}_affinity.pth \
  --test-dir data/test --output-dir outputs/ablation_g4b_full68_{best,latest}

# 消融训练（已完成）
python train_affinity_geometry_g1.py --config config/train/affinity_geometry_g4b_from_g1.yaml
```

| 资产 | SHA256 | 说明 |
| --- | --- | --- |
| `e10a_g4b_fused.pth` | `53e1b5c6f9bf3973723c7bf4c6ce4ebd5eaeaa45210641df9de8f01e510c790f` | 历史部署包，内含 latest(ep20) |
| `stage2_v6/best_model_stage2.pth` | `c4c9827a18ecdda105056d502e5f93a92fe7fc93ada24d63886b20726c7ec557` | 两次提交共同 reference |
| `g2_sam2/best_affinity.pth` | `96f9456e4562e9e0a8fa715160582c8f8aee28deceaf32324ac38ae1570ed445` | **推荐部署**，官方 84.415 |
| `g2_sam2/latest_affinity.pth` | `66b5ceb04b066ec1f859d159a8955421e5f3ccc4fba5528557a5f7fb744b9870` | 未部署 |
| `g4b_gap_weight020/best_affinity.pth` | `b44d69b15b275311c54b3397e1649fd2cd539ff7cf3d9aae5f4dc17cb02a3c22` | **= G2 best**（逐位相同） |
| `g4b_gap_weight020/latest_affinity.pth` | `1334fdcc8472b4da67386b76b25d9099ad6bcef10b6bd9c77ae6908658b8c619` | 历史部署权重 |
| `g4b_from_g1/best_affinity.pth` | 见 `outputs/ABLATION_STATUS_20260911.md` | 本轮消融产物 |

---

## 8. 不要重复的错误

- **不要把代理分当官方分定价。** 代理 74.679 / 79.988 与官方 83.945 / 84.415 差 8–9 分，
  且差值不是常数。
- **不要用"匹配数下降"论证变差。** G2 匹配少 47 个，官方 mIoU 更高。
- **不要用五图的实例数分布外推 68 图。** 五图 ≈158 实例/图，测试集 ≈87。
- **不要把 `best` 和 `latest` 当同义词。** 本链条上 `G4b best` 是 epoch 0，
  `latest` 是 epoch 20，**相差整整一次训练**。
- **不要假定 `best_affinity.pth` 一定训练过。** 它的 `epoch` 字段可能是 0。
  **取权重前先核对 `epoch` 与 `geometry_init_checkpoint`。**
- **不要假定两次提交的 best 是同一规则选的。** G2 的 metrics 没有 deployment 列。
- **五图代理用的是训练集图像**（`train_172` 等），不是独立测试集；历史训练曝光仍不确定。
  这一点 20260911 交接文件 §4 已写过，仍然成立。

---

## 9. 交接阅读顺序

1. 本文件。
2. `docs/HANDOFF_MAINLINE_ABLATION_20260911.md`（上一轮，资产盘点与资产链）。
3. `docs/AFFINITY_GT_DEPLOYMENT_ORACLE_20260911.md`（为什么高阈值与理想 affinity 失配）。
4. `outputs/ABLATION_STATUS_20260911.md`（服务器运行状态，含消融产物哈希）。
