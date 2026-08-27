# E7a 实例语义投票实验（2026-08-27）

## 目标与固定项

E7a 只处理实例类别错配，不改变 G4b affinity、`high=0.65`、`low=0.45/8px`、
seal2、分水岭、最小面积或 255 ID 上限。三组输出的 68 张实例图逐像素完全一致，均为
6178 个实例；差异只来自实例的 ferrite/pearlite 类别。

三臂：

- hard：整实例二值多数票；
- adaptive-core：按实例 distance transform 选择约 40% 自适应核心，语义概率做距离加权
  robust mean；细长、非凸和贴边实例均有定义；
- core+Lab：仅当核心语义分数位于 0.35--0.65 时，将每图 Otsu Lab L* 先验以 0.25 权重
  融合；双峰分离度不足 1.0 时禁用。

## 六张人工验证图

| 投票 | 有效匹配 | 有效 mIoU | GT 惩罚 mIoU | ferrite 数 | ferrite 面积误差 | 代理总分 |
|---|---:|---:|---:|---:|---:|---:|
| hard | 782 | 0.83684 | 0.71992 | 739 | 25.32% | 79.180 |
| adaptive-core | 783 | 0.83680 | 0.72081 | 750 | 26.22% | 78.731 |
| core+Lab | 784 | 0.83654 | 0.72151 | 771 | 28.06% | 77.796 |

核心与 Lab 均让有效匹配和 GT 覆盖出现极小正信号，但持续把更多实例归为 ferrite，导致
人工验证集 ferrite 平均面积进一步变小。由于验证 GT 稀疏且多边形边界可能不精确，不能
只按代理总分否决核心思路；但 core+Lab 的单向偏置过强，不能晋级默认提交。

## 68 张测试集无 GT 审计

| 比较 | 翻转总数 | pearlite→ferrite | ferrite→pearlite | 翻转面积中位数 |
|---|---:|---:|---:|---:|
| hard→core | 207 | 161 | 46 | 1484 px |
| core→core+Lab | 122 | 120 | 2 | 1324 px |
| hard→core+Lab | 287 | 260 | 27 | 1521 px |

Lab 在 6178 个实例中参与 499 次，其中真正越过 0.5 的 122 次几乎都指向 ferrite。这符合
“暗边把亮相核心误判成 pearlite”的假设，但也说明当前 Otsu prior 需要更严格的启用条件或
更低权重。几何一致性审计无任何 mismatch。

## 工程实现

- `utils/semantic_vote.py`：自适应核心、robust mean、每图 Lab prior 和类别融合；
- `utils/post_process.py`：类别审计复用首次投票结果，不重复计算 distance transform；
- distance transform 只在实例包围盒执行，避免每个实例处理整张高分辨率图；
- 每图 `*_class_confidence.json` 记录核心像素、语义分数、颜色分数、Lab 分离度和是否启用；
- submission manifest 记录全部语义投票参数。

配置：

- `config/inference/final_affinity_g4b_high065.yaml`：hard 基线；
- `config/experiments/affinity_g4b_high065_vote_adaptive_core.yaml`；
- `config/experiments/affinity_g4b_high065_vote_adaptive_core_lab.yaml`。

## 判定与下一步

1. 默认仍为 G4b high0.65 + hard；adaptive-core 保留为目检/提交 challenger。
2. core+Lab 当前不晋级；下一版若继续，只允许更窄的不确定区间、更低融合权重，并增加
   “core 与整实例差异/黑边占比”条件，不能仅凭亮度双峰翻转。
3. 进入 E7b 语义专项训练：geometry 全冻结，训练 semantic decoder 与低学习率后层 LoRA；
   增加按实例等权的 adaptive-core loss、针对小实例的局部暗边增强，以及高置信光度一致性。
4. 上采样 `align_corners` 对齐作为独立诊断，不能与 E7b 同轮修改。
