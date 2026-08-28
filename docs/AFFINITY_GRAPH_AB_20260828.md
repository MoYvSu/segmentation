# E10a 单语义 + 方向 affinity 图切分 A/B

## 目的

固定 E10a 为唯一语义来源、固定 G4b checkpoint，仅替换实例几何解码，判断 8 通道
same-instance affinity 是否应直接用于图连通切分，而不是先压成一张 boundary 再分水岭。
本实验完全 GT-free，只在 10 张预先选定的困难测试图上目检，不用于选模打分。

## 阈值含义

图阈值作用于 `P(same instance)`，与融合边界阈值方向相反。当前 boundary `0.65` 大致对应
affinity 连接阈值 `1 - 0.65 = 0.35`，因此扫描短程通道 `0.30/0.35/0.40`；距离 2/4 通道
分别固定为更严格的 `0.55/0.65`，降低长边跨越细小断口造成合并的风险。

初版把每个低 affinity 晶界像素也当作独立图节点，`test_003` 在 428×512 网格产生
32,193 个原始分量，逐片合并既慢又无物理意义。修正版只保留达到面积要求的 affinity
核心，其余晶界带按最近核心做确定性 Voronoi 回填，保证每个像素唯一归属且实例 ID ≤255。

## 结果

| 几何臂 | 10 图实例总数 | 相对 E10a watershed | 结论 |
|---|---:|---:|---|
| E10a + G4b watershed | 1350 | 基线 | 保留 |
| graph short 0.30 | 1603 | +18.7% | 召回过激，碎裂与大块合并并存 |
| graph short 0.40 | 1482 | +9.8% | 数量较温和，但仍有块状漂移与错误合并 |

所有图切分臂均满足像素唯一归属，`unassigned_pixels=0`。但新增实例数不能等同于减少欠分割：
`test_024` 等图仍存在大实例合并，同时其他区域产生碎片；512 affinity 网格的最近核心回填还
带来可见块状边界。因此直接 thresholded connected-components 不替代当前标记分水岭，
`0.35` 单图臂因明显过碎提前淘汰。

报告位于服务器
`outputs/experiments/e10a_graph_ab_v2/graph_ab_summary.json`，本地目检总览位于忽略跟踪的
`downloads/e10a_graph_ab_v2/overview030.png` 与 `overview040.png`。

## 下一步约束

继续固定 E10a 单语义候选和 G4b watershed 主体。几何改进应作为局部补缝而非全量替换：

1. 只在当前 marker boundary 的疑似泄漏处读取方向 affinity，补齐短而连续的弱边；
2. 不对整图做 Voronoi/连通分量重划分；
3. 保持原始 boundary elevation、seal2、面积过滤和 255 上限不变；
4. 首轮只扫一个补缝置信门槛和最大生长长度，避免重新引入复杂自适应后处理。
