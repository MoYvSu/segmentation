# Retired or superseded knowledge

Entries in this file are historical context only and must not be applied as
current guidance. Keep a retired item only when its invalidation or replacement
will prevent a likely future mistake; otherwise delete it during compaction.

Use this shape:

```markdown
## YYYYMMDD-topic

- status: retired
- retired_on: YYYY-MM-DD
- previous_scope: where it used to apply
- reason: evidence that invalidated or superseded it
- replacement: active entry or documentation pointer, if any
```

## 20260831-center-offset-flow

- status: retired
- retired_on: 2026-08-31
- previous_scope: center heatmap、naive flow 与 global center-offset 实例几何训练/部署
- reason: center 监督破坏共享边界表征；naive EDT flow 产生多吸引域；tile-local offset 重复中心并使面积项崩溃；学习式 offset 没有取得替换 G4b 的完整部署证据。专属入口、配置、损失、增强和测试已移出主工作区。
- replacement: 当前实例几何使用 E10a + G4b affinity + 固定 watershed；数值证据、已删除路径和恢复方式见 `docs/RETIRED_EXPERIMENTS.md`。
- source: 清理前源码 `699dd8cf4f5e131c38e54b01f45ec41b7342f88b`。

## 20260831-direct-affinity-graph

- status: retired
- retired_on: 2026-08-31
- previous_scope: graph-v1 连通分量回填与 graph-v2 affinity 最大生成森林部署
- reason: graph-v1 黑盒总分 `83.17` 低于 E10a watershed 的 `83.94`；graph-v2 产生不自然笔直边界。部署工具、专属回填函数和对应测试已移出主工作区。
- replacement: G4b/G7 继续保留 affinity target、Oracle 连通重建和恢复审计，但最终部署固定使用 watershed；证据见 `docs/AFFINITY_GRAPH_AB_20260828.md`。
- source: 清理前源码 `699dd8cf4f5e131c38e54b01f45ec41b7342f88b`。
