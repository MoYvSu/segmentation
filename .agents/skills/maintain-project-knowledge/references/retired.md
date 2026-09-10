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

## 20260910-affinity-mean-ceiling

- status: retired
- retired_on: 2026-09-10
- previous_scope: 从特定直线的 mean 融合峰值 .50，推断 clean60/e110 主要受读出错配限制。
- reason: .50 只适用于特定界面方向，不是普遍上限。真实 mean/top2 对照中，新版匹配仅
  792→806，mIoU .8408→.8335，低于既有 mean/high=.59 的820匹配；选定漏口多个原始方向
  同时偏弱，且 top2 后种子仍穿界。读出有影响，但不足以作为首因或自动改配置的依据。
- replacement: observations.md 的 20260908-clean60-affinity-calibration；详细证据见
  docs/AFFINITY_CLEAN60_DIAGNOSIS_20260910.md。

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
