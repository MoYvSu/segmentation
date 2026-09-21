# Validated experience

## 20260914-confidential-artifacts-across-branches

- status: validated
- last_verified: 2026-09-14
- scope: 赛题数据与派生产物的Git暂存、历史、跨分支传播
- finding: 根目录忽略outputs不等于忽略output，分支间忽略规则也不自动同步；退出跟踪只改变最新树，
  不能清除历史。必须检查完整待推送祖先，并防止内部快照或旧克隆重新带回产物。
- evidence: 本次远程只读镜像确认两个旧分支仍跟踪348/366个output文件，另有历史标注数组与可视化；
  用户未提供新的图片外泄链接，不能把担忧写成已确认新外泄。`tests/test_confidential_artifacts_guard.py`
  五项测试覆盖暂存路径、改名/内嵌图片、干净最新树与污染祖先、内部引用、跨分支安装及既有hook保护。
- reuse_conditions: 按用户明确的最高级项目保密要求，所有分支及新克隆遵守根AGENTS.md；使用
  `tools/confidential_artifacts_guard.py`和`docs/CONFIDENTIAL_ARTIFACTS.md`，安全公开脚本与本地产物分开存放。
- limits: Git hooks不代替内容审阅，不控制浏览器上传或未安装防护的克隆；分支历史清理不等于GitHub缓存/PR
  或第三方副本全部清除。原始私有镜像、恢复bundle和refs/codex快照不得推送。

This file contains evidence-backed project knowledge with explicit scope and
limits. Entries here may guide future tasks but are not automatically global
rules.

Promote an observation here only after its cause or invariant has been checked
against actual code, output, configuration, environment, or experiment
evidence. Remove the active copy from `observations.md` when promoting it.

Use this shape:

```markdown
## YYYYMMDD-topic

- status: validated
- last_verified: YYYY-MM-DD
- scope: where the lesson applies
- finding: concise reusable conclusion
- evidence: reproducible pointer and relevant version
- reuse_conditions: when to apply it
- limits: when not to apply it
```

## 20260831-deployment-proxy-area-merge-tradeoff

- status: validated
- last_verified: 2026-08-31
- scope: affinity/watershed experiments selected by the six-image deployment proxy
- finding: A higher total deployment proxy score can be dominated by correcting ferrite mean area through fewer, larger predicted instances while GT coverage becomes worse. Treat the total as a multi-objective proxy, not evidence of geometry improvement by itself.
- evidence: G7 epoch 14 at fixed `high=0.65` raised the six-image proxy `77.5258 -> 81.3535` and improved ferrite area error `0.2862 -> 0.2135`, while predictions fell `1302 -> 1243`, valid matches fell `784 -> 747`, and GT-penalized IoU fell `0.7217 -> 0.6908`. Its fixed test A/B against G4b+E10a under the same inference/fusion protocol reduced total instances `6178 -> 5884` on 51 of 68 images, almost entirely through ferrite count `4315 -> 3989`; the ten-image monitor also showed a monotonic short-affinity increase. This independently repeats the G5 under-segmentation pattern documented in `docs/AFFINITY_INFERENCE_AUDIT_20260826.md`.
- reuse_conditions: When an affinity, boundary, fusion, or watershed candidate gains aggregate deployment-proxy score, decompose the gain and run a fixed-protocol test A/B before calling it a geometry improvement or replacing the stable path.
- limits: Unlabeled test counts cannot prove which individual merges are wrong, and G7 has no official black-box score. The evidence validates the compensating-error risk and review procedure, not an unconditional rule that fewer instances are worse.

## 20260831-inference-artifact-download-scope

- status: validated
- last_verified: 2026-08-31
- scope: downloading inference and instance-visualization artifacts for local review
- finding: Default local deliveries should go under the repository-root `downloads/` directory and contain only the raw instance-ID maps (`*_inst.png`), instance-script color visualizations (`*_inst_color.png`), class mappings (`*_class.json`), and one batch `submission_manifest.json`. Intermediate masks, boundary maps, marker-boundary maps, and class-confidence JSON are not part of the default review bundle.
- evidence: A full G7 output download contained 477 files because inference emitted seven per-image artifacts for 68 images plus one manifest (`68 * 7 + 1`). The user confirmed that the review workflow needs only the instance map, scripted visualization, and original class JSON, with `downloads/` as the preferred destination.
- reuse_conditions: Apply whenever inference outputs are downloaded without an explicitly requested artifact list or destination.
- limits: Preserve or download extra diagnostics when the user explicitly asks for them or when a named debugging task requires those exact intermediate maps; do not delete complete remote outputs merely because the local bundle is filtered.
