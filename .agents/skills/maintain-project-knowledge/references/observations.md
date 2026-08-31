# Candidate observations

This file holds potentially reusable findings that still need validation or
scope clarification. It is not a chronological task log.

Before adding an entry, search active and retired references for the same
lesson. Record only candidates that pass the admission filter in `SKILL.md`.

Use this shape:

```markdown
## YYYYMMDD-topic

- status: observation
- last_verified: YYYY-MM-DD
- scope: affected workflow, module, environment, or experiment family
- finding: concise statement
- evidence: current reproducible evidence or pointer
- reuse_hypothesis: why this may matter in a future task
- verification_gap: what remains uncertain
- limits: known exceptions or counterevidence
```

## 20260831-fused-deployment-contract

- status: observation
- last_verified: 2026-08-31
- scope: 83.94 E10a + G4b fused checkpoint 推理、提交校验与打包
- finding: 正式提交推理应读取 fused checkpoint 内冻结的 fusion/inference 参数；外部 YAML
  只负责路径和模型构建环境。源三 checkpoint 入口保留作实验与权重对照，不应作为正式提交入口。
- evidence: `models/fused_deployment.py::frozen_deployment_contract`、
  `tools/run_fused_affinity_submission.py` 与 `tests/test_submission_package.py`；本地已验证契约复制、
  格式校验及确定性 ZIP，`docs/REPRODUCTION.md` 记录当前边界。
- reuse_hypothesis: 可避免配置后续编辑使同一组合权重产生不同后处理结果，也能防止复现说明再次
  混淆部署复现、源权重追溯和完整重训练。
- verification_gap: 本地缺少正式 fused/source checkpoint；需在服务器用正式权重完成一次全测试集
  推理、自动校验和打包，确认旧 v1 bundle 的输入尺寸回退与最终预测保持一致。
- limits: 新 bundle 会显式保存输入尺寸和构建配置哈希；旧 v1 bundle 没有输入尺寸字段时仍从构建
  YAML 回退读取，因此在重建组合包前尚未完全消除这一项外部配置依赖。
