# Skill-level rules

This file holds compact rules for recurring workflows that do not belong in the
repository-wide `AGENTS.md`.

A rule must be derived from validated experience, affect future decisions, and
state its scope and exceptions. Keep the supporting evidence in
`validated-experience.md`; do not duplicate the full history here.

Use this shape:

```markdown
## Rule name

- status: active
- last_verified: YYYY-MM-DD
- applies_to: workflow or task class
- rule: imperative guidance
- evidence: validated entry ID
- exceptions: explicit boundaries
```

## Decompose geometry proxy gains before promotion

- status: active
- last_verified: 2026-08-31
- applies_to: affinity, boundary, fusion, and watershed experiment selection
- rule: Before describing an aggregate deployment-proxy gain as improved geometry, compare its mIoU and ferrite-area components, predicted and valid-match counts, GT-penalized IoU, and a fixed-threshold test A/B. Keep the stable path when the gain is dominated by broad instance-count reduction until visual or black-box evidence resolves whether the merges are correct.
- evidence: `20260831-deployment-proxy-area-merge-tradeoff`
- exceptions: A count reduction with direct labeled or official evidence of corrected over-segmentation may be promoted; raw unlabeled instance count alone is never a quality metric.

## Keep local inference downloads minimal

- status: active
- last_verified: 2026-08-31
- applies_to: inference-result and visualization downloads
- rule: Unless the user specifies otherwise, download to the repository-root `downloads/` directory and include only `*_inst.png`, `*_inst_color.png`, `*_class.json`, and `submission_manifest.json`. Do not include masks, boundary or marker-boundary maps, or class-confidence JSON by default.
- evidence: `20260831-inference-artifact-download-scope`
- exceptions: Include additional artifacts only when explicitly requested or necessary for the stated diagnostic task; keep the complete server output intact.
