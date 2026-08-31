---
name: maintain-project-knowledge
description: Capture, validate, consolidate, retire, and promote reusable knowledge for segmentationv2. Use when work reveals a reusable experiment lesson, environment pitfall, repository behavior, or effective or failed approach, and when reviewing existing project knowledge. Do not use for ordinary progress notes, one-off mistakes, or facts directly readable from current code.
---

# Maintain Project Knowledge

Maintain a compact, evidence-backed project memory under this Skill's
`references/` directory. This is a knowledge base, not a chronological work log.

## Route the work

- For a new candidate lesson, read [references/observations.md](references/observations.md)
  and search the other active references for duplicates before recording it.
- To confirm or revise a lesson, read
  [references/validated-experience.md](references/validated-experience.md).
- To consolidate workflow-specific guidance, read
  [references/skill-rules.md](references/skill-rules.md).
- To invalidate or supersede knowledge, read
  [references/retired.md](references/retired.md).
- For a full maintenance pass, read all four references plus the repository
  `AGENTS.md`. Do not load all references for a simple capture.

## Apply the admission filter

Record an item only when it is non-obvious, supported by evidence available in
the current task, and plausibly useful in a future task. Good candidates include
experiment lessons, recurring environment behavior, repository-specific
contracts, and bounded descriptions of approaches that worked or failed.

Do not record:

- an accidental typo, one-time command mistake, or transient failure with no
  reusable cause;
- raw progress, chat history, long command output, or a result that belongs in
  an experiment report;
- a fact that the current code, configuration, or normal documentation exposes
  directly and cheaply;
- speculation presented as a rule, secrets, credentials, personal data, or
  machine-specific identifiers that are not required to reproduce the lesson.

## Use the lifecycle

Move knowledge through this path:

```text
temporary observation/error -> validated experience -> Skill-level rule
                            -> AGENTS.md project-level rule
```

1. **Observation:** record the finding, evidence, scope, reuse hypothesis, and
   verification gap in `observations.md`. One occurrence may be enough to record
   a candidate, but never enough by itself to make a project rule.
2. **Validated experience:** promote only after current evidence confirms the
   cause or invariant and its applicability limits. Move the concise result to
   `validated-experience.md`; do not leave an active duplicate behind.
3. **Skill-level rule:** promote workflow guidance to `skill-rules.md` when it
   reliably changes decisions for a class of tasks but is not a repository-wide
   hard constraint.
4. **Project-level rule:** update root `AGENTS.md` only during a deliberate rule
   review, after the item is stable, cross-task reusable, current, and globally
   applicable. Keep the detailed evidence in this Skill and add only the
   distilled instruction to `AGENTS.md`.

Promotion is a judgment, not a counter. Consider together:

- whether actual evidence verifies the claim;
- whether the situation can plausibly recur;
- whether the lesson transfers across tasks;
- whether it still matches the current repository, environment, and goals;
- whether it is truly global rather than local to one workflow.

## Maintain and retire

- Before adding an item, search for overlap and update the existing entry when
  that preserves its meaning.
- During maintenance, merge duplicates, correct overbroad claims, compress old
  evidence, and check paths, configs, environments, and assumptions for drift.
- Move invalidated or superseded knowledge to `retired.md` with the reason and
  replacement pointer. Remove it from active references and stop applying it.
- If later evidence contradicts an `AGENTS.md` rule, treat the conflict as a
  maintenance issue: report it and perform a deliberate project-rule update;
  do not silently keep using the stale rule.
- Do not create a maintenance pass merely to prolong an otherwise complete
  task. Capture only the reusable result that the task actually established.

## Entry fields

Use a short stable ID such as `YYYYMMDD-topic`. Active entries should state:

- `status` and `last_verified`;
- scope and concise finding;
- evidence or reproducible pointer;
- reuse conditions and known limits;
- source commit/config/log when relevant.

Report what was added, merged, promoted, or retired and why. Distinguish an
automatic reference update from the stricter review required for `AGENTS.md`.
