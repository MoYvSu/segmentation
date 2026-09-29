# Designing questions and classes

Jev answers only the question you wrote, using only the state you sent. Most weak results
trace back to one of four causes: the question was really several questions, the state
carried a conclusion instead of evidence, the classes overlapped, or nothing in the list fit.

## One judgment per question

Ask what a knowledgeable person could decide in a second with the right context in view.

| Weak | Better |
| --- | --- |
| "Analyse this failure and decide what to do" | `is_timeout`: "Is the failure in `log` caused by a network or lock timeout?" + `is_deterministic`: "Would the failure in `log` reproduce on a clean re-run?" |
| "Is this PR good?" | Separate checks for "does `diff` match `description`", "does `diff` add or change tests", "does `diff` touch files outside the stated scope" |
| "Rate this bug" | One score for user impact, one for reproducibility, combined in your own logic |

Combine the answers yourself, with weights or rules you can explain to the user.

## Evidence, not conclusions

State is what the model judges. If you write "this diff looks safe; is it safe?", the answer
will lean toward safe and its probability tells you nothing. Send the diff.

- Put the raw text in the state: the diff hunk, the log lines, the issue body, the excerpt.
- Give each part a name and refer to it with backticks: `` `diff` ``, `` `log` ``,
  `` `items.#412` ``. Jev uses the path to know which part to judge.
- Keep the deciding evidence inside the 4000-character item limit. For a long file, send the
  relevant function, not the whole file.
- Facts the judgment depends on belong in `context`: the team list, the policy, the user's
  actual request. Your opinion does not.

## Classes that decide borderline cases

The model sees the class descriptions, not your intent. A description should say:

1. What belongs: "Existing behaviour is wrong, crashes, or differs from documentation."
2. What does not: "Not a request for new capability, not a question."
3. How it differs from its neighbour when two classes are close: "Prefer `flaky` over `bug`
   when the same test passed on a re-run."
4. A short example when the boundary is subtle.

Always include a catch-all when an item might fit nothing (`other`, `unclear`,
`needs_more_info`). Without one, Jev is forced to pick a wrong class, and the probabilities
spread without telling you why.

Descriptions may be `null` when the label is self-explanatory (`python`, `typescript`), but
a sentence rarely hurts.

## Yes/no checks

- Phrase so that a high probability means yes. "Does `log` show every test passed?" rather
  than "Are there failures?" if you plan to branch on `yes`.
- Name the condition to test, not the outcome you expect. "Does `diff` change authentication
  code?" rather than "Confirm the diff is auth-safe."
- When the boundary is subtle, add `yes` / `no` descriptions. "yes: at least one added or
  changed line inside a function that reads or sets a session"; "no: only comments,
  formatting, or unrelated files".
- A probability near 0.5 means the model is torn, not that the answer is medium. For degrees,
  use a score.

## Score levels

- Ordered lowest to highest, each describing a concrete situation that stands on its own:
  "Major: a core flow is broken for some users" rather than "Major".
- Three to six levels is the sweet spot. More levels make neighbouring ones hard to tell apart
  and lower confidence.
- `score` can land between levels (1.6). Use `level` for the nearest bucket and `score` for
  sorting.

## Ranking

- The query should describe what a hit would contain, in plain language, not keywords.
- Use paths or symbol names as ids and a meaningful excerpt as text: the signature and first
  lines of a function, the heading and first paragraph of a doc.
- Read `any_relevant` first. A rank list always has a first entry; `any_relevant` tells you
  whether that entry is an answer or just the least irrelevant.

## Thresholds

The defaults (`auto_accept 0.85` and `min_margin 0.5` for classify, `yes_at 0.75` /
`no_at 0.25` for checks, `auto_accept 0.7` for score, `relevant_at 0.5` for rank) are
conservative starting points from TypeSafe's cookbooks. Scale them with the stakes:

- Cheap, reversible actions (add a label, open a file): defaults or looser.
- Costly actions (close an issue, skip a test, merge): tighten, and keep the user in the loop
  for anything not `auto`.
- Never let a probability replace a confirmation the user would expect.

## Speculative fan-out

Questions in one call cannot see each other's answers, but they are cheap. Ask every question
that might matter and let your logic pick: "is this a bug report?", "if so, which component?",
"if so, how severe?". Ignore the answers on branches you did not take. Make a second call only
when the first answer changes what evidence you need to fetch.

## Checking your own results

When a result surprises you, inspect in this order: the exact state you sent (did the
deciding evidence make it in?), the question wording, the class descriptions, then the
probabilities. Distinguish missing evidence, an ambiguous question, a model error, and a
service error before changing anything.
