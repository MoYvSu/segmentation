# Building Jev into the user's code

The other references cover using the `jev_*` tools inside your own session. This one is for the
moment the *user's application* needs a decision Jev can make: a support router, a moderation
gate, a document classifier, a relevance ranker, a verification step. Prototype the questions
with `jev_ask` or the CLI, then port the winning payload into the user's stack with the official
SDK. Never shell out to `jev-code` from application code.

This guide adapts TypeSafe's own agent skill ([typesafe-ai/skills](https://github.com/typesafe-ai/skills),
MIT) to the coding-agent setting; see the notice at the end.

## Read the live docs first

**The live TypeSafe docs are the source of truth.** Read them as part of the task rather than
relying on this file, which only gives direction.

- Start from the [documentation index](https://docs.typesafe.ai/llms.txt) and read targeted
  pages. Append `.md` to any page path for Markdown, for example
  `https://docs.typesafe.ai/concepts/state.md`. Resolve relative links against
  `https://docs.typesafe.ai`.
- Before writing an integration, read the current API or SDK page and the page for each
  primitive you use. For a new workflow, open the closest cookbook: it often shows a better
  decomposition than a generic classifier.
- If live access is unavailable, use installed SDK types or local docs, say so, and avoid
  inventing version-dependent details.

| Task | Start here |
| --- | --- |
| Understand the programming model | [System One](https://docs.typesafe.ai/concepts/system-one.md), [how to build with it](https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md) |
| Explore what to build | [Use-case map](https://docs.typesafe.ai/concepts/use-case-map.md), then cookbooks from the index |
| Prepare inputs and questions | [State](https://docs.typesafe.ai/concepts/state.md), [primitives](https://docs.typesafe.ai/primitives.md), then the chosen primitive's page |
| Handle uncertainty | [Confidence](https://docs.typesafe.ai/confidence.md), [confidence-gated routing](https://docs.typesafe.ai/patterns/confidence-routing.md) |
| Write the code | [HTTP API](https://docs.typesafe.ai/api.md), [Python SDK](https://docs.typesafe.ai/sdk/python.md), [JavaScript SDK](https://docs.typesafe.ai/sdk/javascript.md) |
| Update an older integration | [Migration guide](https://docs.typesafe.ai/migrating-to-v1.md) and the installed SDK's reference |

## Find the useful shape

Start from the behaviour the user wants: what will the application show, select, change, or
hand off? Work backward to the judgments it needs. Keep known rules, calculations, exact lookups,
and execution in code; add Jev where semantic understanding is needed. Preserve the user's stack
and scope.

Classification is the obvious pattern, not the only one. Combine primitives around the goal:

- **Route and fill known arguments.** One request can select a handler and its typed
  parameters. Ask the branch-specific questions up front and consume only the relevant answers.
  See [function calling](https://docs.typesafe.ai/cookbooks/function_calling.md) and
  [speculative fan-out](https://docs.typesafe.ai/patterns/fan-out.md).
- **Select instead of generate.** Find candidate values or spans in code (regex, parsing), let a
  judgment pick the intended one, then copy or normalise it. The model never writes a value.
  See [pre-parsed value extraction](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook.md),
  [date extraction](https://docs.typesafe.ai/cookbooks/date_extraction_cookbook.md), and
  [structure recovery](https://docs.typesafe.ai/cookbooks/autoformat.md).
- **Find and judge evidence.** Retrieve candidates, judge their relevance, select context.
  See [re-ranking](https://docs.typesafe.ai/cookbooks/rerank_typesafe.md),
  [line-by-line search](https://docs.typesafe.ai/cookbooks/semantic_find.md),
  [classifying RAG passages](https://docs.typesafe.ai/cookbooks/classifying_rag_passages.md), and
  [hierarchical classification](https://docs.typesafe.ai/cookbooks/hierarchical_classification.md).
- **Turn judgments into reusable data.** Score dimensions once, then let code or user controls
  change weights, thresholds, and views without re-running inference. See
  [composite scoring](https://docs.typesafe.ai/patterns/composite-scoring.md) and
  [feature discovery](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery.md).
- **Verify and escalate.** Check claims or fields against evidence; send uncertain or failing
  cases to a person or a reasoning model. See
  [citation checks](https://docs.typesafe.ai/cookbooks/citation_check.md),
  [LLM guardrails](https://docs.typesafe.ai/cookbooks/llm_guardrails.md), and the
  [extraction cascade](https://docs.typesafe.ai/cookbooks/sde_cascade.md).
- **Respond to changing state.** Code retains goals and observations; fresh judgments guide the
  next bounded step. Keep inferred state separate from observed facts and check freshness before
  applying a result to a changed situation.

For an open-ended request, offer the two or three directions that best serve the goal and
recommend one. For a concrete request, pick the pattern and build.

## Design the judgments

Choose by what the answer means, then read that primitive's page:

| Need | Primitive | Important distinction |
| --- | --- | --- |
| One of a defined set | [Choice](https://docs.typesafe.ai/primitives/choice.md) | Picks one option; the distribution compares competing options |
| Whether a condition holds | [Noul](https://docs.typesafe.ai/primitives/noul.md) | Probability of yes, no separate confidence; one per label when several may apply |
| Degree along a described dimension | [Score](https://docs.typesafe.ai/primitives/score.md) | Probability-weighted position on ordered levels; comparable per-item Scores give a graded ranking |

Give each question enough **state**: source text, identities, relationships, policies, current
facts. Prefer named JSON fields when context has several parts and reference them with backticked
paths such as `` `ticket.messages[0].text` ``. Put the judgment in **instructions** and the
possible answers in **criteria**. Ids are for code and are not sent to the model; make each
question complete on its own.

Ask one narrow, coherent judgment per question. Split independently useful dimensions without
destroying the relationship being judged. Strings work for simple questions; use objects or arrays
for definitions, contrasts, exclusions, and examples. Score levels must describe concrete
situations. Include a no-match outcome when nothing may fit, and a separate presence judgment when
it is independently useful. For source-value selection, check that the intended value is among
the candidates: the model cannot choose an omitted one.

## Compose and verify

- **Ask independent questions over the same state together**, including speculative ones. They
  run in parallel and cannot see each other's answers. Make a second request only when an answer
  is needed to fetch evidence, build new state, or choose the next options.
- Use probabilities and confidence to guide behaviour, with thresholds evaluated on the user's
  data and stakes. Choice and Score confidence summarises distribution concentration, not
  workflow correctness or permission to act. A Noul near 0.5 means similar probability for yes
  and no, not medium intensity. Ignore uncertainty on branches the code did not take.
- Keep policy explicit and raw judgments reusable: weighted scores for compensating
  preferences, separate conditions for "any serious violation" rules. Changing a weight or a
  display filter should not require re-running inference.
- Typed output guarantees the interface, not truth. Validate on representative cases and inspect
  failures by looking at the exact state, questions, answers, and composition. Separate missing
  evidence, model errors, code errors, and service failures.
- Keep API credentials server-side in web apps. Read the key from the environment
  (`TYPESAFE_API_KEY`), never from source.

## SDK quick reference

Python (`pip install typesafe-sdk`, Python 3.10+):

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient() as client:  # reads TYPESAFE_API_KEY, defaults to jev-latest
    result = client.system_one(
        state={"message": message},
        questions={
            "department": Choice(instructions="Which team should handle `message`?",
                                 criteria={"billing": "Payments, refunds", "technical": "Bugs, outages", "other": None}),
            "is_urgent": Noul(instructions="Does `message` convey urgency?"),
            "frustration": Score(instructions="How frustrated is the author of `message`?",
                                 criteria=["Calm", "Concerned but civil", "Angry"]),
        },
    )
answer = result.choices["department"]
if answer.confidence < 0.5:
    route_to_human(message)
```

JavaScript (`npm install @typesafe-ai/sdk`, Node 20+):

```ts
import { choice, noul, TypeSafeClient } from "@typesafe-ai/sdk";

const client = new TypeSafeClient();
const { answers } = await client.systemOne({
  state: { message },
  questions: {
    department: choice("Which team should handle `message`?", { billing: "Payments, refunds", technical: "Bugs, outages", other: null }),
    is_urgent: noul("Does `message` convey urgency?"),
  },
});
if (answers.department.confidence >= 0.8) route(answers.department.choice);
```

Both SDKs expose retries, typed answers, and the raw HTTP response; read the SDK page for the
installed version before relying on a detail.

---

*Adapted from the TypeSafe agent skill, Copyright (c) 2026 TypeSafe AI, MIT License. Permission
is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, subject to the condition that the above copyright notice and this
permission notice are included in all copies or substantial portions of the Software. THE
SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.*
