# Tool reference

Every tool takes one JSON object and returns one JSON object. The same payloads work through
MCP (`jev_classify`), the Pi extension, the OpenCode custom tool, and the CLI
(`jev-code classify --input payload.json`).

Shared conventions:

- `items` / `candidates`: arrays of `{ "id": string, "text": string }`. Ids are echoed back
  verbatim and must be unique. Texts longer than 4000 characters are truncated and the result
  carries `"truncated": true`.
- `context`: optional text or JSON object every item is judged against. Raw evidence only.
- Every result includes `model` and, when the API reports it, `usage`
  (`{ input_tokens, output_tokens }`). `model` echoes the id the host reports: `jev-latest` on
  TypeSafe, `typesafe/jev-1.13` through OpenRouter, `typesafe-ai/jev` through Vercel AI Gateway.
- Thresholds are optional inputs; the effective values are echoed under `thresholds`.

## jev_classify

Label each item with exactly one class. One request, one Choice question per item.

Input:

```json
{
  "instructions": "Route each GitHub issue to the team that owns it.",
  "items": [
    { "id": "#412", "text": "Login page shows a blank screen after the SSO redirect on Safari" },
    { "id": "#415", "text": "Please add CSV export to the billing report" }
  ],
  "classes": {
    "frontend": "Rendering, layout, browser-specific behaviour, client-side state",
    "backend": "APIs, jobs, database, authentication flows on the server",
    "feature_request": "Asks for something new rather than reporting broken behaviour",
    "other": "Does not fit any team above, or too vague to route"
  },
  "context": "Teams: frontend owns the React app, backend owns the Rails API.",
  "auto_accept": 0.85,
  "min_margin": 0.5
}
```

Output:

```json
{
  "summary": { "items": 2, "auto": 2, "review": 0, "by_label": { "frontend": 1, "feature_request": 1 } },
  "results": [
    {
      "id": "#412", "label": "frontend", "probability": 0.91, "margin": 0.84, "confidence": 0.87,
      "decision": "auto",
      "probabilities": { "frontend": 0.91, "backend": 0.07, "feature_request": 0.01, "other": 0.01 }
    },
    { "id": "#415", "label": "feature_request", "probability": 0.97, "margin": 0.95, "confidence": 0.95, "decision": "auto", "probabilities": { "...": 0 } }
  ],
  "thresholds": { "auto_accept": 0.85, "min_margin": 0.5 },
  "model": "jev-latest",
  "usage": { "input_tokens": 410, "output_tokens": 24 }
}
```

`decision` is `auto` only when the top probability is at or above `auto_accept` **and** the gap
to the runner-up is at or above `min_margin`. Limits: 64 items, 250 classes.

## jev_check

Independent yes/no questions about one piece of evidence. One request, one Noul per check.

Input:

```json
{
  "state": {
    "pr_description": "Adds retry to the payment webhook. No behaviour change for successful calls.",
    "diff": "--- a/webhooks.py\n+++ b/webhooks.py\n@@ ... @@\n+    for attempt in range(3):\n+        try:\n+            return post(url, body)\n+        except Timeout:\n+            continue\n+    raise WebhookFailed()",
    "test_log": "12 passed, 0 failed"
  },
  "checks": {
    "matches_description": "Does `diff` do what `pr_description` says, and nothing else?",
    "touches_payments": "Does `diff` change code on the payment path?",
    "tests_green": { "question": "Does `test_log` show that every test passed?", "yes": "Zero failures reported", "no": "At least one failure, error, or skipped-as-failed entry" }
  },
  "yes_at": 0.75,
  "no_at": 0.25
}
```

Output:

```json
{
  "summary": { "checks": 3, "yes": 3, "no": 0, "uncertain": 0 },
  "results": [
    { "id": "matches_description", "question": "Does `diff` do what `pr_description` says, and nothing else?", "probability": 0.88, "verdict": "yes" },
    { "id": "touches_payments", "question": "...", "probability": 0.97, "verdict": "yes" },
    { "id": "tests_green", "question": "...", "probability": 0.99, "verdict": "yes" }
  ],
  "thresholds": { "yes_at": 0.75, "no_at": 0.25 },
  "model": "jev-latest"
}
```

`verdict` is `yes` at or above `yes_at`, `no` at or below `no_at`, otherwise `uncertain`.
Limit: 64 checks.

## jev_score

Rate items on one ordered scale. One request, one Score question per item.

Input:

```json
{
  "instructions": "How severe is this test failure for shipping today?",
  "levels": [
    "Cosmetic or flaky: no product impact, safe to ignore for now",
    "Minor: real bug with an easy workaround",
    "Major: a core flow is broken for some users",
    "Critical: data loss, security, or the app does not start"
  ],
  "items": [
    { "id": "test_checkout_total", "text": "AssertionError: expected 19.99 got 19.989999" },
    { "id": "test_login_sso", "text": "TimeoutError: SSO callback never received (retried 3x)" }
  ],
  "auto_accept": 0.7
}
```

Output:

```json
{
  "summary": { "items": 2, "auto": 1, "review": 1, "mean_score": 1.62, "by_level": { "1": 1, "2": 1 } },
  "legend": { "0": "Cosmetic or flaky...", "1": "Minor...", "2": "Major...", "3": "Critical..." },
  "results": [
    { "id": "test_checkout_total", "score": 1.08, "level": 1, "label": "Minor: real bug with an easy workaround", "confidence": 0.81, "decision": "auto", "probabilities": { "0": 0.1, "1": 0.74, "2": 0.14, "3": 0.02 } },
    { "id": "test_login_sso", "score": 2.15, "level": 2, "label": "Major: a core flow is broken for some users", "confidence": 0.55, "decision": "review" }
  ],
  "thresholds": { "auto_accept": 0.7 },
  "model": "jev-latest"
}
```

`score` is a position from `0` to `levels.length - 1` and may fall between levels; `level`
is the nearest integer. Limits: 64 items, 20 levels.

## jev_rank

Rank candidates by relevance to a query. One request, one Noul per candidate plus one
`any_relevant` Noul.

Input:

```json
{
  "query": "where is the retry policy for outbound HTTP calls configured?",
  "candidates": [
    { "id": "src/http/client.ts", "text": "export const client = createClient({ retries: env.HTTP_RETRIES ?? 3, backoff: 'exponential' })" },
    { "id": "src/http/routes.ts", "text": "router.get('/health', () => ok())" },
    { "id": "docs/runbook.md", "text": "On-call rotation and escalation contacts." }
  ],
  "top_k": 2,
  "relevant_at": 0.5
}
```

Output:

```json
{
  "any_relevant": 0.96,
  "summary": { "candidates": 3, "relevant": 1, "returned": 2 },
  "ranked": [
    { "rank": 1, "id": "src/http/client.ts", "relevance": 0.94, "relevant": true },
    { "rank": 2, "id": "src/http/routes.ts", "relevance": 0.06, "relevant": false }
  ],
  "thresholds": { "relevant_at": 0.5 },
  "model": "jev-latest"
}
```

Check `any_relevant` before trusting rank 1: a sorted list always has a top entry, even when
nothing fits. Limit: 250 candidates.

## jev_ask

Raw System One call. Use it for mixed question types over one state or for speculative
fan-out (ask every question that might matter; use the answers you need).

Input:

```json
{
  "state": { "message": "Our API started returning 500s twenty minutes ago and orders are failing." },
  "questions": {
    "department": { "type": "choice", "instructions": "Which team should handle `message`?", "criteria": { "billing": "Payments, refunds, invoices", "technical": "Bugs, outages, integrations", "sales": "Pricing, plans" } },
    "is_urgent": { "type": "noul", "instructions": "Does `message` convey urgency or time pressure?" },
    "frustration": { "type": "score", "instructions": "How frustrated does the author of `message` appear?", "criteria": ["Calm, factual", "Concerned but civil", "Angry, strong language"] }
  }
}
```

Output is the API response verbatim:

```json
{
  "model": "jev-latest",
  "answers": {
    "department": { "type": "choice", "choice": "technical", "probabilities": { "billing": 0.03, "technical": 0.96, "sales": 0.01 }, "confidence": 0.93 },
    "is_urgent": { "type": "noul", "noul": 0.99 },
    "frustration": { "type": "score", "score": 1.2, "legend": { "0": "Calm, factual", "1": "Concerned but civil", "2": "Angry, strong language" }, "confidence": 0.7 }
  },
  "usage": { "input_tokens": 300, "output_tokens": 40 }
}
```

Criteria shapes: `choice` needs an object with at least 2 options; `score` needs an array of at
least 2 levels; `noul` optionally takes `{ "true": ..., "false": ... }`. The optional `model`
overrides the default for one call and must use the host's id format: `jev-1.13` on TypeSafe or
OpenRouter, `typesafe-ai/jev` on Vercel AI Gateway.
