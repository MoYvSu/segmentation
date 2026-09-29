# CLI fallback

When the `jev_*` tools are not registered in the current harness, the same operations are
available from bash. The CLI reads the API key from the environment (`TYPESAFE_API_KEY`,
`OPENROUTER_API_KEY`, or `AI_GATEWAY_API_KEY`; the key's prefix picks the host).

```bash
jev-code doctor                       # is the key set, which harnesses are wired
jev-code classify --input payload.json
jev-code check    --json '{"state":"...","checks":{"ok":"..."}}'
cat payload.json | jev-code score     # stdin works when piped
jev-code rank --input candidates.json --pretty
jev-code ask --input questions.json
```

Without a global install, prefix with npx: `npx -y @french-castle/jev-code classify ...`.

Payload shapes are identical to the tools; see [tools.md](tools.md). Output is JSON on stdout
(compact when piped, pretty on a terminal or with `--pretty`). Errors go to stderr with exit
code 2 for input or configuration problems and 1 for API failures.

To wire the tools permanently, ask the user to run `jev-code setup` (or
`npx -y @french-castle/jev-code setup`); it installs this skill and registers the tool in
every detected harness.
