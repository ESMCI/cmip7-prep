# AGENTS.md

The agent instructions for this repository are in [CLAUDE.md](CLAUDE.md). The
file name is Claude-specific, but the content applies to every coding agent.

Read `CLAUDE.md` first. It has the ground rules, a short project overview, the
top gotchas, and an index of the detailed docs in `docs/llm/`. Load a
`docs/llm/` doc only when your task needs it.

Key rules from `CLAUDE.md`:
- Do not commit or push unless the user asks in the current request.
- When you change the package, update any stale references in `docs/llm/` and `CLAUDE.md` in the same change.
