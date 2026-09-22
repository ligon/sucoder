# SuCoder — project notes

These rules are SuCoder-repo-specific. They live here (outside the gitnexus
block) instead of in `~/.sucoder/system_prompt.org`, which is the system-wide
default for all mirrors and is meant to stay project-agnostic.

- If introducing new configuration, update `README.org` and `config.example.yaml`
  so the example stays in sync with the parser.
- Test the Python code paths with `pytest` (run from the SuCoder mirror root);
  add targeted tests for behavior changes.
- If `.sucoder/handoff.org` exists, read it completely at session start and
  treat a `READY` handoff as the current operational task.
- **Never put a closing keyword next to an issue number anywhere GitHub
  parses: commit messages *and PR bodies*.** GitHub scans every commit merged
  to the default branch, and the body of every PR merged into it, for `close`,
  `closes`, `fixes`, `resolves` (and their variants) followed by `#N`, and
  closes issue N. It does not read the surrounding English. Issue 14 was closed
  three times this way: by a stale `Closes #14` left in a commit body after the
  scope was cut, by a merge commit that said "does not close #14", and by a PR
  body that said "Does not close #14". The rule is about the tokens, not the
  intent: write `issue 14` when you mean to discuss it, and write `Closes #N`
  only when the PR really closes it.
- **GitNexus is NOT read-only — do not conclude otherwise from the hook spam.**
  Agents repeatedly misread the `Cannot execute write operations in a read-only
  database` FTS messages as "the index can't be refreshed from here." Those are
  a *non-fatal* full-text-index auto-ensure inside the MCP server's *query* path
  (note the "Will retry on next query"); they never block queries or
  re-indexing. The `gitnexus analyze` CLI opens the DB read-write and refreshes
  fine — verified 2026-07-01: `gitnexus analyze --skills --embeddings` succeeded
  in ~27s and cleared the staleness (the MCP then served HEAD immediately).
  If the index is stale, just run `analyze`; don't assume you can't.
- **Always pass `--skip-agents-md` to `analyze`.** Without it, every run
  rewrites the `<!-- gitnexus:start -->` block below with fresh symbol counts
  and leaves `AGENTS.md` (and `CLAUDE.md`, a symlink to it) dirty in an
  otherwise clean tree — a spurious diff to explain in every review:

      node .gitnexus/run.cjs analyze --skip-agents-md

  Use `--no-stats` instead if you want the block refreshed but without the
  volatile counts, and `--index-only` to skip all file injection. Note the
  advice *inside* the generated block says plain `analyze` and cannot be
  fixed there: `analyze` rewrites that whole region, and it reinstalls
  `.claude/skills/gitnexus/` too (both are untracked/generated). This bullet
  is outside the block, which is the only reason it survives.

---

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **sucoder**. Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/sucoder/context` | Codebase overview, check index freshness |
| `gitnexus://repo/sucoder/clusters` | All functional areas |
| `gitnexus://repo/sucoder/processes` | All execution flows |
| `gitnexus://repo/sucoder/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
