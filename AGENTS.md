# SuCoder — project notes

These rules are SuCoder-repo-specific. They live here (outside the gitnexus
block) instead of in `~/.sucoder/system_prompt.org`, which is the system-wide
default for all mirrors and is meant to stay project-agnostic.

- If introducing new configuration, update `README.org` and `config.example.yaml`
  so the example stays in sync with the parser.
- Test the Python code paths with `pytest` (run from the SuCoder mirror root);
  add targeted tests for behavior changes.
- **GitNexus is NOT read-only — do not conclude otherwise from the hook spam.**
  Agents repeatedly misread the `Cannot execute write operations in a read-only
  database` FTS messages as "the index can't be refreshed from here." Those are
  a *non-fatal* full-text-index auto-ensure inside the MCP server's *query* path
  (note the "Will retry on next query"); they never block queries or
  re-indexing. The `gitnexus analyze` CLI opens the DB read-write and refreshes
  fine — verified 2026-07-01: `gitnexus analyze --skills --embeddings` succeeded
  in ~27s and cleared the staleness (the MCP then served HEAD immediately).
  If the index is stale, just run `analyze`; don't assume you can't.

---

