# SuCoder — project notes

These rules are SuCoder-repo-specific. They live here (outside the gitnexus
block) instead of in `~/.sucoder/system_prompt.org`, which is the system-wide
default for all mirrors and is meant to stay project-agnostic.

- If introducing new configuration, update `README.org` and `config.example.yaml`
  so the example stays in sync with the parser.
- Test the Python code paths with `pytest` (run from the SuCoder mirror root);
  add targeted tests for behavior changes.

---

