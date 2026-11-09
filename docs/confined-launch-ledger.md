# Confined launch, issue 90

## 1. Task

Bound the tmux command size and detect a job that exits during initial readiness polling.

## 2. Existing machinery

Search tier: GitNexus upstream impact plus ripgrep and direct source inspection.
`sucoder/mirror.py:_launch_confined` stages batch/timer/prepare files over stdin;
`_build_batch_script` preserves the job cgroup, clone cwd, environment and keeper.
`_build_remote_agent_cmd_str` retains the post-agent login shell.
`_confined_job_state_quiet` tolerates probe errors; None means absent/terminal.
Covered by `tests/test_mirror.py` confined launch/readiness cases and
`tests/test_batch_script.py` executable batch tests.

## 3. Definitions and conventions

The pane command is `bash -lc <quoted agent command>` with configured exports
outside that login shell. Script data travels over executor stdin, never argv.
A recorded job ID is not evidence of a live job.

## 4. Invariants

Keep the existing login-shell, cwd, local-tier, managed-state and timer behavior.
Never overwrite a script referenced by another launch. Retain scripts for queued
jobs and retries; do not remove them on launcher exit. Probe errors keep waiting.

## 5. Reuse decisions

Extend `_launch_confined` with exclusive private pane-script staging, reusing
the existing batch builder with a short invocation. Extend readiness using the
existing quiet state probe. Extend the launch error with a bounded job-log tail.
No new configuration or scheduler lifecycle is needed.

## 6. Open questions

None for implementation. Real Slurm acceptance requires a cluster launch;
local executable tests can verify tmux transport and shell semantics.

## Verification (2026-11-09, Sue)

OK against sections 2-5: the production diff changes only `_launch_confined`
and `_confined_session_ready`; no replacement scheduler/state/timer is introduced.
Real local tmux accepted a pane script exceeding 24 KiB with exact argument,
multiline, environment and cwd preservation. Exclusive creation rejects a second
writer without truncation. A staging failure submits no job; an absent job stops
initial polling promptly and reports the concrete log and tmux failure marker.

Run from this worktree with the project environment (the existing environment's
Python was used with `-m pytest` to avoid installing a second environment):

```sh
poetry run pytest -q tests/test_mirror.py tests/test_batch_script.py tests/test_codex_remote.py tests/test_slurm_timer_script.py
poetry run pytest -q tests/test_cli.py tests/test_codex_state_renew.py tests/test_codex_state_release.py tests/test_codex_remote_control.py tests/test_remote_control.py
poetry run mypy --follow-imports=silent sucoder/mirror.py
```

Results: 377 passed; 479 passed; mypy passed. GitNexus staged detection and manual
diff review were performed before commits. Graph hunk mapping also listed the
adjacent `launch_agent` method, whose source is unchanged.

The full `pytest -q --tb=short` run passed 2704 tests, skipped eight, and failed
59. All 55 baseline failure IDs reproduced in an isolated archive of starting
commit `1db3566`. The four additional failures were assertions expecting the old
inline batch layout; all are covered by the final passing 377-test run. One
additional staging-failure regression was added after full-suite collection.
The full suite was not repeated after those test-only corrections.
Logs: `/tmp/sucoder-issue-90-pytest.log`,
`/tmp/sucoder-issue-90-final-focused.log`, and
`/tmp/sucoder-issue-90-launch-regressions.log`.
Baseline archive/results: `/tmp/sucoder-issue-90-baseline-vp1gl9ub`.

No real Slurm submission or installed-launcher deployment was performed.
The dream branch, handoff, and implementation were left unchanged.
