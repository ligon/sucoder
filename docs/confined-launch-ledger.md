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
