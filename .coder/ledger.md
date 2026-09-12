# Prior-Art Ledger - Slurm timer integration

Search tier: GitNexus query/context/impact plus source and test inspection.
Baseline: PR #13 (`2c24354`) and local main (`258accb`).

## 1. Task

Integrate PR #13's deadline warnings and regression coverage with main's
timer supervision. Keep monitoring through scheduler errors.

## 2. Existing machinery

| Machinery | Source at baseline | Tests | Decision |
|---|---|---|---|
| SSH timer staging and readiness | `sucoder/cli.py:1129` on main | `tests/test_timer_lifecycle.py::test_timer_ssh_timeout_is_advisory` | Reuse main |
| Allocation-scoped locks and readiness handshake | `sucoder/timer_lifecycle.py:3` on main | `test_timer_survives_starter_and_reuses_owner`, `test_concurrent_starters_create_one_timer`, `test_missing_executable_reports_failure` | Reuse main |
| Confined timer launch | `sucoder/mirror.py:2391` on main | `tests/test_batch_script.py`, `tests/test_mirror.py::test_launch_confined_stages_and_starts_deadline_timer` | Keep `bash --ensure`; extend atomic staging |
| Shared warning loop and builder | `sucoder/slurm_timer.py:102` on PR #13 | `tests/test_slurm_timer_script.py` warning-chain tests | Extend scheduler error handling |
| Snapshot implementation | `sucoder/slurm_timer.py:65` on PR #13 | Real repository-pair tests and `test_watchdog_warns_and_snapshots_without_changing_index` | Reuse |
| Scheduler observation contract | `sucoder/cli.py:765` on PR #13 | Slurm state-query tests in `tests/test_cli.py` | Reuse distinction between errors and empty successful queries |

## 3. Definitions and conventions

- Scheduler errors are unknown state: "An ssh/squeue failure is NOT evidence
  the job is dead" (`sucoder/cli.py:765`, PR #13).
- Successful empty queries indicate disappearance; retain PR #13's three
  consecutive observations before ending the watchdog.
- `snapshot_minutes` is a periodic cadence; zero disables periodic snapshots,
  while threshold snapshots remain enabled (`build_timer_script` docstring).
- `--ensure` reports `SUCODER_TIMER_STARTED` or `SUCODER_TIMER_REUSED`; a
  successful SSH return alone does not establish readiness.

## 4. Invariants

- Preserve main's target/node/allocation locks, immutable SSH script names,
  readiness diagnostics, and advisory timer failures. Do not restore `pkill`.
- Never cancel the user's allocation from the timer.
- Preserve the agent's real Git index and existing snapshot implementation.
- Warning urgency only increases; skipped thresholds must not fire later.
- Scheduler failures must not stop monitoring or suppress periodic snapshots.
- Keep unrelated startup safety fixes from `258accb` intact.

## 5. Reuse decisions

- Reuse main's supervision rather than implementing another startup check.
- Extend the existing warning loop and bash test driver to distinguish failed
  queries from successful empty queries, including recovery and snapshot tests.
- Retain PR #13's confined atomic staging, warning order, lifecycle wording,
  and mutation-derived regression tests; adapt assertions to supervision.
- Keep compatibility warning files cleared when a new watchdog starts.

## 6. Open questions

No implementation decision is blocked. Cluster smoke testing and issue #15's
batch-environment Git availability need a real allocation. Issues #14 and #16
remain outside this integration.

Prepared by Sue (2026-09-12).
