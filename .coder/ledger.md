# Prior-Art Ledger - Codex remote-control launches

Search tier: GitNexus query/context/impact, source and tests, installed Codex
0.154.0 help, and official OpenAI command/configuration documentation.
Baseline: main `df2f7b4`. Prepared by Sue, 2026-09-25.
Implementation: `bd5df6b` on `feat/codex-remote-control`.

## 1. Task

Launch Codex remote control inside SuCoder's existing remote job/tmux lifecycle,
carry the SuCoder prelude and model/permission settings through Codex config,
and distinguish the service from a terminal conversation in sessions, messaging,
peek, and renewal. Keep ordinary harness launches compatible.

## 2. Existing machinery

| Machinery | Source at baseline | Tests | Decision |
|---|---|---|---|
| Command selection and argv parsing | `sucoder/cli.py:1588` | `test_launch_commands_forward_harness_and_model` | Reuse `--agent-command`; keep `--agent` as an executable selector |
| Harness flag and prompt assembly | `sucoder/mirror.py:3155`, `sucoder/config.py:278` | `test_launch_agent_supports_overrides`, `test_agent_doc_injected_for_non_claude` | Extend for the Codex remote-control subcommand |
| Remote prelude staging | `sucoder/mirror.py:2176` | `test_build_remote_agent_cmd_str_externalizes_prelude` | Reuse stdin staging and sentinel substitution; give service launches unique files |
| Remote tmux and confined submission | `sucoder/mirror.py:2480`, `sucoder/mirror.py:2914` | `test_launch_confined_wraps_agent_in_bash_lc`, `tests/test_batch_script.py` | Reuse cgroup, cwd, cache and watchdog setup |
| Atomic launcher records | `sucoder/session.py:19` | `tests/test_remote.py` session round trips and atomic-save tests | Extend with a service launch description for renewal |
| Scheduler/tmux discovery and pane liveness | `sucoder/sessions_report.py:71`, `sucoder/cli.py:3911` | `tests/test_sessions_report.py`, session collection tests in `test_cli.py` | Extend live probes with a tmux mode marker; records are not the registry |
| Message planning and pane reading | `sucoder/messaging.py:179`, `sucoder/cli.py:3799` | `tests/test_messaging.py`, message/peek tests in `test_cli.py` | Refuse service stdin even with `--force`; identify peek output as service logs |
| Checkpoint sentinel and relaunch | `sucoder/cli.py:3340`, `sucoder/cli.py:3492`, `sucoder/renew.py:190` | `tests/test_renew.py` | Retain the sentinel; preserve service command/model and explain that it is not a chat delivery |

## 3. Definitions and conventions

- `--agent-command` is shell-tokenized argv, not a shell program (`cli.py:1595`).
  Shell metacharacters must remain data.
- A live tmux session is not proof of a live agent: the window deliberately
  ends in `exec bash -l` (`mirror.py:2488`).
- An unanswered pane/scheduler query is unknown, not evidence of an exit
  (`sessions_report.py:262`). Enumerate from the scheduler and tmux.
- Confined services, like agents, must run in the sbatch job cgroup, with the
  existing per-job local working clone when configured (`mirror.py:2382`).
- Codex `remote-control` runs in the foreground; `remote-control start`
  starts a daemon (official developer command reference). SuCoder will run
  either launch spelling in the foreground so tmux owns the process.
- Renewal already writes `~/.cache/sucoder/renew-requested`; it does not send
  terminal input. Reading that sentinel is cooperative, not an acknowledgement.

## 4. Invariants and assumptions

- Preserve existing agent profile defaults outside remote-control launches.
- Preserve the full prelude, including target instructions and skills, as one
  TOML string in `developer_instructions`; never pass it as a positional prompt.
- Keep model and permission configuration as argv values, with correct quoting.
- Record the requested launch only for a newly submitted job or tmux window;
  reattaching must not relabel an existing session from the requested command.
  Renewal may heal a record from the actual live pane's launch metadata.
- A live tmux marker must identify a service even if launcher records are lost.
- Terminal message delivery cannot address a remote-control conversation.
- Reuse the existing watchdog/snapshotter, atomic session save, and renewal loop.
  Do not add a second daemon supervisor or change allocation cancellation rules.
- No live service, pairing action, or cluster allocation is part of local tests.

## 5. Reuse decisions

- Extend the launch path with a small, pure Codex subcommand adapter: executable
  detection alone cannot distinguish a TUI from an app-server service.
- Reuse remote prelude externalization for the complete config override.
- Extend pane reports with mode metadata separate from process liveness.
- Store the service command/model before generated flags for renewal, without
  resolved credentials or the composed prelude; rebuild those on replacement.
- Keep checkpoint sentinels and snapshots. Make the service limitation explicit
  and restart the same service mode instead of the configured default TUI.
- Use create-or-reuse explicitly for non-confined service tmux launches. A real
  private-socket probe showed `new-session -A -d` tries to attach to an existing
  session and fails without a TTY; the ordinary terminal path is unchanged.

## 6. Open questions

No implementation decision is blocked. Authentication/pairing, cluster egress,
and concurrent allocations sharing Codex state require a live cluster check.
Local tests used fake agents and private tmux sockets.

Verification, 2026-09-25 (Sue):

- OK (sections 2-5): `sucoder/agent_mode.py:46` recognizes subcommands without
  confusing model/config values for commands. `mirror.py:3434` reuses the
  permission intent and maps it to config; `mirror.py:3288` keeps the complete
  prelude. Unicode, control characters, shell metacharacters, and a second
  queued prelude were checked through TOML parsing and an actual tmux pane.
- OK (sections 3-4): `sessions_report.py:110` extends the existing live probe;
  `messaging.py:204` excludes service panes even with force. Existing liveness,
  WIP ages, and legacy command-only probe parsing remain covered.
- OK (sections 2-5): `session.py:38`, `mirror.py:3123`, and `cli.py:3340` retain
  command/model through renewal, save before polling, preserve existing job
  metadata on reuse, and rebuild the prelude. `cli.py:3549` still checkpoints
  through a file sentinel, without typing into service stdin.
- `.venv/bin/python -m pytest -q tests/test_remote_control.py`: 51 passed.
- `umask 022; .venv/bin/python -m pytest -q --tb=short`: 1119 passed, one
  pre-existing failure, `test_direct_collaborate_rejects_node`: startup reports
  the agent cannot read the temporary config before reaching its assertion.
- With the host's default umask `0007`, the suite has 1114 passes and six
  failures. All six reproduce in an untouched `git archive df2f7b4` checkout.
  The additional five are SSH Include fixtures created group-writable and
  timer fixtures expecting 0755 directories; these pass with umask 022.
- Mypy on the six changed source files reports the same 69 diagnostics as
  the baseline's five existing files; no new diagnostics. Command:
  `.venv/bin/python -m mypy --follow-imports=silent sucoder/agent_mode.py sucoder/mirror.py sucoder/cli.py sucoder/messaging.py sucoder/session.py sucoder/sessions_report.py`.
- Changed Python sources parse using Python 3.9 syntax. Installed Codex 0.154.0
  accepts the generated config/subcommand argument layout with `--help`.
- GitNexus was refreshed with `--skip-agents-md`; `detect-changes --scope all`
  reports the expected launch, session persistence, discovery, and messaging
  paths. Its CRITICAL rating reflects the shared launch/persistence callers;
  no other feature paths were intentionally modified.

Official references:
- https://learn.chatgpt.com/docs/developer-commands?surface=cli#codex-remote-control
- https://learn.chatgpt.com/docs/config-file/config-reference
