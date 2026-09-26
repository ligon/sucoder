# Prior-Art Ledger - Codex remote-control launches

Search tier: GitNexus query/context/impact, source and tests, installed Codex
0.154.0 help, and official OpenAI command/configuration documentation.
Baseline: main `df2f7b4`. Prepared by Sue, 2026-09-25.
Implementation: `bd5df6b`, prompt repair `bf8989d`, on `feat/codex-remote-control`.
Follow-up: preserve instructions across remote-client overrides, 2026-09-26.

## 1. Task

Launch Codex remote control inside SuCoder's existing remote job/tmux lifecycle,
carry the SuCoder prelude through native instructions and Codex config,
retain model/permission settings,
and distinguish the service from a terminal conversation in sessions, messaging,
peek, and renewal. Keep ordinary harness launches compatible.

## 2. Existing machinery

| Machinery | Source at baseline | Tests | Decision |
|---|---|---|---|
| Command selection and argv parsing | `sucoder/cli.py:1588` | `test_launch_commands_forward_harness_and_model` | Reuse `--agent-command`; keep `--agent` as an executable selector |
| Harness flag and prompt assembly | `sucoder/mirror.py:3155`, `sucoder/config.py:278` | `test_launch_agent_supports_overrides`, `test_agent_doc_injected_for_non_claude` | Extend for the Codex remote-control subcommand |
| Remote prelude staging | `sucoder/mirror.py:2176` | `test_build_remote_agent_cmd_str_externalizes_prelude` | Reuse stdin staging and sentinel substitution; give service launches unique files |
| Native context-file staging | `sucoder/mirror.py:2216` (current) | `test_staged_installer_runs_in_final_clone_and_keeps_launches_separate` | Reuse private agent-side staging for the standalone Codex context installer |
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
- Preserve the full prelude, including target instructions and skills, in native
  mirror-scoped Codex instructions; never pass it as a positional prompt.
  Codex 0.154.0 replaces the server's `developer_instructions` when a client
  supplies `thread/start.developerInstructions`. Passing argv correctly alone
  does not establish that the model receives the prompt.
- Preserve existing `AGENTS.md` and local override instructions. Install the
  native context in the actual launch directory after local-tier preparation.
  Do not change the shared Codex home, authentication, or hook trust.
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
- Reuse private agent-side file staging for a standalone context installer.
  A local `AGENTS.override.md` carries the prelude and existing native project
  instructions. Refresh only generated regions, preserve user text, and use
  an atomic write. A copied `AGENTS.md` is refreshed on each service launch;
  the bootstrap prompt also directs the agent to read current project guidance.
  Exclude the generated file through the repository's local Git exclude file.
  Verify generated-region hashes before refresh; preserve manual and concurrent
  edits. Refuse tracked or symlinked overrides rather than alter shared sources.
  Keep the developer config override as a fallback for clients using another cwd.
- Extend pane reports with mode metadata separate from process liveness.
- Store the service command/model before generated flags for renewal, without
  resolved credentials or the composed prelude; rebuild those on replacement.
- Keep checkpoint sentinels and snapshots. Make the service limitation explicit
  and restart the same service mode instead of the configured default TUI.
- Use create-or-reuse explicitly for non-confined service tmux launches. A real
  private-socket probe showed `new-session -A -d` tries to attach to an existing
  session and fails without a TTY; the ordinary terminal path is unchanged.

## 6. Open questions

Scope is each SuCoder mirror, as assumed while the optional scope question was
unanswered. No host-wide Codex settings are changed. Native instructions apply
only when the client selects the mirror or a directory beneath it; a different
cwd still depends on the developer config surviving client overrides.
Authentication/pairing, cluster egress, and concurrent allocations sharing Codex
state require a live cluster check. Local protocol probes use an isolated Codex
home and a loopback fake model endpoint, without pairing or paid model requests.

Verification, 2026-09-26 (Sue):

- OK (sections 2-5): `sucoder/agent_mode.py:46` retains subcommand detection,
  service flags, model overrides, and terminal/service distinction. The initial
  argv-only checks were insufficient: a real `thread/start` with client
  developer instructions discarded the server's developer config.
- OK (sections 2-5): `sucoder/mirror.py:3445` stages the standalone installer
  through `_write_context_prelude_file`; each launch has a unique immutable
  script. It runs in the final job cwd, after local-disk preparation. An actual
  staged-script test verifies cwd, Unicode, shell metacharacters, and separation
  between queued launches.
- OK (sections 3-4): `sucoder/codex_context.py:60` adds the full prelude to
  mirror-local native instructions. Tests preserve global/project instructions,
  existing local override text, edits outside generated regions, linked-worktree
  scoping, and updates to the original AGENTS.md. Edited generated regions,
  concurrent changes, symlinked overrides, and tracked overrides are preserved
  and rejected explicitly. Git-local excludes cover output and temporary files.
- OK (sections 3-4): `tests/test_codex_context.py:158` runs installed Codex
  0.154.0 app-server with an isolated home and a loopback fake model endpoint.
  The captured model request contains the complete 40+ KiB prelude, client
  instructions, project instructions, global instructions, and native base
  instructions even though the server developer config was replaced.
- `umask 022; .venv/bin/python -m pytest -q --tb=short`: 1132 passed, one
  pre-existing failure, `test_direct_collaborate_rejects_node`: startup reports
  the agent cannot read the temporary config before reaching its assertion.
  The full run includes all 64 remote-control/context cases.
- The original six failures under default umask 0007 were reproduced in an
  untouched `git archive df2f7b4` checkout. Five disappear with umask 022
  (SSH Include permissions and timer directory modes); the config-readability
  failure remains. No unrelated fixes are included.
- `.venv/bin/python -m mypy --follow-imports=silent sucoder/codex_context.py`
  passes. Including mirror.py reports its same two baseline diagnostics.
  Both changed runtime sources parse using Python 3.9 syntax.
- GitNexus refreshed with `--skip-agents-md`; `detect-changes --scope staged`
  reports the expected installer and launch paths (eight flows, HIGH because
  launch_agent is shared). Session persistence, messaging, and renewal behavior
  remain covered by the full suite.

Official references:
- https://learn.chatgpt.com/docs/developer-commands?surface=cli#codex-remote-control
- https://learn.chatgpt.com/docs/config-file/config-reference
- https://learn.chatgpt.com/docs/agent-configuration/agents-md
