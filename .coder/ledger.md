# Prior-Art Ledger - Codex launches and node-local SQLite state

Current design task: node-local SQLite with durable checkpoints, 2026-10-15.
Baseline: main `ffd0447`; design only, no runtime implementation.
Search tier: GitNexus CLI query/context, source and named tests, installed Codex
0.162.1 help and database filename inventory, and official OpenAI/SQLite docs.
The earlier remote-control evidence below remains version-specific.

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

Design an opt-in extension of local-disk tiering that runs Codex SQLite on the
compute node, preserves state across allocations outside Git, and separates
concurrent allocations. Deliver a reviewed design and Savio validation plan
before implementation; see `docs/codex-local-state.org`.

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

Current baseline anchors for the state design:

| Machinery | Source at ffd0447 | Tests inspected | Decision |
|---|---|---|---|
| Local clone, cache and temporary paths | `sucoder/local_tier.py:37`, `:57`, `:89` | `test_paths_runtime_job_id_expands_and_literal_is_quoted`, `test_cache_exports_cover_tools_and_tmpdir_but_not_installs` | Extend placement; keep state outside clone |
| Confined runtime setup | `sucoder/mirror.py:2416`, `:2480` | `test_launch_confined_local_tier_stages_prepare_script`, `tests/test_batch_script.py` | Reuse in-job cwd and private tmux socket |
| Unconfined preparation | `sucoder/mirror.py:2572` | `test_launch_agent_unconfined_local_tier_clones_then_launches_in_clone` | Reuse job-local root |
| Timer and deadline snapshots | `sucoder/slurm_timer.py:82`, `:204`, `:261` | `test_watchdog_warns_and_snapshots_without_changing_index`, `test_scheduler_outage_does_not_suspend_periodic_snapshots` | Extend requests, keep SQLite I/O out of warning loop |
| Node-local timer ownership | `sucoder/timer_lifecycle.py:9` | `test_concurrent_starters_create_one_timer`, `test_locks_work_when_home_does_not_support_flock` | Reuse local-lock and readiness pattern, not NFS locks |
| Atomic launcher metadata | `sucoder/session.py:19`, `:84` | `test_service_session_roundtrip_and_legacy_defaults`, session-save tests in `tests/test_remote.py` | Extend with validated state provenance |
| Overlapping renewal | `sucoder/renew.py:232`, `:244`, `:253`; `sucoder/cli.py:3391`, `:3601` | `test_loop_draining_relaunches_checkpoints_and_cancels_old`, `test_loop_relaunch_failure_backs_off` | Add acknowledged service drain before restoration |
| Release and WIP retirement | `sucoder/cli.py:2932`, `:3130`, `:3176` | release cases in `tests/test_cli.py` | Preserve last state checkpoint independently of WIP retirement |
| Native context and fake model protocol | `sucoder/codex_context.py:120`, `tests/test_codex_context.py:158` | `test_codex_model_receives_prelude_when_client_replaces_developer_instructions` | Reuse isolated protocol-testing approach for resume |
| Confined adoption and readiness | `sucoder/mirror.py:3045`, `:3058`, `:3233` | scheduler-adoption cases in `tests/test_mirror.py` | Extend only for an authorized replacement; tmux alone is not state-owner readiness |

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
- An allocation is not a Codex conversation: `/local/job<ID>` survives a Codex
  restart inside the allocation and is discarded at allocation end
  (`sucoder/local_tier.py:10`).
- Renewal is explicitly overlapping: `run_renew_loop` calls `relaunch()` before
  `scancel(old_job)` (`sucoder/renew.py:244`, `:253`). A design requiring the old
  allocation to end before every restore contradicts the current lifecycle.
- Git WIP snapshots exclude ignored files and are not SQLite backups
  (`sucoder/slurm_timer.py:82`; `docs/local-disk-tiering.org`).
- Official Codex configuration separates `sqlite_home` / `CODEX_SQLITE_HOME`
  from `CODEX_HOME`; config takes precedence over the SQLite environment value.
- Codex 0.162.1 local inventory contains `state_5`, `thread_history_1`, `goals_1`,
  `queue_1`, `memories_1`, and `logs_2` SQLite files. Filename presence alone
  does not prove their redirection or a cross-database recovery contract.
- Installed 0.162.1 help describes a shared daemon and `--no-daemon` for both
  the TUI and bare `remote-control`. The adapter at `agent_mode.py:46` only
  removes `start`; its earlier foreground assumption needs revalidation.

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
- New state must be outside Git, private to the agent user, and isolated by
  stable scope plus run UUID; job IDs alone are not durable identities.
- A generation is durable only after complete publication and verification;
  pending, failed and local-only copies do not advance recovery status.
- Per-database online backups do not establish a single consistent instant
  across separate SQLite databases. Warm restore eligibility is a release gate.
- A still-running predecessor cannot be continued unless an acknowledged drain
  proves the service stopped and published its sealed checkpoint. Scheduler
  errors and stale PID/heartbeat records are unknown ownership.
- A historical sealed manifest is not current transfer authority: offer,
  single-successor claim, revoke and parent pin/prune operations need a short
  durable metadata critical section. Never steal its lock by age.
- Confined scheduler adoption must exclude only a specifically authorized
  drained predecessor. Replacement readiness must identify a different job,
  restored parent and owned Codex server before the old allocation is cancelled.
- Failed/pending replacement metadata must retain the old allocation and run.
  Managed TUI commands need their own validated renewal descriptor; the existing
  `service_launch` accepts only remote-control, not all Codex launches.
- State placement must not change `CODEX_HOME`, authentication or hook trust.
  Existing shared state is never silently imported, overwritten or deleted.

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

State design decisions (proposed, not implemented):

- Extend existing launch staging with a stdlib-only state runner after the
  final cwd is known. Reuse `config_override` for an absolute `sqlite_home`.
- Extend the existing timer to enqueue local checkpoint requests; use one
  single-flight worker per state run, never inline shared-storage I/O.
- New immutable filesystem generations hold SQLite backups and provenance;
  Git refs, WIP retention and shared mutable database files do not fit.
- Extend session metadata and renewal with acknowledged drain/checkpoint
  handoff. New allocation may coexist with the old allocation after the old
  service stops; an unacknowledged handoff never pretends to be continuity.
- New durable offer/claim/revoke and parent-pin records use one short
  atomic-mkdir metadata lock per scope. Capture/upload never hold it.
  No concurrent same-history forks in the first version; upstream thread IDs
  and external files must be isolated before forks can be supported.
- Enforce one owner/unresolved successor per lineage even when claims name
  different historical generations. Ordinary clean-exit offers may be unbound
  until a claim binds their successor exactly once.
- Keep version-specific storage inventory and warm recovery as explicit
  compatibility gates. Unknown databases and external transcript dependencies
  must be resolved on Savio before production opt-in.

## 6. Open questions

Current state-design gates: determine which databases and maintenance locks
actually follow `sqlite_home` on Savio; prove cross-database warm recovery and
resuming with a changed cwd; establish queue/job replay behavior; verify that
each managed launch owns an independent foreground server. No cluster was
queried or changed in this design session. Proposed configuration belongs only
in the design until parser, README and example can be changed together.

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

Spike evidence, 2026-10-15 (Sue), Slice 0 read-only inventory, job 39969504
on n0033.savio4 (4 cores, confined cgroup):

- Prior art not cited above: `$CODEX_HOME/sqlite-backups/n0140-job39450011/`
  `recovery-tools/` (`RECOVERY.org`, `cutover.py`, `snapshot.py`,
  `checkpoint.py`, `codex-local`), written 2026-10-01 in job 39450011 on
  n0140 against the managed Codex 0.160.0 daemon. It stopped that daemon,
  took SQLite online backups of the six shared databases, restarted it with
  `CODEX_SQLITE_HOME` plus `-c sqlite_home=<local>`, and verified through
  `/proc/<pid>/fd` that every open `.sqlite*` handle was under
  `/local/job39450011/codex-sqlite/active`. A 15-minute loop then wrote
  per-database online backups (`Connection.backup`, `quick_check`, sha256,
  `SNAPSHOT.json`, keep last two) to the shared home. Its diagnosis counted
  1299 database-locked warnings in a four-hour window on Lustre. It checked
  `.sqlite*` handles only: lock files and sidecar directories, the TUI and
  `--no-daemon` modes, override precedence and the current 0.162.1 build are
  what this spike adds. `$CODEX_HOME/sqlite-lustre-moved-2026-10-14/` holds
  a second manual relocation (databases plus `stale-writer-locks/`); no
  accompanying note was found.
- Reused here: `sucoder/agent_mode.py:config_override` encodes the
  `sqlite_home` argument; `tests/test_codex_context.py:158` supplies the
  isolated-home, loopback fake-model app-server driver.
- `$HOME` (`/global/home`) is NFS mounted `local_lock=none`, which is why the
  real Codex home is a symlink to `/global/scratch/fsa/fc_jevons/ligon/`
  `codex-home` on Lustre (`flock`); `/local` is ext4. The npm `codex` 0.162.1
  native binary and the managed daemon's `releases/0.162.1-*/bin/codex` have
  the same sha256, so the spike's binary is the production binary.
- The 0.162.1 binary names a seventh database, `agent_message_board_1.sqlite`,
  absent from the design's inventory table (`docs/codex-local-state.org:60-67`).
- The live daemon (pid 3630696, started 2026-10-14 20:56, working under
  `/local/job39919092/`) runs on another node; nothing in this spike talks to
  it, and no `daemon`/`doctor` command is run against the real home.
- Results (same day): all six databases follow `-c sqlite_home`, which beats
  `config.toml`, which beats `CODEX_SQLITE_HOME` (cases c3, c4, c10); a relative value resolves
  against the cwd; a missing directory is created `0777` masked by umask; an
  unwritable one fails the launch with exit 1 and no fallback. Thread-writer
  locks, `installation_id`, plugin and arg0 locks, the `sessions/` rollout
  JSONL and the TUI's `history.jsonl` stay in `CODEX_HOME`. The app-server and
  TUI hold their own database descriptors with no daemon socket; the
  `--no-daemon` server creates `/tmp/codex-daemon-<uid>/<hash>` and needs
  credentials to go further. Stdin EOF leaves all data in WAL. Full record:
  `docs/codex-local-state-spike.org`.
