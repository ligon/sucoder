# Prior-Art Ledger - Codex launches and node-local SQLite state

Current task: remote-control identity and reachability, 2026-10-20.
Baseline: main `77b7a29`; implementation authorized by the human's active goal.
Search tier: source/call-site searches, named tests, matching upstream Codex
0.163.0 source, installed schema, and read-only live RPC/SQLite/log inspection.
GitNexus query and impact were attempted; its native module requires glibc
2.29, unavailable on this host, including with the existing OpenSSL library.
The current charter and design are `docs/codex-remote-control.org`.

Earlier state-design task: node-local SQLite with durable checkpoints, 2026-10-15.
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

Make concurrent Codex remote-control owners independent across nodes, pair
with the selected owner rather than the shared daemon, distinguish local
readiness from relay connectivity, and retain the existing state/recovery
and allocation lifecycle. The human authorized implementation while away.

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

Current remote-control anchors at `77b7a29` (earlier anchors below are history):

| Machinery | Source | Tests inspected | Decision |
|---|---|---|---|
| Owned app-server and Unix socket | `sucoder/codex_remote.py:14`, `:100` | `test_terminal_uses_same_server_and_cleans_up`, `test_signal_stops_only_owned_processes` | Extend, retain supervisor lifecycle |
| Mode/legacy detection and launch assembly | `sucoder/agent_mode.py:46`, `sucoder/mirror.py:3490` | `tests/test_remote_control.py`, `test_confined_remote_launch_stages_toml_context_inside_batch` | Extend, normalize legacy service to owned app-server |
| State scope, runner argv, checkpoint root | `sucoder/codex_state.py:60`, `:183` | `test_scope_id_is_a_stable_digest_of_the_unsanitized_tuple`, runner-command cases | Reuse; add remote-isolation and temporary-root intents |
| Canonical home and bind namespace | `sucoder/codex_state_runner.py:1422`, `:1479`, `:3450` | `test_runner_places_a_real_codex_app_server`, restored/reused run cases | Extend with private installation file; reuse recovery engine |
| Run ownership, restore and transcript replication | `sucoder/codex_state_runner.py:3450` | `test_restore_places_a_sealed_generation_in_another_allocation`, `test_a_real_codex_thread_resumes_from_a_restored_generation` | Reuse; do not create a second recovery mechanism |
| State reporting, drain and renewal | `sucoder/mirror.py:4067`, `sucoder/cli.py:4132` | `tests/test_codex_state_status.py`, `tests/test_codex_state_renew.py` | Extend actual-run discovery when no local-tier root exists |
| In-allocation request routing | `sucoder/mirror.py:386` (`_CodexStateRoute`) | state status/drain routing cases | Reuse for remote status and pairing |

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

- An installation identity is read from `CODEX_HOME/installation_id`, even
  when `sqlite_home` is private. Confirmed in installed 0.163.0 and matching
  upstream `codex-rs/core/src/installation_id.rs`; canonical home remains
  shared unless that file is bound privately.
- Codex 0.163.0 stores `remote_control_enrollments` in `state_5.sqlite`, keyed
  by WebSocket URL, account and app-server client name. Its cached enrollment
  does not carry/check the installation ID (`remote_control/enroll.rs:293`).
  A restored private database therefore also needs its enrollment cache reset
  before an owner with a new installation identity opens it.
- Unix WebSocket transports use a null/global enrollment key; changing a
  diagnostic client's `initialize.clientInfo.name` does not isolate it.
- Pairing claimed, a local socket, and a connected relay are separate facts.
  This session paired successfully while relay connects returned HTTP 409
  with `Remote app server already online`; the phone displayed an old node.
- `codex remote-control pair` addresses Codex's managed daemon. Pairing a
  SuCoder-owned server must use its exact private socket and verify the
  returned installation/environment identity.

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

- Each new remote server owner gets a fresh installation identity and private
  enrollment storage. Reconnecting to that running owner does not rotate it.
- New remote-control launches reuse the state runner even without local-tier
  clone placement. Their private temporary local root is retained after exit,
  especially when a final checkpoint fails; durable replicas/generations use
  the existing engine and remain outside Git.
- Remote identity is not copied from a predecessor. Renewal pairs a fresh
  owner; carrying a paired identity across renewals needs separate fencing.
- Never reset enrollment in shared, live, or source-generation databases.
  Only the owned, stopped, private database may be sanitized before startup.
- Missing isolation tools or explicitly disabled managed state must refuse
  unsafe remote control rather than silently launch against shared state.
- A status/pair operation never starts a daemon, allocates a node, resumes a
  thread, sends inference, or trusts an old PID without its start identity.

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

- Extend `CodexStateLaunch` and the existing runner with remote-isolation and
  temporary-local-root intents; retain configured local-tier placement.
- New bounded, stdlib-only Unix WebSocket RPC helper: existing state helper
  speaks stdio for one-shot thread/cwd repair, not the owned socket, and the
  native CLI pairing command targets another daemon.
- Extend state/run discovery using a pane marker for actual runtime paths;
  reuse existing request routing and lifecycle records instead of inventing
  a second global service registry.
- Expose truthful status and explicit pairing for the selected mirror/target.
  Generate terminal QR output on the launcher; the target needs no QR package.

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

Current remote-control validation gates: prove two isolated owners can enroll
simultaneously, prove restored enrollment is not reused, test disconnected and
stale-owner status/pairing, and perform live relay checks without altering any
existing server. Android conversation opening requires the human's phone;
do not claim that gate passed from pairing or relay status alone.

Earlier state-design questions and historical evidence follow.

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
- Outcomes 3-7 (same day, job 39975196 on n0038.savio4, `scripts/codex_state_outcomes.py`,
  record section "Outcomes 3--7"): `thread/resume` needs the rollout JSONL at the
  absolute path in `threads.path`; without it every history read fails, and
  with it a restored database is overridden by the file on resume (12 of 20
  warm restores showed more turns than their capture): for thread history the
  generation does not determine what resumes, the shared rollout does.
  `sessions/` is relocatable because the stored path is not canonicalized: a
  symlink works but redirects every process; a per-process `bwrap` bind mount
  of local disk over `CODEX_HOME/sessions` (and `.tmp`, `tmp`,
  `thread-writer-locks`, `shell_snapshots`) leaves about ten metadata calls
  and no data I/O on Lustre per turn, with the shared home untouched and
  Codex's sandbox still working (`symlink` and `bind` sub-tests, 2026-10-15;
  evidence under `evidence-outcomes/job39969504-bind/`). Episode tests
  (`scripts/codex_state_episode.py`, `evidence-outcomes/job39969504-episode/`):
  rollout skew in either direction keeps the model correct, but a database
  ahead of the rollout leaves the SQLite history permanently wrong (projection
  watermark past the file's end: phantom turns shown, real turns missing); per-turn tail replication 13 ms, identical at the
  next turn/start; residue 1.7 metadata calls per turn with skills, AGENTS and
  history.jsonl bound, not load-bearing; TUI and remote-control start under
  bwrap; compaction appends to the rollout.
  20/20 seeded abrupt-stop interleavings restored every completed turn from both
  warm and post-crash captures. Unattended `thread/resume` executes a queued
  message once and, with an active goal, issues ~20 model requests per second
  indefinitely; a goal set to `paused` is a usable hold, the queue has none. Two
  servers with different homes on one `sqlite_home` are two writers with no
  refusal; the same home refuses the second with "already has an active writer".
  Median turn 0.51 s all-Lustre, 0.21 s databases-local, 0.042 s home-local too.
  Outcome 6 and the protocol half of 5 need the runner and were not run.

Slice 1 implemented, 2026-10-15 (Sue): `sucoder/codex_state_runner.py`,
`sucoder/codex_state.py`, `slurm.codex_state` in `config.py`, wiring in
`mirror.py:launch_agent`, `tests/test_codex_state.py`.

- Reused: the staged-Python pattern of `_wrap_codex_service_context`
  (`python3 <staged> <command>`), `local_tier.cache_exports_sh` for the
  runtime `SUCODER_LOCAL_ROOT`, `config_override` for the `-c` value, and
  `_write_context_prelude_file`'s staging transport.
- Measured while implementing: bwrap stays alive as the parent of the
  sandboxed command and does not relay signals (SIGTERM to bwrap exits -15
  and `--die-with-parent` then SIGKILLs the command; SIGTERM to its child
  runs the handler and bwrap propagates the code); a bound file's
  descriptor shows the mount-point path from outside the namespace, so
  placement is verified by `st_dev` of `/proc/<pid>/fd/<n>`.
- Pre-existing hazard, not fixed here: the staged context installer
  (`codex_context.py`) starts with `from __future__ import annotations`
  (3.7+) and fails to compile on Python 3.6, which is the default `python3`
  on some node images (n0038); the state runner is kept 3.6-clean and
  compile-checked in its tests.
- Checked after wiring: the TUI runs under the runner in a tmux pane (prompt
  executed, state placed, rollout local, `/quit` exits 0); Codex installs its
  own SIGINT handler, so the runner's inherited ignore is harmless; the real
  home has every bind directory and `history.jsonl`, and lacks `AGENTS.md`
  and `AGENTS.override.md`, which the runner skips. The runner logs to
  `runner.log` once Codex may own the terminal. The scope digests the
  resolved mirror path (`_resolve_remote_path`), one SSH round trip.
  In remote-control modes the supervisor's server and client invocations
  both carry the placeholder; the runner's placement check covers both.
- Baseline failure, not fixed here: `tests/test_codex_context.py::
  test_codex_model_receives_prelude_when_client_replaces_developer_instructions`
  asserts a top-level `instructions` field that Codex 0.162.1 no longer
  sends; everything else passes (1203).
- GitNexus cannot run on this node image (`libssl.so.3` missing for its
  native database); impact was checked by grepping callers and is listed
  in the wiring commit.

---

# Prior-Art Ledger - Message replies from the agent's transcript (2026-10-23)

Task: charter `.coder/charter-reply-from-transcript.md` (signed off 2026-10-23),
design `docs/reply-from-transcript.org` draft 3, evidence
`docs/reply-transcripts-evidence.org`. Branch `feat/reply-from-transcript`,
baseline `60f25b9`. Prepared by the prior-art pass of the workshop engine;
read-only except this section. Search tier: source and named tests, GitNexus
CLI `impact` (repo sucoder), shape-only parses of the two lyn transcripts
the charter allows (entry types and counts, no text dumped).

## R1. Existing machinery (reuse or extend)

- `sucoder/messaging.py`: `new_message_id` (4 hex today), `frame`,
  `REPLY_PREFIX`, `_BARE_REPLY`, `capture_pane_command`, `extract_reply`
  (screen reader, stays as the fallback), `Recipient`, `send_keys_command`,
  `_in_job`, `parse_send_output`, `plan_recipients`. Pure; the CLI does I/O.
- `sucoder/sessions_report.py:job_tmux_command` (l.124): srun --overlap step,
  falling back to `ssh -n node` for a default-socket session; *inner* is
  `shlex.quote`d. The probe's job route reuses it unchanged (via
  `messaging._in_job`). `SessionEntry.session_keys` (l.327) already maps a
  job to its `<mirror>--<target>` records: the bridge from a Recipient to
  its RemoteSession.
- `sucoder/cli.py`: `message` (l.4885, the `--wait` loop polls
  `capture_pane_command` every `_MESSAGE_POLL_SECONDS`, sleeps through
  `_message_sleep`), `EXIT_UNAVAILABLE = 3` (l.5083), `_session_relay_applies`,
  `_delegate_to_session_relay` (forwards argv; must forward `--reply`),
  `relay_app` with `relay_main` (HOME/config refusal), `_relay_sender`
  (reads `SUDO_USER`), `relay_message`, `relay_peek`, `peek` (l.5284),
  `_plan_session_recipients`, `_control_for_recipient`,
  `_capture_over_tunnel` (l.4014; a nonzero remote rc is not a transport
  failure).
- `sucoder/session.py:RemoteSession`: optional fields loaded with
  `data.get`, saved atomically; legacy files load blank. `managed_owner`
  carries `scope` and `durable_root` (the managed-Codex identity).
- `sucoder/mirror.py:launch_agent` (l.3480): `_detect_agent_type`
  (l.7853) gives `AgentType.CLAUDE`; argv is built before nvm/tmux wrapping.
  The "record only when the tmux session is created" pattern exists for
  `managed_owner` (has-session probe at l.3835, `_record_managed_owner`
  l.3912; confined record at l.3427).
- `sucoder/codex_state_runner.py`: run layout
  `DURABLE/codex/<scope>/runs/<run>/{run.json,rollouts/<thread>.jsonl}`,
  `run.json.allocation = {"job": "<SLURM_JOB_ID as string>", "node": ...}`
  (l.3580); `status.json["replica"]` with `lag_bytes` and per-thread
  `local_bytes/replicated_bytes/relative_path` (l.2262), node-local only.
  Precedent for a 3.6-clean standalone helper and its tests:
  `tests/test_codex_remote_identity.py:99` (`ast.parse(feature_version=(3,6))`),
  `tests/test_codex_state.py:184` (py_compile under python3.6, skipped if absent).
- Tests to keep green: `tests/test_messaging.py` (33), `tests/test_session_relay.py`
  (18), `tests/test_cli.py` message/peek block (`_message_report` l.4845,
  `_run_message`, `test_message_wait_*` l.4950-4988, `_run_peek`).

## R2. Definitions and conventions observed

- Frame: `[message <id> from <sender> via sucoder, <local %Y-%m-%d %H:%M %Z>] <flat text>`.
  Relay sender: `"<from> (via <SUDO_USER>)"` or `"<SUDO_USER>@<host> (relay)"`.
- Remote helpers are standalone stdlib files, 3.6-clean; laptop code 3.9+.
- GitNexus impact (upstream): `message`, `peek`, `extract_reply`,
  `new_message_id`, `RemoteSession`, `launch_agent` LOW;
  `_build_tmux_launch_command` HIGH (7 impacted, 3 processes) -> do not
  edit it; insert `--session-id` at the argv stage in `launch_agent`.
- Tooling: `.venv/bin/pytest` and `.venv/bin/mypy` have a broken shebang
  (`/home/ligon/Projects/Assistant/.venv/bin/python`). Use
  `umask 022; .venv/bin/python -m pytest -q` and `.venv/bin/python -m mypy`.

## R3. Facts added by this pass (lyn, shape-only)

- Claude 2.1.x turn-end dequeue (charter section 5, first risk): observed 3
  times in the lyn session. Shape: `queue-operation enqueue` (content =
  frame) -> `queue-operation dequeue` (no content) -> ... ->
  `type: user` with string content byte-equal to the enqueued text. So the
  `user` form is the dequeued delivery; `queued` must mean "enqueue seen and
  no delivery of any shape", not "no remove/attachment".
- Mid-turn absorb confirmed again (8 times): enqueue -> remove
  (`absorbed_mid_turn`) -> `attachment queued_command` with `prompt` == frame.
- Codex rollout top keys `{ordinal, payload, timestamp, type}`;
  `session_meta.payload.thread_source == "user"`; non-sucoder user messages
  (AGENTS/environment preambles, `<send_user_message_question_reply>`)
  are `role: user` items too, so "stop at next delivery of any input" may
  close a window early (gives `seen`, never a wrong reply: acceptable).
  `task_complete` carries `turn_id`; `compacted.payload.replacement_history`.

## R4. Invariants the implementation must keep

- Never `replied` without a tagged line in an assistant text block (or a
  valid mailbox file); never `absent` without an entry stamped after
  send-minus-skew.
- Existing screen-path tests pass unchanged: in them the recipient has
  `pane="claude"` and NO session record, and a pane timeout exits 0. So
  the adapter must be chosen from a recorded identity (session record
  matched by `session_keys`/mirror--target and the same job id), not from
  the pane name; no identity -> `no-adapter` -> screen path with today's
  exit semantics (0 on timeout). `test_messaging.py:207` (id length 4) is
  the one test the documented change rewrites.
- The pane name is not a reliable harness signal anyway: managed Codex runs
  under `python3` (the runner); Claude's exe is a versioned file.
- Record the Claude uuid only when SuCoder creates the session (same rule
  as `managed_owner`); a reattach keeps the old uuid. A stale or missing
  uuid file, or one with nothing newer than the send, must fall through to
  the design's step-4 fallback rather than end the search.
- `codex-remote-control` recipients stay refused for `message`.
- Relay: the sent log is the relay account's; each line records the
  caller (`SUDO_USER`), and `relay peek --reply` refuses others' ids.
  Sanitise and cap in the printing process, which in the relay is the
  relay process itself, before output reaches the caller.
- Mixed recipients: the exit is the worst status. "Worst" needs an explicit
  order; proposed 3 (could not look) > 1 (ambiguous / delivery failure) >
  4 (no answer) > 0, with a screen-fallback recipient contributing 0 or 1
  as today.

## R5. Reuse decisions

| quantity | decision | why |
|---|---|---|
| message id | extend `new_message_id` to `token_hex(4)` | one-line change; callers monkeypatch it |
| frame | reuse `frame` | unchanged by charter |
| frame anchoring / delivery match | new pure fn in a reader module | anchored whole-frame, optional `<pasted_content ...>` |
| sent log `~/.sucoder/sent.jsonl` | new small module (append/lookup) | no prior art; mirror `session.py` path style |
| Claude `--session-id` | extend `launch_agent` argv stage + new optional `RemoteSession` field | avoid HIGH-risk `_build_tmux_launch_command` |
| transcript readers (Claude, Codex) | new pure adapters + shared core | no prior art |
| managed-Codex identification | new, reading `run.json`/`rollouts/` layout of the runner | runner is the authority on layout |
| probe | new standalone 3.6 module, shipped zlib+base64 via `python3 -I -c` | precedent: `codex_state_runner.py` tests |
| probe transport | reuse `job_tmux_command` / `_capture_over_tunnel` | already handles srun vs ssh -n |
| mailbox | new, inside the probe | no prior art |
| statuses / exit codes | new table; reuse `EXIT_UNAVAILABLE` (3) | 4 is new |
| sanitising/caps | new pure fn in `messaging.py` | applies in direct and relay printing |
| screen fallback | reuse `capture_pane_command` + `extract_reply` | charter keeps it |
| `peek --reply` / relay restriction | extend `peek`, `relay_peek`, `_delegate_to_session_relay` args | |
| docs | extend `default_system_prompt.org` l.122-130, README `*** Messaging a running session` l.981 | config.example.yaml only if config added |

## R6. Open questions (non-blocking; engine decides and reports)

- Baseline test failures differ from charter section 3. Full run at
  `60f25b9` (295s): failed `test_restore_refusals_place_nothing_and_never_start_codex`,
  `test_a_real_codex_thread_resumes_from_a_restored_generation`,
  `test_a_durable_root_below_absent_directories_is_created` (NOT listed);
  `test_restore_places_a_sealed_generation_in_another_allocation` and
  `test_reuse_keeps...` passed. A targeted rerun failed only the first.
  All are in `tests/test_codex_state_runner.py`, which this task does not
  touch; they are order/timing-dependent. Judge the hygiene line as "no
  new failures outside that file's restore/reuse tests, reproduced on the
  baseline".
- Mailbox under `<mirror>/.sucoder/mail` when the agent's cwd is a
  node-local clone (`/local/job<J>/mirrors/<token>`): readable only by the
  job route; a login-host probe reads `~/.sucoder/mail` only.
- `CLAUDE_CONFIG_DIR` via `/proc` is only visible on the agent's node; a
  login-host probe uses the default `~/.claude` unless the record says
  otherwise.
- Renewal replacement records: whether the new uuid rides in
  `replacement` and is promoted, or is simply written on the successor's
  launch; fallback covers the gap either way.

---

# Implementation notes - Message replies from the agent's transcript (2026-10-23)

Engine implementation pass on `feat/reply-from-transcript`, after the
prior-art section above. Decisions taken where the charter or design left
room:

- **Where the reader lives.** All transcript logic (adapters, shared reader,
  identification, mailbox) is in the standalone probe `sucoder/reply_probe.py`
  (stdlib, Python 3.6) because it must run on the cluster; the tests import
  its pure functions. `sucoder/replies.py` is the laptop half (sent log,
  identity from the session record, shipped command, status/exit table).
- **Frame match.** The anchored match is against the frame *header*
  (`[message <id> from <sender> via sucoder, <time>]`, `messaging.frame_header`),
  which is what the design calls the whole frame; the body is not compared.
- **Freshness** (`absent` vs `unrecognized`): any entry stamped at or after
  send minus 30s, in the parsed region or in up to 256 KB before the frame's
  first copy.
- **Probe route.** Login host first (shared home, managed-Codex replica);
  if nothing is identified there and the recipient is a job, inside the job
  via `job_tmux_command`, and it stays there.
- **Screen fallback.** No identity -> screen only (today's exit codes).
  `no-adapter`, `unrecognized` and `unreadable` also read the pane each poll;
  a screen reply wins (0), otherwise the probe status stands.
- **Worst-of order** 3 > 1 > 4 > 0, as proposed above; failed deliveries
  contribute 1 and unreachable clusters 3, as before.
- **Recorded identity.** New optional `RemoteSession.harness` and
  `claude_session_id`. Unmanaged Codex is identified only when `harness:
  codex` was recorded, which happens on the confined path (no tmux probe was
  added to unconfined non-Claude launches).
- **`_launch_confined`** (GitNexus HIGH) got one additive line recording the
  identity beside `managed_owner`; `_build_tmux_launch_command` is untouched.
- **Test isolation.** No test fixture isolated HOME, so `tests/conftest.py`
  now redirects `replies.sent_log_path`; one early run (before that fixture)
  created the real `~/.sucoder/sent.jsonl` with 7 fixture lines, which was
  deleted.
- **mypy.** `cli.py` carries 66 pre-existing errors (typer/click Context
  typing); none on changed lines. `mirror.py`'s two pre-existing errors were
  fixed (typing only).
