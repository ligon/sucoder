# Workshop Charter — Message replies from the agent's transcript

> Agreed with the human in phases 0–1 on 2026-10-23. The engine (phases 2–6)
> treats §2a as the oracle it loops against. §2b is decided after the engine,
> by the lyn agent and the human, because it needs live cluster sessions.
> The design is `docs/reply-from-transcript.org` (draft 3). The evidence is
> `docs/reply-transcripts-evidence.org`.

## §0 Question, restated

A `sucoder message` sender should learn, reliably and promptly, two things:
whether the recipient agent took the message in, and what it answered. This
must not depend on what the agent's screen happens to show. It must work for
Claude Code and Codex recipients, both directly and through the `coder` sudo
relay. It must never report a reply that wasn't made, and never report
"absent" when the truth is "couldn't look".

Premises challenged in phase 0 and how they were resolved:
- *"Harness-agnostic"* cannot literally hold for transcripts, which are
  private per-harness formats. Resolution: a shared reader core plus one
  small adapter per harness. A mailbox, the only truly agnostic channel, is
  also checked.
- *Reading better does not make agents answer.* LSMS_Library took in
  message 1733 and ignored it. Resolution: report `seen`, with a new exit
  status, and strengthen the system-prompt wording. Making agents comply is
  not in scope.
- *The engine cannot run live checks.* Resolution: §2b is done after the
  engine, outside it.

## §1 Scope & non-goals

**In scope (one slice):**
- 8-hex message ids. The frame is otherwise unchanged.
- A local sent-message log, `~/.sucoder/sent.jsonl`.
- Matching the frame anchored at the start of the message, using the whole
  frame.
- Claude launched by SuCoder with `--session-id <uuid>`, and the uuid
  recorded in `RemoteSession`.
- Pure Claude and Codex adapters and a shared reader with fixture tests.
- Managed-Codex identification from runner metadata and `thread_source`.
- A standalone remote probe, Python 3.6 compatible and shipped base64-encoded
  over the existing routes, that emits one fixed-schema JSON line.
- The mailbox check.
- Statuses, and exit codes 0/1/3/4.
- Laptop-side sanitising and caps, in both the direct and the relay paths.
- `message --wait` wired to the probe, with fallback to the screen reader.
- `peek --reply <id>`, plus `relay peek --reply` limited to the caller's
  own ids.
- `default_system_prompt.org` and `README.org` updates. `config.example.yaml`
  is updated only if config is added.

**Out of scope / non-goals:**
- Changing inbound delivery (`tmux send-keys` stays).
- Removing the screen reader (`messaging.extract_reply` stays as the
  fallback).
- Inferring answers from untagged assistant text.
- Pruning mailbox files.
- Making agents comply.
- IRC or any other bus.
- Claude's peer socket.
- Byte-offset caching between polls.
- Harnesses other than Claude Code and Codex. Those get `no-adapter`, which
  means the screen fallback.
- Running anything against a real cluster, pushing, or opening a PR.

## §2a Definition of done — engine oracle (each line decidable by a test or command)

The work is complete when ALL of the following hold on branch
`feat/reply-from-transcript`.

**Reader, using fixture tests built from real lines.** The lines come from
`docs/reply-transcripts-evidence.org` and the shapes listed in the design,
trimmed and anonymised where needed.

- [ ] **Claude** fixtures produce the right status and reply:
  - [ ] Idle delivery (`type: user`, string content).
  - [ ] Pasted delivery (`<pasted_content id=…>` wrapper).
  - [ ] Mid-turn delivery (`queued_command` attachment).
  - [ ] Queue-only → `queued`.
  - [ ] A compaction summary (`isCompactSummary: true`) claiming `REPLY <id>: ok` → NOT `replied`.
  - [ ] `api-request-blob` and `last-prompt` copies, which are ignored.
  - [ ] A `tool_result` holding the frame, which is not a delivery.
  - [ ] A `tool_use` input holding `REPLY <id>:`, which is not a reply.
  - [ ] A `REPLY <id>:` line inside a longer text block → `replied`.
  - [ ] A bare `REPLY:` → counts only with no other input delivered between.
  - [ ] Stop at the next delivered input, so a later reply is not credited.
  - [ ] Sidechain entries, which are ignored.
- [ ] **Codex** fixtures produce the right status and reply:
  - [ ] Main-thread delivery (`response_item` user message).
  - [ ] A `phase: commentary` reply mid-turn → `replied`.
  - [ ] A `compacted.replacement_history` replay, which is not a delivery.
  - [ ] A forked sub-agent thread (`thread_source: subagent`), which is not chosen.
  - [ ] `agent_message` items, which are ignored.
- [ ] **Frame anchoring:** a delivery whose body embeds another sender's
  frame after other text does not match that frame. A test covers this.
- [ ] **`absent`** is returned only when at least one parsed entry is stamped
  after the send time minus the skew allowance. Otherwise the result is
  `unrecognized`. There are tests for both.
- [ ] **Turn end:** when the turn has ended after the delivery (Claude
  `stop_reason: end_turn`, Codex `task_complete`), the reader reports
  `seen` with `turn_ended: true`. A test covers it.
- [ ] **Partial last line:** a final line without a newline is skipped, not
  counted as a parse error. A test covers it.

**Probe and transport:**
- [ ] The probe is a standalone module file inside the package (like
  `codex_state_runner.py`). It passes
  `ast.parse(src, feature_version=(3, 6))`, and `py_compile` under
  `python3.6` when one is installed (skipped otherwise).
- [ ] The shipped command
  (`python3 -I -c "import base64,zlib;exec(...)"` with base64 JSON args)
  runs locally, through `bash -c` with the quoting `job_tmux_command`
  applies. Run against fixture files, it prints exactly one JSON line
  carrying `schema` and `status`. A test executes it.
- [ ] Any exception inside the probe yields `{"status": "unrecognized", …}`,
  never a traceback on stdout. A test covers it.
- [ ] The probe reads at most `CLAUDE_CONFIG_DIR` and `CODEX_HOME` from any
  `/proc/*/environ`, and nothing else from that file. Shown by code review
  and a unit test.
- [ ] **Managed Codex:** identification from a fixture run directory
  (`run.json`/`status.json`/rollouts with one `user` and one `subagent`
  thread) picks the user thread. A test covers it.
- [ ] **Mailbox:** a regular file owned by the uid, newer than the send, at
  most 64 KB → `replied` with `source: mailbox`. A symlink, a stale file, or
  an oversize file is ignored. There are tests.

**CLI behaviour (mocked transport):**
- [ ] `message` ids are 8 hex characters. Each delivered recipient appends
  one line to `~/.sucoder/sent.jsonl` with id, frame, recipient fields, the
  UTC send time, and the sending account. A test uses a temporary HOME.
- [ ] **`message --wait` exit codes:**
  - [ ] `replied` → 0.
  - [ ] `seen` / `queued` / `absent` at the deadline → 4.
  - [ ] `ambiguous` → 1.
  - [ ] `unreadable`, `unrecognized`, or every poll failing in transport → 3.
  - [ ] With several recipients, the exit is the worst of their statuses.
  - [ ] `no-adapter` falls back to the existing screen path, whose existing
    tests still pass.
  - [ ] Polling stops early at `replied`, and at `seen` with `turn_ended`.
  - [ ] Tests cover each of the above.
- [ ] **`peek --reply <id>`** reads the sent log. An unknown id → a usage
  error. `relay peek --reply` answers only ids whose logged sending account
  is the relay caller (`SUDO_USER`). Tests cover this.
- [ ] **Sanitising:**
  - [ ] Reply text printed by `--wait` and `peek --reply` has C0/C1 controls
    stripped (except `\n` and `\t`), Unicode `Cf` and bidi characters
    stripped, and is capped at 2 KB and 20 lines.
  - [ ] The relay path applies the same treatment.
  - [ ] A test with an OSC 52 payload and a bidi override covers it.
- [ ] **Claude `--session-id`:**
  - [ ] SuCoder-launched Claude sessions get `--session-id <uuid4>`, unless
    the user's command or flags already contain `--session-id`, `--resume`,
    `-r`, `--continue`, or `-c`.
  - [ ] The uuid is stored in a new optional `RemoteSession` field, and
    records without it still load.
  - [ ] Tests cover command building and the round-trip.
- [ ] **Docs:**
  - [ ] `default_system_prompt.org` states: reply with a `REPLY <id>: …`
    line in your own message text as soon as you see the message, and the
    mailbox is optional.
  - [ ] `README.org` documents the statuses, exit 4, `peek --reply`, the
    sent log, the mailbox, and the relay restriction.

**Hygiene:**
- [ ] The full `pytest` suite passes under `umask 022`. The only exceptions
  allowed are the four known pre-existing failures named in §3.
- [ ] `mypy` is clean on every touched module.
- [ ] `node .gitnexus/run.cjs detect-changes --repo sucoder` has been run.
  Any HIGH/CRITICAL impact is named in the engine's report.
- [ ] No commit message contains a closing keyword next to an issue number.

## §2b Post-engine acceptance (lyn agent + human; needs live sessions)

- [ ] **Live checks:**
  - [ ] An idle Claude login session (the Savio helper) → `replied`.
  - [ ] A busy managed-Codex job → `replied` or `seen`, matching the
    transcript.
  - [ ] A busy Claude job → `queued`/`seen`/`replied`, matching the
    transcript.
  - [ ] All three both directly and through the `coder` relay, after the
    root-owned `/usr/local/bin/sucoder` is reinstalled from the branch.
- [ ] **A message the agent ignores** ends `seen` with exit 4, not 0.
- [ ] **Human review** of the diff and the README wording. ← judgement call
  (Ethan)

### §2b results (2026-10-23, lyn agent, branch at 910835c)

- [x] **Busy managed Codex** (CerealDemand, job 40401999):
  - [x] Direct: `replied (transcript): ok.`, exit 0.
  - [x] Through the `coder` relay: the same, exit 0.
  - [x] `peek --reply` gives the same answer, directly and through the relay.
  - [x] `relay peek --reply` for an id that `coder` did not send exits 2.
- [x] **Idle Claude login session** (the Savio helper): `replied (transcript): ok`, exit 0.
- [x] **Busy Claude, through the relay:** the message was absorbed mid-turn
  next to a tool result (the queued path), and the result was
  `replied (transcript)`, exit 0.
- [x] **Ignored message:** `seen` (turn ended), exit 4.
- [ ] **Not exercised live:** a fresh launch with `--session-id`. The
  helper predates this change, so its record was annotated by hand with
  its known session id (backup in `~/.sucoder/sessions/`). `collaborate`
  force-pushes to the mirror and runs `git reset --hard` there, so it was
  not run just to test. The launch path is covered by unit tests only.
- [ ] Human review. ← Ethan

## §3 Constraints & assumptions

- **Budget:** the engine runs at most 2 implement↔red-team iterations.
- **Tools allowed:** read and edit the repo; run pytest, mypy, and the
  GitNexus CLI (`impact` before editing a symbol, `detect-changes` before
  committing). Commit on `feat/reply-from-transcript`.
- **Not allowed:**
  - ssh, srun, or `sucoder` commands against real hosts;
  - pushes or PRs;
  - edits outside the repo;
  - reading transcripts outside the repo, except the two lyn files named in
    the design review, read-only.
- **Shell:** run with `umask 022`. The user's shell is umask 0002, which
  makes files group-writable by `coder`.
- **Must not break:**
  - existing message, peek, and relay behaviour beyond the documented
    changes;
  - the existing `extract_reply` tests;
  - session records written before this change;
  - the remote-control refusal for `codex-remote-control` agents.
- **Known pre-existing test failures** (not caused by this work). The
  list was corrected after the engine run; the original four names were
  stale.
  - `test_codex_app_server.py::test_a_restored_thread_works_in_the_successors_clone`
    fails on `main` too (the bwrap sandbox environment).
  - Order-dependent tests in `test_codex_state_runner.py` (for example
    `test_the_drain_state_machine_in_process`,
    `test_restore_refusals_place_nothing_and_never_start_codex`) pass when
    run alone.
- **Conventions:**
  - match the surrounding code style and comment density;
  - remote helpers are standalone files kept Python 3.6 compatible;
  - laptop code may assume Python 3.9 or newer;
  - pure logic in modules, I/O in the CLI;
  - README.org and config.example.yaml are kept in sync with the parser.

## §4 Links

- Design: `docs/reply-from-transcript.org`.
- Evidence: `docs/reply-transcripts-evidence.org`.
- Prior-art ledger: `.coder/ledger.md`. The engine adds a section for this
  task and does not rewrite others' sections.
- Issue 52 (closed). PRs 55, 56, 61 (relay and the screen reader).
  `sucoder/messaging.py`, `sucoder/sessions_report.py` (`job_tmux_command`,
  `PANE_PROBE_SH`), `sucoder/codex_state_runner.py`, `sucoder/session.py`,
  `sucoder/cli.py` (`message`, `peek`, `relay_app`).

## §5 Open questions & risks

- **The Claude delivery shape when a queued message is dequeued at turn
  end** was not observed (the evidence, section 3). The adapter must
  handle it as either a `user` entry or an attachment, and a fixture should
  cover the `user` form.
- **Unmanaged Codex** identification (by `session_meta.cwd`) has no live
  evidence yet. Implement it, and verify it in §2b if a session exists.
- **`--session-id` and the leftover shell's `--continue`/`/clear`:** these
  create a new transcript file. The reader's fallback must cover this; it
  is a risk for §2b, not a blocker.
- **Exit 4** is a new public exit code. If a script relied on `--wait`
  returning 0 when nothing came back, it changes behaviour. README must
  say so.
