"""Synthetic rollout discovery, rendering, and cached reader jobs."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from sucoder import cli, dream_codex_sessions as inventory, dream_codex_view as view, dream_notes, dream_view
from sucoder import dream_reader_job as reader
from sucoder.config import CodexStateConfig, RemoteConfig, SlurmConfig
from sucoder.dream import DreamError
from tests.test_dream_reader_job import (NativeV3, OPTIONS, SID, age, at, clock, item, lines, mirror, no_live,
                                        no_real_native, record, runner, saved, v4_script, workspace)

TID = "aabbeeff-0102-0304-0506-0708090a0b0c"
STAMP = "2026-11-28T10:13:14Z"


@pytest.fixture(autouse=True)
def synthetic_processes(tmp_path, monkeypatch):
    scan = inventory.live_threads
    monkeypatch.setattr(inventory, "live_threads", lambda: scan(tmp_path / "proc"))


def entry(kind, **payload):
    return dict(timestamp=STAMP, type=kind, payload=payload)


def meta(cwd, tid=TID, **fields):
    defaults = dict(id=tid, cwd=str(cwd), thread_source="user", source="vscode", originator="interactive")
    return entry("session_meta", **dict(defaults, **fields))


def message(role, texts, authors=(), kinds=(), **extra):
    content = [dict(type="output_text" if role == "assistant" else "input_text", text=text) for text in texts]
    tags = dict(content_item_metadata=[dict(provenance=dict(type=p)) if p else {} for p in authors],
                content_item_kinds=list(kinds))
    return entry("response_item", type="message", role=role, content=content,
                 internal_chat_message_metadata_passthrough=tags, **extra)


def rollout(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lines(records))
    age(path)
    return path


def managed(root, tid=TID, scope="a" * 64):
    run = root / "codex" / scope / "runs/sample-run"
    run.mkdir(parents=True, exist_ok=True)
    (run / "run.json").write_text(json.dumps(dict(scope=scope, run=run.name, allocation=dict(job="42"))))
    return run / "rollouts" / (tid + ".jsonl")


def local(root, tid=TID):
    return root / "2026/11/28" / ("rollout-2026-11-28T08-00-00-" + tid + ".jsonl")


def test_inventory_scopes_exclusions_quiet_and_complete_cut(tmp_path, monkeypatch):
    repo = tmp_path / "mirrors/sample"
    source = SimpleNamespace(path=repo, name="sample")
    durable = tmp_path / "configured"
    disk = tmp_path / "disk"
    target = RemoteConfig(slurm=SlurmConfig("partition", "account", local_disk=str(disk),
                                           codex_state=CodexStateConfig(durable_root=str(durable))))
    config = SimpleNamespace(targets=dict(cluster=target), mirrors={})
    local_home = tmp_path / "home/.codex/sessions"
    good = rollout(managed(durable), [meta(repo)])
    pinned = good.stat().st_size
    with good.open("ab") as stream:
        stream.write(b"partial")
    age(good)
    rollout(local(local_home), [meta(repo), message("user", ["unmanaged duplicate"])])
    ids = [f"{n:08x}" + TID[8:] for n in range(1, 11)]
    cases = [meta(repo, ids[0], thread_source="subagent", source={"subagent": {}}),
             meta(repo, ids[1], forked_from_id=TID), meta(repo, TID), meta(repo.parent, ids[3]),
             meta(repo, ids[4]), meta(repo, ids[5]), meta(repo, ids[6]), meta(repo, ids[7]),
             meta(repo, ids[8]), meta(repo, ids[9], thread_source=None)]
    for n, record in enumerate(cases):
        # A parent's session_meta on line two must never admit this fork.
        path = rollout(managed(durable, ids[n], "b" * 64), [record, meta(repo)] if n == 1 else [record])
        if n == 6:
            age(path, 1)
    other = "eeeeeeee" + TID[8:]
    rollout(local(local_home, other), [meta(repo.parent / "worktree", other, source="exec", originator="codex_exec")])
    unmanaged = "dddddddd" + TID[8:]
    rollout(local(local_home, unmanaged), [meta(repo, unmanaged)])
    # A valid-looking later line cannot rescue a file without a session_meta first.
    rollout(local(local_home, "cccccccc" + TID[8:]), [message("assistant", ["prelude"]), meta(repo)])
    options = dict(config=config, home=tmp_path / "home", live=[ids[5]], excluded=[ids[4][:8]])
    rows = {r["thread_id"]: r for r in inventory.sessions(source, **options)}
    assert set(rows) == {TID, unmanaged, *[ids[i] for i in (0, 1, 4, 5, 6, 7, 8, 9)]}
    assert (rows[TID]["path"], rows[TID]["bytes"], rows[TID]["state"]) == (str(good), pinned, "finished")
    assert rows[TID]["modified"] == good.stat().st_mtime
    assert rows[ids[5]]["state"] == "live" and rows[ids[6]]["state"] == "recent"
    assert [(rows[ids[n]]["state"], rows[ids[n]]["reason"]) for n in (0, 1, 4, 9)] == [
        ("excluded", "subagent"), ("excluded", "fork"), ("excluded", "dream.reader_exclude_sessions"),
        ("excluded", "unknown thread source")]
    run = disk / "job42/state/codex" / ("a" * 64) / "sample-run"
    sessions_dir = run / "home/sessions"
    preferred = rollout(local(sessions_dir), [meta(repo), message("assistant", ["new bytes"])])
    status = dict(relative_path=str(preferred.relative_to(sessions_dir)), local_bytes=20, replicated_bytes=10)
    (run / "status.json").write_text(json.dumps(dict(replica=dict(threads={TID: status}))))
    found = [r for r in inventory.sessions(source, **options) if r["thread_id"] == TID]
    assert [r["path"] for r in found] == [str(preferred)]
    monkeypatch.setattr(inventory, "MAX_SCAN", 20)
    with pytest.raises(DreamError, match="allowance"):
        inventory.sessions(source, **options)
    monkeypatch.setattr(inventory, "MAX_ENTRIES", 0)
    with pytest.raises(DreamError, match="entry allowance"):
        inventory.sessions(source, **options)


@pytest.mark.parametrize("source,originator,typed", [
    ("vscode", "interactive", "human"), ("exec", "interactive", "unattributed"), ("vscode", "codex_exec", "unattributed")])
def test_codex_render_origin_per_item_and_exact_offsets(source, originator, typed):
    roles = ["user", "harness", "agents_md", "tool", "developer_instructions", "skills", None]
    texts = [f"item {i}" for i in range(len(roles))] + ["missing metadata", "[message 0123abcd from Sue via sucoder, now] hello"]
    kinds = ["user.text", "goal.internal_context", "agents_md.instructions", "tool", "instructions", "skills"]
    records = [meta("/project", source=source, originator=originator), message("user", texts, roles, kinds),
               message("developer", ["harness instructions"]), message("assistant", ["progress"], phase="commentary"),
               message("assistant", ["done"], phase="final_answer"),
               entry("response_item", type="agent_message", author="worker", recipient="root",
                     content=[dict(type="text", text="worker answer")])]
    raw = lines(records)
    text, mapping = dream_view.render(raw, TID, 0, view.CONTRACT)
    blocks = mapping["blocks"]
    assert [b["origin"] for b in blocks] == [typed] + ["harness"] * 5 + ["unattributed"] * 2 + [
        "agent", "harness", "agent", "agent", "agent"]
    assert "HARNESS goal.internal_context\nitem 1" in text and "AGENT worker to root\nworker answer" in text
    assert "AGENT commentary\nprogress" in text and "AGENT final_answer\ndone" in text
    for block, body in zip(blocks, texts):
        assert block["offset"] == len(lines(records[:1]))
        delivered = text.encode()[block["start"]:block["end"]]
        assert delivered.endswith(("\n" + body + "\n\n").encode())
        citation = dict(byte_start=block["start"], byte_end=block["end"])
        assert dream_view.user_quote(mapping, citation) == (block["origin"] == "human")


def test_render_clips_after_redaction_and_omits_mirrors_and_replacement_history():
    secret = "-----BEGIN PRIVATE KEY-----\n" + "s" * 1000 + "\n-----END PRIVATE KEY-----"
    value = "begin " + "x" * 1100 + secret + "END" * 200
    records = [meta("/repo"), dict(message("user", ["copy"], ["user"]), metadata=dict(inherited_user_message=True)),
               entry("compacted", message="one summary", replacement_history=[message("user", ["copy"])]),
               message("developer", [value]), entry("response_item", type="function_call", name="shell", arguments=value),
               entry("response_item", type="custom_tool_call", name="patch", input=value)]
    records.extend(entry("response_item", type=kind, output=value)
                   for kind in ("function_call_output", "custom_tool_call_output"))
    for kind in ("event_msg", "turn_context", "token_usage_record", "world_state", "inter_agent_communication_metadata"):
        records.append(entry(kind, type="item_completed", text="hidden"))
    for kind in ("reasoning", "additional_tools", "configuration_update"):
        records.append(entry("response_item", type=kind, content="hidden"))
    text, mapping = dream_view.render(lines(records), TID, 0, view.CONTRACT)
    assert text.count("one summary") == 1 and "copy" not in text and "hidden" not in text
    assert [b["origin"] for b in mapping["blocks"]] == ["summary", "harness", "tool", "tool", "tool", "tool"]
    assert mapping["redactions"] == 5 and mapping["dropped_records"] == 10
    assert "s" * 10 not in text and "TOOL CALL shell" in text and "TOOL CALL patch" in text
    assert text.count("bytes omitted") == 5
    for block, (head, tail) in zip(mapping["blocks"][1:], [(960, 480), (300, 0), (300, 0), (240, 120), (240, 120)]):
        expected = dream_view._clip(value.replace(secret, "[REDACTED]"), head, tail, dream_view.CONTRACT)
        assert text.encode()[block["start"]:block["end"]].endswith(("\n" + expected.strip("\n") + "\n\n").encode())


def test_full_job_and_tail_are_bound_to_the_original_rollout(mirror, monkeypatch, tmp_path):
    repo, config_path, config, _ = mirror
    question = message("user", ["Keep the timeout."], ["user"])
    answer = message("assistant", ["I kept the timeout."])
    path = rollout(managed(repo.parent / ".sucoder-state"), [meta(repo), question, answer])
    process = tmp_path / "proc/123"
    (process / "fd").mkdir(parents=True)
    (process / "exe").symlink_to("/bin/codex")
    descriptor = process / "fd/17"
    descriptor.symlink_to(local(tmp_path / "sessions"))
    with pytest.raises(DreamError, match="is live"):
        reader.prepare(config, cwd=repo, session=TID, source="codex", overrides=OPTIONS)
    descriptor.unlink()
    # Automatic selection must find this thread even with a finished Claude session present.
    age(path, 7200)
    decision = item("decision", "The human chose the timeout.", [(at(path, "Keep the timeout"), "Keep the timeout.")],
                    "human said", "human", "2026-11-28", by="human")
    decision["citations"][0]["role"] = "support"
    native = NativeV3({0: [decision]})
    result = reader.start(config, cwd=repo, overrides=OPTIONS, transport=native)
    assert result["state"] == "accepted" and result["source"] == ["codex", TID, ""]
    assert native.calls[0]["input"]["session"]["harness"] == "codex"
    job = Path(result["job_path"])
    header = saved(job, "job.json")
    assert (header["reader_contract"], header["source_harness"], header["render_contract"]) == (
        reader.READER_V5, "codex", view.CONTRACT)
    store = dream_notes.Store(Path(result["store"])).source("codex", TID)
    note = store.notes(accepted_only=True)[0]["note"]
    assert note["source"] == result["source"] and note["contract"] == dream_notes.NOTE_V2
    assert note["items"][0]["citations"][0]["quote"] == "Keep the timeout."
    end = path.stat().st_size
    assert store.covered() == [[0, end]]
    assert reader.run(config, job) == result
    assert reader.prepare(config, cwd=repo, session=TID, overrides=OPTIONS)["state"] == "covered"
    with pytest.raises(DreamError, match="immutable"):
        reader.prepare(config, job, source="claude")
    with pytest.raises(DreamError, match="in this mirror"):
        reader.prepare(config, cwd=repo, session=TID, source="claude", overrides=OPTIONS)
    tail = lines([message("assistant", ["tail fact"])])
    with path.open("ab") as stream:
        stream.write(tail)
    age(path)
    monkeypatch.chdir(repo)
    cli_args = ["--config", str(config_path), "dream", "read-session", "--prepare-only", "--session", TID,
                "--source", "codex", "--harness", "codex", "--model", "fixture-model"]
    command = runner.invoke(cli.app, cli_args)
    assert command.exit_code == 0, command.output
    prepared = json.loads(command.output)
    assert prepared["ranges"] == [[end, end + len(tail)]]
    second = Path(prepared["job_path"])
    fact = item("fact", "The agent added a fact.", [(end, "tail fact")], "agent reported", "agent")
    fact["citations"][0]["role"] = "support"
    completed = reader.run(config, second, transport=NativeV3({0: [fact]}))
    assert completed["state"] == "accepted" and store.covered() == [[0, end + len(tail)]]
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"vscode", b"editor"))
    with pytest.raises(DreamError, match="session_meta changed"):
        reader.run(config, second)
    path.write_bytes(raw)
    assert reader.run(config, second) == completed
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(raw)
    replacement.replace(path)
    with pytest.raises(DreamError, match="replaced"):
        reader.run(config, second)


def test_codex_chunks_use_turn_starts_and_bind_even_a_metadata_free_window(monkeypatch):
    records = [meta("/project")]
    for n in range(2):
        records += [entry("event_msg", type="task_started"), message("user", ["request"], ["user"]),
                    message("assistant", [str(n) * 400])]
    raw = lines(records)
    binding = view.identity(lines(records[:1]), TID)
    chunks = reader._chunks(raw, 0, TID, 1024, view.CONTRACT, binding)
    assert [c["byte_start"] for c in chunks] == [0, len(lines(records[:4]))]
    assert all(c["bytes"] <= 1024 for c in chunks)
    tail = lines(records[4:])
    with pytest.raises(DreamError, match="identity"):
        dream_view.render(tail, TID, len(raw) - len(tail), view.CONTRACT)
    with pytest.raises(DreamError, match="identity"):
        dream_view.render(raw, SID, 0, view.CONTRACT)
    with pytest.raises(DreamError, match="identity"):
        dream_view.render(tail, SID, len(raw) - len(tail), view.CONTRACT, binding=binding)
    with pytest.raises(DreamError, match="identity"):
        dream_view.render(lines([meta("/repo", thread_source="subagent"), meta("/repo")]), TID, 0, view.CONTRACT)
    monkeypatch.setattr(dream_view, "MAX_RAW_BYTES", 700)
    chunks = reader._chunks(raw, 0, TID, 65536, view.CONTRACT, binding)
    assert all(c["byte_end"] - c["byte_start"] <= 700 for c in chunks)


@pytest.mark.parametrize("old_reader", [reader.READER_V4, reader.READER_V5])
def test_v4_replays_and_v5_claude_produces_identical_chunks_items_and_estimate(mirror, monkeypatch, old_reader):
    repo, _, config, path = mirror
    with monkeypatch.context() as patch:
        patch.setattr(reader, "READER", old_reader)
        # The first rollout adapter prepared v5 Claude jobs with the v3 renderer.
        patch.setitem(reader.SOURCE_VIEWS, "claude", (dream_view.CONTRACT_V3,))
        legacy = reader.prepare(config, cwd=repo, session=SID, overrides=OPTIONS)
    old_job = Path(legacy["job_path"])
    old = saved(old_job, "job.json")
    fresh_config = replace(config, log_dir=config.log_dir.parent / "v5-logs")
    fresh = reader.prepare(fresh_config, cwd=repo, session=SID, overrides=OPTIONS)
    new_job = Path(fresh["job_path"])
    new = saved(new_job, "job.json")
    assert (old["render_contract"], new["render_contract"]) == ("claude-view-v3", "claude-view-v4")
    for key in ("gaps", "estimate", "session", "settings"):
        assert old[key] == new[key]
    assert len(old["chunks"]) == len(new["chunks"])
    for prev, current in zip(old["chunks"], new["chunks"]):
        assert prev["render_sha256"] != current["render_sha256"]  # The map names its contract.
        assert dict(prev, render_sha256=current["render_sha256"]) == current
    assert old.get("source_harness") == ("claude" if old_reader == reader.READER_V5 else None)
    assert new["source_harness"] == "claude"
    assert reader._views(new, time.monotonic() + 10) == reader._views(old, time.monotonic() + 10)
    values = v4_script(path)
    result = reader.run(config, old_job, transport=NativeV3(values))
    assert reader.run(config, old_job) == result
    current = reader.run(fresh_config, new_job, transport=NativeV3(values))
    assert current["state"] == result["state"] == "accepted"
    stored = [dream_notes.Store(Path(r["store"])).source("claude", SID).notes(accepted_only=True)[0]["note"]
              for r in (result, current)]
    assert dict(stored[0], written_at=STAMP, render_contract=new["render_contract"]) == dict(stored[1], written_at=STAMP)
    assert reader.run(fresh_config, new_job) == current


def test_v5_reader_refuses_a_human_decision_from_a_claude_relay(mirror):
    repo, _, config, path = mirror
    quote = "Keep the timeout at 30 seconds."
    path.write_bytes(lines([record("user", "[message from Sue via sucoder, now] " + quote, STAMP, 1,
                                   origin=dict(kind="human"))]))
    age(path)
    prepared = reader.prepare(config, cwd=repo, session=SID, source="claude", overrides=OPTIONS)
    job = Path(prepared["job_path"])
    header = saved(job, "job.json")
    assert (header["reader_contract"], header["render_contract"]) == (reader.READER_V5, "claude-view-v4")
    assert reader._records(path.read_bytes(), 0, header["render_contract"])[0]["turn"]
    decision = item("decision", quote, [(0, quote)], "human said", "human", by="human")
    decision["citations"][0]["role"] = "support"
    native = NativeV3({0: [decision]})
    with pytest.raises(reader.Rejected, match="no item of a chunk survives"):
        reader.run(config, job, transport=native)
    assert [task["role"] for task in native.calls] == ["reader"]
    assert "AGENT (sucoder relay)" in native.calls[0]["input"]["chunk"]["text"]
    assert "origin agent" in saved(job, "dropped-000001-000000.json")["dropped"][0]["reason"]
    source = dream_notes.Store(Path(header["store"])).source("claude", SID)
    assert source.notes() == [] and source.coverage() == []


def test_codex_typed_provenance_yields_to_a_complete_leading_relay_only():
    header = "[message from Sue via sucoder, now]"
    texts = [" \n<pasted_content length='32'>\t" + header + " peer",
             "I received " + header, "[message 0123abcd from Sue via sucoder]"]
    raw = lines([meta("/repo"), message("user", texts, ["user"] * len(texts))])
    text, mapping = dream_view.render(raw, TID, 0, view.CONTRACT)
    assert [b["origin"] for b in mapping["blocks"]] == ["agent", "human", "human"]
    assert text.count("AGENT (sucoder relay)") == 1


def test_codex_refuses_misattributed_paraphrased_and_summary_quotes(mirror):
    repo, _, config, _ = mirror
    records = [meta(repo), message("user", ["harness only", "typed"], ["agents_md", "user"]),
               message("user", ["[message 0000aaaa from Bob via sucoder, now] peer"], ["user"]),
               entry("compacted", message="summary only", replacement_history=[])]
    path = rollout(managed(repo.parent / ".sucoder-state"), records)
    job = reader.prepare(config, session=TID, cwd=repo, overrides=OPTIONS)
    header = saved(Path(job["job_path"]), "job.json")
    rendered = reader._views(header, time.monotonic() + 10)[0]
    for marker, quote, error in [("harness only", "harness only", "origin harness"), ("typed", "Typed", "does not match"),
                                 ("via sucoder", "peer", "origin agent"), ("summary only", "summary only", "summary")]:
        value = item("decision", "A choice.", [(at(path, marker), quote)], "human said", "human", by="human")
        value["citations"][0]["role"] = "support"
        with pytest.raises(reader.Rejected, match=error):
            reader._resolve(value, rendered, header["chunks"][0], "item", reader.READER_V5)
