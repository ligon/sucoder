"""Private session-note store (docs/dream/read-and-organize.org, notes-store)."""
import copy
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import pytest

from sucoder import dream, dream_notes
from sucoder.dream import DreamError, canonical, digest
from tests.test_dream import add_registry, git, workspace  # noqa: F401


IDENTITY = "6f1c2b0e-3d4a-4f5b-8c6d-7e8f9a0b1c2d"
SESSION = "dcdc7362-0000-4000-8000-000000000001"
DATA = (b'{"type":"user","message":"Run the dream tests, please."}\n'
        b'{"type":"assistant","message":"Running pytest -n 2 now."}\n'
        b'{"type":"tool_result","content":"56 passed in 3.1s"}\n'
        b'{"type":"assistant","message":"All dream tests passed."}\n')
FIRST = len(DATA.splitlines(keepends=True)[0])
PROBE = ("import fcntl, os, sys\n"
         "fd = os.open(sys.argv[1], os.O_WRONLY)\n"
         "try:\n"
         "    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
         "except BlockingIOError:\n"
         "    sys.exit(1)\n")


def mode(path):
    return stat.S_IMODE(os.lstat(path).st_mode)


def span(data, start, end):
    return [start, end, hashlib.sha256(data[start:end]).hexdigest()]


def cite(data, text):
    start = data.index(text.encode())
    return dict(byte_start=start, byte_end=start + len(text), quote=text)


def note(data=DATA, written_at="2026-11-26T12:00:00Z", text="The dream tests passed."):
    return dict(contract="session-note-v1", source=["claude", SESSION, ""],
                ranges=[span(data, 0, FIRST), span(data, FIRST, len(data))], render_contract="claude-view-v2",
                reader=dict(harness="claude", model="claude-sonnet-4-6"), written_at=written_at, items=[
        dict(kind="fact", text=text, when="2026-11-26", citations=[cite(data, "56 passed")],
             basis="output shows", origin="tool"),
        dict(kind="decision", text="The human asked for the dream tests.", when=None,
             citations=[cite(data, "Run the dream tests")], basis="human said", origin="human", by="human"),
        dict(kind="command", text="Ran the dream tests in parallel.", when=None,
             citations=[cite(data, "Running pytest -n 2"), cite(data, "56 passed")],
             basis="output shows", origin="tool", command="pytest -n 2", outcome="worked"),
        dict(kind="open_question", text="Whether the slow tests ran\nis not shown.", when=None,
             citations=[cite(data, "All dream tests passed.")], basis="agent reported", origin="agent"),
        dict(kind="correction", text="The index should say the tests run in parallel.", when=None,
             citations=[cite(data, "Running pytest -n 2")], basis="agent reported", origin="agent",
             target=".sucoder/wiki/index.org::#testing"),
    ])


def review(verdict="accept", findings=()):
    return dict(verdict=verdict, reviewer=dict(harness="codex", model="gpt-5.5"), findings=list(findings),
                reviewed_at="2026-11-26T13:00:00.250Z")


@pytest.fixture
def store(tmp_path):
    return dream_notes.Store(tmp_path / "logs" / "dream-notes" / IDENTITY)


@pytest.fixture
def session(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_bytes(DATA)
    return path


def leased_note(store, session, value=None):
    value = value or note()
    source = store.source("claude", SESSION)
    with source.lease():
        source.add_coverage(session, value["ranges"])
        return source, source.write_note(value)


# Schema

def test_a_note_with_every_kind_and_a_review_are_accepted():
    value = note()
    assert dream_notes.validate_note(value) is value
    assert dream_notes.validate_review(review("reject", ["The quote is outside the cited range."]))
    for change in (lambda n: n.update(items=[]),
                   lambda n: n.update(source=["codex", SESSION, "subagents/agent-a1.jsonl"]),
                   lambda n: n["items"][1].update(by="agent", origin="agent", basis="agent reported"),
                   lambda n: n["items"][2].update(outcome="unknown", basis="agent reported", origin="agent"),
                   lambda n: n["items"][4].update(target=".sucoder/wiki/topics/testing.org"),
                   lambda n: n["items"][0].update(text="caf\u00e9 " + "x" * 1995),
                   lambda n: n.update(written_at="2026-11-26T12:00:00.123456Z")):
        changed = copy.deepcopy(value)
        change(changed)
        dream_notes.validate_note(changed)


NOTE_REJECTIONS = [
    (lambda n: n.pop("items"), "Invalid object fields"),
    (lambda n: n.update(note_id="0" * 64), "Invalid object fields"),
    (lambda n: n.update(contract="session-note-v2"), "contract"),
    (lambda n: n.update(source=["claude", SESSION]), "harness, source_id, subpath"),
    (lambda n: n.update(source=["Claude Code", SESSION, ""]), "harness"),
    (lambda n: n.update(source=["claude", SESSION, "../x.jsonl"]), "subpath"),
    (lambda n: n.update(ranges=[]), "at least one raw range"),
    (lambda n: n.update(ranges=n["ranges"][::-1]), "sorted"),
    (lambda n: n["ranges"].append(list(n["ranges"][-1])), "sorted"),
    (lambda n: n.update(ranges=sorted(n["ranges"] + [[0, FIRST, "f" * 64]])), "two digests"),
    (lambda n: n["ranges"][0].__setitem__(2, "0" * 63), "raw range"),
    (lambda n: n["ranges"][0].__setitem__(0, True), "raw range"),
    (lambda n: n["ranges"][0].__setitem__(1, 0), "raw range"),
    (lambda n: n.update(render_contract="claude view"), "render_contract"),
    (lambda n: n["reader"].pop("model"), "Invalid object fields"),
    (lambda n: n["reader"].update(harness="Claude"), "harness"),
    (lambda n: n.update(written_at="2026-11-26T12:00:00+00:00"), "UTC"),
    (lambda n: n.update(written_at="2026-02-30T12:00:00Z"), "UTC"),
    (lambda n: n.update(items=n["items"] * 41), "at most 200"),
    (lambda n: n["items"][0].update(text="x" * 2001), "at most 2000"),
    (lambda n: n["items"][0].update(text="   "), "nonempty"),
    (lambda n: n["items"][0].update(text="bell\x07"), "control"),
    (lambda n: n["items"][0].update(kind="summary"), "kind"),
    (lambda n: n["items"][0].update(kind=["fact"]), "kind"),
    (lambda n: n["items"][0].pop("when"), "Invalid object fields"),
    (lambda n: n["items"][0].update(when="26 Nov 2026"), "ISO date"),
    (lambda n: n["items"][0].update(when="2026-13-01"), "ISO date"),
    (lambda n: n["items"][0].update(citations=[]), "at least one citation"),
    (lambda n: n["items"][0]["citations"][0].update(byte_start=FIRST - 2, byte_end=FIRST + 2), "inside one of"),
    (lambda n: n["items"][0]["citations"][0].update(byte_start=len(DATA), byte_end=len(DATA) + 4), "inside one of"),
    (lambda n: n["items"][0]["citations"][0].update(byte_end=n["items"][0]["citations"][0]["byte_start"]), "raw offsets"),
    (lambda n: n["items"][0]["citations"][0].update(quote="q" * 501), "at most 500"),
    (lambda n: n["items"][0]["citations"][0].pop("quote"), "Invalid object fields"),
    (lambda n: n["items"][0].update(basis="output suggests"), "basis"),
    (lambda n: n["items"][0].update(origin="dispatch"), "origin"),
    (lambda n: n["items"][0].update(basis="human said"), "requires origin human"),
    (lambda n: n["items"][0].update(by="agent"), "Invalid object fields"),
    (lambda n: n["items"][1].pop("by"), "Invalid object fields"),
    (lambda n: n["items"][1].update(by="team"), "human or agent"),
    (lambda n: n["items"][1].update(origin="agent", basis="agent reported"), "decision by the human"),
    (lambda n: n["items"][1].update(basis="agent reported"), "decision by the human"),
    (lambda n: n["items"][2].pop("outcome"), "Invalid object fields"),
    (lambda n: n["items"][2].update(outcome="passed"), "worked, failed or unknown"),
    (lambda n: n["items"][2].update(basis="agent reported", origin="agent"), "requires basis output shows"),
    (lambda n: n["items"][2].update(outcome="failed", basis="harness recorded"), "requires basis output shows"),
    (lambda n: n["items"][2].update(command=""), "Command"),
    (lambda n: n["items"][4].pop("target"), "Invalid object fields"),
    (lambda n: n["items"][4].update(target="/etc/passwd"), "correction target"),
    (lambda n: n["items"][4].update(target=".sucoder/wiki/../../x.org"), "correction target"),
    (lambda n: n["items"][4].update(target="README.org"), "correction target"),
    (lambda n: n["items"][4].update(target=".sucoder/wiki/index.org::#bad anchor"), "correction target"),
]


@pytest.mark.parametrize("change,message", NOTE_REJECTIONS)
def test_each_note_rule_is_enforced(change, message):
    value = note()
    change(value)
    with pytest.raises(DreamError, match=message):
        dream_notes.validate_note(value)


def test_a_note_may_not_exceed_256_kib():
    value = note()
    value["items"] = [dict(value["items"][3], text="x" * 2000) for _ in range(150)]
    with pytest.raises(DreamError, match="256 KiB"):
        dream_notes.validate_note(value)


@pytest.mark.parametrize("change,message", [
    (lambda r: r.update(verdict="maybe"), "accept or reject"),
    (lambda r: r.pop("reviewer"), "Invalid object fields"),
    (lambda r: r["reviewer"].update(model=""), "model"),
    (lambda r: r.update(findings="fine"), "findings"),
    (lambda r: r.update(findings=[""]), "finding"),
    (lambda r: r.update(findings=["x" * 2001]), "finding"),
    (lambda r: r.update(reviewed_at="yesterday"), "UTC"),
    (lambda r: r.update(note_id="0" * 64), "Invalid object fields"),
])
def test_each_review_rule_is_enforced(change, message):
    value = review()
    change(value)
    with pytest.raises(DreamError, match=message):
        dream_notes.validate_review(value)


# Location, identity and permissions

def test_the_store_is_keyed_by_the_accepted_registry_identity(workspace):
    repo, _, config, _ = workspace
    identity = str(uuid.uuid4())
    add_registry(repo, identity)
    source = dream.GitSource(config, "sample", time.monotonic() + 30)
    store = dream_notes.Store.for_repository(config, source, "HEAD")
    assert store.root == Path(config.log_dir).resolve() / "dream-notes" / identity
    assert all(mode(p) == 0o700 for p in (store.root.parent, store.root, store.root / "sources"))
    inside = replace(config, log_dir=repo / "logs")
    with pytest.raises(DreamError, match="outside the inspected repository"):
        dream_notes.Store.for_repository(inside, source, "HEAD")
    assert not (repo / "logs").exists()


def test_a_provisional_repository_identity_is_refused(workspace):
    _, _, config, _ = workspace
    source = dream.GitSource(config, "sample", time.monotonic() + 30)
    with pytest.raises(DreamError, match="provisional"):
        dream_notes.Store.for_repository(config, source, "HEAD")
    assert not (Path(config.log_dir) / "dream-notes").exists()


def test_store_root_defaults_to_the_sucoder_logs_and_needs_a_canonical_uuid(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    unset = SimpleNamespace(log_dir=None)
    assert dream_notes.store_root(unset, IDENTITY) == (tmp_path / ".sucoder/logs").resolve() / "dream-notes" / IDENTITY
    for bad in ("../" + IDENTITY, IDENTITY.upper(), "{" + IDENTITY + "}", "", None):
        with pytest.raises(DreamError, match="canonical repository_uuid"):
            dream_notes.store_root(unset, bad)
    for bad in (tmp_path / "elsewhere" / IDENTITY, tmp_path / "dream-notes" / "not-a-uuid"):
        with pytest.raises(DreamError):
            dream_notes.Store(bad)
    assert not (tmp_path / "elsewhere").exists() and not (tmp_path / "dream-notes").exists()


def test_a_shared_or_foreign_store_is_refused(tmp_path, monkeypatch, session):
    shared = tmp_path / "shared" / "dream-notes"
    shared.mkdir(parents=True)
    shared.chmod(0o750)
    with pytest.raises(DreamError, match="private and account-owned"):
        dream_notes.Store(shared / IDENTITY)
    assert not (shared / IDENTITY).exists()
    store = dream_notes.Store(tmp_path / "logs" / "dream-notes" / IDENTITY)
    source, note_id = leased_note(store, session)
    for path in (store.root, store.root / "sources" / source.key, store.root / "sources" / source.key / "notes"):
        path.chmod(0o755)
        with pytest.raises(DreamError, match="private and account-owned"):
            source.notes()
        path.chmod(0o700)
    stored = store.root / "sources" / source.key / "notes" / (note_id + ".json")
    stored.chmod(0o640)
    with pytest.raises(DreamError, match="private"):
        source.notes()
    stored.chmod(0o600)
    assert len(source.notes()) == 1
    real = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: real + 1)
    with pytest.raises(DreamError, match="private and account-owned"):
        dream_notes.Store(store.root)
    with pytest.raises(DreamError, match="private and account-owned"):
        source.coverage()


# Lease

def test_a_second_lease_on_the_same_source_is_refused_while_the_first_is_held(store):
    first, second = store.source("claude", SESSION), store.source("claude", SESSION)
    subagent = store.source("claude", SESSION, "subagents/agent-a1.jsonl")
    lease = store.root / "sources" / first.key / "lease"
    probe = [sys.executable, "-I", "-c", PROBE, str(lease)]
    with first.lease():
        assert mode(lease) == 0o600 and mode(lease.parent) == 0o700
        with pytest.raises(DreamError, match="busy") as caught:
            with second.lease():
                pass
        assert SESSION in str(caught.value)
        assert subprocess.run(probe, timeout=30).returncode == 1
        with subagent.lease():
            pass
        with pytest.raises(DreamError, match="already holds"):
            with first.lease():
                pass
    assert subprocess.run(probe, timeout=30).returncode == 0
    with second.lease():
        pass


def test_writes_without_the_lease_are_refused(store, session):
    value = note()
    source = store.source("claude", SESSION)
    for write in (lambda: source.add_coverage(session, value["ranges"]), lambda: source.write_note(value),
                  lambda: source.attach_review("0" * 64, review())):
        with pytest.raises(DreamError, match="requires the source lease"):
            write()
    assert source.coverage() == [] and source.notes() == [] and source.record() is None
    with source.lease():
        source.add_coverage(session, value["ranges"])
        note_id = source.write_note(value)
    with pytest.raises(DreamError, match="requires the source lease"):
        source.attach_review(note_id, review())
    assert source.coverage() == value["ranges"]
    with source.lease():
        lease = store.root / "sources" / source.key / "lease"
        lease.unlink()
        lease.touch(mode=0o600)
        with pytest.raises(DreamError, match="no longer exclusive"):
            source.attach_review(note_id, review())
    assert source.notes()[0]["review"] is None


# Coverage

def test_coverage_is_recorded_sorted_unmerged_and_verified_against_the_file(store, session, tmp_path):
    source = store.source("claude", SESSION)
    early, middle, late = span(DATA, 0, 40), span(DATA, 30, 60), span(DATA, 100, 120)
    record = store.root / "sources" / source.key / "coverage.json"
    with source.lease():
        assert source.add_coverage(session, [late, middle, early]) == [early, middle, late]
        before = record.read_bytes()
        assert source.add_coverage(session, [middle]) == [early, middle, late]
        assert record.read_bytes() == before
        for wrong in ([0, 10, "0" * 64], span(DATA + b"tail", len(DATA), len(DATA) + 4), [5, 5, "0" * 64]):
            with pytest.raises(DreamError):
                source.add_coverage(session, [wrong])
        assert record.read_bytes() == before
    assert mode(record) == 0o600 and mode(record.parent / "source.json") == 0o600
    assert source.coverage() == [early, middle, late]
    assert source.covered() == [[0, 60], [100, 120]]
    info = os.stat(session)
    assert source.record() == dict(contract="session-coverage-v1", source=["claude", SESSION, ""],
                                   device=info.st_dev, inode=info.st_ino)
    assert source.verify(session) == dict(size=len(DATA), same_file=True, rewritten=[])
    with source.lease():
        source.add_coverage(session, [span(DATA, 60, 100)])
    assert source.covered() == [[0, 120]]
    rewritten = DATA[:100] + b"X" * 20 + DATA[120:]
    session.write_bytes(rewritten)
    assert source.verify(session)["rewritten"] == [late]
    with source.lease():
        with pytest.raises(DreamError, match="two digests"):
            source.add_coverage(session, [span(rewritten, 100, 120)])
    session.write_bytes(DATA[:50])
    assert source.verify(session)["rewritten"] == [middle, span(DATA, 60, 100), late]
    copied = tmp_path / "copied.jsonl"
    copied.write_bytes(DATA)
    with source.lease():
        source.add_coverage(copied, [span(DATA, 0, 10)])
    assert source.record()["inode"] == info.st_ino
    assert source.verify(copied) == dict(size=len(DATA), same_file=False, rewritten=[])


# Notes and reviews

def test_notes_are_immutable_and_rewriting_one_is_a_no_op(store, session):
    value = note()
    source = store.source("claude", SESSION)
    with source.lease():
        with pytest.raises(DreamError, match="coverage before writing"):
            source.write_note(value)
        source.add_coverage(session, value["ranges"])
        note_id = source.write_note(value)
        assert note_id == digest(canonical(value))
        stored = store.root / "sources" / source.key / "notes" / (note_id + ".json")
        assert stored.read_bytes() == canonical(value) + b"\n"
        assert mode(stored) == 0o600 and mode(stored.parent) == 0o700
        inode = stored.stat().st_ino
        assert source.write_note(copy.deepcopy(value)) == note_id
        assert stored.stat().st_ino == inode
        earlier = note(written_at="2026-11-25T09:00:00Z", text="A different fact.")
        other_id = source.write_note(earlier)
        assert other_id != note_id
        later_id = source.write_note(note(written_at="2026-11-26T12:00:00.5Z", text="A later fact."))
        with pytest.raises(DreamError, match="source differs"):
            source.write_note(dict(value, source=["codex", SESSION, ""]))
    assert [row["note_id"] for row in source.notes()] == [other_id, note_id, later_id]
    assert source.notes()[1] == dict(note_id=note_id, note=value, review=None)
    stored.write_bytes(canonical(earlier) + b"\n")
    with pytest.raises(DreamError, match="differs from its identity"):
        source.notes()
    with source.lease():
        with pytest.raises(DreamError, match="differs"):
            source.write_note(value)
    assert stored.read_bytes() == canonical(earlier) + b"\n"


def test_a_review_attaches_once_and_only_accepted_notes_are_organizer_input(store, session):
    source, accepted = leased_note(store, session)
    with source.lease():
        rejected = source.write_note(note(text="An unsupported fact."))
        pending = source.write_note(note(text="An unreviewed fact."))
        with pytest.raises(DreamError, match="No such session note"):
            source.attach_review("0" * 64, review())
        with pytest.raises(DreamError, match="SHA-256"):
            source.attach_review("../" + accepted[3:], review())
        source.attach_review(accepted, review())
        source.attach_review(accepted, review())
        with pytest.raises(DreamError, match="already has a different review"):
            source.attach_review(accepted, review("reject", ["Changed my mind."]))
        source.attach_review(rejected, review("reject", ["The quote does not support the fact."]))
    stored = store.root / "sources" / source.key / "notes" / (accepted + ".review.json")
    assert stored.read_bytes() == canonical(review()) + b"\n" and mode(stored) == 0o600
    verdicts = {row["note_id"]: row["review"] and row["review"]["verdict"] for row in source.notes()}
    assert verdicts == {accepted: "accept", rejected: "reject", pending: None}
    assert [row["note_id"] for row in source.notes(accepted_only=True)] == [accepted]


def test_sources_lists_recorded_sources_only(store, session):
    leased_note(store, session)
    subagent = store.source("claude", SESSION, "subagents/agent-a1.jsonl")
    with subagent.lease():
        subagent.add_coverage(session, [span(DATA, 0, 10)])
    with store.source("codex", SESSION).lease():
        pass
    assert sorted((s.triple, s.key) for s in store.sources()) == sorted(
        (s.triple, s.key) for s in (store.source("claude", SESSION), subagent))
    stray = store.root / "sources" / ("0" * 64)
    stray.mkdir(mode=0o700)
    (stray / "coverage.json").touch(mode=0o600)
    with pytest.raises(DreamError, match="without its source.json"):
        store.sources()
    (stray / "coverage.json").unlink()
    (store.root / "sources" / "README").touch()
    with pytest.raises(DreamError, match="Unexpected entry"):
        store.sources()


# Escapes and symlinks

@pytest.mark.parametrize("triple", [
    ("claude", SESSION, "/home/coder/.claude/x.jsonl"), ("claude", SESSION, "../x.jsonl"),
    ("claude", SESSION, "subagents/../../x.jsonl"), ("claude", SESSION, "./x.jsonl"), ("claude", SESSION, "a//b"),
    ("claude", SESSION, "subagents/"), ("claude", SESSION, "a\\b"), ("claude", SESSION, "a\nb"),
    ("../claude", SESSION, ""), ("claude", "../" + SESSION, ""), ("claude", "a/b", ""), ("claude", "", ""),
    ("claude", SESSION, None)])
def test_source_keys_refuse_anything_that_could_escape(store, triple):
    with pytest.raises(DreamError):
        store.source(*triple)
    assert os.listdir(store.root / "sources") == []


def test_the_store_never_follows_a_symlink(tmp_path, session):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "dream-notes").symlink_to(elsewhere)
    with pytest.raises(DreamError, match="symlink"):
        dream_notes.Store(linked / "dream-notes" / IDENTITY)
    store = dream_notes.Store(tmp_path / "logs" / "dream-notes" / IDENTITY)
    source = store.source("claude", SESSION)
    directory = store.root / "sources" / source.key
    directory.symlink_to(elsewhere)
    with pytest.raises(DreamError, match="symlink"):
        with source.lease():
            pass
    for read in (source.coverage, source.notes, store.sources):
        with pytest.raises(DreamError, match="symlink"):
            read()
    directory.unlink()
    assert not any(elsewhere.iterdir())
    source, note_id = leased_note(store, session)
    outside = tmp_path / "outside.json"
    cover = lambda: source.add_coverage(session, [span(DATA, 0, 10)])  # noqa: E731
    for name, write in (("coverage.json", cover), ("source.json", cover),
                        ("notes/" + note_id + ".json", lambda: source.write_note(note()))):
        original = directory / name
        saved = original.read_bytes()
        outside.write_bytes(saved)
        outside.chmod(0o600)
        original.unlink()
        original.symlink_to(outside)
        with pytest.raises(DreamError, match="symlink"):
            source.notes()
        with source.lease():
            with pytest.raises(DreamError, match="symlink"):
                write()
        assert outside.read_bytes() == saved
        original.unlink()
        original.write_bytes(saved)
        original.chmod(0o600)
    lease = directory / "lease"
    lease.unlink()
    lease.symlink_to(tmp_path / "decoy")
    with pytest.raises(DreamError, match="symlink"):
        with source.lease():
            pass
    assert not (tmp_path / "decoy").exists()
    assert len(source.notes()) == 1
