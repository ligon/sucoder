"""Private session-note store (docs/dream/read-and-organize.org, notes-store).

Readers record the raw ranges they read and the notes they wrote here, in the
agent account, never on the memory branch; the organizer later consolidates
reviewed notes. Contracts session-note-v1 and session-coverage-v1.

The store checks the note schema and its internal consistency. It does not
hold transcripts, so whether a quote matches its cited bytes, and whether the
cited blocks have the stated origin, is the reader job's check against the
admitted rendering. Nothing here invokes a model or writes git state.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import errno
import functools
import hashlib
import os
from pathlib import Path
import re
import stat
import uuid

from . import dream, dream_run
from .dream import DreamError, canonical, digest


NOTE_CONTRACT = "session-note-v1"
COVERAGE_CONTRACT = "session-coverage-v1"
MAX_NOTE_BYTES = 256 * 1024
MAX_ITEMS = 200
MAX_TEXT = 2000
MAX_QUOTE = 500
# The spec sets no caps for these; they bound what a read may load.
MAX_FINDINGS = 200
MAX_RANGES = 8192
MAX_COVERAGE_BYTES = 1024 * 1024
MAX_RECORD_BYTES = 16 * 1024
READ_BLOCK = 1024 * 1024
# Fields each item kind adds to the common ones.
KINDS = dict(fact=(), decision=("by",), command=("command", "outcome"), open_question=(), correction=("target",))
ITEM_FIELDS = ("kind", "text", "when", "citations", "basis", "origin")
NOTE_FIELDS = ("contract", "source", "ranges", "render_contract", "reader", "written_at", "items")
REVIEW_FIELDS = ("verdict", "reviewer", "findings", "reviewed_at")
RECORD_FIELDS = ("contract", "source", "device", "inode")
BASES = ("output shows", "agent reported", "human said", "harness recorded")
ORIGINS = ("human", "agent", "tool", "harness", "unattributed")
OUTCOMES = ("worked", "failed", "unknown")
VERDICTS = ("accept", "reject")
SHA256 = re.compile(r"[0-9a-f]{64}")
UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,6})?Z")


def _guard(function):
    """Report storage failures, including a refused symlink, as dream errors."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise DreamError("The session-note store refuses a symlink or misplaced file: " + str(exc), 3) from None
            raise DreamError("Session-note store unavailable: " + str(exc), 3) from None
    return wrapper


# Validation (session-note-v1)

def _prose(value, label, limit):
    # Quotes come from renderings, which keep line breaks and tabs.
    if (not isinstance(value, str) or not value.strip() or len(value) > limit or
            any(ord(c) < 32 and c not in "\t\n\r" for c in value)):
        raise DreamError(label + " must be nonempty text of at most " + str(limit) +
                         " characters, without control characters other than line breaks and tabs")
    return value


def _harness(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", value):
        raise DreamError(label + " must be a simple lowercase harness name")
    return value


def _agent(value, label):
    dream._object(value, ("harness", "model"))
    _harness(value["harness"], label + " harness")
    dream._text(value["model"], label + " model")
    return value


def _utc(value, label):
    if isinstance(value, str) and UTC.fullmatch(value):
        try:
            datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S")
            return value
        except ValueError:
            pass
    raise DreamError(label + " must be a UTC ISO 8601 time such as 2026-11-26T12:00:00Z")


def _instant(value):
    """A sortable form of a checked UTC time, whose fractional seconds are optional."""
    return value[:19] + (value[19:-1] or ".").ljust(7, "0")


def _date(value):
    if isinstance(value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        try:
            datetime.strptime(value, "%Y-%m-%d")
            return value
        except ValueError:
            pass
    raise DreamError("Item when must be an ISO date such as 2026-11-26, or null when the session does not establish it")


def _triple(value):
    """The source key (harness, source_id, subpath), as a JSON list."""
    if not isinstance(value, list) or len(value) != 3 or not all(isinstance(part, str) for part in value):
        raise DreamError("A session source is [harness, source_id, subpath]")
    harness, source_id, subpath = value
    _harness(harness, "Source harness")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", source_id):
        raise DreamError("Source id must be a simple session or thread id")
    if subpath:
        try:
            dream._path(subpath)
        except DreamError:
            raise DreamError("Source subpath must be empty or a relative path without empty, . or .. components") from None
    return value


def _range(value):
    if (not isinstance(value, list) or len(value) != 3 or type(value[0]) is not int or type(value[1]) is not int or
            not 0 <= value[0] < value[1] or not isinstance(value[2], str) or not SHA256.fullmatch(value[2])):
        raise DreamError("A raw range is [byte_start, byte_end, raw_sha256] with 0 <= byte_start < byte_end")
    return (value[0], value[1], value[2])


def _consistent(ranges, label):
    extents = {}
    for start, end, sha in ranges:
        if extents.setdefault((start, end), sha) != sha:
            raise DreamError(label + " records one byte range with two digests; the source was rewritten")


def _ranges(value, label, ordered):
    """Raw ranges as tuples; ``ordered`` requires their stored, sorted, unique form."""
    if not isinstance(value, list) or not value:
        raise DreamError(label + " must list at least one raw range")
    result = [_range(item) for item in value]
    if ordered and result != sorted(set(result)):
        raise DreamError(label + " must be sorted, without duplicates")
    _consistent(result, label)
    return result


def _target(value):
    path, separator, anchor = value.partition("::#") if isinstance(value, str) else ("", "", "")
    try:
        dream._path(path)
    except DreamError:
        path = ""
    if (not path.startswith(".sucoder/wiki/") or not path.endswith(".org") or
            (separator and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", anchor))):
        raise DreamError("A correction target is a wiki page path such as .sucoder/wiki/topics/NAME.org, "
                         "with an optional ::#anchor")
    return value


def _item(item, ranges):
    kind = item.get("kind") if isinstance(item, dict) else None
    if not isinstance(kind, str) or kind not in KINDS:
        raise DreamError("Item kind must be one of: " + ", ".join(KINDS))
    dream._object(item, ITEM_FIELDS + KINDS[kind])
    _prose(item["text"], "Item text", MAX_TEXT)
    if item["when"] is not None:
        _date(item["when"])
    if item["basis"] not in BASES:
        raise DreamError("Item basis must be one of: " + ", ".join(BASES))
    if item["origin"] not in ORIGINS:
        raise DreamError("Item origin must be one of: " + ", ".join(ORIGINS))
    # Only human blocks support what the human said (Session sources, origin classes).
    if item["basis"] == "human said" and item["origin"] != "human":
        raise DreamError("Basis human said requires origin human")
    citations = item["citations"]
    if not isinstance(citations, list) or not citations:
        raise DreamError("Each item needs at least one citation")
    for citation in citations:
        dream._object(citation, ("byte_start", "byte_end", "quote"))
        start, end = citation["byte_start"], citation["byte_end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise DreamError("A citation needs raw offsets with 0 <= byte_start < byte_end")
        if not any(low <= start and end <= high for low, high, _ in ranges):
            raise DreamError("A citation must lie inside one of the note's ranges")
        _prose(citation["quote"], "Citation quote", MAX_QUOTE)
    if kind == "decision":
        if item["by"] not in ("human", "agent"):
            raise DreamError("A decision is by human or agent")
        if item["by"] == "human" and (item["origin"] != "human" or item["basis"] != "human said"):
            raise DreamError("A decision by the human requires origin human and basis human said")
    elif kind == "command":
        _prose(item["command"], "Command", MAX_TEXT)
        if item["outcome"] not in OUTCOMES:
            raise DreamError("A command outcome is worked, failed or unknown")
        if item["outcome"] != "unknown" and item["basis"] != "output shows":
            raise DreamError("A command outcome of worked or failed requires basis output shows")
    elif kind == "correction":
        _target(item["target"])


def validate_note(note):
    """Check a session-note-v1 note's schema and internal consistency."""
    dream._object(note, NOTE_FIELDS)
    if note["contract"] != NOTE_CONTRACT:
        raise DreamError("Session notes use contract " + NOTE_CONTRACT)
    _triple(note["source"])
    ranges = _ranges(note["ranges"], "Note ranges", True)
    if (not isinstance(note["render_contract"], str) or
            not re.fullmatch(r"[a-z][a-z0-9-]{0,63}-v[1-9][0-9]{0,5}", note["render_contract"])):
        raise DreamError("render_contract must name a rendering contract such as claude-view-v2")
    _agent(note["reader"], "Reader")
    _utc(note["written_at"], "written_at")
    items = note["items"]
    if not isinstance(items, list) or len(items) > MAX_ITEMS:
        raise DreamError("Note items must be a list of at most " + str(MAX_ITEMS))
    for item in items:
        _item(item, ranges)
    if len(canonical(note)) > MAX_NOTE_BYTES:
        raise DreamError("A session note may not exceed 256 KiB")
    return note


def validate_review(review):
    """Check an independent review of one note."""
    dream._object(review, REVIEW_FIELDS)
    if review["verdict"] not in VERDICTS:
        raise DreamError("A review verdict is accept or reject")
    _agent(review["reviewer"], "Reviewer")
    findings = review["findings"]
    if not isinstance(findings, list) or len(findings) > MAX_FINDINGS:
        raise DreamError("Review findings must be a list of at most " + str(MAX_FINDINGS))
    for finding in findings:
        _prose(finding, "Review finding", MAX_TEXT)
    _utc(review["reviewed_at"], "reviewed_at")
    if len(canonical(review)) > MAX_NOTE_BYTES:
        raise DreamError("A review may not exceed 256 KiB")
    return review


# Private storage

def _identity(value):
    try:
        identity = str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError):
        identity = None
    if identity is None or identity != value:
        raise DreamError("The session-note store is keyed by a canonical repository_uuid")
    return identity


def store_root(config, repository_uuid):
    """``<log_dir>/dream-notes/<repository_uuid>``, beside the dream job directories."""
    return (config.log_dir or Path("~/.sucoder/logs")).expanduser().resolve() / "dream-notes" / _identity(repository_uuid)


def _private_dir(name, dir_fd=None, create=False):
    """Open a private, account-owned directory without following a symlink."""
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=dir_fd)
        except FileExistsError:
            pass
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)
    info = os.fstat(fd)
    # As for dream job directories (dream_explore._job).
    if info.st_uid != os.geteuid() or info.st_mode & 0o077:
        os.close(fd)
        raise DreamError("Session-note store directories must be private and account-owned: " + str(name), 3)
    return fd


def _load(fd, name, limit):
    """A stored canonical JSON value, or None when absent; never follows a symlink."""
    try:
        raw = dream_run._read(fd, name, limit)
    except FileNotFoundError:
        return None
    value = dream._json(raw)
    if raw != canonical(value) + b"\n":
        raise DreamError("Stored session-note file is not in canonical form: " + name, 3)
    return value


def _publish(fd, name, data, conflict):
    """Write an immutable file once: identical content is a no-op, different content is refused."""
    try:
        existing = dream_run._read(fd, name, MAX_NOTE_BYTES + 1)
    except FileNotFoundError:
        existing = None
    if existing is not None and existing != data:
        raise DreamError(conflict, 3)
    # Linking never replaces, so a racing different write is refused here too.
    dream_run._store(fd, name, data)


def _replace(fd, name, data):
    """Atomically replace a mutable record (coverage.json) in the same directory."""
    temporary = ".pending-" + uuid.uuid4().hex
    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
    renamed = False
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
        renamed = True
    finally:
        if not renamed:
            os.unlink(temporary, dir_fd=fd)
    os.fsync(fd)


def _record(fd):
    value = _load(fd, "source.json", MAX_RECORD_BYTES)
    if value is None:
        return None
    dream._object(value, RECORD_FIELDS)
    if value["contract"] != COVERAGE_CONTRACT or any(type(value[k]) is not int or value[k] < 0 for k in ("device", "inode")):
        raise DreamError("Invalid session source record", 3)
    _triple(value["source"])
    return value


@contextmanager
def _session_file(path):
    try:
        handle = os.open(Path(path), os.O_RDONLY | os.O_NONBLOCK)
    except OSError as exc:
        raise DreamError("Cannot open the session file: " + str(exc), 3) from None
    try:
        info = os.fstat(handle)
        if not stat.S_ISREG(info.st_mode):
            raise DreamError("A session source must be a regular file", 3)
        yield handle, info
    finally:
        os.close(handle)


def _hash(handle, start, end):
    """SHA-256 of the file's bytes [start, end), or None when the file is shorter."""
    value = hashlib.sha256()
    position = start
    while position < end:
        block = os.pread(handle, min(READ_BLOCK, end - position), position)
        if not block:
            return None
        value.update(block)
        position += len(block)
    return value.hexdigest()


class Store:
    """The private store of one repository's session notes and coverage."""

    @_guard
    def __init__(self, root):
        root = Path(root).expanduser().absolute()
        if root.parent.name != "dream-notes":
            raise DreamError("Open the session-note store at store_root(config, repository_uuid)")
        _identity(root.name)
        self.root = root
        root.parent.parent.mkdir(parents=True, exist_ok=True)
        top = _private_dir(str(root.parent), create=True)
        try:
            fd = _private_dir(root.name, top, create=True)
        finally:
            os.close(top)
        try:
            os.close(_private_dir("sources", fd, create=True))
        finally:
            os.close(fd)

    @classmethod
    def for_repository(cls, config, source, revision):
        """The store keyed by the memory registry at ``revision`` of a dream.GitSource."""
        identity = dream._repository_uuid(source, source.commit(revision))
        if identity["provisional"]:
            raise DreamError("Session notes need the memory registry's repository_uuid; "
                             "refusing a provisional repository identity", 3)
        root = store_root(config, identity["repository_uuid"])
        if root == source.path or source.path in root.parents:
            raise DreamError("The session-note store must lie outside the inspected repository")
        return cls(root)

    @contextmanager
    def _sources(self):
        top = _private_dir(str(self.root.parent))
        try:
            fd = _private_dir(self.root.name, top)
        finally:
            os.close(top)
        try:
            sources = _private_dir("sources", fd)
        finally:
            os.close(fd)
        try:
            yield sources
        finally:
            os.close(sources)

    def source(self, harness, source_id, subpath=""):
        return SessionSource(self, harness, source_id, subpath)

    @_guard
    def sources(self):
        """Every recorded source, by key; a directory holding only a lease records nothing yet."""
        found = []
        with self._sources() as fd:
            for name in sorted(os.listdir(fd)):
                if not SHA256.fullmatch(name):
                    raise DreamError("Unexpected entry in the session-note store: " + name, 3)
                child = _private_dir(name, fd)
                try:
                    record = _record(child)
                    if record is None:
                        if any(n != "lease" and not n.startswith(".pending-") for n in os.listdir(child)):
                            raise DreamError("Session source " + name + " holds records without its source.json", 3)
                        continue
                finally:
                    os.close(child)
                entry = SessionSource(self, *record["source"])
                if entry.key != name:
                    raise DreamError("Session source record does not match its key: " + name, 3)
                found.append(entry)
        return found


class SessionSource:
    """One (harness, source_id, subpath) source: its record, coverage, lease and notes."""

    def __init__(self, store, harness, source_id, subpath=""):
        self.store = store
        self.triple = _triple([harness, source_id, subpath])
        self.harness, self.source_id, self.subpath = self.triple
        self.key = digest(canonical(self.triple))
        self._held = None

    def _label(self):
        return canonical(self.triple).decode()

    @_guard
    def _acquire(self):
        try:
            import fcntl
        except ImportError:
            raise DreamError("The session-note store requires advisory file locking", 3) from None
        with self.store._sources() as sources:
            directory = _private_dir(self.key, sources, create=True)
        try:
            handle = os.open("lease", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory)
            try:
                info = os.fstat(handle)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                    raise DreamError("A session lease must be a private, regular, account-owned file", 3)
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise DreamError("Session source is busy; another reader holds its lease: " + self._label(), 3) from None
            except BaseException:
                os.close(handle)
                raise
        except BaseException:
            os.close(directory)
            raise
        return directory, handle

    @contextmanager
    def lease(self):
        """Hold the source's non-blocking exclusive lease; coverage and notes are written only inside it."""
        if self._held is not None:
            raise DreamError("This handle already holds the lease on " + self._label())
        self._held = self._acquire()
        try:
            yield self
        finally:
            directory, handle = self._held
            self._held = None
            os.close(handle)
            os.close(directory)

    def _leased(self):
        if self._held is None:
            raise DreamError("Writing session coverage, notes or reviews requires the source lease: " + self._label())
        directory, handle = self._held
        held = os.fstat(handle)
        try:
            current = os.stat("lease", dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is None or (current.st_dev, current.st_ino) != (held.st_dev, held.st_ino):
            raise DreamError("The source lease file was replaced; the lease is no longer exclusive", 3)
        return directory

    @contextmanager
    def _opened(self):
        """The source directory, or None before any lease created it."""
        if self._held is not None:
            yield self._held[0]
            return
        with self.store._sources() as sources:
            try:
                fd = _private_dir(self.key, sources)
            except FileNotFoundError:
                fd = None
        try:
            yield fd
        finally:
            if fd is not None:
                os.close(fd)

    def _check_record(self, fd):
        record = _record(fd)
        if record is not None and record["source"] != self.triple:
            raise DreamError("Session source record does not match its key", 3)
        return record

    def _coverage(self, fd):
        if fd is None:
            return []
        record = self._check_record(fd)
        value = _load(fd, "coverage.json", MAX_COVERAGE_BYTES)
        if value is None:
            return []
        if record is None:
            raise DreamError("Session coverage exists without its source record", 3)
        dream._object(value, ("contract", "ranges"))
        if value["contract"] != COVERAGE_CONTRACT:
            raise DreamError("Session coverage uses contract " + COVERAGE_CONTRACT, 3)
        ranges = _ranges(value["ranges"], "Recorded coverage", True)
        if len(ranges) > MAX_RANGES:
            raise DreamError("Recorded coverage exceeds its range allowance", 3)
        return ranges

    @_guard
    def record(self):
        """source.json: the triple and the file's device and inode when first seen, or None."""
        with self._opened() as fd:
            return None if fd is None else self._check_record(fd)

    @_guard
    def coverage(self):
        """Every raw range read, as [byte_start, byte_end, raw_sha256], sorted and never merged."""
        with self._opened() as fd:
            return [list(r) for r in self._coverage(fd)]

    @_guard
    def covered(self):
        """The union of the recorded ranges, as [byte_start, byte_end] pairs."""
        with self._opened() as fd:
            ranges = self._coverage(fd)
        union = []
        for start, end, _ in ranges:
            if union and start <= union[-1][1]:
                union[-1][1] = max(union[-1][1], end)
            else:
                union.append([start, end])
        return union

    @_guard
    def add_coverage(self, path, ranges):
        """Record raw ranges read from the session file at ``path``, after checking their digests."""
        fd = self._leased()
        added = _ranges(ranges, "Coverage ranges", False)
        record = self._check_record(fd)
        existing = self._coverage(fd)
        with _session_file(path) as (handle, info):
            for start, end, sha in added:
                if _hash(handle, start, end) != sha:
                    raise DreamError("Coverage range does not match the session file: " + str([start, end]))
        merged = sorted(set(existing) | set(added))
        _consistent(merged, "Session coverage")
        if len(merged) > MAX_RANGES:
            raise DreamError("Session coverage exceeds its range allowance", 3)
        if record is None:
            record = dict(contract=COVERAGE_CONTRACT, source=self.triple, device=info.st_dev, inode=info.st_ino)
            dream_run._store(fd, "source.json", canonical(record) + b"\n")
        if merged != existing:
            value = dict(contract=COVERAGE_CONTRACT, ranges=[list(r) for r in merged])
            _replace(fd, "coverage.json", canonical(value) + b"\n")
        return [list(r) for r in merged]

    @_guard
    def verify(self, path):
        """Check every recorded range against the session file; a mismatch marks it rewritten."""
        with self._opened() as fd:
            record = None if fd is None else self._check_record(fd)
            ranges = self._coverage(fd)
        with _session_file(path) as (handle, info):
            rewritten = [list(r) for r in ranges if _hash(handle, r[0], r[1]) != r[2]]
        same = None if record is None else (record["device"], record["inode"]) == (info.st_dev, info.st_ino)
        return dict(size=info.st_size, same_file=same, rewritten=rewritten)

    def _note(self, notes, note_id, covered):
        value = _load(notes, note_id + ".json", MAX_NOTE_BYTES + 1)
        if value is None:
            return None
        validate_note(value)
        if digest(canonical(value)) != note_id or value["source"] != self.triple:
            raise DreamError("Stored session note differs from its identity: " + note_id, 3)
        if any(tuple(r) not in covered for r in value["ranges"]):
            raise DreamError("Stored session note admits ranges missing from coverage: " + note_id, 3)
        return value

    @_guard
    def write_note(self, note):
        """Store an immutable note under the SHA-256 of its canonical form; return that note_id."""
        fd = self._leased()
        validate_note(note)
        if note["source"] != self.triple:
            raise DreamError("The note's source differs from this session source")
        covered = set(self._coverage(fd))
        if any(tuple(r) not in covered for r in note["ranges"]):
            raise DreamError("Record every range a note admits in the source's coverage before writing the note")
        data = canonical(note)
        note_id = digest(data)
        notes = _private_dir("notes", fd, create=True)
        try:
            _publish(notes, note_id + ".json", data + b"\n",
                     "A stored note differs from its identity; preserved without overwriting: " + note_id)
        finally:
            os.close(notes)
        return note_id

    @_guard
    def attach_review(self, note_id, review):
        """Attach the one immutable review of a stored note."""
        fd = self._leased()
        if not isinstance(note_id, str) or not SHA256.fullmatch(note_id):
            raise DreamError("note_id must be a note's SHA-256 identity")
        validate_review(review)
        try:
            notes = _private_dir("notes", fd)
        except FileNotFoundError:
            raise DreamError("No such session note: " + note_id) from None
        try:
            if self._note(notes, note_id, set(self._coverage(fd))) is None:
                raise DreamError("No such session note: " + note_id)
            _publish(notes, note_id + ".review.json", canonical(review) + b"\n",
                     "This note already has a different review; reviews are immutable: " + note_id)
        finally:
            os.close(notes)

    @_guard
    def notes(self, accepted_only=False):
        """Stored notes as dicts of note_id, note and review (or None), oldest first.

        ``accepted_only`` keeps only notes whose review accepts them, which is
        all the organizer reads.
        """
        result = []
        with self._opened() as fd:
            if fd is None:
                return []
            covered = set(self._coverage(fd))
            try:
                notes = _private_dir("notes", fd)
            except FileNotFoundError:
                return []
            try:
                if self._check_record(fd) is None:
                    raise DreamError("Session notes exist without their source record", 3)
                ids, reviews = set(), set()
                for name in os.listdir(notes):
                    if re.fullmatch(r"[0-9a-f]{64}\.json", name):
                        ids.add(name[:64])
                    elif re.fullmatch(r"[0-9a-f]{64}\.review\.json", name):
                        reviews.add(name[:64])
                    elif not name.startswith(".pending-"):
                        raise DreamError("Unexpected entry among session notes: " + name, 3)
                if reviews - ids:
                    raise DreamError("A session review has no note", 3)
                for note_id in ids:
                    note = self._note(notes, note_id, covered)
                    review = None
                    if note_id in reviews:
                        review = validate_review(_load(notes, note_id + ".review.json", MAX_NOTE_BYTES + 1))
                    if accepted_only and (review is None or review["verdict"] != "accept"):
                        continue
                    result.append(dict(note_id=note_id, note=note, review=review))
            finally:
                os.close(notes)
        result.sort(key=lambda row: (_instant(row["note"]["written_at"]), row["note_id"]))
        return result
