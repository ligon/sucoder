"""The wiki-only memory branch that reviewed dreams update in place.

See docs/dream/essay-alignment.org, S1. The branch holds only ``.sucoder/wiki/``
and shares no history with code branches; git history is its audit log. Its
first use seeds a root commit from the launch commit's wiki subtree, so an
accepted companion carries over. Publication itself stays in dream_run.publish,
which checks the frozen base and moves the branch in one ref transaction.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile

from .dream import DreamError, MAX_TREE_BYTES

PREFIX = ".sucoder/wiki/"
# S5: a read-only copy of the branch's wiki inside a checkout, where the pages'
# file: links (written relative to <project>/.sucoder/wiki/) resolve.
COPY_CONTRACT = "sucoder-memory-copy-v1"
MARKER = ".sucoder-memory-copy.json"
EXCLUDES = ("/.sucoder/wiki/", "/.sucoder/.wiki-*")
MAX_COPY_FILES = 4096
MAX_COPY_BYTES = 8 * 1024 * 1024


def branch_ref(name):
    """Full ref for a validated ``dream.memory_branch`` value (config.py)."""
    if not isinstance(name, str) or not name or name.startswith(("refs/", "dream/", "-")):
        raise DreamError("dream.memory_branch must be a plain branch name outside dream/")
    return "refs/heads/" + name


def configured(config, mirror):
    """The mirror's memory ref, or None when the mirror keeps candidate branches."""
    settings = config.mirrors[mirror].dream if mirror in config.mirrors else None
    name = (settings or config.dream).memory_branch
    return branch_ref(name) if name else None


def check_tree(source, commit):
    """Refuse a memory branch that carries anything besides the companion wiki."""
    names = source.git("ls-tree", "-r", "-z", "--name-only", commit, limit=MAX_TREE_BYTES)
    stray = [n for n in names.split(b"\0") if n and not n.startswith(PREFIX.encode())]
    if stray:
        raise DreamError("Memory branch holds files outside " + PREFIX + "; choose another dream.memory_branch", 3)


def tip(source, ref):
    """Current memory commit, or None before the first dream."""
    if source.git("symbolic-ref", "--quiet", ref, missing=True) is not None:
        raise DreamError("Memory branch must not be a symbolic ref", 3)
    value = source.git("rev-parse", "--verify", "--quiet", "--end-of-options", ref + "^{commit}", missing=True)
    return value.decode().strip() if value else None


def ensure(source, ref, seed_commit, directory, deadline):
    """Return the memory tip, seeding an orphan root from ``seed_commit`` if absent."""
    from . import dream_run
    current = tip(source, ref)
    if current is None:
        with tempfile.TemporaryDirectory(prefix=".memory-", dir=directory) as scratch:
            index = Path(scratch) / "index"
            wiki = source.git("rev-parse", "--verify", "--quiet", seed_commit + ":" + PREFIX.rstrip("/"), missing=True)
            if wiki and source.git("cat-file", "-t", wiki.decode().strip()).strip() == b"tree":
                dream_run._git_write(source, index, ["read-tree", "--prefix=" + PREFIX, wiki.decode().strip()], deadline)
                origin = "the companion wiki at " + seed_commit
            else:
                dream_run._git_write(source, index, ["read-tree", "--empty"], deadline)
                origin = "an empty tree (no companion wiki at " + seed_commit + ")"
            tree = dream_run._git_write(source, index, ["write-tree"], deadline)
            message = ("Seed dream memory\n\nRoot of " + ref + ", from " + origin + ".\n"
                       "Reviewed dreams update this branch in place; it holds only " + PREFIX + ".\n")
            commit = dream_run._git_write(source, index, ["commit-tree", tree, "-F", "-"], deadline, message.encode())
            try:
                dream_run._git_write(source, index, ["update-ref", "--no-deref", "--stdin"], deadline,
                                     ("start\ncreate " + ref + " " + commit + "\nprepare\ncommit\n").encode())
            except DreamError:
                if tip(source, ref) is None:
                    raise
                # Another dream seeded it first; use theirs.
        current = tip(source, ref)
    check_tree(source, current)
    return current


def _git(checkout, *args, data=None, timeout=10):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", GIT_NO_LAZY_FETCH="1")
    return subprocess.run(["git", "--no-replace-objects", "-C", str(checkout), *args], input=data,
                          capture_output=True, env=env, timeout=timeout, check=False)


def _entries(checkout, commit, timeout):
    """The wiki's regular files at ``commit`` as (relative path, bytes), byte for byte."""
    listed = _git(checkout, "ls-tree", "-r", "-z", "--full-tree", commit, "--", PREFIX, timeout=timeout)
    if listed.returncode:
        raise DreamError("Memory tree is unreadable: " + listed.stderr.decode(errors="replace").strip())
    rows = []
    for row in filter(None, listed.stdout.split(b"\0")):
        meta, name = row.split(b"\t", 1)
        mode, kind, oid = meta.decode().split()
        name = name.decode("utf-8")
        if kind != "blob" or mode not in ("100644", "100755") or not name.startswith(PREFIX):
            raise DreamError("Memory branch holds a non-regular wiki entry: " + name)
        rows.append((name[len(PREFIX):], oid))
    if len(rows) > MAX_COPY_FILES:
        raise DreamError("Memory wiki has too many files to copy")
    batch = _git(checkout, "cat-file", "--batch", data=b"".join(oid.encode() + b"\n" for _, oid in rows),
                 timeout=timeout)
    if batch.returncode:
        raise DreamError("Memory blobs are unreadable")
    out, at, files, total = batch.stdout, 0, [], 0
    for name, oid in rows:
        end = out.index(b"\n", at)
        head = out[at:end].split()
        if len(head) != 3 or head[0].decode() != oid or head[1] != b"blob":
            raise DreamError("Memory blob listing is inconsistent")
        size = int(head[2])
        total += size
        if total > MAX_COPY_BYTES:
            raise DreamError("Memory wiki exceeds the copy allowance")
        files.append((name, out[end + 1:end + 1 + size]))
        at = end + 1 + size + 1
    return files


def _exclude(checkout, timeout):
    """List the copy in the repository's local info/exclude, once."""
    found = _git(checkout, "rev-parse", "--git-path", "info/exclude", timeout=timeout)
    if found.returncode:
        return
    path = Path(found.stdout.decode().strip())
    path = path if path.is_absolute() else Path(checkout) / path
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    missing = [line for line in EXCLUDES if line not in text.splitlines()]
    if missing:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(("" if not text or text.endswith("\n") else "\n") +
                         "# sucoder: read-only copy of the dream memory wiki\n" + "\n".join(missing) + "\n")


def _owned_copy(path):
    """The marker of a copy this module wrote, or None for anything else at ``path``."""
    try:
        if path.is_symlink() or not path.is_dir():
            return None
        marker = path / MARKER
        if marker.is_symlink() or not marker.is_file():
            return None
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) and value.get("contract") == COPY_CONTRACT else None


def materialize(checkout, branch, *, timeout=10):
    """Copy the memory branch's wiki into ``checkout``/.sucoder/wiki, read-only (S5).

    Pages link into the project relative to that directory, so the copy is
    where their links resolve. Only a directory carrying this module's marker
    is ever replaced; a checkout that tracks files there (legacy candidate
    mode) or holds anything else there is left alone. Returns a state dict:
    written, current, absent, tracked, unmanaged or busy.
    """
    checkout = Path(checkout)
    ref = branch_ref(branch)
    top = _git(checkout, "rev-parse", "--show-toplevel", timeout=timeout)
    if top.returncode or Path(top.stdout.decode().strip()).resolve() != checkout.resolve():
        raise DreamError("Copy the memory wiki into the root of a git work tree")
    found = _git(checkout, "rev-parse", "--verify", "--quiet", "--end-of-options", ref + "^{commit}", timeout=timeout)
    if found.returncode:
        return dict(state="absent", ref=ref)
    commit = found.stdout.decode().strip()
    destination = checkout / PREFIX.rstrip("/")
    result = dict(ref=ref, commit=commit, path=str(destination))
    tracked = _git(checkout, "ls-files", "-z", "--", PREFIX, timeout=timeout)
    if tracked.returncode or tracked.stdout:
        return dict(result, state="tracked", reason="the checkout tracks files under " + PREFIX)
    parent = destination.parent
    if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
        return dict(result, state="unmanaged", reason=str(parent) + " is not a plain directory")
    parent.mkdir(exist_ok=True)
    lock = os.open(parent / ".wiki-lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(result, state="busy")
        present = destination.exists() or destination.is_symlink()
        marker = _owned_copy(destination) if present else None
        if present and marker is None:
            return dict(result, state="unmanaged", reason=str(destination) + " was not written by sucoder")
        _exclude(checkout, timeout)
        if marker is not None and marker.get("ref") == ref and marker.get("commit") == commit:
            return dict(result, state="current")
        files = _entries(checkout, commit, timeout)
        mask = os.umask(0)
        os.umask(mask)
        staging = Path(tempfile.mkdtemp(prefix=".wiki-", dir=parent))
        try:
            for name, data in files:
                target = staging / name
                target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
            note = dict(contract=COPY_CONTRACT, ref=ref, commit=commit,
                        note="Read-only copy of the dream memory branch; sucoder replaces it, so edits here are lost.")
            fd = os.open(staging / MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(note, sort_keys=True) + "\n")
            # mkdtemp's 0700 becomes the mask's usual directory mode; keep a setgid bit
            # inherited from a group-shared parent, as the subdirectories do.
            os.chmod(staging, (0o755 & ~mask) | (os.stat(parent).st_mode & 0o2000))
            if marker is not None:
                retired = parent / (".wiki-old-" + secrets.token_hex(6))
                os.rename(destination, retired)
                os.rename(staging, destination)
                shutil.rmtree(retired, ignore_errors=True)
            else:
                os.rename(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        return dict(result, state="written", files=len(files))
    finally:
        os.close(lock)


def checkout(config, path=None, branch=None):
    """``sucoder dream checkout``: the read-only copy in any checkout (S5).

    The checkout needs the memory branch locally; the human's own clone gets
    it with ``git fetch <mirror> sucoder/memory:sucoder/memory``.
    """
    from . import dream_explore
    root, name, item, _, _ = dream_explore.project(config, path)
    name = branch or (item.dream or config.dream).memory_branch
    if not name:
        raise DreamError("No dream.memory_branch is configured for " + str(root) + "; pass --branch")
    return dict(schema_version=1, operation="dream_checkout", **materialize(root, name))


def main(argv=None):
    """``python -m sucoder.dream_memory checkout PATH BRANCH``: the launcher runs
    this as the agent account, so the copy in the agent's mirror is the agent's."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3 or args[0] != "checkout":
        print("usage: python -m sucoder.dream_memory checkout PATH BRANCH", file=sys.stderr)
        return 2
    try:
        value = materialize(Path(args[1]), args[2])
    except (DreamError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(json.dumps(dict(state="failed", error=str(exc))))
        return 1
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
