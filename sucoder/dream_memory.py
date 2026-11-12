"""The wiki-only memory branch that reviewed dreams update in place.

See docs/dream/essay-alignment.org, S1. The branch holds only ``.sucoder/wiki/``
and shares no history with code branches; git history is its audit log. Its
first use seeds a root commit from the launch commit's wiki subtree, so an
accepted companion carries over. Publication itself stays in dream_run.publish,
which checks the frozen base and moves the branch in one ref transaction.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

from .dream import DreamError, MAX_TREE_BYTES

PREFIX = ".sucoder/wiki/"


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
