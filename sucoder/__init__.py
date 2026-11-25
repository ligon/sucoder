"""sucoder – Unix-sandboxed agent collaboration toolkit."""

import re
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

# The oldest interpreter SuCoder's staged helpers are written for, on the
# controller (pyproject's python = "^3.9") and on every target.
PYTHON_FLOOR = (3, 9)

__all__ = ["PYTHON_FLOOR", "__version__"]

# Written into the checkout's git directory, so it follows that checkout.
_CACHE_NAME = "sucoder-version"
_DESCRIBE = re.compile(r"v?(?P<base>[0-9][^-]*)-(?P<distance>[0-9]+)-g(?P<commit>[0-9a-f]+)")


def _metadata_version() -> str:
    try:
        return version("sucoder")
    except PackageNotFoundError:
        return "0.0.0"


def _git_dir(root: Path) -> Optional[Path]:
    """The git directory of the checkout at *root*: ``.git`` or a worktree's."""
    dotgit = root / ".git"
    if dotgit.is_dir():
        return dotgit
    if dotgit.is_file():
        text = dotgit.read_text().strip()
        if text.startswith("gitdir: "):
            path = Path(text[len("gitdir: "):])
            return path if path.is_absolute() else (root / path).resolve()
    return None


def _head_commit(git_dir: Path) -> Optional[str]:
    """HEAD's commit from plain file reads: no git process, no work-tree scan."""
    head = (git_dir / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head if re.fullmatch(r"[0-9a-f]{40,64}", head) else None
    ref = head[len("ref: "):]
    common = git_dir
    if (git_dir / "commondir").is_file():
        common = (git_dir / (git_dir / "commondir").read_text().strip()).resolve()
    for base in (git_dir, common):
        loose = base / ref
        if loose.is_file():
            return loose.read_text().strip() or None
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    return None


def _checkout_version(root: Path) -> Optional[str]:
    """The build's version string for the checkout at *root*, or None.

    An editable install runs from a checkout, and its package metadata keeps
    the version it was installed at however far the checkout has moved since.
    This reproduces pyproject's poetry-dynamic-versioning format from the
    checkout's own tags: ``BASE`` on a tag, else ``BASE.devN+gCOMMIT``.
    ``git describe`` runs only when HEAD differs from the cached commit; it
    took up to 0.7 s on Savio's shared filesystems. Uncommitted edits are
    not detected: a work-tree scan took 4 s there.
    """
    if not (root / "pyproject.toml").is_file():
        return None
    git_dir = _git_dir(root)
    if git_dir is None:
        return None
    try:
        commit = _head_commit(git_dir)
    except OSError:
        commit = None
    if commit is None:
        return None
    cache = git_dir / _CACHE_NAME
    try:
        cached_commit, cached_version = cache.read_text().split()
        if cached_commit == commit:
            return cached_version
    except (OSError, ValueError):
        pass
    try:
        described = subprocess.run(
            ["git", "-C", str(root), "describe", "--tags", "--long", "--match", "v[0-9]*", commit],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        described = ""
    match = _DESCRIBE.fullmatch(described)
    if match is None:
        # No reachable tag, or git refused (another user's checkout): the
        # commit is still known, the release it descends from is not.
        return "0+g" + commit[:7]
    base, distance = match.group("base"), int(match.group("distance"))
    value = base if distance == 0 else f"{base}.dev{distance}+g{match.group('commit')}"
    try:
        cache.write_text(f"{commit} {value}\n")
    except OSError:
        pass
    return value


def _version() -> str:
    """The running code's version: its checkout's, else the package metadata's."""
    try:
        found = _checkout_version(Path(__file__).resolve().parent.parent)
    except Exception:  # noqa: BLE001 - a version string must never fail an import
        found = None
    return found or _metadata_version()


def __getattr__(name: str) -> str:
    # Computed on first use, so importing a submodule costs nothing.
    if name == "__version__":
        value = _version()
        globals()["__version__"] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
