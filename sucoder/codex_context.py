"""Install mirror-scoped Codex instructions before starting a service (ledger 5).

This module is also staged as a standalone script on the agent host. Keep it
stdlib-only: the remote host need not have the SuCoder package installed.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path


CONTEXT_BEGIN = "<!-- sucoder:context:begin -->\n"
CONTEXT_END = "<!-- sucoder:context:end -->\n"
PROJECT_BEGIN = "<!-- sucoder:project:begin -->\n"
PROJECT_END = "<!-- sucoder:project:end -->\n"


def _checksum(text: str) -> str:
    return "<!-- sucoder:sha256=" + hashlib.sha256(text.encode("utf-8")).hexdigest() + " -->\n"


def _replace_block(text: str, begin: str, end: str, body: str) -> str:
    """Replace only a complete, unambiguous generated region."""
    if begin in body or end in body:
        raise ValueError("SuCoder instruction text contains a reserved block marker.")
    payload = body + "\n"
    block = begin + _checksum(payload) + payload + end
    if begin not in text and end not in text:
        return block + text
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError("Ambiguous SuCoder blocks in AGENTS.override.md; file left intact.")
    start, stop = text.index(begin), text.index(end)
    if stop < start:
        raise ValueError("Malformed SuCoder block in AGENTS.override.md; file left intact.")
    previous = text[start + len(begin):stop]
    checksum, separator, payload = previous.partition("\n")
    if not separator or checksum + separator != _checksum(payload):
        raise ValueError(
            "A generated region in AGENTS.override.md was edited; file left intact. "
            "Move edits outside the generated markers or into AGENTS.md before relaunching."
        )
    return text[:start] + block + text[stop + len(end):]


def _git_path(workspace: Path, name: str) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--git-path", name], cwd=workspace,
        check=True, capture_output=True, text=True,
    )
    path = Path(result.stdout.strip())
    return path if path.is_absolute() else workspace / path


def install_context(workspace: Path, prelude: str) -> Path:
    """Preserve native project instructions alongside the complete prelude.

    The override is local to the actual working clone, including a Slurm job's
    local-disk clone. Codex reads it independently of client developer overrides.
    """
    destination = workspace / "AGENTS.override.md"
    lock_path = _git_path(workspace, "info/sucoder-codex-context.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_symlink():
            raise ValueError("Cannot install SuCoder context into symlinked AGENTS.override.md.")
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", "AGENTS.override.md"],
            cwd=workspace, capture_output=True,
        )
        if tracked.returncode == 0:
            raise ValueError(
                "AGENTS.override.md is tracked by Git; local SuCoder context requires "
                "an untracked override. File left intact."
            )
        existing = destination.read_bytes().decode("utf-8") if destination.exists() else ""
        content = existing
        if not existing.strip() or PROJECT_BEGIN in existing or PROJECT_END in existing:
            project = workspace / "AGENTS.md"
            original = project.read_bytes().decode("utf-8") if project.exists() else ""
            body = (
                "Project instructions copied at service launch. If AGENTS.md exists, "
                "read its current contents before working to pick up subsequent edits.\n\n" + original
            )
            content = _replace_block(content, PROJECT_BEGIN, PROJECT_END, body)
        content = _replace_block(content, CONTEXT_BEGIN, CONTEXT_END, prelude)
        # Exclude before creating files so a concurrent git add cannot sweep up
        # generated instructions, including a temporary file left by interruption.
        exclude = _git_path(workspace, "info/exclude")
        old = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        additions = [pattern for pattern in ("/AGENTS.override.md", "/.sucoder-codex-context-*")
                     if pattern not in old.splitlines()]
        if additions:
            with exclude.open("a", encoding="utf-8") as stream:
                stream.write(("\n" if old and not old.endswith("\n") else "")
                             + "\n".join(additions) + "\n")
        if content != existing:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=workspace,
                prefix=".sucoder-codex-context-", delete=False,
            ) as staged:
                staged.write(content)
                staged.flush()
                os.fsync(staged.fileno())
                if destination.exists():
                    os.fchmod(staged.fileno(), destination.stat().st_mode & 0o777)
            current = destination.read_bytes().decode("utf-8") if destination.exists() else ""
            if destination.is_symlink() or current != existing:
                raise ValueError("AGENTS.override.md changed during installation; file left intact.")
            os.replace(staged.name, destination)
    return destination


def run(prelude: str, command: list[str]) -> None:
    """Run in the final service cwd, after the existing local-tier preparation."""
    try:
        path = install_context(Path.cwd(), prelude)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"SuCoder: cannot install Codex workspace instructions: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"SuCoder workspace instructions: {path}", file=sys.stderr, flush=True)
    os.execvp(command[0], command)
