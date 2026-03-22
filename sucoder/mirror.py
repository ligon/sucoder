"""High-level operations for managing agent mirrors."""

from __future__ import annotations

import datetime as _dt
import logging
import os
import pwd
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Mapping, NoReturn, Optional, Sequence, Tuple

import yaml

from .config import (
    AGENT_PROFILES,
    DEFAULT_LAUNCH_MODES,
    AgentFlagTemplates,
    AgentLauncher,
    AgentType,
    Config,
    MirrorSettings,
    RemoteConfig,
)
from .executor import CommandError, CommandExecutor, RemoteExecutor
from .permissions import (
    apply_agent_repo_permissions,
    check_parent_traversable,
    ensure_directory,
    ensure_directory_mode,
)
from .skills_version import validate_skills_version
from .workspace_prefs import WorkspacePrefs


class MirrorError(RuntimeError):
    """Raised when mirror operations fail."""


@dataclass
class WorktreeInfo:
    """Status information for a single git worktree."""

    path: Path
    branch: Optional[str]       # None if detached HEAD
    head_commit: str             # Short SHA
    is_main: bool                # True for the main worktree (the mirror root)
    commits_ahead: int           # Commits ahead of base branch
    last_commit_summary: str     # One-line log of HEAD
    last_commit_date: str        # Relative date of HEAD
    is_dirty: bool               # Has uncommitted changes
    modified_count: int          # Number of modified/staged files
    untracked_count: int         # Number of untracked files
    diff_stat: Optional[str]     # --stat summary (only when requested)


def _parse_worktree_porcelain(output: str) -> List[Dict[str, str]]:
    """Parse ``git worktree list --porcelain`` output into a list of dicts.

    Each dict has keys: ``worktree``, ``HEAD``, and optionally ``branch``
    (absent when the worktree has a detached HEAD).
    """
    worktrees: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line:
            if current:
                worktrees.append(current)
                current = {}
            continue
        if line.startswith("worktree "):
            current["worktree"] = line[len("worktree "):]
        elif line.startswith("HEAD "):
            current["HEAD"] = line[len("HEAD "):]
        elif line.startswith("branch "):
            current["branch"] = line[len("branch "):]
        elif line == "detached":
            current["detached"] = "true"
        elif line == "bare":
            current["bare"] = "true"
    if current:
        worktrees.append(current)
    return worktrees


@dataclass
class MirrorContext:
    """Descriptor for a specific mirror derived from configuration."""

    config: Config
    settings: MirrorSettings

    @property
    def canonical_path(self) -> Path:
        return self.settings.canonical_repo

    @property
    def mirror_path(self) -> Path:
        return self.config.mirror_root / self.settings.mirror_dirname

    @property
    def remote_name(self) -> str:
        return self.settings.branch_prefixes.human

    @property
    def agent_prefix(self) -> str:
        return self.settings.branch_prefixes.agent

    @property
    def agent_launcher(self) -> AgentLauncher:
        return self.settings.agent_launcher

    @property
    def skills(self) -> List[Path]:
        return list(self.settings.skills)

    @property
    def is_remote(self) -> bool:
        return self.settings.is_remote

    @property
    def remote_mirror_path(self) -> Optional[str]:
        """Path to the mirror on the remote host (as a string, not local Path)."""
        remote = self.settings.remote
        if remote is None:
            return None
        return str(remote.mirror_root / self.settings.mirror_dirname)


class MirrorManager:
    """Perform operations against configured mirrors."""

    def __init__(
        self,
        config: Config,
        executor: CommandExecutor,
        logger: logging.Logger,
        prompt_handler: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.config = config
        self.executor = executor
        self.logger = logger
        self._prompt_handler = prompt_handler

    def context_for(self, mirror_name: str) -> MirrorContext:
        try:
            settings = self.config.mirrors[mirror_name]
        except KeyError as exc:
            raise MirrorError(f"Mirror `{mirror_name}` not found in configuration.") from exc
        return MirrorContext(config=self.config, settings=settings)

    # ------------------------------------------------------------------ Commands
    def ensure_clone(self, ctx: MirrorContext, *, skip_lfs: bool = True) -> None:
        """Ensure the mirror exists, cloning if necessary.

        When *skip_lfs* is ``True`` (the default), ``GIT_LFS_SKIP_SMUDGE=1``
        is set during the clone so that LFS-tracked files are checked out as
        pointer files instead of triggering downloads that may fail.
        """
        self._validate_canonical(ctx)
        safe_paths = self._ensure_canonical_safe_directory(ctx)
        mirror_path = ctx.mirror_path
        ensure_directory(mirror_path.parent)
        ensure_directory_mode(self.executor, mirror_path.parent, "2770")

        if self._is_git_repo(mirror_path):
            self.logger.info("Mirror already exists at %s", mirror_path)
            self._verify_remote(ctx)
            self._enforce_permissions(ctx)
            self._allow_direnv_if_present(mirror_path)
            return

        self.logger.info("Cloning %s into %s", ctx.canonical_path, mirror_path)
        config_args: List[str] = []
        for path in safe_paths:
            config_args.extend(["-c", f"safe.directory={path}"])

        clone_args = [
            "git",
            *config_args,
            "clone",
            "--no-hardlinks",
            "--no-recurse-submodules",
            "--origin",
            ctx.remote_name,
            str(ctx.canonical_path),
            str(mirror_path),
        ]

        clone_env: Optional[Dict[str, str]] = None
        if skip_lfs:
            clone_env = {"GIT_LFS_SKIP_SMUDGE": "1"}

        try:
            self.executor.run_agent(
                clone_args,
                check=True,
                cwd=str(self.config.mirror_root),
                env=clone_env,
            )
        except CommandError as exc:
            stderr = exc.result.stderr.lower()
            if "permission denied" in stderr or "unable to access './config'" in stderr:
                raise MirrorError(
                    "Failed to clone canonical repository as the agent user. "
                    "Ensure the canonical path and its parents grant the `coder` group "
                    "read and execute permissions."
                ) from exc
            raise

        self.executor.run_agent(
            ["git", "config", "core.sharedRepository", "group"],
            check=True,
            cwd=str(mirror_path),
        )
        self.executor.run_agent(
            ["git", "remote", "set-url", "--push", ctx.remote_name, "no_push"],
            check=True,
            cwd=str(mirror_path),
        )
        self.executor.run_agent(
            ["git", "config", "receive.denyCurrentBranch", "refuse"],
            check=True,
            cwd=str(mirror_path),
        )

        ensure_directory_mode(self.executor, mirror_path, "2770", as_agent=True)
        self._enforce_permissions(ctx)
        self._allow_direnv_if_present(mirror_path)

    def prepare_canonical(
        self,
        ctx: MirrorContext,
        *,
        use_sudo: bool = False,
        setup_agent_remote: bool = True,
    ) -> None:
        """Adjust ownership/permissions and optionally configure human-side access to the agent mirror."""
        canonical = ctx.canonical_path
        if not canonical.exists():
            raise MirrorError(f"Canonical repository not found at {canonical}")

        # Ensure both the working tree and the git dir are group-readable so the agent
        # can traverse and clone. The git dir may be separate (e.g., worktree), so
        # handle both paths.
        git_dir = _resolve_git_dir(canonical)
        target_paths = {canonical, git_dir}
        commands = []
        for path in sorted(target_paths):
            commands.extend(
                [
                    ["chgrp", "-R", self.config.agent_group, str(path)],
                    ["chmod", "-R", "g+rx", str(path)],
                    ["chmod", "-R", "g-w", str(path)],
                    ["find", str(path), "-type", "d", "-exec", "chmod", "g+s", "{}", "+"],
                ]
            )

        for cmd in commands:
            run_args = ["sudo"] + cmd if use_sudo and not self.executor.dry_run else cmd
            self.executor.run_human(run_args, check=True)

        # Verify that every parent directory is traversable by the agent.
        blocking = check_parent_traversable(
            canonical,
            agent_user=self.config.agent_user,
            agent_group=self.config.agent_group,
        )
        if blocking:
            paths_str = "\n  ".join(str(p) for p in blocking)
            cmds_str = " ".join(str(p) for p in blocking)
            raise MirrorError(
                f"The agent user cannot traverse to {canonical} because these "
                f"parent directories lack world-execute (o+x):\n"
                f"  {paths_str}\n"
                f"Fix with one of:\n"
                f"  chmod o+x {cmds_str}\n"
                f"  # or, more targeted:\n"
                f"  setfacl -m g:{self.config.agent_group}:x {cmds_str}"
            )

        self.logger.info(
            "Canonical repository at %s prepared for agent group %s (git dir %s)",
            canonical,
            self.config.agent_group,
            git_dir,
        )

        if setup_agent_remote:
            self._configure_agent_remote(ctx)
            self._write_agent_fetch_helper(ctx)

    def sync(self, ctx: MirrorContext) -> None:
        """Fetch updates from the canonical repository.

        For remote mirrors, pushes from the local canonical to the
        remote mirror via SSH tunnel to the data transfer node.
        """
        if ctx.is_remote:
            self._pull_from_remote(ctx)
            self._sync_remote(ctx)
            return

        mirror_path = self._ensure_mirror_exists(ctx)
        self.logger.info("Fetching updates from %s", ctx.remote_name)

        self.executor.run_agent(
            ["git", "fetch", "--prune", "--no-recurse-submodules",
             ctx.remote_name],
            check=True,
            cwd=str(mirror_path),
        )

    def _resolve_remote_path(self, ctx: MirrorContext) -> str:
        """Return the absolute remote mirror path, resolving ~ via SSH."""
        raw = ctx.remote_mirror_path
        if raw is None:
            raise MirrorError("No remote mirror path configured.")
        if not raw.startswith("~"):
            return raw

        cached = getattr(self, "_resolved_remote_home", None)
        if cached is None:
            # Debug: check ControlMaster state before attempting the command.
            sock = getattr(self.executor, "control_socket_path", None)
            node = getattr(self.executor, "login_node", None)
            if sock and node:
                import subprocess as _dbg_sp
                from pathlib import Path as _P
                sock_exists = _P(sock).exists()
                self.logger.info(
                    "DEBUG _resolve_remote_path: socket %s exists=%s",
                    sock, sock_exists,
                )
                if sock_exists:
                    chk = _dbg_sp.run(
                        ["ssh", "-o", f"ControlPath={sock}",
                         "-O", "check", node],
                        capture_output=True, text=True, check=False,
                    )
                    self.logger.info(
                        "DEBUG mux check rc=%d stdout=%r stderr=%r",
                        chk.returncode, chk.stdout.strip(),
                        chk.stderr.strip(),
                    )
                    # Also try with -v for verbose SSH output
                    probe = _dbg_sp.run(
                        ["ssh", "-v",
                         "-o", "ControlMaster=auto",
                         "-o", f"ControlPath={sock}",
                         node, "echo probe-ok"],
                        capture_output=True, text=True, check=False,
                        timeout=10,
                    )
                    self.logger.info(
                        "DEBUG probe rc=%d stdout=%r stderr=%r",
                        probe.returncode, probe.stdout.strip(),
                        probe.stderr[:500],
                    )
            result = self.executor.run_agent(
                ["bash", "-c", "echo $HOME"],
                check=True,
            )
            cached = result.stdout.strip()
            self._resolved_remote_home = cached

        return raw.replace("~", cached, 1)

    def _remote_git_env(self, ctx: MirrorContext) -> tuple:
        """Return ``(url, env)`` for git operations against the remote mirror.

        Builds the SSH transport command using the ControlMaster sockets
        so that ``git fetch``/``git push`` can reach the mirror via the
        login node.
        """
        remote = ctx.settings.remote
        assert remote is not None

        remote_path = self._resolve_remote_path(ctx)

        login_node = getattr(self.executor, "login_node", None)
        gateway = remote.gateway

        ssh_cmd_parts = ["ssh"]
        control_path = getattr(self.executor, "control_socket_path", None)
        if control_path:
            from .tunnel import _control_socket_path as _gw_sock
            ssh_cmd_parts.extend([
                "-o", "ControlMaster=auto",
                "-o", f"ControlPath={control_path}",
            ])
            # For login-node targets, include a ProxyCommand fallback
            # through the gateway.  Skip it for compute nodes — the
            # gateway cannot reach them directly.
            is_compute = getattr(self.executor, "is_compute_node", False)
            if gateway and not is_compute:
                gw_socket = _gw_sock(gateway)
                ssh_cmd_parts.extend([
                    "-o",
                    f"ProxyCommand=ssh -o ControlMaster=auto "
                    f"-o ControlPath={gw_socket} "
                    f"-W %h:%p {gateway}",
                ])

        git_ssh_cmd = " ".join(shlex.quote(p) for p in ssh_cmd_parts)
        host = login_node or gateway
        url = f"{host}:{remote_path}"

        env = dict(os.environ)
        env["GIT_SSH_COMMAND"] = git_ssh_cmd
        return url, env

    def _pull_from_remote(self, ctx: MirrorContext) -> None:
        """Fetch agent commits from the remote mirror into canonical.

        This must run *before* ``_sync_remote`` so that work the agent
        committed on the mirror is not lost when the canonical repo
        force-pushes over it.

        Strategy:
        1. Fetch the mirror's branch into a temporary ref — always safe.
        2. If canonical is already up-to-date, nothing to do.
        3. If the mirror is strictly ahead (fast-forward), update
           canonical automatically.
        4. If histories have diverged, warn the user and let them
           decide whether to continue (discarding mirror-only commits)
           or abort so they can reconcile manually.
        """
        import subprocess

        url, env = self._remote_git_env(ctx)
        base = ctx.settings.default_base_branch or "main"
        tmp_ref = "refs/sucoder/mirror-head"

        self.logger.info("Fetching agent commits from remote mirror")
        result = self.executor.run_human(
            ["git", "fetch", url, f"{base}:{tmp_ref}"],
            check=False,
            cwd=str(ctx.canonical_path),
            env=env,
        )
        if result.returncode != 0:
            # Mirror may be empty (first run) or unreachable.
            self.logger.warning(
                "Could not fetch from remote mirror (rc=%d): %s",
                result.returncode,
                (result.stderr or "").strip(),
            )
            return

        canon = str(ctx.canonical_path)

        # Resolve both tips.
        def _rev(ref: str) -> Optional[str]:
            r = subprocess.run(
                ["git", "rev-parse", "--verify", ref],
                capture_output=True, text=True, cwd=canon,
            )
            return r.stdout.strip() if r.returncode == 0 else None

        local_head = _rev(f"refs/heads/{base}")
        mirror_head = _rev(tmp_ref)

        if not mirror_head:
            return  # nothing fetched
        if mirror_head == local_head:
            self.logger.info("Canonical and mirror are in sync")
            return

        # Check if local is ancestor of mirror (fast-forward possible).
        is_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", base, tmp_ref],
            capture_output=True, cwd=canon,
        ).returncode == 0

        if is_ancestor:
            # Fast-forward: mirror is strictly ahead.
            ahead = subprocess.run(
                ["git", "log", "--oneline", f"{base}..{tmp_ref}"],
                capture_output=True, text=True, cwd=canon,
            ).stdout.strip()
            self.logger.info(
                "Mirror is ahead of canonical — fast-forwarding:\n%s",
                ahead,
            )
            subprocess.run(
                ["git", "update-ref", f"refs/heads/{base}", mirror_head],
                check=True, cwd=canon,
            )
            # Update the working tree to match.
            subprocess.run(
                ["git", "reset", "--hard", base],
                check=True, cwd=canon,
            )
            return

        # Histories have diverged — need user input.
        only_on_mirror = subprocess.run(
            ["git", "log", "--oneline", f"{base}..{tmp_ref}"],
            capture_output=True, text=True, cwd=canon,
        ).stdout.strip()
        only_on_canonical = subprocess.run(
            ["git", "log", "--oneline", f"{tmp_ref}..{base}"],
            capture_output=True, text=True, cwd=canon,
        ).stdout.strip()

        print("\n⚠  Mirror and canonical have diverged.")
        print(f"\nCommits only on the mirror ({url}):")
        print(f"  {only_on_mirror.replace(chr(10), chr(10) + '  ')}")
        print(f"\nCommits only in canonical:")
        print(f"  {only_on_canonical.replace(chr(10), chr(10) + '  ')}")
        print()

        answer = input(
            "Continue syncing? This will DISCARD the mirror-only commits.\n"
            "  [y] Continue and discard mirror commits\n"
            "  [n] Abort so you can reconcile manually\n"
            "  Choice [n]: "
        ).strip().lower()

        if answer != "y":
            raise MirrorError(
                "Aborting sync — mirror has diverged commits that need "
                "manual reconciliation.  The mirror branch is available "
                f"locally at {tmp_ref} for inspection."
            )

    def _sync_remote(self, ctx: MirrorContext) -> None:
        """Push local canonical commits to the remote mirror.

        Uses the login node ControlMaster for git transport — no
        tunnel needed when the login node has internet access.
        """
        url, env = self._remote_git_env(ctx)

        self.logger.info("Pushing to remote mirror %s", url)
        self.executor.run_human(
            ["git", "push", url, "--all", "--force"],
            check=True,
            cwd=str(ctx.canonical_path),
            env=env,
        )

    def ensure_remote_clone(self, ctx: MirrorContext) -> None:
        """Ensure the mirror exists on the remote host.

        Initialises a bare-ish clone on the remote if it does not
        already exist, then pushes all branches from canonical.
        """
        remote = ctx.settings.remote
        if remote is None:
            raise MirrorError("ensure_remote_clone called on a non-remote mirror.")

        remote_path = ctx.remote_mirror_path
        assert remote_path is not None

        # Resolve to absolute path so we don't fight tilde quoting.
        abs_remote_path = self._resolve_remote_path(ctx)

        # Check if remote mirror is a valid git repo.
        check = self.executor.run_agent(
            ["git", "rev-parse", "--git-dir"],
            check=False,
            cwd=abs_remote_path,
        )
        if check.returncode == 0:
            self.logger.info("Remote mirror already exists at %s", remote_path)
        else:
            # Clean up broken directory from a previously failed init.
            self.executor.run_agent(
                ["rm", "-rf", abs_remote_path],
                check=False,
            )
            self.logger.info("Initialising remote mirror at %s", remote_path)
            self.executor.run_agent(
                ["mkdir", "-p", abs_remote_path],
                check=True,
            )
            base = ctx.settings.default_base_branch or "main"
            self.executor.run_agent(
                ["git", "init", "-b", base],
                check=True,
                cwd=abs_remote_path,
            )

        # Always ensure the config is correct (may have been missed
        # by a failed earlier init).
        self.executor.run_agent(
            ["git", "config", "receive.denyCurrentBranch", "updateInstead"],
            check=True,
            cwd=abs_remote_path,
        )

        # Pull any agent commits before overwriting the mirror.
        self._pull_from_remote(ctx)

        # Push canonical content to the remote via tunnel.
        self._sync_remote(ctx)

        # Ensure HEAD points to the correct branch so that
        # updateInstead keeps the working tree in sync.
        base = ctx.settings.default_base_branch or "main"
        self.executor.run_agent(
            ["git", "symbolic-ref", "HEAD", f"refs/heads/{base}"],
            check=True,
            cwd=abs_remote_path,
        )
        # Reset the working tree to match the branch tip.
        self.executor.run_agent(
            ["git", "reset", "--hard", base],
            check=True,
            cwd=abs_remote_path,
        )

    # (Tunnel helper removed — git transport now goes through the login
    # node ControlMaster directly, no port-forward tunnel needed.)

    def start_task(
        self,
        ctx: MirrorContext,
        *,
        task_name: str,
        base_branch: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> str:
        """Create and switch to a new agent branch based on the chosen base."""
        mirror_path = self._ensure_mirror_exists(ctx)
        self.sync(ctx)

        base = base_branch or ctx.settings.default_base_branch
        remote_ref = f"refs/remotes/{ctx.remote_name}/{base}"
        human_branch = f"{ctx.remote_name}/{base}"

        self.logger.info("Updating local tracking branch %s", human_branch)
        try:
            self.executor.run_agent(
                ["git", "show-ref", "--verify", remote_ref],
                check=True,
                cwd=str(mirror_path),
            )
        except CommandError as exc:
            raise MirrorError(
                f"Base branch `{base}` not found for mirror {ctx.settings.name}. "
                "Ensure the canonical repository has that branch or specify --base."
            ) from exc

        self.executor.run_agent(
            ["git", "branch", "-f", human_branch, remote_ref],
            check=True,
            cwd=str(mirror_path),
        )

        sanitized_task = _sanitize_task_name(task_name)
        stamp = timestamp or _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d%H%M%S")
        agent_branch = f"{ctx.agent_prefix}/{sanitized_task}-{stamp}"
        self.logger.info("Creating task branch %s", agent_branch)

        self.executor.run_agent(
            ["git", "checkout", "-B", agent_branch, remote_ref],
            check=True,
            cwd=str(mirror_path),
        )

        return agent_branch

    def status(self, ctx: MirrorContext) -> str:
        """Return a status summary for the mirror."""
        mirror_path = self._ensure_mirror_exists(ctx)
        lines: List[str] = []

        fetch_url = self._remote_url(ctx, mirror_path, push=False) or "unknown"
        push_url = self._remote_url(ctx, mirror_path, push=True) or "unknown"
        lines.append(f"Remote {ctx.remote_name}: fetch={fetch_url}; push={push_url}")

        git_dir = _resolve_git_dir(mirror_path)
        mirror_mode = self._mode_string(mirror_path)
        git_mode = self._mode_string(git_dir)
        lines.append(
            f"Mirror perms: {mirror_path} {mirror_mode} (git dir {git_dir} {git_mode})"
        )

        lines.append("Agent access:")
        lines.append(
            f"  canonical: {self._agent_access_summary(ctx.canonical_path, require_write=False)}"
        )
        lines.append(
            f"  mirror read: {self._agent_access_summary(mirror_path, require_write=False)}"
        )
        lines.append(
            f"  mirror write: {self._agent_access_summary(mirror_path, require_write=True)}"
        )

        result = self.executor.run_agent(
            ["git", "status", "-sb"],
            check=True,
            cwd=str(mirror_path),
        )
        lines.append("Git status:")
        status_output = result.stdout.strip()
        lines.append(status_output or "(clean)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Worktree inspection (read-only, runs as human user)
    # ------------------------------------------------------------------

    def _run_query(self, ctx: MirrorContext, args, **kwargs):
        """Run a read-only query: locally for local mirrors, via SSH for remote."""
        if ctx.is_remote:
            return self.executor.run_agent(args, **kwargs)
        return self.executor.run_human(args, **kwargs)

    def list_worktrees(
        self,
        ctx: MirrorContext,
        *,
        include_diff: bool = False,
        base_branch: Optional[str] = None,
    ) -> List[WorktreeInfo]:
        """Discover git worktrees in the mirror and gather status for each.

        For local mirrors, commands run as the human user (read-only).
        For remote mirrors, commands run via SSH through the executor.
        """
        mirror_path = self._ensure_mirror_exists(ctx)
        base = base_branch or ctx.settings.default_base_branch

        result = self._run_query(
            ctx,
            ["git", "worktree", "list", "--porcelain"],
            check=True,
            cwd=str(mirror_path),
        )
        parsed = _parse_worktree_porcelain(result.stdout)
        if not parsed:
            return []

        # For remote mirrors, determine the main worktree from the first entry.
        main_worktree_path = (
            parsed[0].get("worktree", "") if ctx.is_remote
            else str(mirror_path.resolve())
        )
        infos: List[WorktreeInfo] = []
        for entry in parsed:
            wt_path_str = entry.get("worktree", "")
            wt_path = Path(wt_path_str)
            if not ctx.is_remote and not wt_path.is_dir():
                continue

            raw_branch = entry.get("branch")
            branch = raw_branch.removeprefix("refs/heads/") if raw_branch else None
            head_sha = entry.get("HEAD", "")[:7]
            is_main = (
                wt_path_str == main_worktree_path if ctx.is_remote
                else str(wt_path.resolve()) == main_worktree_path
            )

            # Gather per-worktree details; tolerate failures gracefully.
            commits_ahead = self._wt_commits_ahead(ctx, wt_path, base, ctx.remote_name)
            summary, date = self._wt_last_commit(ctx, wt_path)
            modified, untracked, dirty = self._wt_dirty_status(ctx, wt_path)
            diff_stat = self._wt_diff_stat(ctx, wt_path, base, ctx.remote_name) if include_diff else None

            infos.append(WorktreeInfo(
                path=wt_path,
                branch=branch,
                head_commit=head_sha,
                is_main=is_main,
                commits_ahead=commits_ahead,
                last_commit_summary=summary,
                last_commit_date=date,
                is_dirty=dirty,
                modified_count=modified,
                untracked_count=untracked,
                diff_stat=diff_stat,
            ))

        return infos

    def worktrees_summary(
        self,
        ctx: MirrorContext,
        *,
        include_diff: bool = False,
        base_branch: Optional[str] = None,
        include_main: bool = False,
    ) -> str:
        """Return a human-readable summary of worktrees in the mirror."""
        infos = self.list_worktrees(ctx, include_diff=include_diff, base_branch=base_branch)
        base = base_branch or ctx.settings.default_base_branch
        display_path = ctx.remote_mirror_path if ctx.is_remote else str(ctx.mirror_path)
        mirror_path = ctx.mirror_path

        non_main = [i for i in infos if not i.is_main]
        main_info = next((i for i in infos if i.is_main), None)

        lines: List[str] = []
        lines.append(f"Worktrees for mirror '{ctx.settings.name}' ({display_path}):")
        lines.append("")

        if include_main and main_info:
            lines.extend(self._format_worktree_block(main_info, base, mirror_path))
            lines.append("")

        if not non_main:
            lines.append("  No active worktrees.")
            return "\n".join(lines)

        for info in non_main:
            lines.extend(self._format_worktree_block(info, base, mirror_path))
            lines.append("")

        return "\n".join(lines).rstrip()

    def _format_worktree_block(
        self,
        info: WorktreeInfo,
        base: str,
        mirror_path: Path,
    ) -> List[str]:
        """Format a single worktree as indented lines."""
        lines: List[str] = []
        # Show path relative to mirror root when possible.
        try:
            rel = info.path.resolve().relative_to(mirror_path.resolve())
            display_path = str(rel) if str(rel) != "." else "(main worktree)"
        except ValueError:
            display_path = str(info.path)

        label = "(main worktree)" if info.is_main else display_path
        lines.append(f"  {label}")
        branch_display = info.branch or "(detached HEAD)"
        lines.append(f"    Branch: {branch_display} @ {info.head_commit}")

        if not info.is_main and info.commits_ahead >= 0:
            lines.append(f"    Ahead:  {info.commits_ahead} commit{'s' if info.commits_ahead != 1 else ''} (vs {base})")

        if info.last_commit_summary:
            lines.append(f"    Last:   {info.head_commit} {info.last_commit_summary} ({info.last_commit_date})")

        status_parts: List[str] = []
        if info.modified_count:
            status_parts.append(f"{info.modified_count} modified")
        if info.untracked_count:
            status_parts.append(f"{info.untracked_count} untracked")
        if status_parts:
            lines.append(f"    Status: dirty ({', '.join(status_parts)})")
        else:
            lines.append(f"    Status: clean")

        if info.diff_stat:
            for stat_line in info.diff_stat.strip().splitlines():
                lines.append(f"    {stat_line}")

        return lines

    # -- Worktree helper queries -----------------------------------------
    # For local mirrors these run as the human user (read-only).
    # For remote mirrors they go via SSH through _run_query.

    def _wt_commits_ahead(self, ctx: MirrorContext, wt_path: Path, base: str, remote: str) -> int:
        """Count commits in the worktree HEAD that are not in the base branch."""
        result = self._run_query(
            ctx,
            ["git", "rev-list", "--count", f"{remote}/{base}..HEAD"],
            check=False,
            cwd=str(wt_path),
        )
        if result.returncode != 0:
            return -1
        try:
            return int(result.stdout.strip())
        except ValueError:
            return -1

    def _wt_last_commit(self, ctx: MirrorContext, wt_path: Path) -> Tuple[str, str]:
        """Return (subject, relative-date) for HEAD."""
        result = self._run_query(
            ctx,
            ["git", "log", "-1", "--format=%s\t%cr", "HEAD"],
            check=False,
            cwd=str(wt_path),
        )
        if result.returncode != 0 or not result.stdout.strip():
            return ("", "")
        parts = result.stdout.strip().split("\t", 1)
        if len(parts) == 2:
            return (parts[0], parts[1])
        return (parts[0], "")

    def _wt_dirty_status(self, ctx: MirrorContext, wt_path: Path) -> Tuple[int, int, bool]:
        """Return (modified_count, untracked_count, is_dirty)."""
        result = self._run_query(
            ctx,
            ["git", "status", "--porcelain"],
            check=False,
            cwd=str(wt_path),
        )
        if result.returncode != 0:
            return (0, 0, False)
        modified = 0
        untracked = 0
        for line in result.stdout.splitlines():
            if line.startswith("??"):
                untracked += 1
            elif line.strip():
                modified += 1
        return (modified, untracked, modified + untracked > 0)

    def _wt_diff_stat(self, ctx: MirrorContext, wt_path: Path, base: str, remote: str) -> Optional[str]:
        """Return diff --stat between base and HEAD."""
        result = self._run_query(
            ctx,
            ["git", "diff", f"{remote}/{base}...HEAD", "--stat"],
            check=False,
            cwd=str(wt_path),
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return result.stdout.strip()

    def launch_agent(
        self,
        ctx: MirrorContext,
        *,
        sync: bool = True,
        task_name: Optional[str] = None,
        base_branch: Optional[str] = None,
        extra_args: Optional[Sequence[str]] = None,
        command_override: Optional[Sequence[str]] = None,
        env_override: Optional[Mapping[str, str]] = None,
        supports_inline_prompt: Optional[bool] = None,
    ) -> int:
        """Launch the configured agent command within the mirror working tree."""
        mirror_path = self._ensure_mirror_exists(ctx)

        if task_name:
            self.logger.info("Preparing task branch %s", task_name)
            self.start_task(ctx, task_name=task_name, base_branch=base_branch)
        elif sync:
            self.sync(ctx)

        launcher = ctx.agent_launcher
        base_command = list(command_override) if command_override else list(launcher.command)
        if not base_command:
            raise MirrorError(
                f"Agent launcher command for mirror {ctx.settings.name} is empty."
            )
        # Store detected agent type so helpers (e.g., _file_read_hint) can
        # produce agent-appropriate output without threading args everywhere.
        self._detected_agent_type = _detect_agent_type(base_command)
        command = list(base_command)
        if extra_args:
            command.extend(extra_args)

        env: Dict[str, str] = dict(launcher.env) if launcher.env else {}
        if env_override:
            env.update(env_override)
        env_to_use = env or None

        prelude = self._compose_context_prelude(ctx)
        inline_prompt_supported = (
            supports_inline_prompt
            if supports_inline_prompt is not None
            else launcher.accepts_inline_prompt
            if launcher.accepts_inline_prompt is not None
            else self._supports_inline_prompt(command)
        )

        self._maybe_run_poetry_auto_install(ctx, mirror_path)

        # Get merged templates (per-mirror > global > agent profile)
        templates = self._get_merged_templates(command, launcher)
        command = self._apply_agent_flag_templates(command, ctx, launcher, templates)

        # Inject system prompt via native flag if available, otherwise trailing text
        if prelude:
            if templates.system_prompt:
                # Use CLI-native system prompt flag (e.g., --system-prompt for Claude)
                # Template is just the flag; content is added as a separate argument
                flag_tokens = shlex.split(templates.system_prompt)
                if flag_tokens:
                    # Add flag and content as separate args to preserve content with spaces
                    command = self._insert_after_executable(command, flag_tokens + [prelude])
            elif inline_prompt_supported:
                # Fallback: append as trailing text
                command = list(command) + [prelude]
            else:
                self.logger.warning(
                    "Context prelude available but not injected because command %s "
                    "has no system_prompt template and does not accept inline prompts.",
                    command[0],
                )

        command = self._wrap_with_nvm(command, launcher)

        # Determine launch mode: explicit config > agent profile default > subprocess
        effective_mode = self._get_effective_launch_mode(command, launcher)

        if ctx.is_remote:
            # Wrap in tmux so the session survives SSH disconnects.
            tmux_name = f"sucoder-{ctx.settings.name}"
            agent_cmd_str = shlex.join(command)

            # If a SLURM allocation is active, cancel it when the agent
            # exits so we don't burn idle compute time.
            slurm_job_id = getattr(self.executor, "slurm_job_id", None)
            if slurm_job_id:
                agent_cmd_str = (
                    f"{agent_cmd_str}; scancel {slurm_job_id} 2>/dev/null"
                )

            # new-session -A attaches if it already exists, creates if not.
            command = [
                "tmux", "new-session", "-A",
                "-s", tmux_name,
                agent_cmd_str,
            ]

        self.logger.info("Starting agent command: %s", shlex.join(command))

        if effective_mode == "exec":
            # Replace current process with agent (preserves TTY)
            self._exec_agent(command, mirror_path, env_to_use)
            # _exec_agent never returns; this is unreachable but satisfies type checker
            return 0  # pragma: no cover
        else:
            # Use subprocess.run (can capture exit code)
            result = self.executor.run_agent(
                command,
                check=False,
                cwd=str(mirror_path),
                env=env_to_use,
                capture_output=False,
            )

            if result.returncode != 0:
                raise MirrorError(
                    f"Agent command exited with code {result.returncode} "
                    f"for mirror {ctx.settings.name}."
                )

            return result.returncode

    def _get_effective_launch_mode(
        self,
        command: Sequence[str],
        launcher: AgentLauncher,
    ) -> Literal["subprocess", "exec"]:
        """Determine the launch mode for this agent command."""
        # Explicit config takes precedence
        if launcher.launch_mode is not None:
            return launcher.launch_mode

        # Fall back to agent type default
        agent_type = _detect_agent_type(command)
        return DEFAULT_LAUNCH_MODES.get(agent_type, "subprocess")

    def _exec_agent(
        self,
        command: List[str],
        cwd: Path,
        env: Optional[Dict[str, str]],
    ) -> NoReturn:
        """Replace current process with agent (preserves TTY).

        This function never returns - the current process is replaced by the agent.
        Use this for interactive CLIs that require proper terminal passthrough.
        """
        agent_user = self.executor.agent_user
        current_user = pwd.getpwuid(os.getuid()).pw_name

        # Determine if we need to switch users
        use_sudo = self.executor.use_sudo_for_agent and (agent_user != current_user)

        # Prepare the command execution
        final_command = list(command)
        command_str = shlex.join(final_command)

        # Construct verification script — quote agent_user to prevent shell injection
        quoted_user = shlex.quote(agent_user)
        check = (
            f'if [ "$(whoami)" != {quoted_user} ]; then '
            f'echo "Error: running as $(whoami), expected {quoted_user}" >&2; '
            f"exit 1; "
            f"fi"
        )

        # Change into the working directory inside the script rather than
        # mutating os.chdir() which would alter global state.
        cd_prefix = f"cd {shlex.quote(str(cwd))} &&"

        # We always wrap in bash to ensure the check runs
        # Use 'exec' to replace the bash process with the final command
        script = f"{check}; {cd_prefix} exec {command_str}"
        final_command = ["bash", "-lc", script]

        if use_sudo:
            # Pass env vars via 'env K=V' prefix so they survive sudo
            # (sudo strips the caller's environment by default).
            if env:
                env_args = ["env"] + [f"{k}={v}" for k, v in env.items()]
                final_command = ["sudo", "-u", shlex.quote(agent_user)] + env_args + final_command
            else:
                final_command = ["sudo", "-u", shlex.quote(agent_user)] + final_command
        else:
            # Non-sudo path: mutations are acceptable since execvp replaces
            # the process immediately after.
            os.chdir(cwd)
            if env:
                os.environ.update(env)

        self.logger.debug("Exec'ing agent (replaces current process): %s", final_command)
        os.execvp(final_command[0], final_command)

    def _maybe_run_poetry_auto_install(self, ctx: MirrorContext, mirror_path: Path) -> None:
        """Offer or run `poetry install` for Poetry-based projects."""
        if ctx.is_remote:
            return  # Poetry install is handled on the remote side, not from local.
        project_file = mirror_path / "pyproject.toml"
        if not project_file.exists():
            return

        prefs = WorkspacePrefs.load(mirror_path)
        decision = prefs.poetry_auto_install()

        if decision is None:
            message = (
                "Poetry project detected. Allow the agent to run `poetry install` "
                "automatically before launch (will be remembered for this mirror)?"
            )
            if self._prompt_handler is None:
                self.logger.info(
                    "Skipping `poetry install` auto-setup because no prompt handler is available."
                )
                prefs.set_poetry_auto_install(False)
                prefs.save()
                return
            try:
                decision = bool(self._prompt_handler(message))
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("Prompt handler failed (%s); assuming opt-out.", exc)
                decision = False
            prefs.set_poetry_auto_install(decision)
            prefs.save()

        if not decision:
            return

        self.logger.info("Running `poetry install` before launching the agent.")
        try:
            self.executor.run_agent(
                ["poetry", "install"],
                check=True,
                cwd=str(mirror_path),
                capture_output=True,
            )
        except CommandError as exc:
            if self._poetry_python_version_error(exc):
                prefs.set_poetry_auto_install(False)
                prefs.save()
                self.logger.warning(
                    "Poetry auto-install disabled for %s due to incompatible Python interpreter.",
                    ctx.settings.name,
                )
                for highlight in self._poetry_error_highlights(exc):
                    self.logger.warning("  %s", highlight)
                self.logger.warning(
                    "Configure a compatible interpreter with `poetry env use` "
                    "and re-run `poetry install` manually before retrying."
                )
                return
            raise MirrorError(
                "`poetry install` failed while preparing mirror "
                f"{ctx.settings.name}."
            ) from exc

    def _poetry_python_version_error(self, error: CommandError) -> bool:
        """Detect Poetry failures caused by an incompatible Python interpreter."""
        combined = "\n".join(
            part for part in (error.result.stdout, error.result.stderr) if part
        ).lower()
        return (
            "python version" in combined
            and "not supported by the project" in combined
        )

    def _poetry_error_highlights(self, error: CommandError) -> Sequence[str]:
        """Extract notable lines from Poetry output for logging."""
        highlights = []
        for stream in (error.result.stdout, error.result.stderr):
            if not stream:
                continue
            for line in stream.splitlines():
                line = line.strip()
                if not line:
                    continue
                if "Python version" in line or "compatible version" in line:
                    highlights.append(line)
        return highlights

    def bootstrap(
        self,
        ctx: MirrorContext,
        *,
        use_sudo: bool = False,
        setup_agent_remote: bool = True,
        sync: bool = True,
        task_name: Optional[str] = None,
        base_branch: Optional[str] = None,
        extra_args: Optional[Sequence[str]] = None,
        command_override: Optional[Sequence[str]] = None,
        env_override: Optional[Mapping[str, str]] = None,
        supports_inline_prompt: Optional[bool] = None,
        skip_lfs: bool = True,
    ) -> int:
        """One-shot helper to prepare canonical, ensure clone, and launch the agent."""
        if not ctx.is_remote:
            self.prepare_canonical(
                ctx,
                use_sudo=use_sudo,
                setup_agent_remote=setup_agent_remote,
            )
        if ctx.is_remote:
            self.ensure_remote_clone(ctx)
            self._configure_target_remote(ctx)
        else:
            self.ensure_clone(ctx, skip_lfs=skip_lfs)
        return self.launch_agent(
            ctx,
            sync=sync,
            task_name=task_name,
            base_branch=base_branch,
            extra_args=extra_args,
            command_override=command_override,
            env_override=env_override,
            supports_inline_prompt=supports_inline_prompt,
        )

    def _get_merged_templates(
        self,
        command: Sequence[str],
        launcher: AgentLauncher,
    ) -> AgentFlagTemplates:
        """Get merged flag templates based on agent type detection and config precedence."""
        agent_type = _detect_agent_type(command)
        profile = AGENT_PROFILES.get(agent_type, AGENT_PROFILES[AgentType.UNKNOWN])
        global_flags = self.config.agent_launcher.flags if self.config.agent_launcher else None
        return _merge_flag_templates(launcher.flags, global_flags, profile)

    # ------------------------------------------------------------------ Helpers
    def _apply_agent_flag_templates(
        self,
        command: Sequence[str],
        ctx: MirrorContext,
        launcher: AgentLauncher,
        templates: AgentFlagTemplates,
    ) -> List[str]:
        """Translate generic intents into agent-specific flags."""
        if not command:
            return []

        command_list = list(command)

        # needs_yolo intent
        needs_yolo = launcher.needs_yolo
        if needs_yolo is None:
            needs_yolo = self._default_needs_yolo(command_list)
        if needs_yolo and templates.yolo:
            tokens = self._render_flag_template(templates.yolo)
            if tokens and not self._tokens_present_any(command_list, tokens):
                command_list = self._insert_after_executable(command_list, tokens)

        # workdir intent
        if launcher.workdir and templates.workdir:
            tokens = self._render_flag_template(
                templates.workdir,
                path=str(launcher.workdir),
            )
            command_list.extend(tokens)

        # writable_dir intent
        writable_dirs = self._resolved_writable_dirs(ctx, launcher, command_list)
        if writable_dirs and templates.writable_dir:
            for path in writable_dirs:
                tokens = self._render_flag_template(
                    templates.writable_dir,
                    path=str(path),
                )
                if not tokens:
                    continue
                if self._tokens_present(command_list, tokens):
                    continue
                command_list.extend(tokens)

        # default_flag intent
        if templates.default_flag and launcher.default_flags:
            for flag in launcher.default_flags:
                tokens = self._render_flag_template(
                    templates.default_flag,
                    flag=flag,
                )
                command_list.extend(tokens)

        # skills intent
        if templates.skills:
            skill_paths = self._resolved_skill_paths_for_flags(ctx)
            if skill_paths:
                template = templates.skills
                if "{paths" in template:
                    joined = ",".join(str(path) for path in skill_paths)
                    tokens = self._render_flag_template(template, paths=joined)
                    command_list.extend(tokens)
                else:
                    for path in skill_paths:
                        tokens = self._render_flag_template(template, path=str(path))
                        command_list.extend(tokens)

        return command_list

    def _resolved_writable_dirs(
        self,
        ctx: MirrorContext,
        launcher: AgentLauncher,
        command: Sequence[str],
    ) -> List[Path]:
        if launcher.writable_dirs:
            return list(launcher.writable_dirs)

        executable = Path(command[0]).name if command else ""
        if executable != "codex":
            return []

        home_dir = self._agent_home_directory()
        return [home_dir] if home_dir else []

    def _default_needs_yolo(self, command: Sequence[str]) -> bool:
        """Determine if yolo mode should be enabled by default for this agent.

        Returns True if the detected agent type has a yolo template in its profile.
        """
        if not command:
            return False

        agent_type = _detect_agent_type(command)
        profile = AGENT_PROFILES.get(agent_type, AGENT_PROFILES[AgentType.UNKNOWN])

        # Enable yolo by default only if the agent profile defines a yolo template
        if not profile.yolo:
            return False

        executable = Path(command[0]).name
        sandbox_mode = os.environ.get("CODER_SANDBOX_MODE")
        if sandbox_mode and sandbox_mode.lower() == "read-only":
            self.logger.info(
                "Detected CODER_SANDBOX_MODE=read-only; injecting yolo flags for %s to grant write access.",
                executable,
            )
        return True

    def _render_flag_template(self, template: str, **values: str) -> List[str]:
        try:
            rendered = template.format(**values)
        except KeyError as exc:
            self.logger.warning("Missing placeholder %s while rendering flag template %s", exc, template)
            return []
        if not rendered.strip():
            return []
        return shlex.split(rendered)

    @staticmethod
    def _insert_after_executable(command: Sequence[str], tokens: Sequence[str]) -> List[str]:
        if not command:
            return list(tokens)
        return [command[0], *tokens, *command[1:]]

    @staticmethod
    def _tokens_present(command: Sequence[str], tokens: Sequence[str]) -> bool:
        if not tokens:
            return False
        return all(token in command for token in tokens)

    @staticmethod
    def _tokens_present_any(command: Sequence[str], tokens: Sequence[str]) -> bool:
        if not tokens:
            return False
        return any(token in command for token in tokens)

    def _resolved_skill_paths_for_flags(self, ctx: MirrorContext) -> List[Path]:
        """Return skill paths to expose to agents, falling back to default skills."""
        paths: List[Path] = []
        seen: set[Path] = set()

        for entry in ctx.skills:
            candidate = Path(entry).expanduser()
            if candidate.exists() and candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)

        default_dir = self._default_skills_dir()
        if default_dir and default_dir.exists() and default_dir not in seen:
            seen.add(default_dir)
            paths.append(default_dir)

        return paths

    @staticmethod
    def _default_skills_dir() -> Path:
        return Path("~/.sucoder/skills").expanduser()

    def _wrap_with_nvm(self, command: Sequence[str], launcher: AgentLauncher) -> List[str]:
        """Wrap the agent command so it runs under a specific nvm-managed Node version.

        # TODO: The default NVM_DIR assumes ~/.nvm for the agent user.  Custom NVM
        # installs (e.g., via Homebrew or a non-standard path) may need the
        # agent_launcher.nvm.dir config option to be set explicitly.
        """
        nvm_settings = launcher.nvm
        if nvm_settings is None:
            return list(command)

        nvm_dir = nvm_settings.dir
        if nvm_dir is None:
            home = self._agent_home_directory()
            if home is None:
                raise MirrorError(
                    "NVM wrapping requested but the agent home directory could not be resolved."
                )
            nvm_dir = home / ".nvm"

        nvm_dir_str = str(nvm_dir)
        version = nvm_settings.version
        command_str = shlex.join(command)

        script = (
            f'export NVM_DIR={shlex.quote(nvm_dir_str)}; '
            f'[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" || '
            f'{{ echo "nvm.sh not found in $NVM_DIR" >&2; exit 1; }}; '
            f'nvm use {shlex.quote(version)} >/dev/null || exit 1; '
            f'exec {command_str}'
        )
        return ["bash", "-lc", script]

    def _remote_url(self, ctx: MirrorContext, mirror_path: Path, *, push: bool) -> Optional[str]:
        args = ["git", "remote", "get-url"]
        if push:
            args.append("--push")
        args.append(ctx.remote_name)
        try:
            result = self.executor.run_agent(
                args,
                check=True,
                cwd=str(mirror_path),
            )
            url = result.stdout.strip()
            return url or None
        except CommandError as exc:
            self.logger.debug(
                "Failed to read remote url (push=%s) for %s: %s",
                push,
                ctx.remote_name,
                exc,
            )
            return None

    @staticmethod
    def _mode_string(path: Path) -> str:
        try:
            mode = path.stat().st_mode & 0o7777
            return f"{mode:04o}"
        except OSError:
            return "????"

    def _agent_access_summary(self, path: Path, *, require_write: bool) -> str:
        try:
            exists = path.exists()
            is_dir = path.is_dir()
        except OSError:
            return "unreachable"

        if not exists:
            return "missing"

        checks: List[Tuple[str, str]] = [("-r", "readable")]
        if is_dir:
            checks.append(("-x", "executable"))
        if require_write:
            checks.append(("-w", "writable"))

        for flag, label in checks:
            result = self.executor.run_agent(
                ["test", flag, str(path)],
                check=False,
            )
            if result.returncode != 0:
                return f"no {label}"
        return "ok"

    def _agent_home_directory(self) -> Optional[Path]:
        try:
            info = pwd.getpwnam(self.config.agent_user)
        except KeyError:
            self.logger.debug(
                "Agent user %s not found while resolving home directory.", self.config.agent_user
            )
            return None
        home = Path(info.pw_dir)
        if not home.exists():
            self.logger.debug(
                "Resolved home directory %s for agent user %s does not exist.",
                home,
                self.config.agent_user,
            )
            return None
        return home

    def _ensure_mirror_exists(self, ctx: MirrorContext) -> Path:
        if ctx.is_remote:
            return self._ensure_remote_mirror_exists(ctx)
        mirror_path = ctx.mirror_path
        if not self._is_git_repo(mirror_path):
            raise MirrorError(f"Mirror at {mirror_path} does not exist. Run agents-clone first.")
        return mirror_path

    def _ensure_remote_mirror_exists(self, ctx: MirrorContext) -> Path:
        """Verify the remote mirror exists, returning the local mirror_path placeholder.

        For remote mirrors the actual mirror lives on the remote host.
        We check via SSH and return ``ctx.mirror_path`` as a local
        reference (the executor translates paths automatically).
        """
        remote_path = ctx.remote_mirror_path
        assert remote_path is not None
        abs_path = self._resolve_remote_path(ctx)
        check = self.executor.run_agent(
            ["git", "rev-parse", "--git-dir"],
            check=False,
            cwd=abs_path,
        )
        if check.returncode != 0:
            raise MirrorError(
                f"Remote mirror at {remote_path} does not exist. Run agents-clone first."
            )
        return ctx.mirror_path

    def _validate_canonical(self, ctx: MirrorContext) -> None:
        canonical = ctx.canonical_path
        if not canonical.exists():
            raise MirrorError(f"Canonical repository not found at {canonical}")
        if not os.access(canonical, os.R_OK):
            raise MirrorError(f"Canonical repository not readable at {canonical}")

        # Verify the agent cannot write to the canonical repo.
        write_result = self.executor.run_agent(
            ["test", "-w", str(canonical)],
            check=False,
        )
        if write_result.returncode == 0:
            raise MirrorError(
                f"Canonical repository at {canonical} is writable by agent user "
                f"{self.config.agent_user!r}. "
                f"Run `sucoder prepare-canonical` to fix permissions."
            )

    def _verify_remote(self, ctx: MirrorContext) -> None:
        if self.executor.dry_run:
            self.logger.info("Dry-run mode: skipping remote verification.")
            return

        mirror_path = ctx.mirror_path
        remote_url = (
            self.executor.run_agent(
                ["git", "config", "--get", f"remote.{ctx.remote_name}.url"],
                check=True,
                cwd=str(mirror_path),
            ).stdout.strip()
            or None
        )

        expected = str(ctx.canonical_path)
        if remote_url != expected:
            raise MirrorError(
                f"Remote {ctx.remote_name} points to {remote_url}, expected {expected}."
            )

    def _enforce_permissions(self, ctx: MirrorContext) -> None:
        mirror_path = ctx.mirror_path
        apply_agent_repo_permissions(
            self.executor,
            mirror_path,
            agent_group=self.config.agent_group,
        )

        git_dir = _resolve_git_dir(mirror_path)
        try:
            self.executor.run_agent(
                [
                    "find",
                    str(git_dir),
                    "-type",
                    "d",
                    "-exec",
                    "chmod",
                    "g+s",
                    "{}",
                    "+",
                ],
                check=True,
            )
        except CommandError as exc:
            self.logger.warning(
                "Failed to enforce setgid on %s: %s",
                git_dir,
                exc.result.stderr.strip() if exc.result.stderr else exc,
            )

    def _allow_direnv_if_present(self, mirror_path: Path) -> None:
        """Trust a checked-in .envrc so Poetry layouts apply for the agent user."""
        envrc = mirror_path / ".envrc"
        if not envrc.exists():
            return

        if shutil.which("direnv") is None:
            self.logger.info(
                "Skipping direnv allow in %s because direnv is not installed.",
                mirror_path,
            )
            return

        try:
            self.executor.run_agent(
                ["direnv", "allow"],
                check=True,
                cwd=str(mirror_path),
            )
        except CommandError as exc:
            message = ""
            if exc.result.stderr:
                message = exc.result.stderr.strip()
            elif exc.result.stdout:
                message = exc.result.stdout.strip()
            else:
                message = str(exc)
            self.logger.warning(
                "Failed to allow direnv in %s: %s",
                mirror_path,
                message,
            )

    def _ensure_canonical_safe_directory(self, ctx: MirrorContext) -> List[str]:
        """Allow the agent to treat the canonical repo as safe for git operations.

        # TODO: safe.directory entries are only added, never cleaned up.  Over time
        # the global git config may accumulate stale paths for mirrors that no longer
        # exist.  Consider adding a periodic prune step.
        """
        candidates = list(self._canonical_safe_directories(ctx))

        result = self.executor.run_agent(
            ["git", "config", "--global", "--get-all", "safe.directory"],
            check=False,
        )
        existing = {line.strip() for line in result.stdout.splitlines() if line.strip()}

        for path_str in candidates:
            if path_str in existing:
                continue
            try:
                self.executor.run_agent(
                    ["git", "config", "--global", "--add", "safe.directory", path_str],
                    check=True,
                )
                existing.add(path_str)
            except CommandError as exc:
                message = exc.result.stderr.strip() if exc.result.stderr else str(exc)
                self.logger.warning(
                    "Failed to add %s to git safe.directory for %s: %s",
                    path_str,
                    self.executor.agent_user,
                    message,
                )

        return candidates

    @staticmethod
    def _is_git_repo(path: Path) -> bool:
        return (path / ".git").exists()

    def _canonical_safe_directories(self, ctx: MirrorContext) -> List[str]:
        """Return ordered list of paths that should be trusted for git access."""
        canonical_configured = ctx.canonical_path
        git_dir_configured = canonical_configured / ".git"

        candidates: List[Path] = [
            canonical_configured,
            git_dir_configured,
            canonical_configured.resolve(),
            _resolve_git_dir(canonical_configured).resolve(),
        ]

        seen: List[str] = []
        for candidate in candidates:
            try:
                path_str = str(candidate)
            except OSError:
                continue
            if path_str not in seen:
                seen.append(path_str)
        return seen

    def _compose_context_prelude(self, ctx: MirrorContext) -> str:
        blocks: List[str] = []

        system_block = self._system_prompt_block()
        if system_block:
            blocks.append(system_block)

        # Append target-specific prompt snippet if the active target
        # defines one (e.g., platform-specific performance guidance).
        target_block = self._target_prompt_block(ctx)
        if target_block:
            blocks.append(target_block)

        blocks.extend(self._skill_blocks(ctx))

        if not blocks:
            return ""

        separator = "\n\n"
        prelude = separator.join(blocks).strip()
        self.logger.info(
            "Injecting %d context block(s) into agent session.", len(blocks)
        )
        return prelude

    def _system_prompt_block(self) -> Optional[str]:
        prompt_path: Optional[Path] = self.config.system_prompt
        if prompt_path is None:
            prompt_path = self._default_system_prompt_path()
            if not prompt_path.exists():
                return None

        if not prompt_path.exists():
            self.logger.warning("Configured system prompt not found: %s", prompt_path)
            return None

        try:
            content = prompt_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            self.logger.warning(
                "Failed to read system prompt at %s: %s", prompt_path, exc
            )
            return None

        header = f"SYSTEM PROMPT ({prompt_path})"
        return f"{header}\n{content}"

    def _target_prompt_block(self, ctx: MirrorContext) -> Optional[str]:
        """Return an additional prompt block from the active target, if any."""
        remote = ctx.settings.remote
        if remote is None or remote.system_prompt_extra is None:
            return None

        prompt_path = remote.system_prompt_extra
        if not prompt_path.exists():
            self.logger.warning(
                "Target system_prompt_extra not found: %s", prompt_path
            )
            return None

        try:
            content = prompt_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            self.logger.warning(
                "Failed to read target prompt at %s: %s", prompt_path, exc
            )
            return None

        header = f"TARGET PROMPT ({prompt_path})"
        return f"{header}\n{content}"

    def _skill_blocks(self, ctx: MirrorContext) -> List[str]:
        entries: List[Path] = list(ctx.skills)
        default_catalog = self._default_skills_catalog_path()
        if default_catalog:
            entries.append(default_catalog)

        blocks: List[str] = []
        seen: set[Path] = set()
        validated_dirs: set[Path] = set()

        # Check if version validation should be skipped
        skip_version_check = os.environ.get("SUCODER_SKIP_SKILLS_VERSION") == "1"

        for entry in entries:
            resolved = Path(entry).expanduser()

            # Validate skills repository version if directory contains VERSION file
            if resolved.is_dir() and not skip_version_check:
                # Check if this directory (or parent) has VERSION file
                version_file = resolved / "VERSION"
                if version_file.exists() and resolved not in validated_dirs:
                    validate_skills_version(resolved)
                    validated_dirs.add(resolved)
                    self.logger.debug("Skills version validated for: %s", resolved)

            if resolved.is_dir():
                catalog = self._find_catalog_file(resolved)
                if catalog:
                    block = self._render_skill_catalog(catalog, seen)
                    if block:
                        blocks.append(block)
                skill_file = self._find_skill_file(resolved)
                if skill_file:
                    block = self._render_skill_file(skill_file, seen)
                    if block:
                        blocks.append(block)
                continue

            catalog = self._normalize_catalog_path(resolved)
            if catalog:
                block = self._render_skill_catalog(catalog, seen)
                if block:
                    blocks.append(block)
                continue

            skill_path = self._normalize_skill_file_path(resolved)
            if skill_path:
                block = self._render_skill_file(skill_path, seen)
                if block:
                    blocks.append(block)

        return blocks

    @staticmethod
    def _default_system_prompt_path() -> Path:
        return Path("~/.sucoder/system_prompt.org").expanduser()

    @staticmethod
    def _supports_inline_prompt(command: Sequence[str]) -> bool:
        if not command:
            return False
        executable = Path(command[0]).name
        return executable in {"codex", "coder", "claude", "gemini"}

    @staticmethod
    def _default_skills_catalog_path() -> Optional[Path]:
        base = Path("~/.sucoder").expanduser()
        for name in ["SKILLS.org", "skills.org", "SKILLS.md", "skills.md"]:
            candidate = base / name
            if candidate.exists():
                return candidate
        return None

    def _find_skill_file(self, directory: Path) -> Optional[Path]:
        for name in [
            "SKILL.org",
            "Skill.org",
            "skill.org",
            "SKILL.md",
            "Skill.md",
            "skill.md",
        ]:
            candidate = directory / name
            if candidate.exists():
                return candidate
        return None

    def _normalize_skill_file_path(self, path: Path) -> Optional[Path]:
        candidate = path.expanduser()
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

        if candidate.exists() and candidate.is_dir():
            return self._find_skill_file(candidate)

        directory = candidate.parent
        stem = candidate.stem
        for name in [stem, "SKILL", "Skill", "skill"]:
            for ext in [".org", ".md"]:
                check = directory / f"{name}{ext}"
                if check.exists():
                    return check.resolve()
        return None

    def _find_catalog_file(self, directory: Path) -> Optional[Path]:
        for name in [
            "SKILLS.org",
            "Skills.org",
            "skills.org",
            "SKILLS.md",
            "Skills.md",
            "skills.md",
        ]:
            candidate = directory / name
            if candidate.exists():
                return candidate
        return None

    def _normalize_catalog_path(self, path: Path) -> Optional[Path]:
        candidate = path.expanduser()
        if candidate.exists():
            if candidate.is_dir():
                return self._find_catalog_file(candidate)
            return candidate.resolve()

        directory = candidate.parent
        stem = candidate.stem
        for name in [stem, "SKILLS", "Skills", "skills"]:
            for ext in [".org", ".md"]:
                check = directory / f"{name}{ext}"
                if check.exists():
                    return check.resolve()
        return None

    def _render_skill_file(self, skill_file: Path, seen: set[Path]) -> Optional[str]:
        resolved = self._normalize_skill_file_path(skill_file)
        if not resolved:
            self.logger.debug("Skill entry %s not found, skipping.", skill_file)
            return None

        if resolved in seen:
            return None
        seen.add(resolved)

        try:
            body = resolved.read_text(encoding="utf-8").strip()
        except OSError as exc:
            self.logger.warning("Failed to read skill file %s: %s", resolved, exc)
            return None

        metadata = _read_skill_metadata(resolved)
        if metadata:
            name, description = metadata
            header = f"SKILL: {name}"
            if description:
                header += f" — {description}"
            self.logger.info("Loaded skill %s (%s)", name, resolved)
        else:
            header = f"SKILL FILE: {resolved}"
            self.logger.info("Loaded skill file %s", resolved)

        resources = self._render_resource_summary(resolved)
        if resources:
            return f"{header}\n{body}\n\n{resources}"
        return f"{header}\n{body}"

    def _render_skill_catalog(self, catalog: Path, seen: set[Path]) -> Optional[str]:
        resolved = self._normalize_catalog_path(catalog)
        if not resolved:
            self.logger.debug("Skill catalog %s not found, skipping.", catalog)
            return None

        if resolved in seen:
            return None
        seen.add(resolved)

        try:
            content = resolved.read_text(encoding="utf-8")
        except OSError as exc:
            self.logger.warning("Failed to read skills catalog %s: %s", resolved, exc)
            return None

        header = "SKILL CATALOG"
        metadata = _read_skill_metadata(resolved)
        if metadata:
            name, description = metadata
            header = f"SKILL CATALOG: {name}"
            if description:
                header += f" — {description}"

        lines: List[str] = [header]
        references = self._parse_skill_catalog(resolved, content)
        if not references:
            lines.append("(No additional skills referenced.)")
        else:
            lines.append("The following skills are available on demand:")
            for ref in references:
                lines.append(self._format_skill_reference(ref))

        return "\n".join(lines)

    def _parse_skill_catalog(self, catalog: Path, content: str) -> List[Path]:
        references: List[Path] = []
        seen: set[Path] = set()

        def add_reference(raw: str) -> None:
            target = raw.strip()
            if not target:
                return
            target = target.strip('<>"\'')
            target = target.split("::", 1)[0]
            path = (
                Path(target).expanduser()
                if target.startswith("~") or target.startswith("/")
                else catalog.parent / target
            )
            normalized = (
                self._normalize_skill_file_path(path)
                or self._normalize_catalog_path(path)
                or path
            )
            if normalized not in seen:
                seen.add(normalized)
                references.append(normalized)

        for match in re.finditer(r"file:([^\s\]]+)", content):
            add_reference(match.group(1))

        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.lower().startswith("file:"):
                add_reference(stripped[5:])
            elif stripped.startswith("- "):
                candidate = stripped[2:].split()[0]
                if candidate and not candidate.startswith("file:") and "://" not in candidate:
                    add_reference(candidate)

        return references

    @staticmethod
    def _readable_skill_name(path: Path, metadata: Optional[Tuple[str, str]]) -> Tuple[str, str]:
        if metadata:
            name, description = metadata
            return name, description
        return path.stem, ""

    def _file_read_hint(self, path: Path) -> str:
        """Return an agent-appropriate file-read command hint for the given path."""
        agent_type = getattr(self, "_detected_agent_type", AgentType.UNKNOWN)
        if agent_type == AgentType.CODEX:
            return f"codex read {path}"
        if agent_type == AgentType.CLAUDE:
            return f"Read tool: {path}"
        if agent_type == AgentType.GEMINI:
            return f"read {path}"
        return f"load {path}"

    def _format_skill_reference(self, reference: Path) -> str:
        normalized = (
            self._normalize_skill_file_path(reference)
            or self._normalize_catalog_path(reference)
            or reference.expanduser()
        )
        metadata = _read_skill_metadata(normalized) if normalized.exists() else None
        name, description = self._readable_skill_name(normalized, metadata)
        line = f"- {name}"
        if description:
            line += f" — {description}"
        if normalized.exists():
            line += f" (load with `{self._file_read_hint(normalized)}`)"
        return line

    def _render_resource_summary(self, skill_file: Path) -> str:
        skill_dir = skill_file.parent
        sections: List[str] = []

        references_section = self._render_reference_section(skill_dir)
        if references_section:
            sections.append(references_section)

        scripts_section = self._render_scripts_section(skill_dir)
        if scripts_section:
            sections.append(scripts_section)

        assets_section = self._render_assets_section(skill_dir)
        if assets_section:
            sections.append(assets_section)

        if not sections:
            return ""
        return "RESOURCES\n" + "\n\n".join(sections)

    def _render_reference_section(self, skill_dir: Path) -> str:
        references_dir = skill_dir / "references"
        if not references_dir.exists():
            return ""

        files = sorted(p for p in references_dir.rglob("*") if p.is_file())
        if not files:
            return ""

        lines = [
            "References (load specific files when needed):",
        ]
        for path in files[:20]:
            rel = path.relative_to(skill_dir)
            lines.append(f"- {rel} — load with `{self._file_read_hint(path)}`")
        if len(files) > 20:
            lines.append(f"- ... ({len(files) - 20} more)")
        return "\n".join(lines)

    def _render_scripts_section(self, skill_dir: Path) -> str:
        scripts_dir = skill_dir / "scripts"
        if not scripts_dir.exists():
            return ""

        files = sorted(p for p in scripts_dir.rglob("*") if p.is_file())
        if not files:
            return ""

        lines = [
            "Scripts (review before running; execute manually as needed):",
        ]
        for path in files[:20]:
            rel = path.relative_to(skill_dir)
            extension = path.suffix.lower()
            if extension in {".py"}:
                suggestion = f"python {path}"
            elif extension in {".sh", ".bash"}:
                suggestion = f"bash {path}"
            else:
                suggestion = str(path)
            lines.append(f"- {rel} — e.g., `{suggestion}`")
        if len(files) > 20:
            lines.append(f"- ... ({len(files) - 20} more)")
        return "\n".join(lines)

    def _render_assets_section(self, skill_dir: Path) -> str:
        assets_dir = skill_dir / "assets"
        if not assets_dir.exists():
            return ""

        files = sorted(p for p in assets_dir.rglob("*") if p.is_file())
        if not files:
            return ""

        lines = [
            "Assets (supporting files to incorporate into outputs):",
        ]
        for path in files[:20]:
            rel = path.relative_to(skill_dir)
            lines.append(f"- {rel}")
        if len(files) > 20:
            lines.append(f"- ... ({len(files) - 20} more)")
        return "\n".join(lines)

    def _configure_agent_remote(self, ctx: MirrorContext) -> None:
        """Ensure the canonical repo has a remote pointing at the agent mirror."""
        canonical = ctx.canonical_path
        remote_name = ctx.agent_prefix
        remote_url = str(ctx.mirror_path)

        # Verify canonical is a git repository.
        git_dir = canonical / ".git"
        if not git_dir.exists():
            self.logger.debug(
                "Canonical repository %s does not look like a non-bare git repo; skipping remote setup.",
                canonical,
            )
            return

        result = self.executor.run_human(
            ["git", "remote", "get-url", remote_name],
            check=False,
            cwd=str(canonical),
        )
        if result.returncode != 0:
            self.logger.info("Adding remote %s -> %s", remote_name, remote_url)
            self.executor.run_human(
                ["git", "remote", "add", remote_name, remote_url],
                check=True,
                cwd=str(canonical),
            )
        else:
            existing_url = result.stdout.strip()
            if existing_url != remote_url:
                self.logger.info(
                    "Updating remote %s URL from %s to %s",
                    remote_name,
                    existing_url,
                    remote_url,
                )
                self.executor.run_human(
                    ["git", "remote", "set-url", remote_name, remote_url],
                    check=True,
                    cwd=str(canonical),
                )

        fetch_key = f"remote.{remote_name}.fetch"
        desired_spec = (
            f"+refs/heads/{ctx.agent_prefix}/*:refs/remotes/{remote_name}/{ctx.agent_prefix}/*"
        )
        fetch_specs = self.executor.run_human(
            ["git", "config", "--get-all", fetch_key],
            check=False,
            cwd=str(canonical),
        )
        existing_specs = {line.strip() for line in fetch_specs.stdout.splitlines()}
        if desired_spec not in existing_specs:
            self.logger.info(
                "Adding fetch spec for remote %s: %s", remote_name, desired_spec
            )
            self.executor.run_human(
                ["git", "config", "--add", fetch_key, desired_spec],
                check=True,
                cwd=str(canonical),
            )

        mirror_path = ctx.mirror_path
        mirror_dotgit = mirror_path / ".git"
        safe_paths = [str(mirror_path), str(mirror_dotgit)]
        existing_safe = self.executor.run_human(
            ["git", "config", "--global", "--get-all", "safe.directory"],
            check=False,
        )
        known = {line.strip() for line in existing_safe.stdout.splitlines()}
        for path_str in safe_paths:
            if path_str in known:
                continue
            self.logger.info("Trusting agent mirror path %s", path_str)
            self.executor.run_human(
                ["git", "config", "--global", "--add", "safe.directory", path_str],
                check=True,
            )

    def _configure_target_remote(self, ctx: MirrorContext) -> None:
        """Ensure the canonical repo has a remote pointing at the remote target mirror.

        For a target named ``savio`` and mirror ``K-Aggregators``, this
        adds a remote ``savio`` with URL ``gateway:~/mirrors/K-Aggregators``
        so the human can ``git fetch savio`` to pull back agent work.
        """
        remote = ctx.settings.remote
        if remote is None:
            return
        canonical = ctx.canonical_path
        git_dir = canonical / ".git"
        if not git_dir.exists():
            return

        # Derive target name from the session state or fall back to gateway hostname.
        obj = {}
        try:
            import click as _click
            obj = (_click.get_current_context().obj or {})
        except RuntimeError:
            pass
        target_name = obj.get("target_name") or remote.gateway.split(".")[0]

        remote_url = f"{remote.gateway}:{ctx.remote_mirror_path}"

        result = self.executor.run_human(
            ["git", "remote", "get-url", target_name],
            check=False,
            cwd=str(canonical),
        )
        if result.returncode != 0:
            self.logger.info("Adding remote %s -> %s", target_name, remote_url)
            self.executor.run_human(
                ["git", "remote", "add", target_name, remote_url],
                check=True,
                cwd=str(canonical),
            )
        else:
            existing_url = result.stdout.strip()
            if existing_url != remote_url:
                self.logger.info(
                    "Updating remote %s URL from %s to %s",
                    target_name, existing_url, remote_url,
                )
                self.executor.run_human(
                    ["git", "remote", "set-url", target_name, remote_url],
                    check=True,
                    cwd=str(canonical),
                )

    def _write_agent_fetch_helper(self, ctx: MirrorContext) -> None:
        """Create or refresh a helper script to fetch and list agent branches."""
        canonical = ctx.canonical_path
        scripts_dir = canonical / "scripts"
        script_path = scripts_dir / "fetch-agent-branches.sh"

        if self.executor.dry_run:
            self.logger.info("DRY-RUN: would ensure helper script at %s", script_path)
            return

        scripts_dir.mkdir(parents=True, exist_ok=True)
        remote_name = ctx.agent_prefix
        prefix = ctx.agent_prefix
        remote_default = shlex.quote(remote_name)
        prefix_default = shlex.quote(prefix)
        script_contents = f"""#!/usr/bin/env bash
set -euo pipefail

remote=${{1:-{remote_default}}}
prefix=${{2:-{prefix_default}}}

git fetch --no-recurse-submodules "${{remote}}"
git for-each-ref "refs/remotes/${{remote}}/${{prefix}}/" --format='%(refname:strip=2)'
"""
        current = script_path.read_text(encoding="utf-8") if script_path.exists() else ""
        if current != script_contents:
            script_path.write_text(script_contents, encoding="utf-8")
            script_path.chmod(0o755)
            self.logger.info("Wrote helper script %s", script_path)


def _sanitize_task_name(raw: str) -> str:
    """Sanitize a task name for use in a git branch."""
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-"
    cleaned = []
    for char in raw.lower():
        if char in allowed:
            cleaned.append(char)
        elif char.isalnum():
            cleaned.append(char.lower())
        else:
            cleaned.append("-")

    sanitized = "".join(cleaned).strip("-")
    sanitized = "-".join(filter(None, sanitized.split("-")))
    if not sanitized:
        raise MirrorError("Task name produces an empty branch component after sanitization.")
    return sanitized


def _resolve_git_dir(canonical: Path) -> Path:
    """Return the directory whose permissions should be shared with the agent."""
    git_dir = canonical / ".git"
    if git_dir.is_dir():
        return git_dir
    return canonical


def _read_skill_metadata(skill_file: Path) -> Optional[Tuple[str, str]]:
    """Extract (title, description) metadata from an Org or Markdown skill file."""
    try:
        content = skill_file.read_text(encoding="utf-8")
    except OSError:
        return None

    stripped = content.lstrip()

    if stripped.startswith("---"):
        lines = content.splitlines()
        yaml_lines: List[str] = []
        for line in lines[1:]:
            if line.strip().startswith("---"):
                break
            yaml_lines.append(line)
        if yaml_lines:
            try:
                data = yaml.safe_load("\n".join(yaml_lines)) or {}
            except yaml.YAMLError:
                data = {}
            name = data.get("name") or data.get("title")
            description = data.get("description") or data.get("summary")
            if name:
                return (str(name), str(description or ""))

    title: Optional[str] = None
    org_description: Optional[str] = None
    for line in content.splitlines():
        stripped_line = line.strip()
        if stripped_line.lower().startswith("#+title:"):
            title = stripped_line.split(":", 1)[1].strip() or None
        elif stripped_line.lower().startswith("#+description:"):
            org_description = stripped_line.split(":", 1)[1].strip() or None
        if title and org_description:
            break

    if not title:
        return None
    return (title, org_description or "")


def _detect_agent_type(command: Sequence[str]) -> AgentType:
    """Detect agent type from command executable."""
    if not command:
        return AgentType.UNKNOWN
    executable = Path(command[0]).name
    if executable == "claude":
        return AgentType.CLAUDE
    if executable == "codex":
        return AgentType.CODEX
    if executable == "gemini":
        return AgentType.GEMINI
    return AgentType.UNKNOWN


def _merge_flag_templates(
    per_mirror: AgentFlagTemplates,
    global_config: Optional[AgentFlagTemplates],
    profile: AgentFlagTemplates,
) -> AgentFlagTemplates:
    """Merge flag templates with precedence: per-mirror > global > profile.

    For each field, use the first non-None value in precedence order.
    """

    def _pick(field_name: str) -> Optional[str]:
        # Per-mirror has highest priority
        val = getattr(per_mirror, field_name)
        if val is not None:
            return val
        # Then global config
        if global_config is not None:
            val = getattr(global_config, field_name)
            if val is not None:
                return val
        # Finally, agent profile
        return getattr(profile, field_name)

    return AgentFlagTemplates(
        yolo=_pick("yolo"),
        writable_dir=_pick("writable_dir"),
        workdir=_pick("workdir"),
        default_flag=_pick("default_flag"),
        skills=_pick("skills"),
        system_prompt=_pick("system_prompt"),
    )
