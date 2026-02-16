"""Typer-powered CLI entry point."""

from __future__ import annotations

import os
import shlex
import pwd
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import click

try:
    from click.shell_completion import CompletionItem as ClickCompletionItem
except (ImportError, AttributeError):  # pragma: no cover - defensive
    ClickCompletionItem = None  # type: ignore[assignment]

import typer

from . import __version__
from .config import (
    AgentLauncher,
    BranchPrefixes,
    Config,
    ConfigError,
    MirrorSettings,
    _detect_git_toplevel,
    build_default_config,
    load_config,
)
from .executor import CommandExecutor
from .logging_utils import setup_logger
from .mirror import MirrorError, MirrorManager
from .startup_checks import StartupError, run_startup_checks

app = typer.Typer(help="sucoder – Unix-sandboxed agent collaboration toolkit for managing agent mirrors.")


def _default_config_path() -> Path:
    return Path("~/.sucoder/config.yaml").expanduser()


def _load_config_path(config_path: Optional[Path]) -> Path:
    if config_path:
        return Path(config_path).expanduser()
    return _default_config_path()


def _get_config(ctx: typer.Context) -> Config:
    obj = ctx.obj or {}
    config = obj.get("config")
    if config is None:
        raise typer.Exit(code=2)
    return config


def _get_use_sudo_for_agent(ctx: Optional[click.Context], config: Optional[Config] = None) -> bool:
    obj = (ctx.obj if ctx and ctx.obj else {}) or {}
    use_sudo = obj.get("use_sudo_for_agent")
    if use_sudo is None:
        use_sudo = True

    # If we're already running as the agent user, skip sudo to avoid failures in restricted environments.
    if config and config.agent_user == pwd.getpwuid(os.getuid()).pw_name:
        return False

    return bool(use_sudo)


def _get_config_for_completion(ctx: typer.Context) -> Optional[Config]:
    """Best-effort resolution of configuration during shell completion."""
    obj = ctx.obj or {}
    config = obj.get("config")
    if isinstance(config, Config):
        return config

    config_param = ctx.params.get("config") if ctx.params else None
    try:
        config_path = _load_config_path(config_param)
    except Exception:
        return None

    try:
        return load_config(config_path)
    except ConfigError:
        pass

    try:
        return build_default_config()
    except ConfigError:
        return None


def _mirror_completion(
    ctx: typer.Context, param, incomplete: str
) -> List[Any]:
    config = _get_config_for_completion(ctx)
    if not config:
        return []
    items: List[Any] = []
    for name, settings in sorted(config.mirrors.items()):
        if incomplete and not name.startswith(incomplete):
            continue
        help_text = str(settings.canonical_repo)
        if ClickCompletionItem is not None:
            items.append(ClickCompletionItem(name, help=help_text))
        else:
            items.append(name)
    return items


def _build_executor(
    config: Config,
    logger,
    dry_run: bool,
    *,
    use_sudo_for_agent: bool = True,
) -> CommandExecutor:
    return CommandExecutor(
        human_user=config.human_user,
        agent_user=config.agent_user,
        agent_group=config.agent_group,
        logger=logger,
        dry_run=dry_run,
        use_sudo_for_agent=use_sudo_for_agent,
    )


def _prompt_yes_no(message: str) -> bool:
    return typer.confirm(message, default=True)


def _build_manager(config: Config, logger, dry_run: bool) -> MirrorManager:
    ctx = None
    try:
        ctx = click.get_current_context()
    except RuntimeError:
        ctx = None

    executor = _build_executor(
        config,
        logger,
        dry_run=dry_run,
        use_sudo_for_agent=_get_use_sudo_for_agent(ctx, config),
    )
    return MirrorManager(config, executor, logger, prompt_handler=_prompt_yes_no)


def _resolve_mirror_name(ctx: typer.Context, mirror: Optional[str]) -> str:
    """Return the mirror name, defaulting to the sole mirror when omitted.

    Resolution order when *mirror* is ``None``:

    1. If the config contains exactly one mirror, use it.
    2. Detect the git root of the current working directory and match it
       against the ``canonical_repo`` of every configured mirror.
    3. If no configured mirror matches, create an ephemeral
       :class:`MirrorSettings` from the git root (analogous to
       :func:`build_default_config`) and inject it into the in-memory
       config so downstream code can use it normally.
    4. If we're not inside a git repo at all, fall through to the
       original "Multiple mirrors configured" error.
    """
    if mirror is not None:
        return mirror
    config = _get_config(ctx)
    names = list(config.mirrors.keys())
    if len(names) == 1:
        return names[0]

    # Step 1 & 2: try git-based detection
    try:
        git_toplevel = _detect_git_toplevel()
    except ConfigError:
        # Not inside a git repo – fall through to the error.
        raise typer.BadParameter(
            f"Multiple mirrors configured; specify one of: {', '.join(sorted(names))}",
            param_hint="MIRROR",
        )

    # Step 1: match cwd's repo root against configured mirrors.
    resolved_toplevel = git_toplevel.resolve()
    for name, settings in config.mirrors.items():
        if settings.canonical_repo.resolve() == resolved_toplevel:
            return name

    # Step 2: create an ephemeral mirror for an unconfigured repo.
    mirror_name = git_toplevel.name
    if mirror_name not in config.mirrors:
        prefixes = BranchPrefixes(human=config.human_user, agent=config.agent_user)
        launcher = config.agent_launcher or AgentLauncher()
        ephemeral = MirrorSettings(
            name=mirror_name,
            canonical_repo=git_toplevel,
            mirror_name=mirror_name,
            branch_prefixes=prefixes,
            agent_launcher=launcher,
            skills=list(config.skills),
        )
        # config.mirrors is typed as Mapping but is a plain dict at runtime.
        config.mirrors[mirror_name] = ephemeral  # type: ignore[index]
        return mirror_name

    # Name collision with an existing mirror – require explicit selection.
    raise typer.BadParameter(
        f"Multiple mirrors configured; specify one of: {', '.join(sorted(names))}",
        param_hint="MIRROR",
    )


def _agent_shorthand(name: str) -> List[str]:
    """Turn a short agent name into a command list."""
    return [name]


def _parse_agent_command(command: Optional[str]) -> Optional[List[str]]:
    if command is None:
        return None
    parts = shlex.split(command)
    return parts or None


def _parse_agent_env(entries: Optional[List[str]]) -> Optional[Dict[str, str]]:
    if not entries:
        return None
    env: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise typer.BadParameter(
                f"Environment override must be KEY=VALUE, received `{entry}`."
            )
        key, value = entry.split("=", 1)
        if not key:
            raise typer.BadParameter("Environment variable name cannot be empty.")
        env[key] = value
    return env


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML (defaults to ~/.sucoder/config.yaml).",
    ),
    use_sudo_for_agent: bool = typer.Option(
        True,
        "--agent-sudo/--no-agent-sudo",
        help="Use sudo to impersonate the agent user when running agent commands (default: enabled).",
    ),
) -> None:
    """Load configuration once and store it on the Typer context."""
    config_explicitly_set = config is not None
    default_path = _default_config_path()
    is_default_config = False
    config_path: Optional[Path] = None

    if config_explicitly_set:
        # User passed --config explicitly; always load from file.
        config_path = Path(config).expanduser()  # type: ignore[arg-type]
        try:
            loaded_config = load_config(config_path)
        except ConfigError as exc:
            typer.echo(f"Configuration error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    elif default_path.exists():
        # Default config file exists; load it.
        config_path = default_path
        try:
            loaded_config = load_config(config_path)
        except ConfigError as exc:
            typer.echo(f"Configuration error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    else:
        # Zero-config mode: derive configuration from the environment.
        try:
            loaded_config = build_default_config()
        except ConfigError as exc:
            typer.echo(f"Configuration error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        is_default_config = True

    try:
        run_startup_checks(
            loaded_config,
            config_path,
            use_sudo=use_sudo_for_agent,
        )
    except StartupError as exc:
        if is_default_config:
            typer.echo(f"Warning: {exc}", err=True)
        else:
            typer.echo(f"Startup validation failed: {exc}", err=True)
            raise typer.Exit(code=2) from exc

    ctx.obj = {
        "config": loaded_config,
        "config_path": config_path,
        "use_sudo_for_agent": use_sudo_for_agent,
        "is_default_config": is_default_config,
    }


@app.command("version")
def version() -> None:
    """Display version information."""
    typer.echo(__version__)


@app.command("agents-clone")
def agents_clone(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without executing."),
    lfs: bool = typer.Option(
        False,
        "--lfs/--no-lfs",
        help="Download Git LFS objects during clone (default: skip LFS to avoid failures).",
    ),
) -> None:
    """Clone the canonical repository into an agent-controlled mirror."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    manager = _build_manager(config, logger, dry_run=dry_run)
    try:
        manager.ensure_clone(manager.context_for(mirror), skip_lfs=not lfs)
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command("prepare-canonical")
def prepare_canonical(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without executing."),
    use_sudo: bool = typer.Option(
        False,
        "--sudo/--no-sudo",
        help="Prefix corrective commands with sudo (default: --no-sudo).",
    ),
    agent_remote: bool = typer.Option(
        True,
        "--agent-remote/--no-agent-remote",
        help="Configure a remote named after the agent prefix pointing to the mirror (default: enabled).",
    ),
) -> None:
    """Fix ownership and permissions on the canonical repository."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    manager = _build_manager(config, logger, dry_run=dry_run)
    try:
        manager.prepare_canonical(
            manager.context_for(mirror),
            use_sudo=use_sudo,
            setup_agent_remote=agent_remote,
        )
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command("sync")
def sync(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without executing."),
) -> None:
    """Fetch updates from the canonical repository."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    manager = _build_manager(config, logger, dry_run=dry_run)
    try:
        manager.sync(manager.context_for(mirror))
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command("start-task")
def start_task(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    task: str = typer.Argument(..., help="Task identifier used to name the branch."),
    base: Optional[str] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base branch name (falls back to mirror default).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without executing."),
) -> None:
    """Create and check out a task branch for the agent."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    manager = _build_manager(config, logger, dry_run=dry_run)
    try:
        branch = manager.start_task(
            manager.context_for(mirror),
            task_name=task,
            base_branch=base,
        )
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(branch)


@app.command("status")
def status(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
) -> None:
    """Display git status for the mirror."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    manager = _build_manager(config, logger, dry_run=False)
    try:
        output = manager.status(manager.context_for(mirror))
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(output)


@app.command("agents-run")
def agents_run(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        "-t",
        help="Optional task identifier for creating a fresh branch before launch.",
    ),
    base: Optional[str] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base branch to use when creating a task branch.",
    ),
    sync: bool = typer.Option(
        True,
        "--sync/--no-sync",
        help="Fetch latest canonical changes before launching (ignored when --task is used).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without executing."),
    agent: Optional[str] = typer.Option(
        None,
        "--agent",
        "-a",
        help="Agent to use (e.g. claude, codex, gemini).",
    ),
    agent_command: Optional[str] = typer.Option(
        None,
        "--agent-command",
        help="Override the full agent command (example: --agent-command 'foo --flag').",
    ),
    agent_env: Optional[List[str]] = typer.Option(
        None,
        "--agent-env",
        help="Override or add agent environment variables (repeat as KEY=VALUE).",
        metavar="KEY=VALUE",
    ),
    inline_prompt: Optional[bool] = typer.Option(
        None,
        "--inline-prompt/--no-inline-prompt",
        help="Force whether context prelude text is appended to the agent command.",
    ),
    lfs: bool = typer.Option(
        False,
        "--lfs/--no-lfs",
        help="Download Git LFS objects during clone (default: skip LFS to avoid failures).",
    ),
    extra_args: Optional[List[str]] = typer.Argument(
        None,
        help="Additional arguments appended to the agent launch command.",
        metavar="ARGS...",
    ),
) -> None:
    """Launch the configured agent inside the mirror working tree."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    manager = _build_manager(config, logger, dry_run=dry_run)
    command_override = _parse_agent_command(agent_command) or (_agent_shorthand(agent) if agent else None)
    env_override = _parse_agent_env(agent_env)
    try:
        manager.launch_agent(
            manager.context_for(mirror),
            sync=sync,
            task_name=task,
            base_branch=base,
            extra_args=extra_args,
            command_override=command_override,
            env_override=env_override,
            supports_inline_prompt=inline_prompt,
        )
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command("collaborate")
def collaborate(
    ctx: typer.Context,
    mirror: Optional[str] = typer.Argument(None, help="Mirror name defined in configuration.", shell_complete=_mirror_completion),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        "-t",
        help="Optional task identifier used to create a task branch before launch.",
    ),
    base: Optional[str] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base branch to use when creating a task branch (defaults to mirror setting).",
    ),
    sync: bool = typer.Option(
        True,
        "--sync/--no-sync",
        help="Fetch latest canonical changes before launching (ignored when --task is used).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without executing."),
    use_sudo: bool = typer.Option(
        False,
        "--sudo/--no-sudo",
        help="Prefix canonical permission commands with sudo (default: --no-sudo).",
    ),
    agent_remote: bool = typer.Option(
        True,
        "--agent-remote/--no-agent-remote",
        help="Configure a remote pointing at the agent mirror during canonical prep (default: enabled).",
    ),
    agent: Optional[str] = typer.Option(
        None,
        "--agent",
        "-a",
        help="Agent to use (e.g. claude, codex, gemini).",
    ),
    agent_command: Optional[str] = typer.Option(
        None,
        "--agent-command",
        help="Override the full agent command (example: --agent-command 'foo --flag').",
    ),
    agent_env: Optional[List[str]] = typer.Option(
        None,
        "--agent-env",
        help="Override or add agent environment variables (repeat as KEY=VALUE).",
        metavar="KEY=VALUE",
    ),
    inline_prompt: Optional[bool] = typer.Option(
        None,
        "--inline-prompt/--no-inline-prompt",
        help="Force whether context prelude text is appended to the agent command.",
    ),
    lfs: bool = typer.Option(
        False,
        "--lfs/--no-lfs",
        help="Download Git LFS objects during clone (default: skip LFS to avoid failures).",
    ),
    extra_args: Optional[List[str]] = typer.Argument(
        None,
        help="Additional arguments appended to the agent launch command.",
        metavar="ARGS...",
    ),
) -> None:
    """Prepare canonical, ensure the mirror exists, and launch the agent in one step."""
    mirror = _resolve_mirror_name(ctx, mirror)
    config = _get_config(ctx)
    logger = setup_logger(f"sucoder.{mirror}", config.log_dir, verbose)
    manager = _build_manager(config, logger, dry_run=dry_run)
    command_override = _parse_agent_command(agent_command) or (_agent_shorthand(agent) if agent else None)
    env_override = _parse_agent_env(agent_env)
    try:
        manager.bootstrap(
            manager.context_for(mirror),
            use_sudo=use_sudo,
            setup_agent_remote=agent_remote,
            sync=sync,
            task_name=task,
            base_branch=base,
            extra_args=extra_args,
            command_override=command_override,
            env_override=env_override,
            supports_inline_prompt=inline_prompt,
            skip_lfs=not lfs,
        )
    except MirrorError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command("mirrors-list")
def mirrors_list(ctx: typer.Context) -> None:
    """Display configured mirrors with their canonical repositories."""
    config = _get_config(ctx)
    entries = sorted(config.mirrors.items())
    if not entries:
        typer.echo("No mirrors configured.")
        return

    name_width = max(len("Mirror"), *(len(name) for name, _ in entries))
    branch_width = max(len("Base"), *(len(settings.default_base_branch) for _, settings in entries))

    header = f"{'Mirror':<{name_width}}  {'Base':<{branch_width}}  Canonical Repo  Mirror Path"
    typer.echo(header)
    typer.echo("-" * len(header))

    for name, settings in entries:
        canonical = str(settings.canonical_repo)
        mirror_path = str(config.mirror_root / settings.mirror_dirname)
        base = settings.default_base_branch
        typer.echo(f"{name:<{name_width}}  {base:<{branch_width}}  {canonical}  {mirror_path}")


@app.command("skills-list")
def skills_list(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase console logging."),
) -> None:
    """List configured skill paths and highlight accessibility issues."""
    config = _get_config(ctx)
    logger = setup_logger("sucoder.skills", config.log_dir, verbose)
    executor = _build_executor(
        config,
        logger,
        dry_run=False,
        use_sudo_for_agent=_get_use_sudo_for_agent(ctx, config),
    )

    path_usage: Dict[Path, Set[str]] = defaultdict(set)
    for mirror_name, settings in config.mirrors.items():
        for entry in settings.skills:
            path_usage[entry].add(mirror_name)

    default_skills_dir = Path("~/.sucoder/skills").expanduser()
    path_usage.setdefault(default_skills_dir, set()).add("default")

    catalog_path = MirrorManager._default_skills_catalog_path()
    if catalog_path:
        path_usage[catalog_path].add("catalog")
        path_usage.setdefault(catalog_path.parent, set()).add("catalog")

    if not path_usage:
        typer.echo("No skill paths configured.")
        raise typer.Exit(code=1)

    exit_code = 0
    for raw_path in sorted(path_usage.keys(), key=lambda p: str(p)):
        contexts = ", ".join(sorted(path_usage[raw_path]))
        try:
            path = raw_path
            exists = path.exists()
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Failed to stat skill path %s: %s", raw_path, exc)
            typer.secho(f"[ERROR] {raw_path} (contexts: {contexts}) – {exc}", fg="red")
            exit_code = 1
            continue

        if exists:
            if path.is_dir():
                readable = path.is_dir() and _agent_can_access_path(path, executor)
                status = "OK" if readable else "UNREADABLE"
                color = "green" if readable else "yellow"
                typer.secho(
                    f"[{status}] directory {path} (contexts: {contexts})",
                    fg=color,
                )
                if readable:
                    entries = _collect_directory_preview(path, executor)
                    if entries:
                        typer.echo(f"  sample: {', '.join(entries)}")
                else:
                    exit_code = 1
            else:
                readable = _agent_can_read_file(path, executor)
                status = "OK" if readable else "UNREADABLE"
                color = "green" if readable else "yellow"
                typer.secho(
                    f"[{status}] file {path} (contexts: {contexts})",
                    fg=color,
                )
                if not readable:
                    exit_code = 1
        else:
            typer.secho(
                f"[MISSING] {path} (contexts: {contexts})",
                fg="red",
            )
            exit_code = 1

    raise typer.Exit(code=exit_code)


def _agent_can_read_file(path: Path, executor: CommandExecutor) -> bool:
    """Return True when the agent user can read the file."""
    if not path.exists() or path.is_dir():
        return False
    result = executor.run_agent(
        ["test", "-r", str(path)],
        check=False,
    )
    return result.returncode == 0


def _agent_can_access_path(path: Path, executor: CommandExecutor) -> bool:
    """Return True when the agent user can read and execute the directory."""
    if not path.exists() or not path.is_dir():
        return False
    result = executor.run_agent(
        ["test", "-r", str(path), "-a", "-x", str(path)],
        check=False,
    )
    return result.returncode == 0


def _collect_directory_preview(path: Path, executor: CommandExecutor, limit: int = 6) -> List[str]:
    """Return a small, sorted sample of entries from a directory."""
    try:
        result = executor.run_agent(
            ["ls", "-1", str(path)],
            check=False,
        )
        if result.returncode != 0:
            typer.secho(f"  failed to list {path}: {result.stderr.strip()}", fg="yellow")
            return []
        entries = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
    except OSError as exc:  # pragma: no cover - defensive
        typer.secho(f"  failed to list {path}: {exc}", fg="yellow")
        return []
    if len(entries) > limit:
        extras = len(entries) - limit
        trimmed = entries[:limit]
        trimmed.append(f"...(+{extras})")
        return trimmed
    return entries
