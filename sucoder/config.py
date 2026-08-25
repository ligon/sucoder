"""Configuration loading for the sucoder."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional

import yaml


class AgentType(Enum):
    """Known agent CLI types for profile-based flag selection."""

    CLAUDE = auto()
    CODEX = auto()
    GEMINI = auto()
    AIDER = auto()
    OPENCODE = auto()
    GOOSE = auto()
    KIMI = auto()
    UNKNOWN = auto()


@dataclass
class BranchPrefixes:
    human: str = ""
    agent: str = "coder"

    def __post_init__(self) -> None:
        if not self.human:
            self.human = os.environ.get("USER", "")


@dataclass
class SlurmConfig:
    """SLURM job parameters for compute-node targets.

    When present on a :class:`RemoteConfig`, the session allocates a
    compute node via ``salloc --no-shell`` and tunnels through the
    login node to reach it.
    """

    partition: str
    account: str
    time: str = "02:00:00"
    qos: Optional[str] = None
    cpus_per_task: Optional[int] = None  # request a fractional slice on
                                         # shared partitions (e.g. savio4_htc)
    mem: Optional[str] = None            # e.g. "16G"; required on shared
                                         # partitions where the default is tiny
    local_disk: Optional[str] = None     # e.g. "/local" — bypass shared FS
    confined: bool = False               # shared partitions: launch the agent
                                         # via `sbatch` so it runs inside the
                                         # job cgroup, confined to the reserved
                                         # cores instead of the whole node


@dataclass
class RemoteConfig:
    """SSH connection details for running agent sessions on a remote host.

    Also used as the definition of a named *target* in the config
    file (``targets:`` section).  When a target carries a
    ``mirror_root`` it overrides the global one.
    """

    gateway: str                                    # Jump host, e.g. "brc.berkeley.edu"
    transfer_host: str                              # DTN for git transport
    remote_user: Optional[str] = None               # SSH username on the remote host
    mirror_root: Path = field(default_factory=lambda: Path("~/mirrors"))
    ssh_options: Dict[str, str] = field(default_factory=dict)
    x11: Optional[bool] = None                      # Forward X11 on interactive hops
                                                    # (collaborate/attach) so remote
                                                    # sessions can open X windows.
                                                    # None = unset: defaults to ON
                                                    # when the local session has a
                                                    # DISPLAY (see cli._resolve_x11)
    control_persist: str = "7d"                     # ControlMaster idle lifetime (ssh time fmt)
    keepalive_interval: int = 30                    # ServerAliveInterval (seconds)
    keepalive_count_max: int = 120                  # ServerAliveCountMax (probes before teardown)
    slurm: Optional[SlurmConfig] = None             # Compute-node allocation params
    system_prompt_extra: Optional[Path] = None      # Target-specific prompt snippet
    cert_file: Optional[Path] = None                # Local SSH cert (private key) presented to the gateway

    def ssh_control_kwargs(self) -> Dict[str, Any]:
        """Shared SSH kwargs threaded to SshControl and
        ``sshconfig.render_block``.

        Keeping these in one place means the persistence/keepalive knobs
        (``control_persist`` + the two ``keepalive_*`` values) and the
        optional gateway ``cert_file`` are applied identically everywhere a
        connection is built, so a target's config is honoured at every hop
        instead of silently falling back to the dataclass defaults at some
        call sites.  ``cert_file`` is a string path (or ``None``); consumers
        that only care about the gateway hop use it, others ignore it.
        """
        return {
            "control_persist": self.control_persist,
            "keepalive_interval": self.keepalive_interval,
            "keepalive_count_max": self.keepalive_count_max,
            "cert_file": str(self.cert_file) if self.cert_file else None,
            "user": self.remote_user,
        }


@dataclass
class NvmConfig:
    """Configuration for wrapping agent launches with nvm."""

    version: str
    dir: Optional[Path] = None


@dataclass
class McpServerConfig:
    """Definition of a single MCP server."""

    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CredentialConfig:
    """Reference to a secret stored outside SuCoder configuration."""

    pass_entry: str


@dataclass(frozen=True)
class ProviderConfig:
    """Provider metadata used to adapt a model name to a harness launch."""

    credential: str
    protocol: str
    base_url: str
    env_var: str


@dataclass
class AgentLauncher:
    """Configuration for launching the agent process."""

    command: List[str] = field(default_factory=lambda: ["claude"])
    model: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    nvm: Optional[NvmConfig] = None
    accepts_inline_prompt: Optional[bool] = None
    needs_yolo: Optional[bool] = None
    launch_mode: Optional[Literal["subprocess", "exec"]] = None
    writable_dirs: List[Path] = field(default_factory=list)
    workdir: Optional[Path] = None
    default_flags: List[str] = field(default_factory=list)
    flags: "AgentFlagTemplates" = field(default_factory=lambda: AgentFlagTemplates())


@dataclass
class AgentFlagTemplates:
    """Templates for translating generic intents into agent-specific flags.

    Defaults are None; actual values come from AGENT_PROFILES based on
    detected agent type, or from user config overrides.
    """

    yolo: Optional[str] = None
    writable_dir: Optional[str] = None
    workdir: Optional[str] = None
    default_flag: Optional[str] = "{flag}"
    skills: Optional[str] = None
    system_prompt: Optional[str] = None
    mcp_config: Optional[str] = None
    model: Optional[str] = None
    system_prompt_file: Optional[str] = None


@dataclass(frozen=True)
class HarnessCapabilities:
    """Concise capability labels shown by ``sucoder list harnesses``."""

    shell: str
    files: str
    skills: str
    mcp: str
    subagents: str
    providers: str
    approval: str


# Agent profiles provide CLI-specific default flag templates.
# Precedence (highest to lowest): per-mirror config > global config > profile > UNKNOWN
AGENT_PROFILES: Dict[AgentType, AgentFlagTemplates] = {
    AgentType.UNKNOWN: AgentFlagTemplates(
        # Fallback for unrecognized CLIs - user should configure explicitly
        yolo=None,
        writable_dir=None,
        system_prompt=None,
        skills=None,
        mcp_config=None,
        model=None,
        system_prompt_file=None,
    ),
    AgentType.CLAUDE: AgentFlagTemplates(
        yolo="--dangerously-skip-permissions",
        writable_dir="--add-dir {path}",
        system_prompt="--system-prompt",  # Flag only; content added as separate arg
        skills=None,  # Claude doesn't have a direct skills flag
        mcp_config="--mcp-config {path}",
        model="--model {model}",
    ),
    AgentType.CODEX: AgentFlagTemplates(
        yolo="--sandbox danger-full-access --ask-for-approval never",
        writable_dir=None,  # codex uses sandbox permissions instead
        system_prompt=None,  # codex uses trailing text
        skills=None,
        mcp_config=None,
        model="--model {model}",
    ),
    AgentType.GEMINI: AgentFlagTemplates(
        yolo="--yolo",
        writable_dir="--include-directories {path}",
        system_prompt="--prompt-interactive",  # stays interactive after prompt
        skills=None,
        mcp_config=None,
        model="--model {model}",
    ),
    AgentType.AIDER: AgentFlagTemplates(
        yolo="--yes-always",
        writable_dir=None,
        system_prompt=None,
        system_prompt_file="--read {path}",
        skills=None,
        mcp_config=None,
        model="--model {model}",
    ),
    AgentType.OPENCODE: AgentFlagTemplates(
        yolo="--auto",
        writable_dir=None,
        system_prompt="--prompt",
        skills=None,  # OpenCode discovers .agents/skills natively
        mcp_config=None,
        model="--model {model}",
    ),
    AgentType.GOOSE: AgentFlagTemplates(
        yolo=None,
        writable_dir=None,
        # ``goose run --interactive --text`` processes the injected context,
        # then leaves the user in an interactive session.
        system_prompt="--text",
        skills=None,  # Goose discovers Agent Skills natively
        mcp_config=None,
        model="--model {model}",
    ),
    AgentType.KIMI: AgentFlagTemplates(
        yolo="--auto",
        writable_dir="--add-dir {path}",
        system_prompt=None,
        # Kimi custom-agent files preserve the native agent when their body
        # includes ${base_prompt}; MirrorManager supplies that wrapper.
        system_prompt_file="--agent-file {path}",
        skills=None,  # Kimi discovers Agent Skills natively
        mcp_config=None,  # Kimi uses its native configuration
        model="--model {model}",
    ),
}


# These labels describe native harness facilities, not model quality.  In
# particular, Aider may suggest shell commands but does not expose a shell tool
# in its model loop.
HARNESS_CAPABILITIES: Dict[AgentType, HarnessCapabilities] = {
    AgentType.CLAUDE: HarnessCapabilities(
        "yes", "yes", "yes", "yes", "yes", "Claude", "bypass",
    ),
    AgentType.CODEX: HarnessCapabilities(
        "yes", "yes", "yes", "yes", "yes", "custom", "policy",
    ),
    AgentType.GEMINI: HarnessCapabilities(
        "yes", "yes", "yes", "yes", "yes", "Gemini", "yolo",
    ),
    AgentType.AIDER: HarnessCapabilities(
        "suggest", "edit", "no", "no", "no", "multi", "explicit",
    ),
    AgentType.OPENCODE: HarnessCapabilities(
        "yes", "yes", "yes", "yes", "yes", "multi", "auto",
    ),
    AgentType.GOOSE: HarnessCapabilities(
        "yes", "yes", "yes", "yes", "yes", "multi", "policy",
    ),
    AgentType.KIMI: HarnessCapabilities(
        "yes", "yes", "yes", "yes", "yes", "multi", "auto",
    ),
    AgentType.UNKNOWN: HarnessCapabilities(
        "?", "?", "?", "?", "?", "?", "?",
    ),
}

# Default launch modes per agent type.
# "subprocess" uses subprocess.run() - works for agents that don't require a TTY.
# "exec" uses os.execvp() - replaces the process, preserving TTY for interactive agents.
DEFAULT_LAUNCH_MODES: Dict[AgentType, Literal["subprocess", "exec"]] = {
    AgentType.CLAUDE: "subprocess",   # Works fine with subprocess
    AgentType.CODEX: "subprocess",    # Works fine with subprocess
    AgentType.GEMINI: "exec",         # Needs TTY passthrough
    AgentType.AIDER: "subprocess",
    AgentType.OPENCODE: "subprocess",
    AgentType.GOOSE: "subprocess",
    AgentType.KIMI: "subprocess",
    AgentType.UNKNOWN: "subprocess",  # Safe default
}


@dataclass
class MirrorSettings:
    """Configuration for a single mirror repository."""

    name: str
    canonical_repo: Path
    mirror_name: str
    branch_prefixes: BranchPrefixes
    # None means "auto-detect from the canonical repo" (origin/HEAD, then a
    # local main, then master, then the current checkout) — see
    # MirrorManager._resolve_base_branch.
    default_base_branch: Optional[str] = None
    task_branch_prefix: str = "task"
    agent_launcher: AgentLauncher = field(default_factory=AgentLauncher)
    skills: List[Path] = field(default_factory=list)
    mcp_servers: Dict[str, McpServerConfig] = field(default_factory=dict)
    remote: Optional[RemoteConfig] = None

    @property
    def mirror_dirname(self) -> str:
        return self.mirror_name

    @property
    def is_remote(self) -> bool:
        return self.remote is not None


@dataclass(frozen=True)
class AuditConfig:
    """Controls automatic post-session audit invocation.

    The compliance audit subsystem (skills + code) can run on demand
    via ``sucoder audit`` but can also fire automatically after each
    agent session.  Auto-trigger is opt-in: the default leaves
    behaviour exactly as before (no extra LLM calls at session
    teardown, no new file writes).
    """

    auto_after_session: bool = False
    """If True, run skills+code audits after every agent session.

    Reports are saved under ``<log_dir>/audits/<mirror>-<kind>-<timestamp>.log``.
    A one-line summary is logged to the console.  Audit failures are
    logged at WARNING level but never block session teardown.
    """

    scope: str = "all"
    """Which audits to run: ``"skills"``, ``"code"``, or ``"all"``."""


@dataclass
class Config:
    human_user: str
    agent_user: str = "coder"
    agent_group: str = "coder"
    mirror_root: Path = field(default_factory=Path)
    skills: List[Path] = field(default_factory=list)
    mcp_servers: Dict[str, McpServerConfig] = field(default_factory=dict)
    credentials: Dict[str, CredentialConfig] = field(default_factory=dict)
    providers: Dict[str, ProviderConfig] = field(
        default_factory=lambda: dict(BUILTIN_PROVIDERS)
    )
    system_prompt: Optional[Path] = None
    log_dir: Optional[Path] = None
    agent_launcher: Optional[AgentLauncher] = None  # Global defaults for all mirrors
    mirrors: Mapping[str, MirrorSettings] = field(default_factory=dict)
    targets: Dict[str, RemoteConfig] = field(default_factory=dict)
    audit: AuditConfig = field(default_factory=AuditConfig)

    def resolve_target(self, target_name: Optional[str]) -> Optional[RemoteConfig]:
        """Look up a named target, returning ``None`` for local execution."""
        if target_name is None:
            return None
        if target_name not in self.targets:
            raise ConfigError(
                f"Unknown target `{target_name}`. "
                f"Available targets: {', '.join(sorted(self.targets)) or '(none)'}."
            )
        return self.targets[target_name]

    @property
    def mirrors_dir(self) -> Path:
        return self.mirror_root


BUILTIN_PROVIDERS: Dict[str, ProviderConfig] = {
    "openrouter": ProviderConfig(
        credential="openrouter",
        protocol="openai",
        base_url="https://openrouter.ai/api/v1",
        env_var="OPENROUTER_API_KEY",
    ),
}


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or validated."""


class ConfigWarning(UserWarning):
    """Non-fatal configuration problem (e.g. an ignored/misplaced key)."""


# Keys accepted inside a ``slurm:`` block.  Anything else is ignored
# with a :class:`ConfigWarning` -- most often a target-level option
# mistakenly nested under ``slurm:`` (e.g. ``system_prompt_extra``),
# which the parser silently drops, so the option appears to do nothing.
_VALID_SLURM_KEYS = frozenset({
    "partition", "account", "time", "qos",
    "cpus_per_task", "mem", "local_disk", "confined",
})
# Target-level options commonly misplaced under ``slurm:``; warned about
# with a tailored "move it up a level" hint.
_TARGET_LEVEL_KEYS = frozenset({
    "gateway", "transfer_host", "mirror_root", "ssh_options",
    "control_persist", "keepalive_interval", "keepalive_count_max",
    "system_prompt_extra", "cert_file", "x11",
})


def _expand_path(raw: Optional[str]) -> Optional[Path]:
    if raw is None:
        return None
    return Path(raw).expanduser().resolve()


def load_config(path: Path) -> Config:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a mapping.")

    return _build_config(data, path=path)


def _detect_git_toplevel() -> Path:
    """Return the git repository root for the current working directory."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise ConfigError(
            "Not inside a git repository. Either cd into a git repo or "
            "create a config file at ~/.sucoder/config.yaml."
        ) from exc
    return Path(result.stdout.strip())


KNOWN_AGENTS = ["claude", "codex", "gemini", "aider", "opencode", "goose", "kimi"]
AGENT_PREFERENCE_FILE = Path("~/.sucoder/agent")


def detect_agent_command() -> List[str]:
    """Resolve which agent CLI to use via a four-level cascade.

    1. ``$SUCODER_AGENT`` environment variable
    2. ``~/.sucoder/agent`` preference file (single word)
    3. Auto-detect from PATH (scan for known agents)
    4. Interactive prompt when multiple agents are found

    Raises :class:`ConfigError` if no agent can be resolved.
    """
    # 1. Environment variable
    env_agent = os.environ.get("SUCODER_AGENT", "").strip()
    if env_agent:
        if shutil.which(env_agent):
            return _default_agent_command(env_agent)
        raise ConfigError(
            f"$SUCODER_AGENT is set to {env_agent!r} but it was not found on PATH."
        )

    # 2. Preference file
    pref_path = AGENT_PREFERENCE_FILE.expanduser()
    if pref_path.is_file():
        saved = pref_path.read_text(encoding="utf-8").strip()
        if saved:
            if shutil.which(saved):
                return _default_agent_command(saved)
            raise ConfigError(
                f"Agent {saved!r} (from {pref_path}) was not found on PATH."
            )

    # 3. Auto-detect from PATH
    found = [name for name in KNOWN_AGENTS if shutil.which(name)]

    if len(found) == 1:
        return _default_agent_command(found[0])

    if len(found) == 0:
        raise ConfigError(
            "No supported agent CLI found on PATH. "
            f"Install one of: {', '.join(KNOWN_AGENTS)}, "
            "or set $SUCODER_AGENT."
        )

    # 4. Multiple found — interactive prompt
    return _prompt_agent_choice(found)


def _default_agent_command(name: str) -> List[str]:
    """Return the interactive command for a known harness executable."""
    if name == "goose":
        return ["goose", "run", "--interactive"]
    return [name]


def _prompt_agent_choice(agents: List[str]) -> List[str]:
    """Present a numbered menu and save the user's choice."""
    print("Multiple agent CLIs found on PATH:")
    for i, name in enumerate(agents, 1):
        print(f"  {i}. {name}")

    while True:
        try:
            raw = input(f"Select agent [1-{len(agents)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise ConfigError("No agent selected.") from None
        if raw.isdigit() and 1 <= int(raw) <= len(agents):
            choice = agents[int(raw) - 1]
            break
        print(f"Please enter a number between 1 and {len(agents)}.")

    # Save for next time
    pref_path = AGENT_PREFERENCE_FILE.expanduser()
    pref_path.parent.mkdir(parents=True, exist_ok=True)
    pref_path.write_text(choice + "\n", encoding="utf-8")

    return _default_agent_command(choice)


def build_default_config() -> Config:
    """Build a zero-config Config from the environment and git state.

    Derives all required fields from ``$USER`` and the git repository
    root of the current working directory.

    Raises :class:`ConfigError` if ``$USER`` is unset or the current
    directory is not inside a git repository.
    """
    user = os.environ.get("USER")
    if not user:
        raise ConfigError(
            "$USER is not set. Export USER or create a config file "
            "at ~/.sucoder/config.yaml."
        )

    agent_command = detect_agent_command()

    git_toplevel = _detect_git_toplevel()
    mirror_name = git_toplevel.name
    mirror_root = Path("/var/tmp/coder-mirrors")

    prefixes = BranchPrefixes(human=user, agent="coder")
    launcher = AgentLauncher(command=agent_command)
    mirror = MirrorSettings(
        name=mirror_name,
        canonical_repo=git_toplevel,
        mirror_name=mirror_name,
        branch_prefixes=prefixes,
        agent_launcher=launcher,
    )

    return Config(
        human_user=user,
        agent_user="coder",
        agent_group="coder",
        mirror_root=mirror_root,
        mirrors={mirror_name: mirror},
    )


def _build_config(data: Dict[str, Any], *, path: Path) -> Config:
    human_user = data.get("human_user")
    if not human_user:
        raise ConfigError(f"`human_user` must be set in {path}")

    mirror_root_raw = data.get("mirror_root")
    if not mirror_root_raw:
        raise ConfigError(f"`mirror_root` must be set in {path}")

    log_dir = _expand_path(data.get("log_dir")) if data.get("log_dir") else None
    system_prompt_raw = data.get("system_prompt")
    system_prompt = _expand_path(system_prompt_raw) if system_prompt_raw else None
    if system_prompt and not system_prompt.exists():
        raise ConfigError(f"Configured system_prompt file not found: {system_prompt}")

    global_skills = _parse_skills(data.get("skills"))
    global_mcp_servers = _parse_mcp_servers(data.get("mcp_servers"))
    credentials = _parse_credentials(data.get("credentials"))
    providers = _parse_providers(data.get("providers"))

    # Parse global agent_launcher defaults (applies to all mirrors unless overridden)
    global_agent_launcher = None
    if data.get("agent_launcher") is not None:
        global_agent_launcher = _parse_agent_launcher(data.get("agent_launcher"))

    targets = _parse_targets(data.get("targets"))
    audit = _parse_audit_config(data.get("audit"), path=path)
    mirrors = _parse_mirrors(
        data.get("mirrors"), global_skills=global_skills,
        global_mcp_servers=global_mcp_servers, path=path,
    )

    mirror_root = _expand_path(mirror_root_raw)
    if mirror_root is None:
        raise ConfigError(f"Failed to resolve mirror_root path: {mirror_root_raw!r}")

    return Config(
        human_user=human_user,
        agent_user=data.get("agent_user", "coder"),
        agent_group=data.get("agent_group", data.get("agent_user", "coder")),
        mirror_root=mirror_root,
        skills=global_skills,
        mcp_servers=global_mcp_servers,
        credentials=credentials,
        providers=providers,
        system_prompt=system_prompt,
        log_dir=log_dir,
        agent_launcher=global_agent_launcher,
        mirrors=mirrors,
        targets=targets,
        audit=audit,
    )


_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROVIDER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_PROVIDER_PROTOCOLS = frozenset({"openai", "anthropic", "kimi"})


def _parse_credentials(raw: Any) -> Dict[str, CredentialConfig]:
    """Parse named references to entries in the human user's password store."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("`credentials` must be a mapping of names to settings.")

    credentials: Dict[str, CredentialConfig] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not _PROVIDER_NAME_RE.fullmatch(name):
            raise ConfigError(
                f"Credential name {name!r} must contain only lowercase letters, "
                "digits, '.', '_', or '-'."
            )
        if not isinstance(value, dict):
            raise ConfigError(f"Credential `{name}` must be a mapping.")
        unknown = set(value) - {"pass"}
        if unknown:
            raise ConfigError(
                f"Credential `{name}` has unsupported settings: "
                f"{', '.join(sorted(str(key) for key in unknown))}."
            )
        entry = value.get("pass")
        if (
            not isinstance(entry, str)
            or not entry.strip()
            or entry.startswith("-")
            or "\n" in entry
            or "\r" in entry
        ):
            raise ConfigError(
                f"`credentials.{name}.pass` must be a non-empty pass entry name."
            )
        credentials[name] = CredentialConfig(pass_entry=entry)
    return credentials


def _parse_providers(raw: Any) -> Dict[str, ProviderConfig]:
    """Merge custom provider metadata over SuCoder's built-in providers."""
    providers = dict(BUILTIN_PROVIDERS)
    if raw is None:
        return providers
    if not isinstance(raw, dict):
        raise ConfigError("`providers` must be a mapping of names to settings.")

    for name, value in raw.items():
        if not isinstance(name, str) or not _PROVIDER_NAME_RE.fullmatch(name):
            raise ConfigError(
                f"Provider name {name!r} must contain only lowercase letters, "
                "digits, '.', '_', or '-'."
            )
        if not isinstance(value, dict):
            raise ConfigError(f"Provider `{name}` must be a mapping.")
        unknown = set(value) - {"credential", "protocol", "base_url", "env_var"}
        if unknown:
            raise ConfigError(
                f"Provider `{name}` has unsupported settings: "
                f"{', '.join(sorted(str(key) for key in unknown))}."
            )
        existing = providers.get(name)
        credential = value.get("credential", existing.credential if existing else name)
        protocol = value.get("protocol", existing.protocol if existing else None)
        base_url = value.get("base_url", existing.base_url if existing else None)
        env_var = value.get("env_var", existing.env_var if existing else None)
        if not isinstance(credential, str) or not credential.strip():
            raise ConfigError(f"`providers.{name}.credential` must be a non-empty string.")
        if protocol not in _PROVIDER_PROTOCOLS:
            raise ConfigError(
                f"`providers.{name}.protocol` must be one of "
                f"{', '.join(sorted(_PROVIDER_PROTOCOLS))}."
            )
        if not isinstance(base_url, str) or not base_url.strip():
            raise ConfigError(f"`providers.{name}.base_url` must be a non-empty string.")
        if not isinstance(env_var, str) or not _ENVIRONMENT_NAME_RE.fullmatch(env_var):
            raise ConfigError(
                f"`providers.{name}.env_var` must be a valid environment variable name."
            )
        providers[name] = ProviderConfig(
            credential=credential.strip(),
            protocol=protocol,
            base_url=base_url.strip(),
            env_var=env_var,
        )
    return providers


def _parse_audit_config(raw: Any, *, path: Path) -> AuditConfig:
    """Parse the ``audit:`` block from a sucoder config file.

    Missing or empty → defaults (auto-trigger off, scope ``"all"``).
    """
    if raw is None:
        return AuditConfig()
    if not isinstance(raw, dict):
        raise ConfigError(
            f"`audit` must be a mapping in {path}, got {type(raw).__name__}."
        )

    auto_raw = raw.get("auto_after_session", False)
    if not isinstance(auto_raw, bool):
        raise ConfigError(
            f"`audit.auto_after_session` must be a boolean in {path}, "
            f"got {type(auto_raw).__name__}."
        )

    scope_raw = raw.get("scope", "all")
    if not isinstance(scope_raw, str):
        raise ConfigError(
            f"`audit.scope` must be a string in {path}, "
            f"got {type(scope_raw).__name__}."
        )
    if scope_raw not in ("skills", "code", "all"):
        raise ConfigError(
            f"`audit.scope` must be one of 'skills', 'code', or 'all' "
            f"in {path}, got {scope_raw!r}."
        )

    return AuditConfig(auto_after_session=auto_raw, scope=scope_raw)


def _parse_mirrors(
    raw: Any,
    *,
    global_skills: List[Path],
    global_mcp_servers: Dict[str, McpServerConfig],
    path: Path,
) -> Dict[str, MirrorSettings]:
    if raw is None:
        return {}  # No mirrors configured; zero-config detection will add them.
    if isinstance(raw, list):
        raise ConfigError("`mirrors` must be a mapping of names to settings.")
    if not isinstance(raw, dict):
        raise ConfigError("`mirrors` must be a mapping.")

    mirrors: Dict[str, MirrorSettings] = {}
    for name, value in raw.items():
        if not isinstance(value, dict):
            raise ConfigError(f"Mirror `{name}` must be a mapping.")

        canonical_raw = value.get("canonical_repo")
        if not canonical_raw:
            raise ConfigError(f"Mirror `{name}` requires `canonical_repo`.")

        mirror_name_raw = value.get("mirror_name", name)
        if not isinstance(mirror_name_raw, str):
            raise ConfigError(f"Mirror `{name}` has invalid `mirror_name`; expected string.")
        mirror_name = mirror_name_raw

        prefix_data = value.get("branch_prefixes", {}) or {}
        if not isinstance(prefix_data, dict):
            raise ConfigError(f"`branch_prefixes` for mirror `{name}` must be a mapping.")

        defaults = BranchPrefixes()
        prefixes = BranchPrefixes(
            human=prefix_data.get("human", defaults.human),
            agent=prefix_data.get("agent", defaults.agent),
        )

        launcher = _parse_agent_launcher(value.get("agent_launcher"))
        remote = _parse_remote_config(value.get("remote"))

        skills_raw_present = "skills" in value
        skills = _parse_skills(value.get("skills")) if skills_raw_present else list(global_skills)

        mcp_raw_present = "mcp_servers" in value
        mcp_servers = (
            _parse_mcp_servers(value.get("mcp_servers"))
            if mcp_raw_present
            else dict(global_mcp_servers)
        )

        canonical_repo = _expand_path(canonical_raw)
        if canonical_repo is None:
            raise ConfigError(
                f"Mirror `{name}` canonical repo path `{canonical_raw}` could not be resolved."
            )

        mirrors[name] = MirrorSettings(
            name=name,
            canonical_repo=canonical_repo,
            mirror_name=mirror_name,
            branch_prefixes=prefixes,
            default_base_branch=value.get("default_base_branch"),
            task_branch_prefix=value.get("task_branch_prefix", "task"),
            agent_launcher=launcher,
            skills=skills,
            mcp_servers=mcp_servers,
            remote=remote,
        )

    return mirrors


def _parse_agent_launcher(raw: Any) -> AgentLauncher:
    if raw is None:
        return AgentLauncher()

    if not isinstance(raw, dict):
        raise ConfigError("`agent_launcher` must be a mapping when provided.")

    command_raw = raw.get("command", ["claude"])
    if isinstance(command_raw, str):
        command = [command_raw]
    elif isinstance(command_raw, list) and all(isinstance(item, str) for item in command_raw):
        command = command_raw or ["claude"]
    else:
        raise ConfigError("`agent_launcher.command` must be a string or list of strings.")

    env_raw = raw.get("env", {}) or {}
    if not isinstance(env_raw, dict) or any(
        not isinstance(k, str)
        or not _ENVIRONMENT_NAME_RE.fullmatch(k)
        or not isinstance(v, str)
        for k, v in env_raw.items()
    ):
        raise ConfigError("`agent_launcher.env` must be a mapping of string keys to string values.")

    model = raw.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ConfigError("`agent_launcher.model` must be a non-empty string when provided.")

    nvm_settings = _parse_nvm_settings(raw.get("nvm"))

    accepts_inline_prompt = raw.get("accepts_inline_prompt")
    if accepts_inline_prompt is not None and not isinstance(accepts_inline_prompt, bool):
        raise ConfigError("`agent_launcher.accepts_inline_prompt` must be a boolean when provided.")

    needs_yolo = raw.get("needs_yolo")
    if needs_yolo is not None and not isinstance(needs_yolo, bool):
        raise ConfigError("`agent_launcher.needs_yolo` must be a boolean when provided.")

    launch_mode = raw.get("launch_mode")
    if launch_mode is not None and launch_mode not in ("subprocess", "exec"):
        raise ConfigError(
            f"`agent_launcher.launch_mode` must be 'subprocess' or 'exec', got {launch_mode!r}."
        )

    writable_dirs_raw = raw.get("writable_dirs", [])
    if writable_dirs_raw is None:
        writable_dirs_raw = []
    if not isinstance(writable_dirs_raw, list) or any(not isinstance(entry, str) for entry in writable_dirs_raw):
        raise ConfigError("`agent_launcher.writable_dirs` must be a list of path strings when provided.")
    writable_dirs = [
        resolved
        for resolved in (_expand_path(entry) for entry in writable_dirs_raw)
        if resolved is not None
    ]

    workdir_raw = raw.get("workdir")
    workdir = None
    if workdir_raw is not None:
        if not isinstance(workdir_raw, str):
            raise ConfigError("`agent_launcher.workdir` must be a path string when provided.")
        workdir = _expand_path(workdir_raw)

    default_flags_raw = raw.get("default_flags", [])
    if default_flags_raw is None:
        default_flags_raw = []
    if not isinstance(default_flags_raw, list) or any(not isinstance(flag, str) for flag in default_flags_raw):
        raise ConfigError("`agent_launcher.default_flags` must be a list of strings when provided.")

    flag_templates = _parse_flag_templates(raw.get("flags"))

    return AgentLauncher(
        command=command,
        model=model.strip() if model is not None else None,
        env=dict(env_raw),
        nvm=nvm_settings,
        accepts_inline_prompt=accepts_inline_prompt,
        needs_yolo=needs_yolo,
        launch_mode=launch_mode,
        writable_dirs=writable_dirs,
        workdir=workdir,
        default_flags=default_flags_raw,
        flags=flag_templates,
    )


def _parse_nvm_settings(raw: Any) -> Optional[NvmConfig]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("`agent_launcher.nvm` must be a mapping when provided.")

    version = raw.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ConfigError("`agent_launcher.nvm.version` must be a non-empty string.")

    dir_raw = raw.get("dir")
    dir_path: Optional[Path] = None
    if dir_raw is not None:
        if not isinstance(dir_raw, str):
            raise ConfigError("`agent_launcher.nvm.dir` must be a path string when provided.")
        dir_path = _expand_path(dir_raw)
        if dir_path is None:
            raise ConfigError("Failed to resolve `agent_launcher.nvm.dir`.")

    return NvmConfig(version=version.strip(), dir=dir_path)


def _parse_flag_templates(raw: Any) -> AgentFlagTemplates:
    if raw is None:
        return AgentFlagTemplates()
    if not isinstance(raw, dict):
        raise ConfigError("`agent_launcher.flags` must be a mapping when provided.")

    def _template(key: str) -> Optional[str]:
        value = raw.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ConfigError(f"`agent_launcher.flags.{key}` must be a string when provided.")
        return value

    return AgentFlagTemplates(
        yolo=_template("yolo"),
        writable_dir=_template("writable_dir"),
        workdir=_template("workdir"),
        default_flag=_template("default_flag"),
        skills=_template("skills"),
        system_prompt=_template("system_prompt"),
        mcp_config=_template("mcp_config"),
        model=_template("model"),
        system_prompt_file=_template("system_prompt_file"),
    )


def _parse_positive_int(raw: Any, key: str, *, default: int) -> int:
    """Parse an optional positive-integer config value.

    ``None`` (key absent) returns *default*.  Bools are rejected even
    though ``bool`` is an ``int`` subclass, and non-positive values are
    errors.  Used for the SSH keepalive knobs (``keepalive_interval`` /
    ``keepalive_count_max``).
    """
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ConfigError(f"`{key}` must be a positive integer when provided.")
    if raw <= 0:
        raise ConfigError(f"`{key}` must be a positive integer when provided.")
    return raw


def _parse_targets(raw: Any) -> Dict[str, RemoteConfig]:
    """Parse the top-level ``targets:`` mapping."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("`targets` must be a mapping of names to remote configurations.")
    targets: Dict[str, RemoteConfig] = {}
    for name, value in raw.items():
        parsed = _parse_remote_config(value)
        if parsed is None:
            raise ConfigError(f"Target `{name}` must be a mapping with at least `gateway` and `transfer_host`.")
        targets[name] = parsed
    return targets


def _parse_remote_config(raw: Any) -> Optional[RemoteConfig]:
    """Parse optional ``remote:`` block from a mirror config entry."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("`remote` must be a mapping when provided.")

    gateway = raw.get("gateway")
    if not gateway or not isinstance(gateway, str):
        raise ConfigError("`remote.gateway` must be a non-empty string.")

    transfer_host = raw.get("transfer_host")
    if not transfer_host or not isinstance(transfer_host, str):
        raise ConfigError("`remote.transfer_host` must be a non-empty string.")

    mirror_root_raw = raw.get("mirror_root", "~/mirrors")
    mirror_root = Path(mirror_root_raw)  # Keep unexpanded; expanded on remote

    remote_user = raw.get("remote_user")
    if remote_user is not None and not isinstance(remote_user, str):
        raise ConfigError("`remote.remote_user` must be a string when provided.")

    ssh_options = raw.get("ssh_options", {})
    if not isinstance(ssh_options, dict):
        raise ConfigError("`remote.ssh_options` must be a mapping when provided.")

    x11 = raw.get("x11")
    if x11 is not None and not isinstance(x11, bool):
        raise ConfigError("`remote.x11` must be a boolean when provided.")

    control_persist = raw.get("control_persist", "7d")
    if not isinstance(control_persist, str):
        raise ConfigError("`remote.control_persist` must be a string (e.g. '7d', '12h', '1d').")

    keepalive_interval = _parse_positive_int(
        raw.get("keepalive_interval"), "remote.keepalive_interval", default=30,
    )
    keepalive_count_max = _parse_positive_int(
        raw.get("keepalive_count_max"), "remote.keepalive_count_max", default=120,
    )

    slurm = _parse_slurm_config(raw.get("slurm"))

    prompt_extra_raw = raw.get("system_prompt_extra")
    prompt_extra = _expand_path(prompt_extra_raw) if prompt_extra_raw else None
    # Don't error if the file is missing — targets may be defined in a
    # shared config but only some machines carry the prompt snippet.
    # The mirror module logs a warning at injection time instead.

    cert_file_raw = raw.get("cert_file")
    if cert_file_raw is not None and not isinstance(cert_file_raw, str):
        raise ConfigError("`remote.cert_file` must be a string path when provided.")
    # Expanded locally: the cert lives on the operator's machine.  Not an
    # error if absent — the operator may not have minted one yet; ssh then
    # falls back to the interactive prompt (and `tunnel doctor` flags it).
    cert_file = _expand_path(cert_file_raw)

    return RemoteConfig(
        gateway=gateway,
        transfer_host=transfer_host,
        remote_user=remote_user,
        mirror_root=mirror_root,
        ssh_options={str(k): str(v) for k, v in ssh_options.items()},
        x11=x11,
        control_persist=control_persist,
        keepalive_interval=keepalive_interval,
        keepalive_count_max=keepalive_count_max,
        slurm=slurm,
        system_prompt_extra=prompt_extra,
        cert_file=cert_file,
    )


def _parse_slurm_config(raw: Any) -> Optional[SlurmConfig]:
    """Parse an optional ``slurm:`` block inside a target/remote config."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("`slurm` must be a mapping when provided.")

    partition = raw.get("partition")
    if not partition or not isinstance(partition, str):
        raise ConfigError("`slurm.partition` must be a non-empty string.")

    account = raw.get("account")
    if not account or not isinstance(account, str):
        raise ConfigError("`slurm.account` must be a non-empty string.")

    time_limit = raw.get("time", "02:00:00")
    if not isinstance(time_limit, str):
        raise ConfigError("`slurm.time` must be a string (e.g. '02:00:00').")

    qos = raw.get("qos")
    if qos is not None and not isinstance(qos, str):
        raise ConfigError("`slurm.qos` must be a string when provided.")

    cpus_per_task = raw.get("cpus_per_task")
    if cpus_per_task is not None:
        # YAML may parse "4" as int already; reject bools (which are int
        # subclasses) and anything non-positive.
        if isinstance(cpus_per_task, bool) or not isinstance(cpus_per_task, int):
            raise ConfigError(
                "`slurm.cpus_per_task` must be a positive integer when provided."
            )
        if cpus_per_task <= 0:
            raise ConfigError(
                "`slurm.cpus_per_task` must be a positive integer when provided."
            )

    mem = raw.get("mem")
    if mem is not None and (not isinstance(mem, str) or not mem.strip()):
        raise ConfigError(
            "`slurm.mem` must be a non-empty string (e.g. '16G', '4000M') when provided."
        )

    local_disk = raw.get("local_disk")
    if local_disk is not None and not isinstance(local_disk, str):
        raise ConfigError("`slurm.local_disk` must be a string path (e.g. '/local').")

    confined = raw.get("confined", False)
    if not isinstance(confined, bool):
        raise ConfigError("`slurm.confined` must be a boolean when provided.")

    # Surface keys that the parser will ignore.  The common case is a
    # target-level option (notably ``system_prompt_extra``) indented one
    # level too deep, under ``slurm:`` instead of beside it -- which
    # silently changes nothing.  Warn rather than error so existing
    # configs keep loading.
    for key in sorted(set(raw) - _VALID_SLURM_KEYS):
        if key in _TARGET_LEVEL_KEYS:
            warnings.warn(
                f"`{key}` is nested under `slurm:` but is a target-level "
                f"option; it is being ignored. Move it up one level, to a "
                f"sibling of `slurm:`.",
                ConfigWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Unknown key `{key}` under `slurm:` is ignored "
                f"(valid keys: {', '.join(sorted(_VALID_SLURM_KEYS))}).",
                ConfigWarning,
                stacklevel=2,
            )

    return SlurmConfig(
        partition=partition,
        account=account,
        time=time_limit,
        qos=qos,
        cpus_per_task=cpus_per_task,
        mem=mem,
        local_disk=local_disk,
        confined=confined,
    )


def _parse_skills(raw: Any) -> List[Path]:
    if raw is None:
        return []
    if isinstance(raw, (str, Path)):
        raise ConfigError("`skills` must be a list of paths when provided.")
    if not isinstance(raw, list):
        raise ConfigError("`skills` must be a list of paths when provided.")

    skills: List[Path] = []
    for entry in raw:
        if not isinstance(entry, str):
            raise ConfigError("`skills` entries must be strings representing paths.")
        expanded = _expand_path(entry)
        if expanded is None:
            continue
        skills.append(expanded)
    return skills


def _parse_mcp_servers(raw: Any) -> Dict[str, McpServerConfig]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("`mcp_servers` must be a mapping of server names to configurations.")

    servers: Dict[str, McpServerConfig] = {}
    for name, value in raw.items():
        if not isinstance(name, str):
            raise ConfigError("`mcp_servers` keys must be strings.")
        if not isinstance(value, dict):
            raise ConfigError(f"`mcp_servers.{name}` must be a mapping.")

        command = value.get("command")
        if not command or not isinstance(command, str):
            raise ConfigError(f"`mcp_servers.{name}.command` must be a non-empty string.")

        args = value.get("args", [])
        if not isinstance(args, list) or any(not isinstance(a, str) for a in args):
            raise ConfigError(f"`mcp_servers.{name}.args` must be a list of strings.")

        env = value.get("env", {})
        if not isinstance(env, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()
        ):
            raise ConfigError(f"`mcp_servers.{name}.env` must be a mapping of strings to strings.")

        servers[name] = McpServerConfig(command=command, args=list(args), env=dict(env))
    return servers
