"""Generate and splice a managed ``~/.ssh/config`` block for a target.

SuCoder keeps warm ControlMaster sockets to a cluster's *cheap* hops --
the gateway, the pinned login node, and the data-transfer node (DTN).
Those tunnels cost no compute money, so keeping them alive removes the
OTP friction from ``sucoder -T <target> collaborate`` and lets plain
``ssh``/Emacs TRAMP ride the same mux.

This module writes a sentinel-scoped block of ``Host`` aliases into the
user's ``~/.ssh/config``.  The ``ControlPath`` for each alias is computed
with :func:`sucoder.tunnel._control_socket_path`, so an alias and the
:class:`~sucoder.tunnel.SshControl` for the same hostname resolve to the
*same* socket -- one mux, shared by SuCoder, ssh, and TRAMP.

The block is fenced by per-target sentinels so re-writing one target's
block never disturbs another target's block or the user's own config.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ssh keywords that govern connection sharing.  If any of these is set by a
# Host/Match stanza that precedes our managed block AND matches our aliases,
# it shadows the alias's own value (ssh uses the first value per keyword).
_SHARE_KEYWORDS = {"controlpath", "controlmaster"}

# "Keyword value" or "Keyword=value"; ssh keywords are case-insensitive.
_DIRECTIVE_RE = re.compile(r"([A-Za-z]\w*)\s*=?\s+(.*)")

from .tunnel import _control_socket_path


def _begin_marker(target: str) -> str:
    return (
        f"# >>> sucoder managed block: {target} >>>  "
        f"(DO NOT EDIT -- run `sucoder -T {target} tunnel config`)"
    )


def _end_marker(target: str) -> str:
    return f"# <<< sucoder managed block: {target} <<<"


def alias_names(target: str) -> Dict[str, str]:
    """Return the ssh_config Host aliases for a target's three cheap hops.

    Keyed ``gw``/``ln``/``dtn`` -> ``"<target>-gw"`` etc.  This is the
    single place alias names are derived; everything else calls here.
    """
    return {
        "gw": f"{target}-gw",
        "ln": f"{target}-ln",
        "dtn": f"{target}-dtn",
    }


@dataclass
class _HopSpec:
    alias: str
    hostname: Optional[str]      # None => emitted as a placeholder comment
    proxy_jump: Optional[str]    # gateway alias, or None for the gateway itself


def render_block(
    target: str,
    gateway: str,
    transfer_host: str,
    *,
    login_node: Optional[str] = None,
    user: Optional[str] = None,
    control_persist: str = "7d",
    keepalive_interval: int = 30,
    keepalive_count_max: int = 120,
) -> str:
    """Render the managed ssh_config block for *target*.

    ``login_node`` may be ``None`` before a login node has been pinned;
    the ``-ln`` alias is then emitted with a placeholder comment instead
    of a ``HostName`` so the block stays syntactically valid.

    ``control_persist`` / ``keepalive_interval`` / ``keepalive_count_max``
    mirror the values used to build the :class:`~sucoder.tunnel.SshControl`
    for the same hops, so plain ``ssh`` and TRAMP riding these aliases get
    the same idle lifetime and keepalive tolerance as SuCoder itself.
    """
    aliases = alias_names(target)
    hops = [
        _HopSpec(aliases["gw"], gateway, None),
        _HopSpec(aliases["ln"], login_node, aliases["gw"]),
        _HopSpec(aliases["dtn"], transfer_host, aliases["gw"]),
    ]

    lines: List[str] = [_begin_marker(target)]
    for hop in hops:
        lines.append(f"Host {hop.alias}")
        if hop.hostname:
            lines.append(f"    HostName {hop.hostname}")
        else:
            lines.append(
                "    # HostName pending -- run `sucoder -T "
                f"{target} tunnel up` to pin the login node"
            )
        if user:
            lines.append(f"    User {user}")
        if hop.proxy_jump:
            lines.append(f"    ProxyJump {hop.proxy_jump}")
        lines.append(f"    ControlPath {_control_socket_path(hop.hostname or hop.alias)}")
        lines.append("    ControlMaster auto")
        lines.append(f"    ControlPersist {control_persist}")
        lines.append(f"    ServerAliveInterval {keepalive_interval}")
        lines.append(f"    ServerAliveCountMax {keepalive_count_max}")
        # Login/DTN are reached only through the gateway; their host keys
        # are stable but may be absent on a fresh client.
        if hop.proxy_jump:
            lines.append("    StrictHostKeyChecking accept-new")
        lines.append("")  # blank line between stanzas
    lines.append(_end_marker(target))
    return "\n".join(lines) + "\n"


def _config_path() -> Path:
    return Path("~/.ssh/config").expanduser()


def _strip_block(text: str, target: str) -> str:
    """Remove an existing managed block for *target* from *text*."""
    begin = re.escape(_begin_marker(target))
    end = re.escape(_end_marker(target))
    # Match the fenced block plus any trailing blank line that follows it.
    pattern = re.compile(rf"{begin}.*?{end}\n?", re.DOTALL)
    return pattern.sub("", text)


def write_block(
    block: str,
    target: str,
    *,
    path: Optional[Path] = None,
) -> Path:
    """Splice *block* into ``~/.ssh/config`` (atomic, 0600).

    Replaces an existing managed block for *target* in place; otherwise
    appends.  Content outside this target's sentinels -- the user's own
    config and other targets' blocks -- is preserved byte-for-byte.
    """
    cfg = path or _config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    try:
        cfg.parent.chmod(0o700)
    except OSError:
        pass

    existing = cfg.read_text(encoding="utf-8") if cfg.is_file() else ""
    had_block = _begin_marker(target) in existing
    stripped = _strip_block(existing, target) if had_block else existing

    # PREPEND, do not append.  ssh uses the *first* value it obtains for
    # each keyword, so host-specific blocks must precede a general
    # ``Host *`` default (ssh_config(5): "more host-specific declarations
    # should be given near the beginning of the file, and general
    # defaults at the end").  A user's ``Host *  ControlPath
    # ~/.ssh/sockets/%r@%h-%p`` would otherwise shadow our ControlPath
    # and break socket reuse -- ssh would look for a socket we never
    # created and fall through to a fresh (re-authenticating) connection.
    stripped = stripped.lstrip("\n")
    new_text = block if not stripped else block + "\n" + stripped

    tmp = cfg.with_name(cfg.name + ".sucoder.tmp")
    tmp.write_text(new_text, encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    os.replace(tmp, cfg)
    return cfg


def remove_block(target: str, *, path: Optional[Path] = None) -> bool:
    """Remove *target*'s managed block from ``~/.ssh/config``.

    Returns ``True`` if a block was present and removed.
    """
    cfg = path or _config_path()
    if not cfg.is_file():
        return False
    existing = cfg.read_text(encoding="utf-8")
    if _begin_marker(target) not in existing:
        return False
    new_text = _strip_block(existing, target)
    tmp = cfg.with_name(cfg.name + ".sucoder.tmp")
    tmp.write_text(new_text, encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    os.replace(tmp, cfg)
    return True


def block_present(target: str, *, path: Optional[Path] = None) -> bool:
    """Return ``True`` if a managed block for *target* exists on disk."""
    cfg = path or _config_path()
    if not cfg.is_file():
        return False
    return _begin_marker(target) in cfg.read_text(encoding="utf-8")


def find_shadowing_hosts(
    target: str, *, path: Optional[Path] = None,
) -> List[Tuple[str, str]]:
    """Return ``[(host_pattern, keyword)]`` for stanzas that shadow the block.

    ssh uses the *first* value it obtains for each keyword.  A ``Host``/
    ``Match`` stanza that appears **before** the managed block, matches one
    of the target's aliases, and sets a connection-sharing keyword
    (``ControlPath``/``ControlMaster``) therefore overrides the alias's own
    value -- so ssh looks for the wrong socket and re-authenticates instead
    of reusing the warm mux.  This is exactly the ``Host * ControlPath
    ~/.ssh/sockets/%r@%h-%p`` trap.

    Only content *before* the managed block is inspected (a stanza after it
    can't win the first-value race).  If the block is absent, the whole
    file is treated as preceding.  Results are de-duplicated, order
    preserved.
    """
    cfg = path or _config_path()
    if not cfg.is_file():
        return []
    text = cfg.read_text(encoding="utf-8")
    begin = _begin_marker(target)
    head = text.split(begin, 1)[0] if begin in text else text
    aliases = set(alias_names(target).values())

    results: List[Tuple[str, str]] = []
    seen = set()
    patterns: Optional[List[str]] = None   # current Host stanza patterns
    label: Optional[str] = None            # human label for the stanza
    is_match = False                       # inside a Match block?

    for raw in head.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _DIRECTIVE_RE.match(line)
        if not m:
            continue
        key = m.group(1).lower()
        val = m.group(2).strip()
        if key == "host":
            patterns = val.split()
            label = "Host " + val
            is_match = False
            continue
        if key == "match":
            patterns = None
            label = "Match " + val
            is_match = True
            continue
        if key not in _SHARE_KEYWORDS:
            continue
        # A sharing keyword: does its enclosing stanza apply to our aliases?
        applies = False
        if is_match:
            # Conservative: a Match block before us may match anything.
            applies = True
        elif patterns:
            for pat in patterns:
                if pat.startswith("!"):
                    continue
                if any(fnmatch.fnmatch(a, pat) for a in aliases):
                    applies = True
                    break
        if applies and label is not None:
            entry = (label, key)
            if entry not in seen:
                seen.add(entry)
                results.append(entry)
    return results


def managed_hostnames(
    target: str, *, path: Optional[Path] = None,
) -> Dict[str, Optional[str]]:
    """Return ``{alias: HostName-or-None}`` parsed from the managed block.

    Used to detect login-node pin drift -- if the cluster reassigned the
    login node, the alias's ``HostName`` no longer matches the pinned node
    and the socket names diverge.
    """
    cfg = path or _config_path()
    if not cfg.is_file():
        return {}
    text = cfg.read_text(encoding="utf-8")
    begin, end = _begin_marker(target), _end_marker(target)
    if begin not in text or end not in text:
        return {}
    block = text.split(begin, 1)[1].split(end, 1)[0]

    result: Dict[str, Optional[str]] = {}
    current: Optional[str] = None
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _DIRECTIVE_RE.match(line)
        if not m:
            continue
        key = m.group(1).lower()
        val = m.group(2).strip()
        if key == "host":
            current = val.split()[0]
            result[current] = None
        elif key == "hostname" and current is not None:
            result[current] = val
    return result
