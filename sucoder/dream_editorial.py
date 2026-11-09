"""Native editorial inspection/materialization (implementation ledger, P5).

These are pure inline operations. The author/reviewer coordinator is responsible
for authority, source references and accepting the materialized candidate.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import re
import subprocess
import time

from . import dream, dream_run, dream_wiki
from .dream import DreamError, canonical


CONTRACT = "native-editorial-org-v1"
NATIVE = Path(__file__).with_suffix(".el")
STATES = ("TODO", "NEXT", "ACTIVE", "WAITING", "REVIEW", "DONE", "MOOT")
TERMINAL = ("DONE", "MOOT")
MAX_ACTIONS = 16


def timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", value):
        raise DreamError("Editorial timestamp must be frozen UTC YYYY-MM-DDTHH:MM:SSZ")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise DreamError("Invalid frozen editorial timestamp") from None
    return value


def action_shape(action):
    dream._object(action, ("kind", "path", "target", "before_sha256"), ("state", "note", "properties", "destination"))
    dream_wiki.editable_path(action["path"])
    dream._text(action["target"], "editorial target")
    if not isinstance(action["before_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", action["before_sha256"]):
        raise DreamError("Editorial action needs the frozen subtree SHA-256")
    if action["kind"] == "transition":
        if action.get("state") not in STATES or "destination" in action:
            raise DreamError("Invalid editorial transition state or destination")
        dream._text(action.get("note"), "timestamped transition note")
        properties = action.get("properties", {})
        if not isinstance(properties, dict):
            raise DreamError("Editorial properties must be an object")
        for key, value in properties.items():
            if (not isinstance(key, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or
                    key in ("ID", "CUSTOM_ID", "TODO", "CLOSED", "PRIORITY", "SCHEDULED", "DEADLINE") or key.startswith("ARCHIVE_")):
                raise DreamError("Editorial actions cannot replace identity/state/archive properties")
            dream._text(value, "editorial property")
    elif action["kind"] == "archive":
        if set(action) - {"kind", "path", "target", "before_sha256", "destination"}:
            raise DreamError("Archive actions only select a frozen task and destination")
        destination = dream._path(action.get("destination"))
        if not destination.startswith(".sucoder/wiki/archive/") or not destination.endswith(".org"):
            raise DreamError("Archive destination must be an Org file under .sucoder/wiki/archive/")
    else:
        raise DreamError("Unsupported editorial action")


def native(files, actions=(), *, at=None, deadline=None):
    """Return changed inline texts, moves and an opt-in rich Org inventory."""
    deadline = deadline or time.monotonic() + 30
    if not isinstance(files, dict) or len(files) > 128:
        raise DreamError("Editorial context must be a bounded file map")
    for path, text in files.items():
        dream._path(path)
        if not path.startswith(".sucoder/wiki/") or not path.endswith(".org") or not isinstance(text, str):
            raise DreamError("Editorial context contains a non-wiki/non-Org page")
    if not isinstance(actions, (tuple, list)):
        raise DreamError("Editorial actions must be an array")
    if len(actions) > MAX_ACTIONS:
        raise DreamError("Editorial action allowance exhausted", 3)
    seen = set()
    for action in actions:
        action_shape(action)
        key = action["kind"], action["path"], action["target"]
        if key in seen:
            raise DreamError("Duplicate editorial action")
        seen.add(key)
        if action["path"] not in files or action.get("destination", action["path"]) not in files:
            raise DreamError("Editorial source and destination must be fully supplied")
    if actions:
        timestamp(at)
    request = canonical(dict(files=files, actions=list(actions), timestamp=at))
    if len(request) > dream_run.MAX_PLAN_BYTES:
        raise DreamError("Native editorial input exceeds its byte allowance", 3)
    try:
        result = subprocess.run(["emacs", "--batch", "-Q", "--script", str(NATIVE)], input=request,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=dream_run._remaining(deadline),
            env=dict(os.environ, TZ="UTC"), check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise DreamError("Native editorial operations unavailable or timed out", 3) from None
    if len(result.stdout) > dream_run.MAX_PLAN_BYTES:
        raise DreamError("Native editorial result exceeds its byte allowance", 3)
    try:
        value = dream._json(result.stdout)
    except DreamError:
        raise DreamError("Native editorial interface returned an invalid result", 3) from None
    if result.returncode:
        raise DreamError("Native editorial operation rejected: " + str(value.get("error", "unknown error"))[:4096])
    dream._object(value, ("contract", "inventory", "files", "moves"))
    if value["contract"] != CONTRACT or not set(value["files"]) <= set(files) or set(value["inventory"]) != set(files):
        raise DreamError("Native editorial result differs from its supplied context")
    return value


def inspect(files, deadline=None):
    return native(files, deadline=deadline)["inventory"]
