"""Configured local-tier memory durability; never traverse mirror remotes."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import re
import subprocess
import tempfile

from . import dream, dream_memory, dream_run
from .config import sanitize_session_token
from .dream import DreamError
from .local_tier import work_path
from .observation_process import Budget
from .observations import WorkspaceBinding


MAX_PACK_BYTES = 64 * 1024 * 1024


class Conflict(DreamError):
    state = "conflict"


def recognize(config, root):
    """Reuse the observation contract; environment claims do not authorize writes."""
    root = Path(root).resolve()
    job = root.parent.parent.name
    if root.parent.name != "mirrors" or not re.fullmatch(r"job[0-9]+", job):
        return None
    value = WorkspaceBinding(root).inspect(config, Budget(5))
    data = value["data"]
    durable, name = data["durable_mirror"]["value"], data["mirror"]["value"]
    if value["status"] != "complete" or data["binding"] != "valid" or not durable or name not in config.mirrors:
        return None
    durable = Path(durable).resolve()
    if durable == root:
        return None
    candidates = set()
    for target in config.targets.values():
        if not target.slurm or not target.slurm.local_disk:
            continue
        token = sanitize_session_token(name)
        disk = Path(target.slurm.local_disk).expanduser().resolve()
        shared = (target.mirror_root.expanduser() / config.mirrors[name].mirror_dirname).resolve()
        if Path(work_path(str(disk), token, int(job[3:]))).resolve() == root and shared == durable:
            candidates.add((str(durable), str(disk), int(job[3:]), token))
    if len(candidates) != 1:
        return None
    path, disk, number, token = candidates.pop()
    return dict(path=path, local_disk=disk, job=number, token=token)


def validate(source, binding):
    dream._object(binding, ("path", "local_disk", "job", "token", "expected_tip"))
    if (type(binding["job"]) is not int or binding["job"] < 0 or
            not all(isinstance(binding[key], str) for key in ("path", "local_disk", "token")) or
            not Path(binding["path"]).is_absolute() or not Path(binding["local_disk"]).is_absolute() or
            sanitize_session_token(binding["token"]) != binding["token"] or
            Path(work_path(binding["local_disk"], binding["token"], binding["job"])).resolve() != source.path or
            Path(binding["path"]).resolve() == source.path or
            (binding["expected_tip"] is not None and (not isinstance(binding["expected_tip"], str) or
                not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", binding["expected_tip"])))):
        raise DreamError("Invalid frozen local-tier memory binding", 3)
    origin = source.git("config", "--get", "remote.origin.url", missing=True)
    if not origin or not Path(origin.decode().strip()).is_absolute() or Path(origin.decode().strip()).resolve() != Path(binding["path"]).resolve():
        raise DreamError("Local-tier memory origin changed; no durable publication authorized", 3)


def durable_source(source, binding):
    validate(source, binding)
    path = Path(binding["path"]).resolve()
    settings = replace(source.config.mirrors[source.name], canonical_repo=path, mirror_name=path.name)
    config = replace(source.config, mirror_root=path.parent, mirrors={source.name: settings}, targets={})
    return dream.GitSource(config, source.name, source.deadline)


def _copy_objects(source, target, commit, directory):
    """Offline object transfer, with no URL rewriting, hooks or mirror remotes."""
    if target.git("cat-file", "-e", commit + "^{commit}", missing=True) is not None:
        return
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_TERMINAL_PROMPT="0", GIT_ALLOW_PROTOCOL="", GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0")
    with tempfile.TemporaryFile() as output:
        try:
            packed = subprocess.run(["git", "--no-replace-objects", "-c", "core.hooksPath=/dev/null",
                "-c", "gc.auto=0", "-c", "maintenance.auto=false", "-C", str(source.path),
                "pack-objects", "--stdout", "--revs"], input=(commit + "\n").encode(), stdout=output,
                stderr=subprocess.PIPE, env=env, timeout=dream_run._remaining(source.deadline), check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DreamError("Local memory object transfer unavailable", 3) from exc
        output.seek(0)
        data = output.read(MAX_PACK_BYTES + 1)
    if packed.returncode or len(data) > MAX_PACK_BYTES:
        raise DreamError("Local memory object transfer failed or exceeded 64 MiB; reviewed work is retained", 3)
    with tempfile.TemporaryDirectory(prefix=".memory-objects-", dir=directory) as temporary:
        dream_run._git_write(target, Path(temporary) / "index", ["unpack-objects", "-q"], target.deadline, data)
    if target.commit(commit) != commit:
        raise DreamError("Copied memory objects do not bind the expected commit", 3)


def synchronize(source, recognized, ref, directory):
    binding = dict(recognized, expected_tip=None)
    durable = durable_source(source, binding)
    remote = dream_memory.tip(durable, ref)
    local = dream_memory.tip(source, ref)
    binding["expected_tip"] = remote
    if remote is None:
        if local is not None:
            raise Conflict("Local memory has unpublished generations; resume their original job before planning another", 3)
        return binding
    dream_memory.check_tree(durable, remote)
    _copy_objects(durable, source, remote, directory)
    if local == remote:
        return binding
    if local is not None and source.git("merge-base", "--is-ancestor", local, remote, missing=True) is None:
        raise Conflict("Local memory diverged from its durable mirror; preserve it and reconcile before planning", 3)
    with tempfile.TemporaryDirectory(prefix=".memory-sync-", dir=directory) as temporary:
        action = f"create {ref} {remote}" if local is None else f"update {ref} {remote} {local}"
        dream_run._git_write(source, Path(temporary) / "index", ["update-ref", "--no-deref", "--stdin"],
            source.deadline, ("start\n" + action + "\nprepare\ncommit\n").encode())
    return binding


def publish(plan, source, commit, directory):
    binding = plan["memory_origin"]
    durable = durable_source(source, binding)
    ref = plan["integration_ref"]
    head = durable.git("symbolic-ref", "--quiet", "HEAD", missing=True)
    if head is not None and head.decode().strip() == ref:
        raise Conflict("Durable memory branch is checked out; reviewed work is retained privately", 3)
    current = dream_memory.tip(durable, ref)
    if current == commit or (current and durable.git("merge-base", "--is-ancestor", commit, current, missing=True) is not None):
        return
    if current != binding["expected_tip"]:
        raise Conflict("Durable memory branch moved since planning; reviewed generation is retained privately; re-plan", 3)
    _copy_objects(source, durable, commit, directory)
    action = f"create {ref} {commit}" if current is None else f"update {ref} {commit} {current}"
    with tempfile.TemporaryDirectory(prefix=".memory-publish-", dir=directory) as temporary:
        try:
            dream_run._git_write(durable, Path(temporary) / "index", ["update-ref", "--no-deref", "--stdin"],
                durable.deadline, ("start\n" + action + "\nprepare\ncommit\n").encode())
        except DreamError as exc:
            if dream_memory.tip(durable, ref) != current:
                raise Conflict("Durable memory branch moved during publication; reviewed generation is retained privately", 3) from exc
            raise


def status(plan, source, commit):
    durable = durable_source(source, plan["memory_origin"])
    tip = dream_memory.tip(durable, plan["integration_ref"])
    contained = tip == commit or (tip is not None and durable.git("merge-base", "--is-ancestor", commit, tip, missing=True) is not None)
    return dict(state="published" if contained else "conflict", durable_mirror=str(durable.path),
        integration=dict(state="integrated", evidence=dict(integration_commit=tip)) if contained else
            dict(state="not_integrated", evidence=None))
