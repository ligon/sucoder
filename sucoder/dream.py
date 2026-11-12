"""Inference-free dream planning (dream-implementation-ledger.org, sections 3-5).

Evidence adapters and future role execution adapters are independent. Nothing
here selects an agent, treats a plan as accepted progress, or writes git state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time
import uuid

from .observation_process import Budget
from .observations import WorkspaceBinding


SCHEMA_VERSION = 1
MAX_CATALOG_BYTES = 256 * 1024
MAX_BLOB_BYTES = 16 * 1024 * 1024
MAX_TREE_BYTES = 256 * 1024


class DreamError(ValueError):
    def __init__(self, message, exit_code=2):
        super().__init__(message)
        self.exit_code = exit_code


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _object(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise DreamError("Invalid object fields; required: " + ", ".join(required))
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096 or any(ord(c) < 32 for c in value):
        raise DreamError(label + " must be a nonempty string without control characters")
    return value


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise DreamError("Duplicate JSON field: " + key)
        value[key] = item
    return value


def _json(data):
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_pairs)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise DreamError("Invalid JSON: " + str(exc)) from None


@dataclass(frozen=True)
class Limits:
    max_units: int = 6
    max_source_bytes: int = 1024 * 1024
    chunk_bytes: int = 64 * 1024
    max_role_invocations: int = 40
    max_prior_bytes: int = 64 * 1024
    max_proposal_bytes: int = 128 * 1024
    max_consolidation_bytes: int = 224 * 1024
    max_elapsed_seconds: int = 15 * 60
    max_repairs: int = 1

    @classmethod
    def parse(cls, value):
        _object(value, (), asdict(cls()))
        limits = cls(**value)
        for key, item in asdict(limits).items():
            if type(item) is not int or item < (0 if key == "max_repairs" else 1):
                raise DreamError(key + " must be a positive integer (max_repairs may be zero)")
        if limits.max_repairs > 1:
            raise DreamError("At most one repair iteration is supported")
        if limits.chunk_bytes < 4 or limits.chunk_bytes > limits.max_source_bytes:
            raise DreamError("chunk_bytes must be at least four and no greater than max_source_bytes")
        return limits


class GitSource:
    """Offline, byte-preserving git inspection; no checkout, hooks or filters.

    The dispatch helper decodes and strips stdout. These bounded binary reads
    are new because exact evidence offsets/digests must survive (ledger, 2).
    """
    def __init__(self, config, name, deadline):
        if name not in config.mirrors:
            raise DreamError("Unknown configured mirror: " + name)
        settings = config.mirrors[name]
        self.path = (config.mirror_root.expanduser() / settings.mirror_dirname).resolve()
        self.name = name
        self.config = config
        self.deadline = deadline
        self.binding = WorkspaceBinding(self.path)
        self.check_binding()

    def check_binding(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise DreamError("Planning deadline exhausted", 3)
        value = self.binding.inspect(self.config, Budget(min(5, remaining)))
        data = value["data"]
        if (value["status"] != "complete" or data["binding"] != "valid" or
                data["git_root"]["value"] != str(self.path) or data["mirror"]["value"] != self.name):
            raise DreamError("Configured mirror binding is unavailable, ambiguous, or changed: " + self.name, 3)

    def git(self, *args, limit=MAX_TREE_BYTES, missing=False):
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0",
                   GIT_ALLOW_PROTOCOL="", GIT_LITERAL_PATHSPECS="1")
        remaining = min(5, self.deadline - time.monotonic())
        if remaining <= 0:
            raise DreamError("Planning deadline exhausted", 3)
        # Spool outside the repository, avoiding unbounded in-memory pipe
        # buffers. Known blob sizes are checked before cat-file is invoked.
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            try:
                result = subprocess.run(
                    ["git", "--no-replace-objects", "-c", "core.hooksPath=/dev/null",
                     "-c", "credential.helper=", "-c", "gc.auto=0", "-c", "maintenance.auto=false",
                     "-C", str(self.path), *args], env=env, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=errors, timeout=remaining, check=False)
            except (OSError, subprocess.TimeoutExpired):
                raise DreamError("Local git source inspection unavailable: " + self.name, 3) from None
            output.seek(0)
            data = output.read(limit + 1)
        if result.returncode:
            if missing:
                return None
            raise DreamError("Local git object unavailable in mirror: " + self.name, 3)
        if len(data) > limit:
            raise DreamError("Git inspection exceeds its byte allowance", 3)
        return data

    def commit(self, revision):
        _text(revision, "commit")
        return self.git("rev-parse", "--verify", "--end-of-options", revision + "^{commit}").decode().strip()

    def entry(self, commit, path):
        data = self.git("ls-tree", "-z", commit, "--", path)
        if not data:
            return None
        rows = data.rstrip(b"\0").split(b"\0")
        if len(rows) != 1:
            raise DreamError("Source path must select exactly one git entry")
        info, name = rows[0].split(b"\t", 1)
        mode, kind, oid = info.decode("ascii").split()
        if name != path.encode("utf-8") or mode not in ("100644", "100755") or kind != "blob":
            raise DreamError("Source must be a regular git blob, not a symlink, directory, or gitlink")
        return oid

    def blob(self, oid, limit=MAX_BLOB_BYTES):
        size = int(self.git("cat-file", "-s", oid).strip())
        if size > limit:
            raise DreamError("Source blob exceeds the inspection byte allowance; select a smaller file", 3)
        return self.git("cat-file", "blob", oid, limit=limit)


def _path(value):
    _text(value, "path")
    if "\\" in value or any(part in ("", ".", "..") for part in value.split("/")):
        raise DreamError("Source path must be a canonical repository-relative path")
    return value


def _repository_uuid(source, commit):
    oid = source.entry(commit, ".sucoder/wiki/sources.json")
    if oid is None:
        return {"repository_uuid": str(uuid.uuid4()), "provisional": True}
    value = _json(source.blob(oid, MAX_CATALOG_BYTES))
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise DreamError("Unsupported source registry schema")
    try:
        identity = str(uuid.UUID(value["repository_uuid"]))
    except (KeyError, ValueError, TypeError, AttributeError):
        raise DreamError("Invalid source registry repository_uuid") from None
    return {"repository_uuid": identity, "provisional": False}


def _chunks(data, start, size, max_chunks):
    """Contiguous UTF-8 byte coverage; prefer whole lines to split text."""
    result = []
    position = 0
    while position < len(data):
        if len(result) >= max_chunks:
            raise DreamError("Extraction, verification, consolidation and repair reserve exceed max_role_invocations")
        end = min(position + size, len(data))
        if end < len(data):
            newline = data.rfind(b"\n", position, end)
            if newline >= position:
                end = newline + 1
            else:
                while data[end] & 0xC0 == 0x80:
                    end -= 1
        result.append(dict(byte_start=start + position, byte_end=start + end,
                           sha256=digest(data[position:end])))
        position = end
    return result


def _admit(unit, source, identity, limits):
    locator = _object(unit["locator"], ("repository", "commit", "path"), ("lines",))
    path = _path(locator["path"])
    commit = source.commit(locator["commit"])
    oid = source.entry(commit, path)
    if oid is None:
        raise DreamError("Selected source path is missing: " + path, 3)
    data = source.blob(oid)
    try:
        data.decode("utf-8")
    except UnicodeError:
        raise DreamError("Source must be UTF-8 text: " + path) from None
    if b"\0" in data:
        raise DreamError("Source contains binary NUL bytes: " + path)
    # Split only at LF: str.splitlines would also split control characters,
    # changing the git passage's byte offsets (ledger, section 3).
    count = data.count(b"\n") + int(bool(data) and not data.endswith(b"\n"))
    selected = locator.get("lines", [1, count])
    if (not isinstance(selected, list) or len(selected) != 2 or
            any(type(n) is not int for n in selected) or not 1 <= selected[0] <= selected[1] <= count):
        raise DreamError("lines must be an inclusive, nonempty one-based range within the blob")
    start, end = 0, len(data)
    if "lines" in locator:
        offset = 0
        for number, line in enumerate(io.BytesIO(data), 1):
            if number == selected[0]:
                start = offset
            offset += len(line)
            if number == selected[1]:
                end = offset
                break
    admitted = data[start:end]
    if len(admitted) > limits.max_source_bytes:
        raise DreamError("Selected evidence exceeds max_source_bytes")
    return dict(unit_id=unit["unit_id"], kind="git", shareable=unit["shareable"],
                source_id=digest(canonical([identity["repository_uuid"], "git", path])),
                repository=source.name, repository_uuid=identity["repository_uuid"],
                commit=commit, path=path, blob=oid, lines=selected,
                byte_start=start, byte_end=end, bytes=len(admitted), sha256=digest(admitted),
                chunks=_chunks(admitted, start, limits.chunk_bytes,
                               (limits.max_role_invocations - 1 - limits.max_repairs) // 2))


def _prior_wiki(source, base, limits, *, topic=None, units=(), policy="all-active-v1"):
    from . import dream_prior
    data = source.git("ls-tree", "-r", "-z", base, "--", ".sucoder/wiki/")
    files, selected, omitted = [], [], []
    total = 0
    for row in data.split(b"\0"):
        if not row:
            continue
        info, name = row.split(b"\t", 1)
        mode, kind, oid = info.decode("ascii").split()
        if kind != "blob" or mode not in ("100644", "100755"):
            raise DreamError("Prior wiki must contain regular files only")
        path = _path(name.decode("utf-8"))
        files.append(dict(path=path, blob=oid))
    chosen, selection = dream_prior.choose(source, files, limits, topic, units, policy)
    for item in files:
        path, oid = item["path"], item["blob"]
        if path in chosen:
            content = source.blob(oid, limits.max_prior_bytes)
            total += len(content)
            if total > limits.max_prior_bytes:
                raise DreamError("Prior wiki exceeds max_prior_bytes; explicitly increase the allowance")
            selected.append(dict(path=path, blob=oid, byte_start=0, byte_end=len(content), sha256=digest(content)))
        elif dream_prior.active(path):
            size = int(source.git("cat-file", "-s", oid).strip())
            omitted.append(dict(path=path, blob=oid, byte_start=0, byte_end=size, reason="unselected_active_topic"))
        else:
            omitted.append(dict(path=path, reason="archive_or_metadata"))
    result = dict(tree_digest=digest(canonical(files)), files=files, selected=selected,
                  omitted=omitted, selected_bytes=total, progress="not_evaluated")
    if selection is not None:
        result["selection"] = selection
    return result


def build_plan(config, mirror, request, base, *, maintenance=None, topic=None, inspection_deadline=None,
               prior_policy="bounded-prior-v1", publication=None, pinned_base=None):
    """Freeze selections, never imply inference or shared completion.

    ``publication="in_place"`` (essay-alignment S1) records that a reviewed
    result advances ``base`` itself, the wiki-only memory branch, instead of
    creating a candidate ref; the key is absent from candidate plans.
    ``pinned_base`` lets revalidation of such a plan rebuild against its frozen
    base after the branch has moved on to the plan's own generation; the
    caller establishes that, and the base must still be an ancestor.
    """
    from . import dream_transcript, dream_view
    _object(request, ("schema_version", "repository", "integration_ref", "units"), ("limits",))
    if type(request["schema_version"]) is not int or request["schema_version"] != SCHEMA_VERSION:
        raise DreamError("Unsupported source request schema_version")
    if request["repository"] != mirror:
        raise DreamError("Source request repository must match MIRROR")
    _text(base, "base")
    if base != request["integration_ref"] or not base.startswith(("refs/heads/", "refs/remotes/")) or base.startswith("refs/heads/dream/"):
        raise DreamError("--base must match a full non-dream integration_ref under refs/heads/ or refs/remotes/")
    if publication not in (None, "in_place") or (publication and not base.startswith("refs/heads/")):
        raise DreamError("In-place publication requires a local memory branch under refs/heads/")
    if (maintenance is None) != (topic is None):
        raise DreamError("--maintenance and --topic must be supplied together")
    if maintenance is not None:
        _text(maintenance, "maintenance reason")
        if not isinstance(topic, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", topic):
            raise DreamError("topic must be a simple topic name")
    limits = Limits.parse(request.get("limits", {}))
    units = request["units"]
    if not isinstance(units, list) or not 1 <= len(units) <= limits.max_units:
        raise DreamError("Source request requires between one and max_units units")
    seen = set()
    for unit in units:
        _object(unit, ("unit_id", "kind", "locator", "shareable"))
        uid = _text(unit["unit_id"], "unit_id")
        if uid in seen:
            raise DreamError("Duplicate unit_id: " + uid)
        seen.add(uid)
        if unit["kind"] not in ("git", *dream_transcript.KINDS, dream_view.KIND):
            raise DreamError("Unsupported evidence kind; use git, claude-jsonl, codex-jsonl or claude-view")
        if type(unit["shareable"]) is not bool:
            raise DreamError("shareable must explicitly be true or false")
        if unit["kind"] == "git":
            _object(unit["locator"], ("repository", "commit", "path"), ("lines",))
            _text(unit["locator"]["repository"], "locator repository")
        elif unit["kind"] == dream_view.KIND:
            _object(unit["locator"], ("path", "thread_id", "byte_start", "byte_end"))
        else:
            _object(unit["locator"], ("path", "thread_id"), ("byte_end",))
    deadline = time.monotonic() + 30
    if inspection_deadline is not None:
        deadline = min(deadline, inspection_deadline)
    target = GitSource(config, mirror, deadline)
    target.git("check-ref-format", base)
    if target.git("symbolic-ref", "--quiet", base, missing=True) is not None:
        raise DreamError("The designated integration ref must not be symbolic")
    base_commit = target.commit(base)
    if pinned_base is not None:
        if not publication or target.git("merge-base", "--is-ancestor", pinned_base, base_commit, missing=True) is None:
            raise DreamError("A pinned base must be an ancestor of the in-place memory branch", 3)
        base_commit = pinned_base
    if publication:
        from . import dream_memory
        dream_memory.check_tree(target, base_commit)
    identity = _repository_uuid(target, base_commit)
    sources = {mirror: target}
    identities = {mirror: identity}
    admitted, total = [], 0
    for unit in units:
        alias = unit["locator"]["repository"] if unit["kind"] == "git" else mirror
        if alias not in sources:
            sources[alias] = GitSource(config, alias, deadline)
            identities[alias] = _repository_uuid(sources[alias], sources[alias].commit("HEAD"))
        item = (_admit(unit, sources[alias], identities[alias], limits) if unit["kind"] == "git" else
                dream_view.admit(unit, alias, identities[alias], limits, deadline) if unit["kind"] == dream_view.KIND else
                dream_transcript.admit(unit, alias, identities[alias], limits, deadline))
        total += item["bytes"]
        if total > limits.max_source_bytes:
            raise DreamError("Selected evidence exceeds max_source_bytes")
        admitted.append(item)
    prior = _prior_wiki(target, base_commit, limits, topic=topic, units=admitted, policy=prior_policy)
    count = sum(len(item["chunks"]) for item in admitted)
    required = 2 * count + 1
    if required + limits.max_repairs > limits.max_role_invocations:
        raise DreamError("Extraction, verification, consolidation and repair reserve exceed max_role_invocations")
    if prior["selected_bytes"] + limits.max_proposal_bytes >= limits.max_consolidation_bytes:
        raise DreamError("Prior context and proposal allowance leave no consolidation space for briefs/metadata")
    for source in sources.values():
        source.check_binding()
    if pinned_base is None and target.commit(base) != base_commit:
        raise DreamError("Integration ref changed during planning; re-plan", 3)
    plan = dict(schema_version=SCHEMA_VERSION, operation="dream_plan", state="planned",
                repository=mirror, repository_uuid=identity["repository_uuid"],
                integration_ref=base, base_commit=base_commit,
                repositories={name: dict(path=str(source.path), **identities[name]) for name, source in sources.items()},
                units=admitted, prior_wiki=prior, maintenance=maintenance, topic=topic,
                limits=asdict(limits), work=dict(chunks=count, extraction_calls=count,
                    verification_calls=count, consolidation_calls=1, repair_reserve=limits.max_repairs,
                    required_role_invocations=required, selected_bytes=total),
                coverage=[dict(unit_id=item["unit_id"], state="planned", bytes=item["bytes"],
                               examined_bytes=0) for item in admitted],
                usage=dict(role_invocations=0, tokens=None, currency=None),
                execution=dict(available=False, reason="Planning performs no inference; dream run checks native execution capabilities"),
                candidate_commit=None, candidate_ref=None,
                integration=dict(state="not_integrated", evidence=None))
    if publication:
        plan["publication"] = publication
    return plan


def _open_private_directory(path):
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        os.close(fd)
        raise DreamError("Dream plan directory must be owned by the current account and private")
    return fd


def create_plan(config, mirror, sources, base, *, maintenance=None, topic=None, publication=None):
    """``publication=None`` derives in-place publication from the mirror's
    configured memory branch, so an explicit plan against it behaves like a
    bare dream; any other base keeps candidate publication."""
    try:
        if publication is None:
            from . import dream_memory
            publication = "in_place" if base == dream_memory.configured(config, mirror) else None
        catalog = os.open(Path(sources), os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(catalog, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise DreamError("Source catalog must be a regular JSON file")
            data = stream.read(MAX_CATALOG_BYTES + 1)
        if len(data) > MAX_CATALOG_BYTES:
            raise DreamError("Source catalog exceeds its byte allowance")
        plan = build_plan(config, mirror, _json(data), base, maintenance=maintenance, topic=topic,
                          publication=publication)
        plan_id = digest(canonical(plan))
        plan["plan_id"] = plan_id
        root = (config.log_dir or Path("~/.sucoder/logs")).expanduser().resolve() / "dream"
        for repository in plan["repositories"].values():
            repo = Path(repository["path"])
            if root == repo or repo in root.parents:
                raise DreamError("Dream plans must be stored outside every inspected repository")
        root.parent.mkdir(parents=True, exist_ok=True)
        fd = _open_private_directory(root)
        try:
            try:
                os.mkdir(plan_id, mode=0o700, dir_fd=fd)
            except FileExistsError:
                pass
            child = os.open(plan_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                info = os.fstat(child)
                if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
                    raise DreamError("Existing plan directory is not private")
                expected = canonical(plan) + b"\n"
                temporary = ".plan-" + uuid.uuid4().hex
                output = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=child)
                try:
                    with os.fdopen(output, "wb") as stream:
                        stream.write(expected)
                        stream.flush()
                        os.fsync(stream.fileno())
                    try:
                        os.link(temporary, "plan.json", src_dir_fd=child, dst_dir_fd=child, follow_symlinks=False)
                    except FileExistsError:
                        previous = os.open("plan.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=child)
                        with os.fdopen(previous, "rb") as stream:
                            info = os.fstat(stream.fileno())
                            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or
                                    stat.S_IMODE(info.st_mode) & 0o077 or stream.read(len(expected) + 1) != expected):
                                raise DreamError("Existing plan does not match its private content identity")
                    os.fsync(child)
                finally:
                    os.unlink(temporary, dir_fd=child)
            finally:
                os.close(child)
        finally:
            os.close(fd)
        return dict(plan, plan_path=str(root / plan_id / "plan.json"))
    except DreamError:
        raise
    except (OSError, UnicodeError) as exc:
        raise DreamError("Cannot read sources or store a private dream plan: " + str(exc), 3) from None
