"""Inference-free dream data plane (implementation ledger, sections 3-6).

The fixture backend proves quotation delivery and git publication mechanics,
not semantic extraction. Its receipts intentionally use a separate contract.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time
import uuid
from urllib.parse import quote

from . import dream
from . import dream_transcript
from .dream import DreamError, canonical, digest


MAX_PLAN_BYTES = 2 * 1024 * 1024
CONTRACT = "git-quotation-fixture-v1"
ORG_VALIDATOR = Path(__file__).with_name("dream_validate.el")


def _read(fd, name, limit):
    handle = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=fd)
    with os.fdopen(handle, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or
                stat.S_IMODE(info.st_mode) & 0o077):
            raise DreamError("Dream artifacts must be private regular files owned by this account")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise DreamError("Dream artifact exceeds its byte allowance")
    return data


def _store(fd, name, data):
    """Publish once; an interrupted write is retryable, never a valid artifact."""
    temporary = ".pending-" + uuid.uuid4().hex
    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
        except FileExistsError:
            if _read(fd, name, len(data)) != data:
                raise DreamError("Existing dream artifact differs; preserved without overwriting: " + name)
        os.fsync(fd)
    finally:
        os.unlink(temporary, dir_fd=fd)


@contextmanager
def _directory(config, plan_path):
    path = Path(plan_path).expanduser().absolute()
    identity = path.parent.name
    root = (config.log_dir or Path("~/.sucoder/logs")).expanduser().resolve() / "dream"
    if (not re.fullmatch(r"[0-9a-f]{64}", identity) or path.name != "plan.json" or
            path != root / identity / "plan.json"):
        raise DreamError("Use the stored plan.json under the configured private dream log directory")
    # Unlike the planner, opening a run must never create absent directories.
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(root_fd)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise DreamError("Dream root is not private and account-owned")
        child = os.open(identity, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            info = os.fstat(child)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise DreamError("Dream plan directory is not private and account-owned")
            yield child, path.parent
        finally:
            os.close(child)
    finally:
        os.close(root_fd)


def _load(fd, identity, mirror):
    plan = dream._json(_read(fd, "plan.json", MAX_PLAN_BYTES))
    if not isinstance(plan, dict) or plan.get("plan_id") != identity:
        raise DreamError("Plan identity does not match its storage directory")
    payload = {k: v for k, v in plan.items() if k != "plan_id"}
    if digest(canonical(payload)) != identity or plan.get("repository") != mirror:
        raise DreamError("Frozen plan content or mirror differs from its identity")
    _plan_shape(plan)
    return plan


def _plan_shape(plan):
    """Check persisted input types before any consumer indexes into the plan."""
    try:
        dream._object(plan, ("schema_version", "operation", "state", "repository", "repository_uuid",
            "integration_ref", "base_commit", "repositories", "units", "prior_wiki", "maintenance", "topic",
            "limits", "work", "coverage", "usage", "execution", "candidate_commit", "candidate_ref", "integration", "plan_id"),
            ("publication",))
        if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
            raise DreamError("Unsupported frozen plan schema")
        if plan["operation"] != "dream_plan" or plan["state"] != "planned":
            raise DreamError("Expected a frozen dream plan")
        ref = dream._text(plan["integration_ref"], "integration_ref")
        if not ref.startswith(("refs/heads/", "refs/remotes/")) or ref.startswith("refs/heads/dream/"):
            raise DreamError("Invalid frozen integration ref")
        if "publication" in plan and (plan["publication"] != "in_place" or not ref.startswith("refs/heads/")):
            raise DreamError("Invalid frozen publication mode")
        limits = dream.Limits.parse(plan["limits"])
        if not isinstance(plan["units"], list) or not 1 <= len(plan["units"]) <= limits.max_units:
            raise DreamError("Invalid frozen unit selection")
        for unit in plan["units"]:
            if unit.get("kind") in dream_transcript.KINDS:
                dream._object(unit, ("unit_id", "kind", "shareable", "source_id", "repository", "repository_uuid",
                    "path", "thread_id", "byte_start", "byte_end", "bytes", "sha256", "chunks", "normalization"))
                dream._text(unit["path"], "transcript path")
                dream._text(unit["thread_id"], "thread_id")
                if (type(unit["shareable"]) is not bool or type(unit["byte_end"]) is not int or
                        type(unit["byte_start"]) is not int or unit["byte_start"] != 0 or
                        type(unit["bytes"]) is not int or not 0 < unit["byte_end"] == unit["bytes"] <= limits.max_source_bytes):
                    raise DreamError("Invalid frozen transcript selection")
                dream._object(unit["normalization"], ("contract", "sha256", "records", "unknown_events"))
                if not isinstance(unit["chunks"], list) or not unit["chunks"]:
                    raise DreamError("Invalid frozen transcript chunks")
                continue  # Full semantic reconstruction below validates all digests and ranges.
            dream._object(unit, ("unit_id", "kind", "shareable", "source_id", "repository", "repository_uuid",
                "commit", "path", "blob", "lines", "byte_start", "byte_end", "bytes", "sha256", "chunks"))
            for field in ("unit_id", "source_id", "repository", "repository_uuid", "commit", "blob", "sha256"):
                dream._text(unit[field], field)
            dream._path(unit["path"])
            if (unit["kind"] != "git" or type(unit["shareable"]) is not bool or
                    any(type(unit[key]) is not int or unit[key] < 0 for key in ("byte_start", "byte_end", "bytes")) or
                    not isinstance(unit["lines"], list) or len(unit["lines"]) != 2 or
                    any(type(n) is not int or n < 1 for n in unit["lines"]) or
                    not isinstance(unit["chunks"], list) or not unit["chunks"]):
                raise DreamError("Invalid frozen unit schema")
            for chunk in unit["chunks"]:
                dream._object(chunk, ("byte_start", "byte_end", "sha256"))
                if (any(type(chunk[key]) is not int or chunk[key] < 0 for key in ("byte_start", "byte_end")) or
                        not isinstance(chunk["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", chunk["sha256"])):
                    raise DreamError("Invalid frozen chunk schema")
        dream._object(plan["prior_wiki"], ("tree_digest", "files", "selected", "omitted", "selected_bytes", "progress"), ("selection",))
        if "selection" in plan["prior_wiki"]:
            selection = plan["prior_wiki"]["selection"]
            dream._object(selection, ("contract", "strategy", "seed_paths"))
            if (selection["contract"] != "bounded-prior-v1" or selection["strategy"] not in ("all", "topic", "source_associations", "orientation") or
                    not isinstance(selection["seed_paths"], list)):
                raise DreamError("Invalid frozen prior-context selection")
            for path in selection["seed_paths"]:
                dream._path(path)
        if not isinstance(plan["repositories"], dict) or plan["repository"] not in plan["repositories"]:
            raise DreamError("Invalid frozen repository bindings")
        for repository in plan["repositories"].values():
            dream._object(repository, ("path", "repository_uuid", "provisional"))
            dream._text(repository["path"], "repository path")
            uuid.UUID(repository["repository_uuid"])
            if type(repository["provisional"]) is not bool:
                raise DreamError("Invalid frozen repository identity")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        if isinstance(exc, DreamError):
            raise
        raise DreamError("Invalid frozen plan schema: " + str(exc)) from None


def _own_generation(config, mirror, plan, fd, deadline):
    """True when an in-place plan's memory branch has moved on to (or past) one of
    this plan's own published generations, so revalidation pins the frozen base.
    Another dream's move is not ours: revalidation then fails as for candidates."""
    if not in_place_plan(plan):
        return False
    own = []
    for name in os.listdir(fd):
        if re.fullmatch(r"candidate-[0-9a-f]{64}\.txt", name):
            own.append(_read(fd, name, 128).decode().strip())
    if not own:
        return False
    source = dream.GitSource(config, mirror, deadline)
    tip = source.commit(plan["integration_ref"])
    if tip == plan["base_commit"]:
        return False
    for commit in own:
        if (re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) and
                source.git("show", "-s", "--format=%P", commit, missing=True) == (plan["base_commit"] + "\n").encode() and
                source.git("merge-base", "--is-ancestor", commit, tip, missing=True) is not None):
            return True
    return False


def _validate(config, mirror, plan, deadline, *, pinned=False):
    """Reconstruct the planner's full contract, reusing admission (ledger 2, 6)."""
    try:
        request = dict(schema_version=1, repository=mirror, integration_ref=plan["integration_ref"],
                       limits=plan["limits"], units=[dream_transcript.request_unit(u) if u["kind"] in dream_transcript.KINDS else dict(unit_id=u["unit_id"], kind=u["kind"],
                           shareable=u["shareable"], locator=dict(repository=u["repository"],
                               commit=u["commit"], path=u["path"], lines=u["lines"])) for u in plan["units"]])
        current = dream.build_plan(config, mirror, request, plan["integration_ref"],
            maintenance=plan["maintenance"], topic=plan["topic"], inspection_deadline=deadline,
            prior_policy=plan["prior_wiki"].get("selection", {}).get("contract", "all-active-v1"),
            publication=plan.get("publication"), pinned_base=plan["base_commit"] if pinned else None)
        # A provisional UUID is a frozen proposal, not a newly accepted registry.
        # Rebuilding the plan generates a fresh proposal; retain only the old UUID.
        for name, repository in current["repositories"].items():
            frozen = plan["repositories"][name]
            if repository["provisional"] and frozen["provisional"]:
                identity = str(uuid.UUID(frozen["repository_uuid"]))
                repository["repository_uuid"] = identity
                if name == mirror:
                    current["repository_uuid"] = identity
                for unit in current["units"]:
                    if unit["repository"] == name:
                        unit["repository_uuid"] = identity
                        logical = unit["path"] if unit["kind"] == "git" else unit["thread_id"]
                        unit["source_id"] = digest(canonical([identity, unit["kind"], logical]))
        expected = {k: v for k, v in plan.items() if k != "plan_id"}
        if expected["execution"] == dict(available=False, reason="Snapshot, receipt and role execution slices are not implemented"):
            current["execution"] = expected["execution"]  # Preserve original git-only frozen plans.
        if canonical(current) != canonical(expected):
            raise DreamError("Frozen plan no longer matches source binding, integration ref, registry or wiki; re-plan", 3)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        if isinstance(exc, DreamError):
            raise
        raise DreamError("Invalid frozen plan schema: " + str(exc)) from None


def _sources(config, plan, deadline):
    sources = {name: dream.GitSource(config, name, deadline) for name in plan["repositories"]}
    root = (config.log_dir or Path("~/.sucoder/logs")).expanduser().resolve() / "dream"
    for name, source in sources.items():
        if str(source.path) != plan["repositories"][name]["path"]:
            raise DreamError("Configured source path changed; re-plan", 3)
        if root == source.path or source.path in root.parents:
            raise DreamError("Dream artifacts must remain outside every inspected repository")
    return sources


def _snapshot(config, mirror, plan, fd, deadline):
    pinned = _own_generation(config, mirror, plan, fd, deadline)
    _validate(config, mirror, plan, deadline, pinned=pinned)
    sources = _sources(config, plan, deadline)
    records = []
    for unit in plan["units"]:
        source = sources[unit["repository"]]
        if unit["kind"] == "git":
            if source.entry(unit["commit"], unit["path"]) != unit["blob"]:
                raise DreamError("Frozen git source object changed or disappeared", 3)
            raw = source.blob(unit["blob"])[unit["byte_start"]:unit["byte_end"]]
        else:
            raw = dream_transcript.read_prefix(dream_transcript.request_unit(unit)["locator"],
                                               plan["limits"]["max_source_bytes"], deadline)
        if len(raw) != unit["bytes"] or digest(raw) != unit["sha256"]:
            raise DreamError("Snapshot bytes differ from the frozen selection", 3)
        name = "evidence-" + unit["sha256"] + ".txt"
        _store(fd, name, raw)
        if unit["kind"] != "git":
            normalized = canonical(dream_transcript.normalize(raw, unit["kind"], unit["thread_id"]))
            if digest(normalized) != unit["normalization"]["sha256"]:
                raise DreamError("Transcript normalization differs from frozen plan", 3)
            _store(fd, "normalized-" + unit["sha256"] + ".json", normalized + b"\n")
        records.append(dict(unit_id=unit["unit_id"], file=name, sha256=digest(raw),
                            bytes=len(raw), state="snapshotted", examined_bytes=0))
    for item in plan["prior_wiki"]["selected"]:
        raw = sources[mirror].blob(item["blob"], plan["limits"]["max_prior_bytes"])
        if digest(raw) != item["sha256"]:
            raise DreamError("Prior wiki snapshot differs from the frozen selection", 3)
        _store(fd, "prior-" + item["sha256"] + ".org", raw)
    for source in sources.values():
        source.check_binding()
    if not pinned and sources[mirror].commit(plan["integration_ref"]) != plan["base_commit"]:
        raise DreamError("Integration ref changed while snapshotting; re-plan", 3)
    result = dict(schema_version=1, operation="dream_snapshot", plan_id=plan["plan_id"],
                  state="snapshotted", coverage=records, usage=plan["usage"],
                  candidate_commit=None, candidate_ref=None, integration=plan["integration"])
    _store(fd, "snapshot.json", canonical(result) + b"\n")
    return result, sources


def snapshot(config, mirror, plan_path):
    with _directory(config, plan_path) as (fd, directory):
        plan = _load(fd, directory.name, mirror)
        deadline = time.monotonic() + dream.Limits.parse(plan["limits"]).max_elapsed_seconds
        result, _ = _snapshot(config, mirror, plan, fd, deadline)
        return dict(result, snapshot_path=str(directory / "snapshot.json"))


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise DreamError("Dream run elapsed allowance exhausted", 3)
    return min(5, remaining)


def _git_write(source, index, args, deadline, data=None):
    """Only coordinator-owned plumbing; never use the user's git index/hooks."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_INDEX_FILE=str(index), GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0",
               GIT_ALLOW_PROTOCOL="", GIT_LITERAL_PATHSPECS="1", GIT_OPTIONAL_LOCKS="0")
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        try:
            result = subprocess.run(["git", "--no-replace-objects", "-C", str(source.path),
                "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
                "-c", "gc.auto=0", "-c", "maintenance.auto=false", "-c", "commit.gpgSign=false",
                *args], input=data, stdout=output, stderr=errors, env=env,
                timeout=_remaining(deadline), check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise DreamError("Dream git publication unavailable or timed out", 3) from None
        if result.returncode:
            raise DreamError("Dream git plumbing or ref transaction failed; preserve results and re-plan", 3)
        output.seek(0)
        value = output.read(4097)
    if len(value) > 4096:
        raise DreamError("Unexpected git publication response", 1)
    return value.decode("ascii").strip()


def _validate_org(files, deadline):
    """Native Org parsing, with no init files, Babel evaluation or export."""
    try:
        result = subprocess.run(["emacs", "--batch", "-Q", "--script", str(ORG_VALIDATOR)],
            input=canonical(files), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=_remaining(deadline), check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise DreamError("Native Org validation unavailable or timed out", 3) from None
    if result.returncode:
        raise DreamError("Native Org validation rejected fixture pages", 1)
    return dream._json(result.stdout)


def _fixture(plan, fd):
    """Structured fixture delivery of every chunk, followed by exact comparison."""
    deliveries, sections, bindings = [], [], []
    run_id = digest(canonical([CONTRACT, plan["plan_id"]]))
    path = ".sucoder/wiki/topics/dream-" + run_id + ".org"
    for number, unit in enumerate(plan["units"]):
        raw = _read(fd, "evidence-" + unit["sha256"] + ".txt", unit["bytes"])
        if digest(raw) != unit["sha256"] or len(raw) != unit["bytes"]:
            raise DreamError("Private evidence changed after snapshotting", 1)
        for chunk in unit["chunks"]:
            start, end = chunk["byte_start"], chunk["byte_end"]
            delivered = raw[start - unit["byte_start"]:end - unit["byte_start"]]
            if digest(delivered) != chunk["sha256"]:
                raise DreamError("Fixture chunk failed exact-byte verification", 1)
            deliveries.append(dict(unit_id=unit["unit_id"], byte_start=start, byte_end=end,
                                   sha256=digest(delivered), extraction="delivered", verification="exact_bytes"))
        target = "source-" + str(number + 1)
        # Org's comma escape prevents admitted block terminators/headlines from
        # becoming structure. The original bytes remain in the private snapshot.
        text = raw.decode("utf-8").replace("\r\n", "\n")
        escaped = re.sub(r"(?m)^(\*|[ \t]*#\+)", r",\1", text)
        section = ("* Admitted quotation " + str(number + 1) + "\n:PROPERTIES:\n:CUSTOM_ID: " + target +
            "\n:END:\nThis is a deterministic fixture quotation, not a synthesized claim.\n"
            "Git commit =" + unit["commit"] + "=; blob =" + unit["blob"] + "=; lines " +
            str(unit["lines"][0]) + "-" + str(unit["lines"][1]) + ".\n" +
            "Source: [[file:../../../" + quote(unit["path"], safe="/") + "][" +
            "admitted git passage]] (repository UUID =" + unit["repository_uuid"] + "=).\n" +
            "#+begin_example\n" + escaped + ("" if escaped.endswith("\n") else "\n") + "#+end_example\n")
        if unit["repository"] != plan["repository"]:
            # A file link into the target repository would misrepresent provenance.
            section = section.replace("[[file:../../../" + quote(unit["path"], safe="/") +
                "][admitted git passage]]", "=" + quote(unit["path"], safe="/") + "=")
        sections.append(section)
        bindings.append(dict(path=path, target=target, sha256=digest(section.encode()), rule="org-subtree-utf8-v1"))
    content = "#+title: Dream quotation fixture\n\n" + "".join(sections)
    bindings.append(dict(path=".sucoder/wiki/index.org", target="dream-" + run_id,
                         sha256=digest(_index_entry(run_id).encode()), rule="org-subtree-utf8-v1"))
    receipt = dict(schema_version=1, contract=CONTRACT, run_id=run_id, plan_id=plan["plan_id"],
        repository_uuid=plan["repository_uuid"], base_commit=plan["base_commit"],
        units=[{key: unit[key] for key in ("unit_id", "kind", "source_id", "repository_uuid", "commit",
            "path", "blob", "lines", "byte_start", "byte_end", "bytes", "sha256", "chunks")} for unit in plan["units"]],
        outputs=bindings, coverage=deliveries, fidelity="exact_quotation_only",
        usage=dict(role_invocations=plan["work"]["required_role_invocations"], tokens=None, currency=None))
    return path, content, receipt


def _index_entry(run_id):
    return ("* Quotation fixture " + run_id + "\n:PROPERTIES:\n:CUSTOM_ID: dream-" + run_id +
            "\n:END:\n[[file:topics/dream-" + run_id + ".org::#source-1][Admitted quotations]]\n")


def run_fixture(config, mirror, plan_path, harness):
    if harness != "fixture":
        raise DreamError("Only the deterministic fixture harness is implemented; no native agent is invoked")
    with _directory(config, plan_path) as (fd, directory):
        plan = _load(fd, directory.name, mirror)
        if any(unit["kind"] != "git" for unit in plan["units"]):
            raise DreamError("The quotation fixture supports git evidence only; select a native role harness for transcripts")
        deadline = time.monotonic() + dream.Limits.parse(plan["limits"]).max_elapsed_seconds
        if any(not unit["shareable"] for unit in plan["units"]):
            raise DreamError("Fixture publication requires every selected unit to be explicitly shareable")
        if any(name != mirror and item["provisional"] for name, item in plan["repositories"].items()):
            raise DreamError("External source repositories need an accepted UUID before fixture publication")
        _, sources = _snapshot(config, mirror, plan, fd, deadline)
        path, content, receipt = _fixture(plan, fd)
        return publish(plan, fd, directory, sources, deadline, path, content, receipt, _index_entry(receipt["run_id"]))


def output_bindings_match(files, parsed, bindings):
    """Versioned exact output rules, including text outside Org subtrees."""
    for binding in bindings:
        path = binding["path"]
        if path not in files:
            return False
        if binding["rule"] == "file-utf8-v1":
            actual = digest(files[path].encode())
        elif binding["rule"] == "org-subtree-utf8-v1":
            actual = parsed.get(path, {}).get(binding["target"])
        else:
            raise DreamError("Unsupported dream output binding rule")
        if actual != binding["sha256"]:
            return False
    return True


def bound_outputs(source, base, bindings, deadline):
    """Read and validate required outputs in one pinned integration tree."""
    files = {}
    try:
        for binding in bindings:
            if binding["path"] in files:
                continue
            blob = source.entry(base, binding["path"])
            if blob is None:
                return False
            files[binding["path"]] = source.blob(blob).decode("utf-8")
        return output_bindings_match(files, _validate_org(files, deadline), bindings)
    except UnicodeError:
        return False
    except DreamError as exc:
        if exc.exit_code not in (1, 2):
            raise
        return False


def publish(plan, fd, directory, sources, deadline, path, content, receipt, index_entry,
            *, receipt_name="fixture-receipt.json", result_name="run.json", wiki_files=None):
    """Shared atomic publication; wiki edits are validated before this call."""
    source = sources[plan["repository"]]
    if len(canonical(receipt)) + len(content.encode()) > plan["limits"]["max_proposal_bytes"]:
        raise DreamError("Fixture proposals exceed max_proposal_bytes", 3)
    run_id = receipt["run_id"]
    in_place = in_place_plan(plan)
    ref = plan["integration_ref"] if in_place else "refs/heads/dream/" + run_id
    previous = {item["path"]: item["blob"] for item in plan["prior_wiki"]["files"]}
    receipt_path = ".sucoder/wiki/receipts/" + run_id + ".json"
    if (wiki_files is None and path in previous) or receipt_path in previous:
        raise DreamError("Fixture output already exists in the integration tree; re-plan")
    index_path = ".sucoder/wiki/index.org"
    if wiki_files is None:
        index = source.blob(previous[index_path]).decode("utf-8") if index_path in previous else "#+title: Companion wiki\n"
        index += "\n" + index_entry
        files = {path: content, index_path: index}
    else:
        files = dict(wiki_files)
        if not files:
            raise DreamError("A wiki change set must not be empty")
        for name, text in files.items():
            dream._path(name)
            if (not name.startswith(".sucoder/wiki/") or not name.endswith(".org") or
                    not isinstance(text, str) or not text.strip()):
                raise DreamError("Wiki edits must be nonempty Org files within the companion")
        if sum(len(text.encode()) for text in files.values()) + len(canonical(receipt)) > plan["limits"]["max_proposal_bytes"]:
            raise DreamError("Wiki proposal exceeds max_proposal_bytes", 3)
    required = dict(files)
    for binding in receipt["outputs"]:
        name = binding["path"]
        if name not in required:
            if wiki_files is None or name not in previous or not name.endswith(".org"):
                raise DreamError("Required output is absent from the proposed and frozen wiki")
            remaining = plan["limits"]["max_consolidation_bytes"] - sum(len(text.encode()) for text in required.values())
            if remaining <= 0:
                raise DreamError("Required wiki validation context exceeds max_consolidation_bytes", 3)
            try:
                required[name] = source.blob(previous[name], remaining).decode("utf-8")
            except UnicodeError:
                raise DreamError("Required frozen wiki output is not UTF-8") from None
    if sum(len(text.encode()) for text in required.values()) > plan["limits"]["max_consolidation_bytes"]:
        raise DreamError("Required wiki validation context exceeds max_consolidation_bytes", 3)
    parsed = _validate_org(required, deadline)
    if not output_bindings_match(required, parsed, receipt["outputs"]):
        raise DreamError("Native Org output differs from its receipt binding", 1)
    files[receipt_path] = canonical(receipt).decode() + "\n"
    registry = ".sucoder/wiki/sources.json"
    if registry not in previous:
        files[registry] = canonical(dict(schema_version=1, repository_uuid=plan["repository_uuid"])).decode() + "\n"
    if sum(len(value.encode()) for value in files.values()) > plan["limits"]["max_consolidation_bytes"]:
        raise DreamError("Fixture staged output exceeds max_consolidation_bytes", 3)
    _store(fd, receipt_name, canonical(receipt) + b"\n")
    # Storing objects may leave harmless unreachable objects on interruption.
    # Only the final compare-and-swap transaction publishes a candidate ref.
    with tempfile.TemporaryDirectory(prefix=".index-", dir=directory) as scratch:
        private_index = Path(scratch) / "index"
        _git_write(source, private_index, ["read-tree", plan["base_commit"]], deadline)
        for name, text in files.items():
            oid = _git_write(source, private_index, ["hash-object", "-w", "--no-filters", "--stdin"], deadline, text.encode())
            _git_write(source, private_index, ["update-index", "--add", "--cacheinfo", "100644", oid, name], deadline)
        tree = _git_write(source, private_index, ["write-tree"], deadline)
        commit_name = "candidate-" + run_id + ".txt"
        try:
            commit = _read(fd, commit_name, 128).decode().strip()
        except FileNotFoundError:
            if in_place:
                message = ("Dream memory generation " + run_id + "\n\nReviewed by the dream coordinator and published "
                           "in place to " + ref + ".\nPlan " + plan["plan_id"] + "; receipt " + receipt_path + ".\n")
            else:
                message = ("Dream candidate " + run_id +
                           "\n\nRequires ordinary review and integration.\n")
            commit = _git_write(source, private_index, ["commit-tree", tree, "-p", plan["base_commit"], "-F", "-"], deadline, message.encode())
            _store(fd, commit_name, (commit + "\n").encode())
        if (not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) or
                source.git("show", "-s", "--format=%T %P", commit).decode().strip() != tree + " " + plan["base_commit"]):
            raise DreamError("Retained candidate object does not match the frozen output tree", 1)
        for item in sources.values():
            item.check_binding()
        if (source.git("symbolic-ref", "--quiet", ref, missing=True) is not None or
                source.git("symbolic-ref", "--quiet", plan["integration_ref"], missing=True) is not None):
            raise DreamError("Candidate and integration refs must not be symbolic", 3)
        existing = source.git("rev-parse", "--verify", ref, missing=True)
        existing = existing.decode().strip() if existing is not None else None
        if in_place:
            # The memory branch moves from the frozen base to this generation in
            # one checked update; a replay after publication finds it already there.
            if existing == commit or (existing is not None and source.git(
                    "merge-base", "--is-ancestor", commit, existing, missing=True) is not None):
                transaction = None  # Already published, possibly with later generations on top.
            elif existing != plan["base_commit"]:
                raise DreamError("Memory branch moved since planning; the reviewed generation is retained "
                                 "privately; re-plan against the new memory tip", 3)
            else:
                transaction = ("start\nupdate " + ref + " " + commit + " " + plan["base_commit"] +
                               "\nprepare\ncommit\n")
        else:
            if existing is not None and existing != commit:
                raise DreamError("Candidate ref already exists with different content", 3)
            action = "create" if existing is None else "verify"
            transaction = ("start\nverify " + plan["integration_ref"] + " " + plan["base_commit"] +
                           "\n" + action + " " + ref + " " + commit + "\nprepare\ncommit\n")
        if transaction is not None:
            _git_write(source, private_index, ["update-ref", "--no-deref", "--stdin"], deadline, transaction.encode())
    result = _run_result(plan, receipt, commit)
    _store(fd, result_name, canonical(result) + b"\n")
    return result


def in_place_plan(plan):
    """A plan frozen to update its wiki-only memory branch (essay-alignment S1)."""
    return plan.get("publication") == "in_place"


def _run_result(plan, receipt, commit):
    if in_place_plan(plan):
        return dict(schema_version=1, operation="dream_run", run_id=receipt["run_id"], plan_id=plan["plan_id"],
            state="published", harness=receipt.get("harness", "fixture"), fidelity=receipt["contract"],
            base_commit=plan["base_commit"], candidate_commit=commit, candidate_ref=None,
            memory_ref=plan["integration_ref"], coverage=receipt["coverage"], usage=receipt["usage"],
            integration=dict(state="integrated", evidence=dict(integration_commit=commit)),
            **({key: receipt[key] for key in ("model", "native_version")} if "harness" in receipt else {}))
    return dict(schema_version=1, operation="dream_run", run_id=receipt["run_id"], plan_id=plan["plan_id"],
        state="candidate_ready", harness=receipt.get("harness", "fixture"), fidelity=receipt["contract"],
        base_commit=plan["base_commit"], candidate_commit=commit,
        candidate_ref="refs/heads/dream/" + receipt["run_id"],
        coverage=receipt["coverage"], usage=receipt["usage"],
        integration=dict(state="not_integrated", evidence=None),
        **({key: receipt[key] for key in ("model", "native_version")} if "harness" in receipt else {}))


def status(config, mirror, plan_path, run_id=None):
    """Check the designated integration tree; candidate ancestry is irrelevant."""
    with _directory(config, plan_path) as (fd, directory):
        plan = _load(fd, directory.name, mirror)
        deadline = time.monotonic() + 30
        fixture_id = digest(canonical([CONTRACT, plan["plan_id"]]))
        native_specs = [name[len("native-"):-len("-spec.json")] for name in os.listdir(fd)
                        if re.fullmatch(r"native-[0-9a-f]{64}-spec\.json", name)]
        if run_id is None:
            if len(native_specs) > 1:
                raise DreamError("Multiple native runs exist; select one with --run RUN_ID")
            run_id = native_specs[0] if native_specs else fixture_id
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{64}", run_id):
            raise DreamError("Invalid --run identity")
        native = run_id != fixture_id
        if native and run_id not in native_specs:
            raise DreamError("Requested native run does not exist")
        if native:
            spec = dream._json(_read(fd, "native-" + run_id + "-spec.json", 16384))
            if spec.get("contract") == "accepted-wiki-progress-v1":
                if spec.get("run_id") != run_id:
                    raise DreamError("Shared-progress specification differs from its storage identity")
                from . import dream_progress
                return dream_progress.restore(config, mirror, plan, fd, spec, deadline)
            if spec.get("contract") in ("source-grounded-wiki-v3", "source-grounded-wiki-v4", "source-grounded-wiki-v5"):
                try:
                    marker = dream._json(_read(fd, "native-" + run_id + "-unchanged.json", MAX_PLAN_BYTES))
                except FileNotFoundError:
                    pass
                else:
                    from . import dream_curate
                    return dream_curate.unchanged_status(config, mirror, plan, fd, run_id, marker, deadline)
        try:
            commit = _read(fd, "candidate-" + run_id + ".txt", 128).decode().strip()
        except FileNotFoundError:
            return dict(schema_version=1, operation="dream_status", plan_id=plan["plan_id"],
                        state="incomplete" if native else "not_run", run_id=run_id,
                        integration=dict(state="not_integrated", evidence=None))
        # Recompute fixture identities and expected receipt, rather than trust a
        # mutable local run record as evidence of shared accepted progress.
        if native:
            spec = dream._json(_read(fd, "native-" + run_id + "-spec.json", 16384))
            if spec.get("contract") in ("source-grounded-wiki-v1", "source-grounded-wiki-v2", "source-grounded-wiki-v3", "source-grounded-wiki-v4", "source-grounded-wiki-v5"):
                from . import dream_wiki
                expected = dream_wiki.restore(plan, fd, run_id, deadline)
            else:
                from . import dream_roles
                expected = dream_roles.restore(plan, fd, run_id, deadline)
        else:
            _, _, expected = _fixture(plan, fd)
        run = _run_result(plan, expected, commit)
        try:
            recorded = dream._json(_read(fd, "native-" + run_id + "-run.json" if native else "run.json", MAX_PLAN_BYTES))
        except FileNotFoundError:
            pass  # Interruption between atomic ref publication and local record.
        else:
            if canonical(recorded) != canonical(run):
                raise DreamError("Stored run record does not match its frozen role plan")
        source = _sources(config, plan, deadline)[mirror]
        if in_place_plan(plan):
            # A generation stays published while the memory branch descends
            # from it; later generations and reverts are new commits on top.
            memory = source.commit(plan["integration_ref"])
            contained = source.git("merge-base", "--is-ancestor", commit, memory, missing=True) is not None
            source.check_binding()
            return dict(run, operation="dream_status", state="published" if contained else "detached",
                        integration=dict(state="integrated", evidence=dict(integration_commit=memory))
                        if contained else dict(state="not_integrated", evidence=None))
        base = source.commit(plan["integration_ref"])
        receipt_path = ".sucoder/wiki/receipts/" + expected["run_id"] + ".json"
        oid = source.entry(base, receipt_path)
        state = "not_integrated"
        evidence = None
        if oid is not None:
            state = "needs_reconciliation"
            evidence = dict(integration_commit=base, receipt_blob=oid, contract=expected["contract"])
            if source.blob(oid, MAX_PLAN_BYTES) == canonical(expected) + b"\n":
                if bound_outputs(source, base, expected["outputs"], deadline):
                    state = "integrated_candidate" if native else "integrated_fixture"
        candidate = source.git("rev-parse", "--verify", run["candidate_ref"], missing=True)
        candidate_state = ("missing" if candidate is None else
                           "available" if candidate.decode().strip() == run["candidate_commit"] else "changed")
        if source.git("symbolic-ref", "--quiet", run["candidate_ref"], missing=True) is not None:
            candidate_state = "changed"
        source.check_binding()
        if source.commit(plan["integration_ref"]) != base:
            raise DreamError("Integration ref changed during status inspection; retry", 3)
        return dict(run, operation="dream_status", candidate_status=candidate_state,
                    state="candidate_ready" if candidate_state == "available" else "candidate_prepared",
                    integration=dict(state=state, evidence=evidence))
