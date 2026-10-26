"""Deterministic verification adapter; receiving path is Python 3.6/stdlb.

No scheduler submission or model process: an existing burst job executes the
frozen request with the same project preparation as a dispatched agent.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import uuid

try:
    from . import dispatch_project as project
    from . import dispatch_worker as worker
    from . import dispatch_artifacts as artifacts
    from .dispatch_protocol import atomic_json
except ImportError:
    import dispatch_project as project
    import dispatch_worker as worker
    import dispatch_artifacts as artifacts
    from dispatch_protocol import atomic_json


HELPERS = ("dispatch_verify.py", "dispatch_project.py", "dispatch_worker.py", "dispatch_protocol.py",
           "dispatch_artifacts.py", "codex_state_runner.py")


def freeze(repository, ref, profile_path, command_file, directory, shared_path, inspect_git,
           script_file=None, timeout=600, local_disk_root="/local", cpus=1, required_evidence=None):
    """Freeze source, command, profile and helpers before an existing sbatch."""
    if type(timeout) is not int or timeout < 1 or type(cpus) is not int or cpus < 1:
        raise ValueError("verification timeout and CPU budget must be positive integers")
    if not isinstance(local_disk_root, str) or not os.path.isabs(local_disk_root) or local_disk_root in ("/", "/tmp", "/var/tmp"):
        raise ValueError("verification requires an explicit receiving local-disk base")
    repository = shared_path(Path(repository), "verification source")
    directory = shared_path(Path(directory), "verification stage")
    if directory.exists():
        raise ValueError("verification stage already exists; use one fresh directory per job")
    if not isinstance(ref, str) or not ref or ref.startswith("-"):
        raise ValueError("verification ref must name committed source")
    config = inspect_git(repository, "config", "--local", "--list").lower()
    common = Path(inspect_git(repository, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    shared_path(common, "verification Git metadata")
    shared_path(Path(inspect_git(repository, "rev-parse", "--absolute-git-dir")), "verification worktree metadata")
    if ("extensions.partialclone=" in config or re.search(r"^remote\..*\.promisor=(?:true|yes|on|1)$", config, re.M)
            or list((common / "objects" / "pack").glob("*.promisor"))):
        raise ValueError("partial/promisor verification sources are unsupported")
    commit = inspect_git(repository, "rev-parse", "--verify", ref + "^{commit}")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise ValueError("verification source did not resolve to a commit")
    prepared = project.read_profile(profile_path, shared_path=shared_path)
    required = [] if required_evidence is None else list(required_evidence)
    for name in required:
        project.relative_name(name)
    if len(set(required)) != len(required):
        raise ValueError("verification required evidence names must be unique")
    if any(n > cpus for n in prepared["profile"]["thread_limits"].values()):
        raise ValueError("project thread limit exceeds verification CPU budget")
    script = None
    if bool(command_file) == bool(script_file):
        raise ValueError("choose exactly one verification command file or script file")
    if script_file:
        script = Path(script_file).read_bytes()
        if not script or len(script) > project.MAX_JSON:
            raise ValueError("verification script must be nonempty and at most one MiB")
        command = ["/bin/bash", str(directory / "verification-script.sh")]
    else:
        with open(str(command_file), "rb") as f:
            encoded = f.read(project.MAX_JSON + 1)
        if len(encoded) > project.MAX_JSON:
            raise ValueError("verification command exceeds one MiB")
        command = json.loads(encoded.decode("utf-8"))
        if not isinstance(command, list) or not command or any(not isinstance(a, str) or "\x00" in a for a in command):
            raise ValueError("verification command file must contain a nonempty JSON argv array")
    directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)  # clear inherited setgid on the directory we created
    request = {"schema_version": 1, "uid": os.getuid(), "id": uuid.uuid4().hex,
               "coordinator_job_id": os.environ.get("SLURM_JOB_ID"),
               "repository": str(repository), "commit": commit, "command": command,
               "timeout_seconds": timeout, "cpus": cpus, "local_disk_root": local_disk_root,
               "required_evidence": required,
               "source_history": "single-commit verification; full current tree, no alternates",
               "project_sha256": prepared["sha256"]}
    names = project.stage_profile(prepared, directory)
    if script is not None:
        project.private_bytes(directory / "verification-script.sh", script)
        names.append("verification-script.sh")
    for name in HELPERS:
        project.private_bytes(directory / name, Path(__file__).with_name(name).read_bytes())
        names.append(name)
    project.private_bytes(directory / "verification-request.json", json.dumps(request, sort_keys=True).encode("utf-8"))
    project.private_bytes(directory / "environment.json", b"{}")
    names.extend(("verification-request.json", "environment.json"))
    atomic_json(str(directory / "verification-digests.json"), {name: project.digest_file(directory / name) for name in names})
    return {"schema_version": 1, "state": "prepared", "directory": str(directory),
            "repository": str(repository), "commit": commit, "project_sha256": prepared["sha256"],
            "run": ["python3", str(directory / "dispatch_verify.py"), str(directory)]}


def read_request(directory):
    directory = Path(directory)
    info = directory.lstat()
    if (not directory.is_absolute() or directory.is_symlink() or info.st_uid != os.getuid()
            or info.st_mode & 0o077):
        raise ValueError("verification stage must be a private owned real directory")
    worker.private_file(str(directory / "verification-digests.json"))
    digests = project.bounded_json(directory / "verification-digests.json")
    required = set(HELPERS) | {"project.json", "project-bootstrap.sh", "project-inputs.sh",
                              "verification-request.json", "environment.json"}
    if not required.issubset(digests):
        raise ValueError("verification helper integrity record is incomplete")
    for name, digest in digests.items():
        if not isinstance(name, str) or os.path.basename(name) != name or name in (".", ".."):
            raise ValueError("invalid verification integrity path")
        path = directory / name
        worker.private_file(str(path))
        if project.digest_file(path) != digest:
            raise ValueError("verification input differs from frozen assignment: " + name)
    request = project.bounded_json(directory / "verification-request.json")
    if request.get("schema_version") != 1 or request.get("uid") != os.getuid():
        raise ValueError("verification request schema or uid mismatch")
    if project.digest_file(directory / "project.json") != request.get("project_sha256"):
        raise ValueError("verification project identity differs from the request")
    return request


def source_identity(work, commit, environment):
    """Compare HEAD, the real index and tracked bytes with the frozen commit.

    A fresh index avoids trusting assume-unchanged/skip-worktree flags left by
    the command. Untracked and ignored runtime outputs remain permitted.
    """
    metadata = Path(work) / ".git"
    if metadata.is_symlink() or not metadata.is_dir():
        raise ValueError("verification Git metadata is no longer a private directory")
    command = ["git", "--no-replace-objects", "--no-optional-locks", "-C", work, "-c", "core.fsmonitor=false",
               "-c", "core.untrackedCache=false"]

    def git(*args, **kwargs):
        return subprocess.check_output(command + list(args), env=kwargs.get("env", environment),
                                       stderr=subprocess.DEVNULL, timeout=30).decode().strip()

    def clean(args, env):
        process = subprocess.run(command + args, env=env, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, timeout=30)
        if process.returncode not in (0, 1):
            raise ValueError("verification source comparison failed")
        return process.returncode == 0

    head = git("rev-parse", "--verify", "HEAD^{commit}")
    index_clean = clean(["diff", "--cached", "--no-ext-diff", "--no-textconv", "--quiet", commit, "--"], environment)
    with tempfile.TemporaryDirectory(prefix="source-check-", dir=environment["TMPDIR"]) as scratch:
        fresh = dict(environment, GIT_INDEX_FILE=os.path.join(scratch, "index"))
        git("read-tree", commit, env=fresh)
        clean(["update-index", "--refresh"], fresh)
        tracked_clean = clean(["diff-files", "--no-ext-diff", "--no-textconv", "--quiet", "--"], fresh)
    return {"state": "matched" if head == commit and index_clean and tracked_clean else "drifted",
            "expected_commit": commit, "actual_commit": head,
            "index_clean": index_clean, "tracked_clean": tracked_clean}


def run(directory, inherited=None, mountinfo_path="/proc/self/mountinfo", ownership_factory=None):
    """Run a frozen deterministic request inside the existing burst allocation."""
    directory = Path(directory)
    request = read_request(directory)
    project.private_bytes(directory / "started.json", json.dumps({"pid": os.getpid(), "started_at": time.time()}).encode("utf-8"))
    old_umask = os.umask(0o077)
    environment = None
    supervisor_environment = None
    work = None
    source_config = None
    ownership = None
    stopped = [None]
    handlers = {}
    record = {"schema_version": 1, "state": "starting", "phase": "receiving_setup", "commit": request["commit"],
              "repository": request["repository"], "source_history": request["source_history"],
              "command_exit_code": None, "coordinator_acceptance": "pending", "started_at": time.time()}

    def publish():
        record["observed_at"] = time.time()
        atomic_json(str(directory / "verification-status.json"), record)

    def interrupted(signum, frame):
        stopped[0] = signum

    try:
        inherited = dict(os.environ) if inherited is None else inherited
        if request.get("coordinator_job_id") and inherited.get("SLURM_JOB_ID") == request["coordinator_job_id"]:
            raise ValueError("verification must use a separate allocation, not the coordinator job")
        root, job, mount = worker.receiving_root(request, inherited, mountinfo_path)
        ownership = (ownership_factory or worker.JobOwnership)(job)
        record.update(local_root=root, job_id=job, mount=mount)
        manifest = {"dispatch_id": request["id"], "directory": str(directory),
                    "environment_file": str(directory / "environment.json"),
                    "runtime_path": ["/usr/local/bin", "/usr/bin", "/bin"], "resources": {"cpus": request["cpus"]}}
        environment = worker.environment_for(manifest, inherited, root)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            handlers[signum] = signal.signal(signum, interrupted)
        work = str(Path(root) / "verification" / request["id"] / "source")
        record.update(work_path=work, phase="cloning")
        publish()
        # Deterministic checks need an exact tree, not history for prior-art
        # research. Retain the real commit and shallow boundary explicitly.
        clone = ('set -eu\nmkdir -p ' + shlex.quote(work) + '\ngit -C ' + shlex.quote(work) + ' init -q\n'
                 + 'git -C ' + shlex.quote(work) + ' -c protocol.file.allow=always fetch --no-tags --no-recurse-submodules --depth=1 '
                 + shlex.quote(request["repository"]) + ' ' + shlex.quote(request["commit"]) + '\n'
                 + 'git -C ' + shlex.quote(work) + ' checkout --detach -q ' + shlex.quote(request["commit"]) + '\n')
        project.private_bytes(directory / "verify-clone.sh", clone.encode("utf-8"))
        worker.run_script(str(directory / "verify-clone.sh"), environment, request["timeout_seconds"],
                          cancellation=stopped, heartbeat=publish)
        source_config = project.digest_file(Path(work) / ".git" / "config")
        record["phase"] = "project_setup"
        publish()
        supervisor_environment = environment
        environment, runtime = project.prepare(str(directory), work, root, environment, worker.run_script,
                                                int(environment["OMP_NUM_THREADS"]), heartbeat=publish, cancellation=stopped)
        record["project"] = runtime
        if project.digest_file(Path(work) / ".git" / "config") != source_config:
            raise ValueError("verification bootstrap changed Git configuration")
        record["source_identity"] = source_identity(work, request["commit"], supervisor_environment)
        if record["source_identity"]["state"] != "matched":
            raise ValueError("verification bootstrap changed tracked source, index or HEAD")
        replacements = {"{worktree}": work, "{python}": environment.get("PROJECT_PYTHON", ""),
                        "{evidence}": environment["SUCODER_DISPATCH_EVIDENCE_DIR"]}
        command = []
        for argument in request["command"]:
            for old, new in replacements.items():
                if old in argument and not new:
                    raise ValueError("project did not supply the requested verification interpreter")
                argument = argument.replace(old, new)
            command.append(argument)
        # Compatibility with the existing burst RUN script contract; values
        # describe this allocation, never its coordinator's /local paths.
        environment.update(PY=environment.get("PROJECT_PYTHON", ""), CLONE=work,
                           DATA=environment.get("LSMS_DATA_DIR", ""), JOBDIR=str(directory))
        command_script = ('set +e\n' + ' '.join(shlex.quote(arg) for arg in command)
                          + ' > ' + shlex.quote(str(directory / 'verification.stdout.log'))
                          + ' 2> ' + shlex.quote(str(directory / 'verification.stderr.log')) + '\n'
                          + 'printf "%s\\n" "$?" > ' + shlex.quote(str(directory / 'command-exit.txt')) + '\nexit 0\n')
        project.private_bytes(directory / "verify-command.sh", command_script.encode("utf-8"))
        record.update(phase="verification", command=command)
        publish()
        worker.run_script(str(directory / "verify-command.sh"), environment, request["timeout_seconds"],
                          cancellation=stopped, heartbeat=publish, cwd=work)
        record["command_exit_code"] = int((directory / "command-exit.txt").read_text().strip())
        record["state"] = "succeeded" if record["command_exit_code"] == 0 else "failed"
    except Exception as exc:
        record.update(state="interrupted" if stopped[0] else "failed", reason=str(exc), failure_type=type(exc).__name__)
    finally:
        cleanup_complete = ownership is not None
        if ownership is not None:
            try:
                worker.stop_allocation(ownership, 5)
            except Exception as exc:
                cleanup_complete = False
                record.update(state="failed", cleanup_error=str(exc))
        if source_config is not None and supervisor_environment is not None:
            try:
                if not cleanup_complete:
                    raise ValueError("source identity cannot be verified before owned processes stop")
                if project.digest_file(Path(work) / ".git" / "config") != source_config:
                    raise ValueError("verification command changed Git configuration")
                record["source_identity"] = source_identity(work, request["commit"], supervisor_environment)
                if record["source_identity"]["state"] != "matched":
                    record.update(state="failed", reason="verification source changed from the frozen commit")
            except Exception as exc:
                record.update(state="failed", source_identity={"state": "unverified", "expected_commit": request["commit"],
                                                              "reason": str(exc)})
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if environment is not None:
            try:
                record["evidence"] = artifacts.publish_evidence(str(directory), required=request.get("required_evidence", []))
                if record["evidence"]["state"] != "complete":
                    record["state"] = "failed"
            except Exception as exc:
                record.update(state="failed", evidence_error=str(exc))
        record.update(phase="finished", finished_at=time.time())
        if stopped[0]:
            record["state"] = "interrupted"
            code = 128 + stopped[0]
        elif record["state"] == "succeeded":
            code = 0
        else:
            code = record["command_exit_code"] or 70
        record["supervisor_exit_code"] = code
        publish()
        atomic_json(str(directory / "verification-result.json"), record)
        os.umask(old_umask)
    return code


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: dispatch_verify.py FROZEN_VERIFICATION_DIRECTORY")
    try:
        sys.exit(run(os.path.abspath(sys.argv[1])))
    except Exception as exc:
        sys.stderr.write("verification request refused (%s)\n" % type(exc).__name__)
        sys.exit(70)
