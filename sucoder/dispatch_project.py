"""Frozen project preparation; Python 3.6, standard library only.

The bootstrap is trusted, explicitly selected code. Its exports are returned
through a bounded JSON receipt rather than inherited from a child shell.
This contract is shared by model workers and deterministic verification jobs.
See docs/dispatch-project-ledger.org sections 2-5.
"""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import shlex
import stat
import sys
import time

try:
    from .dispatch_protocol import ENVIRONMENT_ALLOWLIST, atomic_json
except ImportError:
    from dispatch_protocol import ENVIRONMENT_ALLOWLIST, atomic_json

VERSION = 1
MAX_JSON = 1024 * 1024
PROTECTED = frozenset((
    "HOME", "USER", "LOGNAME", "SHELL", "PATH", "TMPDIR", "PYTHONHOME",
    "PYTHONPYCACHEPREFIX", "PYTHONUSERBASE",
    "LD_PRELOAD", "LD_LIBRARY_PATH", "BASH_ENV", "ENV", "SHELLOPTS",
    "NODE_OPTIONS", "NODE_PATH", "NODE_EXTRA_CA_CERTS",
    "UV_CACHE_DIR", "PIP_CACHE_DIR", "npm_config_cache",
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
)) | frozenset(ENVIRONMENT_ALLOWLIST)


def environment_key(key):
    if (not isinstance(key, str) or len(key) > 128 or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key)
            or key in PROTECTED
            or key.startswith(("SUCODER_", "SLURM_", "SPANK_", "CODEX_", "CLAUDE_", "GIT_", "XDG_",
                               "OPENAI_", "ANTHROPIC_", "LD_", "DYLD_"))):
        raise ValueError("project profile cannot override protected environment key: " + str(key))
    return key


def relative_name(value):
    if (not isinstance(value, str) or not value or "\\" in value or "\x00" in value
            or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts
            or str(PurePosixPath(value)) != value or value == "."):
        raise ValueError("project paths require normalized relative POSIX names")
    return value


def digest_file(path):
    value = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def bounded_json(path):
    with open(str(path), "rb") as stream:
        content = stream.read(MAX_JSON + 1)
    if len(content) > MAX_JSON:
        raise ValueError("project JSON exceeds its one MiB limit")
    value = json.loads(content.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("project JSON must be an object")
    return value


def private_bytes(path, content):
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)


def read_profile(path, shared_path=None):
    """Validate all inputs before dispatch can create files or contact Slurm."""
    path = Path(path).expanduser().resolve()
    profile = bounded_json(path)
    keys = {"schema_version", "bootstrap", "timeout_seconds", "environment", "local_directories",
            "assets", "credentials", "credential_environment", "output_environment", "thread_limits",
            "require_absent_environment"}
    if set(profile) - keys or type(profile.get("schema_version")) is not int or profile["schema_version"] != VERSION:
        raise ValueError("unsupported project profile schema or field")
    timeout = profile.get("timeout_seconds", 120)
    if type(timeout) not in (int, float) or timeout <= 0 or timeout > 86400 or not math.isfinite(timeout):
        raise ValueError("project timeout must be finite, positive and at most one day")
    bootstrap = profile.get("bootstrap")
    if not isinstance(bootstrap, str):
        raise ValueError("project bootstrap must name a shell script")
    bootstrap_path = (path.parent / bootstrap).resolve()
    with bootstrap_path.open("rb") as stream:
        bootstrap_bytes = stream.read(MAX_JSON + 1)
    if not bootstrap_bytes or len(bootstrap_bytes) > MAX_JSON:
        raise ValueError("project bootstrap must be nonempty and at most one MiB")
    frozen = {"schema_version": VERSION, "timeout_seconds": timeout,
              "bootstrap": "project-bootstrap.sh", "bootstrap_sha256": hashlib.sha256(bootstrap_bytes).hexdigest()}
    for field in ("environment", "local_directories", "assets", "credentials", "credential_environment", "thread_limits"):
        if not isinstance(profile.get(field, {}), dict):
            raise ValueError("project " + field + " must be an object")
    environment = profile.get("environment", {})
    if any(not isinstance(v, str) or "\x00" in v or len(v) > 8192 for v in environment.values()):
        raise ValueError("project environment values must be bounded strings without NUL")
    local = profile.get("local_directories", {})
    for field in (environment, local, profile.get("credential_environment", {})):
        for key in field:
            environment_key(key)
    if set(environment) & set(local):
        raise ValueError("project environment and local directories overlap")
    for name in local.values():
        relative_name(name)
    frozen.update(environment=environment, local_directories=local)
    assets = {}
    for name, asset in profile.get("assets", {}).items():
        relative_name(name)
        if not isinstance(asset, dict) or set(asset) != {"path", "sha256"}:
            raise ValueError("project assets require path and sha256")
        if not isinstance(asset["path"], str) or not os.path.isabs(os.path.expanduser(asset["path"])):
            raise ValueError("project asset paths must be absolute shared files")
        asset_path = Path(asset["path"]).expanduser().resolve()
        if shared_path is not None:
            asset_path = shared_path(asset_path, "project asset")
        if not asset_path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", str(asset["sha256"])):
            raise ValueError("project asset requires an existing file and SHA256")
        if digest_file(asset_path) != asset["sha256"]:
            raise ValueError("project asset digest mismatch: " + name)
        assets[name] = {"path": str(asset_path), "sha256": asset["sha256"]}
    frozen["assets"] = assets
    secrets = {}
    for name, source in profile.get("credentials", {}).items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("project credential names require letters, numbers, underscore or hyphen")
        if not isinstance(source, str):
            raise ValueError("project credential source must be a file path")
        source = Path(source).expanduser()
        if not source.is_absolute():
            source = path.parent / source
        fd = os.open(str(source), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError("project credential must be a private regular file owned by this uid")
            content = stream.read(MAX_JSON + 1)
        if len(content) > MAX_JSON:
            raise ValueError("project credential file exceeds one MiB")
        secrets[name] = content
    bindings = profile.get("credential_environment", {})
    if any(name not in secrets for name in bindings.values()):
        raise ValueError("project credential environment names an undeclared credential")
    if (set(bindings) & (set(environment) | set(local))):
        raise ValueError("project credential environment overlaps other bindings")
    outputs = profile.get("output_environment", [])
    if not isinstance(outputs, list) or any(not isinstance(k, str) for k in outputs) or len(set(outputs)) != len(outputs):
        raise ValueError("project output_environment must contain unique environment names")
    for key in outputs:
        environment_key(key)
    if set(outputs) & (set(local) | set(bindings)):
        raise ValueError("bootstrap cannot replace local or credential bindings")
    limits = profile.get("thread_limits", {})
    for key, value in limits.items():
        if key not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment_key(key)
        if type(value) is not int or value < 1:
            raise ValueError("project thread limits must be positive integers")
    if set(limits) & (set(environment) | set(local) | set(bindings) | set(outputs)):
        raise ValueError("project thread limits overlap another binding")
    frozen.update(credentials={name: "project-secret-" + name for name in secrets},
                  credential_environment=bindings, output_environment=outputs, thread_limits=limits)
    absent = profile.get("require_absent_environment", [])
    if not isinstance(absent, list) or any(key not in ENVIRONMENT_ALLOWLIST for key in absent):
        raise ValueError("project absent-environment guard must name supported provider keys")
    frozen["require_absent_environment"] = absent
    canonical = json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"profile": frozen, "content": canonical, "sha256": hashlib.sha256(canonical).hexdigest(),
            "bootstrap": bootstrap_bytes, "secrets": secrets}


def stage_profile(prepared, directory):
    """Freeze helper/profile/reader bytes in the dispatch's private directory."""
    directory = Path(directory)
    private_bytes(directory / "project.json", prepared["content"])
    private_bytes(directory / "project-bootstrap.sh", prepared["bootstrap"])
    command = ["python3", str(directory / "dispatch_project.py"), "--copy-inputs", str(directory)]
    private_bytes(directory / "project-inputs.sh", ("exec " + " ".join(shlex.quote(p) for p in command)
                                                   + ' "$SUCODER_PROJECT_ROOT"\n').encode("utf-8"))
    names = ["project.json", "project-bootstrap.sh", "project-inputs.sh"]
    for name, content in prepared["secrets"].items():
        filename = "project-secret-" + name
        private_bytes(directory / filename, content)
        names.append(filename)
    return names


def inside(path, root):
    root = os.path.realpath(str(root))
    return os.path.realpath(str(path)).startswith(root.rstrip(os.sep) + os.sep)


def local_directory(path, root):
    if not inside(path, root) or os.path.realpath(str(path)) != str(path):
        raise ValueError("project writable path escapes receiving allocation or traverses a symlink")
    os.makedirs(str(path), mode=0o700, exist_ok=True)
    return str(path)


def copy_inputs(directory, project):
    """Copy/hash approved inputs in a supervised child, including slow I/O."""
    directory, project = Path(directory), Path(project)
    frozen = bounded_json(directory / "project.json")
    # The parent has verified and created the real receiving root. Never accept
    # a symlinked staging directory even when this helper is called directly.
    if not project.is_absolute() or str(project.resolve()) != str(project):
        raise ValueError("project input root must be an absolute real directory")
    assets = project / "assets"
    local_directory(assets, project)
    credentials = project / "credentials"
    local_directory(credentials, project)
    asset_receipts = {}
    for name, asset in frozen["assets"].items():
        target = assets / relative_name(name)
        local_directory(target.parent, project)
        # Copy then hash the actual local bytes. A shared file changing after
        # preflight cannot silently substitute the image used on this node.
        with open(asset["path"], "rb") as source:
            fd = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
        if digest_file(target) != asset["sha256"]:
            raise ValueError("receiving project asset digest mismatch: " + name)
        target.chmod(0o400)
        asset_receipts[name] = {"path": str(target), "sha256": asset["sha256"]}
    secret_paths = {}
    for name, filename in frozen["credentials"].items():
        if not re.fullmatch(r"project-secret-[A-Za-z0-9_-]+", filename):
            raise ValueError("invalid frozen credential path")
        target = credentials / name
        private_bytes(target, (directory / filename).read_bytes())
        os.unlink(str(directory / filename))
        secret_paths[name] = str(target)
    atomic_json(str(project / "inputs.json"), {"assets": asset_receipts, "credentials": secret_paths})


def prepare(directory, work, root, environment, run_helper, cpus, heartbeat=None, cancellation=None):
    """Prepare explicit project bindings after the caller verifies source/root.

    run_helper is the existing supervised run_script; deterministic verification
    uses the same callback instead of introducing a second supervision protocol.
    """
    directory, work, root = Path(directory), Path(work), Path(root)
    frozen = bounded_json(directory / "project.json")
    if frozen.get("schema_version") != VERSION:
        raise ValueError("unsupported frozen project profile")
    if digest_file(directory / frozen["bootstrap"]) != frozen["bootstrap_sha256"]:
        raise ValueError("frozen project bootstrap digest mismatch")
    if any(environment.get(key) for key in frozen.get("require_absent_environment", [])):
        raise ValueError("project profile refuses an ambient provider credential that could grant data access")
    deadline = time.monotonic() + frozen["timeout_seconds"]
    project = root / "projects" / environment["SUCODER_DISPATCH_ID"]
    local_directory(project, root)
    assets, credentials = project / "assets", project / "credentials"
    substitutions = {"worktree": str(work), "project_root": str(project),
                     "assets": str(assets), "credentials": str(credentials), "local_root": str(root)}
    result_environment = dict(environment)
    for key, value in frozen["environment"].items():
        environment_key(key)
        result_environment[key] = re.sub(r"\{(worktree|project_root|assets|credentials|local_root)\}",
                                         lambda match: substitutions[match.group(1)], value)
    for key, name in frozen["local_directories"].items():
        environment_key(key)
        result_environment[key] = local_directory(project / relative_name(name), root)
    for key, name in frozen["credential_environment"].items():
        environment_key(key)
        result_environment[key] = str(credentials / name)
    for key, count in frozen["thread_limits"].items():
        if type(count) is not int or not 1 <= count <= cpus:
            raise ValueError("project thread limit exceeds receiving CPU budget")
        result_environment[key] = str(count)
    receipt_path = project / "runtime.json"
    bootstrap_environment = {key: value for key, value in result_environment.items()
                             if key not in ENVIRONMENT_ALLOWLIST}
    bootstrap_environment.update(SUCODER_PROJECT_WORKTREE=str(work), SUCODER_PROJECT_ROOT=str(project),
                                 SUCODER_PROJECT_RESULT=str(receipt_path), SUCODER_PROJECT_ASSETS=str(assets))
    # Reuse supervision for asset copying too: a blocked shared-file read must
    # obey the same cancellation, deadline and owned-process cleanup contract.
    run_helper(str(directory / "project-inputs.sh"), bootstrap_environment, max(0, deadline - time.monotonic()),
               cancellation=cancellation, heartbeat=heartbeat, cwd=str(work))
    run_helper(str(directory / frozen["bootstrap"]), bootstrap_environment, max(0, deadline - time.monotonic()),
               cancellation=cancellation, heartbeat=heartbeat, cwd=str(work))
    info = receipt_path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("project runtime receipt must be a private regular file")
    receipt = bounded_json(receipt_path)
    if receipt.get("schema_version") != VERSION or receipt.get("state") != "ready":
        raise ValueError("project bootstrap did not report a ready runtime")
    if set(receipt) - {"schema_version", "state", "environment", "path_prepend", "runtime"}:
        raise ValueError("unknown project runtime receipt field")
    returned = receipt.get("environment", {})
    if not isinstance(returned, dict) or set(returned) - set(frozen["output_environment"]):
        raise ValueError("project bootstrap returned undeclared environment keys")
    for key, value in returned.items():
        environment_key(key)
        if not isinstance(value, str) or "\x00" in value or len(value) > 8192:
            raise ValueError("invalid project bootstrap environment value")
        result_environment[key] = value
    paths = receipt.get("path_prepend", [])
    if (not isinstance(paths, list) or any(not isinstance(p, str) or os.pathsep in p or "\x00" in p
            or not os.path.isabs(p)
            or not inside(p, root) or not os.path.isdir(p) for p in paths)):
        raise ValueError("project PATH prefixes must be receiving-node directories")
    if any(os.path.lexists(os.path.join(path, name)) for path in paths
           for name in ("node", "codex", "claude", "git", "bash", "sh")):
        raise ValueError("project PATH cannot shadow the native agent or supervisor toolchain")
    result_environment["PATH"] = os.pathsep.join(paths + [environment["PATH"]])
    public_receipt = {"schema_version": VERSION, "state": "ready", "worktree": str(work),
                      "project_root": str(project), "assets": bounded_json(project / "inputs.json")["assets"],
                      "runtime": receipt.get("runtime", {}), "thread_limits": frozen["thread_limits"]}
    # This private nonsecret receipt survives job-local teardown alongside proof.
    atomic_json(str(directory / "project-runtime.json"), public_receipt)
    evidence = directory / "evidence"
    if evidence.is_symlink() or evidence.resolve() != evidence:
        raise ValueError("project evidence directory must not traverse a symlink")
    evidence.mkdir(mode=0o700, exist_ok=True)
    info = evidence.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("project evidence directory must be private and owned")
    atomic_json(str(evidence / "project-runtime.json"), public_receipt)
    return result_environment, public_receipt


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--copy-inputs":
        sys.exit("internal project helper: expected --copy-inputs DIRECTORY PROJECT_ROOT")
    copy_inputs(sys.argv[2], sys.argv[3])
