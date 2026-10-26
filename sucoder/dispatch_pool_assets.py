"""Receiving-side immutable allocation asset reuse (Python 3.6 stdlib)."""
import hashlib
import json
import os
import stat
import subprocess
import time


def _identity(path, directory=False):
    info = os.lstat(path)
    if (os.path.realpath(path) != path or info.st_uid != os.getuid()
            or (not stat.S_ISDIR(info.st_mode) if directory else not stat.S_ISREG(info.st_mode))):
        raise ValueError("pool asset is not a nonredirected owned file/directory")
    if not directory and info.st_mode & 0o222:
        raise ValueError("pool image must have no write permission bits")
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode]


def verify(execution, commit, root, environment):
    assets = execution["assets"]
    allocation_root = root.split("/workers/", 1)[0]
    for name in ("source", "image"):
        path = assets[name]
        if not path.startswith(allocation_root + "/") or os.path.realpath(path) != path:
            raise ValueError("pool assets must be on receiving allocation-local disk")
    source, image = assets["source"], assets["image"]
    env = dict(environment, GIT_NO_LAZY_FETCH="1", GIT_ALLOW_PROTOCOL="", GIT_OPTIONAL_LOCKS="0")
    def git(*args):
        return subprocess.check_output(["git", "-C", source] + list(args), env=env,
                                       stderr=subprocess.DEVNULL, timeout=60).decode().strip()
    if git("rev-parse", "--is-bare-repository") != "true":
        raise ValueError("pool source must be a full private bare repository")
    if os.path.exists(os.path.join(source, "shallow")) or os.path.lexists(os.path.join(source, "objects/info/alternates")):
        raise ValueError("pool source must be self-contained, not shallow or alternate-backed")
    config = git("config", "--local", "--list").lower()
    if "partialclone" in config or ".promisor=true" in config:
        raise ValueError("promisor pool sources are unsupported")
    # Cache is a frozen base-object source, not a moving publication target.
    # prepare.sh fetches the exact requested SHA from this worker's private,
    # self-contained results.git; freshly authored candidates need not already
    # exist in the allocation cache. Never mutate or hydrate the shared cache.
    refs = hashlib.sha256(git("show-ref").encode()).hexdigest()
    identity = dict(source=source, image=image, image_sha256=assets["image_sha256"],
                    source_identity=_identity(source, True), image_identity=_identity(image), refs_sha256=refs)
    receipt = os.path.join(os.path.dirname(image), ".sucoder-pool-" + execution["pool_id"] + ".json")
    lock = receipt + ".lock"
    until = time.monotonic() + 90
    while True:
        try:
            os.mkdir(lock, 0o700)
            break
        except FileExistsError:
            if time.monotonic() >= until:
                raise ValueError("pool asset verification lock remains held; no stale-lock stealing")
            time.sleep(0.1)
    try:
        if os.path.exists(receipt):
            with open(receipt) as stream:
                if json.load(stream) != identity:
                    raise ValueError("verified immutable pool assets changed")
        else:
            digest = hashlib.sha256()
            with open(image, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != assets["image_sha256"]:
                raise ValueError("pool image SHA256 mismatch")
            git("fsck", "--connectivity-only", "--no-reflogs")
            if identity["image_identity"] != _identity(image):
                raise ValueError("pool image changed during verification")
            fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(identity, stream, sort_keys=True)
        return dict(identity, verification_receipt=receipt, commit=commit)
    finally:
        os.rmdir(lock)
