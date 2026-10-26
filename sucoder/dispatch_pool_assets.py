"""Receiving-side immutable allocation asset reuse (Python 3.6 stdlib)."""
import hashlib
import json
import os
import stat
import subprocess
import sys
import time


def validate_local_source(gitdir):
    """Local-copy transport is only for owned, full, independent Git objects."""
    if os.path.realpath(gitdir) != gitdir or os.lstat(gitdir).st_uid != os.getuid():
        raise ValueError("pool local-copy Git metadata must be nonredirected and owned")
    for name in ("shallow", "objects/info/alternates", "objects/info/http-alternates"):
        if os.path.lexists(os.path.join(gitdir, name)):
            raise ValueError("pool local-copy sources must be full and have no alternates")
    objects = os.path.join(gitdir, "objects")
    if os.path.islink(objects) or not os.path.isdir(objects):
        raise ValueError("pool local-copy object directory must not be redirected")
    for parent, directories, files in os.walk(objects):
        for name in directories + files:
            path = os.path.join(parent, name)
            info = os.lstat(path)
            if name.endswith(".promisor"):
                raise ValueError("pool local-copy source must not contain promisor objects")
            if (info.st_uid != os.getuid() or
                    not (stat.S_ISDIR(info.st_mode) if name in directories else stat.S_ISREG(info.st_mode))):
                raise ValueError("pool local-copy objects must be owned regular files/directories, without symlinks")


def verify_copy(source_gitdir, destination_gitdir):
    """Prove local transport made separate files, not alternates or hardlinks."""
    validate_local_source(source_gitdir)
    validate_local_source(destination_gitdir)
    objects = os.path.join(destination_gitdir, "objects")
    count = 0
    for parent, directories, files in os.walk(objects):
        for name in files:
            destination = os.path.join(parent, name)
            if os.stat(destination).st_nlink != 1:
                raise ValueError("pool private source copy retained a hardlinked object")
            original = os.path.join(source_gitdir, os.path.relpath(destination, destination_gitdir))
            if os.path.exists(original):
                left, right = os.stat(original), os.stat(destination)
                if (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino):
                    raise ValueError("pool private source copy retained a hardlinked object")
                count += 1
    return {"independent_object_files": count, "hardlinks": False, "alternates": False, "shallow": False}


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
    validate_local_source(source)
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


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--verify-copy":
        sys.exit("internal pool helper: expected --verify-copy SOURCE_GIT DESTINATION_GIT")
    print(json.dumps(verify_copy(sys.argv[2], sys.argv[3]), sort_keys=True))
