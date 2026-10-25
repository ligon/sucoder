"""Non-destructive, remotely executed mirror initialization.

The protocol is deliberately small: only a successful command ending in one
of the two markers authorizes the caller to continue. SSH failures, shell
noise, inaccessible paths and broken repositories are not absence proofs.
"""

VERIFY_LAUNCH_MIRROR_SH = r'''set -eu
export GIT_ALLOW_PROTOCOL= GIT_NO_LAZY_FETCH=1 GIT_OPTIONAL_LOCKS=0 GIT_TERMINAL_PROMPT=0
path=$1
if [ ! -e "$path" ]; then
    echo "mirror path is absent" >&2
    exit 10
fi
[ -d "$path" ] && [ -e "$path/.git" ] || {
    echo "mirror is not a working repository" >&2; exit 11;
}
# Ref resolution must never hydrate a partial clone or refresh its index.
git -c protocol.allow=never -c protocol.file.allow=never \
    -c core.fsmonitor=false --no-optional-locks -C "$path" \
    rev-parse --verify --quiet 'HEAD^{commit}' || {
    echo "mirror HEAD cannot be verified offline" >&2; exit 11;
}
'''


VERIFY_PUSH_PATHS_PY = r'''
import json
import os
import subprocess
import sys

root = os.path.abspath(sys.argv[1])
payload = json.load(sys.stdin)
os.chdir(root)
env = dict(os.environ, GIT_ALLOW_PROTOCOL="", GIT_NO_LAZY_FETCH="1",
           GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
index = subprocess.check_output(
    ["git", "-c", "core.fsmonitor=false", "--no-optional-locks", "ls-files", "-t", "-z"],
    env=env,
).decode("utf-8", "surrogateescape").split("\0")
tracked = {entry[2:] for entry in index if entry}
skipped = {entry[2:] for entry in index if entry.startswith("S ")}
for status, name in payload["changes"]:
    if os.path.isabs(name) or ".." in name.split("/"):
        sys.exit("Invalid incoming path")
    path = os.path.join(root, name)
    added = status == "A" and (payload["known_base"] or name not in tracked)
    if os.path.lexists(path) and (added or name in skipped):
        sys.exit("Refusing push: local file would be overwritten: " + name)
    if status != "D":
        parent = os.path.dirname(path)
        while parent != root:
            if os.path.islink(parent) or (os.path.exists(parent) and not os.path.isdir(parent)):
                sys.exit("Refusing push: local path blocks incoming files: " + parent)
            parent = os.path.dirname(parent)
'''


INITIALIZE_MIRROR_SH = r'''set -eu
umask 077
path=$1
base=$2
fail() { echo "sucoder: $*; existing files left untouched" >&2; exit 1; }
case "$path" in /*) ;; *) fail "mirror path must be absolute" ;; esac
[ "$path" != / ] || fail "refusing root as a mirror"
[ ! -L "$path" ] || fail "mirror path is a symlink: $path"
if [ ! -e "$path" ]; then
    mkdir -p -- "$(dirname -- "$path")"
    # Exclusive creation: never remove a directory that won this race.
    mkdir -- "$path" || fail "could not exclusively create $path; retry"
fi
[ -d "$path" ] || fail "mirror is not a directory: $path"
cd -- "$path" || fail "cannot enter $path"
if [ -e .git ] || [ -L .git ]; then
    [ ! -L .git ] || fail "mirror .git is a symlink"
    top=$(git rev-parse --show-toplevel) || fail "invalid repository at $path"
    [ "$top" = "$(pwd -P)" ] || fail "Git resolved a different working tree"
    echo SUCODER_MIRROR_EXISTING
else
    # Include hidden files, and never mistake ancestor Git discovery for a
    # repository at this path. Bare repos and arbitrary directories are kept.
    shopt -s nullglob dotglob
    entries=(*)
    [ ${#entries[@]} -eq 0 ] || fail "nonempty directory has no .git: $path"
    mkdir .sucoder-init.lock || fail "another initialization is in progress"
    trap 'rmdir .sucoder-init.lock' EXIT
    git init -b "$base" >&2 || fail "git initialization failed"
    echo SUCODER_MIRROR_CREATED
fi
'''
