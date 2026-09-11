"""Non-destructive, remotely executed mirror initialization.

The protocol is deliberately small: only a successful command ending in one
of the two markers authorizes the caller to continue. SSH failures, shell
noise, inaccessible paths and broken repositories are not absence proofs.
"""

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
