#!/usr/bin/env bash
set -euo pipefail

remote=${1:-coder}
prefix=${2:-coder}

git fetch "${remote}"
git for-each-ref "refs/remotes/${remote}/${prefix}/" --format='%(refname:strip=2)'
