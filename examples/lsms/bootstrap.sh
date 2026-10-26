#!/bin/bash
# Explicit LSMS runtime setup; frozen by --project-profile, never sourced.
set -euo pipefail
umask 077
: "${SUCODER_PROJECT_WORKTREE:?}" "${SUCODER_LOCAL_ROOT:?}" "${LSMS_VENV_SHARE:?}"
: "${LSMS_DATA_DIR:?}" "${LSMS_CONFIG_DIR:?}" "${LSMS_COUNTRIES_ROOT:?}"
cd "$SUCODER_PROJECT_WORKTREE"
test -f "$LSMS_VENV_SHARE/.venv.sqfs"
test -f bin/savio_venv.sh
# The project's mounter has fast paths for existing mounts/extractions. Refuse
# those here so an older image cannot stand in for the explicitly pinned asset.
runtime="$SUCODER_LOCAL_ROOT/venv_img"
if test -L "$runtime" || { test -d "$runtime" && test -n "$(ls -A "$runtime")"; }; then
    echo 'LSMS profile requires a fresh receiving-allocation runtime' >&2
    exit 1
fi
# unsquashfs is under /usr/sbin on Savio; this affects only setup, not the
# model's baseline toolchain. The approved image already exists, so the
# mounter's missing-image/writable-master fallback cannot be selected.
export PATH="/usr/sbin:/sbin:$PATH"
bash bin/savio_venv.sh mount
test -x .venv/bin/python
export LSMS_PROJECT_RUNTIME="$runtime"
.venv/bin/python - <<'PY'
import json, os, pathlib, re, stat, sys
import lsms_library
from lsms_library.paths import countries_root, data_root
from lsms_library.config import config_path, s3_creds_path
from lsms_library import local_tools

work = pathlib.Path(os.environ['SUCODER_PROJECT_WORKTREE']).resolve()
root = pathlib.Path(os.environ['SUCODER_LOCAL_ROOT']).resolve()
runtime = pathlib.Path(os.environ['LSMS_PROJECT_RUNTIME']).resolve()
package = pathlib.Path(lsms_library.__file__).resolve()
countries = pathlib.Path(countries_root()).resolve()
if not str(package).startswith(str(work) + os.sep):
    raise SystemExit('LSMS import is outside the pinned worker checkout')
if countries != work / 'lsms_library' / 'countries':
    raise SystemExit('LSMS countries root differs from the pinned checkout')
if pathlib.Path(sys.prefix).resolve() != runtime:
    raise SystemExit('Python prefix does not identify the newly prepared runtime')
metadata = countries / '.dvc' / 'tmp'
for path in (pathlib.Path(data_root()), pathlib.Path(config_path()).parent,
             pathlib.Path(local_tools._DVC_CACHE_DIR), metadata):
    if not str(path.resolve()).startswith(str(root) + os.sep):
        raise SystemExit('LSMS writable data/configuration is outside the receiving allocation')
shared_l1 = os.environ.get('LSMS_PROFILE_SHARED_L1')
l1_receipt = {'policy': 'private', 'linked_blobs': 0}
if shared_l1:
    # Reuse immutable blob FILES, with every directory and mutable cache entry
    # private. Never symlink a writable directory into the shared L1 cache.
    import yaml
    shared = pathlib.Path(shared_l1).resolve()
    selected = json.loads(os.environ['LSMS_PROFILE_COUNTRIES'])
    hashes = set()
    for country in selected:
        if not isinstance(country, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', country):
            raise SystemExit('invalid explicit L1 country scope')
        for sidecar in (countries / country).rglob('*.dvc'):
            entry = yaml.safe_load(sidecar.read_text()) or {}
            for output in entry.get('outs', []):
                digest = output.get('md5', '')
                if re.fullmatch(r'[0-9a-f]{32}', digest):
                    hashes.add(digest)
    linked = 0
    cache = pathlib.Path(local_tools._DVC_CACHE_DIR)
    for digest in sorted(hashes):
        candidates = [shared / 'files' / 'md5' / digest[:2] / digest[2:], shared / digest[:2] / digest[2:]]
        source = next((p.resolve() for p in candidates if p.is_file()), None)
        if source is None:
            continue
        if not str(source).startswith(str(shared) + os.sep) or source.stat().st_mode & 0o222:
            raise SystemExit('shared L1 projection requires immutable blobs inside the approved root')
        for target in (cache / 'files' / 'md5' / digest[:2] / digest[2:], cache / digest[:2] / digest[2:]):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise SystemExit('L1 projection requires a fresh private cache')
            target.symlink_to(source)
        linked += 1
    l1_receipt = {'policy': 'immutable-file-projection', 'root': str(shared),
                  'countries': selected, 'requested_blobs': len(hashes), 'linked_blobs': linked}
generation = runtime / 'VENV_GENERATION'
receipt = {'schema_version': 1, 'state': 'ready',
           'environment': {'PROJECT_PYTHON': str(work / '.venv' / 'bin' / 'python')},
           'path_prepend': [str(runtime / 'bin')],
           'runtime': {'interpreter': sys.executable, 'prefix': sys.prefix,
                       'python_version': sys.version, 'package': str(package),
                       'countries_root': str(countries), 'data_root': str(data_root()),
                       'dvc_metadata': str(metadata), 'l1_cache': str(local_tools._DVC_CACHE_DIR),
                       'shared_l1': l1_receipt,
                       'config_path': str(config_path()), 'reader_file_present': s3_creds_path().is_file(),
                       'generation': generation.read_text() if generation.is_file() else None,
                       'fresh_runtime': True}}
with open(os.environ['SUCODER_PROJECT_RESULT'], 'w') as f:
    json.dump(receipt, f, sort_keys=True)
PY
