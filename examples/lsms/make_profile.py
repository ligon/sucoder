#!/usr/bin/env python3
"""Create an explicit LSMS profile; never discover or print credentials."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reader', type=Path, help='Explicit approved reader file; omit for hermetic work.')
    parser.add_argument('--shared-l1', type=Path, help='Approved read-only content-addressed blobs; never writable metadata.')
    parser.add_argument('--country', action='append', default=[], help='Country scope for the optional L1 projection.')
    args = parser.parse_args()
    if args.shared_l1 and (not args.reader or not args.country):
        parser.error('--shared-l1 requires an approved --reader and an explicit --country scope')
    image = args.image.expanduser().resolve()
    digest = hashlib.sha256()
    with image.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(data)
    profile = {
        'schema_version': 1, 'bootstrap': str(Path(__file__).with_name('bootstrap.sh').resolve()),
        'timeout_seconds': 300,
        'assets': {'runtime/.venv.sqfs': {'path': str(image), 'sha256': digest.hexdigest()}},
        'environment': {'PYTHONPATH': '{worktree}', 'PYTHONSAFEPATH': '1', 'LSMS_SKIP_AUTH': '1',
                        'LSMS_COUNTRIES_ROOT': '{worktree}/lsms_library/countries',
                        'LSMS_VENV_SHARE': '{assets}/runtime',
                        'LSMS_S3_CREDS': '{project_root}/config/no-reader',
                        'AWS_EC2_METADATA_DISABLED': 'true'},
        'local_directories': {'LSMS_DATA_DIR': 'data', 'LSMS_CONFIG_DIR': 'config'},
        'output_environment': ['PROJECT_PYTHON'],
        'thread_limits': {'OMP_NUM_THREADS': 1, 'OPENBLAS_NUM_THREADS': 1,
                          'MKL_NUM_THREADS': 1, 'LSMS_BUILD_WORKERS': 1},
        'require_absent_environment': ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN'],
    }
    if args.reader:
        profile['environment'].pop('LSMS_S3_CREDS')
        profile['credentials'] = {'reader': str(args.reader.expanduser().resolve())}
        profile['credential_environment'] = {'LSMS_S3_CREDS': 'reader'}
    if args.shared_l1:
        profile['environment']['LSMS_PROFILE_SHARED_L1'] = str(args.shared_l1.expanduser().resolve())
        profile['environment']['LSMS_PROFILE_COUNTRIES'] = json.dumps(args.country)
    fd = os.open(str(args.output), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(profile, f, indent=2, sort_keys=True)
        f.write('\n')
    print(str(args.output))


if __name__ == '__main__':
    main()
