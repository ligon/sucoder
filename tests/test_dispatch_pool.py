"""Existing-allocation pool tests: disposable Git, fake Slurm, no native model."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import pytest

from sucoder import dispatch, dispatch_pool as pool, dispatch_pool_assets as assets
from sucoder import dispatch_worker as worker
from sucoder.local_tier import build_worker_prepare_script, build_worker_snapshot_script
from tests.test_dispatch import source  # existing disposable source fixture, never a real project


def facts():
    return dict(JobId='234', NodeList='n1', Account='a', Partition='p', QOS='q', StartTime='2020-01-01T01:00:00',
                verified_cpus=8, verified_mem_mib=48 * 1024, verified_end=time.time() + 8000)


@pytest.fixture
def attached(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_CLUSTER_NAME', 'cluster')
    monkeypatch.setattr(dispatch, '_shared_path', lambda path, label: path.resolve())
    monkeypatch.setattr(pool, '_job', lambda *args, **kwargs: facts())
    path = tmp_path / 'pool'
    pool.attach(path, 234, node='n1', cpus=4, mem='32G', submit_before=time.time() + 2000,
                worker_deadline=time.time() + 4000, expires=time.time() + 5000,
                source='/local/job234/assets/source.git', image='/local/job234/assets/image.sqfs',
                image_sha256='a' * 64)
    return path


def request(source):
    return {key: source[key] for key in ('repo', 'ref', 'task_file', 'agent', 'executable')}


def test_attach_rejects_wrong_uid_or_multi_node_scheduler(monkeypatch):
    monkeypatch.setattr(dispatch, '_run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout=(
        'JobId=234 UserId=x(%d) JobState=RUNNING NodeList=n1 NumNodes=2 AllocTRES=cpu=8,mem=48G EndTime=2099-01-01T00:00:00' % os.getuid())))
    with pytest.raises(ValueError, match='single-node'):
        pool._job(234)


def test_attach_binds_finite_budget_and_private_metadata(attached):
    profile = json.loads((attached / 'pool.json').read_text())
    assert profile['cpus'] == 4 and profile['mem_mib'] == 32768
    assert profile['uid'] == os.getuid() and profile['node'] == 'n1'
    assert attached.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize('cpus,mem', [(5, '1G'), (1, '33G')])
def test_over_budget_refuses_before_submission(attached, source, monkeypatch, cpus, mem):
    monkeypatch.setattr(dispatch, 'submit', lambda *a, **k: pytest.fail('overcommit submitted'))
    with pytest.raises(ValueError, match='budget exhausted'):
        pool.submit(attached, **request(source), cpus=cpus, mem=mem)


def test_unknown_reserves_aggregate_budget_and_never_retries(attached, source, monkeypatch):
    seen = []
    def submit(*a, **k):
        seen.append(k['_execution'])
        return dict(directory=str(source['repo'] / 'unknown'), state='submission_unknown')
    monkeypatch.setattr(dispatch, 'submit', submit)
    for _ in range(4):
        pool.submit(attached, **request(source), cpus=1, mem='8G')
    with pytest.raises(ValueError, match='budget exhausted'):
        pool.submit(attached, **request(source))
    assert len(seen) == 4 and len({row['dispatch_id'] for row in seen}) == 4
    assert all(row['scope'] == 'step' and row['job_id'] == 234 for row in seen)


def test_submit_cutoff_and_worker_duration_are_enforced(attached, source, monkeypatch):
    monkeypatch.setattr(dispatch, 'submit', lambda *a, **k: pytest.fail('expired submission'))
    with pytest.raises(ValueError, match='deadline'):
        pool.submit(attached, **request(source), time_limit='02:00:00')
    profile = json.loads((attached / 'pool.json').read_text())
    profile['submit_before'] = time.time() - 1
    (attached / 'pool.json').write_text(json.dumps(profile))
    with pytest.raises(ValueError, match='cutoff'):
        pool.submit(attached, **request(source))


def test_lock_is_atomic_and_not_stolen(attached, source):
    (attached / '.reservation-lock').mkdir()
    with pytest.raises(ValueError, match='no automatic stale'):
        pool.submit(attached, **request(source))


def test_staged_pool_command_has_exact_step_and_private_publication(attached, source, monkeypatch):
    seen = {}
    def launch(directory, manifest, argv, report, timeout):
        seen.update(manifest=manifest, argv=argv)
        return report
    monkeypatch.setattr(pool, 'launch_step', launch)
    report = pool.submit(attached, **request(source))
    argv = seen['argv']
    assert argv[0] == 'srun' and '--jobid=234' in argv and '--exclusive' in argv and '--exact' in argv
    assert '--cpu-bind=cores' in argv and '--export=NONE' in argv
    assert not any(arg.startswith('--overlap') or arg == 'sbatch' for arg in argv)
    directory = Path(report['directory'])
    assert (directory / 'results.git/HEAD').exists()
    assert 'SUCODER_LOCAL_ROOT/mirrors/' in (directory / 'prepare.sh').read_text()
    assert 'SUCODER_POOL_SOURCE_REPOSITORY' in (directory / 'prepare.sh').read_text()
    digests = json.loads((directory / 'digests.json').read_text())
    assert digests['dispatch_pool_assets.py'] == hashlib.sha256((directory / 'dispatch_pool_assets.py').read_bytes()).hexdigest()


def test_launch_timeout_keeps_live_srun_and_unknown_reservation(tmp_path, monkeypatch):
    child = SimpleNamespace(pid=987, poll=lambda: None)
    monkeypatch.setattr(pool.subprocess, 'Popen', lambda *a, **k: child)
    directory = tmp_path
    manifest = dict(execution=dict(pool_id='a' * 32, job_id=234, node='n1', cluster='cluster'), dispatch_id='b' * 32, uid=os.getuid())
    report = dict(state='submission_unknown')
    answer = pool.launch_step(directory, manifest, ['fake-srun'], report, .001)
    assert answer['state'] == 'submission_unknown' and answer['frontend_pid'] == 987
    assert 'no retry' in answer['reason']


def test_step_receipt_is_bound_to_pool_and_scheduler_source(attached, source, monkeypatch):
    def launch(directory, manifest, argv, report, timeout):
        execution = manifest['execution']
        identity = dict(schema_version=1, uid=os.getuid(), dispatch_id=manifest['dispatch_id'],
                        job_id=234, cluster='cluster', pool_id=execution['pool_id'], node='n1', step_id='4')
        dispatch._write_json(directory / 'step-source.json', identity)
        dispatch._write_json(directory / 'scheduler-source.json', identity)
        report.update(state='submitted', job_id=234, cluster='cluster', step_id='4')
        from sucoder.dispatch_protocol import atomic_json
        atomic_json(str(directory / 'submission.json'), report)
        return report
    monkeypatch.setattr(pool, 'launch_step', launch)
    report = pool.submit(attached, **request(source))
    path = Path(report['directory'])
    manifest = json.loads((path / 'manifest.json').read_text())
    receipt, state = dispatch._receipt(path, manifest)
    assert state == 'present' and receipt['step_id'] == '4'
    wrong = json.loads((path / 'scheduler-source.json').read_text())
    wrong['step_id'] = '40'
    (path / 'scheduler-source.json').write_text(json.dumps(wrong))
    assert dispatch._receipt(path, manifest) == (None, 'unreadable')


def test_exact_terminal_step_proof_required_to_release(attached, monkeypatch):
    rows = {'a' * 32: dict(state='reserved', cpus=1, mem_mib=8192, directory='/fake')}
    (attached / 'reservations.json').write_text(json.dumps(rows))
    monkeypatch.setattr(pool, '_worker_receipt', lambda *a: (None, None, {}))
    monkeypatch.setattr(dispatch, '_scheduler', lambda *a: dict(available=False, state='unavailable'))
    assert pool.status(attached)['used_cpus'] == 1
    monkeypatch.setattr(dispatch, '_scheduler', lambda *a: dict(available=True, state='RUNNING'))
    assert pool.status(attached)['used_cpus'] == 1
    monkeypatch.setattr(dispatch, '_scheduler', lambda *a: dict(available=True, state='COMPLETED', source='sacct'))
    result = pool.status(attached)
    assert result['used_cpus'] == 0
    assert result['reservations']['a' * 32]['release_proof']['source'] == 'sacct'


def test_stop_only_targets_exact_verified_step(attached, monkeypatch):
    rows = {'a' * 32: dict(state='reserved', cpus=1, mem_mib=8192, directory='/fake')}
    (attached / 'reservations.json').write_text(json.dumps(rows))
    monkeypatch.setattr(pool, '_worker_receipt', lambda *a: (None, None, dict(job_id=234, step_id='4')))
    monkeypatch.setattr(dispatch, '_scheduler', lambda *a: dict(available=True, state='RUNNING'))
    seen = []
    monkeypatch.setattr(dispatch, '_run', lambda argv, **k: (seen.append(argv) or SimpleNamespace(returncode=0)))
    answer = pool.stop_worker(attached, 'a' * 32)
    assert seen == [['scancel', '--clusters=cluster', '--signal=TERM', '234.4']]
    assert answer['reservation'].startswith('held')
    assert json.loads((attached / 'reservations.json').read_text())['a' * 32]['state'] == 'reserved'


def step_receipt(step='0'):
    source = dict(step_id=step, uid=os.getuid(), job_id=234, node='n1', started_at=time.time())
    return dict(job_id=234, uid=os.getuid(), cluster='cluster', step_id=step, node='n1',
                allocation_start='2020-01-01T01:00:00', step_source=source)


def test_step_scheduler_never_accepts_parent_or_sibling(monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'scontrol':
            return SimpleNamespace(returncode=1, stdout='')
        return SimpleNamespace(returncode=0, stdout='234|%s|RUNNING|0:0|n1|2020-01-01T01:00:00\n234.40||COMPLETED|0:0|n1|2020-01-01T01:01:00\n' % os.getuid())
    monkeypatch.setattr(dispatch, '_run', run)
    answer = dispatch._scheduler(step_receipt('4'), 1)
    assert answer['state'] == 'unknown'
    assert calls[0][0] == 'scontrol' and calls[0][-1] == '234.4'


@pytest.mark.parametrize('parent_uid', [None, 'wrong', str(os.getuid())])
def test_blank_accounting_step_uid_requires_bound_parent(monkeypatch, parent_uid):
    def run(argv, **kwargs):
        if argv[0] == 'scontrol':
            return SimpleNamespace(returncode=1, stdout='')
        parent = '' if parent_uid is None else '234|%s|RUNNING|0:0|n1|2020-01-01T01:00:00\n' % parent_uid
        return SimpleNamespace(returncode=0, stdout=parent + '234.0||COMPLETED|0:0|n1|2020-01-01T01:01:00\n')
    monkeypatch.setattr(dispatch, '_run', run)
    answer = dispatch._scheduler(step_receipt(), 1)
    assert (answer.get('state') == 'COMPLETED') == (parent_uid == str(os.getuid()))


def test_live_step_uses_native_step_state_and_parent_incarnation(monkeypatch):
    def run(argv, **kwargs):
        if 'step' in argv:
            return SimpleNamespace(returncode=0, stdout='StepId=234.0 UserId=%s State=RUNNING NodeList=n1 StartTime=2020-01-01T01:01:00' % os.getuid())
        return SimpleNamespace(returncode=0, stdout='JobId=234 UserId=user(%s) NodeList=n1 StartTime=2020-01-01T01:00:00' % os.getuid())
    monkeypatch.setattr(dispatch, '_run', run)
    assert dispatch._scheduler(step_receipt(), 1)['state'] == 'RUNNING'


def test_isolated_prepare_does_not_reset_root_or_share_origin(tmp_path):
    script = build_worker_prepare_script(results_path='/shared/private/results.git', dispatch_id='a' * 32, commit='b' * 40, isolated=True)
    assert 'export SUCODER_LOCAL_ROOT=' not in script
    assert 'git remote set-url origin /shared/private/results.git' in script
    assert '--no-hardlinks' in script
    assert 'SUCODER_LOCAL_ROOT/mirrors/' in build_worker_snapshot_script(results_path='/shared/private/results.git', dispatch_id='a' * 32, isolated=True)


def test_pool_root_uses_actual_job_node_and_private_child(tmp_path):
    disk = tmp_path / 'disk'
    disk.mkdir()
    mounts = tmp_path / 'mounts'
    mounts.write_text('1 0 0:1 / / rw - ext4 /dev/test rw\n')
    manifest = dict(local_disk_root=str(disk), dispatch_id='a' * 32, execution=dict(job_id=234, node='n1'))
    env = dict(SLURM_JOB_ID='234', SLURM_STEP_ID='4', SLURMD_NODENAME='n1')
    root, job, _ = worker.receiving_root(manifest, env, str(mounts))
    assert root == str(disk / 'job234/workers' / ('a' * 32)) and job == '234'
    assert Path(root).stat().st_mode & 0o077 == 0
    with pytest.raises(worker.Refused, match='job/node'):
        worker.receiving_root(manifest, dict(env, SLURM_STEP_ID='batch'), str(mounts))


def test_cached_base_can_prepare_new_private_candidate_without_cache_mutation(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    def git(path, *args):
        return subprocess.check_output(['git', '-C', str(path)] + list(args), text=True).strip()
    git(source, 'init', '-q')
    git(source, 'config', 'user.name', 'Test')
    git(source, 'config', 'user.email', 'test@example.invalid')
    (source / 'tracked').write_text('base')
    git(source, 'add', '.')
    git(source, 'commit', '-qm', 'base')
    cache = tmp_path / 'cache.git'
    subprocess.check_call(['git', 'clone', '-q', '--bare', '--no-hardlinks', str(source), str(cache)])
    cache_refs = git(cache, 'show-ref')
    (source / 'tracked').write_text('private candidate')
    git(source, 'commit', '-qam', 'candidate')
    commit = git(source, 'rev-parse', 'HEAD')
    results = tmp_path / 'results.git'
    subprocess.check_call(['git', 'clone', '-q', '--bare', '--no-hardlinks', str(source), str(results)])
    root = tmp_path / 'job234/workers' / ('a' * 32)
    root.mkdir(parents=True)
    (tmp_path / 'dispatch_pool_assets.py').write_bytes(Path(assets.__file__).read_bytes())
    script = build_worker_prepare_script(results_path=str(results), dispatch_id='a' * 32, commit=commit, isolated=True)
    env = dict(os.environ, SLURM_JOB_ID='234', SUCODER_LOCAL_ROOT=str(root), SUCODER_POOL_SOURCE_REPOSITORY=str(cache),
               TMPDIR=str(root / 'tmp'), UV_CACHE_DIR=str(root / 'cache/uv'), PIP_CACHE_DIR=str(root / 'cache/pip'),
               npm_config_cache=str(root / 'cache/npm'))
    subprocess.check_call(['/bin/bash', '-c', script], env=env)
    work = root / 'mirrors' / ('dispatch_' + 'a' * 32)
    assert git(work, 'rev-parse', 'HEAD') == commit
    assert git(work, 'remote', 'get-url', 'origin') == str(results)
    assert not (work / '.git/objects/info/alternates').exists()
    assert (work / 'tracked').read_text() == 'private candidate'
    assert git(cache, 'show-ref') == cache_refs
    receipt = json.loads((tmp_path / 'source-copy-receiving.json').read_text())
    assert receipt['independent_object_files'] > 0 and receipt['hardlinks'] is False
    assert subprocess.run(['git', '-C', str(cache), 'cat-file', '-e', commit], stderr=subprocess.DEVNULL).returncode != 0


def test_receiving_nested_shared_mount_refuses_before_state(tmp_path):
    disk = tmp_path / 'disk'
    disk.mkdir()
    target = disk / 'job234/workers' / ('a' * 32)
    mounts = tmp_path / 'mounts'
    mounts.write_text('1 0 0:1 / / rw - ext4 /dev/test rw\n2 1 0:2 / %s rw - nfs host:/shared rw\n' % target)
    manifest = dict(local_disk_root=str(disk), dispatch_id='a' * 32, execution=dict(job_id=234, node='n1'))
    with pytest.raises(worker.Refused, match='child root'):
        worker.receiving_root(manifest, dict(SLURM_JOB_ID='234', SLURM_STEP_ID='4', SLURMD_NODENAME='n1'), str(mounts))
    assert not (target / 'state').exists()


def test_pool_image_verified_once_and_mutation_refuses(tmp_path):
    allocation = tmp_path / 'job234'
    cache = allocation / 'assets/source.git'
    cache.mkdir(parents=True)
    subprocess.check_call(['git', 'init', '-q', '--bare', str(cache)])
    # A source ref is necessary to bind a nonempty immutable source namespace.
    source = tmp_path / 'source'
    source.mkdir()
    subprocess.check_call(['git', '-C', str(source), 'init', '-q'])
    subprocess.check_call(['git', '-C', str(source), '-c', 'user.name=Test', '-c', 'user.email=x@y', 'commit', '--allow-empty', '-qm', 'base'])
    subprocess.check_call(['git', '-C', str(cache), 'fetch', '-q', str(source), 'HEAD:refs/heads/base'])
    image = allocation / 'assets/image.sqfs'
    image.write_bytes(b'owned synthetic image')
    image.chmod(0o400)
    root = allocation / 'workers' / ('a' * 32)
    root.mkdir(parents=True)
    execution = dict(pool_id='b' * 32, assets=dict(source=str(cache), image=str(image), image_sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
    first = assets.verify(execution, 'c' * 40, str(root), os.environ)
    assert Path(first['verification_receipt']).exists()
    assert assets.verify(execution, 'd' * 40, str(root), os.environ)['image_sha256'] == first['image_sha256']
    image.chmod(0o600)
    with pytest.raises(ValueError, match='write permission'):
        assets.verify(execution, 'c' * 40, str(root), os.environ)


def test_late_receipt_reconciles_without_new_srun(tmp_path, monkeypatch):
    execution = dict(pool_id='a' * 32, job_id=234, node='n1', cluster='cluster')
    manifest = dict(execution=execution, dispatch_id='b' * 32, uid=os.getuid())
    report = dict(state='submission_unknown')
    source = dict(dispatch_id='b' * 32, uid=os.getuid(), job_id=234, node='n1', cluster='cluster', pool_id='a' * 32, step_id='7')
    (tmp_path / 'step-source.json').write_text(json.dumps(source))
    monkeypatch.setattr(pool.subprocess, 'Popen', lambda *a, **k: pytest.fail('reconciliation launched a duplicate'))
    assert pool.adopt_step(tmp_path, manifest, report)
    assert report['step_id'] == '7' and report['state'] == 'submitted'
    assert json.loads((tmp_path / 'scheduler-source.json').read_text()) == source


def test_worker_duration_gets_early_publication_deadline(attached, source, monkeypatch):
    before = time.time()
    seen = []
    def submit(*args, **kwargs):
        seen.append(kwargs['_execution'])
        return dict(directory=str(source['repo']), state='submission_unknown')
    monkeypatch.setattr(dispatch, 'submit', submit)
    pool.submit(attached, **request(source), time_limit='00:05:00')
    assert before + 300 <= seen[0]['deadline'] <= time.time() + 300
    assert seen[0]['deadline'] < json.loads((attached / 'pool.json').read_text())['worker_deadline']


def test_pool_cli_preserves_evidence_and_snapshot_contract(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from sucoder.dispatch_cli import app
    seen = {}
    def submit(*args, **kwargs):
        seen.update(kwargs)
        return dict(state='submitted')
    monkeypatch.setattr(pool, 'submit', submit)
    result = CliRunner().invoke(app, ['--root', str(tmp_path), 'pool', 'submit', '/shared/pool', '/shared/repo',
        '--ref', 'b' * 40, '--task-file', '/shared/task', '--agent', 'codex', '--executable', '/shared/codex',
        '--require-evidence', 'owner-status.json', '--require-evidence', 'handoff.org', '--snapshot-minutes', '2'])
    assert result.exit_code == 0, result.output
    assert seen['required_evidence'] == ['owner-status.json', 'handoff.org']
    assert seen['snapshot_minutes'] == 2 and seen['directory_root'] == tmp_path


def test_pool_staging_uses_independent_local_object_copy(attached, source, monkeypatch):
    real = dispatch._git
    calls = []
    def git(path, *args, **kwargs):
        calls.append(args)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(dispatch, '_git', git)
    monkeypatch.setattr(pool, 'launch_step', lambda directory, manifest, argv, report, timeout: report)
    original = source['repo'] / '.git'
    before = {str(p.relative_to(original)): p.read_bytes() for p in original.rglob('*') if p.is_file()}
    report = pool.submit(attached, **request(source))
    clone = next(args for args in calls if 'clone' in args)
    assert '--local' in clone and '--no-hardlinks' in clone and '--no-local' not in clone
    destination = Path(report['directory']) / 'results.git'
    assert report['source_copy']['independent_object_files'] > 0
    for p in (destination / 'objects').rglob('*'):
        other = original / p.relative_to(destination)
        if p.is_file() and other.exists():
            assert (p.stat().st_dev, p.stat().st_ino) != (other.stat().st_dev, other.stat().st_ino)
    after = {str(p.relative_to(original)): p.read_bytes() for p in original.rglob('*') if p.is_file()}
    assert before == after
    assert not (destination / 'objects/info/alternates').exists()


@pytest.mark.parametrize('kind', ['alternates', 'http-alternates', 'shallow', 'object-symlink'])
def test_pool_local_copy_refuses_dependent_or_redirected_sources(attached, source, monkeypatch, kind):
    metadata = source['repo'] / '.git'
    if kind in ('alternates', 'http-alternates'):
        (metadata / 'objects/info' / kind).write_text('/unread/foreign/objects\n')
    elif kind == 'shallow':
        commit = subprocess.check_output(['git', '-C', str(source['repo']), 'rev-parse', 'HEAD'], text=True).strip()
        (metadata / 'shallow').write_text(commit + '\n')
    else:
        (metadata / 'objects/link').symlink_to(source['repo'] / 'tracked')
    monkeypatch.setattr(pool, 'launch_step', lambda *a: pytest.fail('unsafe source reached launch'))
    with pytest.raises(ValueError, match='alternates|symlinks'):
        pool.submit(attached, **request(source))
    assert not dispatch.DEFAULT_DIRECTORY_ROOT.exists()


def test_hardlinked_copy_is_rejected(source, tmp_path):
    destination = tmp_path / 'unsafe.git'
    subprocess.check_call(['git', 'clone', '--bare', '--local', '-q', str(source['repo']), str(destination)])
    with pytest.raises(ValueError, match='hardlinked'):
        assets.verify_copy(str(source['repo'] / '.git'), str(destination))


def test_independent_job_keeps_no_local_transport(source, monkeypatch):
    from tests.test_dispatch import scheduler
    scheduler(monkeypatch)
    real = dispatch._git
    calls = []
    def git(path, *args, **kwargs):
        calls.append(args)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(dispatch, '_git', git)
    dispatch.submit(**source)
    clone = next(args for args in calls if 'clone' in args)
    assert '--no-local' in clone and '--local' not in clone


def test_nonregular_object_refuses_without_opening_fifo(source):
    metadata = source['repo'] / '.git'
    os.mkfifo(metadata / 'objects/owned-fifo')
    with pytest.raises(ValueError, match='regular files'):
        assets.validate_local_source(str(metadata))


def test_promisor_marker_refuses_even_without_config(source):
    metadata = source['repo'] / '.git'
    (metadata / 'objects/pack/empty.promisor').write_bytes(b'')
    with pytest.raises(ValueError, match='promisor'):
        assets.validate_local_source(str(metadata))
