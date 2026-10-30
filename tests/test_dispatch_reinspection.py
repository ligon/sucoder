"""Recovered metadata must be fully reread; live/unknown refusal stays strict."""
import builtins
import errno
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from sucoder import dispatch_worker as worker


def identity(root, uid=None, starttime=42):
    base = root / '123'
    base.mkdir(exist_ok=True)
    fields = ['S'] + ['0'] * 19
    fields[19] = str(starttime)
    (base / 'stat').write_text('123 (owned child) ' + ' '.join(fields))
    uid = os.getuid() if uid is None else uid
    (base / 'status').write_text('Uid:\t%s\n' % uid)
    return base


@pytest.fixture
def kernel(tmp_path, monkeypatch):
    base = identity(tmp_path)
    (base / 'fd').mkdir()
    (base / 'environ').write_bytes(('HOME=' + str(tmp_path)).encode())
    now = [0.0]
    monkeypatch.setattr(worker.time, 'monotonic', lambda: now[0])
    def sleep(seconds):
        assert 0 <= seconds <= .005
        now[0] += seconds
    monkeypatch.setattr(worker.time, 'sleep', sleep)
    owner = SimpleNamespace(uid=os.getuid(), members=lambda: [{'pid': 123, 'starttime': 42}])
    child = SimpleNamespace(pid=123)
    real_listdir, real_readlink, real_open = os.listdir, os.readlink, builtins.open
    counts = dict(directory=0, link=0, environment=0, mounts=0)

    def bind(boundary, persistent=False, filesystem='ext4'):
        def denied(name):
            counts[name] += 1
            if name == boundary and (persistent or now[0] < .01):
                raise OSError(errno.EINVAL if name == 'mounts' else errno.EACCES,
                              'invented proc transition', str(base / name))
        def listdir(path):
            if str(path) == str(base / 'fd'):
                denied('directory')
                return ['7']
            return real_listdir(path)
        def readlink(path):
            if str(path) == str(base / 'fd/7'):
                denied('link')
                return str(tmp_path / '.codex/fixture.sqlite')
            return real_readlink(path)
        def open_file(path, *args, **kwargs):
            if str(path) == str(base / 'environ'):
                denied('environment')
            return real_open(path, *args, **kwargs)
        def mounts(pids, **kwargs):
            assert pids == [123]
            denied('mounts')
            return [{'pid': 123, 'path': str(tmp_path / '.codex/fixture.sqlite'),
                     'filesystem': filesystem}]
        monkeypatch.setattr(worker.os, 'listdir', listdir)
        monkeypatch.setattr(worker.os, 'readlink', readlink)
        monkeypatch.setattr(builtins, 'open', open_file)
        runner = SimpleNamespace(inspect_file_mounts=mounts)
        return lambda: worker.inspect_placement_rechecked(
            child, runner, str(tmp_path), str(tmp_path), proc_root=str(tmp_path),
            expected_environment={'HOME': str(tmp_path)}, ownership=owner)
    return bind, now, counts, tmp_path


@pytest.mark.parametrize('boundary', ['directory', 'link', 'environment', 'mounts'])
def test_recovered_live_metadata_requires_a_complete_second_pass(kernel, boundary):
    bind, now, counts, root = kernel
    result = bind(boundary)()
    assert result['inspected_pids'] == [123]
    assert result['databases'] == [str(root / '.codex/fixture.sqlite')]
    assert result['environment_verified'] and not result['misplaced']
    assert counts[boundary] == 2
    assert all(counts[name] >= 1 for name in counts)
    proof = result['inspection_retries'][0]
    assert proof['initial_failure']['proof']['reason'] == 'still_live'
    assert proof['before']['starttime'] == proof['after']['starttime'] == 42
    assert proof['after']['state'] == 'S'
    assert now[0] == pytest.approx(.05, abs=1e-12)


@pytest.mark.parametrize('boundary', ['directory', 'link', 'environment', 'mounts'])
def test_persistent_live_unreadability_still_refuses_after_one_retry(kernel, boundary):
    bind, now, counts, _ = kernel
    with pytest.raises(worker.Refused) as caught:
        bind(boundary, persistent=True)()
    assert counts[boundary] == 2
    assert now[0] == pytest.approx(.1, abs=1e-12)
    diagnostic = caught.value.inspection_failure
    assert diagnostic['proof']['reason'] == 'still_live'
    assert diagnostic['reinspection']['outcome'] == 'refused'
    assert diagnostic['reinspection']['failure']['proof']['reason'] == 'still_live'


@pytest.mark.parametrize('when', ['before', 'after'])
@pytest.mark.parametrize('change', ['uid', 'starttime', 'unknown'])
def test_changed_or_unknown_incarnation_never_passes_reinspection(kernel, monkeypatch, when, change):
    bind, _, _, root = kernel
    invoke = bind('directory')
    original = worker.inspect_placement
    def mutate():
        if change == 'unknown':
            (root / '123/status').unlink()
        else:
            identity(root, uid=os.getuid() + 1 if change == 'uid' else None,
                     starttime=43 if change == 'starttime' else 42)
    def inspected(*args, **kwargs):
        try:
            result = original(*args, **kwargs)
        except worker.Refused:
            if when == 'before':
                mutate()
            raise
        if when == 'after':
            mutate()
        return result
    monkeypatch.setattr(worker, 'inspect_placement', inspected)
    with pytest.raises(worker.Refused):
        invoke()


def test_full_retry_retains_misplaced_store_diagnosis(kernel):
    bind, _, _, root = kernel
    result = bind('directory', filesystem='nfs')()
    assert result['misplaced'] == [str(root / '.codex/fixture.sqlite')]
    assert result['databases'] == []


def test_nonmetadata_refusal_is_not_retried(monkeypatch):
    calls = []
    def refuse(*args, **kwargs):
        calls.append(True)
        raise worker.Refused('effective environment changed')
    monkeypatch.setattr(worker, 'inspect_placement', refuse)
    with pytest.raises(worker.Refused, match='effective environment changed'):
        worker.inspect_placement_rechecked(None, None, '', '')
    assert calls == [True]


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux proc/dumpability boundary')
@pytest.mark.parametrize('recover', [False, True])
def test_real_owned_child_readability_recovery_and_persistent_refusal(tmp_path, monkeypatch, recover):
    code = '''import ctypes,sys
lib=ctypes.CDLL(None,use_errno=True)
print("START",flush=True)
assert sys.stdin.readline().strip()=="GO"
assert lib.prctl(4,0,0,0,0)==0
print("HIDDEN",flush=True)
command=sys.stdin.readline().strip()
if command=="RESTORE":
 assert lib.prctl(4,1,0,0,0)==0
 print("READABLE",flush=True)
 command=sys.stdin.readline().strip()
assert command=="EXIT"
'''
    child = subprocess.Popen([sys.executable, '-I', '-S', '-c', code],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'START'
        initial = worker.process_identity(child.pid, '/proc')
        child.stdin.write('GO\n'); child.stdin.flush()
        assert child.stdout.readline().strip() == 'HIDDEN'
        original = os.listdir
        denied = []
        def listdir(path):
            try:
                return original(path)
            except OSError as exc:
                if str(path) == '/proc/%s/fd' % child.pid:
                    assert exc.errno == errno.EACCES
                    denied.append(True)
                    if recover and len(denied) == 1:
                        child.stdin.write('RESTORE\n'); child.stdin.flush()
                raise
        monkeypatch.setattr(worker.os, 'listdir', listdir)
        def mounts(pids, proc_root='/proc'):
            for pid in pids:
                with open('%s/%s/mountinfo' % (proc_root, pid), 'rb') as stream:
                    stream.read()
            return []
        owner = SimpleNamespace(uid=os.getuid(), members=lambda: [
            {'pid': child.pid, 'starttime': initial['starttime']}])
        args = (child, SimpleNamespace(inspect_file_mounts=mounts), str(tmp_path), str(tmp_path))
        if recover:
            result = worker.inspect_placement_rechecked(*args, ownership=owner)
            assert result['inspected_pids'] == [child.pid]
            assert result['inspection_retries'][0]['after']['starttime'] == initial['starttime']
            assert result['inspection_retries'][0]['after']['state'] not in ('Z', 'X')
            assert denied == [True]
        else:
            with pytest.raises(worker.Refused):
                worker.inspect_placement_rechecked(*args, ownership=owner)
            assert len(denied) == 2
    finally:
        child.stdin.write('EXIT\n'); child.stdin.flush()
        child.wait(timeout=5)
        assert child.returncode == 0, child.stderr.read()


@pytest.mark.parametrize('when', ['before', 'after'])
@pytest.mark.parametrize('change', ['uid', 'starttime', 'partial_mismatch'])
def test_observed_reused_identity_cannot_be_erased_by_later_absence(monkeypatch, when, change):
    """A known replacement disappearing later cannot prove the original PID safe.

    The full second pass may finish before the post-check notices reuse.
    Do not consult a later directory-gone proof after observing that change.
    """
    expected = {'uid': os.getuid(), 'starttime': 42}
    original = dict(expected, state='S')
    replacement = dict(original)
    if change == "partial_mismatch":
        replacement = {"identity_changed": True, "starttime": 43}
    else:
        replacement[change] += 1
    identities = iter([replacement, None] if when == 'before'
                      else [original, replacement])
    inspections, terminal_checks = [], []

    def inspect(*args, **kwargs):
        inspections.append(True)
        if len(inspections) == 1:
            error = worker.Refused('first metadata refusal')
            error.inspection_failure = {
                'pid': 123, 'boundary': 'file descriptors', 'errno': errno.EACCES,
                'proof': {'reason': 'still_live', 'terminal': False,
                          'expected': dict(expected)},
            }
            raise error
        return {'inspected_pids': [] if when == 'before' else [123]}

    def later_gone(*args, **kwargs):
        terminal_checks.append(True)
        return {'terminal': True, 'reason': 'proc_directory_gone',
                'expected': dict(expected)}

    monkeypatch.setattr(worker, 'inspect_placement', inspect)
    monkeypatch.setattr(worker, 'process_identity', lambda *args: next(identities))
    monkeypatch.setattr(worker, 'terminal_proof_after_error', later_gone)
    with pytest.raises(worker.Refused):
        worker.inspect_placement_rechecked(None, None, '', '')
    assert len(inspections) == (1 if when == 'before' else 2)
    assert terminal_checks == []
