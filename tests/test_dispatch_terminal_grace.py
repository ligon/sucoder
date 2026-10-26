"""Finite fake-kernel terminal transitions; no scheduler/native/store access."""
import errno
import os
from types import SimpleNamespace

import pytest
from sucoder import dispatch_worker as worker


def process(root, state='R', started=42, uid=None):
    directory = root / '123'
    directory.mkdir(exist_ok=True)
    fields = [state] + ['0'] * 19
    fields[19] = str(started)
    (directory / 'stat').write_text('123 (fake child) ' + ' '.join(fields) + '\n')
    uid = os.getuid() if uid is None else uid
    (directory / 'status').write_text('Uid:\t%s\t%s\t%s\t%s\n' % ((uid,) * 4))
    return directory


def clock(monkeypatch, callback=lambda now: None):
    value = [0.0]
    def sleep(seconds):
        assert 0 <= seconds <= .005
        value[0] += seconds
        callback(value[0])
    monkeypatch.setattr(worker.time, 'monotonic', lambda: value[0])
    monkeypatch.setattr(worker.time, 'sleep', sleep)
    return value


@pytest.mark.parametrize('boundary', ['directory', 'mounts'])
def test_delayed_terminal_transition_preserves_strict_inspection(tmp_path, monkeypatch, boundary):
    directory = process(tmp_path)
    (directory / 'fd').mkdir()
    value = clock(monkeypatch, lambda now: process(tmp_path, state='Z') if now >= .01 else None)
    original = worker.os.listdir
    def listdir(path):
        if str(path) == str(directory / 'fd') and boundary == 'directory':
            raise OSError(errno.EACCES, 'kernel transition', str(path))
        return original(path)
    monkeypatch.setattr(worker.os, 'listdir', listdir)
    def mounts(*args, **kwargs):
        raise OSError(errno.EINVAL, 'kernel transition', str(directory / 'mountinfo'))
    owner = SimpleNamespace(uid=os.getuid(), members=lambda: [{'pid': 123, 'starttime': 42}])
    result = worker.inspect_placement(SimpleNamespace(pid=123), SimpleNamespace(inspect_file_mounts=mounts),
                                      str(tmp_path), str(tmp_path), proc_root=str(tmp_path), ownership=owner)
    proof = result['inspection_races'][0]['proof']
    assert proof['terminal'] and proof['reason'] == 'same_incarnation_Z'
    assert proof['attempts'] >= 3 and .01 <= value[0] <= .05
    assert result['databases'] == result['misplaced'] == []


@pytest.mark.parametrize('state', ['R', 'S', 'D'])
def test_persistent_live_identity_refuses_at_finite_bound(tmp_path, monkeypatch, state):
    process(tmp_path, state=state)
    value = clock(monkeypatch)
    result = worker.terminal_proof_after_error(123, str(tmp_path), {'uid': os.getuid(), 'starttime': 42})
    assert not result['terminal'] and result['reason'] == 'still_live'
    assert result['attempts'] <= 11 and value[0] <= .05


@pytest.mark.parametrize('changed', ['uid', 'starttime'])
def test_reused_or_changed_identity_never_passes_even_as_zombie(tmp_path, monkeypatch, changed):
    process(tmp_path)
    def change(now):
        if now >= .005:
            process(tmp_path, state='Z', started=43 if changed == 'starttime' else 42,
                    uid=os.getuid() + 1 if changed == 'uid' else os.getuid())
    clock(monkeypatch, change)
    result = worker.terminal_proof_after_error(123, str(tmp_path), {'uid': os.getuid(), 'starttime': 42})
    assert not result['terminal'] and result['reason'] == 'identity_changed'


@pytest.mark.parametrize('missing', ['stat', 'status', 'malformed'])
def test_existing_unknown_process_is_not_terminal(tmp_path, monkeypatch, missing):
    directory = process(tmp_path)
    if missing == 'malformed':
        (directory / 'stat').write_text('malformed')
    else:
        (directory / missing).unlink()
    value = clock(monkeypatch)
    result = worker.terminal_proof_after_error(123, str(tmp_path), {'uid': os.getuid(), 'starttime': 42})
    assert not result['terminal'] and value[0] <= .05


def test_fully_absent_process_is_explicit_terminal_proof(tmp_path, monkeypatch):
    clock(monkeypatch)
    result = worker.terminal_proof_after_error(123, str(tmp_path), {'uid': os.getuid(), 'starttime': 42})
    assert result['terminal'] and result['reason'] == 'proc_directory_gone'


def test_shared_pass_deadline_prevents_new_full_grace(tmp_path, monkeypatch):
    process(tmp_path)
    value = clock(monkeypatch)
    result = worker.terminal_proof_after_error(123, str(tmp_path), {'uid': os.getuid(), 'starttime': 42}, deadline=.01)
    assert not result['terminal'] and value[0] <= .01


def test_unknown_original_identity_cannot_use_later_absence(tmp_path, monkeypatch):
    clock(monkeypatch)
    result = worker.terminal_proof_after_error(123, str(tmp_path), None)
    assert not result['terminal'] and result['reason'] == 'unknown_identity'
    assert result['attempts'] == 0
