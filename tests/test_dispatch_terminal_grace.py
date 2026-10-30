"""Finite fake-kernel terminal transitions; no scheduler/native/store access."""
import errno
import os
from types import SimpleNamespace

import pytest
from sucoder import dispatch_worker as worker


def process(root, state='R', started=42, uid=None, flags=0, pid=123):
    directory = root / str(pid)
    directory.mkdir(exist_ok=True)
    fields = [state] + ['0'] * 19
    fields[19] = str(started)
    fields[6] = str(flags)
    (directory / 'stat').write_text(str(pid) + ' (fake child) ' + ' '.join(fields) + '\n')
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


@pytest.mark.parametrize('state', ['I', 'R', 'D'])
def test_exiting_process_requires_eventual_terminal_proof(tmp_path, monkeypatch, state):
    process(tmp_path, state=state, flags=0x400004)
    value = clock(monkeypatch, lambda now: process(tmp_path, state='Z', flags=0x400004) if now >= .2 else None)
    proof = worker.terminal_proof_after_error(123, str(tmp_path), {'uid': os.getuid(), 'starttime': 42})
    assert proof['terminal'] and proof['reason'] == 'same_incarnation_Z'
    assert proof['first_identity']['state'] == state
    assert proof['first_identity']['flags'] == 0x400004
    assert proof['last_identity']['state'] == 'Z'
    assert proof['exit_grace'] and .2 <= value[0] <= 1


@pytest.mark.parametrize('flags,bound', [(0, .05), (0x400000, .05), ('bad', .05), (0x400004, 1)])
def test_idle_or_exiting_never_means_terminal(tmp_path, monkeypatch, flags, bound):
    process(tmp_path, state='I', flags=flags)
    value = clock(monkeypatch)
    proof = worker.terminal_proof_after_error(123, str(tmp_path), {'uid': os.getuid(), 'starttime': 42})
    assert not proof['terminal']
    assert value[0] <= bound and value[0] == pytest.approx(bound)
    assert proof['attempts'] <= (201 if flags == 0x400004 else 11)


@pytest.mark.parametrize('mismatch', ['first_starttime', 'last_starttime', 'uid'])
def test_observed_identity_change_cannot_be_erased_by_later_missing_metadata(tmp_path, monkeypatch, mismatch):
    import builtins
    import io
    directory = process(tmp_path, state='I', flags=4)
    original = builtins.open
    reads = [0]
    before = (directory / 'stat').read_text()
    after = before.replace(' 42\n', ' 43\n')

    def changed(path, *args, **kwargs):
        if str(path) == str(directory / 'stat'):
            reads[0] += 1
            if mismatch == 'uid' and reads[0] > 1:
                raise FileNotFoundError(errno.ENOENT, 'later vanished stat')
            return io.StringIO(after if mismatch == 'first_starttime' or reads[0] > 1 else before)
        if str(path) == str(directory / 'status') and mismatch == 'uid':
            return io.StringIO('Uid:\t%s\t%s\t%s\t%s\n' % ((os.getuid() + 1,) * 4))
        return original(path, *args, **kwargs)

    def gone(now):
        for item in directory.iterdir():
            item.unlink()
        directory.rmdir()
    value = clock(monkeypatch, gone)
    monkeypatch.setattr(builtins, 'open', changed)
    proof = worker.terminal_proof_after_error(123, str(tmp_path), {'uid': os.getuid(), 'starttime': 42})
    assert not proof['terminal'] and proof['reason'] == 'identity_changed'
    assert proof['attempts'] == 1 and value[0] == 0


def test_two_exiting_pids_share_one_extended_deadline(tmp_path, monkeypatch):
    process(tmp_path, state='I', flags=4)
    process(tmp_path, state='I', flags=4, pid=456)
    value = clock(monkeypatch, lambda now: process(tmp_path, state='Z', flags=4) if now >= .6 else None)
    expected = {'uid': os.getuid(), 'starttime': 42}
    first = worker.terminal_proof_after_error(123, str(tmp_path), expected, deadline=.05, exit_deadline=1)
    assert first['terminal'] and value[0] >= .6
    second = worker.terminal_proof_after_error(456, str(tmp_path), expected, deadline=.05, exit_deadline=1)
    assert not second['terminal'] and value[0] == pytest.approx(1)
    assert second['budget_seconds'] <= .4


@pytest.mark.parametrize('boundary', ['directory', 'mounts'])
def test_inspection_waits_for_flagged_exit_without_hiding_misplaced_peer(tmp_path, monkeypatch, boundary):
    directory = process(tmp_path, state='I', flags=0x400004)
    peer = process(tmp_path, pid=456)
    (directory / 'fd').mkdir()
    (peer / 'fd').mkdir()
    clock(monkeypatch, lambda now: process(tmp_path, state='Z', flags=4) if now >= .2 else None)
    original = worker.os.listdir

    def listdir(path):
        if str(path) == str(directory / 'fd') and boundary == 'directory':
            raise PermissionError(errno.EACCES, 'exit cleanup', str(path))
        return original(path)

    def mounts(pids, **kwargs):
        if pids == [123]:
            raise OSError(errno.EINVAL, 'exit cleanup', str(directory / 'mountinfo'))
        return [{'pid': 456, 'path': '/shared/state.sqlite-wal', 'filesystem': 'lustre'}]
    monkeypatch.setattr(worker.os, 'listdir', listdir)
    owner = SimpleNamespace(uid=os.getuid(), members=lambda: [{'pid': pid, 'starttime': 42} for pid in (123, 456)])
    result = worker.inspect_placement(SimpleNamespace(pid=123), SimpleNamespace(inspect_file_mounts=mounts),
                                      str(tmp_path), str(tmp_path), proc_root=str(tmp_path), ownership=owner)
    assert result['misplaced'] == ['/shared/state.sqlite-wal']
    assert result['inspection_races'][0]['proof']['terminal']
    assert result['inspection_races'][0]['proof']['exit_grace']
