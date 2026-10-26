"""Explicit existing-allocation pools; no allocation creation or job-wide stop."""
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from . import dispatch as control
from .dispatch_protocol import atomic_json

TERMINAL = frozenset(('COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'NODE_FAIL',
                      'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'))


def memory_mib(value):
    match = re.fullmatch(r'([1-9][0-9]*)([KMGT]?)', str(value), re.I)
    if not match:
        raise ValueError('memory must be an integer Slurm memory quantity')
    return math.ceil(int(match[1]) * {'': 1, 'K': 1 / 1024, 'M': 1, 'G': 1024, 'T': 1024 ** 2}[match[2].upper()])


def _job(job_id, timeout=10):
    answer = control._run(['scontrol', 'show', 'job', '--oneliner', str(job_id)], timeout=timeout)
    if answer.returncode:
        raise ValueError('cannot verify allocation with scontrol')
    fields = dict(re.findall(r'(\w+)=([^\s]+)', answer.stdout))
    owner = re.search(r'\(([0-9]+)\)$', fields.get('UserId', ''))
    node = fields.get('NodeList', '')
    if (fields.get('JobId') != str(job_id) or owner is None or int(owner[1]) != os.getuid()
            or fields.get('JobState') != 'RUNNING' or not re.fullmatch(r'[A-Za-z0-9_.-]+', node)
            or fields.get('NumNodes') != '1'):
        raise ValueError('pool requires this uid\'s running single-node allocation')
    tres = dict(part.split('=', 1) for part in fields.get('AllocTRES', '').split(',') if '=' in part)
    try:
        fields['verified_cpus'] = int(tres['cpu'])
        fields['verified_mem_mib'] = memory_mib(tres['mem'])
        fields['verified_end'] = datetime.fromisoformat(fields['EndTime']).timestamp()
        fields['verified_start'] = datetime.fromisoformat(fields['StartTime']).timestamp()
    except (ValueError, KeyError):
        raise ValueError('allocation CPU/memory/end-time evidence is incomplete')
    return fields


def attach(directory, job_id, *, node, cpus, mem, submit_before, worker_deadline,
           expires, source, image, image_sha256, timeout=10):
    directory = control._shared_path(Path(directory), 'pool directory')
    if directory.exists():
        raise ValueError('pool directory must be new')
    if not re.fullmatch(r'[1-9][0-9]*', str(job_id)):
        raise ValueError('pool job ID must be numeric')
    if type(cpus) is not int or cpus < 1:
        raise ValueError('pool CPUs must be positive')
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in (submit_before, worker_deadline, expires)):
        raise ValueError('pool cutoffs must be finite Unix timestamps')
    facts = _job(job_id, timeout)
    if facts['NodeList'] != node or cpus > facts['verified_cpus'] or memory_mib(mem) > facts['verified_mem_mib']:
        raise ValueError('pool node/budget differs from allocation evidence')
    if not time.time() < submit_before < worker_deadline <= expires <= facts['verified_end']:
        raise ValueError('pool cutoffs must fit allocation end and reserve publication time')
    if not re.fullmatch(r'[0-9a-f]{64}', image_sha256):
        raise ValueError('pool image requires full SHA256')
    allocation = '/local/job' + str(job_id) + '/'
    for path in (str(source), str(image)):
        if not path.startswith(allocation) or os.path.normpath(path) != path:
            raise ValueError('asset paths must be receiving-allocation-local absolute paths')
    cluster = os.environ.get('SLURM_CLUSTER_NAME')
    if not cluster or not re.fullmatch(r'[A-Za-z0-9_.-]+', cluster):
        raise ValueError('explicit current Slurm cluster identity is required')
    profile = dict(schema_version=1, pool_id=uuid.uuid4().hex, uid=os.getuid(), cluster=cluster,
                   job_id=int(job_id), node=node, cpus=cpus, mem_mib=memory_mib(mem),
                   submit_before=submit_before, worker_deadline=worker_deadline, expires=expires,
                   assets=dict(source=str(source), image=str(image), image_sha256=image_sha256),
                   allocation_evidence=facts, created_at=time.time())
    directory.mkdir(mode=0o700, parents=True)
    os.chmod(directory, 0o700)  # shared SGID parents must not make a private control directory 2700
    control._write_json(directory / 'pool.json', profile)
    control._write_json(directory / 'reservations.json', {})
    return dict(profile, directory=str(directory))


def _load(directory):
    path = Path(directory)
    if path.is_symlink() or path.resolve() != path or not path.is_dir():
        raise ValueError('pool directory must be an absolute nonredirected directory')
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('pool directory must be private and owned')
    profile, state = control._read_json(path / 'pool.json')
    if (state != 'present' or profile.get('uid') != os.getuid() or profile.get('schema_version') != 1
            or not re.fullmatch(r'[0-9a-f]{32}', str(profile.get('pool_id', '')))):
        raise ValueError('invalid pool identity')
    return path, profile


@contextmanager
def _locked(path):
    lock = path / '.reservation-lock'
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError:
        raise ValueError('pool reservation lock held; no automatic stale lock stealing')
    try:
        yield
    finally:
        lock.rmdir()


def validate_execution(value, resources):
    if (value.get('scope') != 'step' or not re.fullmatch(r'[0-9a-f]{32}', str(value.get('dispatch_id', '')))
            or not re.fullmatch(r'[0-9a-f]{32}', str(value.get('pool_id', '')))
            or type(value.get('job_id')) is not int or value['job_id'] < 1
            or not re.fullmatch(r'[A-Za-z0-9_.-]+', str(value.get('node', '')))
            or not re.fullmatch(r'[A-Za-z0-9_.-]+', str(value.get('cluster', '')))
            or type(value.get('deadline')) not in (int, float) or not math.isfinite(value['deadline'])
            or value['deadline'] <= time.time() + 120):
        raise ValueError('invalid or expired pool execution binding')
    if resources.get('exclude'):
        raise ValueError('pool workers use the bound node, not exclusion routing')


def submit(directory, repo, ref, task_file, agent, executable, *, cpus=1, mem='8G',
           time_limit='00:30:00', timeout=30, **options):
    path, profile = _load(directory)
    if os.environ.get('SLURM_CLUSTER_NAME') != profile['cluster']:
        raise ValueError('pool submission requires its original current Slurm cluster')
    if options.get('dry_run'):
        raise ValueError('pool submission requires an actual reservation; use normal submit --dry-run for staging validation')
    facts = _job(profile['job_id'], timeout)
    if (facts['NodeList'] != profile['node'] or facts['verified_cpus'] < profile['cpus']
            or facts['verified_mem_mib'] < profile['mem_mib'] or facts['verified_end'] < profile['expires']
            or facts['StartTime'] != profile['allocation_evidence']['StartTime']):
        raise ValueError('allocation identity/budget changed since attach')
    resources = control._resources(dict(account=facts['Account'], partition=facts['Partition'],
                                       qos=facts.get('QOS'), cpus=cpus, mem=mem, time=time_limit))
    # Slurm time strings are normalized independently of the scheduler. Worker
    # deadline is an additional hard cap, not a fresh full duration per retry.
    days, clock = time_limit.split('-', 1) if '-' in time_limit else ('0', time_limit)
    parts = [int(p) for p in clock.split(':')]
    seconds = (parts[0] * 60 if len(parts) == 1 else parts[0] * 60 + parts[1]
               if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]) + int(days) * 86400
    if '-' in time_limit and len(parts) < 3:
        seconds = int(days) * 86400 + parts[0] * 3600 + (parts[1] * 60 if len(parts) == 2 else 0)
    dispatch_id = uuid.uuid4().hex
    execution = dict(scope='step', pool_id=profile['pool_id'], dispatch_id=dispatch_id,
                     job_id=profile['job_id'], node=profile['node'], cluster=profile['cluster'],
                     allocation_start=facts['StartTime'],
                     deadline=min(profile['worker_deadline'], time.time() + seconds), assets=profile['assets'])
    result_root = control._shared_path(Path(options.get('directory_root') or control.DEFAULT_DIRECTORY_ROOT), 'result directory')
    with _locked(path):
        if time.time() >= profile['submit_before'] or time.time() + seconds > profile['worker_deadline']:
            raise ValueError('pool submission cutoff/worker deadline would be exceeded')
        reservations = json.loads((path / 'reservations.json').read_text())
        live = [row for row in reservations.values() if row['state'] != 'released']
        if (sum(row['cpus'] for row in live) + cpus > profile['cpus']
                or sum(row['mem_mib'] for row in live) + memory_mib(mem) > profile['mem_mib']):
            raise ValueError('pool aggregate CPU/memory budget exhausted')
        reservations[dispatch_id] = dict(state='reserved', cpus=cpus, mem_mib=memory_mib(mem),
                                          reserved_at=time.time(), execution=execution,
                                          directory=str(result_root / dispatch_id))
        atomic_json(str(path / 'reservations.json'), reservations)
    try:
        report = control.submit(repo, ref, task_file, agent, executable, resources,
                                timeout=timeout, _execution=execution, **options)
    except Exception:
        # There may already be a staged or live srun after an interrupted call.
        # Retain budget; only exact scheduler terminal proof releases it.
        raise
    with _locked(path):
        reservations = json.loads((path / 'reservations.json').read_text())
        reservations[dispatch_id].update(directory=report['directory'], submission=report)
        atomic_json(str(path / 'reservations.json'), reservations)
    return report


def launch_step(directory, manifest, argv, report, timeout):
    started = time.monotonic()
    try:
        with (directory / 'srun-frontend.log').open('ab') as stream:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                                     start_new_session=True, close_fds=True)
    except OSError:
        report.update(state='submission_failed', reason='srun unavailable; reservation retained for explicit reconciliation')
    else:
        report.update(frontend_pid=child.pid, frontend_host=os.uname().nodename,
                      frontend_scope='detached submitter process; not worker signal ownership')
        while time.monotonic() - started < timeout:
            if adopt_step(directory, manifest, report):
                break
            if child.poll() is not None:
                report.update(reason='srun exited before trustworthy receiving-step receipt; reservation retained', frontend_exit=child.returncode)
                break
            time.sleep(0.1)
        else:
            report.update(reason='receiving-step receipt deadline expired; live/unknown submission retained, no retry')
    atomic_json(str(directory / 'submission.json'), report)
    return report


def adopt_step(directory, manifest, report):
    """Reconcile a late receiving receipt without resubmitting or inferring exit."""
    source, state = control._read_json(directory / 'step-source.json')
    execution = manifest['execution']
    if (state != 'present' or source.get('dispatch_id') != manifest['dispatch_id']
            or source.get('uid') != manifest['uid'] or source.get('job_id') != execution['job_id']
            or source.get('pool_id') != execution['pool_id'] or source.get('node') != execution['node']
            or source.get('cluster') != execution['cluster']
            or not re.fullmatch(r'[0-9]+', str(source.get('step_id', '')))):
        return False
    report.update(state='submitted', job_id=execution['job_id'], cluster=execution['cluster'], step_id=source['step_id'])
    existing, existing_state = control._read_json(directory / 'scheduler-source.json')
    if existing_state == 'missing':
        control._write_json(directory / 'scheduler-source.json', source)
    elif existing != source:
        raise ValueError('receiving step conflicts with previously recorded scheduler identity')
    atomic_json(str(directory / 'submission.json'), report)
    return True


def _worker_receipt(directory, profile):
    path, manifest = control._directory(directory)
    execution = manifest.get('execution', {})
    if (execution.get('pool_id') != profile['pool_id'] or execution.get('job_id') != profile['job_id']
            or execution.get('node') != profile['node'] or manifest['cluster'] != profile['cluster']):
        raise ValueError('worker does not belong to this pool')
    receipt, state = control._receipt(path, manifest)
    if receipt is not None and receipt.get('state') == 'submission_unknown':
        if adopt_step(path, manifest, receipt):
            receipt, state = control._receipt(path, manifest)
    if state != 'present' or receipt is None or not re.fullmatch(r'[0-9]+', str(receipt.get('step_id', ''))):
        raise ValueError('worker has no trustworthy numeric step receipt; do not signal parent job')
    return path, manifest, receipt


def status(directory, *, timeout=10):
    path, profile = _load(directory)
    with _locked(path):
        reservations = json.loads((path / 'reservations.json').read_text())
        for row in reservations.values():
            if row['state'] == 'released' or 'directory' not in row:
                continue
            try:
                _, _, receipt = _worker_receipt(row['directory'], profile)
                observation = control._scheduler(receipt, timeout)
            except (ValueError, OSError):
                continue
            row['scheduler'] = observation
            if observation.get('available') and observation.get('state', '').split()[0].rstrip('+') in TERMINAL:
                row.update(state='released', release_proof=observation, released_at=time.time())
        atomic_json(str(path / 'reservations.json'), reservations)
    return dict(pool=profile, reservations=reservations,
                used_cpus=sum(row['cpus'] for row in reservations.values() if row['state'] != 'released'),
                used_mem_mib=sum(row['mem_mib'] for row in reservations.values() if row['state'] != 'released'))


def stop_worker(directory, dispatch_id, *, timeout=10):
    path, profile = _load(directory)
    reservations = json.loads((path / 'reservations.json').read_text())
    row = reservations.get(dispatch_id)
    if row is None or 'directory' not in row:
        raise ValueError('unknown worker; no allocation signal attempted')
    _, _, receipt = _worker_receipt(row['directory'], profile)
    observation = control._scheduler(receipt, timeout)
    if not observation.get('available') or observation.get('state') not in ('RUNNING', 'COMPLETING', 'PENDING'):
        raise ValueError('cannot verify active owned step; no signal attempted')
    target = str(receipt['job_id']) + '.' + receipt['step_id']
    answer = control._run(['scancel', '--clusters=' + profile['cluster'], '--signal=TERM', target], timeout=timeout)
    if answer.returncode:
        raise ValueError('step stop request failed; reservation remains held')
    return dict(state='stop_requested', dispatch_id=dispatch_id, target=target,
                reservation='held until exact scheduler step terminal proof')


def observe_step(receipt, timeout):
    """Slurm step format differs from job format; never use job %T on squeue -s.

    Accounting can omit step UID. Accept it only with an exact parent row
    binding uid, node and original allocation StartTime plus receiving proof.
    """
    timeout = control._finite_seconds(timeout, 'scheduler timeout')
    job, step = str(receipt['job_id']), str(receipt['step_id'])
    source = receipt.get('step_source', {})
    node, start = receipt.get('node'), receipt.get('allocation_start')
    target = job + '.' + step
    if (not re.fullmatch(r'[0-9]+', step) or not receipt.get('cluster') or not node or not start
            or source.get('step_id') != step or source.get('uid') != receipt.get('uid')
            or source.get('job_id') != receipt['job_id'] or source.get('node') != node
            or type(source.get('started_at')) not in (int, float)):
        return dict(available=False, state='unknown_source', reason='step/parent incarnation proof incomplete')
    deadline = time.monotonic() + timeout
    cluster = '--clusters=' + receipt['cluster']
    def remaining():
        return max(.001, deadline - time.monotonic())
    try:
        query = control._run(['scontrol', cluster, '--oneliner', 'show', 'step', target], timeout=remaining())
        if not query.returncode:
            fields = dict(re.findall(r'(\w+)=([^\s]+)', query.stdout))
            owner = re.search(r'(?:^|\()([0-9]+)\)?$', fields.get('UserId', ''))
            try:
                live_start = datetime.fromisoformat(fields.get('StartTime', '')).timestamp()
                parent_start = datetime.fromisoformat(start).timestamp()
                live_incarnation = parent_start <= live_start <= source['started_at'] + 2
            except ValueError:
                live_incarnation = False
            if (fields.get('StepId') == target and owner and int(owner[1]) == receipt['uid']
                    and fields.get('NodeList') == node and live_incarnation
                    and fields.get('State') in ('RUNNING', 'COMPLETING', 'PENDING')):
                # Require the currently live parent incarnation too; a stale
                # private receipt cannot authorize a recycled job/step number.
                parent = control._run(['scontrol', cluster, '--oneliner', 'show', 'job', job], timeout=remaining())
                facts = dict(re.findall(r'(\w+)=([^\s]+)', parent.stdout))
                parent_uid = re.search(r'\(([0-9]+)\)$', facts.get('UserId', ''))
                if (not parent.returncode and facts.get('JobId') == job and parent_uid
                        and int(parent_uid[1]) == receipt['uid'] and facts.get('NodeList') == node
                        and facts.get('StartTime') == start):
                    return dict(available=True, state=fields['State'], node=node, source='scontrol-step', target=target)
        if time.monotonic() >= deadline:
            return dict(available=False, state='unavailable')
        query = control._run(['sacct', cluster, '--jobs=' + job, '--noheader', '--parsable2',
                              '--format=JobIDRaw,UID,State,ExitCode,NodeList,Start'], timeout=remaining())
        if query.returncode:
            return dict(available=False, state='unavailable')
        rows = [line.strip().split('|') for line in query.stdout.splitlines()]
        parents = [row for row in rows if len(row) >= 6 and row[0] == job
                   and row[1] == str(receipt['uid']) and row[4] == node and row[5] == start]
        if len(parents) != 1:
            return dict(available=False, state='unknown_source', reason='accounting parent identity not proven')
        exact = [row for row in rows if len(row) >= 6 and row[0] == target]
        if len(exact) == 1:
            row = exact[0]
            try:
                step_start = datetime.fromisoformat(row[5]).timestamp()
                parent_start = datetime.fromisoformat(start).timestamp()
            except ValueError:
                return dict(available=False, state='unknown_source', reason='step start time unavailable')
            if (row[1] in ('', str(receipt['uid'])) and row[4] == node
                    and parent_start <= step_start <= source['started_at'] + 2):
                return dict(available=True, state=row[2], exit_code=row[3], node=node,
                            source='sacct-step-with-parent', target=target, allocation_start=start, step_start=row[5])
        return dict(available=True, state='unknown', reason='exact step incarnation absent; no result inferred')
    except (OSError, subprocess.TimeoutExpired):
        return dict(available=False, state='unavailable')
