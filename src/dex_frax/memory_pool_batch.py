"""Run verified Fast and Refine A manifests with joint CPU and memory admission."""
import argparse
from collections import Counter, deque
import contextlib
from hashlib import file_digest, sha256
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import signal
from statistics import median
import subprocess
import sys
import time
import traceback

from . import resources
from .dynamic_resources import CpuAdmission
from .memory_pool import BASE, DANGER, GIB, RECOVER, START, JointPool, Lease, fast_warm_grant, requested_total, set_workspace_gate

FAST_HOST_RESERVE_GIB = 2.
IDLE_TIMEOUT_SECONDS = 60.


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def arguments(job, output=None):
    return dict(job['options'], input=job['input'], output=output, source=job['source'],
                mode='frax', profile=job['profile'], refine_flow='A', max_frames=None,
                unified_repair=True, continue_from=job.get('checkpoint'),
                resume_same_round=job.get('resume_same_round', False),
                ground_running=job.get('ground_running', False))


def publications(folder, job):
    from .robot_profile import MODEL_XML

    try:
        run = read(folder/'run.json')
        quality = read(folder/'quality.json')
        identity = read(folder/'identity.json')
        if run['status'] != 'complete' or run['quality'] not in ('accepted','accepted_with_notes','review'):
            return None
        contract = identity['conversion_contract']
        profile = job['profile']
        if contract.get('profile') != profile:
            return None
        if profile == 'fast' and quality['status'] != 'review':
            return None
        if profile == 'refine' and contract.get('refine_flow') != 'A':
            return None
        if (run['quality'] != quality['status'] or identity['input_sha256'] != job['input_sha256']
                or identity['assets'] != job['assets']
                or quality['model_sha256'] != job['assets'][MODEL_XML]
                or run['fingerprint'] != identity['fingerprint']):
            return None
        files = ['quality.json', 'motion.npz', 'motion.pkl']
        training = quality['status'] == 'accepted'
        if (folder/'training.npz').is_file() != training or (training and quality['flags']):
            return None
        if training:
            files.append('training.npz')
        if not all(sha(folder/f) == run['files'][f] for f in files):
            return None
        return dict(status=quality['status'], folder=str(folder), frames=quality['frames'],
                    training=training, flags=quality['flags'],
                    collision_samples=quality['full_self_collision']['collision_samples'])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def verified_seed(folder, job):
    if job['profile'] == 'fast':
        return None, 'Fast does not use refine checkpoints'
    from .cli import identity
    from .refine_resume import locate, verify
    try:
        checkpoint = locate(folder)
        args = argparse.Namespace(**arguments(job))
        if job['source'] == 'smplx':
            args.smplx_prepared = next((p for p in (checkpoint/'input/human.npz',
                                        checkpoint.parent/'input/human.npz') if p.is_file()), None)
            if args.smplx_prepared is None:
                raise ValueError('Continuation SMPL-X prepared geometry missing')
        state = verify(checkpoint, identity(args))
        return str(checkpoint), state
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, str(error)


def prepare(args, content):
    from .cascade import load_manifest

    options = {key: getattr(args, key) for key in (
        'bvh_units', 'bvh_translations', 'foot_contact', 'iterations',
        'initial_iterations', 'allow_local_slowdown', 'repair_budget', 'grounding',
        'input_fps', 'up_axis', 'units', 'smplx_python')}
    options['body_models'] = (str(Path(args.body_models).expanduser().resolve())
                              if args.body_models is not None else None)
    rows = load_manifest(args.manifest, content, allow_smplx=True)
    jobs = []
    for row in rows:
        job = {key: row[key] for key in ('input', 'input_sha256', 'source')}
        job.update(key=row['id'], profile=args.mode, options=options, ground_running=row.get('ground_running', False),
                   audit_workers=args.audit_workers)
        if job['source'] == 'auto' and Path(job['input']).suffix.lower() in ('.pt', '.pkl', '.npz'):
            job['source'] = 'smplx'
        if job['source']=='smplx':
            # The adapter publishes the exact output length after preparation.
            job['frames']=12000
        else:
            job['frames']=resources.output_frames(row['input'])
        if 'continue_from' in row:
            if not isinstance(row['continue_from'], str) or not row['continue_from']:
                raise ValueError('continue_from must be a path for '+row['id'])
            checkpoint, reason = verified_seed(args.manifest.parent/row['continue_from'], job)
            if not checkpoint:
                raise ValueError(f"Invalid checkpoint for {row['id']}: {reason}")
            job['checkpoint']=checkpoint
        jobs.append(job)
    return jobs, options


def worker(connection, job, folder, cpu):
    os.setsid()
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {cpu})
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
                      CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_ENABLE_X64='1')
    os.environ.setdefault('DEX_FRAX_COMPILE_LOCK_DIR',str(resources.state_root()/'compiler'))
    os.environ.setdefault('JAX_COMPILATION_CACHE_DIR',str(resources.state_root()/'jax_cache'))
    os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS','0')
    os.environ.setdefault('DEX_FRAX_KERNEL_CACHE_SIZE','2')
    os.environ['DEX_FRAX_BLOCK_MEMORY']='1'
    os.environ['DEX_SCHEDULE_MEMORY_BYTES']=str(BASE)  # enables safe specialist boundary events
    from .cli import convert
    from .resources import process_memory, release_memory
    from .resource_events import set_gate, phase
    resident_models = None
    reusable = job['profile']=='fast'
    last_trim, last_phase = 0., None
    workspace_active = False

    def gate(event):
        nonlocal last_trim,last_phase
        if workspace_active:
            raise RuntimeError('Return the operator workspace before entering a gated phase')
        stage = event['phase']
        now = time.monotonic()
        if stage != last_phase or now-last_trim >= 15:
            release_memory()
            resident = process_memory(os.getpid())
            if resident:
                if event.get('required_peak_bytes') is not None:
                    event['required_peak_bytes'] += resident-event['pss_bytes']
                event['pss_bytes'] = resident
            last_trim = now
        last_phase = stage
        connection.send(dict(request=event))
        if connection.recv() != 'grant':
            raise InterruptedError('Shared pool worker yielded at a safe boundary')

    def workspace_gate(cores=None, workspace_bytes=0):
        nonlocal workspace_active
        if cores is None:
            release_memory()
            connection.send(dict(workspace_returned=True))
            workspace_active = False
            return ()
        if workspace_active:
            return ()
        connection.send(dict(workspace_request=dict(cores=cores, workspace_bytes=workspace_bytes,
                             cgroup_headroom=resources.memory()['cgroup_headroom'])))
        cpus = connection.recv()
        workspace_active = bool(cpus)
        return cpus

    set_gate(gate)
    try:
        while True:
            last_trim, last_phase = 0., None
            set_workspace_gate(workspace_gate if job['audit_workers']==2 else None)
            with (folder.parent/'worker.log').open('w', buffering=1) as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    try:
                        phase('loading')
                        if sha(Path(job['input'])) != job['input_sha256']:
                            raise ValueError('Input changed after manifest verification')
                        model_s = 0.
                        reused_model = resident_models is not None
                        if not reused_model:
                            from .model_cache import enable_audit_models, prepare_audit_models
                            models = job['models']
                            if models is None:
                                cache_started = time.perf_counter()
                                models = prepare_audit_models(job['assets'], resources.state_root()/'model_cache')
                                connection.send(dict(model_cache=dict(preflight_s=time.perf_counter()-cache_started,
                                                                      models=models)))
                            model_started = time.perf_counter()
                            enable_audit_models(models)
                            resident_models = models
                            model_s = time.perf_counter()-model_started
                        write(folder.parent/'model_load.json',dict(base_mjb_s=model_s, reused_in_worker=reused_model))
                        from .v8.audit import set_geometry_timing_path
                        set_geometry_timing_path(folder.parent/'geometry_timings.jsonl')
                        if job['source']=='smplx' and not job.get('checkpoint'):
                            phase('preprocessing')
                        convert(argparse.Namespace(**arguments(job,str(folder))))
                    except Exception as error:
                        traceback.print_exc()
                        connection.send(dict(result='failed',error=repr(error)))
                        return
            if reusable:
                release_memory()
            connection.send(dict(result='returned'))
            if not reusable:
                break
            assignment = connection.recv()
            if assignment is None:
                break
            job, folder = assignment
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


def tree_pids(pid):
    pending = [pid]
    while pending:
        current = pending.pop()
        yield current
        try:
            pending.extend(map(int,Path(f'/proc/{current}/task/{current}/children').read_text().split()))
        except OSError:
            pass


def tree_pss(pid):
    if sys.platform == 'darwin':
        # macOS reports RSS, so count the helper and worker from one snapshot.
        try:
            rows = subprocess.check_output(['ps', '-ax', '-o', 'pid=,ppid=,rss='], text=True)
        except (OSError, subprocess.SubprocessError):
            return resources.process_memory(pid)
        children = {}
        resident = {}
        for row in rows.splitlines():
            child, parent, rss = map(int, row.split())
            children.setdefault(parent, []).append(child)
            resident[child] = rss * 1024
        pending = [pid]
        total = 0
        while pending:
            current = pending.pop()
            total += resident.get(current, 0)
            pending.extend(children.get(current, ()))
        return total
    return sum(resources.process_memory(current) for current in tree_pids(pid))


def pressure_high():
    try:
        for line in Path('/proc/pressure/memory').read_text().splitlines():
            if line.startswith('full '):
                return float(line.split()[1].split('=')[1]) >= 2.
    except (OSError, ValueError):
        pass
    return False


def close(worker):
    process = worker['process']
    process.join(.3)
    if process.is_alive():
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid,signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        process.join(2)
    if process.is_alive():
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            process.kill()
        process.join(2)
    if process.is_alive():
        raise RuntimeError(f'Worker process did not exit: {process.pid}')
    if worker['lease'].borrowed_cpus:
        # A crashed worker may leave helpers in its process group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    worker['pipe'].close()
    process.close()


def grant_request(worker, added, exclusive):
    """Release deadlock recovery priority when its pending grant is delivered."""
    try:
        worker['pipe'].send('grant')
    except (BrokenPipeError, EOFError, OSError):
        return False, exclusive
    worker['lease'].grant += added
    worker['request'] = None
    if exclusive == worker['job']['key']:
        exclusive = None
    return True, exclusive


def reply_workspace(worker, cpus):
    request = worker['workspace_request']
    try:
        worker['pipe'].send(cpus)
    except (BrokenPipeError, EOFError, OSError):
        return False
    worker['workspace_request'] = None
    if cpus:
        worker['lease'].borrow(cpus, request['workspace_bytes'])
    return True


def wait_reason(request, compiler_owner):
    if request is None:
        return None
    if compiler_owner is not None and request['phase']=='waiting_compile' and request.get('compile_possible'):
        return 'compiler'
    return 'memory'


def close_idle(worker):
    try:
        worker['pipe'].send(None)
    except (BrokenPipeError, EOFError, OSError):
        pass
    close(worker)


def cold_start_due(pending, active, now, last_dispatch, warmup_s):
    """With one wave of work left, give an existing process time to become ready."""
    return active == 0 or pending > active or now - last_dispatch >= warmup_s


def run(args):
    from .paths import asset_fingerprints

    root = args.output.resolve()
    with resources.batch_lock():
        content = args.manifest.read_bytes()
        jobs, options = prepare(args, content)
        snapshot = resources.memory()
        watermarks = tuple(round(value * GIB) for value in args.memory_watermarks)
        if snapshot['total'] < watermarks[2]+BASE:
            raise ValueError('Host memory cannot supply a worker and the configured startup reserve')
        limit = snapshot.get('cgroup_limit')
        if limit is not None and limit < BASE+GIB:
            raise ValueError('Cgroup limit cannot supply a worker and the 1 GiB startup reserve')
        spec = dict(schema=1, manifest_sha256=sha256(content).hexdigest(), options=options)
        if args.mode == 'fast':
            # Refine A batches written before mode selection keep the same manifest contract.
            spec['mode'] = 'fast'
        if args.resume:
            if read(root/'pool_batch.json') != spec:
                raise ValueError('Resume requires the original manifest and conversion options')
        else:
            root.mkdir(parents=True, exist_ok=False)
            write(root/'pool_batch.json', spec)
        assets = asset_fingerprints()
        for job in jobs:
            job['assets'] = assets
        snapshot = resources.memory()
        return schedule(jobs, root, args.ramp, resources.workers(args.workers), snapshot, watermarks)


def schedule(jobs, root, ramp, workers, initial_memory, watermarks=(DANGER, RECOVER, START)):
    ctx = mp.get_context('spawn')
    cpu_groups = resources.cpu_groups()
    pool = JointPool(cpu_groups, watermarks)
    cpu_admission = CpuAdmission((cpu for core in cpu_groups for cpu in core), resources.cpu_count())
    completed = {}
    active = {}
    idle = {}
    pending = deque()
    events = Counter()
    retry_count = Counter()
    oom_count = initial_memory.get('cgroup_oom_kills',0)
    stopped = False
    state = 'failed'
    models = None

    def stop(*_):
        nonlocal stopped
        stopped = True

    def event(kind, **data):
        events[kind] += 1
        with (root/'memory_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(),event=kind,**data))+'\n')

    def retire_idle(cpu, reason):
        w = idle.pop(cpu)
        close_idle(w)
        event('idle_reclaim',cpu=cpu,reason=reason,resident_bytes=w['lease'].resident)

    def status(state, snapshot):
        unused = sum(w['lease'].unused for w in active.values())
        write(root/'status.json',dict(state=state,updated_at=time.time(),pending=[j['key'] for j in pending],
              running={key:dict(pid=w['process'].pid,phase=w['phase'],loan_bytes=max(0,w['lease'].grant-BASE),
                                cpu=w['lease'].cpu, borrowed_cpus=w['lease'].borrowed_cpus,
                                grant_bytes=w['lease'].grant, unused_bytes=w['lease'].unused,
                                resident_bytes=w['lease'].resident, max_resident_bytes=w['max_resident'],
                                waiting=w['request'] is not None,wait_reason=wait_reason(w['request'], compiler_owner),
                                folder=str(w['folder']),
                                checkpoint=w['job'].get('checkpoint')) for key,w in active.items()},
              idle={w['process'].pid:dict(cpu=cpu,resident_bytes=w['lease'].resident) for cpu,w in idle.items()},
              complete=completed,counts=dict(Counter(v['status'] for v in completed.values())),
              memory=snapshot,pool=dict(base_bytes=BASE,unused_bytes=unused,frozen=pool.frozen,
                                        initial_host_available_bytes=initial_memory['available'],
                                        initial_host_budget_bytes=max(0, initial_memory['available']-pool.start),
                                        exclusive_worker=exclusive,
                                        watermarks_gib=dict(danger=pool.danger/GIB, recover=pool.recover/GIB,
                                                            startup=pool.start/GIB),
                                        host_remaining_bytes=snapshot['available']-unused,
                                        cgroup_remaining_bytes=snapshot['cgroup_headroom']-unused),
              cpu=dict(cpu_admission.report(), max_workers=workers),
              counters=dict(events),max_running=max_active,minimum_host_available_bytes=min_available))

    for job in jobs:
        attempts = sorted((root/job['key']).glob('attempt_*'),reverse=True)
        latest = next((value for a in attempts if (value := publications(a/'frax',job)) is not None),None)
        if latest is not None:
            completed[job['key']] = latest
            continue
        for a in attempts:
            if not (a/'frax/refine_state.json').is_file():
                continue
            checkpoint, reason = verified_seed(a/'frax',job)
            if checkpoint:
                job.update(checkpoint=checkpoint, resume_same_round=True)
                break
            event('checkpoint_rejected', key=job['key'], folder=str(a/'frax'), reason=reason)
        pending.append(job)
    compiler_owner = None
    grant_frozen = False
    last_pressure_sample = 0.
    next_start = 0.
    last_dispatch = float('-inf')
    cold_warmups = {}
    previous_write = 0.
    all_wait_since = None
    exclusive = None
    max_active = 0
    min_available = initial_memory['available']
    handlers = {sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}

    try:
        while (active or pending) and not stopped:
            now = time.monotonic()
            for key,w in list(active.items()):
                result = None
                while w['pipe'].poll():
                    try:
                        msg = w['pipe'].recv()
                    except (EOFError, OSError):
                        break
                    if 'request' in msg:
                        w['request'] = msg['request']
                        w['phase'] = w['request']['phase']
                        if w['cold_start'] is not None and w['phase'] == 'ik_block':
                            seconds = now - w['cold_start']
                            cold_warmups.setdefault(w['job'].get('source'), deque(maxlen=5)).append(seconds)
                            event('cold_ik_ready', key=key, seconds=seconds)
                            w['cold_start'] = None
                        if compiler_owner == key and w['phase'] != 'waiting_compile':
                            compiler_owner = None
                        w['lease'].resident = max(w['request']['pss_bytes'],tree_pss(w['process'].pid))
                        lease = w['lease']
                        floor = BASE
                        if w['job'].get('profile') == 'fast' and lease.resident > 0:
                            # Already admitted loading credit belongs to this job. At
                            # later safe boundaries the paused worker can release it.
                            floor = lease.grant if w['phase'] == 'loading' else 0
                        returned = lease.return_unused(floor)
                        if returned:
                            event('return_unused',key=key,phase=w['phase'],bytes=returned)
                        w['wait_started'] = now
                    elif 'model_cache' in msg:
                        models = msg['model_cache']['models']
                        write(root/'model_cache.json', msg['model_cache'])
                        event('model_cache_ready', key=key)
                    elif 'workspace_request' in msg:
                        w['workspace_request'] = msg['workspace_request']
                    elif 'workspace_returned' in msg:
                        lease = w['lease']
                        if lease.borrowed_cpus:
                            lease.resident = tree_pss(w['process'].pid)
                            cpus = lease.borrowed_cpus
                            returned = lease.return_workspace()
                            event('workspace_return', key=key, cpus=cpus, unused_bytes=returned,
                                  resident_bytes=lease.resident)
                    else:
                        result = msg
                        break
                if result is not None or not w['process'].is_alive():
                    code = w['process'].exitcode
                    loan = max(0,w['lease'].grant-BASE)
                    del active[key]
                    if compiler_owner == key:
                        compiler_owner = None
                    if exclusive == key:
                        exclusive = None
                    q = publications(w['folder'],w['job'])
                    reusable = (w['job'].get('profile')=='fast' and (result or {}).get('result')=='returned'
                                and q is not None and not w['lease'].borrowed_cpus and w['process'].is_alive())
                    if reusable:
                        cpu = w['lease'].cpu
                        # Idle resident pages are already charged in host and cgroup usage.
                        w['lease'] = Lease(grant=0,resident=tree_pss(w['process'].pid),cpu=cpu)
                        w['idle_since'] = now
                        idle[cpu] = w
                    else:
                        close(w)
                    if q is None and code == -signal.SIGKILL and not stopped:
                        observed_oom = resources.memory().get('cgroup_oom_kills',0)
                        if observed_oom > oom_count or pressure_high():
                            oom_count = observed_oom
                            retry_count[key] += 1
                            if retry_count[key] <= 2:
                                job = w['job']
                                checkpoint,_ = verified_seed(w['folder'],job)
                                if checkpoint:
                                    job.update(checkpoint=checkpoint, resume_same_round=True)
                                pending.appendleft(job)
                                next_start = now + max(ramp,10.)
                                event('oom_retry',key=key,checkpoint=job.get('checkpoint'))
                                continue
                    if q is None:
                        error = (result or {}).get('error') or f'Worker exited with code {code}'
                        q = dict(status='failed',folder=str(w['folder']),error=error)
                    completed[key] = dict(q, resumed_from=w['job'].get('checkpoint'))
                    event('worker_idle' if reusable else 'worker_exit',key=key,status=q['status'],base_bytes=BASE,loan_bytes=loan,
                          cpu=w['lease'].cpu, borrowed_cpus=w['lease'].borrowed_cpus)
                else:
                    live = tree_pss(w['process'].pid)
                    w['lease'].resident = max(live, (w['request'] or {}).get('pss_bytes', 0))
                    w['max_resident'] = max(w['max_resident'],w['lease'].resident)

            for cpu,w in list(idle.items()):
                if not pending or not w['process'].is_alive():
                    retire_idle(cpu,'queue_drained' if not pending else 'worker_exited')
                else:
                    w['lease'].resident = tree_pss(w['process'].pid)
            snapshot = resources.memory()
            cpu_limit = cpu_admission.sample(pid for w in active.values() for pid in tree_pids(w['process'].pid))
            min_available = min(min_available,snapshot['available'])
            leases = [w['lease'] for w in active.values()]
            host,group = pool.remaining(snapshot,leases)
            if not pool.frozen or now-last_pressure_sample >= .45:
                host,group = pool.update(snapshot,leases,pressure=pressure_high())
                last_pressure_sample = now
            if grant_frozen != pool.frozen:
                grant_frozen = pool.frozen
                event('freeze' if pool.frozen else 'recover',host_bytes=host,cgroup_bytes=group)
            waiting = sorted((w for w in active.values() if w['request'] is not None),
                             key=lambda w: w['wait_started'])
            for w in waiting:
                request = w['request']
                key = w['job']['key']
                compiling = request['phase']=='waiting_compile' and request.get('compile_possible')
                if compiling and compiler_owner is not None:
                    continue
                lease = w['lease']
                job = w['job']
                required = requested_total(request, job['frames'], profile=job.get('profile', 'refine'),
                                           warm=w['warm'], source=job.get('source'), resident=lease.resident)
                added = max(0,required-lease.grant)
                amount = max(0,required-max(lease.grant,lease.resident))
                if amount and not pool.allows(snapshot,leases,amount):
                    continue
                was_exclusive = exclusive == key
                granted, exclusive = grant_request(w, added, exclusive)
                if not granted:
                    continue
                if compiling:
                    compiler_owner = key
                if was_exclusive:
                    event('exclusive_released',key=key,phase=w['phase'])
                if amount:
                    after, _ = pool.remaining(snapshot,leases)
                    event('grant',key=key,phase=w['phase'],bytes=amount,after_host_bytes=after)

            blocked = [w for w in active.values() if w['request'] is not None]
            waiting_for_memory = any(wait_reason(w['request'], compiler_owner)=='memory' for w in blocked)
            if idle and (pool.frozen or waiting_for_memory):
                cpu = max(idle,key=lambda cpu:idle[cpu]['lease'].resident)
                retire_idle(cpu,'memory_pressure')
                continue  # Re-measure the freed pages before making another grant.
            if blocked and len(active) == len(blocked) and len(active)>1:
                all_wait_since = all_wait_since or now
                if now-all_wait_since >= 12:
                    head,*others = sorted(blocked,key=lambda w: w['wait_started'])
                    victim = max(others,key=lambda w: w['lease'].resident)
                    job = victim['job']
                    key = job['key']
                    checkpoint,_ = verified_seed(victim['folder'],job)
                    close(victim)
                    del active[key]
                    if key==compiler_owner:
                        compiler_owner=None
                    if checkpoint:
                        job.update(checkpoint=checkpoint, resume_same_round=True)
                    next_start=now+ramp
                    exclusive=head['job']['key']
                    pending.appendleft(job)
                    event('yield',key=key,checkpoint=job.get('checkpoint'),for_job=exclusive)
                    all_wait_since=None
            else:
                all_wait_since=None

            cache_pending = models is None and active
            can_start = pending and not exclusive and not waiting_for_memory and not cache_pending and len(active)<workers
            free_cpus = pool.free_cpus(leases,cpu_limit) if can_start else []
            warm_cpu = None
            startup_grant = BASE
            if free_cpus and idle:
                # cpu_limit caps the count, not the IDs on which existing processes
                # may run. A hot worker can be bound to any unoccupied eligible CPU.
                for candidate in pool.free_cpus(leases, sum(map(len, cpu_groups))):
                    if candidate not in idle:
                        continue
                    resident = idle[candidate]['lease'].resident
                    hot_grant = fast_warm_grant(resident, pending[0]['frames'])
                    if pool.allows(snapshot, leases, max(0, hot_grant-resident), start=True):
                        warm_cpu, startup_grant = candidate, hot_grant
                        break
            cpu = warm_cpu
            if cpu is None and free_cpus and now>=next_start:
                timings = cold_warmups.get(pending[0].get('source')) if pending[0].get('profile') == 'fast' else None
                warmup_s = min(median(timings), IDLE_TIMEOUT_SECONDS) if timings else 0.
                if (cold_start_due(len(pending), len(active), now, last_dispatch, warmup_s)
                        and pool.allows(snapshot,leases,BASE,start=True)):
                    cpu = next((candidate for candidate in free_cpus if candidate not in idle),None)
            if cpu is not None:
                job=pending[0]
                key=job['key']
                folder=root/key/f'attempt_{time.time_ns()}'/'frax'
                folder.parent.mkdir(parents=True)
                resident = 0
                if warm_cpu is not None:
                    w = idle[cpu]
                    process, connection = w['process'], w['pipe']
                    try:
                        connection.send((job,folder))
                    except (BrokenPipeError, EOFError, OSError):
                        retire_idle(cpu,'worker_exited')
                        continue
                    resident = idle.pop(cpu)['lease'].resident
                else:
                    connection,endpoint=ctx.Pipe()
                    process=ctx.Process(target=worker,args=(endpoint,dict(job,models=models),folder,cpu))
                    try:
                        process.start()
                    except BaseException:
                        connection.close()
                        raise
                    finally:
                        endpoint.close()
                    next_start=now+ramp
                pending.popleft()
                last_dispatch = now
                lease = Lease(cpu=cpu,resident=resident,grant=startup_grant)
                active[key]=dict(job=job,folder=folder,process=process,pipe=connection,
                                 phase='starting',lease=lease,request=None,max_resident=resident,warm=warm_cpu is not None,
                                 cold_start=now if warm_cpu is None and job.get('profile')=='fast' else None,
                                 workspace_request=None)
                leases.append(lease)
                max_active=max(max_active,len(active))
                event('worker_reuse' if warm_cpu is not None else 'worker_start',
                      key=key,cpu=cpu,base_bytes=startup_grant,pid=process.pid,
                      compiler_waiters=len(blocked),checkpoint=job.get('checkpoint'))

            for idle_cpu, w in list(idle.items()):
                if now - w['idle_since'] >= IDLE_TIMEOUT_SECONDS:
                    retire_idle(idle_cpu, 'idle_timeout')

            for key, w in active.items():
                request = w['workspace_request']
                if request is None:
                    continue
                cpus = () if pending or blocked or exclusive else pool.workspace_cpus(
                    snapshot, leases, cpu_limit, w['lease'], **request)
                if reply_workspace(w, cpus):
                    event('workspace_grant' if cpus else 'workspace_serial', key=key, cpus=cpus,
                          bytes=request['workspace_bytes'] if cpus else 0)

            if now-previous_write>=2:
                status('running',snapshot)
                previous_write=now
            if active:
                wait([w['pipe'] for w in active.values()],timeout=.45)
            else:
                time.sleep(.45)
        state = 'interrupted' if stopped else 'complete'
    finally:
        for cpu in list(idle):
            retire_idle(cpu,'batch_finished')
        for key, w in list(active.items()):
            close(w)
            del active[key]
            pending.appendleft(w['job'])
            event('interrupted',key=key)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        status(state, resources.memory())
    return state, completed


def main():
    from .cli import add_repair_budget_option
    from .contract import DEFAULT_FOOT_CONTACT
    from .smplx_cli import add_options

    parser = argparse.ArgumentParser(description='Run Fast or Refine A batches with a shared CPU and memory pool')
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--mode', choices=('fast', 'refine'), default='refine')
    parser.add_argument('--workers', default='auto', help='Maximum workers; memory admission may use fewer')
    parser.add_argument('--ramp', type=float, default=2., help='Minimum seconds between new workers')
    parser.add_argument('--audit-workers', type=int, choices=(1, 2), default=1,
                        help='Use one CPU or request a second CPU for full-trajectory geometry audits')
    parser.add_argument('--memory-watermarks', type=float, nargs=3, metavar=('DANGER', 'RECOVER', 'START'),
                        help='Host memory reserves in GiB (Fast: 2 each; Refine: 8 10 12)')
    parser.add_argument('--resume', action='store_true', help='Verify previous exports and resume from available checkpoints')
    parser.add_argument('--bvh-units', choices=['cm','m','mm'], default='cm')
    parser.add_argument('--bvh-translations', choices=['absolute','offset'], default='absolute')
    parser.add_argument('--foot-contact', choices=['v1','v2'], default=DEFAULT_FOOT_CONTACT)
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--initial-iterations', type=int, default=100)
    parser.add_argument('--allow-local-slowdown', action=argparse.BooleanOptionalAction, default=None)
    add_repair_budget_option(parser)
    add_options(parser)
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    if args.allow_local_slowdown is None:
        args.allow_local_slowdown = args.mode == 'refine'
    if args.mode == 'fast' and (args.allow_local_slowdown or args.repair_budget is not None):
        parser.error('--allow-local-slowdown and --repair-budget require --mode refine')
    if not math.isfinite(args.ramp) or args.ramp<=0 or min(args.iterations,args.initial_iterations)<1:
        parser.error('Worker start interval and iteration counts must be positive')
    if args.memory_watermarks is None:
        args.memory_watermarks = ((FAST_HOST_RESERVE_GIB,)*3 if args.mode == 'fast'
                                  else (DANGER/GIB, RECOVER/GIB, START/GIB))
    else:
        danger, recover, start = args.memory_watermarks
        if not 0 < danger < recover < start < math.inf:
            parser.error('Memory watermarks must be finite, positive and strictly increasing')
    if args.input_fps is not None and (not math.isfinite(args.input_fps) or args.input_fps <= 0):
        parser.error('SMPL-X input fps must be finite and positive')
    try:
        state, completed = run(args)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f'Shared memory batch failed: {error}\n')
    if state!='complete' or any(v['status']=='failed' for v in completed.values()):
        parser.exit(1, f'Shared memory batch ended with {state}. See status.json.\n')


if __name__=='__main__':
    main()
