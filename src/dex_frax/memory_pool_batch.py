"""Run a verified motion manifest with the Refine A shared memory pool."""
import argparse
from collections import Counter, deque
from hashlib import file_digest, sha256
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import signal
import time
import traceback

from . import resources
from .memory_pool import BASE, GIB, START, Lease, Pool, requested_total


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
                mode='frax', profile='refine', refine_flow='A', max_frames=None,
                unified_repair=True, continue_from=job.get('checkpoint'),
                resume_same_round=job.get('resume_same_round', False),
                ground_running=job.get('ground_running', False))


def publications(folder, job):
    try:
        run = read(folder/'run.json')
        quality = read(folder/'quality.json')
        identity = read(folder/'identity.json')
        if run['status'] != 'complete' or run['quality'] not in ('accepted','accepted_with_notes','review'):
            return None
        if (run['quality'] != quality['status'] or identity['input_sha256'] != job['input_sha256']
                or identity['conversion_contract']['refine_flow'] != 'A'
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
        job.update(key=row['id'], options=options, ground_running=row.get('ground_running', False))
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
    os.environ['DEX_FRAX_BLOCK_MEMORY']='1'
    os.environ['DEX_SCHEDULE_MEMORY_BYTES']=str(BASE)  # enables safe specialist boundary events
    from .cli import convert
    from .resources import process_memory, release_memory
    from .resource_events import set_gate, phase
    last_trim, last_phase = 0., None

    def gate(event):
        nonlocal last_trim,last_phase
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

    set_gate(gate)
    with (folder.parent/'worker.log').open('w', buffering=1) as log:
        import contextlib
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                if sha(Path(job['input'])) != job['input_sha256']:
                    raise ValueError('Input changed after manifest verification')
                phase('loading')
                if job['source'] == 'smplx' and not job.get('checkpoint'):
                    # The existing adapter allocates its CPU model in a child.
                    phase('preprocessing')
                convert(argparse.Namespace(**arguments(job,str(folder))))
                connection.send(dict(result='returned'))
            except Exception as error:
                traceback.print_exc()
                connection.send(dict(result='failed', error=repr(error)))
    connection.close()


def tree_pss(pid):
    from .resources import process_memory
    total = 0
    pending = [pid]
    while pending:
        current = pending.pop()
        total += process_memory(current)
        try:
            pending.extend(map(int,Path(f'/proc/{current}/task/{current}/children').read_text().split()))
        except OSError:
            pass
    return total


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
    worker['pipe'].close()
    process.close()


def run(args):
    root = args.output.resolve()
    with resources.batch_lock():
        content = args.manifest.read_bytes()
        jobs, options = prepare(args, content)
        snapshot = resources.memory()
        if snapshot['total'] < START+BASE:
            raise ValueError('Host memory cannot supply a worker and the 12 GiB startup reserve')
        limit = snapshot.get('cgroup_limit')
        if limit is not None and limit < BASE+GIB:
            raise ValueError('Cgroup limit cannot supply a worker and the 1 GiB startup reserve')
        spec = dict(schema=1, manifest_sha256=sha256(content).hexdigest(), options=options)
        if args.resume:
            if read(root/'pool_batch.json') != spec:
                raise ValueError('Resume requires the original manifest and conversion options')
        else:
            root.mkdir(parents=True, exist_ok=False)
            write(root/'pool_batch.json', spec)
        return schedule(jobs, root, args.ramp, resources.workers(args.workers), snapshot)


def schedule(jobs, root, ramp, workers, initial_memory):
    ctx = mp.get_context('spawn')
    cpus = resources.cpu_order()[:workers]
    completed = {}
    active = {}
    pending = deque()
    events = Counter()
    retry_count = Counter()
    oom_count = initial_memory.get('cgroup_oom_kills',0)
    stopped = False
    state = 'failed'

    def stop(*_):
        nonlocal stopped
        stopped = True

    def event(kind, **data):
        events[kind] += 1
        with (root/'memory_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(),event=kind,**data))+'\n')

    def status(state, snapshot):
        unused = sum(w['lease'].unused for w in active.values())
        write(root/'status.json',dict(state=state,updated_at=time.time(),pending=[j['key'] for j in pending],
              running={key:dict(pid=w['process'].pid,phase=w['phase'],loan_bytes=max(0,w['lease'].grant-BASE),
                                resident_bytes=w['lease'].resident, max_resident_bytes=w['max_resident'],
                                waiting=w['request'] is not None,folder=str(w['folder']),
                                checkpoint=w['job'].get('checkpoint')) for key,w in active.items()},
              complete=completed,counts=dict(Counter(v['status'] for v in completed.values())),
              memory=snapshot,pool=dict(base_bytes=BASE,unused_bytes=unused,frozen=pool.frozen,
                                        host_remaining_bytes=snapshot['available']-unused),
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
    pool = Pool()
    compiler_owner = None
    grant_frozen = False
    last_pressure_sample = 0.
    next_start = 0.
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
                        if compiler_owner == key and w['phase'] != 'waiting_compile':
                            compiler_owner = None
                        w['lease'].resident = max(w['request']['pss_bytes'],tree_pss(w['process'].pid))
                        returned = w['lease'].return_unused()
                        if returned:
                            event('return_unused',key=key,phase=w['phase'],bytes=returned)
                        w['wait_started'] = now
                    else:
                        result = msg
                        break
                if result is not None or not w['process'].is_alive():
                    code = w['process'].exitcode
                    loan = max(0,w['lease'].grant-BASE)
                    close(w)
                    del active[key]
                    if compiler_owner == key:
                        compiler_owner = None
                    if exclusive == key:
                        exclusive = None
                    q = publications(w['folder'],w['job'])
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
                    event('worker_exit',key=key,status=q['status'],base_bytes=BASE,loan_bytes=loan)
                else:
                    live = tree_pss(w['process'].pid)
                    w['lease'].resident = max(live, (w['request'] or {}).get('pss_bytes', 0))
                    w['max_resident'] = max(w['max_resident'],w['lease'].resident)

            snapshot = resources.memory()
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
                required = requested_total(request,w['job']['frames'])
                added = max(0,required-lease.grant)
                amount = max(0,required-max(lease.grant,lease.resident))
                if added and not pool.allows(snapshot,leases,amount):
                    continue
                lease.grant += added
                if compiling:
                    compiler_owner = key
                try:
                    w['pipe'].send('grant')
                except (BrokenPipeError, EOFError, OSError):
                    lease.grant -= added
                    if compiling:
                        compiler_owner = None
                    continue
                w['request'] = None
                if amount:
                    after, _ = pool.remaining(snapshot,leases)
                    event('grant',key=key,phase=w['phase'],bytes=amount,after_host_bytes=after)

            blocked = [w for w in active.values() if w['request'] is not None]
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

            can_start = pending and not exclusive and not blocked and len(active)<len(cpus) and now>=next_start
            if can_start and pool.allows(snapshot,leases,BASE,start=True):
                job=pending.popleft()
                key=job['key']
                used={w['cpu'] for w in active.values()}
                cpu=next(c for c in cpus if c not in used)
                folder=root/key/f'attempt_{time.time_ns()}'/'frax'
                folder.parent.mkdir(parents=True)
                connection,endpoint=ctx.Pipe()
                process=ctx.Process(target=worker,args=(endpoint,job,folder,cpu))
                process.start()
                endpoint.close()
                active[key]=dict(job=job,folder=folder,process=process,pipe=connection,cpu=cpu,
                                 phase='starting',lease=Lease(),request=None,max_resident=0)
                max_active=max(max_active,len(active))
                next_start=now+ramp
                event('worker_start',key=key,base_bytes=BASE,pid=process.pid,checkpoint=job.get('checkpoint'))

            if now-previous_write>=2:
                status('running',snapshot)
                previous_write=now
            if active:
                wait([w['pipe'] for w in active.values()],timeout=.45)
            else:
                time.sleep(.45)
        state = 'interrupted' if stopped else 'complete'
    finally:
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

    parser = argparse.ArgumentParser(description='Run Refine A batches with the V3 shared memory pool')
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', default='auto', help='Maximum workers; memory admission may use fewer')
    parser.add_argument('--ramp', type=float, default=2., help='Minimum seconds between new workers')
    parser.add_argument('--resume', action='store_true', help='Verify previous exports and resume from available checkpoints')
    parser.add_argument('--bvh-units', choices=['cm','m','mm'], default='cm')
    parser.add_argument('--bvh-translations', choices=['absolute','offset'], default='absolute')
    parser.add_argument('--foot-contact', choices=['v1','v2'], default=DEFAULT_FOOT_CONTACT)
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--initial-iterations', type=int, default=100)
    parser.add_argument('--allow-local-slowdown', action=argparse.BooleanOptionalAction, default=True)
    add_repair_budget_option(parser)
    add_options(parser)
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    if not math.isfinite(args.ramp) or args.ramp<=0 or min(args.iterations,args.initial_iterations)<1:
        parser.error('Worker start interval and iteration counts must be positive')
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
