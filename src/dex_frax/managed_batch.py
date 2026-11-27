"""Two-stage batch cascade with live resource admission and one compiler."""
import gc
import heapq
from collections import Counter
import copy
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time

from . import resources as res
from .cascade import Worker,_worker,load_manifest,accepted_artifact,STAGES,preview_outcome
from .cli import atomic_json,sha,static_identity
from .resource_policy import ResourcePolicy
from .dispatch import duration_seconds,route,priority,deduplicate
from .memory_learning import MemoryLearning
from .memory_queue import MemoryQueue,pressure,victim as memory_victim
from .initial_memory import InitialMemory
from .stage_memory import StageMemory
from .contract import DEFAULT_FOOT_CONTACT, DEFAULT_REFINE_FLOW, conversion_contract
from .dispatch import pipeline_stages,use_unified

def child(connection,options):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key]='1'
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='1',
                      DEX_FRAX_KERNEL_CACHE_SIZE='2',DEX_FRAX_COMPILE_LOCK_DIR=options['compile_dir'],
                      JAX_COMPILATION_CACHE_DIR=options['persistent_cache_dir'],JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0',
                      DEX_FRAX_IK_BLOCK_FRAMES=str(options['block_frames']))
    if options.get('block_permits'):
        os.environ['DEX_FRAX_BLOCK_MEMORY']='1'
    if options.get('cpu') is not None and hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{options['cpu']})
    if options.get('dynamic'):res.lower_priority()
    from . import cli,model_cache
    cli._FROZEN_IDENTITY=options['identity']
    _worker(connection,initialize=lambda:model_cache.enable(options['model']))

class Slot(Worker):
    supports_stage_memory=True
    def __init__(self,options):
        super().__init__();self.options=options;self.jobs=0;self.started=0.;self.timeout=0.;self.telemetry={};self.phase_peaks={}
        self.memory_request=None;self.wait_started=None;self.waited_s=0.
    @property
    def pid(self):return self.process.pid if self.process is not None else None
    def start(self,job,timeout):
        if self.process is None or not self.process.is_alive():
            self.close();ctx=multiprocessing.get_context('spawn');self.connection,child_conn=ctx.Pipe()
            self.process=ctx.Process(target=child,args=(child_conn,self.options),daemon=True)
            self.process.start();child_conn.close()
        self.started=time.monotonic();self.timeout=timeout;self.connection.send(job)
        self.memory_request=None;self.wait_started=None;self.waited_s=0.
        self.frames_completed=0
        self.phase_peaks={};self.telemetry=dict(phase='starting',stage='loading',compile_possible=job['arguments']['mode']!='v8' and not job['arguments'].get('require_initial_reuse',False) and not job['arguments'].get('initial_from'),cache_state=self.telemetry.get('cache_state','pure'))
    def close(self):
        super().close()
        self.jobs=0;self.telemetry={};self.phase_peaks={}
        self.memory_request=None;self.wait_started=None;self.waited_s=0.

    def memory_wait_s(self):
        return self.waited_s+(time.monotonic()-self.wait_started if self.wait_started is not None else 0.)

    def grant_memory(self):
        self.connection.send(dict(memory_grant=True))
        if self.wait_started is not None:self.waited_s+=time.monotonic()-self.wait_started
        self.memory_request=None;self.wait_started=None

    def poll(self):
        while self.connection.poll(0):
            try:
                result=self.connection.recv()
                if 'memory_request' in result:
                    self.memory_request=result['memory_request'];self.wait_started=time.monotonic();continue
                if 'resource_event' in result:
                    event=result['resource_event'];key=(event['phase'] if event['phase'] in ('ik_prepare','ik_compile','waiting_compile') else event.get('stage',event['phase']))+'|'+event['cache_state']
                    self.phase_peaks[key]=max(self.phase_peaks.get(key,0),event['pss_bytes'])
                    self.frames_completed=max(self.frames_completed,event.get('frames_completed',0))
                    self.telemetry=event;continue
                result['worker_wall_s']=time.monotonic()-self.started;result['memory_wait_s']=self.memory_wait_s();self.jobs+=1;return result
            except (EOFError,OSError):
                self.process.join(.1);code=self.process.exitcode;self.close()
                return dict(status='failed',error_type='WorkerExit',exit_code=code,error='Owned worker exited before completing the export')
        if time.monotonic()-self.started-self.memory_wait_s()>=self.timeout:
            self.close();return dict(status='timeout',elapsed_s=time.monotonic()-self.started)
        return None

def timeout_for(args,mode,frames):
    explicit=getattr(args,mode+'_timeout',None)
    if explicit is None and getattr(args,'profile',None) in ('fast','refine'):
        return float('inf') if args.budget is None else args.budget
    return explicit if explicit is not None else (max(900.,300.+.5*frames) if mode=='frax' else max(1800.,600.+1.5*frames))

def finish(record,response,mode,folder,out,allow_collision_v8=False):
    attempt={**response,'mode':mode,'output':str(folder.relative_to(out))}
    try:selection=accepted_artifact(folder,attempt)
    except (OSError,ValueError,KeyError) as error:
        attempt.update(status='failed',error_type='InvalidExport',error=str(error));selection=None
    if attempt['status']!='complete':
        folder.mkdir(parents=True,exist_ok=True);(folder/'training.npz').unlink(missing_ok=True)
        atomic_json(folder/'run.json',{k:v for k,v in attempt.items() if k!='result'})
    if selection:
        record.update(status='accepted',selected_mode=mode,selected_output=str(folder.relative_to(out)),training_npz=str((folder/'training.npz').relative_to(out)),**selection)
    elif attempt['status']=='input_error':record['status']='input_error'
    else:record['status']='pending'
    result=attempt.pop('result',None)
    if result:
        attempt['conversion_wall_s']=result['total_wall_s'];attempt['initial_reuse']=(result.get('frax') or {}).get('initial_reuse')
        attempt['collision_repair_exhausted']=result.get('collision_repair_exhausted',False)
    record['attempts'].append(attempt)
    return attempt

def run(args,slot_factory=Slot,monitor=res.memory):
    # Shared lease is held across prepare, execution and final identity verification.
    with res.batch_lock():return _run(args,slot_factory,monitor)

def request_peak(request, actual, default_estimate):
    """Block leases must not inherit a previous whole-stage compile peak."""
    resident=max(int(actual),int(request.get('pss_bytes',0)))
    scope=request.get('memory_scope')
    if scope in ('initial_block','initial_finalize') and not request.get('first_block',False):
        extra=max(128*res.MIB,int(request.get('required_peak_bytes',0))-int(request.get('pss_bytes',0)))
        return resident+extra
    return max(default_estimate,int(request.get('required_peak_bytes',0)),int(resident*1.1)+128*res.MIB)


def inflight_forecast(slot, item, queue, source, frames, per_frame, learned_peak):
    """Release the whole-job retry floor once the worker reaches a safe boundary."""
    event = getattr(slot, 'telemetry', {}) or {}
    completed_block = (
        event.get('memory_scope') in ('initial_block', 'initial_finalize')
        and event.get('frames_completed', 0) > 0
    )
    post_initial = event.get('stage') in (
        'fast_refinement', 'local_repair', 'support', 'arms',
        'body_contacts', 'self_collision', 'export_check', 'audit', 'export'
    )
    if completed_block or post_initial:
        item['past_cold_start'] = True

    observed_peak = item['observed_peak']
    if item.get('past_cold_start'):
        # Keep both the length estimate and the locally measured high watermark.
        return max(learned_peak, observed_peak + 64*res.MIB)

    return max(observed_peak, queue.peak(item['mode'], source, frames, per_frame, learned_peak))


_SEGMENT_PHASES = {
    'initial': ('loading', 'preprocessing', 'initial', 'fast_refinement', 'export_check'),
    'repair': ('loading', 'preprocessing', 'local_repair', 'audit', 'export'),
}


def segmented_peak(learning, admission, mode, source, frames, segment, retry_queue=None, *, cache='jax_hot'):
    # A phase with a confirmed underestimate must also constrain admission of
    # the segment that will allocate it, before spawning a resident process.
    estimates = []
    for phase in _SEGMENT_PHASES[segment]:
        guess = learning.stage_peak(mode, source, cache, frames, admission, phase)
        if retry_queue is not None:
            per_frame = admission.value['stages'][mode]['bytes_per_output_frame']
            guess = retry_queue.peak(mode, source, frames, per_frame, guess, stage=phase)
        estimates.append(guess)
    return max(estimates)


def start_reservations(active, usage, stage_memory):
    """Use length-aware forecasts only when adding work to the resident pool."""
    if stage_memory is None:
        return [(item['peak'], usage.get(slot, 0)) for slot, item in active.items()]
    return [
        (max(item['grant'], item['completion_peak'] if slot is stage_memory.protected
             else item['admission_forecast']), usage.get(slot, 0))
        for slot, item in active.items()
    ]


def compiler_reserve(active, cases, admission, learning, candidate=None):
    """Reserve one compiler only while a cold/compilable worker can need it."""
    needed=set()
    for slot,item in active.items():
        mode=item['mode']
        if mode=='frax' and getattr(slot,'memory_request',None) is None and getattr(slot,'telemetry',{}).get('compile_possible',True):
            needed.add((mode,cases[item['index']]['source']))
    if candidate is not None:
        index,mode=candidate
        if mode=='frax':needed.add((mode,cases[index]['source']))
    return max((learning.compiler_extra(mode,source,admission) for mode,source in needed),default=0)


# Exact block-hook transitions validated by 101 regressions and paired outputs.
# This is not a general permission to change IK/solver implementations.
_BLOCK_MEMORY_COMPAT = {'resource_events.py': ('1a071937fe88d64cecc602bb123e362f8f0024eb7ccc60b560a420c2fead431d', '16b705a2f50ef80558aa05c7de1296087a6b26d202a8cff990eda8399353bb43'), 'block_memory.py': (None, '314a83ac3ecb3a4ebcf987f35d5a0a3b0cf27a9e32a1054ce85d8b4856154440'), 'ik.py': ('b6dc0dec4bf54e245a598a5902a0a93ef5937bccb2b525a51d337c08f8fc7773', '58162fed78a7b1f6fb7cda1ad21f3c9cd682513ac966974a28f235b82932711e'), 'stage_memory.py': ('03b3e2519719e1ef01f79f6057ac7595fd40e4c182c32c9b9307f8cdc0d57774', '652ca275adc53c845238dad24d39d7b479b97a5c7a3d17a43c6141c4092c147b')}


_PREVIOUS_BLOCK_STAGE_HASH = '734f8a88240be7bf43524d7ce670a162e5c9a1d4a5b0c96609b691288e880948'


def stage_compiler_reserve(active, cases, admission, learning, slot):
    shared = compiler_reserve(active, cases, admission, learning)
    item = active[slot]
    request = getattr(slot, 'memory_request', {}) or {}
    if item['mode'] == 'frax' and request.get('compile_possible', False):
        shared = max(shared, learning.compiler_extra(item['mode'], cases[item['index']]['source'], admission))
    return shared


# Exact pre-split and tested two-stage file hashes.
_SEGMENT_RESUME_COMPAT = {'cascade.py': ('a00c0c774c627377caa373f21dbf59984530cc07a99dcd720e7e7431808750cf', '15fdf4f773eb117850bbed5a3837fb3adc6cd8c2cce067fdf34d310052d70ede'), 'cli.py': ('141475cff0a9957119a904ecfa9364e718e667969eced3000aa56a101f403b3d', 'f9db5b3263b586760b35525c0bac490ccd29c27f794f70c65e6808c8b0ce998c'), 'memory_learning.py': ('d22688d7bb5f0f2187ad9296492e85f97e1286bbfb79ce096990a896513899cb', 'fee04db9899e549357312c50ca5211a6400d0287e321a70e08a16001bfd9043d'), 'segments.py': (None, 'abf25bbd0aace381062d7caf9b7d7c5cb765cc4f4af0c15217da047a7604542d'), 'v8/convert.py': ('83628f2d771ad58cd04cccfb7b651154d0e459695de4cf650a18490150a05e5e', '4a0d856eaddf096de2b70d83bf3852d3207214361740038766f33a0dc14f1003')}


def scheduler_update(original,snapshot):
    """An explicit resume may change only these resource/scheduling modules.

    Never ignore solver, acceptance, exporter, CLI, asset or dependency changes.
    Preserve original per-artifact identities instead of relabelling outputs.
    """
    if {k:v for k,v in original.items() if k!='implementation'}!={k:v for k,v in snapshot.items() if k!='implementation'}:
        raise ValueError('Resume runtime, assets or solver implementation changed')
    before=original.get('implementation',{});after=snapshot.get('implementation',{})
    changed={k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}
    extra=changed-{'managed_batch.py','resources.py','memory_queue.py','initial_memory.py'}
    segment_changes=extra & _SEGMENT_RESUME_COMPAT.keys()
    if segment_changes:
        if (os.environ.get('DEX_FRAX_ALLOW_SEGMENTED_RESUME')!='1'
                or any((before.get(name),after.get(name)) != _SEGMENT_RESUME_COMPAT[name] for name in segment_changes)
                or any(after.get(name) != pair[1] for name,pair in _SEGMENT_RESUME_COMPAT.items())):
            raise ValueError('Resume conversion implementation changed')
        extra -= segment_changes
    if extra:
        # All block-hook files must match the tested implementation together.
        # Ordinary math/acceptance/exporter/asset/runtime changes remain rejected.
        if (not extra.issubset(_BLOCK_MEMORY_COMPAT)
                or any((before.get(k),after.get(k))!=_BLOCK_MEMORY_COMPAT[k] and not (k=='stage_memory.py' and before.get(k)==_PREVIOUS_BLOCK_STAGE_HASH and after.get(k)==_BLOCK_MEMORY_COMPAT[k][1]) for k in extra)
                or any(after.get(k)!=pair[1] for k,pair in _BLOCK_MEMORY_COMPAT.items())):
            raise ValueError('Resume runtime, assets or solver implementation changed')
    return {k:dict(before=before.get(k),after=after.get(k)) for k in sorted(changed)}


def enable_scheduler_initial_reuse():
    """Worker-local adapter for an explicitly authorized scheduler-only resume.

    The conversion/export identity remains current. Only the reader's comparison
    identity is adapted after the same two-file check used by restore; all source,
    model, solver, parameter, target and artifact checks still run unchanged.
    """
    from . import initial_reuse
    base=initial_reuse.ReusedInitial
    if getattr(base,'scheduler_resume_adapter',False):return
    class SchedulerInitial(base):
        scheduler_resume_adapter=True
        def __init__(self,directory,identity,fallback,strict=False):
            super().__init__(directory,identity,fallback,strict)
            self.scheduler_changes={}
            try:
                previous=json.loads((Path(directory)/'identity.json').read_text())
                keys=('converter','version','implementation','frax_implementation','assets','runtime')
                changes=scheduler_update({k:previous[k] for k in keys},{k:identity[k] for k in keys})
            except (OSError,ValueError,KeyError,TypeError):
                return  # Original strict reader rejects the mismatch.
            if changes:
                self.identity=copy.deepcopy(identity)
                self.identity['implementation']=copy.deepcopy(previous['implementation'])
                self.scheduler_changes=changes
        def __call__(self,*args,**kwargs):
            result=super().__call__(*args,**kwargs)
            if self.scheduler_changes and self.report.get('initial_reuse',{}).get('used'):
                self.report['initial_reuse']['scheduler_compatibility']=self.scheduler_changes
            return result
    initial_reuse.ReusedInitial=SchedulerInitial

def restore(args,out,content,cases,snapshot):
    """Resume only verified attempts; solver or asset changes require a new run."""
    previous=json.loads((out/'batch.json').read_text())
    if previous['manifest_sha256']!=sha_bytes(content) or previous['mode']!=args.mode:
        raise ValueError('Resume manifest or mode changed')
    original=json.loads((out/'batch_identity.json').read_text())
    changes={}
    if original!=snapshot:
        if os.environ.get('DEX_FRAX_ALLOW_SCHEDULER_RESUME')!='1':
            raise ValueError('Resume runtime, assets or solver implementation changed')
        changes=scheduler_update(original,snapshot)
    contract_path=out/'conversion_contract.json'
    if not contract_path.exists() or json.loads(contract_path.read_text())!=conversion_contract(args):
        raise ValueError('Resume conversion contract changed or missing; start a new batch')
    records=copy.deepcopy(previous['records'])
    if [r['case'] for r in records]!=[c['id'] for c in cases]:raise ValueError('Resume case order changed')
    for record in records:
        for attempt in record['attempts']:
            if attempt['status']!='complete':continue
            folder=(out/attempt['output']).resolve()
            if not folder.is_relative_to(out):raise ValueError('Resume output escaped batch directory')
            run=json.loads((folder/'run.json').read_text())
            if run.get('status')!='complete':raise ValueError('Resume completed attempt is inconsistent')
            for name,h in run['files'].items():
                path=(folder/name).resolve()
                if not path.is_relative_to(folder) or sha(path)!=h:raise ValueError('Resume artifact changed: '+name)
            ident=json.loads((folder/'identity.json').read_text())
            if changes:
                scheduler_update({k:ident[k] for k in snapshot},snapshot)
            if ident.get('conversion_contract')!=conversion_contract(args):
                raise ValueError('Resume conversion parameters changed')
        if record['status']=='accepted':
            attempt=next(a for a in reversed(record['attempts']) if a['mode']==record['selected_mode'])
            if not accepted_artifact(out/attempt['output'],dict(attempt,result=json.loads((out/attempt['output']/'run.json').read_text()))):
                raise ValueError('Resume accepted artifact did not validate')
        elif record['status'] not in ('input_error','review','accepted_with_notes','failed'):record['status']='pending'
    atomic_json(out/('batch.before_resume.'+str(time.time_ns())+'.json'),previous)
    if changes:
        atomic_json(out/('scheduler_resume.'+str(time.time_ns())+'.json'),
                    dict(before=original,after=snapshot,changed_files=changes,
                         completed_artifacts_verified=True,original_artifacts_unchanged=True))
    return records,previous

def _run(args,slot_factory,monitor):
    priority_lowered=res.lower_priority() if res.dynamic_mode(args) else None
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='1')
    content=Path(args.manifest).read_bytes();cases,aliases=deduplicate(load_manifest(args.manifest,content))
    nworkers=res.workers(args.workers);out=Path(args.output).resolve();resume=getattr(args,'resume',False)
    if resume and not (out/'batch.json').is_file():raise ValueError('Resume requires an existing batch.json')
    out.mkdir(parents=True,exist_ok=resume)
    atomic_json(out/'source_aliases.json',dict(policy='full_bvh_sha256',duplicates=aliases))
    admission=ResourcePolicy(args,out,nworkers)
    reserve=res.host_reserve(args,monitor());disk_floor=int(args.disk_reserve_gib*res.GIB)
    dynamic=res.dynamic_mode(args)
    from .dynamic_resources import CpuAdmission
    cpu_admission=CpuAdmission(os.sched_getaffinity(0) if hasattr(os,'sched_getaffinity') else range(res.cpu_count()),nworkers) if dynamic else None
    started=time.monotonic();last_checkpoint=started;last_heartbeat=started;loop_seconds=0.;usage={};state='running';phase='prepare';active={};slots=[];last=0.;stopped=False;wait_since=None;last_reason=None
    deadline = float('inf') if args.budget is None else started + args.budget
    records=[dict(case=c['id'],status='pending',attempts=[],selected_mode=None,selected_output=None,training_npz=None) for c in cases]
    snapshot=static_identity();previous=None
    learning=MemoryLearning(res.state_root(),dict(identity=snapshot,execution=dict(
        conversion=conversion_contract(args))),not getattr(args,'no_memory_learning',False))
    if resume:
        records,previous=restore(args,out,content,cases,snapshot)
        if not learning.profiles:
            learning.profiles=copy.deepcopy(previous.get('resources',{}).get('memory_learning',{}).get('profiles',{}))
    frames={};durations={};routes={};snap=monitor();priority_wait=None;observed=dict(max_active_workers=0,worker_pss_peak_bytes=0,total_worker_pss_peak_bytes=0,scope_usage_peak_bytes=0)
    queue=[]
    segmented=getattr(args,'segmented',False)
    initial_memory=InitialMemory() if segmented else None
    ready_initial=set()
    initial_barrier_open=False
    memory_queue=MemoryQueue(nworkers,(previous or {}).get('resources',{}).get('memory_queue'))
    stage_memory=StageMemory() if getattr(args,'profile',None) in ('fast','refine') else None
    oom_kills=snap.get('cgroup_oom_kills',0)
    policy=dict(workers=nworkers,jit_compilers=1,memory_reserve_gib=reserve/res.GIB,
        resource_mode='dynamic' if dynamic else 'manual',memory_target_percent=getattr(args,'memory_target_percent',None),priority_lowered=priority_lowered,
        disk_reserve_gib=args.disk_reserve_gib,worker_peak_base_gib=args.worker_peak_gib,
        v8_worker_peak_base_gib=getattr(args,'v8_worker_peak_gib',args.worker_peak_gib),
        frax_resident_base_gib=getattr(args,'worker_resident_gib',args.worker_peak_gib),
        compile_extra_gib=max(0,args.worker_peak_gib-getattr(args,'worker_resident_gib',args.worker_peak_gib)),
        peak_bytes_per_output_frame=admission.value['stages']['frax']['bytes_per_output_frame'],start_method='spawn',cgroup_limit_bytes=snap['cgroup_limit'],
        frax_peak_bytes_per_output_frame=admission.value['stages']['frax']['bytes_per_output_frame'],
        scheduling='all_frax_then_repair' if segmented else ('frax_checked_local_repairs_v1' if use_unified(args) else 'frax_all_then_independent_v8_v6'),
        stage_barriers=segmented or (args.mode=='auto' and not use_unified(args)),long_seconds=None,
        identity='frozen before workers, recomputed after workers',model='verified MJB, reused per spawn worker; no cross-process fork sharing')
    pool_stats=Counter()
    stop_path=out/'STOP'
    previous_handlers={}
    def on_signal(*_):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGINT,signal.SIGTERM):previous_handlers[sig]=signal.signal(sig,on_signal)

    def save(full=True):
        nonlocal last,last_checkpoint,last_heartbeat
        policy['worker_pool']=dict(policy='reclaim_idle_on_admission_shortfall_v2',fixed_job_recycle=False,
            counters=dict(pool_stats),resident_idle=sum(s not in active and bool(s.pid) for s in slots),
            maximum_jobs_in_resident=max((getattr(s,'jobs',0) for s in slots),default=0))
        policy['resource_policy']=admission.report()
        policy['workers']=admission.value['workers']
        policy['memory_reserve_gib']=reserve/res.GIB
        policy['memory_learning']=learning.report()
        policy['memory_queue']=memory_queue.report()
        if initial_memory:
            policy['initial_memory']=initial_memory.report({slot:item for slot,item in active.items()
                                                            if item.get('segment')=='initial'})
        if stage_memory:policy['stage_memory']=stage_memory.report(active)
        if cpu_admission:
            measured=cpu_admission.report()
            policy['cpu_admission']=dict(measured,load_based_target_ignored=measured.get('target_workers'),
                target_workers=min(admission.value['workers'],nworkers),reason='full_available_cpu',
                external_load_control=False)
        policy.update(frax_resident_base_gib=admission.value['stages']['frax']['base_gib'],
            v8_worker_peak_base_gib=admission.value['stages']['v8']['base_gib'],
            compile_extra_gib=admission.value['compile_extra_gib'],
            worker_peak_base_gib=admission.value['stages']['frax']['base_gib']+admission.value['compile_extra_gib'],
            peak_bytes_per_output_frame=admission.value['stages']['frax']['bytes_per_output_frame'],
            frax_peak_bytes_per_output_frame=admission.value['stages']['frax']['bytes_per_output_frame'])
        policy['worker_pool']['active_pss_bytes']=sum(usage.get(s,0) for s in active)
        policy['worker_pool']['idle_pss_bytes']=sum(usage.get(s,0) for s in slots if s not in active and s.pid)
        measured=[usage.get(s,0) for s in slots if s.pid]
        observed['max_active_workers']=max(observed['max_active_workers'],len(active))
        observed['worker_pss_peak_bytes']=max(observed['worker_pss_peak_bytes'],max(measured,default=0))
        observed['total_worker_pss_peak_bytes']=max(observed['total_worker_pss_peak_bytes'],sum(measured))
        if snap['cgroup_limit'] is not None:observed['scope_usage_peak_bytes']=max(observed['scope_usage_peak_bytes'],snap['cgroup_limit']-snap['cgroup_headroom'])
        now=time.monotonic()
        live=dict(schema=2,mode=args.mode,status=state,phase=phase,
            updated_at=time.time(),elapsed_s=now-started,
            accepted=sum(r['status']=='accepted' for r in records),
            counts=dict(Counter(r['status'] for r in records)),
            resources=policy,memory=snap,resource_wait_reason=last_reason,
            priority_wait=priority_wait,sampled_resource_peaks=observed,
            worker_pss_bytes=sum(measured),scheduler_loop_s=loop_seconds,
            checkpoint_interval_s=30.,checkpoint_age_s=now-last_checkpoint,
            queued=len(queue),
            queued_initial=sum(entry[0][0]==0 for entry in queue) if segmented else None,
            queued_repair=sum(entry[0][0]==1 for entry in queue) if segmented else None,
            initial_ready=len(ready_initial) if segmented else None,
            initial_barrier_open=initial_barrier_open if segmented else None,
            runnable=sum(getattr(s,'memory_request',None) is None for s in active),
            waiting_memory=sum(getattr(s,'memory_request',None) is not None for s in active),
            active=[dict(case=records[v['index']]['case'],mode=v['mode'],segment=v.get('segment'),pid=s.pid,
                estimated_peak_bytes=v['peak'],pss_bytes=usage.get(s,0),
                memory_state='waiting' if getattr(s,'memory_request',None) is not None else 'running',
                granted_peak_bytes=v.get('grant',v['peak']),completion_peak_bytes=v.get('completion_peak',v['peak']),
                admission_forecast_bytes=v.get('admission_forecast',v['peak']),past_cold_start=v.get('past_cold_start',False),
                requested_peak_bytes=v.get('requested_peak'),
                memory_wait_s=s.memory_wait_s() if hasattr(s,'memory_wait_s') else 0.,
                telemetry=getattr(s,'telemetry',{})) for s,v in active.items()])
        if full:
            accepted=[]
            for c,r in zip(cases,records):
                if r['status']=='accepted':accepted.append(dict(case=r['case'],input=c['input'],input_sha256=c['input_sha256'],mode=r['selected_mode'],output=r['selected_output'],training_npz=r['training_npz'],training_sha256=r['training_sha256'],fingerprint=r['fingerprint']))
            atomic_json(out/'accepted.json',dict(schema=1,path_base='batch output directory',motions=accepted))
            atomic_json(out/'batch.json',dict(live,
                stage_order=['frax_initial','frax_repair'] if segmented else (list(pipeline_stages(args)) if args.mode=='auto' else [args.mode]),
                pipeline=getattr(args,'pipeline','unified'),
                unique_sources=len(cases),duplicate_aliases=len(aliases),
                acceptance_policy='common_pose_geometry_v1',manifest_sha256=sha_bytes(content),budget_s=args.budget,
                records=records,resumed=resume,previous_elapsed_s=previous.get('elapsed_s') if previous else None))
            last_checkpoint=time.monotonic()
            live['checkpoint_age_s']=0.
        atomic_json(out/'batch.running.json',live)
        if now-last_heartbeat>=30:
            done=sum(r['status'] in ('accepted','review','accepted_with_notes') for r in records)
            print(f"[batch] running={live['runnable']} waiting_memory={live['waiting_memory']} queued={live['queued']} done={done}/{len(records)}",flush=True)
            last_heartbeat=now
        last=time.monotonic()

    def abort(reason):
        for slot,item in list(active.items()):
            slot.close();finish(records[item['index']],dict(status=reason),item['mode'],item['folder'],out)
        active.clear()

    save()
    try:
        for i,c in enumerate(cases):
            try:
                frames[i]=res.output_frames(c['input']);durations[i]=duration_seconds(c['input'])
                routes[i]=pipeline_stages(args) if args.mode=='auto' else route(args.mode,durations[i],getattr(args,'single_pipeline',False),getattr(args,'pipeline','unified'))
                records[i].update(duration_s=float(durations[i]),route=list(routes[i]))
            except (OSError,ValueError) as error:records[i].update(status='input_error',error=str(error))
        if resume:atomic_json(out/('resume_identity.'+str(time.time_ns())+'.json'),snapshot)
        else:
            atomic_json(out/'batch_identity.json',snapshot)
            atomic_json(out/'conversion_contract.json',conversion_contract(args))
        from .model_cache import prepare
        model=prepare(snapshot['assets'],res.state_root()/'models');res.release_memory()
        cpus=res.cpu_order() if hasattr(os,'sched_getaffinity') else [None]*nworkers
        cache_dir=res.state_root()/'jax_cache';cache_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
        options=dict(identity=snapshot,model=model,compile_dir=str(res.state_root()/'compiler'),persistent_cache_dir=str(cache_dir),block_frames=getattr(args,'ik_block_frames',128),dynamic=dynamic,block_permits=segmented,
            scheduler_resume=bool(resume and os.environ.get('DEX_FRAX_ALLOW_SCHEDULER_RESUME')=='1'))
        queue=[];sequence=0
        def initial_folder(index):
            return out/cases[index]['id']/'frax_initial'

        def segment_for(index):
            if not segmented:return None
            return 'repair' if (initial_folder(index)/'initial_ready.json').is_file() else 'initial'

        if segmented:
            ready_initial.update(i for i in range(len(cases)) if segment_for(i)=='repair')

        def enqueue(index,mode):
            nonlocal sequence
            segment=segment_for(index)
            rank=priority((index,mode),durations)
            if segment=='repair':
                # Filesystem commit time preserves FIFO order across restarts.
                rank=(1,(initial_folder(index)/'initial_ready.json').stat().st_mtime_ns)
            elif segment=='initial':
                rank=(0,durations[index])
            heapq.heappush(queue,(rank,sequence,index,mode))
            sequence+=1
        def defer(slot,item,reason):
            i=item['index'];mode=item['mode'];record=records[i]
            observed_peak=max(item.get('observed_peak',0),usage.get(slot,0)) or item['peak']
            per=admission.value['stages'][mode]['bytes_per_output_frame']
            event=getattr(slot,'telemetry',{})
            underestimated=(reason in ('cgroup_oom_retry','stage_budget_underestimate')
                             or (reason!='initial_resident_saturation' and observed_peak>item.get('grant',item['peak'])))
            memory_queue.defer(mode,cases[i]['source'],frames[i],per,observed_peak,len(active),
                               stage=event.get('stage'),underestimated=underestimated)
            slot.close()
            if item.get('segment')=='initial' and segment_for(i)=='repair':
                ready_initial.add(i)
            finish(record,dict(status='memory_retry',reason=reason),mode,item['folder'],out)
            record['memory_retries']=record.get('memory_retries',0)+1
            del active[slot];enqueue(i,mode)
            print(cases[i]['id'],mode,'memory_retry',reason,flush=True)
            save()
        completion_lengths={}
        for i,route_modes in routes.items():
            for mode in route_modes:
                key=(mode,cases[i]['source'])
                completion_lengths[key]=max(completion_lengths.get(key,0),frames[i])
        for i,r in enumerate(records):
            if r['status'] in ('accepted','input_error','review','accepted_with_notes','failed') or r.get('escalation',{}).get('stop'):continue
            for mode in routes[i]:
                if not any(a['mode']==mode and a['status'] in ('complete','input_error') for a in r['attempts']):
                    enqueue(i,mode);break
        while queue or active:
            loop_started=time.monotonic()
            admission.reload()
            learning.enabled=admission.value.get('memory_learning',not getattr(args,'no_memory_learning',False))
            lengths={}
            if initial_memory:
                per_frame=admission.value['stages']['frax']['bytes_per_output_frame']
                lengths={i:count*per_frame for i,count in frames.items()}
            if stage_memory:
                for (mode,source),length in completion_lengths.items():
                    predicted=max(admission.completion_peak(mode,length),learning.predict(mode,source,'jax_hot',length,admission))
                    predicted=memory_queue.peak(mode,source,length,admission.value['stages'][mode]['bytes_per_output_frame'],predicted)
                    stage_memory.bound=max(stage_memory.bound,predicted)
                    if initial_memory and mode=='frax':
                        scratch=learning.compiler_extra(mode,source,admission)
                        private=segmented_peak(learning,admission,mode,source,length,'initial',memory_queue)
                        initial_memory.compiler_bytes=max(initial_memory.compiler_bytes,scratch)
                        initial_memory.private_bytes=max(initial_memory.private_bytes,private,
                            admission.completion_peak(mode,length)-scratch)
                for item in active.values():
                    if item.get('segment')!='initial':
                        item['completion_peak']=max(item['completion_peak'],stage_memory.bound)
            modes={v['mode'] for v in active.values()} or {entry[3] for entry in queue}
            phase=('frax_repair' if initial_barrier_open else 'frax_initial') if segmented else (
                next(iter(modes)) if len(modes)==1 else 'mixed')
            snap=monitor()
            reserve=res.host_reserve(args,snap)
            # CPU load is telemetry only. Fill every permitted slot; the OS
            # schedules competing work. Host/cgroup memory admission still gates starts.
            if cpu_admission:cpu_admission.sample([os.getpid()]+[s.pid for s in slots])
            dispatch_limit=min(admission.value['workers'],nworkers,memory_queue.limit)
            if stopped or stop_path.exists():state='interrupted';abort('interrupted');break
            if time.monotonic()>=deadline:state='budget_exhausted';abort('budget_exhausted');break
            free=res.shutil.disk_usage(out).free
            if free<disk_floor:state='resource_exhausted';last_reason='disk_reserve';abort('resource_exhausted');break
            for slot,item in list(active.items()):
                response=slot.poll()
                if response is not None:
                    if (response.get('error_type')=='WorkerExit' and response.get('exit_code')==-9
                            and snap.get('cgroup_oom_kills',0)>oom_kills):
                        defer(slot,item,'cgroup_oom_retry');continue
                    mode=item['mode'];i=item['index']
                    learning.observe(slot,item,usage.get(slot,0))
                    if response['status']=='initial_ready' and item.get('segment')=='initial':
                        if segment_for(i)!='repair':
                            raise ValueError('Frax worker did not commit its checkpoint')
                        learning.complete(slot,item,dict(status='complete',stage_only=True),cases[i],frames[i],admission)
                        pool_stats['frax_handoffs']+=1
                        pool_stats['frax_elapsed_s']+=response.get('worker_wall_s',response.get('elapsed_s',0))
                        records[i]['initial_handoff']=str(item['folder'].relative_to(out))
                        slot.close()
                        ready_initial.add(i)
                        del active[slot]
                        enqueue(i,mode)
                        print(cases[i]['id'],'initial_ready',flush=True)
                        save()
                        continue
                    if item.get('segment')=='initial':
                        slot.close()
                    attempt=finish(records[i],response,mode,item['folder'],out,getattr(args,'v8_for_collision_only',False));del active[slot]
                    if attempt['status']=='complete':
                        learning.complete(slot,item,dict(response,stage_only=bool(item.get('segment'))),cases[i],frames[i],admission)
                        memory_queue.complete()
                    print(cases[i]['id'],mode,attempt['status'],attempt.get('quality'),flush=True)
                    if records[i]['status']=='pending':
                        next_stage=routes[i].index(mode)+1
                        if next_stage<len(routes[i]):enqueue(i,routes[i][next_stage])
                        else:records[i]['status']=preview_outcome(records[i])
                    atomic_json(item['folder'].parent/'selection.json',records[i])
            oom_kills=snap.get('cgroup_oom_kills',oom_kills)
            # Drain completions before the one O(workers) PSS sweep. Reuse this
            # sweep for every admission and report; never rescan all workers
            # once per newly filled slot (the old O(workers squared) hot path).
            usage={s:res.process_memory(s.pid) for s in slots}
            initial={slot:item for slot,item in active.items() if item.get('segment')=='initial'}
            other={slot:item for slot,item in active.items() if slot not in initial}
            if segmented and not initial and queue and queue[0][0][0]==1 and not initial_barrier_open:
                if len(ready_initial)!=len(cases):
                    state='initial_incomplete';last_reason='initial_checkpoint_missing';break
                initial_barrier_open=True
                phase='frax_repair'
                print('initial_batch_ready',len(ready_initial),flush=True)
            if initial_memory:
                initial_memory.elect(initial,usage)
                initial_memory.measure_residency(initial,usage,lengths)
            if stage_memory:
                if initial:stage_memory.protected=None
                else:stage_memory.elect(other,usage)
            for slot,item in active.items():
                learning.observe(slot,item,usage.get(slot,0))
                i=item['index'];mode=item['mode']
                item['observed_peak']=max(item.get('observed_peak',0),usage.get(slot,0),getattr(slot,'telemetry',{}).get('pss_bytes',0))
                if slot in initial:
                    item.update(peak=item['grant'],admission_forecast=item['grant'],
                                completion_peak=initial_memory.private_bytes
                                if slot is initial_memory.owner else item['grant'])
                    continue
                # Refine can revisit audit after repair; reserve the complete
                # lifecycle rather than release a supposedly finished stage.
                source=cases[i]['source'];per_frame=admission.value['stages'][mode]['bytes_per_output_frame']
                forecast=learning.predict(mode,source,getattr(slot,'telemetry',{}).get('cache_state','pure'),
                                          frames[i],admission,reuse=item.get('reuse',False))
                estimate=max(forecast,int(item['observed_peak']*1.1)+128*res.MIB)
                if stage_memory:
                    item['completion_peak']=max(item['completion_peak'],admission.completion_peak(mode,frames[i]),estimate)
                    # Reassess the length-aware forecast on every scheduler pass.
                    # This extra reserve gates launches only. Stage grants below
                    # still use actual grants, so running work can make progress.
                    if item.get('segment'):
                        cache=getattr(slot,'telemetry',{}).get('cache_state','pure')
                        segment_peak=segmented_peak(learning,admission,mode,source,frames[i],
                                                    item['segment'],memory_queue,cache=cache)
                        item['admission_forecast']=max(segment_peak,item['observed_peak']+64*res.MIB)
                    else:
                        item['admission_forecast']=inflight_forecast(
                            slot,item,memory_queue,source,frames[i],per_frame,forecast)
                    item['peak']=item['grant']
                else:item['peak']=memory_queue.peak(mode,cases[i]['source'],frames[i],admission.value['stages'][mode]['bytes_per_output_frame'],estimate)
            snap=monitor()
            victim=res.reclaim_idle(slots,active,usage,snap,reserve,queue_empty=not queue)
            if victim is not None:
                victim.close()
                pool_stats['pressure_recycles' if queue else 'tail_idle_recycles']+=1
                # Both snapshot and PSS must be refreshed together next loop.
                continue
            critical,waiting=pressure(snap,reserve)
            if critical and active and not initial:
                last_reason='memory_pressure_requeue';priority_wait=None
                slot=memory_victim(active,usage,stage_memory.protected if stage_memory else None)
                defer(slot,active[slot],last_reason)
                # Never credit released pages until a fresh snapshot/PSS sweep.
                continue
            if waiting:
                last_reason='memory_pressure_wait';priority_wait=None
            # Workers request before allocating the next stage and before
            # acquiring shared locks. Existing growth has priority over starts.
            if initial:
                initial_memory.grant_requests(initial,usage,stage_memory.running(other,usage),
                                              snap,reserve,monitor,lengths,pressured=waiting)
                if initial_memory.memory_waiting and not waiting:
                    last_reason='initial_memory_wait';priority_wait=None
                if not any(getattr(s,'memory_request',None) is None for s in other):
                    victim=initial_memory.stalled_victim(initial,usage)
                    if victim is not None:
                        initial_memory.stalled_since=None
                        defer(victim,active[victim],'initial_resident_saturation')
                        continue
            if stage_memory and not (initial and waiting):
                for slot,item in other.items():
                    request=getattr(slot,'memory_request',None)
                    if request is None:continue
                    i=item['index'];mode=item['mode'];stage=request.get('stage','loading')
                    estimate=learning.stage_peak(mode,cases[i]['source'],request.get('cache_state','pure'),frames[i],admission,stage)
                    estimate=request_peak(request,usage.get(slot,0),estimate)
                    scope=request.get('memory_scope')
                    floor_stage=(scope if scope in ('initial_block','initial_finalize') and not request.get('first_block',False) else stage)
                    source=cases[i]['source']
                    per_frame=admission.value['stages'][mode]['bytes_per_output_frame']
                    estimate=memory_queue.peak(mode,source,frames[i],per_frame,estimate,stage=floor_stage)
                    stage_memory.request(slot,item,estimate,stage,retained=max(usage.get(slot,0),request.get('pss_bytes',0)),scope=scope)
                for item in other.values():item['completion_peak']=max(item['completion_peak'],stage_memory.bound)
                initial_reserved=initial_memory.reservations(initial,usage) if initial else []
                for slot in stage_memory.ordered(other):
                    item=other[slot]
                    grant_shared=stage_compiler_reserve(other,cases,admission,learning,slot)
                    peak=item['completion_peak'] if slot is stage_memory.protected else item['requested_peak']
                    running=stage_memory.running(other,usage,slot)+initial_reserved
                    if not res.can_admit(snap,reserve,1,0,running,peak,usage.get(slot,0),grant_shared)[0]:continue
                    stage_memory.grant(slot,item)
                    measured=monitor()
                    for key in ('available','cgroup_headroom'):snap[key]=min(snap[key],measured[key])
                blocked=stage_memory.ordered(other)
                if blocked:
                    last_reason='stage_memory_wait';priority_wait=None
                    owner=stage_memory.protected
                    slot=blocked[0]
                    wanted=other[slot]['completion_peak' if slot is owner else 'requested_peak']
                    shared=stage_compiler_reserve(other,cases,admission,learning,slot)
                    idle=res.reclaim_idle(slots,active,usage,snap,reserve,slot,wanted,shared,
                                          running=stage_memory.running(other,usage,slot)+initial_reserved)
                    if idle is not None:
                        idle.close()
                        pool_stats['stage_wait_idle_recycles']+=1
                        # Discard the old memory sample before any new grant.
                        continue
                    # If estimates were violated and everyone is blocked,
                    # release one non-protected attempt instead of deadlocking.
                    if not initial and len(blocked)==len(active) and len(active)>1:
                        slot=memory_victim(active,usage,owner)
                        defer(slot,active[slot],'stage_budget_underestimate');continue
                waiting=waiting or bool(blocked)
            if initial and initial_memory.memory_waiting:
                waiting=True
            # Do not credit reclaimed pages again while using this sweep's PSS.
            burst_headroom={k:snap[k] for k in ('available','cgroup_headroom')}
            if queue:
                reservations=start_reservations(other,usage,stage_memory)
                if initial:reservations+=initial_memory.reservations(initial,usage)
                memory_queue.recover(snap,reserve,len(active),waiting=waiting,running=reservations,
                                     shared_reserve=compiler_reserve(other,cases,admission,learning))
            dispatch_limit=min(admission.value['workers'],nworkers,memory_queue.limit)
            if queue and not waiting and len(active)>=dispatch_limit:
                last_reason=('memory_backoff_capacity' if memory_queue.limit < min(admission.value['workers'],nworkers)
                             else 'cpu_capacity')
                priority_wait=None
            while queue and not waiting and len(active)<dispatch_limit:
                if segmented and queue[0][0][0]==1 and not initial_barrier_open:
                    last_reason='initial_stage_barrier';priority_wait=None
                    break
                # Complete the entire Frax pass before any independent V8 work.
                if args.mode=='auto' and queue[0][3]=='v8' and any(v['mode']=='frax' for v in active.values()):
                    last_reason='frax_stage_barrier';priority_wait=None
                    break
                idle=res.select_idle(slots,active,usage)
                _,_,index,mode=queue[0]
                cache=getattr(idle,'telemetry',{}).get('cache_state','pure') if idle and idle.pid else 'pure'
                segment=segment_for(index)
                source=cases[index]['source']
                per_frame=admission.value['stages'][mode]['bytes_per_output_frame']
                if segment=='initial':
                    if initial_memory.unsampled_workers>=initial_memory.bootstrap_limit:
                        last_reason='initial_residency_sampling';priority_wait=None
                        break
                    fallback=int(admission.value['stages'][mode]['base_gib']*res.GIB)
                    peak,running,shared=initial_memory.launch_budget(initial,usage,lengths,index,fallback)
                    completion=initial_memory.private_bytes
                    candidate_peak=peak
                else:
                    running=start_reservations(other,usage,stage_memory)
                    if segment=='repair':
                        peak=segmented_peak(learning,admission,mode,source,frames[index],segment,memory_queue,cache=cache)
                    else:
                        peak=learning.predict(mode,source,cache,frames[index],admission)
                        peak=memory_queue.peak(mode,source,frames[index],per_frame,peak)
                    completion=max(peak,admission.completion_peak(mode,frames[index]),learning.predict(mode,source,cache,frames[index],admission))
                    if stage_memory:completion=max(completion,stage_memory.bound)
                    shared=compiler_reserve(other,cases,admission,learning,
                                            candidate=(index,mode) if segment!='repair' else None)
                    candidate_peak=completion if stage_memory and stage_memory.protected is None else peak
                candidate_used=usage.get(idle,0)
                allowed,reason=res.can_admit(snap,reserve,free,disk_floor,running,
                                             candidate_peak,candidate_used,shared)
                if not allowed:
                    last_reason='priority_head_wait:'+reason
                    headroom,growth,required=res.admission_budget(snap,reserve,running,candidate_peak,candidate_used,shared)
                    priority_wait=dict(case=cases[index]['id'],mode=mode,duration_s=float(durations[index]),
                        candidate_peak_bytes=candidate_peak,candidate_resident_bytes=candidate_used,
                        normal_stage_peak_bytes=peak,completion_peak_bytes=completion,
                        compiler_reserve_bytes=shared if segment!='initial' else 0,
                        initial_launch_margin_bytes=shared if segment=='initial' else 0,
                        inflight_growth_bytes=growth,available_headroom_bytes=headroom,
                        required_headroom_bytes=required,shortfall_bytes=max(0,required-headroom),reason=reason)
                    victim=None
                    if reason=='memory_reserve':
                        victim=res.reclaim_idle(slots,active,usage,snap,reserve,idle,candidate_peak,shared,
                                                running=running)
                    if victim is not None:
                        victim.close()
                        pool_stats['dispatch_wait_idle_recycles']+=1
                    else:pool_stats['reservation_waits_preserving_warm']+=1
                    break
                _,_,i,mode=heapq.heappop(queue)
                c=cases[i];r=records[i]
                if idle is None:
                    idle=slot_factory(dict(options,cpu=cpus[len(slots)%len(cpus)]));slots.append(idle)
                folder=initial_folder(i) if segment=='initial' else (out/c['id']/mode if args.mode=='auto' else out/c['id'])
                folder.parent.mkdir(parents=True,exist_ok=True)
                if folder.exists():
                    if not resume and not r.get('memory_retries'):raise FileExistsError(folder)
                    archived=folder.with_name(folder.name+'.interrupted.'+str(time.time_ns()))
                    folder.rename(archived)
                    for attempt in r['attempts']:
                        if attempt['mode']==mode and attempt['status']!='complete' and attempt['output']==str(folder.relative_to(out)):
                            attempt['output']=str(archived.relative_to(out))
                    old_log=folder.parent/'frax_initial.log' if segment=='initial' else (folder.parent/(mode+'.log') if args.mode=='auto' else out/(c['id']+'.log'))
                    if old_log.exists():old_log.rename(old_log.with_name(old_log.name+'.interrupted.'+str(time.time_ns())))
                reuse=None
                arguments=dict(bvh_units=getattr(args,'bvh_units','cm'),bvh_translations=getattr(args,'bvh_translations','absolute'),foot_contact=getattr(args,'foot_contact',DEFAULT_FOOT_CONTACT),input=c['input'],output=str(folder),source=c['source'],mode=mode,iterations=args.iterations,initial_iterations=args.initial_iterations,max_frames=None,legacy_frax=False,reuse_initial_from=reuse,require_initial_reuse=reuse is not None,pipeline=getattr(args,'pipeline','unified'),unified_repair=use_unified(args),repair_similarity=getattr(args,'repair_similarity','soft'),profile=getattr(args,'profile',None),repair_budget=getattr(args,'repair_budget',None),allow_local_slowdown=getattr(args,'allow_local_slowdown',False),scene=getattr(args,'scene',None),resume_recheck=getattr(args,'resume_recheck',False))
                arguments['refine_flow'] = getattr(args, 'refine_flow', DEFAULT_REFINE_FLOW)
                if segment:
                    arguments.update(initial_only=segment=='initial',
                                     initial_from=str(initial_folder(i)) if segment=='repair' else None)
                job_log=folder.parent/'frax_initial.log' if segment=='initial' else (folder.parent/(mode+'.log') if args.mode=='auto' else out/(c['id']+'.log'))
                job=dict(input=c['input'],input_sha256=c['input_sha256'],log=str(job_log),arguments=arguments,
                    stage_memory=bool(stage_memory and getattr(idle,'supports_stage_memory',False)))
                active[idle]=dict(index=i,case=c['id'],mode=mode,folder=folder,peak=peak,grant=peak,admission_forecast=peak,
                                  completion_peak=completion,reuse=reuse is not None,segment=segment)
                if segment=='initial':
                    initial[idle]=active[idle]
                    initial_memory.unsampled_workers+=1
                    if initial_memory.owner is None:initial_memory.owner=idle
                    stage_memory.protected=None
                else:
                    other[idle]=active[idle]
                    if stage_memory:
                        stage_memory.register(idle,active[idle])
                        if initial:stage_memory.protected=None
                pool_stats['resident_reuses' if idle.pid else 'process_starts']+=1
                idle.start(job,min(timeout_for(args,mode,frames[i]),max(.01,deadline-time.monotonic())))
                snap=monitor()
                for key in burst_headroom:
                    burst_headroom[key]=min(burst_headroom[key],snap[key])
                    snap[key]=burst_headroom[key]
                _,waiting=pressure(snap,reserve)
                last_reason=None;wait_since=None;priority_wait=None
            if queue and not active:
                wait_since=wait_since or time.monotonic()
                if not dynamic and args.resource_wait>0 and time.monotonic()-wait_since>=args.resource_wait:state='resource_exhausted';break
            else:wait_since=None
            loop_seconds=time.monotonic()-loop_started
            if time.monotonic()-last>=2:save(full=time.monotonic()-last_checkpoint>=30)
            if queue or active:time.sleep(.1)
        for slot in slots:slot.close()
        if static_identity()!=snapshot:
            state='identity_changed'
            for r in records:
                if r['status']=='accepted':r.update(status='identity_changed',selected_mode=None,selected_output=None,training_npz=None)
        if state=='running':
            state='initial_incomplete' if segmented and len(ready_initial)!=len(cases) else 'complete'
        for r in records:
            if r['status']=='pending':r['status']=state if state!='complete' else preview_outcome(r)
        phase='finished';save()
    except BaseException:
        state='failed';abort('interrupted')
        for r in records:
            if r['status']=='pending':r['status']='interrupted'
        save();raise
    finally:
        for slot in slots:slot.close()
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
        for r in records:
            d=out/r['case'];d.mkdir(exist_ok=True);atomic_json(d/'selection.json',r)
    return records

def sha_bytes(content):
    import hashlib
    return hashlib.sha256(content).hexdigest()
