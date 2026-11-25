"""V8 segment trials with bounded local repair and full validation."""
import math
from pathlib import Path
import os
import pickle
import shutil
import tempfile
import time

import numpy as np
from osqp import OSQPException
from scipy.spatial.transform import Rotation

from .acceptance import evaluate
from .adaptive_refine import window_targets
from .audit_cache import fork
from .cli import atomic_json, sha
from .refine_pipeline import measurements, masks, propose, guard, enrich_jobs, repair_limits, normalized_input_frames
from .refine_queue import RECIPES, jobs_from_masks
from .refine_routing import collision_groups
from .unified_repair import quality_guard
from .segment_planner import plan_segments, next_segment
from .robot_profile import ROOT_EXTRA_DROP_TOLERANCE_M
from .v8.self_collision import CollisionModel

def root_drop_excess(rows, reference):
    """Extra downward displacement in metres per step relative to the source target."""
    return np.maximum(0., -np.diff(rows[:, 2])-np.maximum(0., -np.diff(reference[:, 2])))


def residual_evidence(quality, cache, robot, groups, targets):
    n = len(cache['data']['root_pos'])
    channels = {}
    for module, side, mask in masks(quality, cache, robot, groups, source_pose=True):
        if module in ('torso', 'arms', 'leg_pose'):
            family = 'pose'
        elif module == 'jitter':
            continue
        else:
            family = 'geometry'
        channels[(':'.join(filter(None,(module,side))))] = (family, mask.astype(float))
    report = quality['body_support']
    for name, part in report['contacts'].items():
        for metric, frames_key, tolerance in (
                ('normal_cm', 'normal_violating_frames', 'normal_tolerance_m'),
                ('fixed_xy_cm', 'fixed_xy_violating_frames', 'fixed_xy_tolerance_m')):
            indices = np.asarray(part[frames_key], dtype=int)
            if not indices.size:
                continue
            values = np.asarray(cache['frames']['body_errors'][name][metric], dtype=float)
            scores = np.zeros(n)
            limit = report[tolerance]
            scores[indices] = np.maximum(1., np.nan_to_num(values[indices]/(100.*limit),nan=1.))
            channels[f'{name}:{metric}'] = ('body_support', scores)
    floor = -np.min(cache['floor'],axis=1)
    severity = np.maximum(floor[::8],0.).copy()
    if n > 1:
        between = floor[:-1].reshape(n-1, 8).max(axis=1)
        severity[1:] = np.maximum(severity[1:],between)
        severity[:-1] = np.maximum(severity[:-1],between)
    channels['mesh_floor'] = ('geometry', np.where(severity > .001,severity/.001,0.))
    return channels


def blend(current, solved, lo, hi, start, end, writable):
    output = current.copy()
    alpha = np.ones(hi-lo)
    if lo and start > lo:
        phase = np.linspace(0,np.pi/2,start-lo,endpoint=False)
        alpha[:start-lo] = np.sin(phase)**2
    if hi < len(current) and end < hi:
        phase = np.linspace(np.pi/2,0,hi-end+1)[1:]
        alpha[end-lo:] = np.sin(phase)**2
    if lo:
        alpha[0] = 0.
    if hi < len(current):
        alpha[-1] = 0.
    rows = output[lo:hi]
    rows[writable] += alpha[writable,None]*(solved[writable]-rows[writable])
    return output



def project_joints(candidate, anchors):
    """Keep blended joints within reach of both the preceding row and the next anchor."""
    for i in np.flatnonzero(~anchors.fixed):
        lo, hi = anchors.bounds(i, candidate[i-1])
        candidate[i,3:] = np.clip(candidate[i,3:],lo[3:],hi[3:])


def pose_excess(cache, lo, hi):
    metrics = cache['metrics']
    residuals = [np.maximum(metrics['torso'][lo:hi]-20.,0.)/20.]
    for side, error in metrics['errors'].items():
        for name, limit in [('upper',20),('lower',20),('hand_forward',25),('hand_width',25),('elbow_axis',40)]:
            residuals.append(np.maximum(error[name][lo:hi]-limit,0.)/limit)
        for name in ('thigh_deg','shin_deg'):
            values = np.asarray(cache['frames']['errors'][side][name])[lo:hi]
            residuals.append(np.maximum(values-45.,0.)/45.)
    return np.column_stack(residuals)


def merit(quality, cache, robot, lo, hi):
    steps = slice(8*lo, 8*(hi-1)+1)
    geometry = np.asarray(cache['floor'])[steps]
    collision = cache['full_collision']['samples'][steps]
    support = measurements(cache,robot)['contact'][lo:hi]
    def count(key):
        return sum(lo <= i < hi for i in quality[key]['violating_frames'])
    return np.array([
        np.maximum(-geometry-.001,0.).sum()/.01,
        sum(len(sample[2]) for sample in collision)/8.,
        np.maximum(support-.001,0.).sum()/.01+count('body_support'),
        np.square(pose_excess(cache,lo,hi)).sum(),
        count('temporal_quality'),
    ])



def segment_guard(robot, targets, model, before, after, old_q, new_q, old_cache, new_cache, a, b):
    """Check the actual float32 candidate against each support/channel, including both joins."""
    from .temporal_constraints import regressions
    from .support_quality import preserve_contact
    from .unified_repair import dense_geometry
    from .similarity import palm_envelope

    lo, hi = max(0,a-1), min(len(before),b+1)
    section = slice(lo, hi)
    native = slice(a,b)
    swept = slice(8*lo,8*(hi-1)+1)
    reason = quality_guard(old_q,new_q,soft_similarity=True)
    if reason:
        return reason
    for k in ('root_pos','root_rot','dof_pos'):
        x,y = old_cache['data'][k],new_cache['data'][k]
        if not np.array_equal(x[:a],y[:a]) or not np.array_equal(x[b:],y[b:]):
            return 'outside_writable_segment:'+k
    speed = np.abs(np.diff(after[section,3:],axis=0))*targets['fps']
    prior = np.abs(np.diff(before[section,3:],axis=0))*targets['fps']
    if np.any(speed > np.maximum(prior,robot.speed_limits)+1e-4):
        return 'motor_speed_or_seam'
    old_drop = root_drop_excess(before[section],targets['root_p'][section])
    new_drop = root_drop_excess(after[section],targets['root_p'][section])
    if np.any(new_drop > np.maximum(old_drop,ROOT_EXTRA_DROP_TOLERANCE_M)+2e-6):
        return 'source_relative_root_drop_or_seam'
    if np.any((old_cache['metrics']['torso'][native]+.02) < new_cache['metrics']['torso'][native]):
        return 'framewise_torso_orientation_regression'
    if np.any(np.asarray(new_cache['floor'])[swept] < np.minimum(np.asarray(old_cache['floor'])[swept],-.001)-2e-6):
        return 'framewise_swept_floor_regression'
    for side, ids in robot.sides.items():
        knee = ids['leg'][3]
        if np.any((before[native,knee]>=0)&(after[native,knee]<-1e-7)):
            return 'new_knee_hyperextension:'+side
        for idx in range(lo, hi):
            previous = old_cache['frames']['contact_xyz'][side][idx]
            current = new_cache['frames']['contact_xyz'][side][idx]
            if previous is None and current is None:
                continue
            if previous is None or current is None:
                return 'lost_foot_anchor:'+side
            for axis in (slice(None), slice(0,2), slice(2,3)):
                if np.linalg.norm(current[axis]) > max(np.linalg.norm(previous[axis]),.1)+.0002:
                    return 'foot_anchor_regression:'+side
    contacts = {c['body_name']:c for c in targets.get('body_contacts',())}
    for name, parts in old_cache['frames']['body_errors'].items():
        for key, old in parts.items():
            current = new_cache['frames']['body_errors'][name][key]
            tolerance = old_q['body_support']['normal_tolerance_m' if key=='normal_cm' else 'fixed_xy_tolerance_m']*100.
            for i in range(lo,hi):
                if name not in contacts or not preserve_contact(contacts[name],i) or old[i] is None:
                    continue
                if current[i] is None or current[i] > max(old[i],tolerance)+.0002:
                    return 'body_support_regression:'+name+':'+key
    old_errors = old_cache['metrics']['errors']
    new_errors = new_cache['metrics']['errors']
    reason = palm_envelope({s:{k:v[native] for k,v in e.items()} for s,e in old_errors.items()},
                           {s:{k:v[native] for k,v in e.items()} for s,e in new_errors.items()})
    if reason:
        return reason
    if np.any(pose_excess(new_cache,lo,hi) > pose_excess(old_cache,lo,hi)+.15):
        return 'source_pose_regression'
    if regressions(old_q['temporal_quality'],new_q['temporal_quality']):
        return 'added_joint_jitter'
    samples = new_cache['full_collision']['samples'][swept]
    if any(sample[2] for sample in samples):
        local = window_targets(targets,lo,hi,len(before))
        densities = []
        for rows, cache in ((before,old_cache),(after,new_cache)):
            local['root_r'] = Rotation.from_quat(cache['data']['root_rot'][section]).as_matrix()
            densities.append(dense_geometry(robot,model,local,rows[section])[0])
        if np.any(densities[1] < np.minimum(0.,densities[0])-2e-6):
            return 'collision_pair_regression'
    return None


def publish_trial(trial, path, data):
    """Prepare complete exports, then commit via atomic renames, authoritative PKL last."""
    np.savez_compressed(trial.with_suffix('.npz'), **{k:v for k,v in data.items() if v is not None})
    suffixes = ('.npz','.check.json','.pose_errors.npz','.json','.pkl')
    # On a failed rename restore the last accepted exports, including derivative files.
    with tempfile.TemporaryDirectory(dir=path.parent,prefix='.rollback-') as directory:
        saved = Path(directory)
        replaced = []
        for suffix in suffixes:
            old=path.with_suffix(suffix)
            if old.exists():
                os.link(old,saved/old.name)
        try:
            for suffix in suffixes:
                new=path.with_suffix(suffix)
                os.replace(trial.with_suffix(suffix),new)
                replaced.append(new)
        except OSError:
            for new in reversed(replaced):
                backup=saved/new.name
                if backup.exists():
                    os.replace(backup,new)
                else:
                    new.unlink(missing_ok=True)
            raise


def refine(robot, targets, meta, payload, path, initial_solver, budget, *,
           deadline=None, initial_cache=None, initial_quality=None,
           plan_only=False, previous_attempts=(), checkpoint=None, specialist_step=None, limits=None, next_dispatch=None,
           segment_filter=None):
    from .airborne import protected_height
    from .v8.solver import WindowAnchors, solve as v8_solve
    from .v8.convert import support_solver

    path = Path(path)
    cache = {} if initial_cache is None else initial_cache
    quality = initial_quality if initial_quality is not None else evaluate(path,meta,initial_solver,'frax',cache=cache)
    baseline = quality
    out = np.c_[payload['root_pos'],payload['dof_pos']].astype(float)
    fixed = path.parent/'avoidance_reference.npy'
    reference = np.load(fixed,allow_pickle=False) if fixed.is_file() else out.copy()
    current_rot = Rotation.from_quat(payload['root_rot']).as_matrix()
    targets = dict(targets,source_root_r=targets.get('source_root_r',targets['root_r']),root_r=current_rot)
    airborne = protected_height(targets,len(out))
    fps = targets['fps']
    source_time = np.asarray(payload.get('source_time_s', np.arange(len(out))/fps))
    weights = meta['configuration']['solver']
    model = CollisionModel(robot,dict(meta['configuration'].get('self_collision') or {},coverage='full'))
    groups = collision_groups(model)
    contacts = np.column_stack([v['contact'] for v in targets['sides'].values()] +
                               [c['mode'] for c in targets.get('body_contacts',[])])
    evidence = residual_evidence(quality,cache,robot,groups,targets)
    jobs = plan_segments(evidence,contacts,fps,source_time)

    def target_loss(channels, job):
        core = slice(job['core_start'],job['core_end'])
        return sum(float(values[core].sum()) for group,values in channels.values() if group==job['primary'])
    if plan_only:
        return dict(meta,_segment_plan=dict(segments=jobs))
    limits = repair_limits() if limits is None else limits
    solver_seconds = limits['v8_segment_seconds']
    plugin_seconds = limits['specialist_candidate_seconds']
    began = time.monotonic()
    end = deadline if deadline is not None else began+(.5*normalized_input_frames(meta,len(out)) if budget is None else budget)
    attempts = list(previous_attempts or ())
    parent_hash = sha(path)
    def audit(rows,quats,changed,trial):
        data = dict(payload,root_pos=rows[:,:3].astype(np.float32),dof_pos=rows[:,3:].astype(np.float32),
                    root_rot=np.asarray(quats,dtype=np.float32))
        trial.write_bytes(pickle.dumps(data,protocol=4))
        description = dict(meta,output_sha256=sha(trial))
        atomic_json(trial.with_suffix('.json'),description)
        cached = fork(cache,changed=changed)
        return data,cached,evaluate(trial,description,initial_solver,'frax',cache=cached,changed=changed)

    def post_job(check,cached,lo,hi):
        scoped = []
        for module,side,mask in masks(check,cached,robot,groups,source_pose=True):
            if module not in RECIPES or module == 'jitter':
                continue
            if segment_filter is not None and module in ('torso','arms','leg_pose'):
                continue
            mask = mask.copy()
            mask[:lo]=False
            mask[hi:]=False
            if mask.any():
                scoped.append((module,side,mask))
        posts = []
        for job in jobs_from_masks(scoped,len(out)):
            a,b=max(lo,job['start']),min(hi,job['end'])
            if b-a>=5:
                posts.append(dict(job,start=a,end=b,core_start=max(a,job['core_start']),core_end=min(b,job['core_end'])))
        enriched = enrich_jobs(posts,targets,robot,cached,model)
        for job in enriched:
            return dict(job,recipe=job.get('candidate_order',RECIPES[job['module']])[0])
        return None

    def phase_metrics(result):
        return dict(
            collision_frames=result['full_self_collision']['native_colliding_frames'],
            swept_floor_over_1mm=result['interpolated_floor']['samples_below_minus_1mm'],
            flags=result['flags'])

    with tempfile.TemporaryDirectory(dir=path.parent,prefix='.segment-trial-') as scratch:
        trial = Path(scratch)/'motion.pkl'
        shutil.copyfile(path.with_suffix('.diagnostics.npz'),trial.with_suffix('.diagnostics.npz'))
        phase_started = time.monotonic()
        phase = dict(before=phase_metrics(quality), allocated_s=max(0.,end-phase_started) if math.isfinite(end) else None,
                     elapsed_scope='current_invocation', metrics_scope='shared_geometry')
        entries = [attempt for attempt in attempts if attempt.get('solver')=='v8']
        ledger = {j['segment_id']:dict(j,status='not_attempted') for j in jobs}
        for previous in entries:
            ledger[previous['segment_id']] = dict(previous)
        # Give each unresolved historical timeout one new opportunity per invocation.
        retry_ids = {key for key, item in ledger.items() if item['status'] == 'time_slice'}

        def refresh_plan():
            nonlocal jobs
            jobs = plan_segments(evidence,contacts,fps,source_time)
            current_ids = {j['segment_id'] for j in jobs}
            for item in ledger.values():
                if item['segment_id'] not in current_ids and item['status']=='not_attempted':
                    item['status']='superseded_after_acceptance'
            for job in jobs:
                ledger.setdefault(job['segment_id'],dict(job,status='not_attempted'))

        specialist_seconds = 0.
        while time.monotonic()<end:
            specialist_attempted = False
            if specialist_step is not None:
                started_step = time.monotonic()
                new_payload,new_quality,new_cache,specialist_attempted = specialist_step(payload,quality,cache)
                if new_payload is not payload:
                    payload,quality,cache = new_payload,new_quality,new_cache
                    out = np.c_[payload['root_pos'],payload['dof_pos']].astype(float)
                    targets['root_r'] = Rotation.from_quat(payload['root_rot']).as_matrix()
                    if fixed.is_file():
                        reference = np.load(fixed,allow_pickle=False)
                    parent_hash = sha(path)
                    evidence = residual_evidence(quality,cache,robot,groups,targets)
                    refresh_plan()
                specialist_seconds += time.monotonic() - started_step
            if time.monotonic() >= end:
                break
            eligible_jobs=jobs if segment_filter is None else segment_filter(jobs,payload,quality,cache)
            job = next_segment(eligible_jobs,entries,retry_ids)
            if job is None:
                if specialist_attempted:
                    continue
                break
            lo,hi=job['start'],job['end']
            read_lo,read_hi=job['read_start'],job['read_end']
            changed=(max(0,lo-1),min(len(out),hi+1))
            record=dict(job,solver='v8',module='segment',recipe='v8_continuous',
                        status='rejected',parent_sha256=parent_hash)
            if next_dispatch is not None:
                record['dispatch_index'] = next_dispatch()
            retry_ids.discard(job['segment_id'])
            start = time.monotonic()
            cutoff=min(end,start+solver_seconds)
            record['window_allowance_s']=max(0.,cutoff-start)
            accepted=None
            try:
                local=window_targets(targets,read_lo,read_hi,len(out))
                fixed_frames = airborne[read_lo:read_hi].copy()
                fixed_frames[:lo-read_lo+1] = True
                fixed_frames[hi-read_lo-1:] = True
                record['fixed_frames'] = (np.flatnonzero(fixed_frames)+read_lo).tolist()
                if fixed_frames.all():
                    raise ValueError('no_writable_segment_frames')
                anchors = WindowAnchors(robot,out[read_lo:read_hi],fixed_frames,fps)
                config=dict(weights,_repair_deadline=cutoff)
                try:
                    solved,_=v8_solve(robot,local,config,anchors=anchors)
                    solved,_=support_solver(config)(robot,local,solved,config,anchors=anchors)
                finally:
                    record['solve_elapsed_s'] = time.monotonic() - start
                if time.monotonic() >= cutoff:
                    raise TimeoutError('v8_candidate_deadline')
                if solved.shape!=anchors.rows.shape or not np.isfinite(solved).all():
                    raise ValueError('invalid_solver_rows')
                if not np.array_equal(solved[fixed_frames],anchors.rows[fixed_frames]):
                    raise ValueError('protected_segment_pose_changed')
                writable = slice(lo-read_lo,hi-read_lo)
                candidate=blend(out,solved[writable],lo,hi,job['core_start'],job['core_end'],~fixed_frames[writable])
                project_joints(candidate[read_lo:read_hi],anchors)
                candidate=robot.export_rows(candidate).astype(float)
                if np.array_equal(candidate,out):
                    raise ValueError('unchanged_candidate')
                data,cached,check=audit(candidate,payload['root_rot'],changed,trial)
                # An orientation or severe penetration regression already precludes a local cleanup.
                reason=quality_guard(quality,check,soft_similarity=True)
                if reason:
                    raise ValueError(reason)
                sub=post_job(check,cached,lo+1,hi-1) if time.monotonic()<cutoff else None
                if sub is not None:
                    record['post_plugin']=dict(sub,status='rejected')
                    try:
                        plugin_start = time.monotonic()
                        plugin_deadline = min(cutoff, plugin_start+plugin_seconds)
                        try:
                            repaired,detail=propose(robot,targets,candidate,reference,weights,model,sub,sub['recipe'],
                                                    plugin_deadline)
                        finally:
                            record['post_plugin']['solve_elapsed_s'] = time.monotonic() - plugin_start
                        if time.monotonic() >= plugin_deadline:
                            raise TimeoutError('post_plugin_deadline')
                        repaired=robot.export_rows(repaired).astype(float)
                        # The body-support plugin returns per-frame records.
                        # Only plugins returning a mapping can carry rotations.
                        rr=detail.get('_pelvis_rotations') if isinstance(detail,dict) else None
                        quats=data['root_rot'].copy()
                        if rr is not None:
                            moved=np.any(rr!=targets['root_r'],axis=(1,2))
                            quats[moved]=Rotation.from_matrix(rr[moved]).as_quat()
                        locked = airborne.copy()
                        locked[read_lo:read_hi] |= fixed_frames
                        if (not np.array_equal(repaired[locked],candidate[locked])
                                or not np.array_equal(quats[locked],data['root_rot'][locked])):
                            raise ValueError('protected_segment_pose_changed')
                        new_data,new_cache,new_check=audit(repaired,quats,changed,trial)
                        trial_targets = dict(targets,root_r=Rotation.from_quat(quats).as_matrix()) if rr is not None else None
                        reason=guard(robot,targets,candidate,repaired,reference,
                                     measurements(cached,robot),measurements(new_cache,robot),
                                     check,new_check,cached,new_cache,model,
                                     dict(sub,source_pose_guard=True,surface_support_pose=sub['recipe']=='surface_support_pose'),
                                     new_targets=trial_targets)
                        if reason:
                            raise ValueError(reason)
                        old,new=merit(check,cached,robot,*changed),merit(new_check,new_cache,robot,*changed)
                        if np.any(new>old+1e-5) or not np.any(new<old-1e-4):
                            raise ValueError('post_plugin_no_improvement')
                        candidate,data,cached,check=repaired,new_data,new_cache,new_check
                        record['post_plugin']['status']='combined_for_audit'
                    except (ValueError,RuntimeError,TimeoutError,OSQPException) as error:
                        record['post_plugin']['reason']=str(error)
                        data,cached,check=audit(candidate,payload['root_rot'],changed,trial)
                if not np.isfinite(candidate).all() or np.any(candidate[lo:hi,3:]<robot.lower[3:]-2e-6) or np.any(candidate[lo:hi,3:]>robot.upper[3:]+2e-6):
                    raise ValueError('invalid_configuration')
                reason=segment_guard(robot,targets,model,out,candidate,quality,check,cache,cached,lo,hi)
                if reason:
                    raise ValueError(reason)
                old,new=merit(quality,cache,robot,*changed),merit(check,cached,robot,*changed)
                old_drop=root_drop_excess(out[lo:hi],targets['root_p'][lo:hi])
                new_drop=root_drop_excess(candidate[lo:hi],targets['root_p'][lo:hi])
                old=np.r_[old,np.maximum(old_drop-ROOT_EXTRA_DROP_TOLERANCE_M,0.).sum()]
                new=np.r_[new,np.maximum(new_drop-ROOT_EXTRA_DROP_TOLERANCE_M,0.).sum()]
                candidate_evidence=residual_evidence(check,cached,robot,groups,targets)
                before_target=target_loss(evidence,job)
                after_target=target_loss(candidate_evidence,job)
                record.update(before=old.tolist(),after=new.tolist(),
                              target_loss_before=before_target,target_loss_after=after_target)
                if after_target>=before_target-1e-5:
                    raise ValueError('target_residual_not_improved')
                old=np.r_[old,before_target]
                new=np.r_[new,after_target]
                if np.any(new>old+1e-5) or not np.any(new<old-1e-4):
                    raise ValueError('no_safe_metric_improvement')
                accepted=candidate,data,cached,check
            except TimeoutError as error:
                record.update(status='time_slice',reason=str(error))
            except (ValueError,RuntimeError,OSQPException) as error:
                record['reason']=str(error)
            if accepted is not None:
                candidate,new_payload,_,_ = accepted
                moved = (np.any(candidate[lo:hi]!=out[lo:hi],axis=1) |
                         np.any(new_payload['root_rot'][lo:hi]!=payload['root_rot'][lo:hi],axis=1))
                publish_trial(trial,path,accepted[1])
                out,payload,cache,quality=accepted
                targets['root_r']=Rotation.from_quat(payload['root_rot']).as_matrix()
                parent_hash=sha(path)
                record.update(status='kept',output_sha256=parent_hash,
                              changed_frames=(lo+np.flatnonzero(moved)).tolist())
            record['seconds']=time.monotonic()-start
            if 'solve_elapsed_s' in record:
                record['validation_and_commit_elapsed_s'] = max(
                    0., record['seconds'] - record['solve_elapsed_s']
                    - record.get('post_plugin', {}).get('solve_elapsed_s', 0.))
            entries.append(record)
            attempts.append(record)
            ledger[record['segment_id']]=dict(record)
            if checkpoint:
                checkpoint(attempts)
            atomic_json(path.parent/'cascade_segments.json',dict(status='running',attempts=attempts))
            print('segment','v8',lo,hi,record['status'],record.get('reason',''),flush=True)
            if accepted is not None:
                evidence=candidate_evidence
                refresh_plan()
        eligible_jobs=jobs if segment_filter is None else segment_filter(jobs,payload,quality,cache)
        eligible_ids={job['segment_id'] for job in eligible_jobs}
        for item in ledger.values():
            if item['status']=='not_attempted':
                if item['segment_id'] in eligible_ids or segment_filter is None:
                    item['reason']='phase_budget_exhausted'
                else:
                    item.update(status='deferred',reason='window_order')
        attempted=sum(i['status'] in ('kept','rejected','time_slice') for i in ledger.values())
        pending=sum(i['status']=='not_attempted' for i in ledger.values())
        pending_retries = [j['segment_id'] for j in eligible_jobs if ledger[j['segment_id']]['status']=='time_slice']
        coverage = dict(eligible=attempted+pending,attempted=attempted,not_attempted=pending,
                        retry_pending=pending_retries,unavailable=None,segments=list(ledger.values()))
        phase.update(after=phase_metrics(quality),elapsed_s=max(0.,time.monotonic()-phase_started-specialist_seconds))
    description=dict(meta,output_sha256=sha(path))
    atomic_json(path.with_suffix('.json'),description)
    phase['attempt_elapsed_s'] = sum(a.get('seconds',0.) for a in entries)
    exhausted = bool(pending or pending_retries) and (segment_filter is None or time.monotonic()>=end)
    summary=dict(status='pending_budget' if quality['flags'] and exhausted else quality['status'],
                 kept_candidates=sum(a['status']=='kept' for a in attempts),attempts=attempts,coverage={'v8':coverage},
                 candidate_limit_s=solver_seconds,phases={'v8':phase},
                 pending_segments=[[j['start'],j['end'],j['core_start'],j['core_end']] for j in eligible_jobs],
                 residual_actions=jobs,before=baseline,after=quality,after_sha256=description['output_sha256'],
                 final_fresh_audit_equal=None,independent_audit_scope='pending_final_output')
    atomic_json(path.parent/'cascade_segments.json',summary)
    description['stage_times_s']['segment_cascade']=time.monotonic()-began-specialist_seconds
    description['self_collision']['export_check']=quality['self_collision']
    return dict(description,_unified_quality=quality,_segment_audit_cache=cache)
