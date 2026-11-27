"""Finish a provisional geometry repair by stretching its new velocity pulses."""
import copy
import hashlib
import json
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation,Slerp

from .cli import atomic_json,sha
from .retiming import _retime_diagnostics,inserted_geometry_regression,resample_targets
from .source_integrity import eligible_timing_intervals
from .temporal_quality import assess
from .v8.self_collision import CollisionModel


def pulse_edges(before, after, start, end):
    """Only source-relative isolated pulses are eligible for this route."""
    old = before.get('temporal_quality',{}).get('joints',{})
    new = after.get('temporal_quality',{}).get('joints',{})
    edges = set()
    for name,stats in new.items():
        previous = old.get(name,{})
        added = set(stats['violating_frames'])-set(previous.get('violating_frames',()))
        if not added:
            continue
        impulses = set(stats.get('added_velocity_impulse_frames',()))
        if not added <= impulses or any(f < start or f >= end for f in added):
            return ()
        edges.update(f for f in impulses if f+1 in impulses and f-1 not in impulses
                     and (f in added or f+1 in added))
    return tuple(sorted(edges))


def context_hash(rows, rotations, start, end):
    digest = hashlib.sha256()
    for values in (rows,rotations):
        digest.update(np.ascontiguousarray(values[max(0,start-2):end+2],dtype=np.float64).tobytes())
    return digest.hexdigest()


def proposal_path(folder, order):
    return Path(folder)/'.contact-timing'/f'{int(order)}.npz'


def save_proposal(folder, order, before, after, rotations, span, edges, gain):
    path = proposal_path(folder,order)
    path.parent.mkdir(exist_ok=True)
    a,b = span
    np.savez_compressed(path,rows=after[a:b])
    return dict(order=order,span=[a,b],edges=list(edges),gain=gain,
                parent=context_hash(before,rotations,a,b),sha256=sha(path))


def restore_proposals(previous, folder, entries):
    restored = []
    for entry in entries:
        old = proposal_path(previous,entry['order'])
        if not old.is_file() or sha(old)!=entry['sha256']:
            continue
        new = proposal_path(folder,entry['order'])
        new.parent.mkdir(exist_ok=True)
        shutil.copyfile(old,new)
        restored.append(entry)
    return restored


def source_grid(n, edges):
    ticks = np.ones(n-1,dtype=np.int64)
    ticks[list(edges)] += 1
    keyframes = np.r_[0,np.cumsum(ticks)]
    return np.interp(np.arange(keyframes[-1]+1),keyframes,np.arange(n))


def new_jitter(before, after, index):
    """Map the independently detected output defects back to the old grid."""
    old = before.get('temporal_quality',{}).get('joints',{})
    new = after.get('temporal_quality',{}).get('joints',{})
    for name,stats in new.items():
        previous = set(old.get(name,{}).get('violating_frames',()))
        if any(int(np.floor(index[f])) not in previous and int(np.ceil(index[f])) not in previous
               for f in stats['violating_frames']):
            return True
    return False


def clean_subdivision(payload, rows, diagnostics, quality, eligible, edges, deadline, clock):
    """Try the mandatory pulse edges first, then neighbors for seam continuity."""
    n = len(rows)
    if any(f<0 or f>=n-1 or not eligible[f] for f in edges):
        return None
    neighbors = sorted({f+d for f in edges for d in (-1,1)
                        if 0<=f+d<n-1 and eligible[f+d]}-set(edges))
    choices = [set(edges),*(set(edges)|{f} for f in neighbors)]
    if neighbors:
        choices.append(set(edges)|set(neighbors))
    # Exact same detector as publication, evaluated on the source-time grid.
    names = [str(name) for name in payload['joint_names']]
    timing_keys = {prefix+field for prefix in ('Left_','Right_') for field in
                   ('thigh','shin','foot_target_r','upper','lower','elbow_axis','hand_forward','hand_width')}
    timing_keys.update(('root_r','torso_target'))
    for intervals in choices:
        if clock()>=deadline:
            break
        index = source_grid(n,intervals)
        selected = np.column_stack([np.interp(index,np.arange(n),col) for col in rows[:,3:].T])
        diag = {k:v for k,v in diagnostics.items() if k in timing_keys}
        _retime_diagnostics(diag,index,n)
        screen = dict(temporal_quality=assess(selected,payload['fps'],names,diag))
        if not new_jitter(quality,screen,index):
            return index, sorted(intervals)
    return None


def _write_candidate(path, payload, rows, index, meta, diagnostics, sampled, robot, intervals):
    grid = np.arange(len(rows))
    exact = index==np.rint(index)
    actual = Slerp(grid,Rotation.from_quat(payload['root_rot']))(index)
    quat = actual.as_quat().astype(np.float32)
    quat[exact] = payload['root_rot']
    stored = robot.export_rows(np.column_stack([np.interp(index,grid,col) for col in rows.T]))
    source_time = np.interp(index,grid,payload['source_time_s'])
    source_fps = float(payload.get('source_fps',meta['input_fps']))
    data = dict(payload,root_pos=stored[:,:3],dof_pos=stored[:,3:],root_rot=quat,
                source_time_s=source_time,source_frame_float=source_time*source_fps,source_fps=source_fps)
    path.write_bytes(pickle.dumps(data,protocol=4))
    np.savez_compressed(path.with_suffix('.npz'),**{k:v for k,v in data.items() if v is not None})
    diagnostics = dict(diagnostics)
    nearest = _retime_diagnostics(diagnostics,index,len(rows))
    if 'head_r' in sampled:
        from .v8.source import head_relative_target
        diagnostics['head_relative_target'] = head_relative_target(sampled['torso_r'],sampled['head_r'])
    for side,values in sampled['sides'].items():
        for name,value in values.items():
            key = side+'_'+name
            if key in diagnostics and isinstance(value,np.ndarray):
                diagnostics[key] = value
    diagnostics['source_time_s'] = source_time
    np.savez_compressed(path.with_suffix('.diagnostics.npz'),**diagnostics)
    description = copy.deepcopy(meta)
    for side,segments in description['estimated_contact_segments'].items():
        selected = []
        for s in segments:
            hits = np.flatnonzero((nearest>=s['start'])&(nearest<s['end']))
            if len(hits):
                selected.append(dict(s,start=int(hits[0]),end=int(hits[-1])+1))
        description['estimated_contact_segments'][side] = selected
    time_map = path.with_suffix('.time_map.npz')
    np.savez_compressed(time_map,output_time_s=np.arange(len(index))/payload['fps'],
                        source_time_s=source_time,source_frame_float=source_time*source_fps,
                        source_fps=source_fps,source_sha256=meta['input_sha256'],
                        policy='contact_impulse_local_slowdown_v1',pre_final_index=index)
    timing = description.setdefault('retiming',{})
    timing.update(changed=True,time_map=time_map.name,source_time_sha256=sha(time_map),
                  added_frames=int(timing.get('added_frames',0))+len(index)-len(rows),
                  final_adaptation=dict(candidate='contact_impulse_local_slowdown',
                      stage='after_temporal_cleanup',stretched_intervals=intervals,
                      added_frames=len(index)-len(rows),original_frames_fixed=True))
    description.update(output_frames=len(index),output_sha256=sha(path))
    atomic_json(path.with_suffix('.json'),description)
    return description,stored,Rotation.from_quat(quat).as_matrix()


def finish(path, meta, quality, solve, mode, context, entries, deadline):
    """Try the strongest non-stale geometry group and publish atomically."""
    import time
    from .acceptance import evaluate
    from .quality_v8 import SPEED_FLAG
    from .refine_pipeline import repair_limits
    from .unified_repair import quality_guard

    if context is None or not entries or SPEED_FLAG in quality['flags']:
        return meta,quality,False
    robot,targets,_ = context
    payload = pickle.loads(path.read_bytes())
    original = np.c_[payload['root_pos'],payload['dof_pos']].astype(float)
    rotations = payload['root_rot']
    candidates = []
    for entry in sorted(entries,key=lambda e:-e['gain']):
        a,b = entry['span']
        saved = proposal_path(path.parent,entry['order'])
        if context_hash(original,rotations,a,b)!=entry['parent'] or not saved.is_file() or sha(saved)!=entry['sha256']:
            continue
        with np.load(saved,allow_pickle=False) as file:
            rows = file['rows']
        if rows.shape!=original[a:b].shape or not np.isfinite(rows).all():
            continue
        candidates.append((entry,rows))
    if not candidates:
        meta.setdefault('retiming',{})['contact_impulse_attempt'] = dict(decision='stale_context')
        return meta,quality,False
    source_path = path.with_suffix('.source_review.json')
    review = json.loads(source_path.read_text()) if source_path.is_file() else None
    eligible = eligible_timing_intervals(review,payload['source_time_s'])
    with np.load(path.with_suffix('.diagnostics.npz'),allow_pickle=False) as archive:
        diag = {k:archive[k] for k in archive.files}
    # Disjoint proposals share one output clock. If their combination fails,
    # retry the strongest candidate by itself.
    chosen = []
    for item in candidates:
        a,b = item[0]['span']
        if all(b<=other[0]['span'][0] or a>=other[0]['span'][1] for other in chosen):
            chosen.append(item)
    attempts = (chosen,chosen[:1]) if len(chosen)>1 else (chosen,)
    planning_deadline = min(deadline,time.monotonic()+repair_limits()['specialist_candidate_seconds'])
    model = CollisionModel(robot,dict(meta['configuration'].get('self_collision') or {},coverage='full'))
    failures = []
    for group in attempts:
        if time.monotonic()>=deadline:
            break
        provisional = original.copy()
        edges = set()
        for entry,rows in group:
            a,b = entry['span']
            provisional[a:b] = rows
            edges.update(entry['edges'])
        plan = clean_subdivision(payload,provisional,diag,quality,eligible,edges,planning_deadline,time.monotonic)
        if plan is None:
            failures.append(dict(edges=sorted(edges),reason='blocked_source_or_unresolved_pulse'))
            continue
        index,intervals = plan
        folder = path.parent
        with tempfile.TemporaryDirectory(prefix='.contact-timing-',dir=folder.parent) as temp:
            work = Path(temp)
            candidate = work/'candidate'
            candidate.mkdir()
            blob = candidate/path.name
            sampled = resample_targets(targets,robot,index)
            next_meta,rows,actual = _write_candidate(blob,payload,provisional,index,meta,diag,sampled,robot,intervals)
            checked = evaluate(blob,next_meta,solve,mode)
            mapped = dict(sampled,source_root_r=sampled['root_r'],root_r=actual)
            geometry = inserted_geometry_regression(robot,mapped,model,original,rows,index)
            old_collision = quality['full_self_collision']
            collision = checked['full_self_collision']
            reason = geometry or quality_guard(quality,checked,True)
            added_flags = sorted(set(checked['flags'])-set(quality['flags']))
            if not reason and added_flags:
                reason = 'new_quality_flags:'+','.join(added_flags)
            if not reason and new_jitter(quality,checked,index):
                reason = 'new_jitter_on_source_time'
            if not reason and (collision['collision_samples']>=old_collision['collision_samples']
                               or collision['native_colliding_frames']>old_collision['native_colliding_frames']
                               or collision['minimum_distance_m']<min(old_collision['minimum_distance_m'],0.)-2e-6):
                reason = 'insufficient_collision_improvement'
            if reason:
                failures.append(dict(edges=intervals,reason=reason))
                continue
            next_meta['retiming']['final_adaptation']['decision'] = 'adopted_after_full_audit'
            next_meta['retiming']['pre_final_checkpoint'] = 'pre_final_retime'
            next_meta['self_collision']['export_check'] = checked['self_collision']
            next_meta['solver_velocity_check'] = checked['step_velocity']
            if 'unified_repair' in next_meta:
                next_meta['unified_repair'] = dict(next_meta['unified_repair'],
                    report_path='pre_final_retime/repair.json',status=checked['status'])
            atomic_json(blob.with_suffix('.json'),next_meta)
            shutil.copyfile(folder/'identity.json',candidate/'identity.json')
            if source_path.is_file():
                shutil.copyfile(source_path,candidate/source_path.name)
            original_folder = work/'original'
            folder.rename(original_folder)
            try:
                candidate.rename(folder)
                original_folder.rename(folder/'pre_final_retime')
            except OSError:
                if folder.exists():
                    folder.rename(candidate)
                original_folder.rename(folder)
                raise
            return next_meta,checked,True
    meta.setdefault('retiming',{})['contact_impulse_attempt'] = dict(decision='no_candidate_passed',attempts=failures)
    return meta,quality,False
