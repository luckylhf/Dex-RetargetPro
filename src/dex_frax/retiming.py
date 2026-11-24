"""Opt-in local slowdown of prepared targets; never re-estimate calibration.

The output remains uniformly sampled. A monotone source-time map makes every
pose comparison refer to the corresponding original instant, not a shifted
frame number. This changes timing, not the source pose or acceptance limits.
"""
import copy
import time
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

POLICY='prepared_target_local_root_slowdown_v2'
DIRECTIONS={'thigh','shin','upper','lower','elbow_axis','hand_forward','hand_width','forearm_width'}
DISCRETE={'contact','phase','mode','anchors','pivots','source_expected_contact','source_airborne','confidence','ground_contact_confidence','platform_anchor','platform_index','source_scene_support','source_support_uncertain'}
STATIC={'robot_hull','hull','world_origin','local_axis'}


def time_map(root_r,root_p,fps,angular_limit,linear_limit,*,eligible=None,joint_positions=None,joint_limits=None,keep_original_frames=False):
    rates = [fps, *(value for value in (angular_limit, linear_limit) if value is not None)]
    if not np.isfinite(rates).all() or min(rates)<=0:
        raise ValueError('Invalid retiming rates')
    n=len(root_r)
    if n<2 or np.shape(root_r)!=(n,3,3) or np.shape(root_p)!=(n,3):raise ValueError('Invalid retiming targets')
    angles=Rotation.from_matrix(root_r[:-1].transpose(0,2,1)@root_r[1:]).magnitude()
    lengths=np.linalg.norm(np.diff(root_p,axis=0),axis=1)
    dt=np.full(n-1,1/fps)
    if angular_limit is not None:dt=np.maximum(dt,angles/angular_limit)
    if linear_limit is not None:dt=np.maximum(dt,lengths/linear_limit)
    joint_stretched=[]
    if joint_positions is not None:
        joints=np.asarray(joint_positions,dtype=float)
        limits=np.asarray(joint_limits,dtype=float)
        if (joints.shape!=(n,len(limits)) or not np.isfinite(joints).all() or
                not np.isfinite(limits).all() or np.any(limits<=0)):
            raise ValueError('Invalid robot joint retiming inputs')
        joint_dt=np.max(np.abs(np.diff(joints,axis=0))/limits,axis=1)
        joint_stretched=np.flatnonzero(joint_dt>1/fps+1e-12).tolist()
        dt=np.maximum(dt,joint_dt)
    blocked=[]
    if eligible is not None:
        eligible=np.asarray(eligible,dtype=bool)
        if eligible.shape!=(n-1,):raise ValueError('Invalid source interval mask')
        blocked=np.flatnonzero((~eligible)&(dt>1/fps+1e-12)).tolist()
        dt=np.where(eligible,dt,1/fps)
    changed=np.flatnonzero(dt>1/fps+1e-12)
    if not len(changed):return np.arange(n,dtype=float),dict(policy=POLICY,changed=False,source_duration_s=(n-1)/fps,output_duration_s=(n-1)/fps,stretched_intervals=[],blocked_source_review_intervals=blocked)
    if keep_original_frames:
        dt = np.ceil(dt*fps-1e-10)/fps
    clock=np.r_[0.,np.cumsum(dt)]
    count=int(np.ceil(clock[-1]*fps-1e-9))+1
    output_time=np.arange(count)/fps
    source_index=np.interp(output_time,clock,np.arange(n))
    # The final <=1 frame is a hold, not a global rescaling or a dropped tail.
    exact=np.abs(source_index-np.rint(source_index))<1e-9
    source_index[exact]=np.rint(source_index[exact])
    source_index[-1]=n-1
    return source_index,dict(policy=POLICY,changed=True,source_duration_s=(n-1)/fps,output_duration_s=(count-1)/fps,
        source_frames=n,output_frames=count,stretched_intervals=changed.tolist(),
        angular_target_rad_s=angular_limit,linear_target_m_s=linear_limit,
        source_angular_max_rad_s=float(angles.max()*fps),source_linear_max_m_s=float(lengths.max()*fps),
        final_hold_s=float(output_time[-1]-clock[-1]), blocked_source_review_intervals=blocked,
        joint_stretched_intervals=sorted(set(joint_stretched).intersection(changed.tolist())))


def apply(targets,robot,angular_limit,linear_limit,**options):
    fps=float(targets['fps'])
    indices,report=time_map(targets['root_r'],targets['root_p'],fps,angular_limit,linear_limit,**options)
    if not report['changed']:return targets,indices/fps,report
    out=resample_targets(targets,robot,indices)
    report.update(downstream_ik_required=True,interpolated_trajectory_is_seed_only=True,
                  calibration='preserved from full original sequence',
                  contact_phases='nearest original phase; no redetection',
                  reference='original source at source_time_s')
    return out,indices/fps,report


def resample_targets(targets,robot,indices):
    fps=float(targets['fps'])
    n=len(targets['root_r'])
    nearest=np.clip(np.floor(indices+.5).astype(int),0,n-1)
    exact=indices==np.rint(indices)
    left=np.floor(indices).astype(int);right=np.minimum(left+1,n-1);alpha=indices-left
    def visit(value,key=None):
        if isinstance(value,np.ndarray):
            if key in STATIC or not value.ndim or len(value)!=n:return value.copy()
            if key in DISCRETE or value.dtype.kind in 'biuUS':return value[nearest].copy()
            if value.shape[1:]==(3,3):
                out=Slerp(np.arange(n),Rotation.from_matrix(value))(indices).as_matrix()
                out[exact]=value[left[exact]]
                return out
            shape=(len(indices),)+(1,)*(value.ndim-1)
            out=value[left]*(1-alpha.reshape(shape))+value[right]*alpha.reshape(shape)
            if key in DIRECTIONS:
                length=np.linalg.norm(out,axis=-1,keepdims=True)
                if np.any(length<1e-8):raise ValueError('Ambiguous antipodal source direction during retiming')
                out=out/length
            out[exact]=value[left[exact]]
            return out
        if isinstance(value,dict):return {k:visit(v,k) for k,v in value.items()}
        if isinstance(value,list):return [visit(v) for v in value]
        return copy.deepcopy(value)
    out=visit(targets)
    for side,v in out['sides'].items():
        v['segments']=[]
        for segment in targets['sides'][side].get('segments',[]):
            hits=np.flatnonzero((nearest>=segment['start'])&(nearest<segment['end']))
            if len(hits):v['segments'].append(dict(segment,start=int(hits[0]),end=int(hits[-1]+1)))
        active=(v['contact']>0)&~exact
        # Discrete anchors/pivots must not blend across contact switches.
        # Rebuild only the existing active-foot geometric target, not contact
        # detection, scale, floor, confidence, or source calibration.
        if np.any(active):
            R=v['foot_target_r'][active];g=robot.sides[side]
            v['foot_target_p'][active]=v['anchors'][active]-np.einsum('nij,nj->ni',R,v['pivots'][active])
            v['foot_target_p'][active,2]=-np.min(R[:,2,:]@g['hull'].T,axis=1)
        for key in ('heel','toe'):
            v[key+'_speed']=np.linalg.norm(np.gradient(v[key],axis=0)*fps,axis=1)
    from .airborne import preserve_targets
    preserve_targets(out)
    return out


def motor_causes(joints, fps, names, limits, intervals, source_time):
    rates = np.abs(np.diff(joints,axis=0))*fps
    return [dict(interval=int(i),source_interval_s=source_time[i:i+2].tolist(),
                 joints={name:dict(speed_rad_s=float(value),limit_rad_s=float(limit),
                                   planning_target_rad_s=float(limit*.999))
                         for name,value,limit in zip(names,rates[i],limits) if value>limit*.999})
            for i in intervals]


def fit_inserted_frames(robot, targets, weights, rows, index, rotations=None, *, deadline=float('inf')):
    """Fit Z and hinges at added poses with interpolated root XY."""
    from scipy.optimize import least_squares
    from .v8.solver import Objective

    old = np.arange(len(rows))
    exact = index == np.rint(index)
    if not np.array_equal(index[exact], old):
        raise ValueError('Motor retiming must retain each original pose exactly once')
    fitted = np.column_stack([np.interp(index, old, col) for col in np.asarray(rows, float).T])
    fixed = np.flatnonzero(exact)
    inserted = np.flatnonzero(~exact)
    joint_step = np.asarray(robot.speed_limits)/targets['fps']
    if rotations is not None:
        targets = dict(targets, root_r=rotations)
    for i in inserted:
        end = fixed[np.searchsorted(fixed, i)]
        lo = robot.lower.copy()
        hi = robot.upper.copy()
        lo[3:] = np.maximum.reduce((lo[3:], fitted[i-1,3:]-joint_step, fitted[end,3:]-(end-i)*joint_step))
        hi[3:] = np.minimum.reduce((hi[3:], fitted[i-1,3:]+joint_step, fitted[end,3:]+(end-i)*joint_step))
        if np.any(hi < lo-1e-9):
            raise ValueError('Inserted pose cannot reach both fixed ends within motor limits')
        movable = hi-lo > 1e-9
        movable[:2] = False
        seed = np.clip(fitted[i],lo,hi)
        seed[:2] = fitted[i,:2]
        obj = Objective(robot,targets,i,weights)
        def configuration(values):
            if time.monotonic() >= deadline:
                raise TimeoutError('inserted_frame_ik_deadline')
            trial = seed.copy()
            trial[movable] = values
            return trial
        result = least_squares(lambda v: obj.fun(configuration(v)), seed[movable],
            jac=lambda v: obj.jac(configuration(v))[:,movable], bounds=(lo[movable],hi[movable]),
            max_nfev=50, ftol=1e-6, xtol=1e-6, gtol=1e-5)
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError(f'Local retiming IK failed at inserted output frame {i}')
        fitted[i] = configuration(result.x)
    return fitted


def _retime_diagnostics(data, index, count):
    base = np.arange(count)
    nearest = np.clip(np.floor(index+.5).astype(int),0,count-1)
    for name,array in data.items():
        if not array.ndim or len(array)!=count:
            continue
        if (array.dtype.kind in 'biuUS' or name=='solver_records' or name.endswith(
                ('_contact','_phase','_mode','_anchors','_pivots','_confidence','_platform_anchor'))):
            data[name] = array[nearest]
        elif array.shape[1:]==(3,3):
            data[name] = Slerp(base,Rotation.from_matrix(array))(index).as_matrix()
        else:
            compact = array.reshape(count,-1)
            data[name] = np.column_stack([np.interp(index,base,col) for col in compact.T]).reshape(
                (len(index),*array.shape[1:]))
    return nearest


def inserted_geometry_regression(robot, targets, model, parent, fitted, index):
    """Compare each new interval against the same original poses and sample times."""
    from .discontinuities import root_export_impulses
    from .inserted_frame_repair import impulse_regression, spans
    from .post_leg_contact import contact_errors, contact_regressions
    from .unified_repair import dense_geometry

    clock = np.arange(len(parent))
    linear = np.column_stack([np.interp(index,clock,col) for col in parent.T])
    def impulses(rows):
        reference = targets.get('source_root_r',targets['root_r'])
        return dict(root_discontinuities=root_export_impulses(rows[:,:3],targets['root_r'],
            targets['root_p'],reference,targets['fps']))
    if impulse_regression(impulses(linear),impulses(fitted)):
        return 'inserted_root_impulse_regression'
    tolerance = model.cfg['penetration_tolerance_m']
    for a,b in spans(index,len(fitted)):
        lo,hi = max(0,a-2),min(len(fitted),b+2)
        region = slice(lo,hi)
        local = {'root_r':targets['root_r'][region]}
        distance,floor = dense_geometry(robot,model,local,linear[region])
        new_distance,new_floor = dense_geometry(robot,model,local,fitted[region])
        if np.any((new_distance < -tolerance) & ((distance >= -tolerance) | (new_distance < distance-2e-6))):
            return f'inserted_collision_regression:{a}:{b}'
        if np.any(new_floor < np.minimum(floor,-.001)-2e-6):
            return f'inserted_floor_regression:{a}:{b}'
        before = contact_errors(robot,targets,linear,lo,hi)
        after = contact_errors(robot,targets,fitted,lo,hi)
        if contact_regressions(before,after):
            return f'inserted_support_regression:{a}:{b}'
    return None


def final_candidate(path, meta, quality, solve, mode, context=None, *, deadline=float('inf')):
    """Try one fixed-anchor retiming for the exact exported hinges.

    Candidate acceptance uses a full fresh audit. The original queue and
    reference stay together on their original time grid.
    """
    import json
    import pickle
    import shutil
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from .acceptance import evaluate
    from .cli import atomic_json, sha
    from .inserted_frame_repair import recover
    from .quality_v8 import SPEED_FLAG
    from .refine_pipeline import repair_limits
    from .robot_profile import joint_angular_limits
    from .source_integrity import eligible_timing_intervals
    from .v8.self_collision import CollisionModel

    if SPEED_FLAG not in quality['flags']:
        if context is not None:
            meta.setdefault('retiming',{})['final_adaptation'] = dict(decision='within_rated_joint_speed',inserted_ik_frames=0,
                                                                    stage='after_temporal_cleanup')
            atomic_json(path.with_suffix('.json'),meta)
        return meta,quality
    if context is None:
        meta.setdefault('retiming',{})['final_adaptation'] = dict(decision='prepared_ik_context_unavailable')
        atomic_json(path.with_suffix('.json'),meta)
        return meta,quality
    def defer():
        meta.setdefault('retiming',{})['final_adaptation'] = dict(
            candidate='final_joint_slowdown', stage='after_temporal_cleanup', decision='pending_budget')
        atomic_json(path.with_suffix('.json'),meta)
        return meta,quality

    if time.monotonic() >= deadline:
        return defer()
    window_seconds = repair_limits()['specialist_candidate_seconds']
    folder = path.parent
    payload = pickle.loads(path.read_bytes())
    q = payload['dof_pos']
    fps = float(payload['fps'])
    rotations = Rotation.from_quat(payload['root_rot'])
    source_time = payload.get('source_time_s',np.arange(len(q))/fps)
    limits = np.asarray(joint_angular_limits(payload['joint_names']))
    source_path = path.with_suffix('.source_review.json')
    review = json.loads(source_path.read_text()) if source_path.is_file() else None
    exceeds_speed = np.any(np.abs(np.diff(q.astype(float),axis=0))*fps > limits+1e-4,axis=1)
    index, attempt = time_map(rotations.as_matrix(),payload['root_pos'],fps,None,None,
                             eligible=eligible_timing_intervals(review,source_time)&exceeds_speed,
                             joint_positions=q,joint_limits=limits,keep_original_frames=True)
    attempt['candidate'] = 'final_joint_slowdown'
    attempt['stage'] = 'after_temporal_cleanup'
    if time.monotonic() >= deadline:
        return defer()
    if not attempt['changed']:
        attempt['decision'] = 'blocked_or_no_stretch'
    else:
        attempt['motor_causes'] = motor_causes(q,fps,payload['joint_names'],limits,attempt['joint_stretched_intervals'],source_time)
        with TemporaryDirectory(prefix='.final-timing-',dir=folder.parent) as temp:
            work = Path(temp)
            candidate = work/'candidate'
            candidate.mkdir()
            blob = candidate/path.name
            grid = np.arange(len(q))
            next_source = np.interp(index,grid,source_time)
            source_fps = float(payload.get('source_fps',meta['input_fps']))
            robot, targets, weights = context
            sampled = resample_targets(targets,robot,index)
            actual_r = Slerp(grid,rotations)(index)
            original_rows = np.c_[payload['root_pos'],q].astype(float)
            linear = np.column_stack([np.interp(index,grid,col) for col in original_rows.T])
            try:
                fitted = fit_inserted_frames(robot,sampled,weights,original_rows,index,actual_r.as_matrix(),
                                             deadline=min(deadline,time.monotonic()+window_seconds))
            except (RuntimeError, ValueError, TimeoutError) as error:
                attempt['ik_rejection'] = str(error)
                fitted = linear
            if time.monotonic() >= deadline:
                return defer()
            attempt.update(inserted_ik_frames=len(index)-len(q),original_frames_fixed=True,
                           inserted_root_xy='linear_between_original_poses')
            quat = actual_r.as_quat().astype(np.float32)
            exact = index == np.rint(index)
            quat[exact] = payload['root_rot']
            payload.update(root_pos=fitted[:,:3].astype(np.float32),dof_pos=fitted[:,3:].astype(np.float32),root_rot=quat,
                           source_time_s=next_source,source_frame_float=next_source*source_fps,source_fps=source_fps)
            blob.write_bytes(pickle.dumps(payload,protocol=4))
            np.savez_compressed(blob.with_suffix('.npz'),**{k:v for k,v in payload.items() if v is not None})
            with np.load(path.with_suffix('.diagnostics.npz'),allow_pickle=False) as loaded:
                diagnostics = {k:loaded[k] for k in loaded.files}
            nearest = _retime_diagnostics(diagnostics,index,len(q))
            for side, values in sampled['sides'].items():
                for key in values:
                    name = side+'_'+key
                    if name in diagnostics and isinstance(values[key], np.ndarray):
                        diagnostics[name] = values[key]
            diagnostics['source_time_s'] = next_source
            np.savez_compressed(blob.with_suffix('.diagnostics.npz'),**diagnostics)
            next_meta = copy.deepcopy(meta)
            for side,old in next_meta['estimated_contact_segments'].items():
                new = []
                for segment in old:
                    ids = np.flatnonzero((nearest>=segment['start'])&(nearest<segment['end']))
                    if len(ids):new.append(dict(segment,start=int(ids[0]),end=int(ids[-1])+1))
                next_meta['estimated_contact_segments'][side] = new
            timing = next_meta.setdefault('retiming',{})
            saved = blob.with_suffix('.time_map.npz')
            np.savez_compressed(saved,output_time_s=np.arange(len(index))/fps,source_time_s=next_source,
                                source_frame_float=next_source*source_fps,source_fps=source_fps,
                                source_sha256=meta['input_sha256'],policy='final_joint_slowdown_v1',pre_final_index=index)
            timing.update(changed=True,time_map=saved.name,source_time_sha256=sha(saved),
                          final_adaptation=attempt,added_frames=int(timing.get('added_frames',0))+len(index)-len(q))
            next_meta.update(output_frames=len(index),output_sha256=sha(blob))
            atomic_json(blob.with_suffix('.json'),next_meta)
            try:
                if time.monotonic() >= deadline:
                    return defer()
                cache = {}
                checked = evaluate(blob,next_meta,solve,mode,cache=cache)
                if time.monotonic() >= deadline:
                    return defer()
                repair_path = work/path.name
                shutil.copyfile(blob.with_suffix('.diagnostics.npz'),repair_path.with_suffix('.diagnostics.npz'))
                model = CollisionModel(robot,dict(meta['configuration'].get('self_collision') or {},coverage='full'))
                data,checked,_,repairs = recover(robot,sampled,model,payload,blob,repair_path,
                    next_meta,checked,cache,solve,index,deadline,mode=mode,window_seconds=window_seconds)
                attempt['inserted_frame_repairs'] = repairs
                if time.monotonic() >= deadline:
                    return defer()
                if any(item['status']=='kept' for item in repairs):
                    next_meta['output_sha256'] = sha(blob)
                    fresh = evaluate(blob,next_meta,solve,mode)
                    if {k:v for k,v in fresh.items() if k!='audit_s'} != {k:v for k,v in checked.items() if k!='audit_s'}:
                        raise RuntimeError('Inserted-frame incremental and full audits disagree')
                    checked = fresh
                actual = dict(sampled,source_root_r=sampled['root_r'],root_r=Rotation.from_quat(data['root_rot']).as_matrix())
                geometry_reason = inserted_geometry_regression(robot,actual,model,original_rows,
                    np.c_[data['root_pos'],data['dof_pos']].astype(float),index)
            except (ValueError,RuntimeError) as exc:
                attempt.update(decision='candidate_audit_error',reason=str(exc))
                meta.setdefault('retiming',{})['final_adaptation'] = attempt
                atomic_json(path.with_suffix('.json'),meta)
                return meta,quality
            flags = set(checked['flags'])
            def rejection():
                if geometry_reason:
                    return geometry_reason
                if SPEED_FLAG in flags or flags-set(quality['flags']):
                    return 'remaining_speed_or_new_quality_flag'
                return None

            reason = rejection()
            if time.monotonic() >= deadline:
                return defer()
            if reason and fitted is not linear:
                # Preserving the original piecewise path is a useful second
                # candidate when new-sample IK moves a foot or loses clearance.
                attempt['ik_rejection'] = reason
                data = dict(payload,root_pos=linear[:,:3].astype(np.float32),dof_pos=linear[:,3:].astype(np.float32))
                blob.write_bytes(pickle.dumps(data,protocol=4))
                np.savez_compressed(blob.with_suffix('.npz'),**{k:v for k,v in data.items() if v is not None})
                next_meta['output_sha256'] = sha(blob)
                atomic_json(blob.with_suffix('.json'),next_meta)
                checked = evaluate(blob,next_meta,solve,mode)
                flags = set(checked['flags'])
                geometry_reason = inserted_geometry_regression(robot,actual,model,original_rows,
                    np.c_[data['root_pos'],data['dof_pos']].astype(float),index)
                fitted = linear
                reason = rejection()
            if time.monotonic() >= deadline:
                return defer()
            if reason:
                attempt.update(decision='rejected_by_full_audit',candidate_flags=checked['flags'],reason=reason)
            else:
                attempt.update(decision='adopted_after_full_audit',candidate_flags=checked['flags'],
                               selected='fixed_anchor_linear' if fitted is linear else 'fixed_anchor_ik')
                next_meta['retiming']['pre_final_checkpoint'] = 'pre_final_retime'
                next_meta['self_collision']['export_check'] = checked['self_collision']
                next_meta['solver_velocity_check'] = checked['step_velocity']
                if 'unified_repair' in next_meta:
                    next_meta['unified_repair'] = dict(next_meta['unified_repair'],
                        report_path='pre_final_retime/repair.json',status=checked['status'])
                atomic_json(blob.with_suffix('.json'),next_meta)
                shutil.copyfile(folder/'identity.json',candidate/'identity.json')
                if source_path.is_file():shutil.copyfile(source_path,candidate/source_path.name)
                original = work/'original'
                folder.rename(original)
                try:
                    candidate.rename(folder)
                    original.rename(folder/'pre_final_retime')
                except OSError:
                    if folder.exists():folder.rename(candidate)
                    original.rename(folder)
                    raise
                return next_meta,checked
    meta.setdefault('retiming',{})['final_adaptation'] = attempt
    atomic_json(path.with_suffix('.json'),meta)
    return meta,quality
