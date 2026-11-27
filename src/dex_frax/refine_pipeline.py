"""Frax export -> persistent measured-window repairs -> independent full audit.

No per-clip window cap. Each finite candidate family gets a fair visit. Safe
adoptions persist immediately; a user budget leaves a resumable pending queue.
"""
from .robot_profile import POSE_SIZE
import copy
import json
import os
import pickle
import shutil
import tempfile
import time
from pathlib import Path
import numpy as np
from .airborne import airborne_mask, protected_height
from .contract import (DEFAULT_REFINE_FLOW, DEFAULT_REFINE_SECONDS_PER_FRAME, LEGACY_CHECKPOINT_FLOW,
                       REFINE_FLOWS, UNLIMITED_REPAIR_BUDGET)
from osqp import OSQPException
from scipy.spatial.transform import Rotation
from .adaptive_refine import policy,pose_mask,window_targets,_stage
from .audit_cache import fork
from .cross_foot_repair import RECIPE as CROSS_FOOT
from .local_refine import fit_window,torso_errors,upper_ids
from .refine_dispatch_budget import FiniteDispatch
from .refine_queue import WorkQueue,jobs_from_masks,time_sliced
from .refine_routing import (LEGACY_SAFETY_ORDER, SAFETY_ORDER, WINDOW_POSE, alignment_reference,
                             checked_geometry_frames, clearance_trial_family, collision_groups, collision_masks,
                             core_frames, dependencies, group_jitter_jobs, movable_submodel,
                             coupled_leg_window, leg_phase_repair, next_window_job, ready_jobs, stalled_window, submodel, window_jobs)
from .unified_repair import quality_guard,dense_geometry,collision_seams,arm_window_pending
from .similarity import palm_envelope,score
from .temporal_constraints import root_speed_bounds
from .pelvis_contact import BODY_DRIFT_M, RECIPE as PELVIS_CONTACT
from .v8.self_collision import CollisionModel,scan,_repair_window

POLICY=dict(name='geometry_before_temporal_v8',core_frames=101,padding_frames=20,
            collision_tiers_deg=[10.,20.,30.],
            maximum_windows=None,whole_clip_repair_seconds=None,
            context_padding_frames=[20,50,100],neighbor_reactivations=None,final_fresh_audit=True,
            shared_safety_kernel=['support','dense_floor','leg_collision'],swept_geometry_steps=8,
            short_safety_core_frames=21,short_temporal_core_frames=11,jitter_guard='per_joint_and_union')
A_POLICY=dict(POLICY, name='specialists_then_residual_jitter_v1')
WINDOW_POLICY=dict(POLICY, name='alternating_support_phase_windows_v6')
COUPLED_LEGS='coupled_leg_clearance'
PHASE_LEGS=('phase_leg_clearance','phase_leg_rollback')
PLATFORM_PHASE=('platform_phase','platform_leg_only')
PELVIS_CONTACT_RESTORE='pelvis_contact_restore'
FOOT_GEOMETRY='foot_geometry_nonlinear'
SUPPORT_LEG_RECIPES={
    'ground_left_leg':('grounded_leg_windows','Left'),
    'ground_right_leg':('grounded_leg_windows','Right'),
    'swing_left_leg':('inferred_swing_leg_windows','Left'),
    'swing_right_leg':('inferred_swing_leg_windows','Right'),
}


def repair_limits():
    """Read the packaged per-attempt limits for shared-budget refine."""
    return json.loads(Path(__file__).with_name('refine_budget.json').read_text())


def normalized_input_frames(meta, fallback):
    count = meta.get('input_frames', fallback)
    fps = meta.get('input_fps', 50.)
    if abs(fps - 50.) <= 1e-5:
        return count
    return int(np.floor((count - 1) * 50. / fps + 1e-8)) + 1


def repair_workload(pending, flags, frames):
    """Size local modules by the largest reachable window and whole-motion passes by length."""
    from .temporal_repair import ANGULAR, JITTER
    from .swept_height import FLAG as SWEPT_FLOOR

    workload = {"motion": frames, "discontinuities": frames}
    for job in pending:
        if job["next_recipe"] is None and not job["time_sliced_recipes"]:
            continue
        module = job["module"]
        padding = 10 if module in {"dense_floor", "jitter", "support", "body", "leg_collision"} else 100
        span = min(frames, job["core_end"] + padding) - max(0, job["core_start"] - padding)
        workload[module] = max(workload.get(module, 0), span)
    if JITTER in flags:
        workload["jitter"] = frames
    if ANGULAR in flags:
        workload["root_angular"] = frames
    if SWEPT_FLOOR in flags:
        workload["dense_floor"] = frames
    return workload


def measurements(cache,robot):
    frames=cache['frames'];n=len(frames['body_heights']);floor=cache['floor'][::8]
    bodies=list(robot.surface_models);feet={g['foot'] for g in robot.sides.values()}
    contact=np.zeros(n)
    for values in frames['contact_xyz'].values():
        contact=np.maximum(contact,np.array([0 if v is None else np.linalg.norm(v)/100 for v in values]))
    return dict(contact=contact,floor=floor.min(axis=1),
        foot_floor=floor[:,[i for i,b in enumerate(bodies) if b in feet]].min(axis=1),
        body_floor=floor[:,[i for i,b in enumerate(bodies) if b not in feet]].min(axis=1),
        errors=cache['metrics']['errors'],torso=cache['metrics']['torso'],
        legs={side:np.column_stack([frames['errors'][side][key] for key in ('thigh_deg','shin_deg')])
              for side in robot.sides})


def masks(quality,cache,robot,groups=None,torso_reference=None,source_pose=False):
    if not quality['flags'] and not source_pose:return []
    m=measurements(cache,robot);n=len(m['torso']);result=[];cfg=policy()
    def add(stage,indices,side=None):
        mask=np.zeros(n,bool);mask[np.asarray(indices,int)]=True
        if mask.any():result.append((stage,side,mask))
    add('support',np.flatnonzero((m['contact']>cfg['geometry_acceptance_m'])|(m['foot_floor']<-.001)))
    add('body',(quality.get('body_support') or {}).get('violating_frames',[]))
    add('body',np.flatnonzero(m['body_floor']<-.001))
    dense=[]
    for i in (quality.get('interpolated_floor') or {}).get('violating_intervals',[]):dense.extend([i,min(n-1,i+1)])
    add('dense_floor',dense)
    torso_limit=20. if torso_reference is None else np.maximum(20.,np.asarray(torso_reference)+.02)
    add('torso',np.flatnonzero(m['torso']>torso_limit))
    if groups is None:
        count=len(cache['full_collision']['key'][1]);groups={'collision':list(range(count))}
    for stage,mask in collision_masks(cache['full_collision']['samples'],groups,n).items():
        add(stage,np.flatnonzero(mask))
    for joint,data in (quality.get('temporal_quality') or {}).get('joints',{}).items():
        add('jitter',data.get('violating_frames',data.get('added_reversal_frames',[])),joint)
    for side,errors in m['errors'].items():
        mask=pose_mask({side:errors},50.,cfg)
        for key,values in errors.items():
            limit=20. if key in ('upper','lower') else 25.
            if key!='elbow_axis' and np.percentile(values,95)>limit:mask|=values>limit
        add('arms',np.flatnonzero(mask),side)
        if source_pose:
            from .adaptive_refine import sustained
            deviation=np.max(m['legs'][side],axis=1)
            add('leg_pose',np.flatnonzero(sustained(deviation>45.,max(2,round(cfg['pose_duration_s']*50.)))),side)
    # Merge same-stage masks before generating stable grid windows.
    merged={}
    for stage,side,mask in result:merged[(stage,side)]=merged.get((stage,side),np.zeros(n,bool))|mask
    return [(stage,side,mask) for (stage,side),mask in merged.items()]


def guard(robot,t,before,after,original,old_m,new_m,old_q,new_q,old_cache,new_cache,model,job,*,new_targets=None):
    a,b=job['start'],job['end'];stage=job['module'];n=len(before)
    joint=job.get('hand_leg_route')
    pelvis=new_targets is not None
    if after.shape!=before.shape or not np.isfinite(after).all():return 'shape_or_nonfinite'
    if not np.array_equal(before[:a],after[:a]) or not np.array_equal(before[b:],after[b:]):return 'outside_window_changed'
    lo=max(0,a-1);hi=min(n,b+1)
    tolerance=np.deg2rad(json.loads(Path(__file__).with_name('quality_pose.json').read_text())['joint_limit_tolerance_deg'])
    if np.any(after[a:b,3:]<robot.lower[3:]-tolerance) or np.any(after[a:b,3:]>robot.upper[3:]+tolerance):return 'effective_joint_limits'
    speed_limit=robot.speed_limits+1e-4
    if pelvis or job.get('source_pose_guard'):
        speed_limit=np.maximum(speed_limit,abs(np.diff(before[lo:hi,3:],axis=0))*t['fps']+1e-6)
    if np.any(abs(np.diff(after[lo:hi,3:],axis=0))*t['fps']>speed_limit):return 'speed_or_seam'
    knees=[g['leg'][3] for g in robot.sides.values()]
    if np.any((before[a:b,knees]>=0)&(after[a:b,knees]<-1e-7)):return 'new_knee_hyperextension'
    if joint:
        from .hand_leg_repair import columns,owned_pairs
        allowed,_,_=columns(model,job['hand_leg_pair_ids'],joint=='cooperate')
        fixed=np.setdiff1d(np.arange(POSE_SIZE),allowed)
        if not np.array_equal(before[:,fixed],after[:,fixed]):return 'hand_leg_changed_protected_joint'
        arm_ids=sum([g['arm'][:3] for g in robot.sides.values()],[])
        if np.any(abs(after[a:b,arm_ids]-original[a:b,arm_ids])>np.deg2rad(30.)+2e-6):return 'fixed_reference_shoulder_cap'
    elif job.get('surface_support_pose'):
        from .contact_phase import support_errors
        g=robot.sides[job['side']]
        fixed=np.setdiff1d(np.arange(POSE_SIZE),[0,1,2]+sum([s['leg'] for s in robot.sides.values()],[])+g['arm'])
        if not np.array_equal(before[:,fixed],after[:,fixed]):return 'protected_body_changed'
        if np.any(abs(after[a:b,:2]-before[a:b,:2])>.065001):return 'root_corridor'
        prior,limit=support_errors(robot,t,before[a:b],a)
        value,_=support_errors(robot,t,after[a:b],a)
        if np.any(value>np.maximum(prior,limit)+2e-6):return 'support_component_regression'
        old_speed=np.linalg.norm(np.diff(before[lo:hi,:3],axis=0),axis=1)*t['fps']
        new_speed=np.linalg.norm(np.diff(after[lo:hi,:3],axis=0),axis=1)*t['fps']
        if np.any(new_speed>np.maximum(root_speed_bounds(t['root_p'][lo:hi],t['fps']),old_speed)+1e-5):return 'root_speed'
    elif stage in ('arms','collision'):
        sides=[job['side']] if job['side'] else list(robot.sides)
        fixed=np.setdiff1d(np.arange(POSE_SIZE),sum([robot.sides[s]['arm'] for s in sides],[]))
        if not np.array_equal(before[:,fixed],after[:,fixed]):return 'protected_body_changed'
    if not pelvis and not job.get('surface_support_pose') and not np.array_equal(before[:,:2],after[:,:2]):return 'pelvis_horizontal_changed'
    if stage=='leg_pose':
        fixed=np.setdiff1d(np.arange(POSE_SIZE),robot.sides[job['side']]['leg'])
        if not np.array_equal(before[:,fixed],after[:,fixed]):return 'leg_pose_changed_protected_joint'
    if job.get('source_pose_guard'):
        for side in robot.sides:
            if stage=='leg_pose' and side==job['side']:continue
            previous=old_m['legs'][side][a:b]
            updated=new_m['legs'][side][a:b]
            if (updated>np.maximum(previous,25.)+3.).any():return 'source_leg_direction_regression'
    if stage=='leg_collision' and not joint:
        fixed=np.setdiff1d(np.arange(POSE_SIZE),[2]+sum([g['leg'] for g in robot.sides.values()],[]))
        if not np.array_equal(before[:,fixed],after[:,fixed]):return 'leg_repair_changed_upper_body'
    if stage in ('collision','torso'):
        for g in robot.sides.values():
            if stage=='collision' and np.any(abs(after[a:b,g['arm'][:3]]-original[a:b,g['arm'][:3]])>np.deg2rad(30.)+2e-6):return 'fixed_reference_shoulder_cap'
    if pelvis:
        from .pelvis_contact import check_motion
        reason=check_motion(t,new_targets,before,after,old_m,new_m,old_cache,new_cache,old_q,new_q,a,b)
        if reason:return reason
    elif np.any(new_m['torso'][a:b]>old_m['torso'][a:b]+.02):return 'torso_frame_regression'
    if joint:
        # Source-relative envelope, not a ratchet against each incidental gain.
        from .palm_constraints import SourcePalms
        palms=SourcePalms(robot,t,original,a,b,list(robot.sides),np.arange(POSE_SIZE))
        reason='palm_absolute_envelope' if any(palms.evaluate(after[i],i-a)[2]>np.deg2rad(.001) for i in range(a,b)) else None
    else:
        reason=palm_envelope({s:{k:v[a:b] for k,v in e.items()} for s,e in old_m['errors'].items()},
                             {s:{k:v[a:b] for k,v in e.items()} for s,e in new_m['errors'].items()})
    if reason:return reason
    if np.any(new_m['contact'][a:b]>np.maximum(.001,old_m['contact'][a:b])+1e-7):return 'foot_anchor_regression'
    local=window_targets(t,lo,hi,n)
    from .audit_cache import clear_dense_window
    certified_clear=(stage not in ('collision','leg_collision') and not joint and
        clear_dense_window(old_cache,lo,hi,n) and clear_dense_window(new_cache,lo,hi,n))
    if certified_clear:
        old_f=np.asarray(old_cache['floor'])[lo*8:(hi-1)*8+1]
        new_f=np.asarray(new_cache['floor'])[lo*8:(hi-1)*8+1]
    else:
        changed=window_targets(new_targets,lo,hi,n) if pelvis else local
        old_d,old_f=dense_geometry(robot,model,local,before[lo:hi]);new_d,new_f=dense_geometry(robot,model,changed,after[lo:hi])
        if np.any(new_d<np.minimum(old_d,0.)-2e-6):return 'new_or_deeper_collision_sample'
    if stage in ('collision','leg_collision'):
        owned=owned_pairs(model,job['hand_leg_pair_ids'],joint=='cooperate') if joint else collision_groups(model)[stage]
        old_m['owned_collision_deficit']=float(np.maximum(-old_d[:,owned],0.).sum())
        new_m['owned_collision_deficit']=float(np.maximum(-new_d[:,owned],0.).sum())
    if stage in ('leg_collision','jitter') or joint or job.get('restore_support_deficit') or job.get('grounded_leg_repair'):
        from .protected_collision_route import slide_guard
        slip = slide_guard(t,old_cache,new_cache,lo,hi)
        if slip:return slip
    if stage=='leg_collision' or joint:
        from .inserted_frame_repair import impulse_regression
        if impulse_regression(old_q,new_q):return 'root_impulse_regression'
        from .leg_similarity import tradeoff
        def leg_angles(c):return np.column_stack([c['frames']['errors'][s][k][a:b] for s in ('Left','Right') for k in ('thigh_deg','shin_deg')])
        result=tradeoff(leg_angles(old_cache),leg_angles(new_cache),old_m['owned_collision_deficit'],new_m['owned_collision_deficit'])
        new_m['leg_shape_tradeoff']=result
        if result['dominated']:return 'leg_shape_regression_without_collision_gain'
    if np.any(new_f<np.minimum(old_f,-.001)-2e-6):return 'dense_floor_regression'
    for name,values in old_cache['frames']['body_errors'].items():
        for key,old in values.items():
            for i in range(a,b):
                contact=next((c for c in t.get('body_contacts',[]) if c.get('body_name')==name),None)
                from .support_quality import preserve_contact
                if contact is not None and not preserve_contact(contact,i):continue
                new=new_cache['frames']['body_errors'][name][key][i]
                # All inferred contacts keep existing treatment; trusted supports
                # also get independent final gates. Let existing errors improve.
                if old[i] is None or new is None:continue
                if pelvis:
                    tolerance_key='body_contact_height_tolerance_m' if key=='normal_cm' else 'body_contact_anchor_tolerance_m'
                    tolerance_cm=t[tolerance_key]*100.
                    limit=min(tolerance_cm,old[i]+BODY_DRIFT_M*100.) if old[i]<=tolerance_cm else old[i]
                    if new>limit+1e-5:return 'body_support_regression'
                elif new>max(old[i],.1)+.05:return 'body_support_regression'
    before_j=set((old_q.get('temporal_quality') or {}).get('violating_frames',[]))
    after_j=set((new_q.get('temporal_quality') or {}).get('violating_frames',[]))
    if after_j-before_j:return 'new_jitter_frames'
    from .temporal_constraints import regressions
    added_joints=regressions(old_q.get('temporal_quality') or {},new_q.get('temporal_quality') or {})
    if added_joints:return 'new_jitter_joint:'+sorted(added_joints)[0]
    if pelvis:return quality_guard(old_q,new_q,orientation_budget_deg=5.)
    return quality_guard(old_q,new_q,True)


def improved(stage,side,a,b,old,new,old_m,new_m,*,jitter_joints=()):
    if old['flags'] and not new['flags']:return True
    if stage in ('collision','leg_collision'):
        x=old['full_self_collision'];y=new['full_self_collision']
        return (y['collision_samples']<x['collision_samples'] or y['minimum_distance_m']>x['minimum_distance_m']+1e-6
            or new_m.get('owned_collision_deficit',float('inf'))<old_m.get('owned_collision_deficit',0.)-1e-7)
    if stage=='torso':return np.sum(np.maximum(new_m['torso'][a:b]-20.,0.))<np.sum(np.maximum(old_m['torso'][a:b]-20.,0.))-.01
    if stage=='arms':return score(new_m['errors'][side])<score(old_m['errors'][side])-1e-5
    if stage=='leg_pose':
        def leg_loss(m):return np.sum(np.maximum(m['legs'][side][a:b]-25.,0.)**2)
        return leg_loss(new_m)<leg_loss(old_m)-.01
    if stage=='jitter':
        before=old['temporal_quality'];after=new['temporal_quality']
        joints=jitter_joints or (side,)
        def count(report):
            return sum(len(report.get('joints',{}).get(name,{}).get('violating_frames',[])) for name in joints)
        return (len(after['violating_frames'])<len(before['violating_frames']) or
                (count(after)<count(before) and len(after['violating_frames'])<=len(before['violating_frames'])))
    if stage=='body':
        before,after=old['body_support'],new['body_support']
        if len(after['violating_frames'])<len(before['violating_frames']):return True
        if len(after['violating_frames'])==len(before['violating_frames']):
            def excess(report):
                return sum(max(0.,(part.get(metric) or 0.)-report[limit])
                           for part in report['contacts'].values() for metric,limit in
                           (('max_trusted_normal_m','normal_tolerance_m'),('max_trusted_fixed_xy_m','fixed_xy_tolerance_m')))
            if excess(after)<excess(before)-1e-5:return True
    if stage=='dense_floor':return new['interpolated_floor']['samples_below_minus_1mm']<old['interpolated_floor']['samples_below_minus_1mm'] or new['interpolated_floor']['minimum_body_visual_cm']>old['interpolated_floor']['minimum_body_visual_cm']+1e-5
    return (np.maximum(new_m['contact'][a:b]-.001,0.).sum()<np.maximum(old_m['contact'][a:b]-.001,0.).sum()-1e-7
        or np.maximum(-new_m['floor'][a:b]-.001,0.).sum()<np.maximum(-old_m['floor'][a:b]-.001,0.).sum()-1e-7)


def phase_leg_fit(robot,t,out,weights,model,job,recipe,deadline,progress):
    from .adaptive_refine import MinimumChangeObjective
    a,b=job['phase_leg_window']
    legs=[column for side in robot.sides.values() for column in side['leg']]
    flight=job['leg_phase']=='flight'
    ids=legs if flight else [2,*legs]
    protected=protected_height(t,len(out))
    if flight:
        protected=protected & ~t['reconstructible_flight']
    guide=out
    if recipe=='phase_leg_rollback':
        guide=out.copy()
        grid=np.arange(a,b)
        core_start=max(a+2,job['core_start'])
        core_end=min(b-3,job['core_end']-1)
        core_end=max(core_start,core_end)
        entering=np.maximum(0.,(grid-a-1)/max(1,core_start-a-1))
        leaving=np.maximum(0.,(b-2-grid)/max(1,b-2-core_end))
        weight=np.minimum(1.,np.minimum(entering,leaving))
        weight=np.sin(weight*np.pi/2.)**2
        weight[protected[a:b]]=0.
        window=np.ix_(grid,ids)
        guide[window]+=weight[:,None]*(t['raw_phase_reference'][window]-out[window])
    options=dict(weights,_pelvis_height_only=True,_coupled_swept_support=True,_coupled_leg_clearance=True,
                 _preserve_support_step=True,_phase_elastic_search=not flight,
                 _initial_leg_collision_cfg=model.cfg,_repair_deadline=deadline-2.5,
                 _local_fit_progress=progress)
    fitted,detail=fit_window(robot,t,out,options,a,b,15,ids_override=ids,local_checks=True,
                            allow_support=True,seed=None if guide is out else guide,
                            objective_factory=lambda i:MinimumChangeObjective(guide[i]),
                            protected_frames=protected[a:b],flight_legs_only=flight)
    detail.update(policy=recipe,phase=job['leg_phase'],window=[a,b],movable_joint_columns=ids)
    return fitted,detail


def platform_phase_fit(robot,t,out,weights,model,job,recipe,deadline,progress):
    from .adaptive_refine import MinimumChangeObjective
    a,b=job['platform_window']
    indices=[foot['platform_index'] for foot in t['sides'].values()]
    measured=t['inferred_platform_start']
    inferred=np.logical_or.reduce([index>=measured for index in indices])
    measured_contact=np.logical_or.reduce([(index>=0)&(index<measured) for index in indices])
    mutable=inferred & ~measured_contact & ~airborne_mask(t,len(out)) & ~np.asarray(t.get('source_support_uncertain',False),bool)
    scene=np.asarray(t.get('source_scene_support',False),bool)
    targets=dict(t,source_scene_support=scene & ~mutable,_inferred_platform_repair=measured)
    protected=protected_height(targets,len(out)) | measured_contact
    ids=[2]+[column for side in robot.sides.values() for column in side['leg']]
    if recipe=='platform_leg_only':
        ids=robot.sides[job['side']]['leg']
        protected |= t['sides'][job['side']]['contact']>0
    options=dict(weights,_pelvis_height_only=True,_coupled_swept_support=True,
                 _preserve_support_step=True,_temporal_repair_gain=1.3,_phase_elastic_search=True,
                 _initial_leg_collision_cfg=model.cfg,_repair_deadline=deadline-2.5,
                 _local_fit_progress=progress)
    fitted,detail=fit_window(robot,targets,out,options,a,b,9,ids_override=ids,local_checks=True,
                            allow_support=True,objective_factory=lambda i:MinimumChangeObjective(out[i]),
                            protected_frames=protected[a:b])
    detail.update(policy=recipe,window=[a,b],movable_joint_columns=list(ids),
                  _platform_protected_frames=protected)
    return fitted,detail


def propose(robot,t,out,original,weights,model,job,recipe,deadline,progress=None,unfixed_waist=None):
    a,b=job['start'],job['end'];stage=job['module'];side=job['side'];n=len(out)
    local=window_targets(t,a,b,n);cfg=dict(weights,_repair_deadline=deadline,_pelvis_height_only=True);candidate=out.copy();detail={}
    progress={} if progress is None else progress
    if recipe in PHASE_LEGS:
        return phase_leg_fit(robot,t,out,weights,model,job,recipe,deadline,progress)
    if recipe in PLATFORM_PHASE:
        return platform_phase_fit(robot,t,out,weights,model,job,recipe,deadline,progress)
    if recipe == CROSS_FOOT:
        from .cross_foot_repair import ContactWindow
        return ContactWindow(robot,t,out,model,job,deadline).run(weights)
    if recipe in (PELVIS_CONTACT,PELVIS_CONTACT_RESTORE,FOOT_GEOMETRY):
        from .pelvis_contact import propose as settle
        rows,rotations,detail=settle(robot,t,out,job,deadline,
                                    restore_violations=recipe==PELVIS_CONTACT_RESTORE,
                                    foot_geometry=recipe==FOOT_GEOMETRY)
        if recipe==FOOT_GEOMETRY:
            return rows,detail
        return rows,dict(detail,_pelvis_rotations=rotations)
    if recipe=='coupled_body_support':
        from .contact_phase import body_proposal
        return body_proposal(robot,t,out,job,deadline,dict(cfg,_initial_leg_collision_cfg=model.cfg),progress)
    if recipe=='surface_support_pose':
        from .contact_phase import arm_proposal
        return arm_proposal(robot,t,out,job,deadline)
    if recipe.startswith('hand_leg'):
        from .hand_leg_repair import propose as joint_proposal
        return joint_proposal(model,t,out,original,cfg,job,recipe,deadline,progress)
    if recipe==COUPLED_LEGS:
        from .adaptive_refine import MinimumChangeObjective
        lo,hi=job['coupled_leg_window']
        ids=[2]+[column for side in robot.sides.values() for column in side['leg']]
        cfg.update(_coupled_swept_support=True,_coupled_leg_clearance=True,_preserve_support_step=True,
                   _initial_leg_collision_cfg=model.cfg,_local_fit_progress=progress,
                   _repair_deadline=deadline-2.5)
        rows,detail=fit_window(robot,t,out,cfg,lo,hi,5,ids_override=ids,local_checks=True,allow_support=True,
                               objective_factory=lambda i: MinimumChangeObjective(out[i]),
                               protected_frames=protected_height(t,n)[lo:hi])
        detail.update(policy=COUPLED_LEGS,movable_joint_columns=ids,window=[lo,hi])
        return rows,detail
    if recipe in ('hip_roll_only','legs_preserve_support'):
        from .protected_collision import propose as protected_proposal
        ids = ([3+robot.names.index(job['hip_roll_joint'])] if recipe=='hip_roll_only'
               else [2]+[i for g in robot.sides.values() for i in g['leg']])
        rows,detail = protected_proposal(robot,t,out,model,a,b,deadline,leg_columns=ids)
        return rows,dict(detail,movable_joint_columns=ids)
    if stage in ('support','dense_floor','leg_collision'):
        from .adaptive_refine import MinimumChangeObjective
        digits=''.join(c for c in recipe if c.isdigit())
        tier=float(digits) if digits else (10. if recipe.endswith('more') else 5.)
        cfg.update(_restore_geometry=True,_safety_joint_deg=tier,_safety_reference=original,_local_fit_progress=progress)
        support_leg = SUPPORT_LEG_RECIPES.get(recipe)
        residual_only = support_leg is not None
        if residual_only:
            cfg['_restore_violations_only'] = True
        ids=np.asarray([2]+sum([g['leg'] for g in robot.sides.values()],[]))
        from .local_geometry import project_root_height
        protected = protected_height(t,n)[a:b]
        if support_leg is not None:
            window_key,moving_side=support_leg
            ids=np.asarray(robot.sides[moving_side]['leg'])
            protected = protected & ~t[window_key][moving_side][a:b]
            cfg['_preserve_support_step'] = True
        vertical=None if residual_only else project_root_height(robot,t,out,a,b,protected_frames=protected)
        seed=vertical[0] if vertical is not None else None
        candidate,detail=fit_window(robot,t,out,cfg,a,b,24,ids_override=ids,seed=seed,
            local_checks=True,allow_support=True,objective_factory=lambda i:MinimumChangeObjective(out[i]),
            protected_frames=protected)
        detail.update(policy='shared_native_swept_support_geometry_v1',tier_deg=tier,
                      movable_joint_columns=ids.tolist())
        if support_leg is not None:
            detail.update(support_evidence=window_key,moving_side=moving_side)
        return candidate,detail
    progress={} if progress is None else progress
    cfg['_branch_progress']=progress.setdefault('branches',{})
    cfg['_support_progress']=progress.setdefault('support',{})
    from .torso_branch import TimedObjective
    from .v8.solver import Objective
    def fit(seed=None,ids=None,iterations=6):
        protected=protected_height(t,n)[a:b] if stage=='arms' else None
        return fit_window(robot,t,out,cfg,a,b,iterations,ids_override=ids,seed=seed,local_checks=True,
            objective_factory=lambda i:TimedObjective(Objective(robot,t,i,cfg),deadline),
            protected_frames=protected)
    if stage=='collision':
        from .arm_policy import collision_options
        tier=float(''.join(c for c in recipe if c.isdigit()))
        options=dict(collision_options(tier,6),_reference=original,_palm_reference=out,_progress=progress,**collision_seams(model,t,out,a,b))
        options['internal_overlap']=not(options['freeze_start'] and options['freeze_end'])
        if recipe.startswith('sweep'):
            from .trajectory_escape import propose as escape
            candidate,detail=escape(model,t,out,cfg,a,b,options,deadline)
        elif recipe.startswith('continuous'):
            from .trajectory_recovery import seed as continuous_seed
            from .palm_constraints import SourcePalms
            axis=recipe.split('_')[1];offset=dict(zero=(0.,0.,0.),xpos=(1.,0.,0.),xneg=(-1.,0.,0.),ypos=(0.,1.,0.),yneg=(0.,-1.,0.),zpos=(0.,0.,1.),zneg=(0.,0.,-1.))[axis]
            seed,seed_detail=continuous_seed(model,t,out,cfg,a,b,dict(options,_initial_shoulder_offset=offset),deadline)
            candidate,validation=_repair_window(model,t,seed,cfg,a,b,list(robot.sides),dict(options,_elastic_recovery=True,_project_seed=True,_source_palm_constraints=True,iterations=64,watchdog_seconds=max(.01,deadline-time.monotonic())))
            detail=dict(continuous_seed=seed_detail,validation=validation,offset=offset)
        elif recipe.startswith('direct'):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('collision_candidate_deadline')
            options['watchdog_seconds'] = remaining
            candidate,detail=_repair_window(model,t,out,cfg,a,b,list(robot.sides),options)
        else:
            from .shoulder_recovery import propose as recover
            candidate,detail=recover(model,t,out,cfg,a,b,options,deadline)
        detail.update(tier_deg=tier,**{k:options[k] for k in ('freeze_start','freeze_end')})
    elif stage=='leg_pose':
        candidate,detail=fit_window(robot,t,out,cfg,a,b,6,ids_override=robot.sides[side]['leg'],
                                    local_checks=True,allow_support=True,protected_frames=protected_height(t,n)[a:b])
    elif stage=='arms':
        protected = protected_height(t,n)[a:b]
        if recipe=='fit':
            from .experimental_arm_refine import refine
            candidate[a:b],detail=refine(robot,local,out[a:b],cfg,model.cfg,side,max(.01,deadline-time.monotonic()),6,
                freeze_start=a>0,freeze_end=b<n,soft_similarity=True,movable_collision_only=True,
                protected_frames=protected)
        else:
            from .unified_repair import propose_arm_branch
            seed=out.copy();ids=robot.sides[side]['arm']
            route={'guided_current':'current_best','guided_original':'original_frax'}.get(recipe,recipe)
            result,detail=propose_arm_branch(robot,local,out[a:b],original[a:b],cfg,side,route)
            if recipe.startswith('guided_'):
                from .experimental_arm_refine import refine
                candidate[a:b],join=refine(robot,local,out[a:b],cfg,model.cfg,side,max(.01,deadline-time.monotonic()),12,
                    freeze_start=a>0,freeze_end=b<n,soft_similarity=True,movable_collision_only=True,branch_hint=result,
                    protected_frames=protected)
            else:
                seed[a:b,ids]=result[:,ids];candidate,join=fit(seed,ids)
            detail=dict(branch=detail,join=join,guided_from_safe_baseline=recipe.startswith('guided_'))
    elif stage=='torso':
        if recipe=='fit':candidate,detail=fit()
        else:
            from .torso_branch import source_waist
            from .v8.arms import plan_and_refine
            guided=source_waist(robot,t,out,a,b,deadline)
            seed=guided.copy();branch={'executed':False}
            if recipe=='source_branch_collision' and 'aligned' not in progress:seed[a:b],branch=plan_and_refine(robot,local,guided[a:b],cfg)
            arm_ids=np.asarray(sum([g['arm'] for g in robot.sides.values()],[]))
            # Once waist FK matches the source, join only the arms; optimizing
            # waist again can re-enter the very branch this candidate fixes.
            if 'aligned' in progress:
                candidate=out.copy();candidate[a:b]=np.asarray(progress['aligned']);detail={'resumed_alignment':True}
            else:
                candidate,detail=fit_window(robot,t,guided,cfg,a,b,6,ids_override=arm_ids,seed=seed,local_checks=True,
                    objective_factory=lambda i:TimedObjective(Objective(robot,t,i,cfg),deadline))
                progress['aligned']=candidate[a:b].tolist()
            if unfixed_waist is None:unfixed_waist=torso_errors(robot,t,out)>20.
            aligned=alignment_reference(original,candidate,unfixed_waist,a,b)
            detail=dict(fit=detail,branch=branch,_aligned_reference=aligned[a:b].tolist())
            if recipe.endswith('_collision'):
                from .arm_policy import collision_options
                from .shoulder_recovery import propose as recover
                detail['collision_tiers']=[]
                for tier in (10.,20.,30.):
                    if not scan(model,candidate[a:b],local['root_r'],8)[0]['collision_samples']:break
                    if time.monotonic()>=deadline:raise TimeoutError('candidate_time_slice')
                    options=dict(collision_options(tier,6),_reference=aligned,_palm_reference=out,_progress=progress.setdefault('tier'+str(tier),{}),**collision_seams(model,t,candidate,a,b))
                    try:candidate,recovery=recover(model,t,candidate,cfg,a,b,options,deadline)
                    except (ValueError,RuntimeError) as exc:
                        if time_sliced(exc):raise
                        detail['collision_tiers'].append(dict(degrees=tier,reason=str(exc)));continue
                    detail['collision_tiers'].append(dict(degrees=tier,detail=recovery))
                # The provisional torso pose was never written. Only the whole
                # transaction can pass the external guard.
    elif stage=='jitter':
        from .adaptive_refine import MinimumChangeObjective
        cfg['joint_acceleration']=float(cfg.get('joint_acceleration',.18))*(2 if recipe=='temporal' else 4)
        cfg['_temporal_repair_gain']=2. if recipe=='temporal' else 4.
        names=job.get('jitter_joints') or [side]
        if any(name not in robot.names for name in names):raise ValueError('Jitter task requires its measured joints')
        columns=[3+robot.names.index(name) for name in names]
        j=columns[0]
        # Correct the measured body region; leg/waist jitter cannot be routed
        # into a shoulder-only solver. Supports remain hard local constraints.
        ids=next((np.asarray(g[k]) for g in robot.sides.values() for k in ('leg','arm') if j in g[k]),np.array([j]))
        leg_jitter = any(j in g['leg'] for g in robot.sides.values())
        if leg_jitter:ids=np.asarray([2]+sum([g['leg'] for g in robot.sides.values()],[]))
        guide=out
        if recipe=='temporal_more':
            from scipy.interpolate import CubicHermiteSpline
            guide=out.copy()
            lo,hi=a+1,b-2
            slopes=(out[lo,columns]-out[lo-1,columns],out[hi+1,columns]-out[hi,columns])
            guide[lo+1:hi,columns]=CubicHermiteSpline([lo,hi],out[np.ix_([lo,hi],columns)],slopes)(np.arange(lo+1,hi))
            cfg.update(_coupled_swept_support=True,_initial_leg_collision_cfg=model.cfg,_temporal_repair_gain=2.,
                       _preserve_support_step=leg_jitter)
        elif leg_jitter:
            cfg['_preserve_support_step']=True
        fixed=protected_height(t,n)[a:b]
        candidate,detail=fit_window(robot,t,out,cfg,a,b,6 if recipe=='temporal' else 12,ids_override=ids,
            local_checks=True,allow_support=True,objective_factory=lambda i:TimedObjective(MinimumChangeObjective(guide[i]),deadline),
            protected_frames=fixed if recipe=='temporal' else None)
    else:
        cfg['max_iterations']=30 if recipe=='solve' else 60
        cfg['_repair_trusted_support']=stage=='body'
        candidate[a:b],detail=_stage(robot,local,out[a:b].copy(),cfg,stage)
    if stage!='collision':
        if a:candidate[a:a+2]=out[a:a+2]
        if b<n:candidate[b-2:b]=out[b-2:b]
    return candidate,detail


def enrich_jobs(pending, t, robot, cache, model, *, default_a=False):
    """Select the applicable contact, arm and collision recipes for the audited trajectory."""
    samples = cache['full_collision']['samples']
    for job in pending:
        if job['module']=='leg_collision':
            a,b=job['core_start'],job['core_end']
            pairs={tuple(model.names[k]) for _,_,ids in samples[max(0,8*a-7):8*b] for k in ids}
            hips = {side for side in ('l','r') for pair in pairs
                    if 'pelvis_collision' in pair and any(f'hip_{axis}_{side}_link_collision' in pair for axis in ('roll','yaw'))}
            job['candidate_order']=['legs_preserve_support','legs5','legs10']
            if len(hips)==1 and all('pelvis_collision' in pair for pair in pairs):
                job['hip_roll_joint']='hip_roll_'+next(iter(hips))+'_joint'
                job['candidate_order'].insert(0,'hip_roll_only')
        if job['module']=='support':
            # Existing coupled geometry kernel handles support and collision
            # together; avoid a support-only proposal that immediately
            # fails the common geometry guard. No new candidate family.
            job['candidate_order']=['geometry5_elastic','geometry10_elastic']
        if default_a and job['module'] in ('support','dense_floor','leg_collision'):
            from .refine_queue import RECIPES
            order = job.get('candidate_order',RECIPES[job['module']])
            a,b=job['core_start'],job['core_end']
            key='inferred_swing_leg_windows' if job['module']=='leg_collision' else 'grounded_leg_windows'
            windows=t.get(key,{})
            ground=[recipe for recipe,(basis,side) in SUPPORT_LEG_RECIPES.items()
                    if basis==key and side in windows and windows[side][a:b].any()]
            foot_geometry=[FOOT_GEOMETRY] if job['module'] in ('support','dense_floor') else []
            job['candidate_order']=[*ground,*foot_geometry,*order]
        if job['module']=='body':
            from .contact_phase import has_joint_support
            if has_joint_support(t,job['core_start'],job['core_end']):
                job['joint_support']=True
                if default_a:
                    job['restore_support_deficit']=True
                    job['candidate_order']=['coupled_body_support',PELVIS_CONTACT_RESTORE,PELVIS_CONTACT,'solve','solve_more']
                else:
                    job['candidate_order']=[PELVIS_CONTACT,'coupled_body_support','solve','solve_more']
        if job['module']=='arms':
            errors=cache['metrics']['errors'][job['side']];a,b=job['core_start'],job['core_end']
            # Large direction errors deserve a branch trial before repeated
            # small local steps. This orders search; it changes no gate.
            if max(np.percentile(errors[k][a:b],95) for k in ('upper','lower'))>2*policy()['arm_direction_deg']:
                job['candidate_order']=['current_best','guided_current','original_frax','guided_original','fit']
            from .contact_phase import supported_arm_span
            if len(supported_arm_span(robot,t,job['side'],a,b)):
                order=job.get('candidate_order',['fit','current_best','guided_current','original_frax','guided_original'])
                job['candidate_order']=['surface_support_pose',*order]
    from .hand_leg_repair import attach
    return attach(pending,samples,model)


def repair_output(robot,targets,xs,meta,payload,path,solve,repair_budget=None,continue_from=None,resume_recheck=False,repair_modules=None,plan_only=False,v8_candidates=False,segment_cascade=False,resume_same_round=False,*,refine_flow=DEFAULT_REFINE_FLOW,allow_contact_retime=False):
    from .acceptance import evaluate
    from .cli import atomic_json,sha
    from . import contact_timing
    from .refine_checkpoint import artifact, write as write_checkpoint
    from .resource_events import phase
    if refine_flow not in REFINE_FLOWS:
        raise ValueError(f'Unknown refine flow: {refine_flow}')
    legacy_order = refine_flow == 'A'
    window_order = refine_flow == 'C'
    repair_policy = WINDOW_POLICY if window_order else A_POLICY if legacy_order else POLICY
    safety_order = LEGACY_SAFETY_ORDER if legacy_order else SAFETY_ORDER
    if legacy_order:
        v8_candidates = segment_cascade = False
    phase('local_repair',compile_possible=False)
    def growth_checkpoint():
        if 'DEX_SCHEDULE_MEMORY_BYTES' in os.environ:
            phase('local_repair')
    path=Path(path);started=time.monotonic();cache={}
    resume_state=None
    saved_bytes=None
    saved=payload
    if continue_from is not None and (Path(continue_from)/'refine_state.json').exists():
        previous=Path(continue_from)
        resume_state=json.loads((previous/'refine_state.json').read_text())
        if resume_state.get('refine_flow', LEGACY_CHECKPOINT_FLOW) != refine_flow:
            raise ValueError('Continuation refine flow changed')
        if resume_recheck:
            resume_state['states']={}
            atomic_json(path.parent/'recheck_provenance.json',dict(previous=str(previous),previous_identity_sha256=sha(previous/'identity.json'),previous_motion_sha256=sha(previous/'motion.pkl'),policy='preserve motion and fixed reference; fresh audit and queue'))
        saved_bytes=artifact(previous,resume_state,'motion.pkl').read_bytes()
        saved=pickle.loads(saved_bytes)
    head_clipped=robot.clip_head_angles(saved['dof_pos'])
    if head_clipped and resume_state is not None:
        resume_state['states']={}
        resume_recheck=True
    if resume_state is not None or head_clipped:
        staged=path.with_suffix('.pkl.tmp')
        staged.write_bytes(pickle.dumps(saved,protocol=4) if head_clipped else saved_bytes)
        staged.replace(path)
        np.savez_compressed(path.with_suffix('.npz'),**{k:v for k,v in saved.items() if v is not None})
        meta['output_sha256']=sha(path);atomic_json(path.with_suffix('.json'),meta)
    quality=evaluate(path,meta,solve,'frax',cache=cache);before=copy.deepcopy(quality)
    # Initial IK and the independent baseline audit have already completed.
    separate_timing = meta.get('profile') == 'refine'
    total_frames_50hz = normalized_input_frames(meta, len(payload['dof_pos']))
    budget_source = 'explicit'
    if repair_budget is None:
        if separate_timing:
            repair_budget = DEFAULT_REFINE_SECONDS_PER_FRAME * total_frames_50hz
            budget_source = 'default_middle_refine_50hz'
        elif meta.get('profile') == 'fast':
            repair_budget = .5 * total_frames_50hz
            budget_source = 'default_full_input_50hz'
        else:
            budget_source = 'default_unlimited'
    if repair_budget == UNLIMITED_REPAIR_BUDGET:
        repair_budget = None
    budget_scope = 'middle_refine' if separate_timing else 'shared_repair'
    budget_policy = (UNLIMITED_REPAIR_BUDGET if repair_budget is None else
                     'middle_refine_wall_time' if separate_timing else 'shared_wall_time')
    elapsed_before_resume = 0.
    if resume_same_round and resume_state is not None:
        elapsed_before_resume = float(resume_state.get('middle_refine_elapsed_s',0.)
                                      if separate_timing else resume_state['elapsed_s'])
        if not np.isfinite(elapsed_before_resume) or elapsed_before_resume < 0:
            raise ValueError('Invalid saved repair time')
    available_seconds = None if repair_budget is None else max(0., repair_budget - elapsed_before_resume)
    cascade_enabled = segment_cascade and (repair_modules is None or 'segment_solvers' in repair_modules)
    resume_phase = None if not resume_state or resume_recheck else resume_state.get('active_phase')
    prior_attempts = list((resume_state or {}).get('segment_attempts', [])) if not resume_recheck else []
    specialist_path = path.parent/'specialist_report.json'
    repair_started = time.monotonic()
    middle_finished_at = None
    deadline = float('inf') if available_seconds is None else repair_started + available_seconds

    def middle_elapsed():
        end = time.monotonic() if middle_finished_at is None else middle_finished_at
        return end-repair_started

    def budget_details():
        return dict(budget_s=repair_budget, available_budget_s=available_seconds, budget_policy=budget_policy,
                    budget_scope=budget_scope, budget_source=budget_source, full_input_frames_50hz=total_frames_50hz,
                    middle_refine_elapsed_s=elapsed_before_resume+middle_elapsed(),
                    candidate_limits_s=limits)

    limits = repair_limits()
    configured_order = limits.pop('a_order','serial')
    a_order = configured_order if legacy_order else 'serial'
    if a_order not in ('serial','budgeted','phase'):
        raise ValueError(f'Unknown A specialist order: {a_order}')
    phase_order = window_order or a_order=='phase'
    phase_combinations = a_order=='phase'
    if not phase_combinations:
        limits.pop('cross_foot_reconstruction_seconds',None)
    specialist_deadline = deadline
    ground_deadline = deadline
    if a_order!='serial' and available_seconds is not None:
        # Isolated geometry work must give the final jitter pass a time slice.
        if (not separate_timing and 'conversion_added_jitter' in quality['flags']
                and (repair_modules is None or 'jitter' in repair_modules)):
            specialist_deadline -= .25 * available_seconds
        ground_deadline = min(specialist_deadline,repair_started+min(50.,available_seconds/15.))
    specialist_elapsed = 0.
    t=copy.deepcopy(targets);t['source_root_r']=t['root_r'];t['root_r']=Rotation.from_quat(payload['root_rot']).as_matrix()
    if legacy_order and meta.get('source_type') in ('lafan','soma','mocap') and 'source_time_s' in payload:
        from .refine_platform import grounded_leg_windows
        t['grounded_leg_windows']=grounded_leg_windows(t,meta,payload['source_time_s'])
        scene = meta.get('scene_support') or {}
        if scene.get('gravity_inferred_platforms'):
            measured_count = len(scene['boxes'])
            t['inferred_platform_start']=measured_count
            t['inferred_swing_leg_windows'] = {}
            for side,other in (('Left','Right'),('Right','Left')):
                available = ((t['sides'][other]['platform_index']>=measured_count)
                             & (t['sides'][side]['platform_index']<0))
                t['inferred_swing_leg_windows'][side] = available
                ground = available & (t['sides'][side]['contact']>0)
                t['grounded_leg_windows'][side] = t['grounded_leg_windows'].get(side,False) | ground
        elif (meta.get('source_type')=='lafan' and 'scene_support_unconfirmed' in quality['flags']
              and (quality.get('full_self_collision') or {}).get('collision_samples')):
            from .refine_platform import inferred_swing_leg_windows
            t['inferred_swing_leg_windows']=inferred_swing_leg_windows(t,payload['source_time_s'])
    original=np.c_[payload['root_pos'],payload['dof_pos']].astype(float);out=original.copy()
    trusted_support=np.zeros(len(out),bool)
    if phase_combinations:
        flight=airborne_mask(t,len(out))
        t['reconstructible_flight']=(flight & ~np.asarray(t.get('source_support_uncertain',False),bool)
                                    & ~np.asarray(t.get('source_scene_support',False),bool))
        foot_support=np.logical_or.reduce([foot['contact']>0 for foot in t['sides'].values()])
        phase_padding=max(3,round(t['fps']*.25))
    if phase_order:
        from .support_quality import preserve_contact
        for contact in t.get('body_contacts',()):
            for frame in np.flatnonzero(contact['mode']):
                trusted_support[frame] |= preserve_contact(contact,frame)
        support_phases=np.column_stack([trusted_support,*[foot['phase'] for foot in t['sides'].values()]])
        phase_edges=np.r_[0,np.flatnonzero(np.any(support_phases[1:]!=support_phases[:-1],axis=1))+1,len(out)]
        if phase_combinations:
            changed_flight=np.flatnonzero(np.diff(t['reconstructible_flight']))+1
            repair_phase_edges=np.union1d(phase_edges,changed_flight)
    baseline=path.parent/'frax_baseline';baseline.mkdir(exist_ok=False)
    for item in path.parent.glob('motion.*'):
        if item.is_file():shutil.copyfile(item,baseline/item.name)
    atomic_json(baseline/'quality.json',quality)
    if resume_state is not None:
        shutil.rmtree(baseline);shutil.copytree(previous/'frax_baseline',baseline)
        original_data=pickle.loads((baseline/'motion.pkl').read_bytes())
        robot.clip_head_angles(original_data['dof_pos'])
        original=np.c_[original_data['root_pos'],original_data['dof_pos']].astype(float)
        out=np.c_[saved['root_pos'],saved['dof_pos']].astype(float)
        before=json.loads((baseline/'quality.json').read_text())
        payload=saved
        t['root_r']=Rotation.from_quat(payload['root_rot']).as_matrix()
    baseline_sha=sha(baseline/'motion.pkl')
    raw_reference=baseline/'motion.raw_frax.npz'
    if phase_combinations and raw_reference.is_file():
        with np.load(raw_reference,allow_pickle=False) as raw:
            raw_rows=raw['x']
            if (raw_rows.shape==out.shape and np.isfinite(raw_rows).all()
                    and float(raw['fps'])==t['fps']
                    and np.array_equal(raw['root_r'],t['source_root_r'])):
                t['raw_phase_reference']=raw_rows
    queue=WorkQueue(None if resume_state is None else resume_state['states'],
                    retry_timed=resume_state is not None,distinct_contexts=phase_order)
    attempts = [] if resume_state is None or resume_recheck else resume_state['attempts']
    contact_proposals = []
    if resume_state is not None and allow_contact_retime and not resume_recheck and not plan_only:
        contact_proposals = contact_timing.restore_proposals(
            previous,path.parent,resume_state.get('contact_timing_proposals',()))
    finite_dispatch = (FiniteDispatch(queue, attempts, resume_same_round)
                       if legacy_order and separate_timing and phase_order and available_seconds is not None
                       else None)
    kept = 0 if resume_state is None or resume_recheck else resume_state['kept_candidates']
    model=CollisionModel(robot,dict(meta['configuration'].get('self_collision') or {},coverage='full'))
    infeasible_clearance={(a['start'],a['end'],a['clearance_context'],a['clearance_family'])
                          for a in attempts if 'clearance_context' in a and a['status']=='rejected'
                          and 'primal infeasible' in a.get('reason','').lower()}
    groups=collision_groups(model);models={key:submodel(model,ids) for key,ids in groups.items() if ids}
    models['leg_collision']=movable_submodel(model,sum([g['leg'] for g in robot.sides.values()],[]))
    reference=original.copy();unfixed_waist=cache['metrics']['torso']>20.
    if resume_state is not None:
        reference=np.load(artifact(previous,resume_state,'avoidance_reference.npy'),allow_pickle=False)
        unfixed_waist=np.asarray(resume_state['unfixed_waist'],bool)
        if unfixed_waist.shape!=(len(out),):raise ValueError('Invalid waist reference mask')
        if reference.shape!=original.shape or not np.isfinite(reference).all():raise ValueError('Invalid avoidance reference')
        robot.clip_head_angles(reference[:, 3:])
    reference_path=path.parent/'avoidance_reference.npy'
    np.save(reference_path,reference,allow_pickle=False)
    weights=meta['configuration']['solver'];current=measurements(cache,robot)
    torso_reference=((meta.get('frax') or {}).get('body_priority') or {}).get('torso_reference_error_deg')
    if torso_reference is not None and np.shape(torso_reference)!=(len(out),):raise ValueError('Body reference certificate length mismatch')
    def dependency_torso():
        return current['torso'] if torso_reference is None else current['torso']-np.maximum(0.,np.asarray(torso_reference)+.02-20.)
    frax_solver = None
    def frax_candidate(job):
        nonlocal frax_solver
        from .ik import FraxIK
        a, b, start, end = (job[k] for k in ('start','end','core_start','core_end'))
        if frax_solver is None:
            frax_solver = FraxIK(improved=False)
        lead_in = max(0,a-round(.6*t['fps']))
        source = window_targets(t,lead_in,b,len(out))
        source['root_r'] = source['source_root_r']
        frax_rows, _ = frax_solver(robot,source,weights)
        frax_rows = frax_rows[a-lead_in:]
        side = job['side']
        sides = [side] if side in robot.sides else list(robot.sides)
        module = job['module']
        arms = sum([robot.sides[s]['arm'] for s in sides], [])
        legs = sum([robot.sides[s]['leg'] for s in sides], [])
        waist = [i+3 for i,name in enumerate(robot.names) if name.startswith('waist_')]
        columns = {
            'arms': arms, 'collision': arms, 'torso': waist+arms,
            'leg_collision': [2]+legs, 'leg_pose': legs, 'support': [2]+legs,
            'body': [2]+legs+waist+arms, 'dense_floor': [2]+legs,
        }[module]
        alpha = np.ones(b-a)
        if a > 0 and start > a:
            alpha[:start-a] = np.sin(np.linspace(0, np.pi/2, start-a, endpoint=False))**2
        if b < len(out) and end < b:
            alpha[end-a:] = np.sin(np.linspace(np.pi/2, 0, b-end+1)[1:])**2
        candidate = out.copy()
        ix = np.ix_(np.arange(a,b), columns)
        candidate[ix] += alpha[:,None]*(frax_rows[:,columns]-out[ix])
        joints = np.asarray([col for col in columns if col >= 3])
        if len(joints):
            current = out[a:b,joints]
            proposed = candidate[a:b,joints].copy()
            limit = np.maximum(robot.speed_limits[joints-3]/t['fps'], np.abs(np.diff(current,axis=0)))
            proposed[0], proposed[-1] = current[0], current[-1]
            for i in range(1,len(proposed)-1):
                proposed[i] = np.clip(proposed[i],proposed[i-1]-limit[i-1],proposed[i-1]+limit[i-1])
            for i in range(len(proposed)-2,0,-1):
                proposed[i] = np.clip(proposed[i],proposed[i+1]-limit[i],proposed[i+1]+limit[i])
            candidate[a:b,joints] = proposed
        return candidate, dict(origin='frax_diff_ik', columns=columns)

    def split_at_support(jobs, targets):
        from .refine_routing import descendant_bodies
        limits = {}
        for job in jobs:
            module,side = job['module'], job['side']
            if module not in ('torso','arms','leg_pose'):
                yield job
                continue
            key = module,side
            if key not in limits:
                if module == 'torso':
                    columns=[i+3 for i,name in enumerate(robot.names) if name.startswith('waist_')]
                else:
                    columns=robot.sides[side]['arm' if module=='arms' else 'leg']
                movable=descendant_bodies(robot,columns)
                modes=[np.asarray(c['mode'])!=0 for c in targets['body_contacts'] if c['body'] in movable]
                state=np.stack(modes,axis=1) if modes else np.zeros((len(out),1),bool)
                limits[key]=state,np.r_[0,np.flatnonzero(np.any(state[1:]!=state[:-1],axis=1))+1,len(out)]
            state,bounds=limits[key]
            for lo,hi in zip(bounds[:-1],bounds[1:]):
                a,b=max(int(lo),job['core_start']),min(int(hi),job['core_end'])
                if a<b and hi-lo>=5:
                    yield dict(job,core_start=a,core_end=b,context_bounds=[int(lo),int(hi)],
                               active_movable_support=bool(state[a].any()),severity=min(job['severity'],b-a))

    def jobs(include_all=False):
        findings=masks(quality,cache,robot,groups,torso_reference,source_pose=v8_candidates or segment_cascade)
        platform=quality.get('inferred_platform_support')
        if 'inferred_platform_start' in t and platform is not None:
            for side in robot.sides:
                mask=np.zeros(len(out),bool)
                for entry in platform['support_violations']:
                    if entry['side']==side:mask[entry['frame']]=True
                for entry in platform['penetrating_samples']:
                    if entry['side']==side:
                        i,part=divmod(entry['sample'],8)
                        mask[i:i+1+bool(part)]=True
                if mask.any():findings.append(('support',side,mask))
        pending=jobs_from_masks(findings,len(out),POLICY['core_frames'],POLICY['padding_frames'])
        if v8_candidates or segment_cascade:
            pending=list(split_at_support(pending,t))
            torso_limit=20. if torso_reference is None else np.maximum(20.,np.asarray(torso_reference)+.02)
            active_torso=current['torso']>torso_limit
            pending=[j for j in pending if j['module']!='torso' or active_torso[j['core_start']:j['core_end']].any()]
        if not (legacy_order or window_order):
            pending = [job for job in pending if job['module']!='jitter']
        if repair_modules is not None and not include_all:
            pending=[job for job in pending if job['module'] in repair_modules]
        pending=enrich_jobs(pending,t,robot,cache,model,default_a=legacy_order)
        if phase_order:
            pending=group_jitter_jobs(pending,robot)
            leg_mask=collision_masks(cache['full_collision']['samples'],groups,len(out))['leg_collision']
            floor_mask=np.zeros(len(out),bool)
            for i in quality['interpolated_floor']['violating_intervals']:
                floor_mask[i:i+2]=True
            if phase_combinations:
                support_error=floor_mask | (current['contact']>.001)
            for job in pending:
                if phase_combinations and job['module']=='support' and job['side'] in robot.sides:
                    a,b=job['core_start'],job['core_end']
                    job['platform_window']=[max(0,a-phase_padding),min(len(out),b+phase_padding)]
                    job['candidate_order']=list(PLATFORM_PHASE)
                    continue
                if job['module']=='leg_collision':
                    a,b=job['core_start'],job['core_end']
                    window=coupled_leg_window(a,b,leg_mask,floor_mask,trusted_support,phase_edges)
                    if window is not None:
                        job['coupled_leg_window']=window
                        job['candidate_order']=[COUPLED_LEGS,*job['candidate_order']]
                    if phase_combinations and 'hip_roll_joint' not in job:
                        route=leg_phase_repair(a,b,leg_mask,support_error,
                                               t['reconstructible_flight'],foot_support,repair_phase_edges,phase_padding)
                        if route is not None:
                            job['leg_phase'],job['phase_leg_window']=route
                            order=list(PHASE_LEGS if 'raw_phase_reference' in t else PHASE_LEGS[:1])
                            if job['leg_phase']=='flight':
                                order.reverse()
                            job['candidate_order']=[*order,*job['candidate_order']]
                if job['module']=='body' and job.get('joint_support'):
                    job['restore_support_deficit']=True
                    if window_order:
                        job['candidate_order']=['coupled_body_support',PELVIS_CONTACT,'solve','solve_more']
                if job['module']=='arms' and 'candidate_order' in job:
                    order=job['candidate_order']
                    if order[0]!='surface_support_pose' and 'fit' in order:
                        job['candidate_order']=['fit',*(recipe for recipe in order if recipe!='fit')]
                if a_order=='phase' and FOOT_GEOMETRY in job.get('candidate_order',()):
                    a,b=job['core_start'],job['core_end']
                    floor=cache['floor'][8*a:8*(b-1)+1].min()
                    if floor<-.004 or current['contact'][a:b].max()>.004:
                        order=job['candidate_order']
                        job['candidate_order']=[recipe for recipe in order if recipe!=FOOT_GEOMETRY]+[FOOT_GEOMETRY]
        if v8_candidates:
            from .refine_queue import RECIPES
            for job in pending:
                module = job['module']
                job['source_pose_guard'] = True
                if module in {'arms','torso','support','leg_collision','leg_pose','body','dense_floor','collision'}:
                    recipes = job.get('candidate_order', RECIPES[module])
                    job['candidate_order'] = [recipes[0], 'frax_candidate', *recipes[1:]]
        rows = np.c_[out,payload['root_rot']]
        configured = queue.configure(pending,rows)
        if phase_combinations and (repair_modules is None or 'leg_collision' in repair_modules or include_all):
            from .cross_foot_repair import jobs as cross_foot_jobs
            fallback = []
            for job in cross_foot_jobs(model,t,cache['full_collision']['samples']):
                overlaps = any(other['module']=='leg_collision' and queue.choices(other)
                               and other['core_start']<job['core_end'] and other['core_end']>job['core_start']
                               for other in configured)
                if not overlaps or stalled_window(job,attempts):
                    fallback.append(job)
            configured += queue.configure(fallback,rows)
        return configured
    def eligible(pending):
        cm=collision_masks(cache['full_collision']['samples'],groups,len(out))
        if phase_order:
            body_support_error=np.zeros(len(out),bool)
            body_support_error[(quality.get('body_support') or {}).get('violating_frames',[])]=True
            checked=checked_geometry_frames([*prior_attempts,*attempts],len(out))
            ready,blocked=window_jobs(queue,pending,trusted_support,body_support_error,cm['collision'],cm['body_collision'],checked)
            reusable=[]
            for job in ready:
                while queue.choices(job):
                    recipe=queue.recipe(job)
                    family=clearance_trial_family(job['module'],recipe)
                    identity=(job['start'],job['end'],queue.state(job)['context'],family)
                    if identity not in infeasible_clearance:
                        reusable.append(job)
                        break
                    # A different search bound remains eligible in this slot.
                    queue.complete(job,False,recipe=recipe)
            return reusable,blocked
        ready, blocked = ready_jobs(queue,pending,dependency_torso(),cm['collision'],cm.get('body_collision'),attempts,
                                    safety_order=safety_order)
        if not v8_candidates:
            return ready, blocked
        # Let geometry and pose repair make progress in the same round. Every
        # pose candidate still goes through the complete geometry/anchor guard.
        local_pose = [j for j in queue.actionable(pending) if j['module'] in {'arms','torso','leg_pose'}]
        return list({queue.key(j):j for j in ready+local_pose}.values()), blocked

    def sync_segment_result(segment_payload, segment_quality, segment_cache):
        nonlocal payload,out,quality,cache,current
        if payload is not segment_payload:
            payload,quality,cache=segment_payload,segment_quality,segment_cache
            out=np.c_[payload['root_pos'],payload['dof_pos']].astype(float)
            t['root_r']=Rotation.from_quat(payload['root_rot']).as_matrix()
            current=measurements(cache,robot)

    def eligible_segments(segments, segment_payload, segment_quality, segment_cache):
        sync_segment_result(segment_payload,segment_quality,segment_cache)
        ready,_=eligible(jobs())
        waiting=core_frames((job for job in ready if job['module'] not in WINDOW_POSE),len(out))
        result=[]
        for segment in segments:
            start,end=segment['core_start'],segment['core_end']
            if segment['affected_frames']<2 or segment['primary']=='pose':
                continue
            if waiting[start:end].any() and not stalled_window(segment,attempts):
                continue
            channels={s.split(':',1)[0] for s in segment['symptoms']}
            if 'mesh_floor' in channels:
                channels.discard('mesh_floor')
                channels.add('dense_floor')
            if len(channels)>1 or trusted_support[start:end].any():
                result.append(segment)
        return result
    def pending_report(include_all=False):
        pending=jobs(include_all=include_all);_,blocked=eligible(pending)
        cm=collision_masks(cache['full_collision']['samples'],groups,len(out))
        pose_jobs=[j for j in pending if j['module']!='torso' and j['module'] not in safety_order]
        _,local_blocked=dependencies(pose_jobs,dependency_torso(),cm['collision'],cm.get('body_collision'),pending=pending)
        block={queue.key(j):j['reason'] for j in blocked+local_blocked}
        return [dict(j,dependency=block.get(queue.key(j))) for j in queue.pending(pending)]
    active_phase = 'specialist'

    def next_dispatch():
        return len(attempts) + len(prior_attempts)

    def checkpoint(status, include_pending=True):
        elapsed = elapsed_before_resume+(middle_elapsed() if separate_timing else time.monotonic()-repair_started)
        timing = dict(middle_refine_elapsed_s=elapsed) if separate_timing else {}
        write_checkpoint(path.parent,dict(schema=3,policy=repair_policy,refine_flow=refine_flow,status=status,baseline_sha256=baseline_sha,
            output_sha256=sha(path),states=queue.states,attempts=attempts,kept_candidates=kept,pending=pending_report() if include_pending else [],
            fixed_reference='avoidance_reference.npy',avoidance_reference_sha256=sha(reference_path),unfixed_waist=unfixed_waist.tolist(),prepared_targets_sha256=meta['prepared_targets_sha256'],
            active_phase=active_phase,segment_attempts=prior_attempts,contact_timing_proposals=contact_proposals,
            elapsed_s=elapsed, **timing))
    if plan_only:
        pending = pending_report(include_all=True)
        workload = repair_workload(pending, quality['flags'], len(out))
        if segment_cascade:
            workload['ground_settle']=len(out)
        if cascade_enabled:
            from .segment_refine import refine
            planned = refine(robot,targets,meta,payload,path,solve,repair_budget,initial_cache=cache,
                             initial_quality=quality,plan_only=True,previous_attempts=prior_attempts)
            plan = planned['_segment_plan']
            atomic_json(path.parent/'segment_plan.json',plan)
            if plan['segments']:
                workload['segment_solvers'] = len(out)
        return dict(meta, _repair_plan=workload)

    from .temporal_repair import ANGULAR,JITTER,FLAGS,Repair,final_guard
    airborne=protected_height(t,len(out))

    def run_temporal(*, jitter_only=False):
        from .discontinuities import Discontinuities
        from .inserted_frame_repair import impulse_regression
        from . import swept_height

        started_at = time.monotonic()
        cutoff = float('inf') if separate_timing else deadline
        requested = set(quality['flags']) & FLAGS
        if repair_modules is not None:
            requested = {flag for flag in requested if ('jitter' if flag==JITTER else 'root_angular') in repair_modules}
        if jitter_only:
            requested &= {JITTER}
        discontinuities_enabled = (not jitter_only and bool(quality['flags']) and
                          (repair_modules is None or 'discontinuities' in repair_modules))
        remaining_flags = set(quality['flags'])
        near_floor = (not jitter_only and swept_height.FLAG in remaining_flags and remaining_flags <= (FLAGS|{swept_height.FLAG})
                      and (repair_modules is None or 'dense_floor' in repair_modules))
        result = dict(stage='after_geometry_and_segment_solvers', requested=sorted(requested),
                      before_jitter_frames=len(quality['temporal_quality']['violating_frames']),
                      before_root_events=len(quality.get('root_discontinuities') or []))
        if not requested and not discontinuities_enabled and not near_floor:
            return dict(result,status='not_needed',kept=False,budget_exhausted=False), {}
        if started_at>=cutoff:
            return dict(result,status='no_remaining_budget',kept=False,budget_exhausted=True), {}

        with np.load(path.with_suffix('.diagnostics.npz'),allow_pickle=False) as archive:
            fields = ('root_target','root_r','source_time_s','Left_foot_target_p','Right_foot_target_p','torso_target',
                      'Left_upper','Left_lower','Left_hand_forward','Left_hand_width','Left_elbow_axis',
                      'Right_upper','Right_lower','Right_hand_forward','Right_hand_width','Right_elbow_axis',
                      'Left_thigh','Left_shin','Left_foot_target_r','Right_thigh','Right_shin','Right_foot_target_r')
            diagnostics = {key:archive[key] for key in fields if key in archive.files}
        with tempfile.TemporaryDirectory(dir=path.parent,prefix='.temporal-') as temporary:
            trial=Path(temporary)/path.name
            shutil.copyfile(path.with_suffix('.diagnostics.npz'),trial.with_suffix('.diagnostics.npz'))
            def commit_temporal(temporal):
                nonlocal payload,out,quality,cache,current,kept
                rows=robot.export_rows(temporal.rows).astype(float)
                changed=np.flatnonzero(np.any(rows!=out,axis=1)|
                                       np.any(temporal.quaternions!=payload['root_rot'],axis=1))
                result=dict(changed_frames=int(len(changed)),kept=False)
                if not len(changed):
                    return result
                if (temporal.protected_pose_changed(out,rows,0,len(rows)) or
                        not np.array_equal(temporal.quaternions[airborne],payload['root_rot'][airborne])):
                    raise RuntimeError('Temporal proposal changed protected airborne frames')
                data=dict(payload,root_pos=rows[:,:3].astype(np.float32),
                          dof_pos=rows[:,3:].astype(np.float32),root_rot=temporal.quaternions)
                trial.write_bytes(pickle.dumps(data,protocol=4))
                description=dict(meta,output_sha256=sha(trial));atomic_json(trial.with_suffix('.json'),description)
                trial_cache=fork(cache)
                changed_range=(int(changed.min()),int(changed.max())+1)
                proposed=evaluate(trial,description,solve,'frax',cache=trial_cache,changed=changed_range)
                reason=final_guard(quality,proposed,t,cache,trial_cache)
                if reason is None and impulse_regression(quality,proposed):
                    reason='root_impulse_regression'
                if reason is not None:
                    result['reason']=reason
                    return result
                payload,out,quality,cache=data,rows,proposed,trial_cache
                current=measurements(cache,robot)
                t['root_r']=temporal.rotations
                temp=path.with_suffix('.pkl.tmp');temp.write_bytes(trial.read_bytes());temp.replace(path)
                np.savez_compressed(path.with_suffix('.npz'),**{k:v for k,v in data.items() if v is not None})
                for suffix in ('.check.json','.pose_errors.npz'):
                    shutil.copyfile(trial.with_suffix(suffix),path.with_suffix(suffix))
                kept+=1
                checkpoint('running')
                return dict(result,kept=True)


            contact = meta['configuration'].get('contact',{})
            t['body_contact_anchor_tolerance_m'] = float(contact.get('max_anchor_excursion_m',.035))*meta['leg_scale']
            window_seconds = limits['specialist_candidate_seconds']
            temporal = Repair(robot,t,diagnostics,out.copy(),payload['root_rot'].copy(),model,cutoff,
                              allow_unconfirmed_arm_jitter=jitter_only)
            residual_windows = []
            if JITTER in requested:
                temporal.deadline = min(cutoff, time.monotonic()+window_seconds)
                if jitter_only:
                    temporal.jitter(preserve_contact_edges=True)
                else:
                    temporal.jitter()
                temporal.deadline = cutoff
                if time.monotonic()<cutoff:
                    from .sparse_jitter import finish
                    residual_windows = finish(temporal,weights,cutoff,candidate_seconds=window_seconds)
            if ANGULAR in requested and time.monotonic()<cutoff:
                temporal.deadline = min(cutoff, time.monotonic()+window_seconds)
                temporal.angular()
            floor_detail = (swept_height.project(robot,t,out,temporal.rows,temporal.quaternions,cache,
                                                 min(cutoff,time.monotonic()+window_seconds))
                            if near_floor and time.monotonic()<cutoff else None)
            change = commit_temporal(temporal)
            root_report = {}
            if discontinuities_enabled and time.monotonic()<cutoff:
                root_engine = Repair(robot,t,diagnostics,out.copy(),payload['root_rot'].copy(),model,
                                     min(cutoff,time.monotonic()+window_seconds))
                discontinuities = Discontinuities(root_engine,path,meta,diagnostics,repair_modules)
                discontinuities.run()
                root_change = commit_temporal(root_engine)
                root_report = discontinuities.report(out,t['root_r'],root_change['kept'])
                root_report['commit'] = root_change
            result.update(status='completed',events=temporal.events,residual_windows=residual_windows,swept_height=floor_detail,
                          after_jitter_frames=len(quality['temporal_quality']['violating_frames']),
                          after_root_events=len(quality.get('root_discontinuities') or []),
                          remaining=sorted(set(quality['flags']) & requested),
                          budget_exhausted=time.monotonic()>=cutoff,elapsed_s=time.monotonic()-started_at,**change)
        return result,root_report

    def run_segments(specialist_step, specialist_report):
        nonlocal active_phase,payload,out,quality,cache,current,specialist_elapsed,middle_finished_at
        from .segment_refine import refine
        atomic_json(specialist_path, specialist_report())
        active_phase = 'segments'
        checkpoint('running')

        def record_segment(attempts_so_far):
            nonlocal prior_attempts
            prior_attempts = list(attempts_so_far)
            checkpoint('running', include_pending=False)

        saved_report = previous/'cascade_segments.json' if resume_same_round and resume_phase in ('segments','pose') else None
        saved_segments = json.loads(saved_report.read_text()) if saved_report and saved_report.is_file() else {}
        audit_fields = lambda report: {k:v for k,v in report.items() if k!='audit_s'}
        if (saved_segments.get('final_fresh_audit_equal')
                and saved_segments['attempts']==prior_attempts
                and audit_fields(saved_segments['after'])==audit_fields(quality)):
            atomic_json(path.parent/'cascade_segments.json',saved_segments)
            final_meta = copy.deepcopy(meta)
            final_meta['stage_times_s']['segment_cascade'] = sum(
                phase['elapsed_s'] for phase in saved_segments['phases'].values())
            print('refine: restored audited segment result, continuing with temporal cleanup',flush=True)
        else:
            final_meta = refine(robot,targets,meta,payload,path,solve,available_seconds,
                                deadline=deadline,initial_cache=cache,
                                initial_quality=quality,previous_attempts=prior_attempts,checkpoint=record_segment,
                                specialist_step=specialist_step,limits=limits,next_dispatch=next_dispatch,
                                segment_filter=eligible_segments if window_order else None)
            quality = final_meta.pop('_unified_quality')
            cache = final_meta.pop('_segment_audit_cache')
        # The cache stores just the trajectory arrays. Reload the committed export
        # so temporal proposals also retain joint names, timing and frame metadata.
        payload = pickle.loads(path.read_bytes())
        out = np.c_[payload['root_pos'],payload['dof_pos']].astype(float)
        t['root_r'] = Rotation.from_quat(payload['root_rot']).as_matrix()
        current = measurements(cache,robot)
        if window_order:
            active_phase='pose'
            pose_started=time.monotonic()
            while run_specialist_trial(pose_only=True):
                pass
            specialist_elapsed+=time.monotonic()-pose_started
        specialist_snapshot = specialist_report()
        atomic_json(specialist_path, specialist_snapshot)
        middle_finished_at = time.monotonic()
        if separate_timing:
            checkpoint('running', include_pending=False)
        temporal_result,root_report = run_temporal()
        remaining_repair_s = max(0., deadline - (middle_finished_at if separate_timing else time.monotonic()))
        shared_budget_exhausted = remaining_repair_s == 0.
        final_meta['output_sha256'] = sha(path)
        final = evaluate(path,final_meta,solve,'frax',cache={})
        if audit_fields(quality) != audit_fields(final):
            raise RuntimeError('Incremental and fresh final audits disagree')
        segment_path = path.parent/'cascade_segments.json'
        segments = json.loads(segment_path.read_text())
        same_output = segments.get('after_sha256') == final_meta['output_sha256']
        segments['final_fresh_audit_equal'] = True if same_output else None
        segments['independent_audit_scope'] = 'geometry_and_final_output' if same_output else 'final_output_in_repair_report'
        atomic_json(segment_path, segments)
        prior_attempts[:] = segments['attempts']
        total_kept = kept + sum(item.get('status')=='kept' for item in prior_attempts)
        status = segments['status'] if final['flags'] and segments['status']=='pending_budget' else final['status']
        if final['flags'] and shared_budget_exhausted and queue.next(eligible(jobs())[0]) is not None:
            status = 'pending_budget'
        full_timeline = [*(dict(item,solver='plugin') for item in attempts), *prior_attempts]
        full_timeline.sort(key=lambda item: item.get('dispatch_index', -1))
        report = dict(specialist_snapshot, **budget_details())
        report.update(status=status, before=before, after=final,
            refine_flow=refine_flow,temporal_repair=temporal_result, discontinuities=root_report, attempts=full_timeline,
            all_attempts_ordered=all('dispatch_index' in item for item in full_timeline),
            kept_candidates=total_kept, specialist_report='specialist_report.json',
            segment_report='cascade_segments.json', pending_segments=segments.get('pending_segments',[]),
            shared_budget_exhausted=shared_budget_exhausted,
            elapsed_s=time.monotonic()-started, final_sha256=sha(path),
            final_fresh_audit_equal=True)
        atomic_json(path.parent/'repair.json',report)
        final_meta['unified_repair']=dict(report_path='repair.json',status=status,kept_candidates=total_kept)
        final_meta['stage_times_s']['unified_repair']=report['elapsed_s']
        final_meta['stage_times_s']['middle_refine']=middle_elapsed()
        final_meta['stage_times_s']['temporal_cleanup']=temporal_result.get('elapsed_s',0.)
        final_meta['solver_velocity_check']=final['step_velocity']
        final_meta['self_collision']['export_check']=final['self_collision']
        atomic_json(path.with_suffix('.json'),final_meta)
        checkpoint(status)
        return dict(final_meta,_unified_quality=final,_remaining_repair_s=remaining_repair_s)

    continuing_segments = cascade_enabled and resume_phase in ('segments','pose')
    if continuing_segments:
        for name in ('ground_settle.json','ground_settle_evidence.npz'):
            if (previous/name).is_file():
                shutil.copyfile(previous/name,path.parent/name)
    checkpoint('running')
    if not continuing_segments and (repair_modules is None or 'discontinuities' in repair_modules):
        from .inserted_frame_repair import prepared_index, recover
        index = prepared_index(path, meta, payload)
        if index is not None and time.monotonic()<deadline:
            def record_insertion(entry, accepted):
                nonlocal payload,out,quality,cache,current,kept
                if accepted is not None:
                    payload,out,quality,cache=accepted
                    current=measurements(cache,robot)
                    kept+=1
                entry['dispatch_index'] = next_dispatch()
                attempts.append(entry)
                checkpoint('running')

            with tempfile.TemporaryDirectory(dir=path.parent,prefix='.inserted-frames-') as temporary:
                trial=Path(temporary)/path.name
                shutil.copyfile(path.with_suffix('.diagnostics.npz'),trial.with_suffix('.diagnostics.npz'))
                recover(robot,t,model,payload,path,trial,meta,quality,cache,solve,index,
                        deadline,record=record_insertion,window_seconds=limits['specialist_candidate_seconds'])
    with tempfile.TemporaryDirectory(prefix='dex-window-refine-') as temporary:
        trial=Path(temporary)/'motion.pkl'
        shutil.copyfile(path.with_suffix('.diagnostics.npz'),trial.with_suffix('.diagnostics.npz'))
        contact=meta['configuration'].get('contact',{})
        t['body_contact_anchor_tolerance_m']=float(contact.get('max_anchor_excursion_m',.035))*meta['leg_scale']
        ground_report=None
        needs_ground_settle = legacy_order and bool(quality['interpolated_floor']['samples_below_minus_1mm'])
        if (not continuing_segments and (segment_cascade or needs_ground_settle) and time.monotonic()<ground_deadline
                and (repair_modules is None or 'ground_settle' in repair_modules)):
            from . import ground_settle as gs
            from .segment_refine import segment_guard
            times=np.asarray(payload.get('source_time_s',np.arange(len(out))/t['fps']))
            source,basis=gs.source_clearance(meta,t,times)
            ground_jobs,minimum,counts=gs.episodes(source,cache['floor'],t)
            if phase_order:
                ground_jobs=[job for job in ground_jobs
                             if trusted_support[job['core_start']:job['core_end']].any()]
                counts['trusted_body_support_windows']=len(ground_jobs)
            evidence_path=path.parent/'ground_settle_evidence.npz'
            np.savez_compressed(evidence_path,source_time_s=times,robot_mapped_floor_m=minimum,
                                protected=airborne,support=gs.contact_mask(t,len(out)),
                                **{'source_'+k+'_m':v for k,v in source.items()})
            # C already runs a contact-aware joint fit in the specialist
            # queue. Only the inexpensive vertical proposal runs ahead of it.
            methods=('vertical_lift_and_settle',)
            if not phase_order or repair_modules is not None and 'body' not in repair_modules:
                methods+=('support_joint_ik',)
            ground_report=dict(policy='source_and_robot_penetration_first_v1',source=basis,evidence=counts,
                               first_phase=True,evidence_file=evidence_path.name,segments=ground_jobs,
                               candidate_order=list(methods))
            for job in ground_jobs:
                job.update(solver='plugin',status='not_attempted',reason='shared_budget_exhausted')
                if time.monotonic()>=ground_deadline:
                    continue
                a,b,lo,hi=(job[k] for k in ('core_start','core_end','start','end'))
                # Keep one full, unmodified parent until the complete core and its joins pass.
                started_trial=time.monotonic()
                local_deadline=min(ground_deadline,started_trial+limits['specialist_candidate_seconds'])
                job['window_allowance_s']=max(0.,local_deadline-started_trial)
                job['trials']=[]
                job['dispatch_index']=next_dispatch()
                try:
                    seed,detail=gs.settled_seed(robot,t,out,cache['floor'],job)
                    job.update(detail)
                    for method in methods:
                        candidate=seed if method=='vertical_lift_and_settle' else gs.joint_fit(
                            robot,t,out,seed,job,weights,model,local_deadline)
                        from .lift_continuity import smooth_lift_correction
                        candidate = smooth_lift_correction(t,out,candidate,lo,hi)
                        if time.monotonic() >= local_deadline:
                            raise TimeoutError('ground_settle_candidate_deadline')
                        candidate=robot.export_rows(candidate).astype(float)
                        supports=gs.support_report(gs.support_measurements(robot,t,candidate,a,b))
                        result=dict(method=method,supports=supports)
                        job['trials'].append(result)
                        reason=gs.contact_failure(supports)
                        if reason is None:
                            data=dict(payload,root_pos=candidate[:,:3].astype(np.float32),dof_pos=candidate[:,3:].astype(np.float32))
                            trial.write_bytes(pickle.dumps(data,protocol=4))
                            description=dict(meta,output_sha256=sha(trial))
                            atomic_json(trial.with_suffix('.json'),description)
                            updated=fork(cache,changed=(lo,hi))
                            check=evaluate(trial,description,solve,'frax',cache=updated,changed=(lo,hi))
                            core=updated['floor'][8*a:8*(b-1)+1]
                            result['after_swept_min_m']=float(np.min(core))
                            old_boundary=cache['floor'][8*max(0,lo-1):8*min(len(out)-1,hi)+1]
                            new_boundary=updated['floor'][8*max(0,lo-1):8*min(len(out)-1,hi)+1]
                            if np.min(core)<-gs.ROUNDING_M:
                                reason='core_still_crosses_ground'
                            elif np.any(new_boundary<np.minimum(old_boundary,0.)-gs.ROUNDING_M):
                                reason='new_or_deeper_writable_or_seam_penetration'
                            else:
                                reason=segment_guard(robot,t,model,out,candidate,quality,check,cache,updated,lo,hi)
                            if reason is None:
                                temp=path.with_suffix('.pkl.tmp')
                                temp.write_bytes(trial.read_bytes());temp.replace(path)
                                np.savez_compressed(path.with_suffix('.npz'),**{k:v for k,v in data.items() if v is not None})
                                for suffix in ('.check.json','.pose_errors.npz'):
                                    shutil.copyfile(trial.with_suffix(suffix),path.with_suffix(suffix))
                                payload,out,quality,cache=data,candidate,check,updated
                                current=measurements(cache,robot)
                                kept+=1
                                result['status']='kept'
                                job.update(status='kept',reason='grounded_without_penetration',selected=method)
                                break
                        result.update(status='rejected',reason=reason)
                        job.update(status='rejected',reason=reason)
                        if time.monotonic()>=local_deadline:
                            break
                except (ValueError,RuntimeError,TimeoutError,OSQPException) as exc:
                    job.update(status='time_slice' if time_sliced(exc) else 'rejected',reason=str(exc))
                job['elapsed_s']=time.monotonic()-started_trial
                print('ground_settle',a,b,job['status'],job['reason'],flush=True)
                attempts.append(dict(job))
                checkpoint('running')
            atomic_json(path.parent/'ground_settle.json',ground_report)
        def candidate_seconds(job, recipe):
            if recipe == CROSS_FOOT:
                return limits.get('cross_foot_reconstruction_seconds',90)
            seconds = limits['specialist_candidate_seconds']
            if recipe in PLATFORM_PHASE or (recipe in PHASE_LEGS and job['leg_phase']=='support'):
                seconds *= 2
            return seconds

        def run_specialist_trial(*,pose_only=None):
            nonlocal out,payload,quality,cache,current,kept,unfixed_waist
            if time.monotonic() >= specialist_deadline:
                return False
            pending=jobs();ready,blocked=eligible(pending)
            dispatch = None
            if phase_order:
                if pose_only is not None:
                    ready=[j for j in ready if (j['module'] in WINDOW_POSE)==pose_only]
                if finite_dispatch is not None:
                    job, dispatch = finite_dispatch.choose(
                        pending, ready, quality['flags'], specialist_deadline-time.monotonic(),
                        candidate_seconds, (*PHASE_LEGS, *PLATFORM_PHASE, COUPLED_LEGS, CROSS_FOOT, 'coupled_body_support'),
                        nominal_frames=POLICY['core_frames'])
                else:
                    job=next_window_job(queue,ready,attempts[-1] if attempts else None,attempts=attempts)
            elif v8_candidates:
                available = queue.actionable(ready)
                def priority(job):
                    module=job['module']
                    peers=sum(a['module']==module for a in attempts)
                    a,b=job['core_start'],job['core_end']
                    error=job['severity']
                    if module=='torso':
                        error=float(np.max(current['torso'][a:b]))
                    elif module=='arms':
                        error=max(float(np.max(values[a:b])) for values in current['errors'][job['side']].values())
                    elif module=='leg_pose':
                        error=float(np.max(current['legs'][job['side']][a:b]))
                    return (peers, queue.state(job)['visits']//2, job.get('active_movable_support',False), -error, a)
                job = min(available,key=priority,default=None)
            else:
                job = queue.next(ready)
            if job is None or time.monotonic()>=specialist_deadline:
                return False
            growth_checkpoint()
            if time.monotonic()>=specialist_deadline:
                return False
            recipe=queue.recipe(job);entry=dict(job,recipe=recipe,status='rejected',dispatch_index=next_dispatch());begin=time.monotonic()
            verification_seconds = 0.
            if dispatch is not None:
                if dispatch['estimated_seconds'] > specialist_deadline - begin:
                    finite_dispatch.budget_blocked = True
                    return False
                verification_seconds = dispatch['verification_seconds']
                entry['budget_dispatch'] = dispatch
                old_flags = set(quality['flags'])
            before_progress=copy.deepcopy(queue.progress(job)) if phase_order else None
            family=clearance_trial_family(job['module'],recipe) if phase_order else None
            if family is not None:
                entry.update(clearance_family=family,clearance_context=queue.state(job)['context'])
            try:
                slice_s = candidate_seconds(job, recipe)
                local_deadline = min(specialist_deadline - verification_seconds, begin + slice_s)
                entry['window_allowance_s'] = max(0., local_deadline - begin)
                local_model=model if recipe==CROSS_FOOT or recipe.startswith('hand_leg') else models.get(job['module'],models.get('collision',model)) if job['module'] in ('collision','leg_collision','torso') else model
                try:
                    if recipe == 'frax_candidate':
                        candidate,detail=frax_candidate(job)
                    else:
                        candidate,detail=propose(robot,t,out,reference,weights,local_model,job,recipe,local_deadline,queue.progress(job),unfixed_waist)
                finally:
                    entry['solve_elapsed_s'] = time.monotonic() - begin
                if time.monotonic() >= local_deadline:
                    raise TimeoutError('specialist_candidate_deadline')
                aligned=detail.pop('_aligned_reference',None) if isinstance(detail,dict) else None
                platform_protection=detail.pop('_platform_protected_frames',None) if isinstance(detail,dict) else None
                rotation=detail.pop('_pelvis_rotations',None) if recipe in (PELVIS_CONTACT,PELVIS_CONTACT_RESTORE) else None
                entry['detail']=detail
                candidate=robot.export_rows(candidate).astype(float)
                # Preserve the complete in-flight pose against all repair stages.
                constrained_fit = (job['module'] in ('leg_pose','support','dense_floor','leg_collision','jitter')
                                   or job['module']=='arms' and recipe in ('fit','current_best','original_frax',
                                                                          'guided_current','guided_original'))
                locked=airborne
                if platform_protection is not None:
                    locked=platform_protection
                    fixed=np.setdiff1d(np.arange(POSE_SIZE),detail['movable_joint_columns'])
                    if not np.array_equal(candidate[:,fixed],out[:,fixed]):
                        raise ValueError('platform_repair_changed_protected_joint')
                if recipe in PHASE_LEGS and job['leg_phase']=='flight':
                    lo,hi=job['phase_leg_window']
                    locked=airborne.copy()
                    locked[lo+2:hi-2] &= ~t['reconstructible_flight'][lo+2:hi-2]
                    if not np.array_equal(candidate[:,:3],out[:,:3]):
                        raise ValueError('flight_leg_repair_changed_root')
                support_leg=SUPPORT_LEG_RECIPES.get(recipe)
                if support_leg is not None:
                    window_key,moving_side=support_leg
                    movable=t[window_key][moving_side]
                    locked=airborne & ~movable
                    fixed=np.setdiff1d(np.arange(POSE_SIZE),robot.sides[moving_side]['leg'])
                    if not np.array_equal(candidate[:,fixed],out[:,fixed]):
                        raise ValueError('ground_leg_repair_changed_platform_support')
                if constrained_fit and not np.array_equal(candidate[locked],out[locked]):
                    raise ValueError('candidate_changed_protected_frames')
                candidate[locked]=out[locked]
                data=dict(payload,root_pos=candidate[:,:3].astype(np.float32),dof_pos=candidate[:,3:].astype(np.float32))
                new_targets=None
                if rotation is not None:
                    if not np.array_equal(rotation[airborne],t['root_r'][airborne]):
                        raise ValueError('pelvis_contact_changed_protected_rotation')
                    changed=np.any(rotation!=t['root_r'],axis=(1,2))
                    quaternions=payload['root_rot'].copy()
                    if changed.any():
                        quaternions[changed]=Rotation.from_matrix(rotation[changed]).as_quat()
                    data['root_rot']=quaternions
                    new_targets=dict(t,root_r=Rotation.from_quat(quaternions).as_matrix())
                trial.write_bytes(pickle.dumps(data,protocol=4));description=copy.deepcopy(meta);description['output_sha256']=sha(trial)
                atomic_json(trial.with_suffix('.json'),description)
                lo,hi=job['start'],job['end']
                if recipe in (COUPLED_LEGS,CROSS_FOOT,FOOT_GEOMETRY,*PHASE_LEGS,*PLATFORM_PHASE) or (recipe=='coupled_body_support' and job.get('restore_support_deficit')):
                    lo,hi=detail['window']
                trial_cache=fork(cache,changed=(lo,hi))
                proposed=evaluate(trial,description,solve,'frax',cache=trial_cache,changed=(lo,hi))
                newer=measurements(trial_cache,robot)
                checked_job=dict(job,start=lo,end=hi)
                if support_leg is not None or recipe in PLATFORM_PHASE:
                    checked_job['grounded_leg_repair']=True
                if recipe.startswith('hand_leg'):checked_job['hand_leg_route']=detail['route']
                if recipe=='surface_support_pose':checked_job=dict(job,surface_support_pose=True)
                rotation_check={} if new_targets is None else {'new_targets':new_targets}
                reason=guard(robot,t,out,candidate,reference,current,newer,quality,proposed,cache,trial_cache,model,checked_job,
                             **rotation_check)
                if reason is None and recipe in PLATFORM_PHASE:
                    from .inserted_frame_repair import impulse_regression, scene_regression
                    if impulse_regression(quality,proposed):
                        reason='root_impulse_regression'
                    elif scene_regression(quality.get('scene_support'),proposed.get('scene_support')):
                        reason='measured_scene_regression'
                if 'leg_shape_tradeoff' in newer:entry['leg_shape_tradeoff']=newer['leg_shape_tradeoff']
                if reason is None:
                    gain=improved(job['module'],job['side'],lo,hi,quality,proposed,current,newer,
                                  jitter_joints=job.get('jitter_joints',()))
                    if recipe in PLATFORM_PHASE:
                        from .v8.support_acceptance import SUPPORT_ACCEPTANCE_M
                        before_platform=quality['inferred_platform_support']
                        after_platform=proposed['inferred_platform_support']
                        before_error=sum(x['error_m']-SUPPORT_ACCEPTANCE_M for x in before_platform['support_violations'])
                        after_error=sum(x['error_m']-SUPPORT_ACCEPTANCE_M for x in after_platform['support_violations'])
                        gain=(after_error<before_error-1e-5 or
                              len(after_platform['penetrating_samples'])<len(before_platform['penetrating_samples']))
                    if job.get('restore_support_deficit'):
                        from .support_quality import support_deficit
                        old_support,new_support=quality['body_support'],proposed['body_support']
                        old_loss=support_deficit(old_support,cache['frames']['body_errors'],lo,hi)
                        new_loss=support_deficit(new_support,trial_cache['frames']['body_errors'],lo,hi)
                        entry['support_deficit_m']={'before':old_loss,'after':new_loss}
                        gain=(len(new_support['violating_frames'])<len(old_support['violating_frames'])
                              or (len(new_support['violating_frames'])==len(old_support['violating_frames'])
                                  and new_loss<old_loss-1e-5))
                    if not gain:reason='no_target_metric_improvement'
                if reason is None:
                    if phase_order:
                        changed=np.any(candidate!=out,axis=1)
                        if rotation is not None:
                            changed |= np.any(data['root_rot']!=payload['root_rot'],axis=1)
                        indices=np.flatnonzero(changed)
                        if len(indices):
                            entry['changed_span']=[int(indices[0]),int(indices[-1])+1]
                    out=candidate;payload=data;quality=proposed;cache=trial_cache;current=newer;kept+=1;entry['status']='kept'
                    if new_targets is not None:t['root_r']=new_targets['root_r']
                    if aligned is not None:
                        reference[job['start']:job['end']]=np.asarray(aligned)
                        unfixed_waist &= newer['torso']>20.
                        np.save(reference_path,reference,allow_pickle=False)
                    # Commit the playable output before recording its checkpoint.
                    temp=path.with_suffix('.pkl.tmp');temp.write_bytes(trial.read_bytes());temp.replace(path)
                    np.savez_compressed(path.with_suffix('.npz'),**{k:v for k,v in data.items() if v is not None})
                    for suffix in ('.check.json','.pose_errors.npz'):shutil.copyfile(trial.with_suffix(suffix),path.with_suffix(suffix))
                    entry['reason']='improved_and_checked'
                else:
                    entry['reason']=reason
                    if (recipe==CROSS_FOOT and allow_contact_retime
                            and (reason=='new_jitter_frames' or reason.startswith('new_jitter_joint:'))):
                        edges = contact_timing.pulse_edges(quality,proposed,lo,hi)
                        gain = quality['full_self_collision']['collision_samples']-proposed['full_self_collision']['collision_samples']
                        if edges and gain>0:
                            contact_proposals.append(contact_timing.save_proposal(
                                path.parent,entry['dispatch_index'],out,candidate,payload['root_rot'],(lo,hi),edges,gain))
                            entry['contact_timing_deferred'] = list(edges)
                entry['audit_work']=trial_cache['work']
            except (ValueError,RuntimeError,TimeoutError,OSQPException) as exc:
                entry['reason']=str(exc);entry['error_type']=type(exc).__name__
                if getattr(exc,'diagnostics',None) is not None:entry['solver_diagnostics']=exc.diagnostics
                entry['status']='time_slice' if time_sliced(exc) else 'rejected'
                if family is not None and entry['status']=='rejected' and 'primal infeasible' in entry['reason'].lower():
                    infeasible_clearance.add((job['start'],job['end'],entry['clearance_context'],family))
            timed = entry['status']=='time_slice'
            unchanged_timeout = phase_order and timed and before_progress==queue.progress(job)
            queue.complete(job,entry['status']=='kept',timed=timed,recipe=recipe,
                           unchanged_timeout=unchanged_timeout,
                           result_rows=np.c_[out,payload['root_rot']] if entry['status']=='kept' else None)
            entry['elapsed_s']=time.monotonic()-begin
            if 'solve_elapsed_s' in entry:
                entry['validation_and_commit_elapsed_s'] = entry['elapsed_s'] - entry['solve_elapsed_s']
            if dispatch is not None:
                entry['resumable_progress'] = timed and not unchanged_timeout
                entry['cleared_flags'] = sorted(old_flags - set(quality['flags']))
                finite_dispatch.observe(entry)
            attempts.append(entry)
            print(f"refine: {job['module']} {job['start']}:{job['end']} {recipe}: {entry['status']} ({entry.get('reason','')})",flush=True)
            checkpoint('running')
            return True

        if cascade_enabled:
            specialist_elapsed = time.monotonic() - repair_started
            specialist_idle = False

            def specialist_step(segment_payload, segment_quality, segment_cache):
                nonlocal payload,out,quality,cache,current,specialist_elapsed,specialist_idle
                if specialist_idle and payload is segment_payload:
                    return payload,quality,cache,False
                sync_segment_result(segment_payload,segment_quality,segment_cache)
                started_step = time.monotonic()
                attempted = run_specialist_trial(pose_only=False if window_order else None)
                specialist_idle = not attempted
                specialist_elapsed += time.monotonic() - started_step
                return payload,quality,cache,attempted

            def specialist_report():
                ground_file = path.parent/'ground_settle.json'
                return dict(policy=repair_policy, status='specialist_complete', before=before, after=quality,
                            attempts=attempts, kept_candidates=kept, pending=pending_report(include_all=True),
                            baseline_sha256=baseline_sha, output_sha256=sha(path), elapsed_s=specialist_elapsed,
                            elapsed_scope='current_invocation', metrics_scope='shared_geometry',
                            attempt_elapsed_s=sum(item.get('elapsed_s', 0.) for item in attempts),
                            ground_settle_report=ground_file.name if ground_file.is_file() else None,
                            budget_scope=budget_scope)

            return run_segments(specialist_step, specialist_report)

        while (quality['flags'] or v8_candidates or segment_cascade) and run_specialist_trial():
            pass
    middle_finished_at = time.monotonic()
    if separate_timing:
        checkpoint('running', include_pending=False)
    temporal_result,discontinuity_report = run_temporal(jitter_only=legacy_order)
    remaining_repair_s = max(0., deadline - (middle_finished_at if separate_timing else time.monotonic()))
    no_candidate_fits = finite_dispatch is not None and finite_dispatch.budget_blocked
    shared_budget_exhausted = remaining_repair_s == 0. or no_candidate_fits
    # Independently certify the persisted result for specialist-only or legacy V8 execution.
    description=copy.deepcopy(meta);description['output_sha256']=sha(path)
    atomic_json(path.with_suffix('.json'),description)
    final=evaluate(path,description,solve,'frax')
    # Audit_s is diagnostic; all actual flags and measured fields must agree.
    left={k:v for k,v in quality.items() if k!='audit_s'};right={k:v for k,v in final.items() if k!='audit_s'}
    if left!=right:raise RuntimeError('Incremental/full audit disagreement; no training export permitted')
    quality=final
    status='passed' if not final['flags'] else ('accepted_with_notes' if final.get('status')=='accepted_with_notes' else ('pending_budget' if shared_budget_exhausted and queue.next(eligible(jobs())[0]) is not None else 'search_exhausted_review'))
    if status=='search_exhausted_review' and any(v['reason']=='time_sliced_candidates' for v in queue.pending(jobs())):status='time_slice_review'
    if temporal_result and temporal_result['budget_exhausted'] and set(final['flags']) & FLAGS and status=='search_exhausted_review':
        status='time_slice_review'
    checkpoint(status)
    report=dict(policy=repair_policy,refine_flow=refine_flow,status=status,attempts=attempts,kept_candidates=kept,before=before,after=quality,
        repair_modules=None if repair_modules is None else sorted(repair_modules),
        pending=pending_report(include_all=True),collision_groups={k:len(v) for k,v in groups.items()},baseline_sha256=baseline_sha,final_sha256=sha(path),
        elapsed_s=time.monotonic()-started,target_calibration='full_sequence_once_then_slice',full_v8_restarted=False,
        temporal_repair=temporal_result,discontinuities=discontinuity_report,final_fresh_audit_equal=True,
        similarity_mode='soft',shared_budget_exhausted=shared_budget_exhausted,
        a_order=a_order,temporal_reserve_s=None if available_seconds is None else deadline-specialist_deadline,
        dispatch_policy='acceptance_progress_finite' if finite_dispatch is not None else a_order,
        dispatch_budget_blocked=no_candidate_fits,
        **budget_details())
    atomic_json(path.parent/'repair.json',report)
    description['unified_repair']=dict(report_path='repair.json',kept_candidates=kept,similarity_mode='soft',status=status)
    description['stage_times_s']['unified_repair']=report['elapsed_s'];description['elapsed_s']+=report['elapsed_s']
    description['stage_times_s']['middle_refine']=middle_elapsed()
    description['stage_times_s']['temporal_cleanup']=temporal_result.get('elapsed_s',0.)
    description['self_collision']['export_check']=quality['self_collision'];description['solver_velocity_check']=quality['step_velocity']
    atomic_json(path.with_suffix('.json'),description);description['_unified_quality']=quality
    description['_remaining_repair_s']=remaining_repair_s
    if contact_proposals:
        description['_contact_timing_proposals'] = contact_proposals
    return description
