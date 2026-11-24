"""First-pass grounding of contiguous source/robot penetration episodes."""
from pathlib import Path
import time

import numpy as np

from .airborne import protected_height
from .source_geometry import SURFACES, load_surface
from .support_quality import preserve_contact
from .v8.math3d import runs
from .v8.multicontact import contact_local

PENETRATION_M = .001
ROUNDING_M = .000002


def source_clearance(meta, targets, times):
    """Measure source landmarks/hulls independently of the robot trajectory."""
    from .v8.source import adapt
    from .v8.bvh import load
    from .cli import sha
    plane = targets.get('floor')
    source = meta.get('source_type')
    basis = dict(source_type=source, prepared_source_ground_m=plane, calibration=targets.get('ground_calibration'))
    if plane is None or not np.isfinite(plane):
        return {}, dict(basis, reason='source_plane_unavailable')
    if source in ('lafan','soma','mocap'):
        path = Path(meta['input'])
        if sha(path) != meta['input_sha256']:
            raise ValueError('Source changed before ground-settle measurement')
        motion = load(path, **(meta.get('bvh_input_convention') or {}))
        end = min(len(motion.local_p),max(3,int(np.ceil(times[-1]*motion.fps))+2))
        review = meta.get('source_review') or {}
        discarded = review.get('discarded_frames', [])
        repaired = review.get('repaired_frames', [])
        replay_needed = bool(repaired) and 'native_frame_count' in review
        start, stop = review.get('kept_native_interval') or (int(0 in discarded), len(motion.local_p))
        end = min(end, stop)
        if replay_needed:
            from .source_integrity import inspect
            count = review['native_frame_count']
            motion.local_p, motion.local_r = motion.local_p[:count], motion.local_r[:count]
            motion, replay = inspect(motion, source)
            if replay['repaired_frames'] != repaired or replay['discarded_frames'] != discarded:
                raise ValueError('Input audit no longer reproduces the source geometry')
            motion.local_p, motion.local_r = motion.local_p[:end-start], motion.local_r[:end-start]
        else:
            motion.local_p, motion.local_r = motion.local_p[start:end], motion.local_r[start:end]
        native=adapt(motion,source)
        from .v8.contacts import estimate_ground
        plane,calibration=estimate_ground(native,meta['configuration']['contact'])
        basis.update(native_ground_calibration=calibration)
        clock=(start+np.arange(len(motion.local_p)))/motion.fps
        basis.update(geometry='audited_native_bvh_fk' if replay_needed or discarded else 'original_native_bvh_fk', source_fps=motion.fps,
                     source_sha256=meta['input_sha256'],alignment='linear height on source_time_s')
    elif source=='smplx' and all('heel' in f and 'toe' in f for f in targets['sides'].values()):
        native,clock=targets,times
        basis.update(geometry='source_model_heel_toe_before_robot_fit',alignment='prepared source_time_s')
    else:
        return {}, dict(basis, reason='independent_source_geometry_unavailable')
    measurements = {side+'_'+landmark: foot[landmark][:,2]
                    for side,foot in native['sides'].items() for landmark in ('heel','toe')}
    if source in ('soma','lafan'):
        shape = load_surface(source)
        for region,bone in shape['semantic_bones'].items():
            vertices=np.asarray(shape['bones'][bone]['vertices'])
            z=native['r'][bone][:,2,:]@vertices.T
            measurements[region]=z.min(axis=1)+native['p'][bone][:,2]
        basis['source_body_surfaces'] = SURFACES[source]
    names=list(measurements)
    basis.update(channels=names,source_ground_m=float(plane))
    aligned={name: np.interp(times,clock,value-plane) for name,value in measurements.items()}
    return aligned,basis


def contact_mask(targets,n):
    active=np.logical_or.reduce([np.asarray(f['contact'])>0 for f in targets['sides'].values()])
    for part in targets.get('body_contacts',()):
        for i in np.flatnonzero(part['mode']):
            if preserve_contact(part,i):
                active[i]=True
    return active


def episodes(source, swept_floor, targets):
    """Pair source timestamps with robot native and within-step measurements."""
    n=(len(swept_floor)-1)//8+1
    samples=np.asarray(swept_floor).min(axis=1)
    robot=samples[::8].copy()
    if n>1:
        low=np.minimum(samples[:-1].reshape(n-1,8).min(axis=1),samples[8::8])
        robot[:-1]=np.minimum(robot[:-1],low)
        robot[1:]=np.minimum(robot[1:],low)
    if not source:
        return [],robot,dict(reason='no_source_witness')
    depths=np.column_stack(list(source.values()))
    src=depths.min(axis=1)
    both=(src < -PENETRATION_M) & (robot < -PENETRATION_M)
    protected=protected_height(targets,n)
    support=contact_mask(targets,n)
    selected=both & support & ~protected
    counts=dict(source_penetrating_frames=int((src < -PENETRATION_M).sum()),
                robot_penetrating_frames=int((robot < -PENETRATION_M).sum()),
                coincident_frames=int(both.sum()),protected_overlap_frames=int((both&protected).sum()),
                unmarked_overlap_frames=int((both&~support&~protected).sum()),eligible_frames=int(selected.sum()))
    if not np.any(selected):
        counts['reason']='no_coincident_grounded_penetration'
    names=list(source)
    jobs=[]
    for a,b in runs(selected):
        a,b=int(a),int(b)
        # Two immobile context samples on each side of the writable band.
        lo,hi=max(0,a-5),min(n,b+5)
        left=np.flatnonzero(protected[lo:a])
        right=np.flatnonzero(protected[b:hi])
        if len(left):lo+=int(left[-1])+1
        if len(right):hi=b+int(right[0])
        witnesses={names[col]:float(depths[a:b,col].min()) for col in np.flatnonzero((depths[a:b]<-PENETRATION_M).any(axis=0))}
        jobs.append(dict(module='ground_settle',recipe='lift_descend_to_marked_support',
                         start=lo,end=hi,core_start=a,core_end=b,source_witnesses_m=witnesses,
                         robot_floor_min_m=float(robot[a:b].min())))
    return sorted(jobs,key=lambda j:(j['robot_floor_min_m'],j['core_start'])),robot,counts


def support_measurements(robot,t,rows,a,b):
    """Signed heights of each marked support; fixed anchors use their own XY target."""
    records={}
    def add(key,frame,z,xy,xy_limit):
        records.setdefault(key,[]).append((frame,float(z),None if xy is None else float(xy),xy_limit))
    for i in range(a,b):
        robot.set(rows[i],t['root_r'][i])
        for side,group in robot.sides.items():
            foot=t['sides'][side]
            if foot['contact'][i]>0:
                xyz=robot.point(group['foot'],foot['pivots'][i])-foot['anchors'][i]
                add(side+'_foot',i,robot.point(group['foot'],robot.lowest(side))[2],np.linalg.norm(xyz[:2]),.001)
        for c in t.get('body_contacts',()):
            if preserve_contact(c,i):
                pt=robot.point(c['body'],contact_local(robot,c))
                fixed=c['mode'][i]==1
                add(c['body_name'],i,pt[2],np.linalg.norm(pt[:2]-c['anchors'][i,:2]) if fixed else None,
                    t['body_contact_anchor_tolerance_m'])
    return records


def support_report(records):
    result={}
    for key,values in records.items():
        xy=[p[2] for p in values if p[2] is not None]
        result[key]=dict(frames=len(values),min_z_m=min(p[1] for p in values),max_z_m=max(p[1] for p in values),
                         max_abs_z_m=max(abs(p[1]) for p in values),max_xy_m=max(xy,default=None),
                         xy_limit_m=values[0][3])
    return result


def contact_failure(report):
    if not report:
        return 'missing_marked_support'
    for name,part in report.items():
        if part['min_z_m'] < -ROUNDING_M or part['max_z_m'] > PENETRATION_M:
            return 'support_not_on_plane:'+name
        if part['max_xy_m'] is not None and part['max_xy_m'] > part['xy_limit_m']+ROUNDING_M:
            return 'fixed_xy_anchor:'+name
    return None


def settled_seed(robot,t,rows,floor,job):
    a,b,lo,hi=(job[k] for k in ('core_start','core_end','start','end'))
    raw=support_measurements(robot,t,rows,a,b)
    minimum=np.asarray(floor[8*a:8*(b-1)+1]).min(axis=1)
    clear=np.maximum(0.,.00002-minimum)
    required=clear[::8].copy()
    if b-a>1:
        interval=np.maximum(clear[:-1].reshape(-1,8).max(axis=1),clear[8::8])
        required[:-1]=np.maximum(required[:-1],interval)
        required[1:]=np.maximum(required[1:],interval)
    lowest_support=np.full(b-a,np.inf)
    for points in raw.values():
        for frame,height,_,_ in points:
            lowest_support[frame-a]=min(lowest_support[frame-a],height)
    if not np.isfinite(lowest_support).all():
        raise ValueError('missing_marked_support')
    lift=max(float(required.max()),float(-lowest_support.min()),0.)+.001
    # Translate the complete core together, then lower it to the first
    # support/clearance boundary. Incompatible supports proceed to joint IK.
    offsets=lift-np.minimum(lift+lowest_support,lift-required)
    dz=np.zeros(len(rows))
    dz[a:b]=offsets
    if a>lo:
        dz[lo:a]=np.linspace(0.,offsets[0],a-lo+1)[:-1]
    if b<hi:
        dz[b:hi]=np.linspace(offsets[-1],0.,hi-b+1)[1:]
    new=rows.copy()
    new[lo:hi,2]+=dz[lo:hi]
    return new,dict(common_lift_m=lift,settled_offset_min_m=float(offsets.min()),settled_offset_max_m=float(offsets.max()),
                    before_supports=support_report(raw))


def joint_fit(robot,t,rows,seed,job,cfg,model,deadline):
    from .adaptive_refine import MinimumChangeObjective
    from .local_refine import fit_window
    a,b=job['start'],job['end']
    ids=[2]+sum([g['leg'] for g in robot.sides.values()],[])
    for g in robot.sides.values():
        arm_bodies={g[k] for k in ('shoulder','elbow','forearm','wrist','hand','tcp')}
        if any(c['body'] in arm_bodies and any(preserve_contact(c,i) for i in range(a,b))
               for c in t.get('body_contacts',())):
            ids.extend(g['arm'])
    options=dict(cfg,_repair_deadline=deadline,_coupled_swept_support=True,_initial_leg_collision_cfg=model.cfg)
    # Bounded-size temporal solves, with one final transaction over the entire episode.
    draft=seed.copy()
    for i in range(a,b,18):
        if time.monotonic()>=deadline:raise TimeoutError('ground_settle_joint_budget')
        lo,hi=max(a,i-3),min(b,i+21)
        if hi-lo<5:continue
        reference=draft.copy()
        fitted,_=fit_window(robot,t,reference,options,lo,hi,iterations=6,ids_override=np.asarray(ids),
                            allow_support=True,local_checks=True,seed=reference,
                            objective_factory=lambda frame:MinimumChangeObjective(seed[frame]))
        draft[lo:hi]=fitted[lo:hi]
    return draft
