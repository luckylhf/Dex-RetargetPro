"""Measured collision ownership and strict local torso dependencies."""
from collections import Counter
import copy
import numpy as np


def descendant_bodies(robot,columns):
    roots=set(int(robot.m.jnt_bodyid[robot.jids[j-3]]) for j in columns)
    def belongs(body):
        while body:
            if body in roots:return True
            body=int(robot.m.body_parentid[body])
        return False
    return {b for b in range(1,robot.m.nbody) if belongs(b)}


def collision_groups(model):
    robot=model.r
    arms=descendant_bodies(robot,sum([g['arm'] for g in robot.sides.values()],[]))
    legs=descendant_bodies(robot,sum([g['leg'] for g in robot.sides.values()],[]))
    result={'collision':[],'leg_collision':[],'body_collision':[]}
    for k,(a,b) in enumerate(model.pairs):
        bodies={int(model.m.geom_bodyid[a]),int(model.m.geom_bodyid[b])}
        key='collision' if bodies&arms else 'leg_collision' if bodies&legs else 'body_collision'
        result[key].append(k)
    return result


def submodel(model,indices):
    """Solver sees only its movable collision class; adoption still checks all."""
    import mink
    from .ankle_collision import AnkleCollisionAvoidanceLimit
    result=copy.copy(model)
    pairs=model.pairs[np.asarray(indices,int)]
    result.limit=AnkleCollisionAvoidanceLimit(result.m,[([int(a)],[int(b)]) for a,b in pairs],
        minimum_distance_from_collisions=result.cfg['margin_m'],collision_detection_distance=result.cfg['detection_m'])
    result.pairs=np.asarray(result.limit.geom_id_pairs,int)
    result.names=[[result.m.geom(a).name,result.m.geom(b).name] for a,b in result.pairs]
    result.configuration=mink.Configuration(result.m)
    return result


def collision_masks(samples,groups,n):
    out={key:np.zeros(n,bool) for key in groups};owner={i:key for key,ids in groups.items() for i in ids}
    for sample,(_,_,pairs) in enumerate(samples):
        if not pairs:continue
        frame,k=divmod(sample,8)
        for key in {owner[p] for p in pairs}:
            out[key][frame]=True
            if k:out[key][min(n-1,frame+1)]=True
    return out


def dependencies(jobs,torso,arm_collision,body_collision=None,pending=None):
    """Prior unresolved stages block overlapping downstream windows.

    Exhausted tasks remain blockers; they cannot disappear merely because no
    solver route is left. Independent windows can still make progress.
    """
    from .refine_queue import PHASE
    pending=jobs if pending is None else pending
    ready=[];blocked=[]
    for job in jobs:
        a,b=job['start'],job['end'];stage=job['module'];reason=None
        if stage=='body_collision':reason='no_certified_body_collision_module'
        elif PHASE[stage]>1 and np.any(torso[a:b]>20.):reason='waiting_for_torso'
        elif PHASE[stage]>1 and body_collision is not None and np.any(body_collision[a:b]):reason='waiting_for_body_geometry'
        elif stage=='arms' and np.any(arm_collision[a:b]):reason='waiting_for_arm_clearance'
        if reason is None:
            blockers=[j for j in pending if PHASE[j['module']]<PHASE[stage] and j['start']<b and j['end']>a]
            if blockers:reason='waiting_for_'+min(blockers,key=lambda j:PHASE[j['module']])['module']
        if reason:blocked.append(dict(job,reason=reason))
        else:ready.append(job)
    return ready,blocked


def alignment_reference(reference,aligned,unfixed,a,b):
    """Only an unresolved waist may establish a new pre-avoidance reference.

    Correct neighboring frames keep their original reference, even when a
    larger repair window overlaps them. Never rebase an already-correct frame
    on the outcome of previous shoulder avoidance.
    """
    out=reference.copy();indices=np.flatnonzero(np.asarray(unfixed)[a:b])+a
    out[indices]=aligned[indices]
    return out


def movable_submodel(model,columns):
    """All pairs affected by these joints, regardless of repair ownership.

    A leg repair owns leg/body penetration, but it can also push a thigh into a
    fixed hand. Such arm/leg pairs must constrain its candidate solver too.
    """
    bodies=descendant_bodies(model.r,columns)
    indices=[k for k,(a,b) in enumerate(model.pairs)
             if int(model.m.geom_bodyid[a]) in bodies or int(model.m.geom_bodyid[b]) in bodies]
    return submodel(model,indices)


SAFETY_ORDER={'leg_collision':0,'collision':1,'support':2,'dense_floor':3,'body':4,'jitter':5}
LEGACY_SAFETY_ORDER={'leg_collision':0,'support':1,'dense_floor':2,'jitter':3,'body':4,'collision':5}
WINDOW_GEOMETRY=('leg_collision','collision','support','dense_floor')
WINDOW_POSE=frozenset(('arms','torso','leg_pose'))
WINDOW_PRIORITY={name:i for i,name in enumerate((*WINDOW_GEOMETRY,'jitter','body','torso','arms','leg_pose'))}


def core_frames(jobs,n):
    active=np.zeros(n,bool)
    for job in jobs:
        active[job['core_start']:job['core_end']]=True
    return active


def overlaps(a,b,lo,hi):
    return a<hi and lo<b


def coupled_leg_window(start,end,collision,floor,trusted_support,phase_edges):
    """Cover measured defects in the core with a short, contact-aware fit."""
    first=max(0,int(np.searchsorted(phase_edges,start,side='right'))-1)
    last=int(np.searchsorted(phase_edges,end))
    spans=[]
    for left,right in zip(phase_edges[first:last],phase_edges[first+1:last+1]):
        a,b=max(start,left),min(end,right)
        legs=collision[a:b]
        if not legs.any() or not (floor[a:b] | trusted_support[a:b]).any():
            continue
        # Use up to three nearby frames on either side of the scheduling
        # core, leaving two more frames inside the ordinary context for joins.
        lo,hi=max(left,start-3),min(right,end+3)
        flagged=np.flatnonzero(collision[lo:hi] | floor[lo:hi])+lo
        spans.append((int(legs.sum()),int(flagged[0]),int(flagged[-1])+1))
    if spans:
        _,a,b=max(spans,key=lambda span:(span[0],-span[1]))
        return [max(0,a-2),min(len(collision),b+2)]
    return None


def group_jitter_jobs(pending, robot):
    """Use one queue entry for each set of joints sharing a local solver."""
    groups={robot.names[i-3]:'legs' for leg in robot.sides.values() for i in leg['leg']}
    for side,limb in robot.sides.items():
        groups.update({robot.names[i-3]:side+'_arm' for i in limb['arm']})
    result=[]
    grouped={}
    for job in pending:
        if job['module']!='jitter':
            result.append(job)
            continue
        group=groups.get(job['side'],job['side'])
        key=(group,job['core_start'],job['core_end'])
        if key not in grouped:
            grouped[key]=dict(job,side=group,jitter_joints=[],severity=0)
        merged=grouped[key]
        merged['jitter_joints'].append(job['side'])
        merged['severity']+=job['severity']
    return result+list(grouped.values())


def clearance_trial_family(stage, recipe):
    """Map recipes that solve identical support/floor constraints and bounds."""
    if stage in ('support','dense_floor','leg_collision'):
        if recipe in ('geometry5_elastic','solve','legs5'):
            return 5
        if recipe in ('geometry10_elastic','solve_more','legs10'):
            return 10
    return None


def stalled_window(segment, attempts):
    """Offer a new solver after overlapping specialists stop making progress."""
    lo,hi=segment['core_start'],segment['core_end']
    misses=0
    for attempt in reversed(attempts):
        if attempt['module'] not in (*WINDOW_GEOMETRY,'body','jitter'):
            continue
        a,b=attempt.get('changed_span',(attempt['core_start'],attempt['core_end']))
        if not overlaps(a,b,lo,hi):
            continue
        if attempt['status']=='kept':
            break
        if attempt['status'] in ('rejected','time_slice'):
            misses+=1
        if misses>=2:
            return True
    return False


def checked_geometry_frames(attempts,n):
    """Only allow a local temporal follow-up where geometry actually changed."""
    active=np.zeros(n,bool)
    for entry in attempts:
        if entry['module'] in (*WINDOW_GEOMETRY,'body','ground_settle') and entry['status']=='kept':
            a,b=entry.get('changed_span',(entry['start'],entry['end']))
            active[a:b]=True
    return active


def window_jobs(queue,pending,trusted_support,body_support_error,arm_collision,body_collision,geometry_checked=None):
    """Resolve overlapping prerequisites while independent windows stay eligible."""
    n=len(trusted_support)
    actionable=queue.actionable(pending)
    geometry=core_frames((j for j in actionable if j['module'] in WINDOW_GEOMETRY),n)
    support_to_repair=(core_frames((j for j in actionable if j['module']=='body'),n)
                       & trusted_support & body_support_error)
    if geometry_checked is None:
        geometry_checked=np.zeros(n,bool)
    unresolved=core_frames((j for j in pending if j['module'] in (*WINDOW_GEOMETRY,'body')),n)
    ready,blocked=[],[]
    for job in actionable:
        core=slice(job['core_start'],job['core_end'])
        stage=job['module']
        reason=None
        if stage=='body_collision':
            reason='no_certified_body_collision_module'
        elif stage=='body' and not trusted_support[core].any():
            reason='awaiting_trusted_body_support'
        elif stage=='body' and (geometry[core] | body_collision[core]).any():
            if job.get('joint_support') and (trusted_support[core] & body_support_error[core]).any():
                job=dict(job, coupled_support_priority=True)
            else:
                reason='waiting_for_local_geometry'
        elif stage=='jitter' and (geometry[core] | body_collision[core]).any() and not geometry_checked[core].any():
            reason='waiting_for_local_geometry'
        elif stage=='jitter' and support_to_repair[core].any() and not geometry_checked[core].any():
            reason='waiting_for_local_body_support'
        elif stage in WINDOW_POSE and (unresolved[core] | body_collision[core] | arm_collision[core]).any():
            reason='waiting_for_local_geometry'
        if reason:
            blocked.append(dict(job,reason=reason))
        else:
            ready.append(job)
    return ready,blocked


def next_window_job(queue,ready,previous=None,*,attempts=()):
    # Rotate specialist families before revisiting another window of the
    # same family. The window gate still decides what is safe to try.
    family_visits=Counter(entry['module'] for entry in attempts)

    def order(job):
        stage=job['module']
        tier=1 if stage=='body' else 2 if stage in WINDOW_POSE else 0
        if job.get('coupled_support_priority'):
            tier=0
            priority=-1
        else:
            priority=WINDOW_PRIORITY[stage]
        return (tier,family_visits[stage],queue.state(job)['visits'],
                priority,-job['severity'],job['core_start'])
    if previous and previous.get('changed_span') and previous['status']=='kept' and previous['module'] in (*WINDOW_GEOMETRY,'body'):
        lo,hi=previous['changed_span']
        temporal=[job for job in ready if job['module']=='jitter'
                  and overlaps(job['core_start'],job['core_end'],lo,hi)]
        if temporal:
            return min(temporal,key=order)
        if previous['module']=='leg_collision':
            clearance=[job for job in ready if job['module'] in ('dense_floor','support')
                       and overlaps(job['core_start'],job['core_end'],lo,hi)]
            if clearance:
                return min(clearance,key=order)
    if previous and previous['module']=='support' and previous['status']=='kept':
        collisions=[job for job in ready if job['module'] in ('collision','leg_collision')]
        floor=[job for job in ready if job['module']=='dense_floor'
               and job['core_start']<previous['end'] and previous['start']<job['core_end']
               and not any(other['core_start']<job['core_end'] and job['core_start']<other['core_end']
                           for other in collisions)]
        if floor:
            return min(floor,key=order)
    return min(ready,key=order,default=None)


def ready_jobs(queue,pending,torso,arm_collision,body_collision,attempts,*,safety_order=SAFETY_ORDER):
    """Shared safety fits protect each other physically, without queue deadlocks.

    Identical selection is used by normal refinement and safety-scoped runs.
    Existing dependencies continue to protect pose-only work after safety.
    """
    actionable=queue.actionable(pending)
    safety=[job for job in actionable if job['module'] in safety_order]
    if safety:
        visits={name:sum(entry['module']==name for entry in attempts) for name in safety_order}
        chosen=min({job['module'] for job in safety},key=lambda name:(visits[name],safety_order[name]))
        return [job for job in safety if job['module']==chosen],[]
    return dependencies(actionable,torso,arm_collision,body_collision,pending=pending)
