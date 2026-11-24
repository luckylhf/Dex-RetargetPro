"""Correct a foot-stage contact transition before the arm stage commits to its root.

Use both neighbors as boundary conditions and fit stance leg(s) to the
original source pivots at the proposed root height. The frame and the two
incoming/outgoing swept intervals are assessed together.
"""
import numpy as np
from .airborne import protected_height
from scipy.optimize import least_squares
from .contact_phase import lowest, support_errors, transition_windows
from .unified_repair import dense_geometry
from .v8.self_collision import CollisionModel


def landing_fit(robot, targets, rows, frame, height, reference, solver_config,
                *, translate_xy=True, speed_margin=0.):
    feet = [(robot.sides[side], foot) for side, foot in targets['sides'].items()
            if foot['contact'][frame] > 0]
    if not feet:
        return None
    legs = np.array([axis for g, _ in feet for axis in g['leg']])
    ids = np.r_[0,1,legs] if translate_xy else legs
    step = robot.joint_step(targets['fps'],solver_config)[legs-3]-speed_margin
    if translate_xy: step = np.r_[[np.inf,np.inf],step]
    before, current, after = rows[frame-1:frame+2]
    lo = np.maximum.reduce([before[ids]-step, after[ids]-step, robot.lower[ids]])
    hi = np.minimum.reduce([before[ids]+step, after[ids]+step, robot.upper[ids]])
    if translate_xy:
        lo[:2]=current[:2]-.045
        hi[:2]=current[:2]+.045
    if np.any(hi-lo < 1e-8):
        return None
    base = current.copy()
    base[2] = height
    aim = np.clip(reference[ids], lo, hi)
    regularization = .00025
    regularization = np.full(len(legs),regularization)
    if translate_xy: regularization = np.r_[[.015,.015],regularization]
    prior = np.diag(regularization)
    def terms(q, jac):
        row = base.copy()
        row[ids] = q
        robot.set(row, targets['root_r'][frame])
        blocks = []
        for g, foot in feet:
            body, pivot = g['foot'], foot['pivots'][frame]
            local, point = lowest(robot,body,g['hull'])
            if jac:
                blocks.append(robot.jac(body,pivot)[0][:2,ids])
                blocks.append(robot.jac(body,local)[0][2:3,ids])
            else:
                blocks.extend([robot.point(body,pivot)[:2]-foot['anchors'][frame,:2],point[2:3]])
        blocks.append(prior if jac else prior@(q-aim))
        return np.vstack(blocks) if jac else np.concatenate(blocks)
    fit = least_squares(lambda q:terms(q,False),aim,jac=lambda q:terms(q,True),
                        bounds=(lo,hi),max_nfev=65,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    base[ids] = fit.x
    return base


def correct(robot, t, rows, collision_config, solver_config):
    from .temporal_constraints import assess,diagnostics,regressions
    events = []
    windows = transition_windows(rows,t)
    if not windows:
        return rows,events
    result = rows.copy()
    model = CollisionModel(robot,dict(collision_config or {},coverage='full'))
    dg = diagnostics(t)
    protected=protected_height(t,len(rows))
    for a,b,i in windows:
        if i == 0 or i+1 >= len(rows) or protected[i]:
            continue
        old = result[i-1:i+2].astype(np.float32).astype(float)
        local = dict(root_r=t['root_r'][i-1:i+2])
        old_d,old_f = dense_geometry(robot,model,local,old)
        peak = lambda poses:float(np.linalg.norm(np.diff(poses[:,:3],axis=0),axis=1).max()*t['fps'])
        prior_peak = peak(old)
        ref = (old[0]+old[2])*.5
        old_support,tol = support_errors(robot,t,old[1:2],i)
        low,high = max(0,a-20),min(len(rows),b+20)
        signals = {k:v[low:high] for k,v in dg.items()}
        angles = result[low:high,3:].copy()
        prev_jitter = assess(angles,t['fps'],robot.names,signals)
        event = dict(frame=int(i),peak_before=prior_peak,deep_before_mm=float(-1000*old_f.min()),trials=0,rejected={})
        best_peak = prior_peak
        best = None
        for factor in (1.,.8,.6,.4,.25,.12):
            height = old[1,2]+factor*(ref[2]-old[1,2])
            for seed in (ref,old[2],old[1]):
                row = landing_fit(robot,t,result,i,height,seed,solver_config)
                if row is None:continue
                row = row.astype(np.float32).astype(float)
                test = old.copy();test[1] = row
                speed = peak(test)
                if speed >= best_peak-.015:continue
                event['trials']+=1
                value,_=support_errors(robot,t,row[None],i)
                if np.any(value>np.maximum(tol,old_support)+2e-6):
                    why='contact'
                else:
                    ds,fs=dense_geometry(robot,model,local,test)
                    if np.any(ds<np.minimum(0.,old_d)-2e-6):why='collision'
                    elif np.any(fs<np.minimum(-.001,old_f)-2e-6):why='floor'
                    else:
                        angles[i-low]=row[3:]
                        why='jitter' if regressions(prev_jitter,assess(angles,t['fps'],robot.names,signals)) else ''
                if why:
                    event['rejected'][why]=event['rejected'].get(why,0)+1
                else:
                    best=row;best_peak=speed
                    event['deep_after_mm']=float(-1000*fs.min())
        if best is not None:
            orig=result[i].copy()
            result[i]=np.where(best==old[1],orig,best)
        event['kept']=best is not None;event['peak_after']=best_peak
        events.append(event)
    return result,events
