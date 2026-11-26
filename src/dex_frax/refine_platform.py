"""Protect the elevated support while fitting the opposite grounded leg."""

import numpy as np


def lafan_center_height(points):
    """Approximate whole-body height from the native LAFAN skeleton."""
    centers = [(.50, .60 * points['Hips'][:,2] + .40 * points['Neck'][:,2]),
               (.08, points['Head'][:,2])]
    segments = (('UpLeg','Leg',.10,.43), ('Leg','Foot',.0465,.43),
                ('Foot','Toe',.0145,.50), ('Arm','ForeArm',.026,.44),
                ('ForeArm','Hand',.016,.43), ('Hand','Hand__EndSite',.006,.50))
    for side in ('Left','Right'):
        for first,last,mass,fraction in segments:
            start = points[side+first][:,2]
            end = points.get(side+last,points[side+first])[:,2]
            centers.append((mass,(1-fraction)*start+fraction*end))
    return sum(mass*z for mass,z in centers) / sum(mass for mass,_ in centers)


def gravity_support_windows(targets, source_time, eligible):
    """Find stable elevated foot contacts where ballistic motion does not fit."""
    clock = np.asarray(source_time,float)
    left,right = np.searchsorted(clock,clock-.101),np.searchsorted(clock,clock+.101,side='right')
    center_height = lafan_center_height(targets['p'])
    floor = float(targets['floor'])
    result = {side: np.zeros(len(clock),bool) for side in ('Left','Right')}
    for i in np.flatnonzero(eligible):
        a,b = left[i],right[i]
        t = clock[a:b]-clock[i]
        if len(t)<7 or np.ptp(t)<.18 or not (t[0]<0<t[-1]):
            continue
        stable = []
        for side,foot in targets['sides'].items():
            for name in ('heel','toe'):
                point = foot[name][a:b]
                if point[:,2].min()-floor>.08 and np.linalg.norm(
                        point-np.median(point,axis=0),axis=1).max() <= .025:
                    stable.append(side)
                    break
        if not stable:
            continue
        z = center_height[a:b]
        design = np.column_stack([np.ones(len(t)),t,.5*t*t])
        params = np.linalg.lstsq(design,z,rcond=None)[0]
        if abs(params[2]+9.80665)<4.:
            continue
        ball = z+4.903325*t*t
        ball -= design[:,:2] @ np.linalg.lstsq(design[:,:2],ball,rcond=None)[0]
        rms = lambda values: np.sqrt(np.mean(values**2))
        if rms(ball) < max(.012,3*rms(z-design@params)):
            continue
        for side in stable:
            result[side][i] = True
    return result


def inferred_swing_leg_windows(targets, source_time):
    support = gravity_support_windows(targets, source_time, targets['source_airborne'])
    return {side: support[other] & ~support[side]
            for side, other in (('Left', 'Right'), ('Right', 'Left'))}


def grounded_leg_windows(targets, meta, source_time):
    report = meta.get('source_support_review') or {}
    intervals = report.get('unresolved_platforms', ())
    if not intervals or report.get('frame_convention') != 'zero_based_prepared_end_exclusive':
        return {}
    source_review = meta.get('source_review') or {}
    first_native = (source_review.get('kept_native_interval') or [0])[0]
    origin = first_native / float(meta['input_fps'])
    prepared = (np.asarray(source_time, float) - origin) * float(report['fps'])
    elevated = {side: np.zeros(len(prepared), bool) for side in targets['sides']}
    for segment in intervals:
        side = segment['side']
        elevated[side] |= (prepared >= segment['start'] - 1e-6) & (prepared < segment['end'] - 1e-6)
    ambiguous = (np.asarray(targets['source_support_uncertain'], bool)
                 & ~np.asarray(targets.get('source_airborne', False), bool)
                 & ~np.asarray(targets.get('source_scene_support', False), bool))
    return {
        side: ambiguous & ~elevated[side] & elevated[other] & (targets['sides'][side]['contact'] > 0)
        for side, other in (('Left', 'Right'), ('Right', 'Left'))
    }
