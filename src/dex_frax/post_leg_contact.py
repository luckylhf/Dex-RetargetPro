"""Recheck contact after the protected leg solve, while the legs can still move."""
import time

import numpy as np
from osqp import OSQPException

from .airborne import protected_height
from .frax_ground_constraints import ROOT_CORRECTION_SPEED_M_S
from .ground_settle import PENETRATION_M, ROUNDING_M, contact_mask, settled_seed, support_measurements
from .torso_branch import floor_samples
from .v8.math3d import runs
from .v8.self_collision import CollisionModel

POLICY = 'post_leg_contact_relanding_v1'
SUBSTEPS = 8


def contact_errors(robot, targets, rows, a, b):
    """Per-frame errors and allowances for the existing contact annotations."""
    errors = {}
    for name, samples in support_measurements(robot, targets, rows, a, b).items():
        height_limit = (PENETRATION_M if name in ('Left_foot', 'Right_foot')
                        else targets['body_contact_height_tolerance_m'])
        for i, z, xy, xy_limit in samples:
            errors[(i, name, 'height')] = (abs(z), height_limit)
            if xy is not None:
                errors[(i, name, 'xy')] = (xy, xy_limit)
    for side, group in robot.sides.items():
        foot = targets['sides'][side]
        points = {}
        for i in range(a, b):
            if foot['contact'][i] <= 0:
                continue
            robot.set(rows[i], targets['root_r'][i])
            points[i] = robot.point(group['foot'], foot['pivots'][i])[:2].copy()
            if (i-1 in points and foot['phase'][i-1] == foot['phase'][i]
                    and np.array_equal(foot['pivots'][i-1], foot['pivots'][i])
                    and np.array_equal(foot['anchors'][i-1], foot['anchors'][i])):
                slip = np.linalg.norm(points[i]-points[i-1])
                errors[(i, side+'_foot', 'slip')] = (slip, PENETRATION_M)
    return errors


def contact_regressions(before, after):
    return [key for key, (error, tolerance) in after.items()
            if error > max(tolerance, before[key][0])+ROUNDING_M]


def affected_frames(before_floor, after_floor, before_contact, after_contact):
    """A bad fractional sample implicates the two poses that generate it."""
    n = (len(after_floor)-1)//SUBSTEPS+1
    frames = np.zeros(n, dtype=bool)
    samples = np.flatnonzero(np.any(after_floor < np.minimum(before_floor, -PENETRATION_M)-ROUNDING_M, axis=1))
    frames[samples//SUBSTEPS] = True
    frames[(samples+SUBSTEPS-1)//SUBSTEPS] = True
    for i, _, channel in contact_regressions(before_contact, after_contact):
        frames[i] = True
        if channel == 'slip' and i:
            frames[i-1] = True
    return frames


def bounded_lift(parent, seed, source_z, frozen, fps):
    """Limit vertical translation to the existing source-relative step corridor."""
    offsets = np.maximum(0., seed-parent)
    offsets[frozen] = 0.
    step = np.diff(parent)-np.diff(source_z)
    span = np.maximum(ROOT_CORRECTION_SPEED_M_S/fps, np.abs(step))
    for i in range(1, len(offsets)):
        offsets[i] = min(offsets[i], offsets[i-1]+span[i-1]-step[i-1])
    for i in range(len(offsets)-2,-1,-1):
        offsets[i] = min(offsets[i], offsets[i+1]+span[i]+step[i])
    return parent+offsets


def support_lift_limit(robot, targets, rows, a, b):
    """Keep the lifted pelvis within each retained support's height allowance."""
    capacity = np.full(b-a,np.inf)
    for name, samples in support_measurements(robot,targets,rows,a,b).items():
        tolerance = (PENETRATION_M if name in ('Left_foot','Right_foot')
                     else targets['body_contact_height_tolerance_m'])
        for i, height, _, _ in samples:
            capacity[i-a] = min(capacity[i-a],max(tolerance,abs(height))-height)
    return capacity


def check_candidate(robot, targets, model, parent, candidate, lo, hi, before_geometry):
    """Compare each affected body, anchor and collision pair, including joins."""
    from .unified_repair import dense_geometry

    sl = slice(lo, hi)
    old_distance, old_floor, old_contacts = before_geometry
    proposed = candidate[sl].astype(np.float32).astype(float)
    old = parent[sl].astype(np.float32).astype(float)
    if not np.isfinite(proposed).all() or np.any(proposed[:, 3:] < robot.lower[3:]-ROUNDING_M) or np.any(proposed[:, 3:] > robot.upper[3:]+ROUNDING_M):
        return 'joint_limits_or_nonfinite', {}
    if np.any(np.abs(np.diff(proposed[:, 3:], axis=0))*targets['fps'] > robot.speed_limits+1e-4):
        return 'joint_speed_or_join', {}
    source_z = np.diff(targets['root_p'][sl, 2])
    rate = lambda pose: np.abs(np.diff(pose[:, 2])-source_z)*targets['fps']
    if np.any(rate(proposed) > np.maximum(rate(old), ROOT_CORRECTION_SPEED_M_S)+.0002):
        return 'root_z_continuity_or_join', {}
    contacts = contact_errors(robot, targets, candidate.astype(np.float32).astype(float), lo, hi)
    gaps, floor = dense_geometry(robot, model, {'root_r': targets['root_r'][sl]}, proposed)
    tol = model.cfg['penetration_tolerance_m']
    stats = dict(minimum_floor_m=float(floor.min()), collision_samples=int(np.any(gaps < -tol, axis=1).sum()),
                 max_foot_xy_m=max((float(v[0]) for k,v in contacts.items() if k[1].endswith('_foot') and k[2]=='xy'),default=0.),
                 max_foot_slip_m=max((float(v[0]) for k,v in contacts.items() if k[2]=='slip'),default=0.))
    if np.any((gaps < -tol) & ((old_distance >= -tol) | (gaps < old_distance-ROUNDING_M))):
        return 'new_or_deeper_collision', stats
    if np.any(floor < np.minimum(-PENETRATION_M, old_floor)-ROUNDING_M):
        return 'new_or_deeper_floor_penetration', stats
    bad = contact_regressions(old_contacts, contacts)
    if bad:
        return 'support_regression:'+bad[0][1]+':'+bad[0][2], stats
    floor_gain = (old_floor < -PENETRATION_M) & (floor > old_floor+ROUNDING_M)
    contact_gain = contact_regressions(contacts, old_contacts)
    return (None if floor_gain.any() or contact_gain else 'no_contact_improvement'), stats


def reland(robot, targets, landed, legs, cfg, collision_cfg=None, *, smooth_lifts=False):
    """Repair contiguous new regressions before arms get the fixed lower body."""
    from .adaptive_refine import MinimumChangeObjective
    from .local_refine import fit_window
    from .unified_repair import dense_geometry

    n = len(legs)
    old_floor = floor_samples(robot, targets, landed)
    leg_floor = floor_samples(robot, targets, legs)
    before = contact_errors(robot, targets, landed, 0, n)
    after = contact_errors(robot, targets, legs, 0, n)
    protected = protected_height(targets, n)
    active = contact_mask(targets, n)
    suspect = affected_frames(old_floor, leg_floor, before, after)
    cores = runs(suspect & active & ~protected)
    report = dict(policy=POLICY, executed=True, candidate_frames=int(suspect.sum()), segments=[])
    if not cores:
        return legs, report
    model = CollisionModel(robot, dict(collision_cfg or {}, coverage='full'))
    ids = np.asarray([2]+sum([g['leg'] for g in robot.sides.values()], []))
    fixed = np.setdiff1d(np.arange(legs.shape[1]), ids)
    output = legs.copy()
    # A 100 mm clearance correction needs roughly half a second to ramp
    # within the existing source-relative Z rate.
    padding = max(3, round(.5*targets['fps']))
    for a, b in cores:
        a, b = int(a), int(b)
        lo, hi = max(0, a-padding), min(n, b+padding)
        read_lo, read_hi = max(0,lo-1), min(n,hi+1)
        entry = dict(core_start=a,core_end=b,start=lo,end=hi,status='rejected',attempts=[])
        report['segments'].append(entry)
        if hi-lo < 5:
            entry['reason'] = 'window_too_short'
            continue
        parent = output
        q = parent.astype(np.float32).astype(float)
        sl = slice(read_lo, read_hi)
        geometry = dense_geometry(robot, model, {'root_r': targets['root_r'][sl]}, q[sl])
        referenced = (*geometry, contact_errors(robot, targets, q, read_lo, read_hi))
        entry['before'] = dict(minimum_floor_m=float(geometry[1].min()),
                               collision_samples=int(np.any(geometry[0] < -model.cfg['penetration_tolerance_m'], axis=1).sum()))
        # Build the lift seed from the current trajectory if an earlier
        # accepted segment has already touched this window.
        current_floor = leg_floor
        if not np.array_equal(parent[lo:hi], legs[lo:hi]):
            current_floor = leg_floor.copy()
            current_floor[SUBSTEPS*read_lo:SUBSTEPS*(read_hi-1)+1] = floor_samples(
                robot, {'root_r': targets['root_r'][sl]}, parent[sl])
        lifted, lift_report = settled_seed(robot, targets, parent, current_floor, entry)
        entry['lift'] = lift_report
        window = slice(lo, hi)
        frozen = protected[window].copy()
        frozen[[0,1,-2,-1]] = True
        caps = support_lift_limit(robot,targets,parent,lo,hi)
        lifted[window,2] = bounded_lift(parent[window,2],
                                        np.minimum(lifted[window,2],parent[window,2]+caps),
                                        targets['root_p'][window,2], frozen, targets['fps'])
        objective = lambda frame: MinimumChangeObjective(parent[frame])
        for method in ('vertical','joint_fit'):
            attempt = dict(method=method,status='rejected')
            entry['attempts'].append(attempt)
            try:
                if method == 'vertical':
                    candidate = lifted.copy()
                else:
                    options = dict(cfg, _repair_deadline=time.monotonic()+6.,
                                   _coupled_swept_support=True, _initial_leg_collision_cfg=model.cfg,
                                   _source_relative_root_z_rate=ROOT_CORRECTION_SPEED_M_S)
                    candidate,_ = fit_window(robot,targets,parent,options,lo,hi,iterations=6,ids_override=ids,
                                              objective_factory=objective,local_checks=True,allow_support=True,seed=lifted)
                candidate[protected] = parent[protected]
                candidate[[lo,lo+1,hi-2,hi-1]] = parent[[lo,lo+1,hi-2,hi-1]]
                if smooth_lifts:
                    from .lift_continuity import smooth_lift_correction
                    candidate = smooth_lift_correction(targets,parent,candidate,lo+1,hi-1)
                    attempt['lift_continuity'] = True
                if not np.array_equal(candidate[:,fixed],parent[:,fixed]):
                    raise ValueError('protected_root_xy_waist_or_arms_changed')
                reason, stats = check_candidate(robot, targets, model, parent, candidate, read_lo, read_hi, referenced)
                attempt['after'] = stats
                if reason:
                    raise ValueError(reason)
            except (ValueError, RuntimeError, TimeoutError, OSQPException) as error:
                attempt['reason'] = str(error)
            else:
                attempt['status'] = 'kept'
                entry.update(status='kept', selected=method)
                output = candidate
                break
    return output, report
