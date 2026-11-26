"""Lift-and-settle candidate for short, grounded multi-contact intervals."""
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .airborne import protected_height
from .support_quality import preserve_contact
from .temporal_constraints import root_speed_bounds
from .v8.math3d import skew
from .v8.multicontact import contact_local
from .v8.support_acceptance import SUPPORT_ACCEPTANCE_M


RECIPE = 'pelvis_contact'
POLICY = 'lift_settle_pelvis_contact_v1'
BODY_DRIFT_M = .001
MAX_STEP_RAD = np.deg2rad(1.)
MAX_CORE_FRAMES = 8
# Search within the existing source-relative orientation gate.
SOURCE_CAP_RAD = np.deg2rad(json.loads(Path(__file__).with_name('quality_pose.json').read_text())
                           ['root_orientation_max_deg'])
POSE_CAP_RAD = SOURCE_CAP_RAD - np.deg2rad(.025)


def offset_jacobian(offset):
    """Map rotation-vector derivatives to a world-frame angular increment."""
    angle = np.linalg.norm(offset)
    cross = skew(offset)
    squared = cross @ cross
    if angle < 1e-5:
        return np.eye(3) + .5 * cross + squared / 6.
    return (np.eye(3) + (1. - np.cos(angle)) / angle**2 * cross +
            (angle - np.sin(angle)) / angle**3 * squared)


def _surface_point(robot, body, hull):
    local = hull[np.argmin(hull @ robot.rot(body)[2])]
    return local, robot.point(body, local)


def _lowest_surface(robot, surfaces):
    return min(((body, *_surface_point(robot, body, hull)) for body, hull in surfaces),
               key=lambda witness: witness[2][2])


def _window(robot, targets, rows, job):
    a, b = job['start'], job['end']
    start, end = max(a + 2, job['core_start']), min(b - 2, job['core_end'])
    protected = protected_height(targets, len(rows))
    foot_mask = np.logical_or.reduce([values['contact'] > 0 for values in targets['sides'].values()])
    scored = []
    for i in range(start, end):
        if protected[i] or not foot_mask[i]:
            continue
        contacts = [c for c in targets.get('body_contacts', ()) if preserve_contact(c, i)]
        if not contacts:
            continue
        robot.set(rows[i], targets['root_r'][i])
        floor = min(_surface_point(robot, body, s['hull'])[1][2]
                    for body, s in robot.surface_models.items())
        bad = 0.
        for contact in contacts:
            body = contact['body']
            point = robot.point(body, contact_local(robot, contact))
            bad = max(bad, abs(point[2]) - targets['body_contact_height_tolerance_m'])
            if contact['mode'][i] == 1:
                error = np.linalg.norm(point[:2] - contact['anchors'][i, :2])
                bad = max(bad, error - targets['body_contact_anchor_tolerance_m'])
        scored.append((max(0., -floor - SUPPORT_ACCEPTANCE_M, bad), i))
    if not scored or max(scored)[0] <= 0.:
        raise ValueError('pelvis_contact_no_grounded_defect')
    _, center = max(scored)
    left = max(a, center - MAX_CORE_FRAMES // 2 - 2)
    right = min(b, center + MAX_CORE_FRAMES // 2 + 2)
    if right - left < 5:
        raise ValueError('pelvis_contact_window_too_short')
    return left, right


def propose(robot, targets, rows, job, deadline, *, restore_violations=False, foot_geometry=False):
    """Jointly fit root position, small root rotations and support limb hinges."""
    if foot_geometry:
        a = max(job['start'], job['core_start'] - 3)
        b = min(job['end'], job['core_end'] + 3)
    else:
        a, b = _window(robot, targets, rows, job)
    if b-a<5:
        raise ValueError('pelvis_contact_window_too_short')
    reference = rows[a:b]
    rotations = targets['root_r'][a:b]
    source = targets.get('source_root_r', targets['root_r'])[a:b]
    fixed = protected_height(targets, len(rows))[a:b].copy()
    fixed[[0, 1, -2, -1]] = True
    if np.all(fixed):
        raise ValueError('pelvis_contact_window_protected')
    arms = []
    for side, group in robot.sides.items():
        bodies = {group[k] for k in ('shoulder', 'elbow', 'forearm', 'wrist', 'hand', 'tcp')}
        if not foot_geometry and any(c['body'] in bodies and any(preserve_contact(c, i) for i in range(a, b))
               for c in targets.get('body_contacts', ())):
            arms.extend(group['arm'])
    ids = np.asarray([0, 1, 2] + sum([g['leg'] for g in robot.sides.values()], []) + arms)
    n, width = len(reference), len(ids) + 3
    template = np.column_stack((reference[:, ids], np.zeros((n, 3))))
    writable = np.broadcast_to(~fixed[:, None], template.shape).copy()
    if foot_geometry:
        writable[:, :2] = False
        writable[:, -3:] = False
    free = np.flatnonzero(writable.ravel())
    expected, limits = [], []
    surfaces = [(body, v['hull']) for body, v in robot.surface_models.items()]
    height = targets['body_contact_height_tolerance_m']
    horizontal = targets['body_contact_anchor_tolerance_m']
    for i, row, rotation in zip(range(a, b), reference, rotations):
        robot.set(row, rotation)
        feet, support = [], []
        for side, g in robot.sides.items():
            f = targets['sides'][side]
            if f['contact'][i] > 0:
                anchor = f['anchors'][i, :2]
                pivot = f['pivots'][i]
                limit = .9 * SUPPORT_ACCEPTANCE_M
                if restore_violations or foot_geometry:
                    point = robot.point(g['foot'], pivot)
                    floor = _surface_point(robot, g['foot'], g['hull'])[1][2]
                    error = np.linalg.norm(np.r_[point[:2] - anchor, floor])
                    if error <= SUPPORT_ACCEPTANCE_M:
                        limit = max(.99 * SUPPORT_ACCEPTANCE_M, error)
                feet.append((g['foot'], g['hull'], pivot, anchor, limit))
        for contact in targets.get('body_contacts', ()):
            if not preserve_contact(contact, i):
                continue
            body = contact['body']
            hull = np.asarray(contact.get('robot_hull', robot.surface_models[body]['hull']))
            p = _surface_point(robot, body, hull)[1]
            anchor = contact['anchors'][i, :2] if contact['mode'][i] == 1 else None
            z = abs(p[2])
            z_limit = min(height, z + BODY_DRIFT_M * .95) if z <= height else z
            if restore_violations and z > height:
                z_limit = height - .00010
            xy_limit = None
            if anchor is not None:
                xy = np.linalg.norm(p[:2] - anchor)
                xy_limit = min(horizontal, xy + BODY_DRIFT_M * .95) if xy <= horizontal else xy
                if restore_violations and xy > horizontal:
                    xy_limit = horizontal - .00010
            support.append((body, hull, z_limit, anchor, xy_limit))
        expected.append((feet, support))
        angle = Rotation.from_matrix(rotation @ source[i - a].T).magnitude()
        limits.append(max(POSE_CAP_RAD, angle))

    deltas = Rotation.from_matrix(rotations[:-1].transpose(0, 2, 1) @ rotations[1:]).as_rotvec()
    samples, required_lift = [], np.zeros(n)
    for j in range(n):
        for k in range(8 if j < n - 1 else 1):
            u = k / 8
            rr = rotations[j] if not k else rotations[j] @ Rotation.from_rotvec(u * deltas[j]).as_matrix()
            terms = ((j, 1.),) if not k else ((j, 1. - u), (j + 1, u))
            x = sum(coef * reference[i] for i, coef in terms)
            robot.set(x, rr)
            _, _, minimum = _lowest_surface(robot, surfaces)
            floor_limit = -.00085
            if (restore_violations or foot_geometry) and minimum[2] >= -SUPPORT_ACCEPTANCE_M:
                floor_limit = min(-.00099,minimum[2])
            needed = max(0., floor_limit - minimum[2]) if restore_violations or foot_geometry else max(0., .0004 - minimum[2])
            for i, _ in terms:
                required_lift[i] = max(required_lift[i], needed)
            samples.append((j, k, rr, terms, floor_limit))
    lifted = template.copy()
    # Raise the entire window above the floor, then descend until each
    # sampled interval is just clear. This is a seed, never an export.
    lift = float(required_lift.max())
    lifted[:, 2] += lift
    lifted[:, 2] -= lift - required_lift
    lower = np.r_[robot.lower[ids], np.full(3, -POSE_CAP_RAD)]
    upper = np.r_[robot.upper[ids], np.full(3, POSE_CAP_RAD)]
    knee_ids = {g['leg'][3] for g in robot.sides.values()}
    lower = np.broadcast_to(lower, template.shape).copy()
    for col, joint in enumerate(ids):
        if joint in knee_ids:
            lower[:, col] = np.where(template[:, col] >= 0., np.maximum(lower[:, col], 0.), lower[:, col])
    lo = lower.ravel()[free]
    hi = np.broadcast_to(upper, template.shape).ravel()[free]
    fps = targets['fps']
    speed_limits = np.r_[np.full(3, np.inf), robot.speed_limits][ids[3:]] / fps
    prior_steps = np.diff(template, axis=0)
    speeds = np.maximum(speed_limits, np.abs(prior_steps[:, 3:-3]))
    root_bounds = np.maximum(root_speed_bounds(targets['root_p'][a:b], fps) / fps,
                             np.linalg.norm(prior_steps[:, :3], axis=1))
    weight = np.r_[np.full(2, 5.), 2., np.ones(len(ids) - 3), np.full(3, 2.)]
    cached = None

    def evaluate(v):
        nonlocal cached
        if time.monotonic() >= deadline:
            raise TimeoutError('pelvis_contact_deadline')
        if cached is not None and np.array_equal(cached[0], v):
            return cached[1:]
        values = template.copy()
        values.ravel()[free] = v
        result, jac = [], []

        def add(value, gradient, terms):
            value, gradient = np.atleast_1d(value), np.atleast_2d(gradient)
            block = np.zeros((len(value), n * width))
            for i, factor in terms:
                block[:, i * width:(i + 1) * width] = factor * gradient
            result.extend(value)
            jac.append(block[:, free])

        def ball(point, jacobian, limit, terms):
            magnitude = np.linalg.norm(point)
            add(limit - magnitude, -(point @ jacobian) / max(magnitude, 1e-12), terms)

        eye = np.eye(width)
        for j, (old, new) in enumerate(zip(values[:-1], values[1:])):
            if fixed[j] and fixed[j + 1]:
                continue
            step = new - old
            terms = ((j, -1.), (j + 1, 1.))
            add(speeds[j] - step[3:-3], -eye[3:-3], terms)
            add(speeds[j] + step[3:-3], eye[3:-3], terms)
            ball(step[:3], eye[:3], root_bounds[j], terms)
            ball(step[-3:], eye[-3:], MAX_STEP_RAD, terms)

        for j, k, rr, terms, floor_limit in samples:
            if all(fixed[i] for i, _ in terms):
                continue
            val = sum(coef * values[i] for i, coef in terms)
            row = sum(coef * reference[i] for i, coef in terms)
            row[ids] = val[:-3]
            rotation = Rotation.from_rotvec(val[-3:]).as_matrix() @ rr
            robot.set(row, rotation)
            angular = offset_jacobian(val[-3:])

            def point_jac(body, local, p):
                return np.c_[robot.jac(body, local)[0][:, ids], -skew(p - row[:3]) @ angular]

            # All surfaces share a plane, so their lowest witness suffices
            # for the search. The export guard still checks every body.
            body, local, p = _lowest_surface(robot, surfaces)
            add(p[2] - floor_limit, point_jac(body, local, p)[2], terms)
            if k:
                continue
            ball(val[-3:], eye[-3:], POSE_CAP_RAD, terms)
            relative = Rotation.from_matrix(rotation @ source[j].T).as_rotvec()
            add(limits[j] - np.linalg.norm(relative),
                np.r_[np.zeros(width - 3), -(relative @ angular) / max(1e-12, np.linalg.norm(relative))], terms)
            for body, hull, pivot, anchor, limit in expected[j][0]:
                local, p = _surface_point(robot, body, hull)
                foot = robot.point(body, pivot)
                error = np.r_[foot[:2] - anchor, p[2]]
                g = np.vstack((point_jac(body, pivot, foot)[:2], point_jac(body, local, p)[2]))
                ball(error, g, limit, terms)
            for body, hull, z_limit, anchor, xy_limit in expected[j][1]:
                local, p = _surface_point(robot, body, hull)
                g = point_jac(body, local, p)
                add([z_limit - p[2], z_limit + p[2]], [-g[2], g[2]], terms)
                if anchor is not None:
                    ball(p[:2] - anchor, g[:2], xy_limit, terms)
        cached = (v.copy(), np.asarray(result), np.vstack(jac))
        return cached[1:]

    reference_v = template.ravel()[free]
    weights_v = np.tile(weight, n)[free]

    def objective(v):
        d = (v - reference_v) * weights_v
        cost = d @ d
        gradient = d * weights_v
        if lift > 0.:
            correction = np.zeros((n,width))
            correction.ravel()[free] = d
            curvature = np.diff(correction,n=2,axis=0)
            curvature_gradient = np.zeros_like(correction)
            curvature_gradient[:-2] += curvature
            curvature_gradient[1:-1] -= 2.*curvature
            curvature_gradient[2:] += curvature
            cost += .25 * np.sum(curvature**2)
            gradient += .25 * curvature_gradient.ravel()[free] * weights_v
        return .5 * cost, gradient

    guess = np.clip(lifted.ravel()[free], lo, hi)
    fit = minimize(objective, guess, jac=True, method='SLSQP', bounds=list(zip(lo, hi)),
                   constraints=[dict(type='ineq', fun=lambda v: evaluate(v)[0], jac=lambda v: evaluate(v)[1])],
                   options=dict(maxiter=90, ftol=1e-9))
    violation = max(0., -float(evaluate(fit.x)[0].min(initial=0.)))
    if violation > 2e-5 or not np.isfinite(fit.x).all():
        raise ValueError('pelvis_contact_no_feasible_window')
    fitted = template.copy()
    fitted.ravel()[free] = fit.x
    candidate = rows.copy()
    candidate[a:b, ids] = fitted[:, :-3]
    rr = targets['root_r'].copy()
    modified = np.flatnonzero(np.any(fitted[:, -3:] != 0., axis=1))
    rr[a + modified] = Rotation.from_rotvec(fitted[modified, -3:]).as_matrix() @ rotations[modified]
    return candidate, rr, dict(policy=POLICY, window=[a, b], lift_m=lift,
                               max_rotation_deg=float(np.rad2deg(np.linalg.norm(fitted[:, -3:], axis=1)).max()),
                               iterations=int(fit.nit))


def check_motion(targets, updated, before, after, old_m, new_m, old_cache, new_cache, old_q, new_q, a, b):
    """Check the rounded export, including both joins and per-foot support."""
    old_r, new_r = targets['root_r'], updated['root_r']
    if not (np.array_equal(old_r[:a], new_r[:a]) and np.array_equal(old_r[b:], new_r[b:])):
        return 'outside_window_rotation_changed'
    protected = protected_height(targets, len(before))
    if not (np.array_equal(before[protected], after[protected]) and
            np.array_equal(old_r[protected], new_r[protected])):
        return 'protected_contact_changed'
    lo, hi = max(0, a - 1), min(len(before), b + 1)
    offset = Rotation.from_matrix(new_r[lo:hi] @ old_r[lo:hi].transpose(0, 2, 1)).as_rotvec()
    if np.any(np.linalg.norm(offset, axis=1) > POSE_CAP_RAD + 2e-6):
        return 'pelvis_rotation_budget'
    if np.any(np.linalg.norm(np.diff(offset, axis=0), axis=1) > MAX_STEP_RAD + 2e-6):
        return 'pelvis_rotation_seam'
    prior_speed = np.linalg.norm(np.diff(before[lo:hi, :3], axis=0), axis=1) * targets['fps']
    speed = np.linalg.norm(np.diff(after[lo:hi, :3], axis=0), axis=1) * targets['fps']
    bound = root_speed_bounds(targets['root_p'][lo:hi], targets['fps'])
    if np.any(speed > np.maximum(prior_speed, bound) + 1e-5):
        return 'pelvis_translation_speed'
    old_torso, new_torso = old_m['torso'][a:b], new_m['torso'][a:b]
    if np.any(new_torso > np.minimum(np.maximum(old_torso, 20.), old_torso + 5.) + .02):
        return 'torso_frame_regression'
    source = targets.get('source_root_r', old_r)[a:b]
    orientation = lambda rr: Rotation.from_matrix(rr[a:b] @ source.transpose(0, 2, 1)).magnitude()
    if np.any(orientation(new_r) > np.maximum(SOURCE_CAP_RAD, orientation(old_r)) + 1e-7):
        return 'source_root_orientation'
    old_xyz = old_cache['frames']['contact_xyz']
    new_xyz = new_cache['frames']['contact_xyz']
    for side, previous in old_xyz.items():
        for old, new in zip(previous[a:b], new_xyz[side][a:b]):
            if old is not None and (new is None or np.linalg.norm(new) >
                                    max(100. * SUPPORT_ACCEPTANCE_M, np.linalg.norm(old)) + 1e-5):
                return 'foot_anchor_regression:' + side

    def impulse_steps(quality):
        return {(event['signal'], i) for event in quality.get('root_discontinuities') or []
                for i in range(*event['intervals'])}

    if impulse_steps(new_q) - impulse_steps(old_q):
        return 'new_root_impulse'
    return None

