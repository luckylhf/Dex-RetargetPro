"""Budgeted 0.8.4-style candidate with fixed joins and staged movable chains.

The reference for a window is the current accepted trajectory. Both ends,
protected source frames, root XY and root attitude are immutable. Constraints
linearize the actual robot surfaces and material anchors at each SQP step.
The dispatcher owns independent nonlinear, float32, eight-subdivision admission.
"""
import time
from collections import Counter

import mujoco
import numpy as np
import osqp
from mink.limits.collision_avoidance_limit import compute_contact_normal_jacobian
from scipy import sparse
from scipy.spatial.transform import Rotation

from .airborne import protected_height
from .palm_constraints import SourcePalms
from .refine_routing import collision_groups, descendant_bodies
from .robot_profile import WAIST_JOINTS
from .support_quality import preserve_contact


def _check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('protected_collision_window_deadline')


class _Rows:
    """Collect Jacobians supported on one frame or two interpolated frames."""
    def __init__(self, frames, width):
        self.size = frames * width
        self.width = width
        self.rows, self.cols, self.values = [], [], []
        self.low, self.high = [], []

    def add(self, jac, low, high, j, alpha=0.):
        nz = np.flatnonzero(abs(jac) > 1e-12)
        if not len(nz):
            return False
        index = len(self.low)
        self.rows.extend([index] * len(nz))
        self.cols.extend(j * self.width + nz)
        self.values.extend((1 - alpha) * jac[nz])
        if alpha:
            self.rows.extend([index] * len(nz))
            self.cols.extend((j + 1) * self.width + nz)
            self.values.extend(alpha * jac[nz])
        self.low.append(low)
        self.high.append(high)
        return True

    def matrix(self):
        return sparse.csc_matrix((self.values, (self.rows, self.cols)),
                                 shape=(len(self.low), self.size))


def _low_point(robot, body, hull):
    local = hull[np.argmin(hull @ robot.rot(body)[2])]
    return local, robot.point(body, local)


def _foot(robot, side, values, frame):
    body = robot.sides[side]['foot']
    pivot = values['pivots'][frame]
    xy = robot.point(body, pivot)[:2] - values['anchors'][frame, :2]
    low = robot.lowest(side)
    return body, pivot, low, np.r_[xy, robot.point(body, low)[2]]


def _samples(robot, targets, rows, model, start, end, deadline):
    rotations = targets['root_r'][start:end]
    rv = Rotation.from_matrix(rotations[:-1].transpose(0, 2, 1) @ rotations[1:]).as_rotvec()
    samples = []
    for j, matrix in enumerate(rotations):
        for alpha in ((0., .5) if j < len(rv) else (0.,)):
            _check_time(deadline)
            frame = start + j
            rot = matrix if not alpha else matrix @ Rotation.from_rotvec(alpha * rv[j]).as_matrix()
            pose = rows[frame] if not alpha else (rows[frame] + rows[frame + 1]) * .5
            model.update(pose, rot)
            heights = {b: _low_point(robot, b, s['hull'])[1][2]
                       for b, s in robot.surface_models.items()}
            feet, contacts = {}, []
            if not alpha:
                for side, v in targets['sides'].items():
                    if v['contact'][frame]:
                        feet[side] = _foot(robot, side, v, frame)[-1]
                for c in targets.get('body_contacts', ()):
                    if preserve_contact(c, frame):
                        body = c['body']
                        hull = c.get('robot_hull', robot.surface_models[body]['hull'])
                        _, point = _low_point(robot, body, hull)
                        contacts.append((c, point))
            samples.append((j, alpha, rot, model.distances(), heights, feet, contacts))
    return samples


def _difference(frames, width):
    first = sparse.diags([-1., 1.], [0, 1], shape=(frames - 1, frames))
    second = sparse.diags([1., -2., 1.], [0, 1, 2], shape=(frames - 2, frames))
    return tuple(sparse.kron(x, sparse.eye(width), format='csc') for x in (first, second))


def _stage(robot, targets, base, current, model, start, samples, ids, owned, fixed, name, deadline):
    n, w = len(current), len(ids)
    identity = sparse.eye(n * w, format='csc')
    first, second = _difference(n, w)
    # Regularize changes relative to the accepted trajectory, including their
    # velocity and acceleration. Raw source velocities need not be small.
    regularizer = (identity + 3 * first.T @ first + 70 * second.T @ second).tocsc()
    joint_speed = np.r_[np.full(3, np.inf), robot.speed_limits / targets['fps']][ids]
    diff_base = np.diff(base[:, ids], axis=0)
    speed = np.maximum(abs(diff_base), joint_speed)
    knees = [g['leg'][3] for g in robot.sides.values()]
    bounds_lo = np.broadcast_to(robot.lower[ids], (n, w)).copy()
    for k, column in enumerate(ids):
        if column in knees:
            bounds_lo[base[:, column] >= 0., k] = 0.
    bodies = (set(robot.surface_models) if 2 in ids else
              descendant_bodies(robot, ids[ids >= 3]) & robot.surface_models.keys())
    arms = [s for s, g in robot.sides.items() if set(g['arm']) & set(ids)
            and 'hand_forward' in targets['sides'][s]]
    palms = None
    if arms:
        # SourcePalms indexes both source targets and reference at global frames.
        sl = {**targets, 'root_r': targets['root_r'][start:start+n],
              'sides': {s: {k: targets['sides'][s][k][start:start+n]
                            for k in ('hand_forward','hand_width')} for s in arms}}
        palms = SourcePalms(robot, sl, base, 0, n, arms, ids)
    normal = np.empty(3)
    jac1 = np.empty((3, model.m.nv))
    jac2 = np.empty_like(jac1)
    witnesses = np.empty(6)
    status = dict(name=name, axes=len(ids), iterations=0, constraints={}, outcome='already_clear')
    counts = Counter()
    penetration = model.cfg['penetration_tolerance_m']
    # Stages with clean geometry are certified against the current pose, then
    # their joints stay at the accepted values. No unconstrained refit.
    needs_fit = any(np.any(sample[3][owned] < -penetration) for sample in samples)
    if name == 'waist':
        for j in np.flatnonzero(~fixed):
            robot.set(current[j], targets['root_r'][start+j])
            delta = targets['torso_r'][start+j] @ robot.rot(robot.torso).T
            if Rotation.from_matrix(delta).magnitude() > np.deg2rad(20):
                needs_fit = True
                break
    if not needs_fit and np.array_equal(current, base):
        return status
    for iteration in range(3):
        hard, objective = _Rows(n,w), _Rows(n,w)
        priority = 100.
        for j, alpha, rot, baseline_dist, baseline_height, feet, contacts in samples:
            if fixed[j] and (not alpha or fixed[j+1]):
                continue
            _check_time(deadline)
            frame = start+j
            row = current[j] if not alpha else (1-alpha)*current[j] + alpha*current[j+1]
            model.update(row, rot)
            dist = baseline_dist if iteration == 0 and np.array_equal(base[j:j+2], current[j:j+2]) else model.distances()
            data = model.configuration.data
            for k in np.flatnonzero(dist < .009):
                ga, gb = map(int, model.pairs[k])
                d = mujoco.mj_geomDistance(model.m, data, ga, gb, model.cfg['detection_m'], witnesses)
                jac = compute_contact_normal_jacobian(model.m, data, ga, gb, witnesses,
                                                       normal, jac1, jac2)[robot.cols[ids]]
                if d < 0:
                    jac = -jac
                floor = min(.0001, baseline_dist[k])
                counts['collision_bounds'] += hard.add(jac, floor - d - 2e-7, np.inf, j, alpha)
                if owned[k] and baseline_dist[k] < -penetration:
                    gain = priority * (1 if not alpha else .65)
                    counts['collision_objectives'] += objective.add(gain*jac, gain*(.0018-d), 0, j, alpha)
            for body in bodies:
                hull = robot.surface_models[body]['hull']
                heights = hull @ robot.rot(body)[2] + robot.point(body)[2]
                low = np.min(heights)
                if low > .018:
                    continue
                # A flat sole can switch witnesses under an arbitrarily small
                # rotation. Include extremal corners from the whole low band.
                near = np.flatnonzero(heights < low + .004)
                xy = hull[near] @ robot.rot(body)[:2].T
                extremes = np.unique(np.r_[np.argmin(heights),
                                            near[np.argmin(xy,axis=0)], near[np.argmax(xy,axis=0)]])
                bound = min(0., baseline_height[body])
                for vertex in extremes:
                    jac = robot.jac(body, hull[vertex])[0][2,ids]
                    counts['floor_midpoint' if alpha else 'floor_native'] += hard.add(
                        jac, bound - heights[vertex] - 2e-7, np.inf, j, alpha)
            if alpha:
                continue
            for side, reference in feet.items():
                body, pivot, low, error = _foot(robot, side, targets['sides'][side], frame)
                if body not in bodies:
                    continue
                J = np.vstack([robot.jac(body,pivot)[0][:2,ids], robot.jac(body,low)[0][2,ids]])
                # Bound the correction to the signed anchor offset at the input
                # pose. A tight material-point corridor prevents anchor drift
                # when source inference already misplaced a support foot.
                for axis, jac in enumerate(J):
                    center = reference[axis] - error[axis]
                    counts['foot_anchor_XYZ'] += hard.add(jac, center-1e-5, center+1e-5, j)
                    objective.add(55*jac, 55*center, 0, j)
            for c, prior in contacts:
                body = c['body']
                if body not in bodies:
                    continue
                hull = c.get('robot_hull', robot.surface_models[body]['hull'])
                local, point = _low_point(robot, body, hull)
                J = robot.jac(body,local)[0][:,ids]
                axes = (0,1,2) if c['mode'][frame] == 1 else (2,)
                for axis in axes:
                    diff = prior[axis]-point[axis]
                    counts['body_support'] += hard.add(J[axis], diff-.00015, diff+.00015, j)
            if palms is not None:
                C, hi, _ = palms.evaluate(row, j)
                for jac, bound in zip(C, hi):
                    hard.add(jac, -np.inf, bound, j)
                # A small angular-change cost also helps preserve the source
                # palm for the independent nonlinear audit near its boundary.
                for side in arms:
                    J = robot.jac(robot.sides[side]['tcp'])[1][:,ids]
                    for jac in J:
                        objective.add(2*jac, 0., 0, j)
            if name == 'waist':
                rv = Rotation.from_matrix(targets['torso_r'][frame] @ robot.rot(robot.torso).T).as_rotvec()
                if np.linalg.norm(rv) > np.deg2rad(20):
                    for jac, axis_target in zip(robot.jac(robot.torso)[1][:,ids], rv):
                        objective.add(10*jac, 10*axis_target, 0, j)
        if not objective.low or (name != 'waist' and not counts['collision_objectives']):
            break
        q = current[:,ids]
        cap = np.where(ids == 2, .005, .055)
        lo, hi = np.maximum(-cap, bounds_lo-q), np.minimum(cap, robot.upper[ids]-q)
        lo[fixed], hi[fixed] = 0., 0.
        dq = np.diff(q, axis=0)
        lo_v, hi_v = -speed-dq, speed-dq
        if 2 in ids:
            # Keep the accepted root's vertical step to within 0.25 mm.
            z = int(np.flatnonzero(ids == 2)[0])
            lo_v[:,z], hi_v[:,z] = diff_base[:,z]-dq[:,z]-.00025, diff_base[:,z]-dq[:,z]+.00025
        H, C = objective.matrix(), hard.matrix()
        A = sparse.vstack((identity, first, C), format='csc')
        P = (regularizer + H.T@H).tocsc()
        linear = regularizer @ (q-base[:,ids]).ravel() - H.T @ np.asarray(objective.low)
        _check_time(deadline)
        solver = osqp.OSQP()
        solver.setup(P=P,q=linear,A=A,l=np.r_[lo.ravel(),lo_v.ravel(),hard.low],
                     u=np.r_[hi.ravel(),hi_v.ravel(),hard.high],verbose=False,
                     eps_abs=2e-7,eps_rel=1e-5,max_iter=4000,
                     time_limit=max(.002,deadline-time.monotonic()))
        result = solver.solve()
        status['qp_status'] = result.info.status
        status['primal_residual'] = float(result.info.prim_res)
        if result.x is None or result.info.status_val not in (1,2):
            if result.info.status_val != 7 or result.x is None or result.info.prim_res > 2e-5:
                status['outcome'] = 'qp_stopped'
                break
        step = result.x.reshape(n,w)
        step[fixed] = 0.
        current[:,ids] += step
        status['iterations'] += 1
        status['outcome'] = 'optimized'
        if abs(step).max() < 1e-4:
            break
    status['constraints'] = dict(counts)
    return status


def propose(robot, targets, rows, full_model, start, end, deadline, *, protected=None, leg_columns=None):
    """Return a whole-motion proposal for the requested half-open segment."""
    if not 0 <= start < end <= len(rows):
        raise ValueError('invalid protected collision segment')
    stages = []
    result = np.array(rows, dtype=float, copy=True)
    order = ['waist','pelvis_z_and_legs','locked_lower_body_arms'] if leg_columns is None else ['pelvis_z_and_legs']
    info = dict(stage_order=order,
                stages=stages, start=int(start),end=int(end),linearization_subdivisions=2)
    if end-start < 3:
        return result, info
    fixed = (protected_height(targets,len(rows)) if protected is None else np.asarray(protected,bool))
    fixed = fixed[start:end].copy()
    fixed[[0,-1]] = True
    if fixed.all():
        return result, info
    _check_time(deadline)
    base = np.array(rows[start:end], dtype=float, copy=True)
    candidate = result[start:end]
    samples = _samples(robot, targets, rows, full_model, start, end, deadline)
    groups = collision_groups(full_model)
    if leg_columns is None:
        waist = [3+robot.names.index(name) for name in WAIST_JOINTS]
        legs = [2]+[i for side in robot.sides.values() for i in side['leg']]
        arms = [i for side in robot.sides.values() for i in side['arm']]
        axes = zip(order,(waist,legs,arms),('body_collision','leg_collision','collision'))
    else:
        axes = [(order[0],leg_columns,'leg_collision')]
    for name, ids, group in axes:
        owned = np.zeros(len(full_model.pairs),bool)
        owned[groups[group]] = True
        stages.append(_stage(robot, targets, base, candidate, full_model, start, samples,
                             np.asarray(ids), owned, fixed, name, deadline))
    info['changed_frames'] = int(np.any(candidate != base,axis=1).sum())
    return result, info
