"""Smooth corrections on lifted contact spans while preserving support."""

from collections import Counter

import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation

from dex_frax.airborne import protected_height
from dex_frax.frax_ground_constraints import ROOT_CORRECTION_SPEED_M_S, support_targets
from dex_frax.post_leg_contact import contact_errors, contact_regressions
from dex_frax.temporal_constraints import diagnostics
from dex_frax.temporal_quality import assess
from dex_frax.torso_branch import floor_samples


def contact_rows(robot, targets, poses, columns):
    """Linearize stance anchors and sole height around the landing seed."""
    count = len(poses)
    width = len(columns)
    value_blocks, column_blocks, row_blocks = [], [], []
    low, high = [], []

    def append(frame, jacobian, error, tolerance, previous=None):
        error = np.atleast_1d(error)
        for f, derivative in ((frame, jacobian), (frame - 1, previous)):
            if derivative is None:
                continue
            derivative = np.atleast_2d(derivative)
            row, col = np.nonzero(derivative)
            value_blocks.append(derivative[row, col])
            row_blocks.append(row + len(low))
            column_blocks.append(col + f * width)
        margin = np.maximum(tolerance, abs(error))
        low.extend(-margin - error)
        high.extend(margin - error)

    feet = [(group, support_targets(targets['sides'][side], robot.lowest(side)), side)
            for side, group in robot.sides.items()]
    frozen = protected_height(targets, count)
    for group, foot, side in feet:
        last_frame = -2
        for frame in np.flatnonzero(foot['stance_active']):
            robot.set(poses[frame], targets['root_r'][frame])
            pivot = foot['stance_pivot'][frame]
            anchor = foot['stance_anchor'][frame]
            body = group['foot']
            sole = robot.lowest(side)
            point_xy = robot.point(body, pivot)[:2]
            jacobian_xy = robot.jac(body, pivot)[0][:2, columns]
            error = np.r_[point_xy - anchor[:2], robot.point(body, sole)[2] - anchor[2]]
            jacobian = np.vstack([jacobian_xy, robot.jac(body, sole)[0][2, columns]])
            # Keep the original per-axis stance box. Inherited excess may
            # shrink; holding the original pose remains a feasible start.
            if not frozen[frame]:
                append(frame, jacobian, error, .0005)
                distance = np.linalg.norm(error[:2])
                if distance > .0008:
                    append(frame, error[:2] @ jacobian_xy / distance, distance, .00085)
            if frame == last_frame + 1 and foot['stance_stable'][frame] and not (frozen[frame] and frozen[last_frame]):
                slip = point_xy - last_xy
                append(frame, jacobian_xy, slip, .0006, previous=-last_jacobian)
                distance = np.linalg.norm(slip)
                if distance > .0008:
                    unit = slip / distance
                    append(frame, unit @ jacobian_xy, distance, .00085, previous=-unit @ last_jacobian)
            last_frame, last_xy, last_jacobian = frame, point_xy, jacobian_xy
    coords = (np.concatenate(row_blocks), np.concatenate(column_blocks)) if low else ([], [])
    matrix = sparse.csc_matrix((np.concatenate(value_blocks) if low else [], coords),
                               shape=(len(low), count * width))
    return matrix, np.array(low), np.array(high)


def floor_rows(robot, targets, poses, columns):
    """Constrain near-ground witnesses throughout each original interval."""
    rotations = targets['root_r']
    heights = floor_samples(robot, targets, poses)
    indices = np.argwhere(heights < .010)
    bodies = list(robot.surface_models)
    delta = Rotation.from_matrix(rotations[:-1].transpose(0, 2, 1) @ rotations[1:]).as_rotvec()
    width = len(columns)
    jacobians = np.empty((len(indices), width))
    fractions = np.empty(len(indices))
    for index, (sample, body_index) in enumerate(indices):
        frame, part = divmod(sample, 8)
        fraction = part / 8.
        fractions[index] = fraction
        row = poses[frame]
        rotation = rotations[frame]
        if part:
            row = row * (1. - fraction) + poses[frame + 1] * fraction
            rotation = rotation @ Rotation.from_rotvec(delta[frame] * fraction).as_matrix()
        robot.set(row, rotation)
        body = bodies[body_index]
        hull = robot.surface_models[body]['hull']
        local = hull[np.argmin(hull @ robot.rot(body)[2])]
        jacobians[index] = robot.jac(body, local)[0][2, columns]
    sample_indices, body_indices = indices.T
    frame_indices = sample_indices // 8
    before = heights[sample_indices, body_indices]
    frame_cols = frame_indices[:, None] * width + np.arange(width)
    rows = np.repeat(np.arange(len(indices)), width)
    values = np.r_[(jacobians * (1. - fractions[:, None])).ravel(),
                   (jacobians * fractions[:, None]).ravel()]
    adjacent = np.minimum(frame_cols + width, len(poses) * width - 1)
    matrix = sparse.csc_matrix((values, (np.r_[rows, rows], np.r_[frame_cols.ravel(), adjacent.ravel()])),
                               shape=(len(indices), len(poses) * width))
    matrix.eliminate_zeros()
    return heights, matrix, np.minimum(-.00085, before) - before


def smooth_landing(robot, targets, raw, landed, writable):
    """Fit adjacent landing corrections together on the routed lift frames."""
    hips_and_legs = [axis for group in robot.sides.values() for axis in group['leg']]
    columns = np.array([2, *hips_and_legs])
    count, width = len(landed), len(columns)
    identity = sparse.eye(count * width, format='csc')
    step = sparse.diags([-1., 1.], [0, 1], shape=(count - 1, count), format='csc')
    curvature = sparse.diags([1., -2., 1.], [0, 1, 2], shape=(count - 2, count), format='csc')
    velocity = sparse.kron(step, sparse.eye(width), format='csc')
    # Physical units differ: one centimetre of pelvis correction has the
    # same objective scale as roughly six degrees of hinge correction.
    scale = sparse.diags([10., *np.ones(width - 1)])
    temporal = sparse.kron(curvature, scale, format='csc')
    fidelity = sparse.kron(sparse.eye(count), scale @ scale, format='csc')
    hessian = (3.0 * temporal.T @ temporal).tocsc()
    correction = (landed[:, columns] - raw[:, columns]).ravel()
    original_curvature = float(np.linalg.norm(temporal @ correction))
    lower = np.maximum(robot.lower[columns] - landed[:, columns], [-.018, *np.full(width - 1, -.12)])
    upper = np.minimum(robot.upper[columns] - landed[:, columns], [.018, *np.full(width - 1, .12)])
    lower, upper = np.minimum(lower, 0.), np.maximum(upper, 0.)
    locked = protected_height(targets, count) | ~writable
    lower[locked] = upper[locked] = 0.
    existing_step = np.diff(landed[:, columns], axis=0)
    reference_step = existing_step.copy()
    reference_step[:, 0] -= np.diff(targets['root_p'][:, 2])
    speed = np.r_[ROOT_CORRECTION_SPEED_M_S, robot.speed_limits[columns[1:] - 3]] / targets['fps']
    speed = np.maximum(speed[None, :], abs(reference_step))
    contacts, contact_low, contact_high = contact_rows(robot, targets, landed, columns)
    before_floor, floor, floor_low = floor_rows(robot, targets, landed, columns)
    constraints = sparse.vstack([identity, velocity, contacts, floor], format='csc')
    lo = np.r_[lower.ravel(), (-speed - reference_step).ravel(), contact_low, floor_low]
    hi = np.r_[upper.ravel(), (speed - reference_step).ravel(), contact_high, np.full(len(floor_low), np.inf)]
    solver = osqp.OSQP()
    solver.setup(P=sparse.triu(fidelity + hessian, format='csc'), q=hessian @ correction,
                 A=constraints, l=lo, u=hi, verbose=False, max_iter=1800,
                 eps_abs=1e-6, eps_rel=1e-6, check_termination=25)
    result = dict(policy='all_frame_landing_correction_curvature_v1', frames=count,
                  variables=count * width, constraints=constraints.shape[0],
                  protected_frames=int(locked.sum()), candidates=[])
    baseline_contacts = contact_errors(robot, targets, landed, 0, count)
    reference = diagnostics(targets)
    prior_jitter = len(assess(landed[:, 3:], targets['fps'], robot.names, reference)['violating_frames'])
    result.update(before_jitter_frames=prior_jitter, before_correction_curvature=original_curvature)
    selected = landed
    # Exact nonlinear checks can expose contacts or interpolated samples
    # missed by the tangent model. Hold those frames in the next full solve,
    # so the objective can distribute the correction at the new boundaries.
    for iteration in range(6):
        solution = solver.solve(raise_error=False)
        if solution.x is None or solution.info.status_val not in (1, 2, 7):
            result['qp_status'] = solution.info.status
            break
        delta = solution.x.reshape(count, width)
        delta[locked] = 0.
        candidate = landed.copy()
        candidate[:, columns] += delta
        check = candidate.astype(np.float32).astype(float)
        checked_contacts = contact_errors(robot, targets, check, 0, count)
        checked_floor = floor_samples(robot, targets, check)
        floor_regression = checked_floor < np.minimum(before_floor, -.001) - .00003
        body_contacts = contact_regressions(baseline_contacts, checked_contacts)
        jitter = len(assess(check[:, 3:], targets['fps'], robot.names, reference)['violating_frames'])
        delta_step = np.diff(candidate[:, columns], axis=0) - existing_step
        limit_difference = np.maximum(lower - delta, delta - upper).max()
        speed_difference = (abs(reference_step + delta_step) - speed).max()
        curvature_norm = float(np.linalg.norm(temporal @ (check[:, columns] - raw[:, columns]).ravel()))
        stats = dict(iteration=iteration + 1, qp_status=solution.info.status,
                     qp_iterations=int(solution.info.iter), jitter=jitter,
                     frozen_frames=int(locked.sum()), floor_regressions=int(floor_regression.sum()),
                     floor_peak_m=float(checked_floor.min()),
                     contact_regressions=dict(Counter(channel for _, _, channel in body_contacts)),
                     limit_difference=float(limit_difference), speed_difference=float(speed_difference),
                     correction_curvature=curvature_norm)
        result['candidates'].append(stats)
        if (jitter <= prior_jitter and curvature_norm < original_curvature - 1e-7
                and not floor_regression.any() and not body_contacts
                and limit_difference <= 2e-6 and speed_difference <= 2e-6):
            selected = candidate
            result['accepted_iteration'] = iteration + 1
            break
        floor_indices = np.flatnonzero(np.any(floor_regression, axis=1))
        previously_locked = locked.copy()
        locked[floor_indices // 8] = True
        locked[(floor_indices + 7) // 8] = True
        for frame, _, channel in body_contacts:
            locked[frame] = True
            if channel == 'slip' and frame:
                locked[frame - 1] = True
        # Splitting at these locked poses retains the immutable contact and
        # lets adjoining frames smooth into it under the same time objective.
        if np.array_equal(previously_locked, locked):
            break
        lower[locked] = upper[locked] = 0.
        lo[:count * width] = lower.ravel()
        hi[:count * width] = upper.ravel()
        solver.update(l=lo, u=hi)
    result['changed_frames'] = int(np.any(selected != landed, axis=1).sum())
    return selected, result


def smooth_lift_correction(targets, parent, candidate, start, end):
    """Smooth a local Z correction, preserving clip context and joins."""
    length = end - start
    correction = candidate[start:end, 2] - parent[start:end, 2]
    if length < 3 or not np.any(abs(correction) > 1e-9):
        return candidate
    from scipy.sparse.linalg import spsolve

    second = sparse.diags([1., -2., 1.], [0, 1, 2], shape=(length - 2, length), format='csc')
    system = sparse.eye(length, format='csc') + .5 * second.T @ second
    movable = (~protected_height(targets, len(parent))[start:end]) & (correction != 0.)
    movable[[0, -1]] = False
    fixed = ~movable
    result = candidate.copy()
    if movable.any():
        rhs = correction[movable] - system[movable][:, fixed] @ correction[fixed]
        result[np.flatnonzero(movable) + start, 2] = parent[start:end, 2][movable] + spsolve(system[movable][:, movable], rhs)
    return result
