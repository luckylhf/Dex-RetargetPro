"""Reconstruct isolated bad frames from clean temporal context."""
import time

import numpy as np
from .airborne import protected_height
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation


def isolated_frames(event):
    """A reversal brackets the corrupt samples between two observed poses."""
    start, end = event['intervals']
    if 'reversal' in event['reasons'] and 2 <= end-start <= 6:
        return np.arange(start+1, end)
    return np.empty(0, dtype=int)


def context_seeds(values, frames, available, phases):
    """Two-sided interpolation and backward prediction in a stable phase."""
    length = len(values)
    start, end = int(frames[0]), int(frames[-1])
    left = np.arange(max(0, start-4), start)
    right = np.arange(end+1, min(length, end+5))
    left, right = left[available[left]], right[available[right]]
    if left.size >= 2 and right.size >= 2:
        context = np.r_[left, right]
        yield 'neighbor_reconstruction', PchipInterpolator(context, values[context], axis=0)(frames), context
    if right.size >= 3 and np.all(np.diff(right) == 1) and right[0] == end+1:
        if np.all(phases[frames] == phases[right[0]]) and np.all(phases[right] == phases[right[0]]):
            # Estimate a local trajectory from a few later samples and evaluate
            # it at the missing timestamps. Keep those future samples intact.
            offsets = (right-right[0])[:, None]
            speed, origin = np.linalg.lstsq(np.c_[offsets, np.ones(len(right))], values[right], rcond=None)[0]
            yield 'same_phase_backward_reconstruction', origin+(frames-right[0])[:, None]*speed, right


def contact_phases(targets):
    phases = [protected_height(targets,len(targets['source_airborne'])).astype(int)]
    for side in targets['sides'].values():
        phases.append(np.where(np.asarray(side['contact']) > 0, side.get('phase', 1), 0))
    from .support_quality import preserve_contact
    for c in targets.get('body_contacts', ()):
        active = np.flatnonzero(c['mode'])
        mode = np.zeros_like(c['mode'])
        mode[active] = [c['mode'][i] if preserve_contact(c,i) else 0 for i in active]
        phases.append(mode)
    return np.column_stack(phases)


def columns_for(robot, signal):
    if signal.key in ('root_translation','pelvis_height'):
        return np.asarray([0,1,2, *(j for g in robot.sides.values() for j in g['leg'])])
    for group in robot.sides.values():
        for limb in ('arm','leg'):
            if set(signal.columns).intersection(group[limb]):
                return np.asarray(group[limb],dtype=int)
    return np.asarray(signal.columns,dtype=int)


def reconstruction_candidates(repair, signal, frames, available, phases):
    r = repair
    cols = columns_for(r.robot, signal)
    if signal.angular:
        rotations = Rotation.from_quat(r.quaternions)
        origin = rotations[frames[0]-1]
        values = (rotations * origin.inv()).as_rotvec()
    else:
        values = r.rows[:,cols]
    for name, estimate, context in context_seeds(values, frames, available, phases):
        if signal.angular:
            quats = r.quaternions.copy()
            proposed = (Rotation.from_rotvec(estimate)*origin).as_quat()
            proposed[np.einsum('ij,ij->i', proposed, quats[frames]) < 0] *= -1
            quats[frames] = proposed
            yield name, r.rows, quats, Rotation.from_quat(quats).as_matrix(), context
        else:
            rows = r.rows.copy()
            rows[np.ix_(frames,cols)] = np.clip(estimate,r.robot.lower[cols],r.robot.upper[cols]).astype(np.float32)
            yield name, rows, r.quaternions, r.rotations, context


def coupled_contact(repair, seed, a, b, frames, weights, seconds):
    """Fit pelvis and both legs together, preserving the recorded contact schedule."""
    from .adaptive_refine import MinimumChangeObjective
    from .local_refine import fit_window

    r = repair
    ids = np.asarray([0,1,2, *(j for g in r.robot.sides.values() for j in g['leg'])])
    deadline = min(r.deadline,time.monotonic()+seconds)
    config = dict(weights, _repair_deadline=deadline,
                  _initial_leg_collision_cfg=r.model.cfg,
                  _frame_reconstruction=dict(frames=frames))
    target = dict(r.targets, root_r=r.rotations)
    proposal, detail = fit_window(
        r.robot, target, r.rows, config, a, b, iterations=12, ids_override=ids,
        seed=seed, local_checks=True, allow_support=True,
        objective_factory=lambda i: MinimumChangeObjective(seed[i]),
    )
    return proposal.astype(np.float32).astype(float), detail
