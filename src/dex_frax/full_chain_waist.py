"""Floor-constrained line search on the 0.9.0 full-clip waist orientation fit."""
import numpy as np
from scipy.spatial.transform import Rotation

from .airborne import protected_height
from .refine_routing import descendant_bodies


def constrained_waist(robot, t, seed, fitted, ids, cfg):
    """Keep the closest feasible waist fit along each raw-to-fit trajectory ray.

    Check both adjacent intervals whenever a factor changes. A smooth taper
    spreads reductions across frames and prevents waist velocity seams. Root
    and all leg joints remain fixed, so every anchored sole point is invariant.
    """
    n = len(seed)
    protected = protected_height(t, n)
    movable_bodies = descendant_bodies(robot, ids)
    if any(s['foot'] in movable_bodies for s in robot.sides.values()):
        raise ValueError('Waist changes a support-foot kinematic ancestor')
    surfaces = [(body, model['hull']) for body, model in robot.surface_models.items()
                if body in movable_bodies]
    rotations = t['root_r']
    steps = 8
    alpha = np.arange(steps) / steps
    delta = Rotation.from_matrix(rotations[:-1].transpose(0,2,1) @ rotations[1:]).as_rotvec() if n > 1 else []
    def surface(rows, i, limit=None):
        count = steps if i+1 < n else 1
        result = []
        for k, u in enumerate(alpha[:count]):
            q = rows[i] if k == 0 else (1-u)*rows[i] + u*rows[i+1]
            rot = rotations[i] if k == 0 else rotations[i] @ Rotation.from_rotvec(delta[i]*u).as_matrix()
            robot.set(q, rot)
            h = np.array([robot.d.xpos[b,2] + (hull @ robot.d.xmat[b,6:]).min()
                          for b, hull in surfaces])
            if limit is not None and np.any(h < limit[k]-1e-6):
                return False
            result.append(h)
        return True if limit is not None else np.minimum(0., result)

    reference = [surface(seed, i) for i in range(n)]
    weights = (~protected).astype(float)
    angle_change = fitted[:,ids] - seed[:,ids]
    step = np.maximum(robot.joint_step(t['fps'],cfg)[ids-3],
                      np.abs(np.diff(seed[:,ids],axis=0)))
    out = seed.copy()
    dirty = np.ones(n,bool)
    evaluations = 0
    fallback = False
    for iteration in range(28):
        out[:,ids] = seed[:,ids] + weights[:,None]*angle_change
        bad = np.zeros(n,bool)
        touched = np.flatnonzero(dirty & ((weights > 0) | np.r_[weights[1:] > 0,False]))
        for i in touched:
            if not surface(out, i, reference[i]):
                bad[i:min(n,i+2)] = True
        evaluations += len(touched)
        jumps = np.any(abs(np.diff(out[:,ids],axis=0)) > step+1e-7, axis=1)
        bad |= np.r_[False,jumps] | np.r_[jumps,False]
        if not bad.any():
            break
        # A new candidate is still one full trajectory. Preserve both sides of
        # every changed edge, including those next to a protected flight frame.
        weight_cap = np.ones(n)
        violation = np.flatnonzero(bad)
        for offset, ratio in enumerate((.42,.50,.62,.74,.86,.94)):
            weight_cap[np.maximum(0,violation-offset)] = np.minimum(
                weight_cap[np.maximum(0,violation-offset)],ratio)
            weight_cap[np.minimum(n-1,violation+offset)] = np.minimum(
                weight_cap[np.minimum(n-1,violation+offset)],ratio)
        prior = weights.copy()
        weights *= weight_cap
        weights[weights < .0001] = 0.
        changed = weights != prior
        dirty = changed | np.r_[changed[1:],False]
    else:
        # Raw is always feasible for the one-sided floor and speed constraints.
        out = seed.copy()
        weights[:] = 0.
        fallback = True
    if not np.array_equal(out[:,:3],seed[:,:3]):
        raise RuntimeError('Waist solve changed the root')
    return out, dict(method='bounded_descent_along_full_waist_fit',swept_subdivisions=steps,
        checked_moving_body_surfaces=len(surfaces), invariant_feet_and_root=True,
        fixed_source_frames=int(protected.sum()), bounded_waist_frames=int((weights<1).sum()),
        zero_waist_change_frames=int((weights==0).sum()), full_fit_frames=int((weights==1).sum()),
        projection_passes=iteration+1,interval_checks=evaluations,
        returned_full_raw_waist_on_infeasibility=fallback)
