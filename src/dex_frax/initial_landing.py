"""Build a support-based landing seed before the full-film collision pass."""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .airborne import protected_height
from .ground_settle import contact_mask
from .support_quality import preserve_contact
from .torso_branch import floor_samples
from .v8.math3d import runs
from .v8.multicontact import contact_local


def lift_and_land(robot, targets, raw):
    n = len(raw)
    protected = protected_height(targets, n)
    marked = contact_mask(targets, n)
    # Give both ends of every sampled interval room to clear the lowest hull.
    minimum = floor_samples(robot, targets, raw).min(axis=1)
    clearance = minimum[::8].copy()
    low = minimum[:-1].reshape(-1,8).min(axis=1)
    clearance[:-1] = np.minimum(clearance[:-1],low)
    clearance[1:] = np.minimum(clearance[1:],low)
    eligible = (marked | (clearance < -.001)) & ~protected
    support_height = np.full(n, np.inf)
    for i in np.flatnonzero(marked & eligible):
        robot.set(raw[i],targets['root_r'][i])
        heights = [robot.point(g['foot'],robot.lowest(s))[2]
                   for s,g in robot.sides.items() if targets['sides'][s]['contact'][i] > 0]
        heights.extend(robot.point(c['body'],contact_local(robot,c))[2]
                       for c in targets['body_contacts'] if preserve_contact(c,i))
        support_height[i] = min(heights, default=np.inf)

    lifted = raw.copy()
    desired = np.zeros(n)
    segments = []
    for a,b in runs(eligible):
        gap = .001 - clearance[a:b]
        contact = support_height[a:b]
        common = max(0., float(gap.max()), float(-contact.min())) + .001
        lifted[a:b,2] += common
        # Descend the whole episode from this lift. At each source instant a
        # marked contact or any closer shell stops the descent. Frax then fits
        # all active anchors together, including unequal support heights.
        shift = -np.minimum(contact, clearance[a:b]-.0001)
        desired[a:b] = np.where(np.isfinite(contact), shift, np.maximum(0.,shift))
        segments.append(dict(start=int(a), end=int(b), common_lift_m=common,
                             source_marked_frames=int(marked[a:b].sum()),
                             raw_minimum_clearance_m=float(clearance[a:b].min()),
                             contact_descent_offset_range_m=[float(desired[a:b].min()),float(desired[a:b].max())]))
    # Use the landed trajectory as an IK target. Feather the offsets around
    # changes in contact; the swept solver then enforces the real geometry.
    offset = gaussian_filter1d(desired, max(1., .08*targets['fps']), mode='nearest')
    offset[protected] = 0.
    seed = raw.copy()
    seed[:,2] += offset
    center = .5*(robot.lower[3:] + robot.upper[3:])
    q = center + np.remainder(raw[:,3:]-center+np.pi,2.*np.pi)-np.pi
    seed[:,3:] = np.clip(q,robot.lower[3:]+1e-7,robot.upper[3:]-1e-7)
    return lifted,seed,dict(policy='whole_episode_vertical_lift_then_marked_support_landing_v1',
                            protected_frames=int(protected.sum()), segments=segments)


def solve_landing(robot, targets, cfg, raw, solver, progress=None):
    lifted,seed,report = lift_and_land(robot,targets,raw)
    root = targets['root_p'].copy()
    root[:,2] = seed[:,2]
    # Keep the prepared BVH translation on its original time grid. Solve
    # support with height and joints while the source supplies both XY axes.
    seed[:, :2] = root[:, :2]
    fixed = np.zeros(raw.shape[1], dtype=bool)
    fixed[:2] = True
    local = dict(targets,root_p=root,priority_pose=seed,priority_mask=fixed,
                 full_chain_reference=raw,landing=True)
    options = dict(cfg,root_position=max(18.,cfg['root_position']),
                   leg_direction=max(12.,cfg['leg_direction']))
    rows,records = solver(robot,local,options,progress)
    if not np.array_equal(rows[:, :2], root[:, :2]):
        raise RuntimeError('Landing changed the prescribed BVH root XY trajectory')
    report['root_xy'] = dict(mode='prepared_source_xy_prescribed',
                             maximum_step_error_m=0.)
    report['solver'] = dict(solver.report)
    return lifted,rows,records,report
