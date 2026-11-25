"""Widespread self-collision routing and per-window acceptance on the current export."""
import math
import pickle
import time

import numpy as np

from .acceptance import evaluate
from .audit_cache import fork
from .cli import atomic_json, sha


def counts(quality):
    collision = quality['full_self_collision']
    return dict(native_frames=collision['native_colliding_frames'],
                swept_samples=collision['collision_samples'],
                interpolated_intervals=collision['interpolated_colliding_intervals'])


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def plan(cache, airborne, fps):
    """Measure the full film once. Work only on non-airborne colliding cores."""
    samples = cache['full_collision']['samples'][::8]
    n = len(samples)
    mask = np.array([bool(s[2]) for s in samples])
    affected = int(mask.sum())
    episodes = runs(mask)
    longest = max((b-a for a, b in episodes), default=0)
    minimum = max(math.ceil(.8*fps), math.ceil(.05*n))
    minimum_run = math.ceil(.4*fps)
    triggered = affected > minimum and (longest >= minimum_run or len(episodes) >= 5)
    trigger = dict(triggered=triggered, total_frames=n, native_frames=affected,
                   minimum_frames_exclusive=minimum, minimum_contiguous_frames=minimum_run,
                   alternative_minimum_episodes=5, episodes=len(episodes), longest_episode_frames=longest,
                   excluded_airborne_collision_frames=int((mask & airborne).sum()))
    windows = []
    if triggered:
        # Prefer sustained single-pair cores with shallow penetration. They offer
        # many recoverable frames per expensive independent candidate audit.
        width = max(8, round(.72*fps))
        for a, b in runs(mask & ~airborne):
            for indices in np.array_split(np.arange(a, b), math.ceil((b-a)/width)):
                i, j = int(indices[0]), int(indices[-1])+1
                native = samples[i:j]
                depth = max(-s[0] for s in native)
                pair_count = len({pair for s in native for pair in s[2]})
                windows.append(dict(start=max(0,i-2), end=min(n,j+2), core_start=i, core_end=j,
                                    collision_frames=j-i, worst_depth_m=float(depth), pair_count=pair_count))
        windows.sort(key=lambda w: (w['pair_count'] > 1, w['worst_depth_m'] > .025,
                                    -w['collision_frames'], w['worst_depth_m'], w['core_start']))
    return dict(trigger=trigger, windows=windows)


def slide_guard(targets, old_cache, new_cache, lo, hi):
    """Compare motion of the same load-bearing sole point on each step, in metres."""
    for side, target in targets['sides'].items():
        old = old_cache['frames']['contact_xyz'][side]
        new = new_cache['frames']['contact_xyz'][side]
        for i in range(lo+1, hi):
            if old[i] is None or old[i-1] is None:
                continue
            if new[i] is None or new[i-1] is None:
                return 'lost_foot_anchor:'+side
            # The contact label may change from heel to toe within a stance.
            if (target['phase'][i] != target['phase'][i-1] or
                    not np.allclose(target['pivots'][i], target['pivots'][i-1], atol=1e-8, rtol=0.) or
                    not np.allclose(target['anchors'][i,:2], target['anchors'][i-1,:2], atol=1e-8, rtol=0.)):
                continue
            anchor_step = target['anchors'][i,:2] - target['anchors'][i-1,:2]
            def step(errors):
                return np.linalg.norm((errors[i][:2]-errors[i-1][:2])/100. + anchor_step)
            if step(new) > max(step(old), .001) + 2e-6:
                return 'support_foot_xy_step_regression:'+side
    return None


def certify(robot, targets, model, before, after, quality, proposed, cache, updated, a, b):
    from .segment_refine import segment_guard
    from .robot_profile import ROOT_EXTRA_DROP_TOLERANCE_M

    initial, trial = counts(quality), counts(proposed)
    if trial['native_frames'] >= initial['native_frames']:
        return 'no_native_collision_frame_reduction'
    if any(trial[k] > initial[k] for k in ('swept_samples','interpolated_intervals')):
        return 'swept_collision_regression'
    lo, hi = max(0, a-1), min(len(before), b+1)
    swept = slice(8*lo, 8*(hi-1)+1)
    if np.any(updated['floor'][swept] < np.minimum(0., cache['floor'][swept])-2e-6):
        return 'new_or_deeper_ground_penetration'
    old_raw = -np.diff(before[lo:hi,2])
    new_raw = -np.diff(after[lo:hi,2])
    if np.any(new_raw > np.maximum(old_raw, ROOT_EXTRA_DROP_TOLERANCE_M)+2e-6):
        return 'new_or_deeper_raw_root_drop'
    old_drops = {(event['signal'], i) for event in quality.get('root_discontinuities') or ()
                 for i in range(*event['intervals'])}
    if any((event['signal'], i) not in old_drops for event in proposed.get('root_discontinuities') or ()
           for i in range(*event['intervals'])):
        return 'new_root_impulse'
    return (slide_guard(targets, cache, updated, lo, hi) or
            segment_guard(robot, targets, model, before, after, quality, proposed, cache, updated, a, b))


def attempt(robot, targets, model, rows, payload, quality, cache, job, path, trial, meta, solve, deadline, airborne):
    """Return a committed state only after checking its float32 serialization."""
    from .protected_collision import propose
    from .segment_refine import publish_trial

    a, b = job['start'], job['end']
    entry = dict(job, module='protected_collision', solver='protected_collision', status='rejected',
                 before=counts(quality))
    candidate, detail = propose(robot, targets, rows, model, a, b, deadline, protected=airborne)
    entry['detail'] = detail
    if candidate.shape != rows.shape or not np.isfinite(candidate).all():
        return dict(entry, reason='invalid_candidate_rows'), None
    candidate = robot.export_rows(candidate).astype(float)
    if (not np.array_equal(candidate[:a+1], rows[:a+1]) or
            not np.array_equal(candidate[b-1:], rows[b-1:]) or
            not np.array_equal(candidate[airborne], rows[airborne]) or
            not np.array_equal(candidate[:,:2], rows[:,:2])):
        return dict(entry, reason='changed_fixed_rows_or_root_xy'), None
    changed = np.flatnonzero(np.any(candidate[a:b] != rows[a:b], axis=1))+a
    if not len(changed):
        return dict(entry, reason='unchanged_candidate'), None
    # A native-count improvement is necessary. Cheap native FK keeps unhelpful
    # candidates out of the more expensive complete incremental audit.
    native = cache['full_collision']['samples'][::8]
    tol = model.cfg['penetration_tolerance_m']
    delta = 0
    for i in changed:
        model.update(candidate[i], targets['root_r'][i])
        delta += int(model.distances().min() < -tol) - int(bool(native[i][2]))
    entry['proposal_native_reduction'] = -delta
    if delta >= 0:
        return dict(entry, reason='no_native_collision_frame_reduction'), None
    if time.monotonic() >= deadline:
        return dict(entry, reason='deadline_before_audit', status='time_slice'), None
    data = dict(payload, root_pos=candidate[:,:3].astype(np.float32), dof_pos=candidate[:,3:].astype(np.float32))
    trial.write_bytes(pickle.dumps(data, protocol=4))
    description = dict(meta, output_sha256=sha(trial))
    atomic_json(trial.with_suffix('.json'), description)
    changed_range = int(changed[0]), int(changed[-1])+1
    updated = fork(cache, changed=changed_range)
    proposed = evaluate(trial, description, solve, 'frax', cache=updated, changed=changed_range)
    entry['proposed'] = counts(proposed)
    reason = certify(robot, targets, model, rows, candidate, quality, proposed, cache, updated, a, b)
    if reason:
        return dict(entry, reason=reason), None
    if time.monotonic() >= deadline:
        return dict(entry, reason='deadline_before_commit', status='time_slice'), None
    publish_trial(trial, path, data)
    return dict(entry, reason='native_collision_reduction_and_protections_passed', status='kept'), (
        candidate, data, proposed, updated)
