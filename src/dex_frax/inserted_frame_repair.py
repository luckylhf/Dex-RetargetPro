"""Repair motor-retiming poses against their fixed neighbours."""
import pickle
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .v8.support_acceptance import SUPPORT_ACCEPTANCE_M

POLICY = 'fixed_end_inserted_frame_repair_v1'
SUBSTEPS = 8
# Keep the serialized proposal just inside the 1 mm floor tolerance.
FLOOR_ROUNDING_M = 1.5e-5


def prepared_index(path, meta, payload):
    """Use the motor-insertion map, not fractional frames at native source FPS."""
    from .cli import sha

    timing = meta.get('retiming', {})
    if not timing.get('joint_adaptation', {}).get('inserted_ik_frames'):
        return None
    archive = path.with_suffix('.time_map.npz')
    if not archive.is_file() or sha(archive) != timing.get('source_time_sha256'):
        raise ValueError('Motor retiming map missing or changed')
    with np.load(archive, allow_pickle=False) as data:
        if not np.array_equal(data['source_time_s'], payload['source_time_s']):
            raise ValueError('Motor retiming map does not match output timing')
        return data['joint_prepared_index']


def spans(index, frames):
    """Each span is one retimed interval with both original anchors present."""
    from .v8.math3d import runs

    index = np.asarray(index)
    if (frames < 2 or index.shape != (frames,) or not np.isfinite(index).all() or
            np.any(np.diff(index) <= 0)):
        raise ValueError('Invalid motor retiming index')
    inserted = index != np.rint(index)
    anchors = index[~inserted]
    if inserted[0] or inserted[-1] or not np.array_equal(anchors, np.arange(len(anchors))):
        raise ValueError('Missing original poses in motor retiming index')
    return [(int(a), int(b)) for a, b in runs(inserted)]


def rebuild(robot, rows, rotations, old_floor, a, b, index=None):
    """Blend both anchors, then lift only the editable samples for floor clearance."""
    from .torso_branch import floor_samples

    local = rows[a-1:b+1].copy()
    alpha = np.arange(1, len(local)-1)[:, None] / (len(local)-1)
    local[1:-1] = (1.-alpha)*local[0] + alpha*local[-1]
    if index is not None:
        local[1:-1, :2] = np.column_stack([
            np.interp(index[a:b], index[[a-1,b]], axis)
            for axis in local[[0,-1], :2].T])
    floor = floor_samples(robot, {'root_r': rotations[a-1:b+1]}, local)
    weight = np.ones(len(floor))
    weight[:SUBSTEPS] = np.arange(SUBSTEPS)/SUBSTEPS
    weight[-SUBSTEPS:] = np.arange(SUBSTEPS-1, -1, -1)/SUBSTEPS
    target = np.minimum(old_floor, -SUPPORT_ACCEPTANCE_M+FLOOR_ROUNDING_M)
    extra = (target-floor).max(axis=1)[1:-1]
    lift = max(0., float(np.max(extra/weight[1:-1])))
    local[1:-1, 2] += lift
    return local[1:-1], lift


def impulse_regression(before, after):
    previous = {}
    for event in before.get('root_discontinuities') or ():
        a, b = event['intervals']
        for i in range(a, b):
            key = (event['signal'], i)
            previous[key] = max(previous.get(key, 0.), event['peak_rate'])
    return any(event['peak_rate'] > previous.get((event['signal'], i), 0.)+1e-5
               for event in after.get('root_discontinuities') or ()
               for i in range(*event['intervals']))


def scene_regression(before, after):
    if not after:
        return False
    if not before:
        return bool(after['support_violations'] or after['penetrating_samples'])
    old_failures = {(v['side'], v['frame']): v for v in before['support_violations']}
    for item in after['support_violations']:
        old = old_failures.get((item['side'], item['frame']))
        if (old is None or item['error_m'] > old['error_m']+2e-6 or
                (old['within_platform'] and not item['within_platform'])):
            return True
    return (bool(set(after['penetrating_samples'])-set(before['penetrating_samples'])) or
            after['minimum_convex_distance_m'] < before['minimum_convex_distance_m']-2e-6)


def recover(robot, targets, model, payload, path, trial, meta, quality, cache, solve,
            index, deadline, *, mode='frax', record=None, window_seconds=float('inf')):
    """Audit each rebuilt block on the original grid before committing it."""
    from .acceptance import evaluate
    from .airborne import protected_height
    from .audit_cache import fork
    from .cli import atomic_json, sha
    from .segment_refine import publish_trial, segment_guard

    rows = np.c_[payload['root_pos'], payload['dof_pos']].astype(float)
    rotations = Rotation.from_quat(payload['root_rot']).as_matrix()
    protected = protected_height(targets, len(rows))
    attempts = []
    for a, b in spans(index, len(rows)):
        interior = slice(SUBSTEPS*(a-1)+1, SUBSTEPS*b)
        old_collisions = cache['full_collision']['samples'][interior]
        old_floor = np.asarray(cache['floor'])[interior]
        collision_count = sum(bool(sample[2]) for sample in old_collisions)
        floor_count = int(np.any(old_floor < -SUPPORT_ACCEPTANCE_M, axis=1).sum())
        if not (collision_count or floor_count):
            continue
        began = time.monotonic()
        cutoff = min(deadline, began + window_seconds)
        entry = dict(module='inserted_frames', recipe=POLICY, start=a, end=b,
                     fixed_frames=[a-1, b], status='rejected',
                     before=dict(collision_samples=collision_count, floor_samples=floor_count,
                                 minimum_floor_m=float(old_floor.min())))
        accepted = False
        if began >= deadline:
            entry.update(status='time_slice', reason='inserted_frame_budget')
        elif protected[a:b].any():
            entry['reason'] = 'protected_source_frames'
        else:
            try:
                candidate = rows.copy()
                floor = np.asarray(cache['floor'])[SUBSTEPS*(a-1):SUBSTEPS*b+1]
                replacement, lift = rebuild(robot, rows, rotations, floor, a, b, index)
                if time.monotonic() >= cutoff:
                    raise TimeoutError('inserted_frame_deadline')
                candidate[a:b] = robot.export_rows(replacement)
                if np.array_equal(candidate[a:b], rows[a:b]):
                    raise ValueError('unchanged_inserted_pose')
                joints = candidate[a:b, 3:]
                if (np.any(joints < robot.lower[3:]-2e-6) or
                        np.any(joints > robot.upper[3:]+2e-6)):
                    raise ValueError('inserted_joint_limits')
                speed = np.abs(np.diff(candidate[a-1:b+1, 3:], axis=0))*targets['fps']
                if np.any(speed > robot.speed_limits+1e-4):
                    raise ValueError('inserted_motor_speed_or_seam')
                data = dict(payload, root_pos=candidate[:, :3].astype(np.float32),
                            dof_pos=candidate[:, 3:].astype(np.float32))
                trial.write_bytes(pickle.dumps(data, protocol=4))
                digest = sha(trial)
                atomic_json(trial.with_suffix('.json'), dict(meta, output_sha256=digest))
                updated = fork(cache, changed=(a, b))
                checked = evaluate(trial, dict(meta, output_sha256=digest), solve, mode,
                                   cache=updated, changed=(a, b))
                samples = updated['full_collision']['samples'][interior]
                new_floor = np.asarray(updated['floor'])[interior]
                new_collisions = sum(bool(sample[2]) for sample in samples)
                new_floor_count = int(np.any(new_floor < -SUPPORT_ACCEPTANCE_M, axis=1).sum())
                entry['after'] = dict(collision_samples=new_collisions,
                                      minimum_collision_m=min(sample[0] for sample in samples),
                                      floor_samples=new_floor_count, minimum_floor_m=float(new_floor.min()),
                                      peak_motor_speed_ratio=float(np.max(speed/robot.speed_limits)),
                                      peak_root_speed_m_s=float(np.linalg.norm(
                                          np.diff(candidate[a-1:b+1, :3], axis=0), axis=1).max()*targets['fps']),
                                      lift_m=lift)
                if new_collisions:
                    raise ValueError('inserted_pose_or_seam_still_collides')
                if new_floor_count > floor_count:
                    raise ValueError('more_penetrating_inserted_or_seam_samples')
                improved_floor = (old_floor < -SUPPORT_ACCEPTANCE_M) & (new_floor > old_floor+2e-6)
                if not collision_count and not improved_floor.any():
                    raise ValueError('no_local_geometry_improvement')
                reason = segment_guard(robot, targets, model, rows, candidate, quality, checked, cache, updated, a, b)
                if reason:
                    raise ValueError(reason)
                if impulse_regression(quality, checked):
                    raise ValueError('root_impulse_regression')
                if scene_regression(quality.get('scene_support'), checked.get('scene_support')):
                    raise ValueError('measured_scene_support_regression')
                if time.monotonic() >= cutoff:
                    raise TimeoutError('inserted_frame_deadline')
            except TimeoutError as error:
                entry.update(status='time_slice',reason=str(error))
            except (ValueError, RuntimeError) as error:
                entry['reason'] = str(error)
            else:
                publish_trial(trial, path, data)
                entry.update(status='kept', output_sha256=digest,
                             changed_frames=(a+np.flatnonzero(np.any(candidate[a:b]!=rows[a:b], axis=1))).tolist())
                rows, payload, quality, cache = candidate, data, checked, updated
                accepted = True
        entry['elapsed_s'] = time.monotonic()-began
        attempts.append(entry)
        if record is not None:
            record(entry, (payload, rows, quality, cache) if accepted else None)
    return payload, quality, cache, attempts
