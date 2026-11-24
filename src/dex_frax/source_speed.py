"""Review human speeds on normalized geometry, independently of jump repair."""

import numpy as np
from scipy.spatial.transform import Rotation

from .human_motion_constants import (
    HUMAN_EXTREMITY_REVIEW_M_S, HUMAN_HARD_SPEED_LIMITS, HUMAN_KNEE_EXTENSION_REVIEW_RAD_S,
    HUMAN_PELVIS_SPIN_REVIEW_RAD_S, HUMAN_SPRINT_PEAK_M_S,
)
from .robot_profile import SMPLX_LIMB_JOINTS
from .source_integrity import stable_segments
from .v8.math3d import unit


def _record(checks, key, rates, limit, unit_name, origin, applicable=None):
    applicable = np.isfinite(rates) if applicable is None else applicable & np.isfinite(rates)
    selected = rates[applicable]
    checks[key] = dict(
        threshold=limit, unit=unit_name, eligible_intervals=int(applicable.sum()),
        peak=float(selected.max()) if selected.size else None,
        exceedances=[
            dict(interval=[a + origin, b + origin], peak=float(rates[a:b].max()),
                 decision='review', repair_blocking=False)
            for a, b in stable_segments(applicable & (rates > limit))
        ],
    )
    hard = HUMAN_HARD_SPEED_LIMITS.get(key)
    if hard is not None:
        threshold = hard['threshold']
        if (not np.isfinite(threshold) or threshold <= 0 or hard['unit'] != unit_name
                or not all(hard[name] for name in ('axis', 'scope', 'source'))):
            raise ValueError('Invalid hard human speed reference for '+key)
        checks[key]['hard_limit'] = dict(hard)
        checks[key]['hard_exceedances'] = [
            dict(channel=key, interval=[a+origin,b+origin], peak=float(rates[a:b].max()),
                 threshold=threshold, unit=unit_name, source=hard['source'], axis=hard['axis'],
                 scope=hard['scope'], kind='human_speed_hard_limit', decision='review', repair_blocking=True)
            for a,b in stable_segments(applicable & (rates > threshold))
        ]


def record_remaining_speed_limits(source_report, audit, grid, *, source_frames=None):
    """Only exceedances remaining on the corrected poses gate training."""
    events = [event for check in audit['checks'].values() for event in check.get('hard_exceedances', ())]
    if not events:
        return
    for event in events:
        onset, stop = event['interval']
        interval = [float(source_frames[onset]),float(source_frames[stop])] if source_frames is not None else [onset,stop]
        source_report['blocking_events'].append(dict(event, interval=interval,
            frame_grid=grid, checked_interval=[onset,stop]))


def _knee_extension(positions, femur_rotation, fps):
    """Estimate flexion around a stable femur-local axis from bent samples."""
    hip, knee, ankle = positions
    thigh, shin = unit(knee - hip), unit(ankle - knee)
    normal = np.cross(thigh, shin)
    dot = np.sum(thigh * shin, axis=1)
    # Engineering calibration criteria. Avoid nearly straight and deeply bent
    # samples whose bone cross product poorly identifies the femur flexion axis.
    bent = (dot < np.cos(np.deg2rad(25))) & (dot > np.cos(np.deg2rad(140)))
    if bent.sum() < 5:
        return None
    normals = np.einsum('nji,nj->ni', femur_rotation[bent], unit(normal[bent]))
    local_axis = unit(np.median(normals, axis=0))
    if np.median(normals @ local_axis) < .85:
        return None
    axis = femur_rotation @ local_axis
    projected_dot = dot - np.sum(thigh * axis, axis=1) * np.sum(shin * axis, axis=1)
    angles = np.arctan2(np.sum(axis * normal, axis=1), projected_dot)
    return -np.diff(np.unwrap(angles)) * fps


def inspect_speeds(p, r, source, fps, *, ground_running=False, frame_origin=0):
    """Compute reference exceedances in a Z-up world on one sample grid."""
    checks, skipped, landmarks = {}, {}, {}
    root = 'pelvis' if source == 'smplx' else 'Hips'
    torso = next((k for k in ('spine3', 'Spine2', 'Spine1', 'Spine') if k in r), None)
    if ground_running and root in p:
        speed = np.linalg.norm(np.diff(p[root][:, :2], axis=0), axis=1) * fps
        _record(checks, 'ground_running', speed, HUMAN_SPRINT_PEAK_M_S, 'm/s', frame_origin)
        landmarks['ground_running'] = root
    else:
        skipped['ground_running'] = 'requires_declared_unaided_ground_running_and_root'

    if root in r and torso is not None and len(r[root]) >= 2:
        pelvis, chest = r[root], r[torso]
        pelvis_steps, torso_steps = (
            Rotation.from_matrix(x[1:] @ x[:-1].transpose(0, 2, 1)).as_rotvec()
            for x in (pelvis, chest)
        )
        local_up = 1 if source in ('mocap', 'smplx') else 0
        upright = pelvis[:, 2, local_up] >= np.cos(np.deg2rad(30))
        yaw, chest_yaw = pelvis_steps[:, 2], torso_steps[:, 2]
        # Engineering tolerances for mostly vertical, coherent body rotation.
        vertical = np.abs(yaw) >= .90 * np.linalg.norm(pelvis_steps, axis=1)
        coherent = np.abs(yaw - chest_yaw) <= .25 * np.abs(yaw)
        _record(checks, 'pelvis_spin', np.abs(yaw) * fps, HUMAN_PELVIS_SPIN_REVIEW_RAD_S,
                'rad/s', frame_origin, upright[1:] & upright[:-1] & vertical & coherent)
        landmarks['pelvis_spin'] = [root, torso]
    else:
        skipped['pelvis_spin'] = 'requires_pelvis_and_torso_orientations'

    for side in ('Left', 'Right'):
        if source == 'smplx':
            names = {part: side.lower() + '_' + label for part, label in SMPLX_LIMB_JOINTS.items()}
        else:
            hip, knee = ('Leg', 'Shin') if source == 'soma' else ('UpLeg', 'Leg')
            names = {part: side + part for part in ('Foot', 'Hand')}
            names.update(UpLeg=side + hip, Leg=side + knee)
            names['ToeBase'] = next((side + name for name in ('ToeBase', 'Toe', 'Foot__EndSite')
                                     if side + name in p and np.any(p[side + name] != p.get(names['Foot'], 0.))),
                                    names['Foot'])
            names['Hand'] = next((side + name for name in ('HandMiddle1', 'Hand__EndSite', 'Hand')
                                  if side + name in p and (name == 'Hand' or
                                      np.any(p[side + name] != p.get(names['Hand'], 0.)))), names['Hand'])

        key = side + '_knee_extension'
        joints = [names[part] for part in ('UpLeg', 'Leg', 'Foot')]
        speed = (_knee_extension([p[name] for name in joints], r[joints[0]], fps)
                 if joints[0] in r and all(name in p for name in joints) else None)
        if speed is None:
            skipped[key] = 'requires_stable_anatomical_flexion_axis'
        else:
            _record(checks, key, speed, HUMAN_KNEE_EXTENSION_REVIEW_RAD_S, 'rad/s', frame_origin)
            landmarks[key] = joints
        for part, label in (('Hand', 'hand'), ('ToeBase', 'foot')):
            key, name = side + '_' + label, names[part]
            if name not in p:
                skipped[key] = 'missing_end_effector_landmark'
                continue
            speed = np.linalg.norm(np.diff(p[name], axis=0), axis=1) * fps
            _record(checks, key, speed, HUMAN_EXTREMITY_REVIEW_M_S, 'm/s', frame_origin)
            landmarks[key] = name
    return dict(checks=checks, skipped=skipped, landmarks=landmarks,
                hard_speed_limits={key:value['hard_limit'] for key,value in checks.items()
                                   if 'hard_limit' in value})
