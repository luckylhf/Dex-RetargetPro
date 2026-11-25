"""Export quality gates with explicit target provenance; lossless training export."""
from pathlib import Path
from .robot_profile import MODEL_XML, PELVIS_BODY
import time
from .v8.audit import audit
from .quality_v8 import quality
from .v8.support_acceptance import SUPPORT_ACCEPTANCE_M

ACCEPTED_WITH_NOTES = 'accepted_with_notes'
POSE_NOTES = frozenset((
    'sustained_arm_pose_mismatch', 'source_torso_up_mismatch',
    'source_torso_orientation_mismatch',
)) | frozenset(
    side+'_'+key+'_source_mismatch' for side in ('Left','Right')
    for key in ('upper_deg','lower_deg','hand_forward_deg','hand_width_deg',
                'thigh_deg','shin_deg','forearm_width_deg')
)


def disposition(flags):
    flags = set(flags)
    if not flags:
        return 'accepted'
    return ACCEPTED_WITH_NOTES if flags <= POSE_NOTES else 'review'


def preview_outcome(record):
    for attempt in reversed(record['attempts']):
        if attempt.get('status')=='complete' and attempt.get('quality')==ACCEPTED_WITH_NOTES:
            record['preview_output'] = attempt['output']
            return ACCEPTED_WITH_NOTES
    return 'review' if any(a.get('quality')=='review' for a in record['attempts']) else 'failed'


def export_training(path, output, ident):
    import pickle
    import numpy as np
    data = pickle.loads(Path(path).read_bytes())
    arrays = dict(joint_pos=data["dof_pos"], joint_names=np.asarray(data["joint_names"]),
                  body_names=np.array([PELVIS_BODY]), body_pos_w=data["root_pos"][:, None],
                  body_quat_w=data["root_rot"][:, None, [3, 0, 1, 2]], fps=np.array(50.), quat_order=np.array("wxyz"),
                  model_sha256=np.array(ident["assets"][MODEL_XML]), converter=np.array("dex_frax"),
                  converter_fingerprint=np.array(ident["fingerprint"]), storage_format=np.array("root_joint_lossless_v1"))
    for name in ('source_time_s','source_frame_float','source_fps'):
        if name in data:arrays[name]=np.asarray(data[name])
    np.savez_compressed(output, **arrays)
    with np.load(output, allow_pickle=False) as loaded:
        if not all(np.array_equal(loaded[k], v) for k, v in arrays.items()):
            raise ValueError("Training export round trip failed")


def evaluate(pkl, meta, solve, mode, cache=None, changed=None):
    begin = time.perf_counter()
    from .resource_events import phase
    phase('audit')
    check = audit(pkl) if cache is None else audit(pkl,cache=cache,changed=changed)
    result = quality(pkl, meta, check)
    import pickle
    import numpy as np
    from .local_refine import continuity
    payload = pickle.loads(pkl.read_bytes())
    result['continuity'] = continuity(payload['dof_pos'], 50.)
    result['continuity']['hardware_acceleration_limits_available'] = False
    result['head_tracking'] = dict(calibration=meta.get('head_calibration'),
                                  orientation_error_deg=check.get('head_orientation_deg'),
                                  target_clipped_frames_by_joint=check.get('head_target_clipped_frames_by_joint'))
    result['knees'] = {name: dict(min_deg=float(np.rad2deg(payload['dof_pos'][:, j]).min()),
                                 negative_frames=int(np.sum(payload['dof_pos'][:, j] < 0.)))
                       for j, name in enumerate(payload['joint_names']) if 'knee' in name}
    flags = result["flags"]
    from .adaptive_refine import policy, pose_mask
    adaptive_cfg = policy()
    with np.load(pkl.with_suffix('.pose_errors.npz')) as errors:
        angles = {s: {k: errors[s+'_'+k+'_deg'] for k in
            ('upper','lower','hand_forward','hand_width','elbow_axis')} for s in ('Left','Right')}
    bad_pose = pose_mask(angles,50.,adaptive_cfg)
    if bad_pose.any(): flags.append('sustained_arm_pose_mismatch')
    result['adaptive_pose_gate'] = dict(policy=adaptive_cfg, violating_frames=int(bad_pose.sum()))
    if check["minimum_body_visual_cm"] / 100 < -SUPPORT_ACCEPTANCE_M:
        flags.append("full_body_mesh_penetration_over_1mm")
    dense_floor=check.get('interpolated_floor')
    result['interpolated_floor']=dense_floor
    if dense_floor and dense_floor['minimum_body_visual_cm']/100 < -SUPPORT_ACCEPTANCE_M:
        flags.append('interpolated_full_body_mesh_penetration_over_1mm')
    for side, values in check["sides"].items():
        if values["minimum_visual_sole_cm"] / 100 < -SUPPORT_ACCEPTANCE_M:
            flags.append(side + "_sole_penetration_over_1mm")
        contact = values["contact_xyz_cm"]
        if contact and contact["max"] / 100 > SUPPORT_ACCEPTANCE_M:
            flags.append(side + "_contact_anchor_norm_over_1mm")
    result['body_support']=check.get('body_support')
    result['temporal_quality']=check.get('temporal_quality')
    result['full_self_collision']=check.get('full_self_collision')
    if result['body_support'] is None:flags.append('body_support_not_checked')
    elif result['body_support']['violating_frames']:flags.append('trusted_body_support_mismatch')
    if result['temporal_quality'] is None:flags.append('temporal_quality_not_checked')
    elif result['temporal_quality']['violating_frames']:flags.append('conversion_added_jitter')
    if result['full_self_collision'] is None:flags.append('full_self_collision_not_checked')
    elif result['full_self_collision']['collision_samples']:flags.append('full_self_collision_unresolved')
    if (meta.get('source_review') or {}).get('blocking_events'):
        flags.append('source_bad_frame_unresolved')
    if (meta.get('source_support_review') or {}).get('unresolved_platforms'):
        flags.append('scene_support_unconfirmed')
    result['scene_support'] = check.get('scene_support')
    if result['scene_support']:
        if result['scene_support']['support_violations']:
            flags.append('measured_platform_anchor_or_boundary_failed')
        if result['scene_support']['penetrating_samples']:
            flags.append('measured_scene_penetration_over_1mm')
    result['root_discontinuities'] = check.get('root_discontinuities')
    if result['root_discontinuities']:
        flags.append('conversion_added_root_discontinuity')
    from .quality_scope import solver_history,describe
    initial_report=solve.report if solve is not None else meta.get('frax')
    result['initial_solver_history']=solver_history(initial_report)
    initial_steps=result['initial_solver_history']['qp_nonconverged_steps']
    result['final_residual_flags']=sorted(set(flags))
    if initial_steps:

        flags.append("frax_qp_nonconvergence")
    result.update(flags=sorted(set(flags)), backend=mode, support_acceptance_m=SUPPORT_ACCEPTANCE_M,
                  audit_s=time.perf_counter() - begin, status=disposition(flags), pose_notes=sorted(set(flags)&POSE_NOTES),
                  dynamic_validation=False, full_library_validation=False)
    result.update(describe(meta,check,result))
    return result
