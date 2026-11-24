import json
import os
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from dex_frax.human_motion_constants import HUMAN_SPRINT_PEAK_M_S, wrist_profile
from dex_frax.retiming import time_map
from dex_frax.source_integrity import inspect, prepare_smplx_channels, review_channel
from dex_frax.v8.bvh import Motion, resample


def test_smplx_subprocess_uses_its_own_dependencies(tmp_path, monkeypatch):
    from dex_frax import smplx_cli

    solver_site = tmp_path / 'solver-site'
    package = solver_site / 'dex_frax'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'smplx_input.py').write_text('import numpy\nprint("solver source loaded")\n')
    (solver_site / 'numpy.py').write_text('raise RuntimeError("solver dependency leaked into preprocessor")\n')
    monkeypatch.setattr(smplx_cli, '__file__', str(package / 'smplx_cli.py'))
    env = dict(os.environ, PYTHONPATH=str(solver_site))
    result = subprocess.run(smplx_cli._input_command(sys.executable, []), env=env,
                            check=True, capture_output=True, text=True)
    assert result.stdout.strip() == 'solver source loaded'


def test_native_rig_repair_covers_root_and_nonroot_without_rewriting_input():
    count,fps=16,30.
    p=np.zeros((count,3,3))
    R=np.tile(np.eye(3),(count,3,1,1))
    p[:,0,0]=np.arange(count)*.03
    p[8,0,0]+=.90
    R[8,1]=Rotation.from_rotvec([0.,0.,1.]).as_matrix()
    R[8,2]=Rotation.from_rotvec([0.,0.,1.]).as_matrix()
    src=Motion(['Hips','LeftToe','RightHand'],np.array([-1,0,0]),np.zeros((3,3)),p,R,fps,count,fps,'fixture')
    cleaned,report=inspect(src,'lafan')
    assert report['repaired_frames']==[8]
    assert {e['channel'] for e in report['events'] if e['decision']=='repaired'}=={
        'Hips.translation','LeftToe.rotation','RightHand.rotation'}
    np.testing.assert_array_equal(cleaned.local_r[8,2],np.eye(3))
    assert p[8,0,0]>1 and not np.array_equal(R[8,1],np.eye(3))
    np.testing.assert_allclose(cleaned.local_p[:,0,0],np.arange(count)*.03)
    assert resample(cleaned,50.).input_frames==count


def test_keep_continuous_fast_turn_and_threshold_has_no_margin():
    fast=Rotation.from_rotvec(np.c_[np.zeros((18,2)),np.arange(18)*10./30.]).as_matrix()
    clean,marks=review_channel('Hips.rotation',fast,True,30.)
    assert marks==[]
    np.testing.assert_array_equal(clean,fast)
    assert HUMAN_SPRINT_PEAK_M_S==12.34


def test_boundary_steps_are_discarded_on_both_input_types():
    count, fps = 12, 30.
    pos = np.zeros((count, 3, 3))
    rot = np.broadcast_to(np.eye(3), (count, 3, 3, 3)).copy()
    pos[0, 0, 0] = .5
    rot[-1, 1] = Rotation.from_rotvec([0., 0., 1.2]).as_matrix()
    names = ['Hips', 'LeftLeg', 'RightLeg']
    src = Motion(names, np.array([-1, 0, 0]), np.zeros((3, 3)), pos, rot, fps, count, fps, 'fixture')
    cleaned, audit = inspect(src, 'mocap')
    assert audit['discarded_frames'] == [0,11] and audit['repaired_frames'] == []
    assert audit['kept_native_interval'] == [1,11]
    assert audit['blocking_events'] == []
    np.testing.assert_array_equal(cleaned.local_r, rot[1:-1])
    assert len(src.local_p) == count

    fields = dict(transl=pos[:, 0], global_orient=np.zeros((count, 3)), body_pose=np.zeros((count, 63)))
    fields['body_pose'][-1, 2] = 1.2
    fields['body_pose'][6, 2] = 1.2
    _, smplx_audit = prepare_smplx_channels(fields, fps, count, [f'joint{i}' for i in range(22)])
    assert smplx_audit['discarded_frames'] == [0,11]
    assert smplx_audit['repaired_frames'] == [6]
    json.dumps(smplx_audit)

def test_interior_gap_requires_continuous_neighbors():
    count, fps = 16, 30.
    pos = np.zeros((count, 2, 3))
    rot = np.broadcast_to(np.eye(3), (count, 2, 3, 3)).copy()
    pos[1, 0, 0] = .7
    pos[8:, 0, 2] = 1.
    angles = np.deg2rad(np.arange(count) * 120.)
    rot[:, 1] = Rotation.from_rotvec(np.c_[np.zeros((count,2)), angles]).as_matrix()
    src = Motion(['Hips','LeftLeg'], np.array([-1,0]), np.zeros((2,3)), pos, rot, fps, count, fps, 'fixture')
    cleaned, audit = inspect(src, 'mocap')
    assert audit['discarded_frames'] == []
    assert audit['repaired_frames'] == [1]
    assert any(e['channel']=='Hips.translation' and e['interval']==[7,8] for e in audit['blocking_events'])
    assert not any(e['channel']=='LeftLeg.rotation' for e in audit['events'])
    np.testing.assert_array_equal(cleaned.local_p[8:], pos[8:])
    np.testing.assert_array_equal(cleaned.local_r, rot)


def test_single_native_rotation_step_is_smoothed_but_peak_in_continuous_motion_is_preserved():
    count, fps = 32, 30.
    positions = np.zeros((count, 3))
    positions[:, 0] += np.arange(count) * .01
    positions[0, 0] += .7
    hip_steps = np.zeros(count - 1)
    hip_steps[8:13] = [-3.48, -3.85, 36.65, 2.84, -3.79]
    knee_steps = np.zeros(count - 1)
    knee_steps[17:24] = [2.48, 12.33, 19.55, 35.59, 27.78, 12.6, 4.12]
    pose = np.zeros((count, 63))
    pose[:, 0] = np.deg2rad(np.r_[0., hip_steps.cumsum()])
    pose[:, 3] = np.deg2rad(np.r_[0., knee_steps.cumsum()])
    pose[:, 6] = 2. * np.pi + .25
    fields = dict(transl=positions, global_orient=np.zeros((count, 3)), body_pose=pose)
    names = ['pelvis', 'right_hip', 'right_knee'] + [f'joint{i}' for i in range(3, 22)]

    corrected, audit = prepare_smplx_channels(fields, fps, count, names)

    assert audit['discarded_frames'] == [0]
    assert audit['repaired_frames'] == [10, 11]
    events = [e for e in audit['events'] if e['decision'] == 'repaired']
    assert all(e['smoothing']['anchor_frames'] == [9, 12] for e in events)
    knee = next(e for e in audit['events'] if e['channel'] == 'right_knee.rotation')
    assert knee['decision'] == 'review' and not knee['repair_blocking']
    assert knee['kind'] == 'neighborhood_candidate'
    np.testing.assert_array_equal(np.flatnonzero(np.any(corrected['body_pose'] != pose, axis=1)), [10, 11])
    np.testing.assert_array_equal(corrected['body_pose'][:, 1:], pose[:, 1:])
    np.testing.assert_allclose(np.diff(corrected['body_pose'][9:13, 0]),
                               (pose[12, 0] - pose[9, 0]) / 3.)


def test_isolated_reversal_with_only_one_large_step_repairs_just_the_middle_frame():
    # The +10.43 deg step clears the broad neighborhood screen. Its
    # -7.95 deg return is below that screen and above the single-step
    # isolation tolerance, so this case requires the reversal-specific path.
    steps = [.56, .81, .41, 2.76, 2.24, 10.43, -7.95, 1.20, 4.67, 2.95, 2.80, 2.50, 2.40]
    angles = np.deg2rad(np.r_[0., np.cumsum(steps)])
    rotations = Rotation.from_rotvec(np.c_[angles, np.zeros((len(angles), 2))]).as_matrix()
    pos = np.zeros((len(angles), 1, 3))
    motion = Motion(['Hips'], np.array([-1]), np.zeros((1, 3)), pos, rotations[:, None],
                    30., len(angles), 30., 'fixture')

    cleaned, audit = inspect(motion, 'mocap')

    assert audit['repaired_frames'] == [6] and not audit['blocking_events']
    event = next(e for e in audit['events'] if e['decision'] == 'repaired')
    assert event['kind'] == 'smoothed_source_reversal'
    assert event['smoothing']['anchor_frames'] == [5, 7]
    assert np.isclose(np.rad2deg(event['smoothing']['candidate_peak_rate']) / 30, 1.24)
    np.testing.assert_array_equal(cleaned.local_r[:6], motion.local_r[:6])
    np.testing.assert_array_equal(cleaned.local_r[7:], motion.local_r[7:])


def test_same_direction_translation_speed_peak_is_advisory():
    count, fps = 14, 30.
    pos = np.zeros((count, 1, 3))
    pos[:, 0, 0] = np.arange(count) * .025
    pos[7:, 0, 0] += .20
    rot = np.broadcast_to(np.eye(3), (count, 1, 3, 3)).copy()
    source = Motion(['Hips'], np.array([-1]), np.zeros((1, 3)), pos, rot, fps, count, fps, 'fixture')

    corrected, audit = inspect(source, 'lafan')

    assert not audit['repaired_frames'] and not audit['blocking_events']
    peak = next(e for e in audit['events'] if e['kind'] == 'same_direction_speed_change')
    assert peak['interval'] == [6, 7] and not peak['repair_blocking']
    assert peak['smoothing_checks'][0]['reason'] == 'same_direction_speed_change_preserved'
    np.testing.assert_array_equal(corrected.local_p, pos)


def test_same_direction_ankle_speed_change_survives_native_and_resampled_checks():
    from dex_frax.source_integrity import recheck_resampled

    increments = np.array([1.12, 1.84, 3.49, 2., 4.55, 19.74, 3.55, 1.09, 2.80, 2.27, 1.64])
    turns = np.deg2rad(np.r_[0., np.cumsum(increments)])
    count, fps = len(turns), 30.
    pose = np.zeros((count, 63))
    pose[:, 0] = turns
    fields = dict(transl=np.zeros((count, 3)), global_orient=np.zeros((count, 3)), body_pose=pose)
    names = ['pelvis', 'left_ankle'] + [f'joint{i}' for i in range(20)]

    corrected, audit = prepare_smplx_channels(fields, fps, count, names)
    np.testing.assert_array_equal(corrected['body_pose'], pose)
    assert not audit['repaired_frames'] and not audit['blocking_events']
    assert any(e['kind'] == 'same_direction_speed_change' and e['interval'] == [5, 6]
               for e in audit['events'])

    times = np.arange(19) / 50.
    grid = np.interp(times, np.arange(count) / fps, turns)
    rotations = Rotation.from_rotvec(np.c_[grid, np.zeros((len(grid), 2))]).as_matrix()
    audit_50 = recheck_resampled({'left_ankle.rotation': (rotations, True)}, 50., times, fps)
    assert any(e['kind'] == 'same_direction_speed_change' and not e['corroborated'] for e in audit_50['events'])


def test_overlapping_multi_joint_smplx_discontinuities_are_smoothed():
    count, fps = 35, 30.
    pose = np.zeros((count, 63))
    pose[:, 0] = np.deg2rad(np.arange(count) * 2.)
    pose[:, 3] = np.deg2rad(np.arange(count) * 1.)
    pose[:, 6] = 2. * np.pi + .25
    fields = dict(transl=np.zeros((count, 3)), global_orient=np.zeros((count, 3)), body_pose=pose.copy())
    fields['body_pose'][11:16, 0] += np.deg2rad(65.)
    fields['body_pose'][13:19, 3] -= np.deg2rad(50.)
    names = ['pelvis', 'right_hip', 'left_hip'] + [f'joint{i}' for i in range(3, 22)]

    corrected, report = prepare_smplx_channels(fields, fps, count, names)

    np.testing.assert_allclose(corrected['body_pose'], pose, atol=1e-12)
    assert report['repaired_frames'] == list(range(11, 19))
    assert not report['blocking_events']
    assert report['anomaly_segments'][0]['channels'] == ['left_hip.rotation', 'right_hip.rotation']
    assert report['anomaly_segments'][0]['interval'] == [10, 19]


def test_native_smoothing_does_not_reverse_the_arc_of_a_fast_turn():
    count, fps = 20, 30.
    angles = np.zeros(count)
    angles[9], angles[10:] = 120., 240.
    vectors = np.c_[np.zeros((count, 2)), np.deg2rad(angles)]
    rot = Rotation.from_rotvec(vectors).as_matrix()[:, None]
    pos = np.zeros((count, 1, 3))
    source = Motion(['Hips'], np.array([-1]), np.zeros((1, 3)), pos, rot, fps, count, fps, 'fixture')

    corrected, audit = inspect(source, 'mocap')

    assert not audit['repaired_frames']
    candidate = next(e for e in audit['events'] if e.get('smoothing_checks'))
    assert candidate['smoothing_checks'][0]['reason'] == 'candidate_reverses_rotation_direction'
    assert not candidate['repair_blocking']
    np.testing.assert_array_equal(corrected.local_r, rot)


def test_robot_joint_time_adaptation_and_blocked_source_interval():
    q=np.array([[0.],[0.],[.16],[.16],[.16]])
    R=np.tile(np.eye(3),(len(q),1,1))
    p=np.zeros((len(q),3))
    indices,report=time_map(R,p,50.,8.,8.,joint_positions=q,joint_limits=[2.])
    assert report['joint_stretched_intervals']==[1]
    assert len(indices)>len(q)
    assert np.max(abs(np.diff(np.interp(indices,np.arange(len(q)),q[:,0]))))*50.<=2.+1e-10
    _,blocked=time_map(R,p,50.,8.,8.,joint_positions=q,joint_limits=[2.],eligible=[True,False,True,True])
    assert not blocked['changed'] and blocked['blocked_source_review_intervals']==[1]


def test_nasa_wrist_profile_reports_same_retarg_defaults():
    defaults=wrist_profile()
    assert defaults['soft_deg']==dict(radial=15,ulnar=30,flexion=70,extension=65)
    assert defaults['maximum_deg']==dict(radial=20,ulnar=45,flexion=80,extension=85)
    defaults['maximum_deg']['radial']=0
    assert wrist_profile()['maximum_deg']['radial']==20


def test_a_shape_only_suspect_does_not_block_robot_adaptation():
    from dex_frax.source_integrity import eligible_timing_intervals

    report={'native_fps':30.,'events':[
        {'channel':'Hips.rotation','frame':2,'kind':'isolated_reversal','decision':'review'},
        {'channel':'Hips.translation','interval':[4,5],'kind':'unresolved_large_source_step',
         'decision':'review','repair_blocking':True},
    ]}
    mask=eligible_timing_intervals(report,np.arange(7)/30.)
    np.testing.assert_array_equal(mask,[True,True,True,True,False,True])



def test_human_speed_and_resample_review_are_separate_from_repairs():
    from dex_frax.source_integrity import recheck_bvh

    count, fps = 16, 30.
    pos = np.zeros((count, 1, 3))
    pos[:, 0, 0] = np.arange(count) * 13. / fps
    rotation = np.broadcast_to(np.eye(3), (count, 1, 3, 3)).copy()
    source = Motion(['Hips'], np.array([-1]), np.zeros((1, 3)), pos, rotation, fps, count, fps, 'fixture')
    cleaned, normal = inspect(source, 'mocap')
    assert 'ground_running' in normal['human_speed']['skipped']
    cleaned, review = inspect(source, 'mocap', ground_running=True)
    assert not review['events'] and not review['blocking_events']
    event = review['human_speed']['checks']['ground_running']['exceedances'][0]
    assert event['interval'] == [0, count-1] and not event['repair_blocking']
    resampled = recheck_bvh(resample(cleaned, 50.), 'mocap', review, ground_running=True)
    assert not resampled['events']
    assert resampled['human_speed']['checks']['ground_running']['exceedances']
    np.testing.assert_array_equal(cleaned.local_p, pos)



def test_speed_geometry_checks_extension_axis_and_coherent_upright_yaw():
    from dex_frax.source_speed import inspect_speeds

    frames = 12
    angles = np.radians([90] * 6 + [30] * 5 + [90])
    p = dict(left_hip=np.zeros((frames, 3)),
             left_knee=np.tile([0., 0., -.4], (frames, 1)))
    p['left_ankle'] = p['left_knee'] + .4 * np.c_[np.zeros(frames), np.sin(angles), -np.cos(angles)]
    yaw = Rotation.from_rotvec(np.c_[np.zeros((frames, 2)), np.radians([0] * 6 + [42] * 6)])
    upright = Rotation.from_euler('x', 90., degrees=True)
    body = (yaw * upright).as_matrix()
    r = dict(left_hip=np.broadcast_to(np.eye(3), (frames, 3, 3)), pelvis=body, spine3=body)
    review = inspect_speeds(p, r, 'smplx', 50.)['checks']
    assert review['Left_knee_extension']['exceedances'][0]['interval'] == [5, 6]
    assert len(review['Left_knee_extension']['exceedances']) == 1
    assert review['pelvis_spin']['exceedances'][0]['interval'] == [5, 6]


def test_50hz_reconstructs_only_confirmed_new_spike():
    from dex_frax.source_integrity import recheck_resampled

    native_times = np.arange(13)/30.
    times = np.arange(20)/50.
    native = np.zeros((len(native_times),3))
    samples = np.zeros((len(times),3))
    samples[7,0] = .4
    review = recheck_resampled({'pelvis.translation':(samples,False)},50.,times,30.,
        native_channels={'pelvis.translation':(native,False)},native_times=native_times)
    assert not review['blocking_events']
    assert [e['repaired_frames'] for e in review['events'] if e['decision']=='repaired'] == [[7]]
    np.testing.assert_array_equal(samples,np.zeros_like(samples))


def test_documented_hard_speed_gate_is_distinct_from_review_reference(monkeypatch):
    from dex_frax import source_speed

    count, fps = 16, 30.
    pos = np.zeros((count,1,3))
    pos[:,0,0] = np.arange(count)*16./fps
    rot = np.broadcast_to(np.eye(3),(count,1,3,3)).copy()
    motion = Motion(['Hips'],np.array([-1]),np.zeros((1,3)),pos,rot,fps,count,fps,'fixture')
    monkeypatch.setattr(source_speed,'HUMAN_HARD_SPEED_LIMITS',{
        'ground_running':dict(threshold=15.,unit='m/s',source='synthetic test',
                              axis='horizontal',scope='declared test running')})
    _, review = inspect(motion,'mocap',ground_running=True)
    assert not review['repaired_frames']
    event, = review['blocking_events']
    assert event['kind']=='human_speed_hard_limit' and event['frame_grid']=='native'
    assert event['interval']==[0,count-1]
