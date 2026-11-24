import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dex_frax.source_integrity import inspect, prepare_smplx_channels
from dex_frax.v8.bvh import Motion


def sample(count=35):
    names=['Hips','LeftToe','RightHand','LeftHand']
    t=np.arange(count)
    p=np.zeros((count,len(names),3))
    p[:,0,0]=t*.025
    r=np.repeat(np.eye(3)[None,None],count,axis=0).repeat(len(names),axis=1)
    r[:,2]=Rotation.from_rotvec(np.c_[t*.012,np.zeros((count,2))]).as_matrix()
    r[:,3]=Rotation.from_rotvec(np.c_[np.zeros((count,2)),t*.016]).as_matrix()
    return Motion(names,np.array([-1,0,0,0]),np.zeros((4,3)),p,r,30.,count,30.,'fixture')


@pytest.mark.parametrize('holes', [2, 5])
def test_authenticated_packet_hold_reconstructs_multiple_channels_without_input_mutation(holes):
    m=sample()
    p,r=m.local_p.copy(),m.local_r.copy()
    m.local_p[12:12+holes]=m.local_p[11]
    m.local_r[12:12+holes]=m.local_r[11]
    c,rep=inspect(m,'lafan')
    np.testing.assert_allclose(c.local_p,p,atol=1e-10)
    np.testing.assert_allclose(c.local_r,r,atol=1e-10)
    assert rep['repaired_frames']==list(range(12, 12+holes))
    assert not rep['blocking_events']
    assert not np.array_equal(c.local_r,m.local_r)
    assert all(len(e['evidence_channels'])>=2 for e in rep['events'] if e['kind']=='repeated_capture_packet')


def test_stop_then_resume_at_natural_speed_is_not_a_dropped_packet():
    m=sample()
    for v in (m.local_p,m.local_r):
        v[12:]=np.concatenate([v[11:12],v[11:12],v[12:-2]])
    c,rep=inspect(m,'lafan')
    assert not rep['repaired_frames']
    np.testing.assert_array_equal(c.local_p,m.local_p)
    assert rep['axis_rates']['RightHand']['parent_frame_angular_rad_s']['peak_norm']>0


def test_staggered_holds_in_individual_joints_are_repaired_and_grouped():
    motion = sample()
    expected = motion.local_r.copy()
    motion.local_r[8:13, 2] = expected[7, 2]
    motion.local_r[10:16, 3] = expected[9, 3]

    corrected, report = inspect(motion, 'lafan')

    np.testing.assert_allclose(corrected.local_r, expected, atol=1e-12)
    np.testing.assert_array_equal(corrected.local_p, motion.local_p)
    assert report['repaired_frames'] == list(range(8, 16))
    assert not report['blocking_events']
    assert report['anomaly_segments'] == [{
        'interval': [7, 16], 'channels': ['LeftHand.rotation', 'RightHand.rotation'],
        'repaired_frames': list(range(8, 16)), 'discarded_frames': [],
        'review_channels': [], 'blocking_channels': [],
    }]


def test_unrepairable_fixed_channel_marks_training_gate_and_statistics_stay_advisory():
    m=sample(110)
    m.local_p[20:27,2,0]=.5
    c,rep=inspect(m,'lafan')
    np.testing.assert_array_equal(c.local_p,m.local_p)
    assert any(e['channel']=='RightHand.translation' for e in rep['blocking_events'])
    assert rep['references']['running_knee_extension']['kind']=='activity_group_mean'


def test_fixed_channel_recovers_multiple_frames_from_matching_neighbor_samples():
    motion = sample(110)
    expected = motion.local_p.copy()
    motion.local_p[20:25, 2, 0] = .5

    corrected, report = inspect(motion, 'lafan')

    np.testing.assert_array_equal(corrected.local_p, expected)
    assert report['repaired_frames'] == list(range(20, 25))
    assert not report['blocking_events']
    assert all(e['kind'] == 'fixed_channel_run' for e in report['events'] if e['decision'] == 'repaired')


def test_smplx_native_packet_repair_precedes_resampling_and_preserves_rotvec_branches():
    n=20
    t=np.arange(n)
    fields={'transl':np.c_[t*.035,np.zeros((n,2))],
            'global_orient':np.c_[np.zeros((n,2)),t*.03+2*np.pi],
            'body_pose':np.zeros((n,63))}
    fields['body_pose'][:,0]=t*.023
    good={key:v.copy() for key,v in fields.items()}
    for v in fields.values():
        v[8]=v[7]
    names=['pelvis']+[f'joint{i}' for i in range(21)]
    corrected,report=prepare_smplx_channels(fields,30.,n,names)
    assert report['repaired_frames']==[8]
    np.testing.assert_allclose(corrected['transl'],good['transl'])
    np.testing.assert_allclose(Rotation.from_rotvec(corrected['global_orient']).as_matrix(),
                               Rotation.from_rotvec(good['global_orient']).as_matrix(),atol=1e-10)
    np.testing.assert_array_equal(corrected['global_orient'][:8],good['global_orient'][:8])
    assert fields['transl'][8,0]==fields['transl'][7,0]
