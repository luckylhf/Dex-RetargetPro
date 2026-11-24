from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from dex_frax.discontinuities import Discontinuities, Signal, detect, matches


def test_marks_short_root_pulse_below_existing_speed_limit():
    positions = np.zeros((12, 3))
    positions[5, 2] = -.08
    signal = Signal('root_translation', .035, 'm/s', 8.)
    event, = detect(signal, positions, 50)
    assert event['intervals'] == [4, 6]
    assert event['reasons'] == ['reversal']
    assert event['peak_rate'] == 4.


def test_marks_single_step_without_rebound():
    positions = np.zeros((12, 1))
    positions[5:, 0] = .09
    signal = Signal('pelvis_height', .025, 'm/s')
    event, = detect(signal, positions, 50)
    assert event['intervals'] == [4, 5]
    assert event['reasons'] == ['isolated_step']


def test_rotation_uses_physical_step_and_marks_sustained_rate_separately():
    angles = 178. + np.arange(12)*1.5
    rotation = Rotation.from_euler('z', angles[:, None], degrees=True).as_quat()
    rotation[1::2] *= -1
    signal = Signal('root_rotation', np.deg2rad(4), 'rad/s', 8.)
    assert detect(signal, Rotation.from_quat(rotation).as_matrix(), 50) == []
    event, = detect(signal, Rotation.from_euler('z', (angles*8)[:, None], degrees=True).as_matrix(), 50)
    assert event['reasons'] == ['rate_limit']


def test_stage_matching_uses_timestamp_across_sampling_rates():
    assert matches(dict(intervals=[39, 41]), [dict(intervals=[23, 25])], 50, 30)
    assert not matches(dict(intervals=[39, 41]), [dict(intervals=[39, 41])], 50, 30)


def test_reconstruction_preserves_contact_guard_and_airborne_samples():
    rows = np.zeros((15, 3))
    rows[7, 2] = -.12
    rotation = np.tile(np.eye(3), (len(rows), 1, 1))
    air = np.zeros(len(rows), dtype=bool)
    air[6] = True
    checked = []
    def check(candidate, _, a, b):
        checked.append((a, b))
        np.testing.assert_array_equal(candidate[air], rows[air])
        return None

    repair = SimpleNamespace(
        robot=SimpleNamespace(sides={}, names=[]), targets={'sides': {}, 'source_airborne': air},
        rows=rows, rotations=rotation, quaternions=Rotation.from_matrix(rotation).as_quat(),
        air=air, fps=50, time_left=lambda: True, check=check,
    )
    from pathlib import Path
    run = Discontinuities(repair, Path('unavailable/motion.pkl'), {}, {})
    run.run()
    assert checked
    assert repair.rows[7, 2] == 0.
    assert not run.measure()['root_translation']


def test_backward_prediction_requires_matching_contact_and_keeps_source_context():
    from dex_frax.frame_repair import context_seeds

    values = np.arange(12,dtype=float)[:,None]
    values[5] = 999.
    available = np.ones(12,bool)
    available[5] = False
    phase = np.zeros((12,1),int)
    phase[5:] = 1
    trials = {name: (fit, context) for name, fit, context in
              context_seeds(values,np.array([5]),available,phase)}
    fit, context = trials['same_phase_backward_reconstruction']
    np.testing.assert_allclose(fit, [[5.]])
    assert min(context) == 6
    phase[5] = 0
    assert [name for name, *_ in context_seeds(values,np.array([5]),available,phase)] == [
        'neighbor_reconstruction'
    ]


def test_same_phase_replacement_is_attempted_after_smoothing_fails():
    from pathlib import Path

    rows = np.zeros((15,3))
    rows[7,0] = .25
    rows[8:,0] = .08
    rotation = np.tile(np.eye(3), (len(rows),1,1))
    air = np.zeros(len(rows),dtype=bool)
    phase = np.r_[np.zeros(7),np.ones(8)]

    def check(candidate, _, a, b):
        return None if np.isclose(candidate[7,0], .08) else 'fixture_contact_error'

    robot = SimpleNamespace(sides={}, names=[], lower=np.full(3,-np.inf), upper=np.full(3,np.inf))
    repair = SimpleNamespace(
        robot=robot, targets={'sides': {'Left': {'contact': phase, 'phase': phase}}, 'source_airborne': air},
        rows=rows, rotations=rotation, quaternions=Rotation.from_matrix(rotation).as_quat(), air=air,
        fps=50, time_left=lambda: True, check=check, compensate_feet=lambda _r,_i,seed=None: seed,
    )
    run = Discontinuities(repair,Path('unavailable/motion.pkl'),{}, {})
    run.run()
    assert any(a['method'] == 'trajectory_interpolation' and a['reason'] for a in run.attempts)
    accepted, = [a for a in run.attempts if a['reason'] is None]
    assert accepted['method'] == 'same_phase_backward_reconstruction'
    assert accepted['rebuilt_frames'] == [7]
    np.testing.assert_array_equal(repair.rows[np.arange(15)!=7],rows[np.arange(15)!=7])


def test_overlimit_vertical_reversal_with_forward_progress_is_rebuilt():
    rows = np.zeros((9,3))
    rows[4:,:] = [.10,-.025,-.18]
    rows[5:,:] = [.12,-.065,.01]
    task = Discontinuities.__new__(Discontinuities)
    task.repair = SimpleNamespace(robot=None,rows=rows,rotations=None)
    signal = Signal('root_translation',.035,'m/s',8.,(0,1,2))
    task.repair.rotations = np.broadcast_to(np.eye(3),(len(rows),3,3))
    event, = detect(signal,rows,50)
    assert 'rate_limit' in event['reasons']
    np.testing.assert_array_equal(task.rebuild_frames(signal,event),[4])
    rows[5:,2] = -.37
    assert not task.rebuild_frames(signal,event).size
def test_supported_drop_can_be_detected_without_reversal_or_quiet_neighbors():
    from dex_frax.discontinuities import Signal, detect

    output = np.array([1., .991, .972, .927, .913, .902])[:,None]
    original = np.array([1., .998, .997, .996, .997, .999])
    pelvis = Signal('pelvis_height', .025, 'm/s', columns=(2,))
    assert not detect(pelvis, output, 50.)
    events = detect(pelvis, output, 50., original)
    assert [(e['peak_interval'], e['reasons']) for e in events] == [(2, ['source_relative_drop'])]
    assert not detect(pelvis, output, 50., output[:,0])

