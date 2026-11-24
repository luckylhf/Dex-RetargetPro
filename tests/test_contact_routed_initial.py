import numpy as np
import pytest
import json
from pathlib import Path

from dex_frax.contact_routed_initial import SectionJoins, contact_routes, expanded_sections
from dex_frax.lift_continuity import smooth_lift_correction
from dex_frax.v8.robot import Robot


def test_packaged_version_record_matches_active_routing():
    from dex_frax import __version__
    from dex_frax.identity import REFINE_INITIAL_POLICY

    metadata = json.loads((Path(__file__).resolve().parents[1]/'CURRENT_VERSION.json').read_text())
    assert metadata['package_version'] == __version__
    assert metadata['refine_initial_policy'] == REFINE_INITIAL_POLICY


def test_landing_requires_trusted_nonfoot_support_and_yields_to_flight():
    robot = Robot()
    targets = dict(root_p=np.zeros((7, 3)), source_airborne=[0, 1, 0, 0, 0, 0, 0],
                   source_support_uncertain=[0, 0, 0, 0, 1, 0, 0],
                   sides={'Left': dict(contact=[1, 0, 1, 1, 1, 0, 0],
                                       platform_index=[-1, -1, -1, -1, -1, -1, 0])},
                   body_contacts=[
                       dict(body=robot.sides['Left']['knee'], body_name='knee', mode=[0, 0, 1, 0, 0, 0, 0]),
                       dict(body=robot.torso, body_name='torso', mode=[0, 0, 0, 1, 0, 0, 0]),
                       dict(body=robot.sides['Left']['foot'], body_name='foot', mode=[0, 0, 0, 0, 1, 0, 0]),
                       dict(body=robot.torso, body_name='hint', mode=[0, 0, 0, 0, 0, 1, 0],
                            confidence=np.ones(7), source_geometry_confidence=.5),
                       dict(body=robot.torso, body_name='flight_hint', mode=[0, 1, 0, 0, 0, 0, 0]),
                   ])
    active, evidence = contact_routes(robot, targets)
    assert active.tolist() == [False, False, True, True, False, False, False]
    assert evidence == dict(other_support={'knee': [2], 'torso': [3], 'flight_hint': [1]},
                            feet_only_frames=[0, 6], airborne_frames=[1])


def test_expansion_merges_nearby_sections_and_clips_to_valid_frames():
    active = np.zeros(30, dtype=bool)
    active[[0, 5, 18, 29]] = True
    assert expanded_sections(active, 2) == [(0, 8), (16, 21), (27, 30)]


def test_bad_join_moves_outward_and_preserves_whole_frames():
    joins = SectionJoins.__new__(SectionJoins)
    joins.targets = {}
    joins.legacy = np.zeros((30, 32))
    joins.landing = np.ones((30, 32))
    joins.check_direct = lambda cut, entering: 'motor_speed' if cut in (9, 17) else None
    required = np.zeros(30, dtype=bool)
    required[12:14] = True

    result, active, attempts = joins.assemble(required, 3)

    assert attempts[0]['sections'] == [(9, 17)]
    assert attempts[-1]['sections'] == [(8, 18)]
    assert np.flatnonzero(active).tolist() == list(range(8, 18))
    np.testing.assert_array_equal(result[active], joins.landing[active])
    np.testing.assert_array_equal(result[~active], joins.legacy[~active])


def test_expansion_and_join_search_do_not_cross_flight():
    joins = SectionJoins.__new__(SectionJoins)
    flight = np.zeros(24, dtype=bool)
    flight[[6, 14]] = True
    joins.targets = {'source_airborne': flight}
    joins.legacy = np.zeros((24, 32))
    joins.landing = np.ones((24, 32))
    checked = []
    fitted = []

    def check_direct(cut, entering):
        checked.append((cut, entering))
        return 'motor_speed' if cut in (7, 14) else None

    def fit(rows, cut, entering, section, padding):
        fitted.append(cut)
        return rows, {}

    joins.check_direct = check_direct
    joins.fit = fit
    required = np.zeros_like(flight)
    required[10] = True

    result, active, attempts = joins.assemble(required, 3)

    assert attempts[0]['sections'] == [(7, 14)]
    assert checked == [(7, True), (14, False)]
    assert fitted == [7, 14]
    assert np.flatnonzero(active).tolist() == list(range(7, 14))
    np.testing.assert_array_equal(result[flight], joins.legacy[flight])


def test_unresolved_join_stops_at_flight_boundary():
    joins = SectionJoins.__new__(SectionJoins)
    joins.targets = {'source_airborne': [True, False, False, False, False]}
    joins.legacy = np.zeros((5, 32))
    joins.landing = np.ones((5, 32))
    joins.check_direct = lambda cut, entering: 'motor_speed'

    def fail_fit(*args):
        raise ValueError('insufficient_boundary_context')

    joins.fit = fail_fit
    with pytest.raises(ValueError, match='frame 1 without changing source flight'):
        joins.assemble(np.array([False, True, False, False, False]), 3)


def test_lift_correction_smooths_only_changed_unprotected_rows():
    parent = np.zeros((8, 32))
    lifted = parent.copy()
    lifted[1:7, 2] = [.01, .05, .02, .06, .05, .01]
    protected = np.zeros(8, dtype=bool)
    protected[4] = True
    result = smooth_lift_correction(dict(source_airborne=protected), parent, lifted, 1, 7)
    np.testing.assert_array_equal(result[[0, 1, 4, 6, 7]], lifted[[0, 1, 4, 6, 7]])
    assert np.linalg.norm(np.diff(result[:, 2], n=2)) < np.linalg.norm(np.diff(lifted[:, 2], n=2))
