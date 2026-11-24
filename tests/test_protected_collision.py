import time

import numpy as np
import pytest

from dex_frax.protected_collision import propose
from dex_frax.robot_profile import WAIST_JOINTS
from dex_frax.torso_branch import floor_samples
from dex_frax.v8.robot import Robot
from dex_frax.v8.self_collision import CollisionModel, scan


@pytest.fixture(scope='module')
def model():
    return CollisionModel(Robot(), {'coverage': 'full'})


def contact_scene(model, left_hip_roll, frames=15):
    """Construct foot/foot or hand/thigh overlap using the production meshes."""
    robot = model.r
    row = np.r_[[0., 0., 1.], np.clip(np.zeros(29), robot.lower[3:], robot.upper[3:])]
    row[robot.sides['Left']['leg'][1]] = left_hip_roll
    robot.set(row, np.eye(3))
    row[2] -= min(robot.point(b)[2] + (s['hull'] @ robot.rot(b)[2]).min()
                  for b, s in robot.surface_models.items())
    robot.set(row, np.eye(3))
    repeat = lambda a: np.repeat(np.array(a)[None], frames, axis=0)
    sides = {}
    for side, group in robot.sides.items():
        pivot = robot.lowest(side)
        anchor = robot.point(group['foot'], pivot)
        sides[side] = dict(contact=np.full(frames, side == 'Right'), pivots=repeat(pivot),
                           anchors=repeat(anchor))
    return repeat(row), dict(root_r=repeat(np.eye(3)), torso_r=repeat(robot.rot(robot.torso)),
                             sides=sides, fps=50.)


@pytest.mark.parametrize('roll,stage', [(-.235, 'pelvis_z_and_legs'), (.15, 'locked_lower_body_arms')])
def test_signed_distance_recovery_preserves_locked_chains_support_and_seams(model, roll, stage):
    robot = model.r
    rows, targets = contact_scene(model, roll)
    reference = rows.copy()
    candidate, detail = propose(robot, targets, rows, model, 0, len(rows), time.monotonic()+15)
    old = scan(model, rows, targets['root_r'], 8)[0]
    new = scan(model, candidate, targets['root_r'], 8)[0]
    assert old['native_colliding_frames'] == len(rows)
    assert new['native_colliding_frames'] == 2  # The colliding endpoints remain fixed.
    assert new['interpolated_colliding_intervals'] < old['interpolated_colliding_intervals']
    assert detail['stage_order'] == ['waist','pelvis_z_and_legs','locked_lower_body_arms']
    phases = {s['name']: s for s in detail['stages']}
    assert [s['axes'] for s in detail['stages']] == [3, 13, 14]
    assert phases[stage]['iterations'] > 0
    assert phases[stage]['constraints']['collision_objectives'] > 0
    np.testing.assert_array_equal(candidate[[0, -1]], rows[[0, -1]])
    np.testing.assert_array_equal(rows, reference)

    waist = [3 + robot.names.index(joint) for joint in WAIST_JOINTS]
    fixed = [0, 1, *waist]
    if stage == 'pelvis_z_and_legs':
        for key in ('floor_native', 'floor_midpoint', 'foot_anchor_XYZ'):
            assert phases[stage]['constraints'][key] > 0
        fixed += [axis for g in robot.sides.values() for axis in g['arm']]
    else:
        fixed += [2] + [axis for g in robot.sides.values() for axis in g['leg']]
    np.testing.assert_array_equal(candidate[:, fixed], rows[:, fixed])
    assert floor_samples(robot, targets, candidate).min() >= -2e-6
    group = robot.sides['Right']
    values = targets['sides']['Right']
    for i, pose in enumerate(candidate):
        robot.set(pose, targets['root_r'][i])
        xy = robot.point(group['foot'], values['pivots'][i])[:2]
        np.testing.assert_allclose(xy, values['anchors'][i, :2], atol=3e-5,rtol=0)


def test_protected_source_frames_are_immutable_inside_and_outside_window(model):
    rows, targets = contact_scene(model, .15, 21)
    targets['source_airborne'] = np.arange(len(rows)) == 10
    candidate, detail = propose(model.r, targets, rows, model, 3, 18, time.monotonic()+15)
    assert detail['changed_frames'] > 0
    fixed = [0,1,2,3,10,17,18,19,20]
    np.testing.assert_array_equal(candidate[fixed], rows[fixed])
    with pytest.raises(TimeoutError):
        propose(model.r, targets, rows, model, 3, 18, time.monotonic()-1)
