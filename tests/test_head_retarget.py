"""Head coordinates follow relative source motion and keep collision coverage."""

from dataclasses import replace
import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from conftest import synthetic_mocap
from dex_frax.ik import FraxIK
from dex_frax.robot_profile import HEAD_JOINTS, JOINT_COUNT
from dex_frax.scene_support import audit_scene
from dex_frax.v8 import convert
from dex_frax.v8.robot import Robot
from dex_frax.v8.self_collision import CollisionModel, scan
from dex_frax.v8.source import head_relative_target


def test_head_yaw_pitch_follow_bvh_and_ignore_whole_body_yaw():
    body = synthetic_mocap()
    count = 48
    rotations = np.broadcast_to(np.eye(3), body.local_r[:count].shape).copy()
    motion = replace(body, local_p=body.local_p[:count], local_r=rotations)
    rest = convert.adapt(motion, 'mocap')
    mount = rest['r']['Head'][0].T @ rest['torso_r'][0]

    expected = np.zeros((count, 2))
    sweep = np.sin(np.linspace(0., np.pi, 16))
    expected[:16, 0] = .35 * sweep
    expected[16:32, 1] = -.25 * sweep
    rotations[:, motion.names.index('Head')] = (
        mount @ Rotation.from_euler('ZY', expected).as_matrix() @ mount.T
    )
    rotations[32:, motion.names.index('Hips')] = Rotation.from_euler(
        'Y', np.linspace(0., .25, 16)[:, None]).as_matrix()
    source = convert.adapt(motion, 'mocap')
    np.testing.assert_allclose(head_relative_target(source['torso_r'], source['head_r']),
                               Rotation.from_euler('ZY', expected).as_matrix(), atol=1e-12)

    robot = Robot()
    config = json.loads((convert.ROOT / 'config.json').read_text())
    convert.configure(robot, config['anatomy'])
    targets = convert.prepare(source, robot, config['contact'])
    rows, _ = FraxIK(3, 25, block_frames=32)(robot, targets, config['solver'])
    assert tuple(robot.names[-2:]) == HEAD_JOINTS
    assert rows.shape == (count, 3 + JOINT_COUNT)
    np.testing.assert_allclose(rows[:, -2:], expected, atol=np.deg2rad(2.))
    assert np.all(np.max(abs(np.diff(rows[:, -2:], axis=0)), axis=0) * targets['fps']
                  <= robot.speed_limits[-2:] + 1e-5)


def test_neck_overlap_is_excluded_and_head_hand_penetration_is_reported():
    robot = Robot()
    model = CollisionModel(robot, {'coverage': 'full'})
    x = np.r_[0., 0., 1., np.zeros(JOINT_COUNT)]
    model.update(x, np.eye(3))
    assert not np.any(model.distances() < -model.cfg['penetration_tolerance_m'])
    torso = model.m.geom('waist_pitch_link_collision').id
    for name in ('head_pitch_link_collision', 'camera_head_link_collision'):
        pair = {'waist_pitch_link_collision', name}
        assert pair in [set(item['pair']) for item in model.exclusions]
        assert pair not in [set(names) for names in model.names]
        assert mujoco.mj_geomDistance(
            model.m, model.configuration.data, torso, model.m.geom(name).id, .08, None) < -.01

    x[robot.sides['Left']['arm']] = [
        1.1560742516, 3.3335788713, -2.8448866808, -2.2216023674,
        -1.5455872972, -1.3264502314, .0753598429,
    ]
    hand_head = next(i for i, names in enumerate(model.names)
                     if set(names) == {'head_yaw_link_collision', 'left_tcp_link_collision'})
    _, events, _ = scan(model, np.tile(x, (2, 1)), np.tile(np.eye(3), (2, 1, 1)), 8)
    assert any(hand_head in event['pairs'] for event in events)


def test_scene_box_detects_head_penetration():
    robot = Robot()
    x = np.r_[0., 0., 1., np.zeros(JOINT_COUNT)]
    rotations = np.tile(np.eye(3), (2, 1, 1))
    box = dict(id='near_head', center_m=[.080613, -.053076, 1.597770], half_size_m=[.004] * 3)
    report = audit_scene(robot, np.tile(x, (2, 1)), rotations, {}, [box])
    assert report['penetrating_samples'] == list(range(9))
    assert report['minimum_convex_distance_m'] < -.006
    box['center_m'][0] += .25
    report = audit_scene(robot, np.tile(x, (2, 1)), rotations, {}, [box])
    assert report['penetrating_samples'] == []
