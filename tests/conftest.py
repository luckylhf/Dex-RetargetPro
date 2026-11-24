import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def synthetic_mocap():
    """An authored Y-up skeleton with stationary feet and moving upper limbs."""
    from dex_frax.v8.bvh import Motion

    names = []
    parents = []
    offsets = []

    def joint(name, parent, offset):
        names.append(name)
        parents.append(-1 if parent is None else names.index(parent))
        offsets.append(offset)

    joint('Hips', None, (0., .94, 0.))
    joint('Spine', 'Hips', (0., .19, 0.))
    joint('Neck', 'Spine', (0., .25, 0.))
    joint('Head', 'Neck', (0., .11, 0.))
    for side, sign in (('Left', 1.), ('Right', -1.)):
        joint(side + 'UpLeg', 'Hips', (sign * .10, -.04, 0.))
        joint(side + 'Leg', side + 'UpLeg', (0., -.43, .04))
        joint(side + 'Foot', side + 'Leg', (0., -.43, -.04))
        joint(side + 'ToeBase', side + 'Foot', (0., -.04, .13))
        joint(side + 'Arm', 'Spine', (sign * .19, .19, 0.))
        joint(side + 'ForeArm', side + 'Arm', (sign * .12, -.24, .03))
        joint(side + 'Hand', side + 'ForeArm', (0., -.18, .14))
        joint(side + 'HandIndex1', side + 'Hand', (sign * .03, -.035, .05))
        joint(side + 'HandMiddle1', side + 'Hand', (0., -.045, .06))
        joint(side + 'HandPinky1', side + 'Hand', (-sign * .03, -.035, .05))

    count = 96
    phase = np.linspace(0., 2. * np.pi, count)
    positions = np.broadcast_to(offsets, (count, len(names), 3)).copy()
    rotations = np.broadcast_to(np.eye(3), (count, len(names), 3, 3)).copy()
    rotations[:, names.index('Spine')] = Rotation.from_euler('Y', .08 * np.sin(phase[:, None])).as_matrix()
    rotations[:, names.index('Head')] = Rotation.from_euler('Y', .10 * np.cos(phase[:, None])).as_matrix()
    rotations[:, names.index('RightArm')] = Rotation.from_euler('X', .20 * np.sin(phase[:, None])).as_matrix()
    return Motion(names, np.asarray(parents), np.asarray(offsets), positions, rotations,
                  fps=50., input_frames=count, input_fps=50., source='synthetic_mocap')


@pytest.fixture(scope='session')
def dex_synthetic_clip():
    from dex_frax.ik import FraxIK
    from dex_frax.v8 import convert

    config = json.loads((convert.ROOT / 'config.json').read_text())
    robot = convert.Robot()
    convert.configure(robot, config['anatomy'])
    targets = convert.prepare(convert.adapt(synthetic_mocap(), 'mocap'), robot, config['contact'])
    rows, _ = FraxIK(3, 25, block_frames=32)(robot, targets, config['solver'])
    return config, targets, rows
