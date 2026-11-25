from dex_frax.robot_profile import POSE_SIZE
import jax.numpy as jnp
import numpy as np
import pytest

from dex_frax.frax_ground_constraints import CLEARANCE_M, ROOT_CORRECTION_SPEED_M_S, RootContinuity, root_tube
from dex_frax.ik import FrameBounds, HingeSpace
from dex_frax.refine_routing import descendant_bodies
from dex_frax.v8.robot import Robot


def test_reduced_leg_box_projects_every_moving_hinge_without_changing_root_or_waist():
    robot = Robot()
    hinges = np.array([i for side in robot.sides.values() for i in side['leg']])
    ids = np.r_[2, hinges]
    current = np.clip(np.zeros(POSE_SIZE), robot.lower + 1e-7, robot.upper - 1e-7)
    current[2] = 0.7
    current[hinges[0]] = robot.upper[hinges[0]] - .001
    angular_step = robot.speed_limits[hinges - 3] / 50.
    bounds = FrameBounds(HingeSpace(len(ids)), robot.lower[hinges], robot.upper[hinges],
                         angular_step, 50., previous_columns=hinges,
                         local_columns=np.arange(1, len(ids)))
    bad = jnp.array(current[ids] + np.r_[.2, np.tile([2.,-2.],len(hinges)//2)])
    for first in (False, True):
        projected = np.asarray(bounds.project(bad, dict(previous=current, first=first)))
        assert projected[0] == bad[0]
        assert np.all(projected[1:] <= robot.upper[hinges])
        assert np.all(projected[1:] >= robot.lower[hinges])
        if not first:
            assert np.all(abs(projected[1:]-current[hinges]) <= angular_step+1e-12)
        full = current.copy()
        full[ids] = projected
        fixed = np.setdiff1d(np.arange(POSE_SIZE),ids)
        np.testing.assert_array_equal(current[fixed],full[fixed])


@pytest.mark.parametrize('n', [1,23])
def test_root_corridor_admits_source_relative_speed_and_fixed_geometry_clearance(n):
    robot = Robot()
    poses = np.zeros((n,POSE_SIZE))
    poses[:,2] = .75
    poses[n//2,2] = .05
    rotations = np.tile(np.eye(3),(n,1,1))
    source = np.zeros((n,3))
    source[:,2] = .60+np.arange(n)*.001
    tube = root_tube(robot,dict(priority_pose=poses,root_r=rotations,root_p=source,fps=50.))
    legs = [i for side in robot.sides.values() for i in side['leg']]
    fixed = set(robot.surface_models)-descendant_bodies(robot,legs)
    assert np.all(tube[:,0] <= tube[:,1]+1e-6)
    step = ROOT_CORRECTION_SPEED_M_S/50
    assert np.all(abs(np.diff(tube-source[:,2,None],axis=0)) <= step+1e-6)
    root = RootContinuity(HingeSpace(13),50.,0)
    height = poses[0,2]
    for i in range(n):
        delta = source[i,2]-source[max(0,i-1),2]
        lo,hi = map(float,root.bounds(dict(root_velocity_goal=[0,0,height+delta],
                          first=(i==0),protected_z=False,reference_z=poses[i,2],root_tube=tube[i])))
        height = np.clip(poses[i,2],lo,hi)
        row = poses[i].copy(); row[2] = height
        robot.set(row,rotations[i])
        floor = min(robot.point(b)[2]+(robot.surface_models[b]['hull']@robot.rot(b)[2]).min() for b in fixed)
        assert floor >= CLEARANCE_M-1e-6
