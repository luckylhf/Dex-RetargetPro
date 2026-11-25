from dex_frax.robot_profile import JOINT_COUNT
import numpy as np
from scipy.spatial.transform import Rotation

from dex_frax.frax_ground_constraints import CLEARANCE_M, RootContinuity, SweptFloor, stage_targets, support_targets
from dex_frax.ik import HingeSpace
from dex_frax.torso_branch import floor_samples
from dex_frax.v8.robot import Robot


def test_swept_rows_match_visual_hull_auditor_and_endpoint_jacobian():
    robot=Robot()
    ids=np.r_[2,robot.sides['Left']['leg'],robot.sides['Right']['leg']]
    con=SweptFloor(HingeSpace(len(ids)),robot,50.,ids)
    prior=np.r_[0.,0.,.85,np.zeros(JOINT_COUNT)]
    now=prior.copy()
    now[2]-=.07
    now[robot.sides['Left']['leg'][3]]=.26
    pair=np.stack([prior,now])
    rotations=Rotation.from_euler('y',[[.08],[-.15]]).as_matrix()
    refs=stage_targets(dict(root_r=rotations,full_chain_reference=pair))
    G,gap=con.measure(now,prior,refs['floor_quaternions'][1],False)
    floor=floor_samples(robot,dict(root_r=rotations),pair)[1:]
    expected=floor.min(axis=0)-CLEARANCE_M
    near=expected<.01
    assert near.any()
    np.testing.assert_allclose(gap[near],expected[near],atol=1e-9)
    worst=np.argmin(gap)
    # Root-Z changes the selected incoming sample by its interpolation fraction.
    q=now.copy();q[2]+=1e-7
    _,moved=con.measure(q,prior,refs['floor_quaternions'][1],False)
    np.testing.assert_allclose((moved[worst]-gap[worst])/1e-7,-G[worst,0],atol=1e-7)


def test_root_guard_uses_source_step_and_bounded_reentry_to_protected_height():
    con=RootContinuity(HingeSpace(13),50.,0)
    ref=dict(root_velocity_goal=np.r_[.1,.2,.90],first=False,protected_z=False,reference_z=.70)
    np.testing.assert_allclose(con.bounds(ref),(.896,.904))
    np.testing.assert_allclose(con.bounds(dict(ref,protected_z=True)),(.896,.896))
    np.testing.assert_allclose(con.bounds(dict(ref,protected_z=True,first=True)),(.7,.7))


def test_stance_displacement_constraint_tracks_same_material_point_only():
    values=dict(contact=np.array([1,1,1,0,0]), phase=np.array([1,1,2,0,0]),
                pivots=np.zeros((5,3)),anchors=np.zeros((5,3)),platform_index=[-1,-1,-1,0,0],
                platform_anchor=np.tile([.1,.2,.25],(5,1)))
    goals=support_targets(values,np.array([.03,.02,.01]))
    assert goals['stance_active'].tolist()==[True]*5
    assert goals['stance_stable'].tolist()==[False,True,False,False,True]
    np.testing.assert_allclose(goals['stance_anchor'][-1],[.1,.2,.25])
    np.testing.assert_allclose(goals['stance_pivot'][-1],[.03,.02,.01])
