import numpy as np
from dex_frax.swept_height import distribute

def test_distribute_respects_locked_flight_and_second_endpoint_capacity():
    d=distribute(np.zeros(2),np.array([0.,.001]),np.array([.3,.7]),.00035)
    np.testing.assert_allclose(d,[0.,.0005],atol=1e-12)
    d=distribute(np.zeros(2),np.array([.0001,.001]),np.array([.5,.5]),.00035)
    np.testing.assert_allclose(d,[.0001,.0006],atol=1e-12)

def test_uncorrectable_interpolant_and_existing_lift_are_bounded():
    d=distribute(np.zeros(2),np.array([.0001,.0001]),np.array([0.,1.]),.0005)
    np.testing.assert_allclose(d,[0.,.0001])
    np.testing.assert_array_equal(distribute(d,d,np.array([0.,1.]),.00005),d)


def test_stance_fit_preserves_other_leg_waist_arms_and_both_velocity_seams():
    from dex_frax.foot_transition import landing_fit
    from dex_frax.v8.robot import Robot
    r=Robot();g=r.sides['Left']
    pose=np.r_[0.,0.,1.,np.zeros(29)]
    pose[g['leg'][0]]=-.35;pose[g['leg'][3]]=.65
    r.set(pose,np.eye(3))
    pose[2]-=r.point(g['foot'],r.lowest('Left'))[2]
    r.set(pose,np.eye(3))
    pivot=g['sole'];xy=r.point(g['foot'],pivot)
    rows=np.repeat(pose[None],3,axis=0)
    t=dict(fps=50.,root_r=np.repeat(np.eye(3)[None],3,axis=0),
           sides={'Left':dict(contact=np.ones(3),pivots=np.tile(pivot,(3,1)),anchors=np.tile(xy,(3,1))),
                  'Right':dict(contact=np.zeros(3))})
    config=dict(numerical_joint_speed_deg_s=120.)
    result=landing_fit(r,t,rows,1,pose[2]+.001,pose,config)
    protected=np.setdiff1d(np.arange(32),[0,1,2]+g['leg'])
    np.testing.assert_array_equal(result[protected],pose[protected])
    for other in (rows[0],rows[2]):
        assert np.all(abs(result[3:]-other[3:])<=r.joint_step(50.,config)+1e-8)
    r.set(result,np.eye(3))
    assert abs(r.point(g['foot'],r.lowest('Left'))[2])<.0001
    assert np.linalg.norm(r.point(g['foot'],pivot)[:2]-xy[:2])<.0001


def test_jitter_fit_constrains_slip_between_individually_valid_anchors():
    from dex_frax.local_geometry import Geometry
    from dex_frax.v8.robot import Robot

    robot = Robot()
    foot = robot.sides['Left']
    rotation = np.eye(3)
    pose = np.r_[0., 0., 1., np.zeros(29)]
    robot.set(pose, rotation)
    pose[2] -= robot.point(foot['foot'], robot.lowest('Left'))[2]
    robot.set(pose, rotation)
    pivot = foot['sole']
    anchor = robot.point(foot['foot'], pivot)
    support = dict(contact=np.ones(4), pivots=np.tile(pivot, (4, 1)), anchors=np.tile(anchor, (4, 1)))
    targets = dict(root_r=np.repeat(rotation[None], 4, axis=0), sides={'Left': support})
    reference = np.repeat(pose[None], 4, axis=0)
    geometry = Geometry(robot, targets, reference, 0, 4, [0], lambda body: body == foot['foot'],
                        fixed_frames=(0, 3), coupled_support=True, preserve_support_step=True)
    candidate = np.array([0., .00075, -.00075, 0.])
    constraints, lower, upper = geometry.evaluate(reference[:, 0], linear=True)
    assert np.any(constraints @ candidate > upper) or np.any(constraints @ candidate < lower)
    geometry.evaluate(candidate)
    assert geometry.last['maximum_contact_m'] < .001
    assert geometry.last['contact_regression_m'] > .0005
    geometry.evaluate(candidate / 2.)
    assert geometry.last['contact_regression_m'] == 0.
