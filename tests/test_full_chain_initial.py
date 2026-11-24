import numpy as np
from dex_frax.full_chain_initial import collision_trigger


def test_trigger_counts_native_whole_clip_episodes_including_both_boundaries():
    assert not collision_trigger(range(40),200,50)['triggered']
    report = collision_trigger(range(41),200,50)
    assert report['triggered'] and report['longest_episode_frames']==41
    islands = [i for start in range(0,200,40) for i in range(start,start+9)]
    assert collision_trigger(islands,200,50)['episodes']==5
    assert collision_trigger(islands,200,50)['triggered']
    assert not collision_trigger(range(99),2000,50)['triggered']


def test_full_waist_keeps_support_feet_and_root_fixed_and_improves_torso():
    from dex_frax.v8.robot import Robot
    from dex_frax.pelvis_priority import waist_trajectory
    from dex_frax.full_chain_waist import constrained_waist
    r=Robot()
    seed=np.tile(np.r_[0.,0.,.9,np.zeros(29)],(5,1))
    target=seed.copy()
    j=3+r.names.index('waist_yaw_joint')
    target[:,j]=.12
    r.set(target[0],np.eye(3))
    torso=r.rot(r.torso).copy()
    t=dict(root_r=np.tile(np.eye(3),(5,1,1)),torso_r=np.tile(torso,(5,1,1)),
           fps=50.,source_airborne=np.array([False,False,True,False,False]))
    cfg={'numerical_joint_speed_deg_s':720.}
    fitted,ids=waist_trajectory(r,t,seed,cfg)
    out,stats=constrained_waist(r,t,seed,fitted,ids,cfg)
    locked=np.ones(32,bool)
    locked[ids]=False
    assert np.array_equal(out[:,locked],seed[:,locked])
    assert np.array_equal(out[2],seed[2])
    assert stats['invariant_feet_and_root']
    assert np.any(out[:,j]>0.)


def test_landing_lifts_each_episode_as_a_rigid_vertical_translation():
    from dex_frax.initial_landing import lift_and_land
    from dex_frax.v8.robot import Robot

    robot = Robot()
    rotations = np.repeat(np.eye(3)[None], 12, axis=0)
    raw = np.zeros((12, 32))
    raw[:, 2] = .60
    raw[11, 2] = 1.20
    raw[:, 0] = np.linspace(0., .1, len(raw))
    targets = dict(
        root_r=rotations, fps=50.,
        sides={side: {'contact': np.r_[np.ones(11), 0.]} for side in robot.sides},
        body_contacts=[], source_airborne=np.r_[np.zeros(11, bool), True])

    lifted, seed, report = lift_and_land(robot, targets, raw)

    assert len(report['segments']) == 1
    episode = report['segments'][0]
    assert (episode['start'], episode['end']) == (0, 11)
    assert episode['common_lift_m'] > .05
    np.testing.assert_allclose(lifted[:11, 2] - raw[:11, 2], episode['common_lift_m'])
    np.testing.assert_array_equal(lifted[:, :2], raw[:, :2])
    np.testing.assert_array_equal(lifted[:, 3:], raw[:, 3:])
    assert lifted[11, 2] == seed[11, 2] == raw[11, 2]


def test_landing_prescribes_the_mapped_source_xy_at_every_frame():
    from dex_frax.ik import solve_columns
    from dex_frax.initial_landing import solve_landing
    from dex_frax.v8.robot import Robot

    robot = Robot()
    raw = np.zeros((3, 32))
    raw[:, 2] = .8
    source_root = np.array([[1., 2., .8], [1.01, 1.997, .8], [1.02, 1.991, .8]])
    targets = dict(
        root_p=source_root, root_r=np.repeat(np.eye(3)[None], len(raw), axis=0), fps=50.,
        sides={side: {'contact': np.zeros(len(raw))} for side in robot.sides}, body_contacts=[])

    def solver(robot, target, config, progress):
        np.testing.assert_array_equal(solve_columns(robot, target), np.arange(2, 32))
        np.testing.assert_array_equal(target['priority_pose'][:, :2], source_root[:, :2])
        return target['priority_pose'], []

    solver.report = {}
    _, rows, _, report = solve_landing(robot, targets, dict(root_position=1., leg_direction=1.), raw, solver)
    np.testing.assert_array_equal(np.diff(rows[:, :2], axis=0), np.diff(source_root[:, :2], axis=0))
    assert report['root_xy']['maximum_step_error_m'] == 0.
    np.testing.assert_array_equal(raw[:, :2], 0.)


def test_post_leg_recheck_marks_both_ends_of_swept_penetration_and_slip():
    from dex_frax.post_leg_contact import affected_frames

    floor = np.zeros((33, 2))
    worse = floor.copy()
    worse[11, 0] = -.003
    old = {(4, 'Left_foot', 'slip'): (.0007, .001)}
    new = {(4, 'Left_foot', 'slip'): (.002, .001)}
    np.testing.assert_array_equal(
        affected_frames(floor, worse, old, new), [False, True, True, True, True])


def test_lift_preserves_frozen_joins_and_support_capacity_from_seed():
    from dex_frax.post_leg_contact import bounded_lift

    source = np.arange(8)*.006
    parent = source.copy()
    seed = parent+.045
    seed[4] = parent[4]+.0015
    frozen = np.array([True, True, False, False, False, False, True, True])
    lifted = bounded_lift(parent, seed, source, frozen, 50.)
    np.testing.assert_array_equal(lifted[frozen], parent[frozen])
    assert lifted[3]>parent[3]
    assert np.all(lifted<=seed)
    assert np.all(np.abs(np.diff(lifted)-np.diff(source))<=.004+1e-12)
