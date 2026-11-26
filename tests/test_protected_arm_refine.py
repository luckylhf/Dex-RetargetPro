import numpy as np

from dex_frax import robot_profile as profile
from dex_frax.adaptive_refine import window_targets
from dex_frax.experimental_arm_refine import refine
from dex_frax.v8 import convert as converter


def test_arm_fit_obeys_internal_protected_frame_seams(dex_synthetic_clip):
    cfg, targets, rows = dex_synthetic_clip
    seed = rows[40]
    rotation = targets['root_r'][40]
    robot = converter.Robot()
    converter.configure(robot, cfg['anatomy'])
    count = 13
    targets=window_targets(targets,40,40+count,len(targets['root_p']))
    targets['root_r']=np.broadcast_to(rotation,(count,3,3)).copy()
    targets['body_contacts']=[]
    protected = np.zeros(count,bool)
    protected[6:9] = True
    targets['source_airborne']=protected
    for name in ('source_support_uncertain','source_scene_support'):
        targets[name]=np.zeros(count,bool)
    baseline = np.broadcast_to(seed,(count,len(seed))).copy()
    side='Right'
    arm=robot.sides[side]
    goal=seed.copy()
    joint=arm['arm'][0]
    goal[joint]=np.clip(goal[joint]+.20,robot.lower[joint],robot.upper[joint])
    robot.set(goal,rotation)
    shoulder, elbow, wrist = [robot.point(arm[name]) for name in ('shoulder','elbow','wrist')]
    def direction(a,b):
        vector=b-a
        return vector/np.linalg.norm(vector)
    alignment = {
        'upper':direction(shoulder,elbow),
        'lower':direction(elbow,wrist),
        'hand_forward':robot.rot(arm['tcp'])@profile.TCP_FORWARD,
        'hand_width':robot.rot(arm['tcp'])@profile.TCP_WIDTH,
        'elbow_axis':robot.rot(arm['shoulder'])@profile.ELBOW_FLEXION_AXIS,
        'forearm_width':robot.rot(arm['forearm'])@profile.FOREARM_WIDTH,
    }
    targets['sides'][side].update({name:np.tile(value,(count,1)) for name,value in alignment.items()})

    result,report=refine(robot,targets,baseline,cfg['solver'],cfg['self_collision'],side,10.,3,
                         freeze_start=True,freeze_end=True,soft_similarity=True,movable_collision_only=True,
                         protected_frames=protected)

    np.testing.assert_array_equal(result[protected],baseline[protected])
    assert np.any(abs(result[~protected]-baseline[~protected])>1e-4)
    assert np.all(abs(np.diff(result[:,3:],axis=0))*targets['fps'] <= robot.speed_limits+1e-4)
    assert report['cost'] < report['initial_cost']


def test_grounded_leg_keeps_the_other_elevated_foot_at_source_time():
    from dex_frax.refine_platform import grounded_leg_windows

    times = 1.0 + np.array([0., .02, .031, .04, .06])
    meta = dict(input_fps=30., source_review={'kept_native_interval': [30, 80]},
                source_support_review={'fps': 50., 'frame_convention': 'zero_based_prepared_end_exclusive',
                                       'unresolved_platforms': [
                                           {'side': 'Left', 'start': 1, 'end': 3},
                                           {'side': 'Right', 'start': 2, 'end': 3}]})
    targets = dict(sides={side: {'contact': np.ones(5)} for side in ('Left', 'Right')},
                   source_support_uncertain=np.ones(5, bool),
                   source_airborne=[False, False, True, False, False])
    windows = grounded_leg_windows(targets, meta, times)
    np.testing.assert_array_equal(windows['Right'], [False, True, False, False, False])
    assert not windows['Left'].any()


def test_high_stationary_foot_is_distinguished_from_ballistic_flight():
    from dex_frax.refine_platform import inferred_swing_leg_windows

    times = np.arange(21)/50.
    stable = np.tile([0.,0.,.4], (len(times),1))
    moving = stable.copy()
    moving[:,0] = times
    names = ['Hips','Neck','Head'] + [
        side+joint for side in ('Left','Right')
        for joint in ('UpLeg','Leg','Foot','Toe','Arm','ForeArm','Hand')]
    points = {name:stable.copy() for name in names}
    targets = dict(p=points, floor=0., source_airborne=np.ones(len(times),bool), sides={
        'Left':dict(heel=stable,toe=stable), 'Right':dict(heel=moving,toe=moving)})
    assert inferred_swing_leg_windows(targets,times)['Right'][10]
    for value in points.values():
        value[:,2] -= 4.903325*(times-times[10])**2
    assert not inferred_swing_leg_windows(targets,times)['Right'][10]


def test_inferred_platform_uses_the_regular_support_target_and_checks_export(dex_synthetic_clip):
    from dex_frax.scene_support import audit_gravity, gravity_regression
    from dex_frax.source_support import apply

    cfg,targets,rows = dex_synthetic_clip
    count = 21
    targets = window_targets(targets,0,count,len(rows))
    robot = converter.Robot()
    converter.configure(robot,cfg['anatomy'])
    rows = np.repeat(rows[:1],count,axis=0)
    rows[:,2] += .4
    rotation = np.repeat(targets['root_r'][:1],count,axis=0)
    names = ['Hips','Neck','Head'] + [side+joint for side in robot.sides
                                      for joint in ('UpLeg','Leg','Foot','Toe','Arm','ForeArm','Hand')]
    targets['p'] = {name:np.tile([0.,0.,.8],(count,1)) for name in names}
    targets['source_airborne'] = np.ones(count,bool)
    targets['floor'] = 0.
    robot.set(rows[0],rotation[0])
    for side,g in robot.sides.items():
        foot = targets['sides'][side]
        point = robot.point(g['foot'])
        r = robot.rot(g['foot'])
        source = np.tile([0.,0.,.4],(count,1))
        if side=='Right':
            source[:,0] = np.arange(count)/targets['fps']
        foot.update(heel=source.copy(), toe=source.copy(),
                    foot_r=np.repeat(np.eye(3)[None],count,axis=0),
                    raw_foot_p=np.repeat(point[None],count,axis=0),
                    raw_foot_r=np.repeat(r[None],count,axis=0))

    evidence = apply(targets,robot,[],infer_gravity=True)
    assert not evidence['unresolved_platforms']
    inferred = evidence['gravity_inferred_platforms']
    assert len(inferred)==1 and inferred[0]['side']=='Left'
    assert targets['source_scene_support'][10] and not targets['source_airborne'][10]
    assert targets['sides']['Left']['contact'][10]==0
    diagnostics = {side+'_'+key:foot[key] for side,foot in targets['sides'].items()
                   for key in ('platform_index','platform_anchor')}
    clean = audit_gravity(robot,rows,rotation,diagnostics,0)
    assert clean['support_targets']==count
    assert not clean['support_violations'] and not clean['penetrating_samples']

    rows[10,2] -= .003
    bad = audit_gravity(robot,rows,rotation,diagnostics,0)
    assert bad['support_violations'][0]['frame']==10
    assert any(item['sample']%8 for item in bad['penetrating_samples'])
    assert gravity_regression({'inferred_platform_support': clean},{'inferred_platform_support': bad})

