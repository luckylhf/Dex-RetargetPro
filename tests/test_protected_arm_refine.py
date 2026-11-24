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

