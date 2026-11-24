import json
import pickle

import numpy as np
from scipy.spatial.transform import Rotation

from dex_frax.pelvis_contact import RECIPE, check_motion
from dex_frax.refine_pipeline import repair_output
from dex_frax.v8.anatomy import configure
from dex_frax.v8.robot import Robot
from test_incremental_audit import export_neutral


def test_pelvis_contact_reaches_export_audit_and_is_kept(tmp_path):
    # Controlled 50 Hz contact plateau with one geometrically inferred
    # supporting body region and one planted foot. Insert a height defect.
    p, data, meta, dg = export_neutral(tmp_path, n=40)
    robot = Robot()
    configure(robot, meta['configuration']['anatomy'])
    x = np.r_[0., 0., 1., np.zeros(29)]
    robot.set(x, np.eye(3))
    side = min(robot.sides, key=lambda key: robot.point(robot.sides[key]['foot'], robot.lowest(key))[2])
    foot = robot.sides[side]
    x[2] -= robot.point(foot['foot'], robot.lowest(side))[2]
    robot.set(x, np.eye(3))
    body = foot['foot']
    name = robot.m.body(body).name
    hull = foot['hull']
    repeat = lambda v: np.repeat(np.asarray(v)[None], 40, axis=0)
    anchor = robot.point(body, foot['sole'])
    values = dict(contact=np.ones(40), pivots=repeat(foot['sole']), anchors=repeat(anchor))
    dg.update(root_target=repeat(x[:3]), source_expected_contact=np.ones(40,bool))
    dg.update({side+'_'+k:v for k,v in values.items()})
    dg.update({'body_'+name+'_mode':np.full(40,2), 'body_'+name+'_confidence':np.ones(40),
               'body_'+name+'_anchors':repeat(anchor)})
    contact=dict(body=body,body_name=name,robot_hull=hull.tolist(),source_geometry_confidence=1.,
                 mode=np.full(40,2),confidence=np.ones(40),anchors=repeat(anchor))
    meta['body_contact_models']=[{k:contact[k] for k in ('body','body_name','robot_hull','source_geometry_confidence')}]
    meta['configuration']['contact'].update(height_tolerance_m=.004,max_anchor_excursion_m=.004)
    meta.update(leg_scale=1.,prepared_targets_sha256='fixture',stage_times_s={},elapsed_s=0.,self_collision={},
                support_constraint_failures=[])
    targets=dict(root_p=dg['root_target'],root_r=dg['root_r'],torso_r=dg['torso_target'],fps=50.,
                 source_airborne=np.zeros(40,bool),
                 body_contacts=[contact],body_contact_height_tolerance_m=.004,
                 sides={key:{k[len(key)+1:]:v for k,v in dg.items() if k.startswith(key+'_')}
                        for key in robot.sides})
    data['root_pos']=repeat(x[:3]).astype(np.float32)
    data['root_pos'][19:22,2]-=.012
    p.write_bytes(pickle.dumps(data,protocol=4))
    p.with_suffix('.json').write_text(json.dumps(meta))
    np.savez_compressed(p.with_suffix('.diagnostics.npz'),**dg)
    rows=np.c_[data['root_pos'],data['dof_pos']]
    repair_output(robot,targets,rows,meta,data,p,None,repair_budget=30.,repair_modules={'body'},refine_flow='B')
    report=json.loads((p.parent/'repair.json').read_text())
    entry=next(a for a in report['attempts'] if a['recipe']==RECIPE)
    assert entry['status']=='kept', entry
    assert report['final_fresh_audit_equal']
    assert report['after']['interpolated_floor']['samples_below_minus_1mm']==0
    assert report['after']['body_support']['violating_frames']==[]
    output=pickle.loads(p.read_bytes())
    assert not np.array_equal(output['root_rot'],data['root_rot'])
    with np.load(p.with_suffix('.npz'),allow_pickle=False) as exported:
        np.testing.assert_array_equal(exported['root_rot'],output['root_rot'])
        np.testing.assert_array_equal(exported['root_pos'],output['root_pos'])
    start,stop=entry['detail']['window']
    np.testing.assert_array_equal(data['root_rot'][:start+2],output['root_rot'][:start+2])
    np.testing.assert_array_equal(data['root_rot'][stop-2:],output['root_rot'][stop-2:])


def test_root_rotation_guard_preserves_window_flight_and_existing_angle_gate():
    rows=np.zeros((12,32))
    rr=np.repeat(np.eye(3)[None],12,axis=0)
    t=dict(root_r=rr,source_root_r=rr,root_p=rows[:,:3],fps=50.,source_airborne=np.arange(12)==5)
    meters=dict(torso=np.zeros(12))
    cache=dict(frames=dict(contact_xyz={}))
    def check(angles):
        new=dict(t,root_r=Rotation.from_rotvec(np.c_[angles,np.zeros((12,2))],degrees=True).as_matrix())
        return check_motion(t,new,rows,rows,meters,meters,cache,cache,{}, {},2,10)

    allowed=np.zeros(12)
    allowed[3:5]=[.5,1.]
    assert check(allowed) is None
    allowed[5]=.5
    assert check(allowed)=='protected_contact_changed'
    allowed[5]=0.
    allowed[7]=6.23
    assert check(allowed)=='pelvis_rotation_budget'

