from dex_frax.robot_profile import JOINT_COUNT
import json
import pickle

import numpy as np

from dex_frax.ground_settle import episodes
from dex_frax.refine_pipeline import repair_output
from dex_frax.v8.anatomy import configure
from dex_frax.v8.robot import Robot
from test_incremental_audit import export_neutral


def test_requires_independent_source_and_robot_evidence_ground_and_support():
    n=9
    floor=np.full(((n-1)*8+1,1),-.02)
    t=dict(sides={'Left':dict(contact=np.ones(n))},source_airborne=np.zeros(n,bool))
    assert not episodes({'heel':np.ones(n)},floor,t)[0]
    assert not episodes({'heel':-np.ones(n)},-floor,t)[0]
    t['source_airborne'][3:5]=True
    t['sides']['Left']['contact'][7:]=0
    jobs,_,counts=episodes({'heel':-np.ones(n)},floor,t)
    assert sorted((j['core_start'],j['core_end']) for j in jobs)==[(0,3),(5,7)]
    assert counts['protected_overlap_frames']==2 and counts['unmarked_overlap_frames']==2


def test_first_pass_lifts_and_settles_complete_supported_episode(tmp_path):
    n=40
    p,data,meta,dg=export_neutral(tmp_path,n)
    robot=Robot();configure(robot,meta['configuration']['anatomy'])
    row=np.r_[0.,0.,1.,np.zeros(JOINT_COUNT)]
    robot.set(row,np.eye(3))
    side=min(robot.sides,key=lambda s:robot.point(robot.sides[s]['foot'],robot.lowest(s))[2])
    g=robot.sides[side]
    row[2]-=robot.point(g['foot'],robot.lowest(side))[2]
    robot.set(row,np.eye(3))
    repeat=lambda x:np.repeat(np.asarray(x)[None],n,axis=0)
    anchor=robot.point(g['foot'],g['sole'])
    dg[side+'_anchors']=repeat(anchor)
    dg[side+'_pivots']=repeat(g['sole'])
    dg[side+'_contact']=np.ones(n)
    dg['root_target']=repeat(row[:3])
    dg['source_expected_contact']=np.ones(n,bool)
    sides={s:{key[len(s)+1:]:value for key,value in dg.items() if key.startswith(s+'_')} for s in robot.sides}
    for s,foot in sides.items():
        for mark in ('heel','toe'):
            foot[mark]=repeat([0.,0.,-.012 if s==side else .02])
    targets=dict(sides=sides,root_p=dg['root_target'],root_r=dg['root_r'],torso_r=dg['torso_target'],
                 floor=0.,ground_calibration={'method':'fixture_measured'},fps=50.,source_airborne=np.zeros(n,bool),
                 body_contacts=[],body_contact_height_tolerance_m=.004)
    meta.update(source_type='smplx',leg_scale=1.,prepared_targets_sha256='fixture',stage_times_s={},elapsed_s=0.,
                self_collision={},support_constraint_failures=[])
    data['root_pos']=repeat(row[:3]).astype(np.float32)
    data['root_pos'][:,2]-=.012
    p.write_bytes(pickle.dumps(data,protocol=4));p.with_suffix('.json').write_text(json.dumps(meta))
    np.savez_compressed(p.with_suffix('.diagnostics.npz'),**dg)
    repair_output(robot,targets,None,meta,data,p,None,repair_budget=35.,segment_cascade=True,
                  repair_modules={'ground_settle'},refine_flow='B')
    report=json.loads((p.parent/'ground_settle.json').read_text())
    job,=report['segments']
    assert job['core_start']==0 and job['core_end']==n and job['status']=='kept',job
    assert job['common_lift_m']>.012 and job['selected']=='vertical_lift_and_settle'
    after=job['trials'][-1]
    assert after['after_swept_min_m']>=-.000002
    assert after['supports'][side+'_foot']['max_abs_z_m']<.001
    assert after['supports'][side+'_foot']['max_xy_m']<.001
    assert json.loads((p.parent/'repair.json').read_text())['final_fresh_audit_equal']
