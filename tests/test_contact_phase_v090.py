from dex_frax.robot_profile import POSE_SIZE
import numpy as np
import pytest
from dex_frax.contact_phase import transition_windows,surface_patch
from dex_frax.contact_probe import scene_boxes,surface_at


def test_root_norm_and_contact_change_route_only_nearby_edges():
    rows=np.zeros((100,POSE_SIZE))
    rows[30,0:2]=.12
    rows[70,2]=.35
    phase=np.zeros(100);phase[30:40]=2
    targets=dict(fps=50,sides={'Left':{'phase':phase}})
    windows=transition_windows(rows,targets)
    assert len(windows)==1 and windows[0][2] in (30,31)


def test_heel_and_toe_patches_include_lateral_edge():
    hull=np.array([[x,y,0.] for x in (-1.,0.,1.) for y in (-.2,.2)])
    g=dict(hull=hull,heel=np.array([-1,0,0]),toe=np.array([1,0,0]))
    assert np.array_equal(surface_patch(g,2),hull[-2:])
    assert np.array_equal(surface_patch(g,3),hull[:2])
    assert np.array_equal(surface_patch(g,1),hull)


def test_scene_units_provenance_and_support_footprint(tmp_path):
    import json
    p=tmp_path/'scene.json'
    group=dict(boxes=[dict(id='step',center_m=[1.,1.,.1],half_size_m=[.2,.3,.1])],frame='robot_export_world_z_up_metres')
    p.write_text(json.dumps(group))
    with pytest.raises(ValueError):scene_boxes(p)
    group['provenance']='synthetic fixture'
    p.write_text(json.dumps(group))
    assert surface_at(np.array([1.,1.]),scene_boxes(p))==(.2,'step')
    assert surface_at(np.array([1.5,1.]),scene_boxes(p))==(0.,'floor')
