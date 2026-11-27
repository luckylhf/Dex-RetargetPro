from types import SimpleNamespace

import numpy as np

from dex_frax.contact_timing import new_jitter, pulse_edges, source_grid
from dex_frax.cross_foot_repair import RECIPE, jobs


def test_cross_foot_recovery_groups_neighboring_hits_only_in_supported_floor_motion():
    model = SimpleNamespace(r=SimpleNamespace(sides={'Left':{'foot':2},'Right':{'foot':3}}),
        pairs=[(0,1),(0,2)],m=SimpleNamespace(geom_bodyid=np.array([2,3,4])))
    targets = dict(root_r=np.zeros((140,3,3)),fps=50.,
                   sides={side:dict(contact=np.ones(140),platform_index=np.full(140,-1))
                          for side in ('Left','Right')})
    samples=[(0,0,[]) for _ in range(8*140-7)]
    for frame,pair in ((40,0),(46,0),(90,1)):
        samples[8*frame]=(frame,0,[pair])
    [job]=jobs(model,targets,samples)
    assert job['candidate_order']==[RECIPE]
    assert (job['core_start'],job['core_end'])==(40,48)
    targets['sides']['Right']['platform_index'][41]=0
    assert jobs(model,targets,samples)==[]


def test_timing_only_defers_new_pulses_and_maps_original_samples():
    old=dict(temporal_quality=dict(joints={'knee':dict(violating_frames=[1])}))
    new=dict(temporal_quality=dict(joints={
        'knee':dict(violating_frames=[1,4,5],added_velocity_impulse_frames=[4,5])}))
    assert pulse_edges(old,new,3,7)==(4,)
    index=source_grid(9,{4})
    assert index[~(index==np.rint(index))].tolist()==[4.5]
    assert index[index==np.rint(index)].tolist()==list(range(9))
    assert new_jitter(old,new,index)
    new['temporal_quality']['joints']['knee']['violating_frames'].append(6)
    assert not pulse_edges(old,new,3,7)
