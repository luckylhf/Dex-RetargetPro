import numpy as np

from dex_frax.protected_collision_route import counts, plan, slide_guard


def from_mask(mask):
    native = [(-.008, 4, [4]) if value else (.02, 4, []) for value in mask]
    dense = [v for row in native[:-1] for v in [row]*8]+native[-1:]
    return {'full_collision': {'samples': dense}}


def test_trigger_uses_full_native_frame_distribution_and_excludes_airborne():
    mask = np.zeros(600, bool)
    mask[150:177] = True
    air = np.zeros_like(mask)
    assert not plan(from_mask(mask),air,50)['trigger']['triggered']
    mask[180:255] = True
    air[202:208] = True
    report = plan(from_mask(mask),air,50)
    assert report['trigger']['triggered']
    assert report['trigger']['native_frames']==102
    marked = np.zeros_like(mask)
    for w in report['windows']:
        assert w['core_end']-w['core_start']<=36
        marked[w['core_start']:w['core_end']] = True
    assert np.array_equal(marked,mask & ~air)


def test_small_isolated_events_on_a_long_film_do_not_select_global_candidate():
    mask = np.zeros(5000, bool)
    mask[::100] = True
    assert not plan(from_mask(mask),np.zeros_like(mask),50)['trigger']['triggered']
    mask[:] = False
    for start in range(0,5000,100):
        mask[start:start+6] = True
    assert plan(from_mask(mask),np.zeros_like(mask),50)['trigger']['triggered']


def test_motion_of_a_single_support_pivot_cannot_be_hidden_by_anchor_radius():
    target = dict(phase=np.ones(4),pivots=np.zeros((4,3)),anchors=np.zeros((4,3)))
    def make(values):
        return {'frames': {'contact_xyz': {'Left': [np.array([v,0.,0.]) for v in values]}}}
    baseline=make([2.,2.,2.,2.])
    degraded=make([2.,2.,1.8,1.8])
    assert slide_guard({'sides': {'Left':target}},baseline,degraded,0,4)=='support_foot_xy_step_regression:Left'
    target['phase'][2:]=2
    assert slide_guard({'sides': {'Left':target}},baseline,degraded,0,4) is None


def test_report_counts_always_use_full_body_collision_audit():
    quality = {'self_collision':{'native_colliding_frames':0}, 'full_self_collision':{
        'native_colliding_frames':65,'collision_samples':557,'interpolated_colliding_intervals':71}}
    assert counts(quality)==dict(native_frames=65,swept_samples=557,interpolated_intervals=71)
