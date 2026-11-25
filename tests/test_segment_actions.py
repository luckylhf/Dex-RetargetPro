from dex_frax.robot_profile import JOINT_COUNT
from pathlib import Path

import numpy as np
import pytest

from dex_frax.segment_planner import next_segment, plan_segments
from dex_frax.segment_refine import blend, project_joints, publish_trial, root_drop_excess


def test_unresolved_timeout_can_be_retried_once_after_fresh_segments():
    timeout = dict(segment_id='geometry-0-10', primary='geometry', core_start=0, core_end=10,
                   peak_excess=3., affected_frames=10, status='time_slice')
    fresh = dict(timeout, segment_id='geometry-20-30', core_start=20, core_end=30)
    retry_ids = {timeout['segment_id']}
    assert next_segment([timeout, fresh], [timeout], retry_ids) == fresh
    assert next_segment([timeout], [timeout], retry_ids) == timeout
    retry_ids.discard(timeout['segment_id'])
    assert next_segment([timeout], [timeout], retry_ids) is None


def test_core_ownership_and_first_tries_do_not_follow_context_overlap():
    fps = 50
    n = 175
    body = np.zeros(n)
    body[11:17] = 3
    body[20:23] = 4
    body[35:38] = 2
    body[130:135] = 6
    foot = np.zeros(n)
    foot[5:44] = 4
    evidence = {'left_hand:xy': ('body_support', body), 'shoe': ('geometry', foot)}
    contacts = np.ones((n,2))
    jobs = plan_segments(evidence,contacts,fps,np.arange(n)/fps)
    first = next_segment(jobs,[])
    assert first['core_start'] == 130
    second = next_segment(jobs,[first])
    assert second['primary'] == 'geometry'
    support = [j for j in jobs if j['primary']=='body_support']
    assert len(support)==3
    assert support[0]['core_start']==11 and support[0]['core_end']==23
    assert support[0]['end']>support[1]['start']  # seam ramps overlap; action cores do not.
    ordered=sorted(jobs,key=lambda j:j['core_start'])
    assert all(a['core_end']<=b['core_start'] for a,b in zip(ordered,ordered[1:]))
    assert 'shoe' in support[0]['symptoms']
    assert len({j['segment_id'] for j in jobs})==len(jobs)


def test_clean_source_drop_is_not_a_conversion_artifact():
    source=np.zeros((5,3))
    source[1:,2]=-.08
    robot=source.copy()
    robot[3:,2]-=.05
    np.testing.assert_allclose(root_drop_excess(robot,source),[0,0,.05,0])


def test_v8_pins_protected_poses_through_ik_support_and_projection(dex_synthetic_clip):
    from dex_frax.adaptive_refine import window_targets
    from dex_frax.v8 import convert
    from dex_frax.v8.solver import WindowAnchors, solve

    config, targets, rows = dex_synthetic_clip
    robot = convert.Robot()
    convert.configure(robot, config['anatomy'])
    baseline = rows[40:49]
    target_window = window_targets(targets, 40, 49, len(targets['root_p']))
    fixed = np.array([True, False, False, True, True, False, False, False, True])
    anchors = WindowAnchors(robot, baseline, fixed, 50.)
    weights = config['solver']
    whole_body, _ = solve(robot, target_window, weights, anchors=anchors)
    fast, _ = convert.support_solver(weights)(robot, target_window, whole_body, weights, anchors=anchors)
    legacy = dict(weights, support_refinement='legacy_slsqp')
    slow, _ = convert.support_solver(legacy)(robot, target_window, whole_body, legacy, anchors=anchors)
    proposal = fast.copy()
    proposal[:,3:] = robot.upper[3:]
    projected = blend(baseline, proposal, 0, len(baseline), 1, len(baseline)-1, ~fixed)
    project_joints(projected, anchors)

    for candidate in (whole_body, fast, slow, projected):
        np.testing.assert_array_equal(candidate[fixed], baseline[fixed])
        np.testing.assert_array_equal(candidate[:,:2], baseline[:,:2])
        assert np.all(np.abs(np.diff(candidate[:,3:], axis=0)) <= anchors.step + 1e-8)
        assert np.any(np.abs(candidate[~fixed,3:] - baseline[~fixed,3:]) > 1e-5)


def test_failed_segment_publish_restores_previous_generation(tmp_path,monkeypatch):
    import os
    old=tmp_path/'motion.pkl'
    trial_dir=tmp_path/'trial'
    trial_dir.mkdir()
    new=trial_dir/'motion.pkl'
    for suffix in ('.npz','.check.json','.pose_errors.npz','.json','.pkl'):
        old.with_suffix(suffix).write_bytes(b'old generation')
        new.with_suffix(suffix).write_bytes(b'new generation')
    replace=os.replace
    def fail(source,destination):
        if Path(source)==new.with_suffix('.json'):
            raise OSError('simulated write failure')
        return replace(source,destination)
    monkeypatch.setattr(os,'replace',fail)
    with pytest.raises(OSError,match='simulated write failure'):
        publish_trial(new,old,{'dof_pos':np.zeros((2,JOINT_COUNT))})
    for suffix in ('.npz','.check.json','.pose_errors.npz','.json','.pkl'):
        assert old.with_suffix(suffix).read_bytes()==b'old generation'
