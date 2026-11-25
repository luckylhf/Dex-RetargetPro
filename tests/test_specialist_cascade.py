from dex_frax.robot_profile import POSE_SIZE
import json
import math
import pickle
import time
from types import SimpleNamespace

import numpy as np
import pytest

from dex_frax.refine_pipeline import repair_output
from dex_frax.contract import UNLIMITED_REPAIR_BUDGET
from dex_frax.v8.anatomy import configure
from dex_frax.v8.robot import Robot
from test_incremental_audit import export_neutral


def neutral_clip(folder):
    path, payload, meta, dg = export_neutral(folder, n=40)
    robot = Robot()
    configure(robot,meta['configuration']['anatomy'])
    targets = dict(root_p=dg['root_target'], root_r=dg['root_r'], torso_r=dg['torso_target'], fps=50.,
                   source_airborne=np.zeros(40,bool), body_contacts=[], body_contact_height_tolerance_m=.004,
                   sides={side: {key[len(side)+1:]: value for key,value in dg.items() if key.startswith(side+'_')}
                          for side in robot.sides})
    meta.update(leg_scale=1.,prepared_targets_sha256='fixture',stage_times_s={},elapsed_s=0.,
                self_collision={},support_constraint_failures=[],source_type='smplx')
    path.with_suffix('.json').write_text(json.dumps(meta))
    return robot, targets, path, payload, meta


def test_plan_fresh_and_continuation_share_the_specialist_to_segment_handoff(tmp_path):
    folders = [tmp_path/name for name in ('plan', 'fresh', 'continued')]
    for folder in folders:
        folder.mkdir()
        robot, targets, path, payload, meta = neutral_clip(folder)
        options=dict(segment_cascade=True,repair_budget=0.,plan_only=folder==folders[0],refine_flow='B')
        if folder==folders[2]:
            options['continue_from']=folders[1]
        result=repair_output(robot,targets,None,meta,payload,path,None,**options)
        if options['plan_only']:
            assert result['_repair_plan']['motion']==40
            assert set(json.loads((folder/'segment_plan.json').read_text())) == {'segments'}
            continue
        report=json.loads((folder/'repair.json').read_text())
        assert report['refine_flow'] == 'B'
        assert report['final_fresh_audit_equal']
        assert report['after']==result['_unified_quality']
        assert (folder/report['specialist_report']).is_file()
        segments=json.loads((folder/report['segment_report']).read_text())
        assert set(segments['phases']) == set(segments['coverage']) == {'v8'}
        assert segments['candidate_limit_s'] == 50.
        assert report['candidate_limits_s'] == {'specialist_candidate_seconds': 20, 'v8_segment_seconds': 50}
        assert report['available_budget_s'] == 0.
        assert report['temporal_repair']['stage']=='after_geometry_and_segment_solvers'
        assert json.loads((folder/'refine_state.json').read_text())['active_phase']=='segments'
        np.testing.assert_array_equal(pickle.loads(path.read_bytes())['root_rot'],payload['root_rot'])


def test_default_flow_runs_jitter_in_the_a_specialist_queue(tmp_path, monkeypatch):
    from dex_frax import acceptance, refine_pipeline, segment_refine

    robot, targets, path, payload, meta = neutral_clip(tmp_path)
    evaluate = acceptance.evaluate
    calls = []

    def mark_jitter(*args, **kwargs):
        quality = evaluate(*args, **kwargs)
        quality['flags'] = sorted(set(quality['flags']) | {'conversion_added_jitter'})
        quality['temporal_quality']['joints']['ankle_pitch_l_joint']['violating_frames'] = [18, 19]
        quality['temporal_quality']['violating_frames'] = [18, 19]
        return quality

    def reject(robot, targets, rows, original, weights, model, job, recipe, *args):
        calls.append((job['module'], recipe))
        raise ValueError('test_rejection')

    def unexpected_v8(*args, **kwargs):
        pytest.fail('Flow A entered the V8 segment cascade')

    monkeypatch.setattr(acceptance, 'evaluate', mark_jitter)
    monkeypatch.setattr(refine_pipeline, 'propose', reject)
    monkeypatch.setattr(segment_refine, 'refine', unexpected_v8)
    repair_output(robot, targets, None, meta, payload, path, None,
                  repair_budget=UNLIMITED_REPAIR_BUDGET, segment_cascade=True,
                  repair_modules={'jitter', 'segment_solvers'})
    report = json.loads((tmp_path/'repair.json').read_text())
    assert calls[:2] == [('jitter', 'temporal'), ('jitter', 'temporal_more')]
    assert report['refine_flow'] == 'A'
    assert report['policy']['name'] == 'joint_safety_before_pose_default_v7'
    assert not report['temporal_repair'] and not (tmp_path/'cascade_segments.json').exists()
    assert report['final_fresh_audit_equal']


@pytest.mark.parametrize("elapsed, budget, runs_temporal", [
    (2., None, True),
    (12., 10., False),
    (120., UNLIMITED_REPAIR_BUDGET, True),
])
def test_temporal_uses_only_remaining_shared_budget(tmp_path, monkeypatch, elapsed, budget, runs_temporal):
    from dex_frax import acceptance, refine_pipeline, segment_refine, sparse_jitter, temporal_repair

    robot, targets, path, payload, meta = neutral_clip(tmp_path)
    meta.update(input_frames=61,input_fps=30.)
    events = []
    clock = [time.monotonic()]
    real_refine = segment_refine.refine
    real_evaluate = acceptance.evaluate
    unlimited = budget == UNLIMITED_REPAIR_BUDGET
    expected_budget = None if unlimited else (50.5 if budget is None else budget)

    def geometry(*args, **kwargs):
        if unlimited:
            assert math.isinf(kwargs['deadline'])
            kwargs['specialist_step'] = None
            monkeypatch.setattr(segment_refine, 'next_segment', lambda *args: None)
        else:
            assert kwargs['deadline'] == clock[0] + expected_budget
            kwargs['deadline'] = time.monotonic() - 1.
        result = real_refine(*args, **kwargs)
        clock[0] += elapsed
        events.append('geometry')
        return result

    def evaluate(*args, **kwargs):
        events.append('audit')
        result = real_evaluate(*args, **kwargs)
        result['flags'] = sorted(set(result['flags']) | {temporal_repair.JITTER})
        return result

    monkeypatch.setattr(refine_pipeline, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(acceptance, 'evaluate', evaluate)
    monkeypatch.setattr(segment_refine, 'refine', geometry)
    monkeypatch.setattr(temporal_repair.Repair, 'jitter', lambda self: events.append('temporal'))
    monkeypatch.setattr(sparse_jitter, 'finish', lambda *args, **kwargs: [])
    repair_output(robot, targets, None, meta, payload, path, None, repair_budget=budget,
                  segment_cascade=True, repair_modules={'jitter', 'segment_solvers'}, refine_flow='B')
    report = json.loads((tmp_path / 'repair.json').read_text())
    assert events == (['audit', 'geometry', 'temporal', 'audit'] if runs_temporal else ['audit', 'geometry', 'audit'])
    assert report['budget_s'] == expected_budget
    assert report['available_budget_s'] == expected_budget
    assert report['full_input_frames_50hz'] == 101
    assert report['budget_policy'] == ('unlimited' if unlimited else 'shared_wall_time')
    assert report['shared_budget_exhausted'] is not runs_temporal
    assert report['temporal_repair']['status'] == ('completed' if runs_temporal else 'no_remaining_budget')
    assert report['final_fresh_audit_equal']
    if unlimited:
        segments = json.loads((tmp_path / 'cascade_segments.json').read_text())
        assert segments['phases']['v8']['allocated_s'] is None
        assert segments['candidate_limit_s'] == 50.


def test_direct_collision_passes_remaining_time_to_qp(monkeypatch):
    from dex_frax import refine_pipeline

    watchdog = []

    def solve(model, targets, rows, cfg, start, end, sides, options):
        watchdog.append(options['watchdog_seconds'])
        return rows, {}

    monkeypatch.setattr(refine_pipeline, 'time', SimpleNamespace(monotonic=lambda: 100.))
    monkeypatch.setattr(refine_pipeline, 'window_targets', lambda *args: {})
    monkeypatch.setattr(refine_pipeline, 'collision_seams', lambda *args: dict(freeze_start=True, freeze_end=True))
    monkeypatch.setattr(refine_pipeline, '_repair_window', solve)
    rows = np.zeros((8, POSE_SIZE))
    robot = SimpleNamespace(sides={'Left': {}, 'Right': {}})
    job = dict(module='collision', side=None, start=0, end=8)
    refine_pipeline.propose(robot, {}, rows, rows, {}, None, job, 'direct10', 103.)
    assert watchdog == [3.]
    with pytest.raises(TimeoutError, match='collision_candidate_deadline'):
        refine_pipeline.propose(robot, {}, rows, rows, {}, None, job, 'direct10', 100.)


def test_flow_c_only_blocks_overlapping_jitter_and_does_not_repeat_unchanged_timeouts():
    from dex_frax.refine_queue import WorkQueue
    from dex_frax.refine_routing import next_window_job, window_jobs

    def window(module, start, end):
        return dict(module=module, side=None, start=start, end=end,
                    core_start=start, core_end=end, severity=1)

    queue = WorkQueue(distinct_contexts=True)
    support = window('support', 2, 9)
    overlapping = window('jitter', 7, 12)
    independent = window('jitter', 16, 21)
    body = window('body', 16, 25)
    clear = np.zeros(30, bool)
    ready, blocked = window_jobs(queue, [support, overlapping, independent, body], clear, clear, clear, clear)
    assert ready == [support, independent]
    assert {item['reason'] for item in blocked} == {'waiting_for_local_geometry', 'awaiting_trusted_body_support'}
    assert next_window_job(queue, ready) == support

    queue.complete(support, False, timed=True, unchanged_timeout=True)
    assert queue.choices(support) == ['geometry10_elastic']
    assert next_window_job(queue, ready) == independent


def test_flow_c_body_support_blocks_only_jitter_overlapping_trusted_errors():
    from dex_frax.refine_queue import WorkQueue
    from dex_frax.refine_routing import window_jobs

    def window(module, start, end):
        return dict(module=module, side=None, start=start, end=end,
                    core_start=start, core_end=end, severity=1)

    queue = WorkQueue(distinct_contexts=True)
    body = window('body', 7, 20)
    overlapping = window('jitter', 8, 13)
    stable = window('jitter', 14, 19)
    jobs = [body, overlapping, stable]
    clear = np.zeros(30, bool)
    trusted, errors = clear.copy(), clear.copy()
    trusted[7:20] = True
    errors[10] = True
    ready, blocked = window_jobs(queue, jobs, trusted, errors, clear, clear)
    assert ready == [body, stable]
    assert blocked == [dict(overlapping, reason='waiting_for_local_body_support')]

    for recipe in queue.choices(body):
        queue.complete(body, False, recipe=recipe)
    assert window_jobs(queue, jobs, trusted, errors, clear, clear) == ([overlapping, stable], [])


def test_flow_c_tries_residual_floor_after_kept_support_with_collision_priority():
    from dex_frax.refine_queue import WorkQueue
    from dex_frax.refine_routing import next_window_job

    queue = WorkQueue(distinct_contexts=True)
    support = dict(module='support', side=None, core_start=21, core_end=42, severity=4)
    floor = dict(module='dense_floor', side=None, core_start=22, core_end=33, severity=2)
    other_floor = dict(floor, core_start=49, core_end=60, severity=8)
    kept = dict(module='support', start=0, end=26, status='kept')
    ready = [support, floor, other_floor]
    assert next_window_job(queue, ready) == support
    assert next_window_job(queue, ready, kept) == floor
    assert next_window_job(queue, ready, dict(kept, status='rejected')) == support

    collision = dict(module='leg_collision', side=None, core_start=25, core_end=38, severity=1)
    assert next_window_job(queue, [*ready, collision], kept) == collision


def test_flow_c_prioritizes_joint_support_and_checks_temporal_after_a_safe_change():
    from dex_frax.refine_queue import WorkQueue
    from dex_frax.refine_routing import checked_geometry_frames, next_window_job, window_jobs

    def window(module, start, end, **options):
        return dict(module=module, side=None, start=start, end=end,
                    core_start=start, core_end=end, severity=1, **options)

    queue = WorkQueue(distinct_contexts=True)
    collision = window('leg_collision', 5, 15)
    body = window('body', 8, 18, joint_support=True)
    jitter = window('jitter', 8, 11)
    jobs = [collision, body, jitter]
    clear = np.zeros(25, bool)
    trusted, errors = clear.copy(), clear.copy()
    trusted[8:18] = True
    errors[9] = True
    ready, blocked = window_jobs(queue, jobs, trusted, errors, clear, clear)
    assert next_window_job(queue, ready)['module'] == 'body'
    assert [entry['module'] for entry in blocked] == ['jitter']

    for span, expected in (([12, 14], 'body'), ([9, 10], 'jitter')):
        kept = dict(collision, status='kept', changed_span=span)
        checked = checked_geometry_frames([kept], len(clear))
        ready, _ = window_jobs(queue, jobs, trusted, errors, clear, clear, checked)
        assert next_window_job(queue, ready, kept)['module'] == expected


def test_flow_c_coupled_window_uses_the_defective_contact_phase_and_join_frames():
    from dex_frax.refine_routing import coupled_leg_window

    collision = np.zeros(40, bool)
    floor = collision.copy()
    trusted = collision.copy()
    collision[[8, 9, 16, 17, 18]] = True
    floor[13:20] = True
    edges = np.array([0, 12, 25, 40])
    assert coupled_leg_window(5, 23, collision, floor, trusted, edges) == [11, 22]
    assert coupled_leg_window(5, 12, collision, floor, trusted, edges) is None


def test_flow_c_groups_joint_tasks_sharing_one_temporal_fit():
    from types import SimpleNamespace

    from dex_frax.refine_routing import group_jitter_jobs

    robot=SimpleNamespace(names=['left_knee','right_knee','elbow'],
                          sides={'Left':{'leg':[3],'arm':[5]},'Right':{'leg':[4],'arm':[]}})
    def job(joint,start):
        return dict(module='jitter',side=joint,core_start=start,core_end=start+11,severity=2)

    grouped=group_jitter_jobs([job('left_knee',0),job('right_knee',0),
                               job('elbow',0),job('right_knee',11)],robot)
    assert [entry['jitter_joints'] for entry in grouped] == [
        ['left_knee','right_knee'],['elbow'],['right_knee']]
    assert grouped[0]['severity']==4


def test_flow_c_can_offer_alternative_after_overlapping_failures():
    from dex_frax.refine_routing import stalled_window

    segment=dict(core_start=10,core_end=20)
    failure=dict(module='support',core_start=5,core_end=15,status='rejected')
    independent=dict(failure,core_start=25,core_end=30)
    gain=dict(failure,status='kept',changed_span=[11,13])
    assert stalled_window(segment,[failure,independent,failure])
    assert not stalled_window(segment,[failure,independent])
    assert not stalled_window(segment,[failure,gain,failure])


def test_flow_c_rotates_specialists_before_taking_another_window_of_the_same_kind():
    from dex_frax.refine_queue import WorkQueue
    from dex_frax.refine_routing import next_window_job

    def window(module, start):
        return dict(module=module, side=None, start=start, end=start+11,
                    core_start=start, core_end=start+11, severity=1)

    queue=WorkQueue(distinct_contexts=True)
    collision=window('leg_collision',10)
    other_collision=window('leg_collision',50)
    support=window('support',80)
    floor=window('dense_floor',110)
    jitter=window('jitter',150)
    available=[other_collision,support,floor,jitter]
    history=[dict(collision,status='rejected')]
    assert next_window_job(queue,available,attempts=history) == support
    history.append(dict(support,status='rejected'))
    assert next_window_job(queue,available,attempts=history) == floor
    history.append(dict(floor,status='rejected'))
    assert next_window_job(queue,available,attempts=history) == jitter
