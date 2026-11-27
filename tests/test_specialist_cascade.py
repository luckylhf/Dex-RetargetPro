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


def test_legs_use_a_phase_fit_only_for_flight_or_coupled_support():
    from dex_frax.refine_routing import leg_phase_repair

    collision=np.zeros(80,bool)
    contact_error=np.zeros_like(collision)
    flight=np.zeros_like(collision)
    foot_support=np.ones_like(collision)
    flight[20:40]=True
    foot_support[flight]=False
    boundaries=np.array([0,20,40,64,80])
    collision[[28,51,74]]=True

    assert leg_phase_repair(24,34,collision,contact_error,flight,foot_support,boundaries,8)==('flight',[18,39])
    assert leg_phase_repair(48,55,collision,contact_error,flight,foot_support,boundaries,8) is None
    contact_error[51]=True
    assert leg_phase_repair(48,55,collision,contact_error,flight,foot_support,boundaries,8)==('support',[43,60])
    assert leg_phase_repair(70,77,collision,contact_error,flight,foot_support,boundaries,12)==('support',[62,80])


def test_phase_recipe_reactivates_when_its_extended_context_changes():
    from dex_frax.refine_queue import WorkQueue

    queue=WorkQueue(distinct_contexts=True)
    rows=np.zeros((72,4))
    job=dict(module='leg_collision',side=None,core_start=25,core_end=46,
             phase_leg_window=[10,60],candidate_order=['phase_leg_clearance'])
    first,=queue.configure([job],rows)
    queue.complete(first,False)
    assert not queue.choices(first)

    rows[12,0]=.01
    refreshed,=queue.configure([job],rows)
    assert queue.choices(refreshed)==['phase_leg_clearance']


def test_failed_platform_phase_immediately_tries_the_protected_leg_fallback():
    from dex_frax.refine_queue import WorkQueue
    from dex_frax.refine_routing import next_window_job

    queue=WorkQueue(distinct_contexts=True)
    platform=dict(module='support',side='Left',core_start=20,core_end=40,severity=8,
                  candidate_order=['platform_phase','platform_leg_only'])
    other=dict(module='leg_collision',side=None,core_start=45,core_end=65,severity=9)
    platform,other=queue.configure([platform,other],np.zeros((72,4)))
    queue.complete(platform,False,recipe='platform_phase')
    rejected=dict(platform,recipe='platform_phase',status='rejected',reason='foot_anchor_regression')

    assert next_window_job(queue,[other,platform],rejected,attempts=[rejected]) is platform
    assert queue.recipe(platform)=='platform_leg_only'


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


@pytest.mark.parametrize('detail', [None, [{'iteration': 1}]])
def test_default_flow_runs_jitter_in_the_a_specialist_queue(tmp_path, monkeypatch, detail):
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
        if detail is not None:
            return rows.copy(), detail
        raise ValueError('test_rejection')

    def unexpected_v8(*args, **kwargs):
        pytest.fail('Flow A entered the V8 segment cascade')

    monkeypatch.setattr(acceptance, 'evaluate', mark_jitter)
    monkeypatch.setattr(refine_pipeline, 'propose', reject)
    monkeypatch.setattr(segment_refine, 'refine', unexpected_v8)
    repair_output(robot, targets, None, meta, payload, path, None,
                  segment_cascade=True,
                  repair_modules={'jitter', 'segment_solvers'})
    report = json.loads((tmp_path/'repair.json').read_text())
    assert calls[:2] == [('jitter', 'temporal'), ('jitter', 'temporal_more')]
    if detail is not None:
        assert report['attempts'][0]['detail'] == detail
    assert report['refine_flow'] == 'A'
    assert report['budget_source'] == 'default_unlimited'
    assert not report['shared_budget_exhausted']
    assert all(job['next_recipe'] is None for job in report['pending'])
    assert report['policy']['name'] == 'specialists_then_residual_jitter_v1'
    assert report['temporal_repair']['requested'] == ['conversion_added_jitter']
    assert not (tmp_path/'cascade_segments.json').exists()
    assert report['final_fresh_audit_equal']


@pytest.mark.parametrize('order',['budgeted','phase'])
def test_a_leaves_temporal_time_after_specialists_hit_deadline(tmp_path,monkeypatch,order):
    from dex_frax import acceptance, refine_pipeline, sparse_jitter, temporal_repair

    robot,targets,path,payload,meta = neutral_clip(tmp_path)
    clock = [100.]
    events = []
    evaluate = acceptance.evaluate

    def mark_jitter(*args,**kwargs):
        quality = evaluate(*args,**kwargs)
        quality['flags'] = sorted(set(quality['flags']) | {temporal_repair.JITTER})
        quality['temporal_quality']['joints']['ankle_pitch_l_joint']['violating_frames'] = [18,19]
        quality['temporal_quality']['violating_frames'] = [18,19]
        return quality

    def spend_slice(robot,targets,rows,reference,weights,model,job,recipe,deadline,*args):
        clock[0] = deadline
        events.append('specialist')
        raise TimeoutError('test_candidate_deadline')

    monkeypatch.setattr(acceptance,'evaluate',mark_jitter)
    monkeypatch.setattr(refine_pipeline,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    monkeypatch.setattr(refine_pipeline,'propose',spend_slice)
    monkeypatch.setattr(refine_pipeline,'repair_limits',lambda:dict(
        specialist_candidate_seconds=20,v8_segment_seconds=50,a_order=order))
    monkeypatch.setattr(temporal_repair.Repair,'jitter',lambda self,**kw:events.append('temporal'))
    monkeypatch.setattr(sparse_jitter,'finish',lambda *args,**kwargs:[])
    repair_output(robot,targets,None,meta,payload,path,None,
                  repair_budget=40.,repair_modules={'jitter'})
    report=json.loads((tmp_path/'repair.json').read_text())
    assert events==['specialist','specialist','temporal']
    assert report['temporal_reserve_s']==10.
    assert report['temporal_repair']['status']=='completed'
    assert report['final_fresh_audit_equal']


@pytest.mark.parametrize("elapsed, budget, runs_temporal", [
    (120., None, True),
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
    unlimited = budget in (None, UNLIMITED_REPAIR_BUDGET)
    expected_budget = None if unlimited else budget

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
    assert report['budget_source'] == ('default_unlimited' if budget is None else 'explicit')
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
