from dex_frax.refine_dispatch_budget import FiniteDispatch
from dex_frax.refine_queue import WorkQueue


def window(module, start, end, recipe, **context):
    return dict(module=module, side=None, start=start, end=end, core_start=start,
                core_end=end, severity=end-start, candidate_order=[recipe], **context)


def test_finite_dispatch_groups_contact_failures_and_visits_independent_windows():
    queue = WorkQueue()
    leg = window('leg_collision', 10, 30, 'phase_leg_clearance', phase_leg_window=[5, 45])
    support = window('support', 32, 38, 'geometry5_elastic')
    floor = window('dense_floor', 100, 111, 'solve')
    arm = window('arms', 120, 221, 'fit')
    dispatcher = FiniteDispatch(queue, [])
    pending = [leg, support, floor, arm]
    flags = ['full_self_collision_unresolved', 'sustained_arm_pose_mismatch']

    def choose(ready):
        return dispatcher.choose(pending, ready, flags, 200, lambda *_: 20,
                                 {'phase_leg_clearance'}, nominal_frames=101)

    job, details = choose([leg, arm])
    assert job == leg
    assert details['group_modules'] == ['leg_collision', 'support']
    dispatcher.observe(dict(leg, recipe='phase_leg_clearance', status='time_slice', elapsed_s=20))
    assert choose([leg, support, floor, arm])[0] == floor


def test_finite_dispatch_uses_measured_cost_and_reports_unaffordable_work():
    queue = WorkQueue()
    arm = window('arms', 0, 101, 'fit')
    short = window('arms', 200, 220, 'guided_current')
    history = [dict(arm, recipe='fit', status='time_slice', elapsed_s=20)]
    dispatcher = FiniteDispatch(queue, history)
    flags = ['sustained_arm_pose_mismatch']

    def choose(ready):
        return dispatcher.choose([arm, short], ready, flags, 6, lambda *_: 20,
                                 set(), nominal_frames=101)

    job, details = choose([arm, short])
    assert job == short
    assert details['pose_is_last_gate']
    assert not dispatcher.budget_blocked
    assert choose([arm]) == (None, None)
    assert dispatcher.budget_blocked
