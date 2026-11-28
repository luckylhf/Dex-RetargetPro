import hashlib
import argparse
from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dex_frax import memory_pool, memory_pool_batch, refine_resume
from dex_frax.cli import add_repair_budget_option
from dex_frax.dynamic_resources import CpuAdmission
from dex_frax.memory_pool import BASE, DANGER, GIB, MIB, JointPool, Lease, Pool, requested_total
from dex_frax.memory_pool_batch import grant_request, reply_workspace
from dex_frax.refine_checkpoint import write


def test_refine_budget_option_has_default_and_unlimited_modes():
    parser=argparse.ArgumentParser()
    add_repair_budget_option(parser)
    assert parser.parse_args([]).repair_budget is None
    assert parser.parse_args(['--repair-budget','unlimited']).repair_budget=='unlimited'
    with pytest.raises(SystemExit):
        parser.parse_args(['--repair-budget','1800'])


def test_shared_grants_do_not_double_count_resident_memory():
    pool = Pool()
    snapshot = dict(available=11*GIB, cgroup_headroom=30*GIB)
    lease = Lease(resident=BASE)
    assert pool.allows(snapshot, [lease], 2*GIB)
    lease.grant += 2*GIB
    assert not pool.allows(snapshot, [lease], 2*GIB)
    lease.resident += GIB
    snapshot['available'] -= GIB
    assert pool.remaining(snapshot, [lease])[0] == 9*GIB
    assert lease.return_unused() == GIB
    assert lease.grant == BASE+GIB


def test_shared_pool_uses_distinct_restart_and_startup_watermarks():
    pool = Pool()
    snapshot = dict(available=DANGER-1, cgroup_headroom=30*GIB)
    pool.update(snapshot, [])
    snapshot['available'] = 10*GIB
    for _ in range(2):
        pool.update(snapshot, [])
        assert not pool.allows(snapshot, [], 0)
    pool.update(snapshot, [])
    assert pool.allows(snapshot, [], 2*GIB)
    assert not pool.allows(snapshot, [], BASE, start=True)

    snapshot = dict(available=40*GIB, cgroup_headroom=255*MIB)
    pool.update(snapshot, [])
    assert not pool.allows(snapshot, [], 0)


@pytest.mark.parametrize(('mode','watermarks'), [
    ('fast', (2., 2., 2.)),
    ('refine', (8., 10., 12.)),
])
def test_pool_cli_applies_mode_memory_defaults(monkeypatch, mode, watermarks):
    monkeypatch.setattr(memory_pool_batch.sys, 'argv',
                        ['dex-frax-pool', 'manifest.json', 'outputs', '--mode', mode])
    run = Mock(return_value=('complete', {}))
    monkeypatch.setattr(memory_pool_batch, 'run', run)
    memory_pool_batch.main()
    assert run.call_args.args[0].memory_watermarks == watermarks


def test_compiler_queue_and_memory_wait_are_distinct():
    request = dict(phase='waiting_compile', compile_possible=True)
    assert memory_pool_batch.wait_reason(request, 'owner') == 'compiler'
    assert memory_pool_batch.wait_reason(request, None) == 'memory'
    assert memory_pool_batch.wait_reason(dict(request, compile_possible=False), 'owner') == 'memory'
    assert memory_pool_batch.wait_reason(dict(phase='audit'), 'owner') == 'memory'
    assert memory_pool_batch.wait_reason(None, 'owner') is None


def test_fast_tail_waits_at_most_one_observed_cold_start_for_a_warm_worker():
    due = memory_pool_batch.cold_start_due
    # A short queue can wait for a warm worker. The grace period after the
    # most recent dispatch (cold or reused) ends exactly at the measured cost.
    assert not due(pending=2, active=2, now=103., last_dispatch=100., warmup_s=4.)
    assert due(pending=2, active=2, now=104., last_dispatch=100., warmup_s=4.)
    # Never delay a cold start when capacity would otherwise sit unused
    # behind a deep queue, no worker remains, or we have no warmup sample.
    assert due(pending=3, active=2, now=100., last_dispatch=100., warmup_s=4.)
    assert due(pending=1, active=0, now=100., last_dispatch=100., warmup_s=4.)
    assert due(pending=1, active=2, now=100., last_dispatch=100., warmup_s=0.)


@pytest.fixture
def simulated_fast_pool(monkeypatch, tmp_path):
    batch = memory_pool_batch
    clock = 0
    processes = {}

    def make_pipe():
        inbox = deque()
        slot = SimpleNamespace(poll=lambda: bool(inbox), recv=inbox.popleft, close=lambda: None)
        slot.inbox = inbox
        return slot, slot

    def make_process(target, args):
        pipe, job, _, cpu = args
        process = SimpleNamespace(pid=len(processes)+100, exitcode=None, close=lambda: None, join=lambda _: None)
        process.is_alive = lambda: process.exitcode is None
        process.job = job
        process.age = 0
        process.start = lambda: processes.update({process.pid: (process, pipe)})

        def send(message):
            if message is None:
                process.exitcode = 0
            elif message != 'grant':
                process.job, _ = message
                process.age = 0

        pipe.send = send
        return process

    monkeypatch.setattr(batch, 'mp', SimpleNamespace(get_context=lambda _: SimpleNamespace(
        Process=make_process, Pipe=make_pipe)))
    monkeypatch.setattr(batch, 'pressure_high', lambda: False)
    monkeypatch.setattr(batch, 'publications', lambda folder, job: dict(status='review'))

    def run(available, advance, *, cpus=1, resident=2*GIB, cpu_limit=None, keys=('a', 'b')):
        def tick(*args, **kwargs):
            nonlocal clock
            clock += 1
            assert clock < 18, 'simulated batch stalled'
            for process, pipe in processes.values():
                if process.is_alive() and process.job is not None:
                    process.age += 1
                    if process.age == 1:
                        pipe.inbox.append(dict(model_cache=dict(models={'verified': 'mjbs'})))
                    if advance(process, pipe.inbox):
                        pipe.inbox.append(dict(result='returned'))
                        process.job = None

        def memory():
            live = [p for p, _ in processes.values() if p.is_alive()]
            room = available(clock, live)
            return dict(total=16*GIB, available=room, cgroup_headroom=room)

        monkeypatch.setattr(batch, 'time', SimpleNamespace(monotonic=lambda: clock, time_ns=lambda: clock,
                                                          time=lambda: clock, sleep=tick))
        monkeypatch.setattr(batch, 'wait', tick)
        monkeypatch.setattr(batch.resources, 'memory', memory)
        monkeypatch.setattr(batch.resources, 'cpu_count', lambda: cpus)
        monkeypatch.setattr(batch.resources, 'cpu_groups', lambda: [[cpu] for cpu in range(cpus)])
        monkeypatch.setattr(batch, 'tree_pss', lambda _: resident)
        monkeypatch.setattr(batch, 'CpuAdmission', lambda *args: SimpleNamespace(
            sample=lambda _: cpu_limit(clock) if cpu_limit else cpus, report=dict))
        jobs = [dict(key=key, frames=50, profile='fast', source='bvh') for key in keys]
        state, completed = batch.schedule(jobs, tmp_path, 1, cpus, memory(), (2*GIB,)*3)
        assert state == 'complete'
        assert {key: info['status'] for key, info in completed.items()} == dict.fromkeys(keys, 'review')
        status = batch.read(tmp_path/'status.json')
        assert status['idle'] == {} and status['pool']['unused_bytes'] == 0
        events = [json.loads(line) for line in (tmp_path/'memory_events.jsonl').read_text().splitlines()]
        return status, events

    return run


@pytest.mark.parametrize('condition', ['roomy', 'pressure', 'startup_tight'])
def test_fast_worker_reused_or_reclaimed_on_pressure(simulated_fast_pool, condition):
    def available(clock, live):
        if live and live[0].job is None:
            if condition == 'pressure':
                return GIB
            if condition == 'startup_tight' and clock < 4:
                return 2*GIB + 64*MIB
        return 8*GIB

    status, events = simulated_fast_pool(available, lambda process, inbox: True)
    pressure = condition == 'pressure'
    assert status['counters']['worker_start'] == (2 if pressure else 1)
    assert status['counters'].get('worker_reuse', 0) == (0 if pressure else 1)
    reasons = [event['reason'] for event in events if event['event']=='idle_reclaim']
    assert reasons==(['memory_pressure','queue_drained'] if pressure else ['queue_drained'])
    if condition == 'startup_tight':
        reuse, = [event for event in events if event['event'] == 'worker_reuse']
        assert reuse['time'] >= 4


def test_fast_releases_startup_credit_before_dispatching_another_file(simulated_fast_pool):
    def advance(process, inbox):
        if process.age == 1:
            inbox.append(dict(request=dict(phase='ik_block', pss_bytes=GIB,
                                           required_peak_bytes=GIB+128*MIB)))
        return process.age == 4

    # 7 GiB with 2 GiB protected supports one cold start. A second fits once
    # the first reaches an IK block with a measured 128 MiB workspace.
    status, events = simulated_fast_pool(lambda clock, live: 7*GIB - len(live)*GIB,
                                         advance, cpus=2, resident=GIB)
    assert status['max_running'] == 2
    start_second, = [e for e in events if e['event'] == 'worker_start' and e['key'] == 'b']
    finish_first, = [e for e in events if e['event'] == 'worker_idle' and e['key'] == 'a']
    assert start_second['time'] < finish_first['time']


def test_fast_reuses_an_eligible_warm_cpu_when_the_cpu_limit_drops(simulated_fast_pool):
    def advance(process, inbox):
        if process.job['key'] == 'a':
            if process.age == 1:
                return False
            process.exitcode = 0  # a completed, but its worker is no longer reusable
        return True

    # Both first results arrive together. Only CPU 1 has a warm worker.
    # A host CPU target of one still permits picking CPU 1 for the next job.
    status, events = simulated_fast_pool(lambda clock, live: 16*GIB, advance,
                                         cpus=2, cpu_limit=lambda tick: 2 if tick < 2 else 1,
                                         keys=('a', 'b', 'c'))
    assert status['counters']['worker_start'] == 2
    assert [(e['key'], e['cpu']) for e in events if e['event'] == 'worker_reuse'] == [('c', 1)]


def test_block_request_uses_reported_workspace():
    event = dict(pss_bytes=BASE, phase='ik_block', required_peak_bytes=BASE+55*MIB)
    assert requested_total(event, 16000) == BASE+55*MIB


def test_fast_stage_requests_preserve_native_workspace_and_cold_start_protection():
    event = dict(pss_bytes=GIB, phase='loading')
    fast = dict(profile='fast', source='bvh')
    assert requested_total(event, 50, **fast) == BASE
    warm = requested_total(event, 50, warm=True, **fast)
    assert GIB < warm < BASE
    assert warm == memory_pool.fast_warm_grant(GIB, 50)

    # The native block reports only its own PSS and workspace. Another
    # 256 MiB already resident in children must be included in the total.
    block = dict(event, phase='ik_block', required_peak_bytes=GIB+128*MIB)
    assert requested_total(block, 50, resident=GIB+256*MIB, **fast) == GIB+384*MIB
    assert requested_total(block, 50) == BASE
    assert requested_total(dict(block, first_block=True), 50, **fast) >= BASE
    assert requested_total(dict(event, phase='preprocessing'), 50, profile='fast', source='smplx') >= BASE
    assert requested_total(dict(event, phase='unrecognized_operator'), 50, **fast) >= BASE
    audit = dict(event, phase='audit')
    assert requested_total(audit, 12000, **fast) > requested_total(audit, 50, **fast)
    compiling = dict(event, phase='waiting_compile', compile_possible=True)
    assert requested_total(compiling, 50, **fast) >= 3840*MIB
    assert requested_total(compiling, 50, resident=5*GIB, **fast) >= 5*GIB+512*MIB


def test_deadlock_recovery_releases_priority_after_the_head_receives_its_grant():
    request = dict(phase='local_repair')
    head = dict(job=dict(key='head'), request=request, lease=Lease(), pipe=Mock())
    other = dict(job=dict(key='other'), request=request, lease=Lease(), pipe=Mock())
    head['pipe'].send.side_effect = BrokenPipeError
    assert grant_request(head, 64*MIB, 'head') == (False, 'head')
    assert head['request'] is request and head['lease'].grant == BASE

    assert grant_request(other, 32*MIB, 'head') == (True, 'head')
    head['pipe'].send.side_effect = None
    assert grant_request(head, 64*MIB, 'head') == (True, None)
    head['pipe'].send.assert_called_with('grant')
    assert head['request'] is None and head['lease'].grant == BASE+64*MIB


def test_cpu_and_memory_are_both_required_for_admission():
    pool = JointPool([[0, 1], [2, 3], [4, 5]])
    lease = Lease(cpu=1, resident=BASE)
    leases = [lease]
    roomy = dict(available=16*GIB, cgroup_headroom=16*GIB)
    tight_host = dict(roomy, available=11*GIB)
    request = dict(cores=1, workspace_bytes=512*MIB, cgroup_headroom=16*GIB)

    assert pool.startup_cpu(roomy, leases, 3) == 2
    assert pool.startup_cpu(tight_host, leases, 3) is None
    assert pool.startup_cpu(roomy, leases, 1) is None
    assert pool.workspace_cpus(roomy, leases, 3, lease, **request) == (2,)
    assert pool.workspace_cpus(roomy, leases, 1, lease, **request) == ()
    assert pool.workspace_cpus(roomy, leases, 3, lease, **dict(request, cgroup_headroom=1500*MIB)) == ()
    assert pool.workspace_cpus(dict(roomy, cgroup_headroom=1500*MIB), leases, 3, lease, **request) == ()

    leases.extend([Lease(cpu=2, resident=BASE), Lease(cpu=4, resident=BASE)])
    assert pool.startup_cpu(roomy, leases, 6) == 0
    assert pool.workspace_cpus(roomy, leases, 6, lease, **request) == ()


def test_external_cpu_load_reduces_new_admissions_without_counting_our_worker_twice():
    clock = 0
    process = [(5, 0), (5, 100)]
    samples = [{cpu: (100, 100) for cpu in range(4)},
               {0: (200, 100), 1: (200, 100), 2: (200, 100), 3: (200, 200)}]
    admission = CpuAdmission(range(4), 4, clock=lambda: clock, read_cpu=lambda _: samples[clock],
                             read_process=lambda _: process[clock])
    assert admission.sample([42]) == 0
    clock = 1
    assert admission.sample([42]) == 2
    assert admission.report()['external_cpu_equivalents'] == 2
    pool = JointPool([[0], [1], [2], [3]])
    leases = [Lease(cpu=0, resident=BASE), Lease(cpu=1, resident=BASE)]
    assert pool.startup_cpu(dict(available=32*GIB, cgroup_headroom=32*GIB), leases, admission.target) is None


def test_workspace_preserves_phase_reservation_and_releases_only_unused_bytes():
    lease = Lease(cpu=0, grant=BASE+256*MIB, resident=BASE)
    lease.borrow((2,), 512*MIB)
    lease.resident += 384*MIB  # Native workspace is still resident after the helpers join.
    assert lease.return_workspace() == 384*MIB
    assert lease.borrowed_cpus == () and lease.grant == BASE+256*MIB
    assert lease.unused == 0


def test_failed_workspace_reply_leaves_cpu_and_memory_unreserved():
    request = dict(workspace_bytes=512*MIB)
    worker = dict(workspace_request=request, lease=Lease(cpu=0), pipe=Mock())
    worker['pipe'].send.side_effect = BrokenPipeError
    assert not reply_workspace(worker, (2,))
    assert worker['lease'] == Lease(cpu=0)
    assert worker['workspace_request'] is request


def test_macos_memory_snapshot_includes_operator_children(monkeypatch):
    monkeypatch.setattr(memory_pool_batch, 'sys', SimpleNamespace(platform='darwin'))
    table = '100 1 100\n200 100 90\n201 200 20\n300 1 900\n'
    monkeypatch.setattr(memory_pool_batch, 'subprocess', SimpleNamespace(check_output=lambda *a, **kw: table))
    assert memory_pool_batch.tree_pss(100) == 210*1024


@pytest.mark.parametrize(('profile', 'status', 'training'), [
    ('refine', 'accepted', True),
    ('fast', 'review', False),
])
def test_pool_counts_completed_output_for_its_profile_and_target_assets(tmp_path, profile, status, training):
    from dex_frax.robot_profile import MODEL_XML

    assets = {MODEL_XML: 'robot', 'foot_mesh': 'geometry'}
    job = dict(input_sha256='source', assets=assets, profile=profile)
    contract = dict(profile=profile)
    if profile == 'refine':
        contract['refine_flow'] = 'A'
    identity = dict(input_sha256='source', assets=assets, fingerprint='recorded',
                    conversion_contract=contract)
    quality = dict(status=status, flags=[], frames=10, model_sha256='robot',
                   full_self_collision=dict(collision_samples=0))
    (tmp_path/'identity.json').write_text(json.dumps(identity))
    files = {'quality.json': json.dumps(quality).encode(), 'motion.npz': b'motion', 'motion.pkl': b'poses'}
    if training:
        files['training.npz'] = b'training'
    for name, data in files.items():
        (tmp_path/name).write_bytes(data)
    run = dict(status='complete', quality=status, fingerprint='recorded',
               files={name:hashlib.sha256(data).hexdigest() for name,data in files.items()})
    (tmp_path/'run.json').write_text(json.dumps(run))
    assert memory_pool_batch.publications(tmp_path, job)['training'] is training
    assert memory_pool_batch.publications(tmp_path, dict(job, assets=dict(assets, foot_mesh='other'))) is None
    other = 'fast' if profile == 'refine' else 'refine'
    assert memory_pool_batch.publications(tmp_path, dict(job, profile=other)) is None
    if not training:
        (tmp_path/'training.npz').write_bytes(b'invalid training')
        assert memory_pool_batch.publications(tmp_path, job) is None


def test_operator_returns_loan_on_exception_and_uses_serial_without_a_gate(monkeypatch):
    monkeypatch.setattr(memory_pool, '_workspace_gate', None)
    with memory_pool.operator_workspace(1, 512*MIB) as cpus:
        assert cpus == ()

    gate = Mock(return_value=(2,))
    memory_pool.set_workspace_gate(gate)
    with pytest.raises(RuntimeError, match='simulated operator failure'):
        with memory_pool.operator_workspace(1, 512*MIB) as cpus:
            assert cpus == (2,)
            raise RuntimeError('simulated operator failure')
    assert [call.args for call in gate.call_args_list] == [(1, 512*MIB), ()]


def test_batch_simulates_priority_pressure_recovery_and_crash(monkeypatch, tmp_path):
    batch = memory_pool_batch
    clock = 0
    processes = {}
    pipes = {}
    replies = {}
    residents = {}
    assigned_models = {}
    killpg = Mock()

    def make_process(target, args):
        pipe, job, _, cpu = args
        key = job['key']
        process = SimpleNamespace(pid=100+cpu, exitcode=None, join=Mock(), close=Mock())
        process.is_alive = lambda: process.exitcode is None

        def start():
            processes[key] = process
            pipes[key] = pipe.inbox
            replies[key] = pipe.sent
            residents[process.pid] = BASE
            assigned_models[key] = job['models']

        process.start = start
        return process

    def make_pipe():
        inbox = deque()
        sent = []
        pipe = SimpleNamespace(inbox=inbox, sent=sent, poll=lambda: bool(inbox), recv=inbox.popleft,
                               close=Mock(), send=lambda reply: sent.append((clock, reply)))
        return pipe, pipe

    def workspace(amount=512*MIB, room=16*GIB):
        return dict(workspace_request=dict(cores=1, workspace_bytes=amount, cgroup_headroom=room))

    def tick(*args, **kwargs):
        nonlocal clock
        clock += 1
        if clock == 1:
            pipes['a'].append(dict(request=dict(phase='local_repair', pss_bytes=BASE,
                                                required_peak_bytes=BASE+GIB)))
        if clock == 2:
            pipes['a'].append(dict(model_cache=dict(models={'verified':'mjbs'}, preflight_s=1)))
            pipes['a'].append(workspace())
        if clock == 4:
            pipes['a'].append(workspace(2*GIB, 1650*MIB))
            pipes['b'].append(workspace())
        if clock == 5:
            residents[processes['b'].pid] = BASE+256*MIB
            pipes['b'].append(dict(workspace_returned=True))
            pipes['a'].append(workspace())
        if clock in (7, 8, 9):
            pipes['a'].append(workspace())
        if clock == 10:
            processes['a'].exitcode = -9
            pipes['b'].append(workspace())
        if clock == 11:
            residents[processes['b'].pid] = BASE
            pipes['b'].append(dict(workspace_returned=True))
            pipes['b'].append(dict(result='returned'))
            processes['b'].exitcode = 0
        assert clock < 14, 'simulated batch stalled'

    def memory():
        # At t=2 the necessary grant issued at t=1 still holds 1 GiB. Without
        # that reservation the second task would fit above the startup floor.
        available = {1: 11*GIB, 2: 15*GIB, 5: DANGER-1}.get(clock, 16*GIB)
        return dict(total=32*GIB, available=available, cgroup_headroom=32*GIB)

    # The supervisor uses production policies and message handling, with deterministic I/O.
    monkeypatch.setattr(batch, 'mp', SimpleNamespace(get_context=lambda _: SimpleNamespace(
        Process=make_process, Pipe=make_pipe)))
    monkeypatch.setattr(batch, 'time', SimpleNamespace(monotonic=lambda: clock, time_ns=lambda: clock,
                                                       time=lambda: clock, sleep=tick))
    monkeypatch.setattr(batch, 'wait', tick)
    monkeypatch.setattr(batch, 'os', SimpleNamespace(killpg=killpg))
    monkeypatch.setattr(batch, 'pressure_high', lambda: False)
    monkeypatch.setattr(batch, 'tree_pss', residents.__getitem__)
    monkeypatch.setattr(batch, 'publications', lambda folder, job: dict(status='review') if job['key']=='b' else None)
    monkeypatch.setattr(batch, 'CpuAdmission', lambda *args: SimpleNamespace(
        sample=lambda pids: 2 if clock == 8 else 4, report=lambda: {}))
    monkeypatch.setattr(batch.resources, 'memory', memory)
    monkeypatch.setattr(batch.resources, 'cpu_count', lambda: 8)
    monkeypatch.setattr(batch.resources, 'cpu_groups', lambda: [[0, 1], [2, 3], [4, 5], [6, 7]])

    jobs = [dict(key=key, frames=50) for key in ('a', 'b')]
    state, completed = batch.schedule(jobs, tmp_path, 2, 4, memory())
    assert state == 'complete'
    assert completed['a']['status'] == 'failed' and completed['b']['status'] == 'review'
    assert replies['a'] == [(1, 'grant'), (2, ()), (4, ()), (5, ()), (7, ()), (8, ()), (9, (4,))]
    assert replies['b'] == [(4, (4,)), (10, (0,))]
    killpg.assert_called_once_with(processes['a'].pid, batch.signal.SIGKILL)

    events = [json.loads(line) for line in (tmp_path/'memory_events.jsonl').read_text().splitlines()]
    starts = [(ev['key'], ev['cpu'], ev['time']) for ev in events if ev['event']=='worker_start']
    assert starts == [('a', 0, 0), ('b', 2, 3)]
    assert assigned_models == {'a':None, 'b':{'verified':'mjbs'}}
    assert batch.read(tmp_path/'model_cache.json')['models'] == assigned_models['b']
    assert [(ev['event'], ev['time']) for ev in events if ev['event'] in ('freeze','recover')] == [('freeze', 5), ('recover', 8)]
    returns = [(ev['unused_bytes'], ev['resident_bytes']) for ev in events if ev['event']=='workspace_return']
    assert returns == [(256*MIB, BASE+256*MIB), (768*MIB, BASE)]
    status = json.loads((tmp_path/'status.json').read_text())
    assert status['running'] == {} and status['pool']['unused_bytes'] == 0


def test_v3_checkpoint_migration_keeps_artifact_and_implementation_checks(tmp_path):
    root = Path(refine_resume.__file__).parent
    same = 'unchanged'
    setup = dict(input_sha256=same, assets=same, frax_implementation=same, runtime=same,
                 conversion_contract=dict(profile='refine', refine_flow='A'))
    previous = dict(setup, implementation={
        'refine_resume.py': '634f75d49fbaa93ed99fcad4338553896c086ee22815e3f7fcb49c5b83127c6f',
        'managed_batch.py': '2d8939296717d1379ed129899599371d8dddb7bf741fa3a05240e2e81deaf7eb',
        'v8/audit.py': same,
    })
    current = dict(setup, implementation={
        **{name: hashlib.sha256((root/name).read_bytes()).hexdigest()
           for name in ('refine_resume.py','managed_batch.py','memory_pool.py','memory_pool_batch.py')},
        'v8/audit.py': same,
    })
    (tmp_path/'identity.json').write_text(json.dumps(previous))
    base = tmp_path/'frax_baseline'
    base.mkdir()
    (base/'motion.pkl').write_bytes(b'initial')
    (tmp_path/'motion.pkl').write_bytes(b'motion')
    (tmp_path/'avoidance_reference.npy').write_bytes(b'reference')
    write(tmp_path, dict(schema=3, status='running',
                         baseline_sha256=hashlib.sha256(b'initial').hexdigest(),
                         output_sha256=hashlib.sha256(b'motion').hexdigest(),
                         avoidance_reference_sha256=hashlib.sha256(b'reference').hexdigest()))
    assert refine_resume.verify(tmp_path, current)['schema'] == 3

    # A checkpoint made just before the joint scheduler change keeps its
    # implementation identity checks for all solver and acceptance modules.
    current['implementation']['resources.py'] = hashlib.sha256((root/'resources.py').read_bytes()).hexdigest()
    pre_joint = dict(current, implementation={**current['implementation'],
        'resources.py': 'e2912af478519bbb53fe4936f5b0b1285ab2932d0cc4e29696aba5c280f7edd5',
        'memory_pool.py': 'a7eb8c33c2cbdeff754be5ea3126ef90ea268545e0689a9df8c1fb672d2bb11d',
        'memory_pool_batch.py': 'fb72199814dd5a0a47a22160b2f2bc841cab4707bdcb044095baaa581a9c962d',
        'refine_resume.py': '07a3c74f45e0bbe80b628714582fdaefad5befb2c4b9325af922e6c8c6556a3b',
    })
    (tmp_path/'identity.json').write_text(json.dumps(pre_joint))
    assert refine_resume.verify(tmp_path, current)['schema'] == 3

    current['implementation']['v8/audit.py'] = hashlib.sha256((root/'v8/audit.py').read_bytes()).hexdigest()
    pre_joint['implementation']['v8/audit.py'] = 'c8f536ba479076689032a36a3725fbd4e9befc059774443796e3ec7502439967'
    (tmp_path/'identity.json').write_text(json.dumps(pre_joint))
    assert refine_resume.verify(tmp_path, current)['schema'] == 3

    altered = dict(current, implementation=dict(current['implementation'], **{'v8/audit.py': 'changed'}))
    with pytest.raises(ValueError, match='Continuation identity mismatch: implementation'):
        refine_resume.verify(tmp_path, altered)

    for name in ('motion.json','motion.diagnostics.npz','quality.json'):
        (tmp_path/name).write_text('audit fixture')
    (tmp_path/'run.json').write_text(json.dumps(dict(files={name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
        for name in ('motion.json','motion.diagnostics.npz','quality.json')})))
    pre_joint['implementation']['v8/audit.py'] = 'a035b9ff99d654147a4aae09268bda195f619c2ad4579254683a652ec67f0156'
    (tmp_path/'identity.json').write_text(json.dumps(pre_joint))
    assert refine_resume.verify(tmp_path, current, recheck=True)['schema'] == 3
    with pytest.raises(ValueError, match='Recheck safety policy changed: v8/audit.py'):
        refine_resume.verify(tmp_path, altered, recheck=True)
    (base/'motion.pkl').write_bytes(b'other')
    with pytest.raises(ValueError, match='Fixed Frax reference hash changed'):
        refine_resume.verify(tmp_path, current)
