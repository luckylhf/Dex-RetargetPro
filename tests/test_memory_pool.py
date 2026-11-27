import hashlib
import argparse
import json
from pathlib import Path

import pytest

from dex_frax import refine_resume
from dex_frax.cli import add_repair_budget_option
from dex_frax.memory_pool import BASE, DANGER, GIB, MIB, Lease, Pool, requested_total
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


def test_block_request_uses_reported_workspace():
    event = dict(pss_bytes=BASE, phase='ik_block', required_peak_bytes=BASE+55*MIB)
    assert requested_total(event, 16000) == BASE+55*MIB


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

    altered = dict(current, implementation=dict(current['implementation'], **{'v8/audit.py': 'changed'}))
    with pytest.raises(ValueError, match='Continuation identity mismatch: implementation'):
        refine_resume.verify(tmp_path, altered)
    (base/'motion.pkl').write_bytes(b'other')
    with pytest.raises(ValueError, match='Fixed Frax reference hash changed'):
        refine_resume.verify(tmp_path, current)
