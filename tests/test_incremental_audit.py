from dex_frax.robot_profile import JOINT_COUNT
import copy,json,pickle
import os
from pathlib import Path
import numpy as np
import pytest
from dex_frax.audit_cache import fork


def export_neutral(tmp_path,n=20):
    from dex_frax.v8.robot import Robot
    from dex_frax.v8.anatomy import configure
    from dex_frax.v8 import convert
    r=Robot();cfg=json.loads(Path(convert.__file__).with_name('config.json').read_text());configure(r,cfg['anatomy'])
    x=np.r_[0.,0.,1.,np.zeros(JOINT_COUNT)];r.set(x,np.eye(3))
    repeat=lambda v:np.repeat(np.asarray(v)[None],n,axis=0)
    dg=dict(root_target=repeat(x[:3]),root_r=repeat(np.eye(3)),torso_target=repeat(r.rot(r.torso)),source_expected_contact=np.zeros(n,bool))
    for side,g in r.sides.items():
        v=dict(foot_target_p=r.point(g['foot']),phase=0,contact=0,
            hand_forward=-r.rot(g['tcp'])[:,2],hand_width=r.rot(g['tcp'])[:,0],forearm_width=r.rot(g['forearm'])[:,0],
            elbow_axis=-r.rot(g['shoulder'])[:,1],signed_elbow_bend=g['elbow_geometric_zero'])
        for a,b,k in [('shoulder','elbow','upper'),('elbow','wrist','lower'),('hip','knee','thigh'),('knee','foot','shin')]:
            delta=r.point(g[b])-r.point(g[a]);v[k]=delta/np.linalg.norm(delta)
        dg.update({side+'_'+k:repeat(val) for k,val in v.items()})
    data=dict(root_pos=repeat(x[:3]).astype(np.float32),root_rot=repeat([0.,0.,0.,1.]).astype(np.float32),dof_pos=repeat(x[3:]).astype(np.float32),joint_names=r.names,quat_order='xyzw',fps=50.)
    meta=dict(model=r.path,configuration=cfg,estimated_contact_segments={s:[] for s in r.sides},body_contact_models=[])
    p=tmp_path/'motion.pkl';p.write_bytes(pickle.dumps(data,protocol=4));p.with_suffix('.json').write_text(json.dumps(meta));np.savez_compressed(p.with_suffix('.diagnostics.npz'),**dg)
    return p,data,meta,dg


@pytest.mark.parametrize('bounds',[(0,2),(8,12),(18,20)])
def test_cached_audit_equals_fresh_including_boundary_intervals(tmp_path,bounds):
    from dex_frax.v8.audit import audit
    p,data,meta,dg=export_neutral(tmp_path);cache={};baseline=audit(p,cache=cache)
    before=copy.deepcopy(cache);a,b=bounds;data['dof_pos'][a:b,15]+=.005
    p.write_bytes(pickle.dumps(data,protocol=4));trial=fork(cache)
    actual=audit(p,cache=trial,changed=bounds)
    assert actual==audit(p)
    assert trial['work']==dict(native_frames_recomputed=b-a,cache_hit=True)
    assert np.array_equal(cache['data']['dof_pos'],before['data']['dof_pos'])
    assert cache['frames']['torso_orientation']==before['frames']['torso_orientation']
    # Dishonest/incorrect range cannot reuse measurements outside it.
    wrong=fork(cache);assert audit(p,cache=wrong,changed=(5,6))==actual
    assert not wrong['work']['cache_hit']
    dg['Left_upper'][:,0]+=.001;np.savez_compressed(p.with_suffix('.diagnostics.npz'),**dg)
    audit(p,cache=trial,changed=bounds);assert not trial['work']['cache_hit']


def _collision_worker_with_receipt(cpu, path, configuration, rows, rotations, models=None):
    from dex_frax.v8.audit import _collision_worker

    report = _collision_worker(cpu, path, configuration, rows, rotations, models)
    Path(configuration['test_receipt']).write_text(json.dumps(dict(pid=os.getpid(), cached=models is not None)))
    return report


def test_parallel_geometry_preserves_full_sweep_samples_and_incremental_cache(monkeypatch, tmp_path):
    from scipy.spatial.transform import Rotation
    from dex_frax import memory_pool, model_cache
    from dex_frax.paths import asset_fingerprints
    from dex_frax.robot_profile import PELVIS_BODY
    from dex_frax.v8 import audit
    from dex_frax.v8.robot import Robot

    robot = Robot()
    count = 98
    rows = np.zeros((count, JOINT_COUNT+3))
    rows[:, 2] = robot.neutral_pos[robot.bid(PELVIS_BODY)][2]
    rows[48:51, 2] -= .5
    rows[:, 3] = np.linspace(-.1, .1, count)
    root_angles = np.zeros((count, 3))
    root_angles[:, 2] = np.linspace(0, .1, count)
    rotations = Rotation.from_rotvec(root_angles).as_matrix()
    receipt = tmp_path/'helper.pid'
    meta = dict(profile='refine', configuration={'test_receipt':str(receipt)})
    serial_cache, parallel_cache = {}, {}
    serial = audit.geometry_checks(robot, meta, rows, rotations, serial_cache, None, False)
    models = model_cache.prepare_audit_models(asset_fingerprints(), tmp_path/'models')
    assert not any(info['reused'] for info in models.values())
    warm = model_cache.prepare_audit_models(asset_fingerprints(), tmp_path/'models')
    assert all(info['reused'] and info['sha256']==models[name]['sha256'] for name,info in warm.items())
    for name in ('_model','_xml','_audit_models'):
        monkeypatch.setattr(model_cache, name, getattr(model_cache,name))
    model_cache.enable_audit_models(warm)
    cached_robot = Robot()
    second_robot = Robot()
    assert cached_robot.m is second_robot.m and cached_robot.d is not second_robot.d
    cached = audit.geometry_checks(cached_robot, meta, rows, rotations, {}, None, False)
    assert cached[:2] == serial[:2] and np.array_equal(cached[2], serial[2])

    cpu = min(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 0
    calls = []

    def gate(*args):
        calls.append(args)
        return (cpu,) if args else ()

    monkeypatch.setattr(memory_pool, '_workspace_gate', gate)
    monkeypatch.setattr(audit, '_collision_worker', _collision_worker_with_receipt)
    timing = {}
    parallel = audit.geometry_checks(cached_robot, meta, rows, rotations, parallel_cache, None, False, timing)
    worker = json.loads(receipt.read_text())
    assert worker['pid'] != os.getpid() and worker['cached']
    assert timing['geometry_mode']=='parallel'
    binary_bytes = sum(Path(info['binary']).stat().st_size for info in warm.values())
    assert timing['workspace_bytes'] == 2*binary_bytes+256*memory_pool.MIB+count*64*1024
    assert all(timing[key]>=0 for key in ('helper_startup_s','input_receive_s','helper_robot_s','collision_model_s',
                                         'full_collision_scan_s','result_transfer_s','helper_shutdown_s','workspace_return_s'))
    assert parallel[:2] == serial[:2]
    assert np.array_equal(parallel[2], serial[2])
    assert (parallel[2] < -.001).any()
    for name in ('arm_collision', 'full_collision'):
        assert parallel_cache[name]['samples'] == serial_cache[name]['samples']
    assert len(calls) == 2 and calls[0][0] == 1 and calls[1] == ()

    rows[48:51, 2] += .03
    incremental = audit.geometry_checks(robot, meta, rows, rotations, parallel_cache, (48, 51), True)
    assert len(calls) == 2
    monkeypatch.setattr(memory_pool, '_workspace_gate', None)
    fresh = audit.geometry_checks(robot, meta, rows, rotations, {}, None, False)
    assert incremental[:2] == fresh[:2]
    assert np.array_equal(incremental[2], fresh[2])

    variant = Path(warm['collision']['binary'])
    with variant.open('ab') as stream:
        stream.write(b'invalid cache')
    with pytest.raises(ValueError, match='Cached MJB hash mismatch'):
        model_cache.prepare_audit_models(asset_fingerprints(), tmp_path/'models')


def test_relocated_audit_uses_verified_runtime_assets(tmp_path):
    from dex_frax.cli import sha
    from dex_frax.paths import asset_fingerprints
    from dex_frax.v8.audit import audit

    path, payload, meta, _ = export_neutral(tmp_path, n=5)
    original = audit(path, write=False)
    meta['model_sha256'] = sha(meta['model'])
    meta['model'] = str(tmp_path / 'previous-host/tiangong3_torq.xml')
    path.with_suffix('.json').write_text(json.dumps(meta))
    assets = asset_fingerprints()
    assets['.DS_Store'] = 'legacy-finder-metadata'
    identity = tmp_path / 'identity.json'
    identity.write_text(json.dumps(dict(assets=assets)))
    assert audit(path, write=False) == original
    assets['model.json'] = 'different-asset'
    identity.write_text(json.dumps(dict(assets=assets)))
    with pytest.raises(ValueError, match='asset hashes differ'):
        audit(path, write=False)


def test_asset_identity_excludes_metadata_but_tracks_model_changes(tmp_path):
    from dex_frax.paths import asset_fingerprints

    xml = tmp_path / 'dex.xml'
    xml.write_text('<mujoco/>')
    original = asset_fingerprints(tmp_path)
    for name in ('.DS_Store', '._dex.xml', 'README.md', 'LICENSE'):
        (tmp_path / name).write_text('metadata')
    assert asset_fingerprints(tmp_path) == original
    xml.write_text('<mujoco model="changed"/>')
    assert asset_fingerprints(tmp_path) != original


def test_final_gate_reads_support_jitter_full_collision_and_separates_history(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from dex_frax import acceptance
    p,data,meta,dg=export_neutral(tmp_path)
    np.savez_compressed(p.with_suffix('.pose_errors.npz'),**{s+'_'+k+'_deg':np.zeros(20) for s in ('Left','Right') for k in ('upper','lower','hand_forward','hand_width','elbow_axis')})
    check=dict(minimum_body_visual_cm=0.,interpolated_floor={'minimum_body_visual_cm':0.},sides={},
        body_support={'violating_frames':[5]},temporal_quality={'violating_frames':[8]},full_self_collision={'collision_samples':1})
    monkeypatch.setattr(acceptance,'audit',lambda _:copy.deepcopy(check))
    monkeypatch.setattr(acceptance,'quality',lambda *args:{'flags':[]})
    result=acceptance.evaluate(p,meta,SimpleNamespace(report={'qp_nonconverged_steps':1}),'frax')
    assert set(result['flags'])=={'trusted_body_support_mismatch','conversion_added_jitter','full_self_collision_unresolved','frax_qp_nonconvergence'}
    assert 'frax_qp_nonconvergence' not in result['final_residual_flags']
    assert result['initial_solver_history']['qp_nonconverged_steps']==1
