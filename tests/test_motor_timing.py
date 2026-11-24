import numpy as np

from dex_frax.retiming import final_candidate, fit_inserted_frames, resample_targets, time_map
from dex_frax.v8 import convert as converter


def test_inserted_frame_uses_ik_and_preserves_original_poses(dex_synthetic_clip):
    # Perturb one shoulder motor to require retiming at 50 Hz.
    cfg, targets, baseline = dex_synthetic_clip
    robot = converter.Robot()
    converter.configure(robot, cfg['anatomy'])
    rows = baseline.copy()
    col = robot.names.index('shoulder_roll_l_joint') + 3
    rows[70:73,col] += np.cumsum([.10,.17,.10])
    rows[73:83,col] += np.linspace(.34,0.,10)
    index, report = time_map(targets['root_r'],rows[:,:3],50.,None,None,
                             joint_positions=rows[:,3:],joint_limits=.999*robot.speed_limits,
                             keep_original_frames=True)
    assert report['changed']
    fitted = fit_inserted_frames(robot,resample_targets(targets,robot,index),cfg['solver'],rows,index)
    original = index == np.rint(index)
    np.testing.assert_array_equal(fitted[original],rows)
    assert np.all(abs(np.diff(fitted[:,3:],axis=0))*50 <= robot.speed_limits+1e-4)
    assert np.all(fitted[:,3:] <= robot.upper[3:]+1e-6)
    assert np.all(fitted[:,3:] >= robot.lower[3:]-1e-6)
    linear = np.column_stack([np.interp(index,np.arange(len(rows)),c) for c in rows.T])
    np.testing.assert_array_equal(fitted[:, :2], linear[:, :2])
    assert not np.allclose(fitted[~original],linear[~original])


def test_final_retiming_defers_without_touching_trajectory_when_budget_is_exhausted(tmp_path):
    from dex_frax.quality_v8 import SPEED_FLAG

    motion = tmp_path/'motion.pkl'
    motion.write_bytes(b'trajectory must not be opened or edited')
    quality = {'flags':[SPEED_FLAG]}
    meta, checked = final_candidate(motion,{},quality,None,'frax',context=(None,None,None),deadline=0.)
    assert meta['retiming']['final_adaptation']['decision']=='pending_budget'
    assert checked is quality
    assert motion.read_bytes()==b'trajectory must not be opened or edited'


def test_limited_acceptance_still_reports_unfinished_work():
    from dex_frax.pipeline_explain import _render_repair_progress, _repair_progress

    work = _repair_progress(dict(status='pending_budget',shared_budget_exhausted=True,budget_s=90.))
    lines = _render_repair_progress(dict(status='accepted_with_notes',profile='refine',
        repair_progress=work,retiming={'pre_final_checkpoint':'pre_final_retime'}))
    assert any('仍有任务待处理' in line for line in lines)
    assert any('`pre_final_retime`' in line for line in lines)
