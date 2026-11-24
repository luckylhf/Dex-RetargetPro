"""Build and run short regressions from verified Robot-Retarget archives."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.spatial.transform import Rotation

from dex_frax.adaptive_refine import window_targets
from dex_frax.airborne import protected_height
from dex_frax.contact_routed_initial import contact_routes, expanded_sections
from dex_frax.initial_reuse import target_digest
from dex_frax.refine_pipeline import propose
from dex_frax.segment_refine import project_joints
from dex_frax.source_integrity import prepare_smplx_channels
from dex_frax.temporal_quality import assess
from dex_frax.torso_branch import floor_samples
from dex_frax.v8 import convert
from dex_frax.v8.self_collision import CollisionModel, scan
from dex_frax.v8.solver import WindowAnchors, solve


REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO/'examples'
REGRESSION = EXAMPLES/'regression'
PACK = REGRESSION/'quick'

# Half-open frame ranges. Source checks use native 30 Hz, the rest use output 50 Hz.
# For historical failure controls, the expected failure must be detected.
CASES = [
    ('boy-arm-join', 'arm', 'dance_boy', 182, 240, []),
    ('boy-legs-in-flight', 'legs', 'dance_boy', 192, 231, []),
    ('boy-v8-anchors', 'v8', 'dance_boy', 206, 240, []),
    ('boy-jitter-340', 'temporal', 'dance_boy', 320, 360, [340, 341]),
    ('boy-jitter-598', 'temporal', 'dance_boy', 575, 619, [598, 599]),
    ('girl-jitter-742', 'temporal', 'dance_girl', 725, 760, [742, 743]),
    ('girl-jitter-6793', 'temporal', 'dance_girl', 6780, 6810, [6793, 6794]),
    ('guangyuan-jitter-281', 'temporal', 'dance_guangyuan', 269, 299, [281, 282]),
    ('gbtc-jitter-29', 'temporal', 'gbtc', 17, 43, [29, 30]),
    ('gbtc-jitter-217', 'temporal', 'gbtc', 203, 233, [217, 218]),
    ('gbtc-jitter-411', 'temporal', 'gbtc', 397, 427, [411, 412]),
    ('nobody-clean-control', 'clean', 'nobody', 291, 322, []),
    ('girl-ankle-collision', 'collision', 'dance_girl', 10007, 10035, [10021]),
    ('guangyuan-ankle-collision', 'collision', 'dance_guangyuan', 8437, 8466, [8451]),
    ('ground-hip-ankle-collision', 'collision', 'ground1_subject5', 5457, 5486, [5471]),
    ('ground-enter-body-support', 'routing', 'ground1_subject5', 358, 445, [382, 420]),
    ('ground-exit-body-support', 'routing', 'ground1_subject5', 845, 930, [869, 889]),
    ('girl-airborne-contact-first', 'flight', 'dance_girl', 206, 245, [221, 222]),
    ('guangyuan-collar-reversal', 'source', 'dance_guangyuan', 2086, 2109, [2098]),
    ('guangyuan-ankle-fast-motion', 'source', 'dance_guangyuan', 4217, 4240, []),
]


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def setup(metadata=None):
    config = json.loads((convert.ROOT/'config.json').read_text())
    robot = convert.Robot()
    anatomy = metadata['configuration']['anatomy'] if metadata else config['anatomy']
    convert.configure(robot, anatomy if metadata else convert.robot_profile.wrist_profile(anatomy))
    return robot


def capture_boy(pack):
    reference = pack/'source/reference_boy'
    metadata = json.loads((reference/'motion.json').read_text())
    motion = npz(reference/'motion.npz')
    with (reference/'boy-verified-short-targets.pkl').open('rb') as stream:
        captured = pickle.load(stream)
    targets = captured['targets']
    assert target_digest(targets) == metadata['prepared_targets_sha256']
    targets['root_r'] = Rotation.from_quat(motion['root_rot']).as_matrix()
    return dict(captured, motion=motion, meta=metadata)


def archive_motion(archive, case, version):
    folder = archive/version/case/'frax'
    receipt = json.loads((folder/'run.json').read_text())
    assert receipt['status'] == 'complete'
    for name in ('motion.npz', 'motion.diagnostics.npz'):
        assert sha(folder/name) == receipt['files'][name], name
    return npz(folder/'motion.npz'), npz(folder/'motion.diagnostics.npz')


def save_clip(path, motion, selection):
    per_frame = ('root_pos','root_rot','dof_pos','source_time_s','source_frame_float')
    fields = {key:motion[key][selection] for key in per_frame if key in motion}
    fields.update({key:motion[key] for key in ('fps','source_fps','joint_names','quat_order','robot')
                   if key in motion})
    np.savez_compressed(path,**fields)


def build(pack, history_root):
    manifest = pack/'windows.json'
    if manifest.exists():
        raise FileExistsError(f'Use a new directory for rebuilding {manifest}')
    archive = history_root/'compare_084_103_20261123.UHCgRF'
    targets = history_root/'window_recheck_104_20261123/1.0.3-origin'
    if not archive.is_dir() or not targets.is_dir():
        raise FileNotFoundError('Restore the historical payload and pass its location with --history-root')
    boy = capture_boy(pack)
    native = npz(pack/'source/guangyuan.npz')
    windows = pack/'windows'
    windows.mkdir(parents=True, exist_ok=True)
    clips = pack/'clips'
    clips.mkdir(parents=True, exist_ok=True)
    entries = []
    archive_cache = {}
    target_cache = {}
    for name, group, case, lo, hi, expected in CASES:
        version = '0.8.4' if name=='girl-jitter-6793' else '1.0.3'
        sel = slice(lo,hi)
        clip = clips/(name+'.npz')
        info = dict(name=name, group=group, motion=case, frames=[lo,hi],
                    fps=30 if group=='source' else 50, expected_frames=expected,
                    clip=str(clip.relative_to(pack)))
        if group in ('arm','legs','v8'):
            t = boy['targets']
            m = boy['motion']
            data = dict(targets=window_targets(t,lo,hi,len(t['root_p'])),
                        rows=np.c_[m['root_pos'][sel],m['dof_pos'][sel]].astype(float),
                        weights=boy['weights'], meta=boy['meta'])
            save_clip(clip,m,sel)
            info['origin'] = boy['meta']['prepared_targets_sha256']
        elif group in ('routing','flight'):
            if case not in target_cache:
                source = targets/case/'archived_targets.pkl'
                with source.open('rb') as stream:
                    target_cache[case] = pickle.load(stream)
            t = target_cache[case]
            data = dict(targets=window_targets(t,lo,hi,len(t['root_p'])))
            if (case,version) not in archive_cache:
                archive_cache[case,version] = archive_motion(archive,case,version)
            save_clip(clip,archive_cache[case,version][0],sel)
            info['origin'] = target_digest(t)
        elif group=='source':
            data = dict(fields={key:native[key][sel] for key in ('transl','body_pose','global_orient')},
                        names=native['names'].tolist())
            np.savez_compressed(clip,**data['fields'],fps=native['fps'],
                                original_frame=np.arange(lo,hi),joint_names=native['names'])
            info['origin'] = str(native['source_sha256'])
        else:
            if (case,version) not in archive_cache:
                archive_cache[case,version] = archive_motion(archive,case,version)
            m,d = archive_cache[case,version]
            save_clip(clip,m,sel)
            data = dict(motion={key:m[key][sel] for key in ('root_pos','root_rot','dof_pos','source_time_s')},
                        names=m['joint_names'].tolist(),
                        diagnostics={key:value[sel] for key,value in d.items()
                                     if value.ndim and len(value)==len(m['dof_pos'])})
            info['origin'] = version
            if case=='gbtc':
                if (case,'0.8.4') not in archive_cache:
                    archive_cache[case,'0.8.4'] = archive_motion(archive,case,'0.8.4')
                clean,diag = archive_cache[case,'0.8.4']
                np.testing.assert_allclose(m['source_time_s'][sel],clean['source_time_s'][sel],atol=1e-9)
                data['clean_q'] = clean['dof_pos'][sel]
                data['clean_diag'] = {k:v[sel] for k,v in diag.items()
                                      if v.ndim and len(v)==len(clean['dof_pos'])}
        path = windows/(name+'.pkl')
        path.write_bytes(pickle.dumps(data,protocol=4))
        entries.append(dict(info,sha256=sha(path),clip_sha256=sha(clip)))
    manifest.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')


def solve_window(group, data):
    robot = setup(data['meta'])
    t,q,w = (data[key] for key in ('targets','rows','weights'))
    protected = protected_height(t,len(q))
    assert protected.any() and (~protected).any()
    model = CollisionModel(robot,dict(data['meta']['configuration']['self_collision'],coverage='full'))
    if group=='v8':
        fixed = protected.copy()
        fixed[[0,-1]] = True
        anchors = WindowAnchors(robot,q,fixed,t['fps'])
        config = dict(w,_repair_deadline=time.monotonic()+12.)
        fit,_ = solve(robot,t,config,anchors=anchors)
        support,_ = convert.support_solver(config)(robot,t,fit,config,anchors=anchors)
        projected = support.copy()
        project_joints(projected,anchors)
        for stage in (fit,support,projected):
            np.testing.assert_array_equal(stage[fixed],q[fixed])
            assert np.all(abs(np.diff(stage[:,3:],axis=0))<=anchors.step+1e-8)
        proposals = [projected]
    else:
        sides = ['Right'] if group=='arm' else ['Left','Right']
        proposals = []
        for side in sides:
            job = dict(start=0,end=len(q),side=side,module='arms' if group=='arm' else 'leg_pose')
            fit,detail = propose(robot,t,q,q,w,model,job,'fit',time.monotonic()+10.)
            if group=='arm':
                assert detail['cost'] < detail['initial_cost'] * .5
            proposals.append(fit)
    for candidate in proposals:
        np.testing.assert_array_equal(candidate[protected],q[protected])
        assert np.any(candidate[~protected]!=q[~protected])
        if group!='v8':
            assert np.all(abs(np.diff(candidate[:,3:],axis=0))*t['fps']<=robot.speed_limits+1e-4)
    return dict(protected_frames=int(protected.sum()))


def verify(entry, data):
    group = entry['group']
    lo,hi = entry['frames']
    expected = set(entry['expected_frames'])
    fps = entry['fps']
    if group in ('arm','legs','v8'):
        return solve_window(group,data)
    if group=='source':
        before = data['fields']
        after,audit = prepare_smplx_channels(before,fps,hi-lo,data['names'])
        channel = 'left_collar' if expected else 'left_ankle'
        index = data['names'].index(channel)-1
        changed = set((lo+np.flatnonzero(np.any(after['body_pose'].reshape(-1,21,3)[:,index]!=
                                               before['body_pose'].reshape(-1,21,3)[:,index],axis=1))).tolist())
        assert changed == expected, (channel,changed,expected)
        assert not any(e.get('channel')==channel+'.rotation' for e in audit['blocking_events'])
        return dict(changed_native_frames=sorted(changed))
    if group in ('routing','flight'):
        t = data['targets']
        active,evidence = contact_routes(setup(),t)
        flight = protected_height(dict(source_airborne=t['source_airborne']),hi-lo)
        sections = [(int(a)+lo,int(b)+lo) for a,b in expanded_sections(active,round(.25*fps),flight)]
        if group=='flight':
            assert not active.any() and not sections
            assert expected.issubset({lo+i for i in evidence['airborne_frames']})
        else:
            assert not flight.any() and active.any() and not active.all()
            assert all(any(a<=f<b for a,b in sections) for f in expected), sections
            assert not (sections[0][0]==lo and sections[-1][1]==hi)
        return dict(sections=sections, airborne_frames=len(evidence['airborne_frames']))
    m = data['motion']
    if group in ('temporal','clean'):
        detected = set(lo+i for i in assess(m['dof_pos'],fps,data['names'],data['diagnostics'])['violating_frames'])
        if group=='temporal':
            assert expected.issubset(detected), (expected,detected)
            if 'clean_q' in data:
                assert not assess(data['clean_q'],fps,data['names'],data['clean_diag'])['violating_frames']
            return dict(detected_frames=sorted(detected))
        assert not detected
    robot = setup()
    model = CollisionModel(robot,{'coverage':'full'})
    rows = np.c_[m['root_pos'],m['dof_pos']]
    rotations = Rotation.from_quat(m['root_rot']).as_matrix()
    report,_,_ = scan(model,rows,rotations,8)
    penetrations = int((floor_samples(robot,{'root_r':rotations},rows).min(axis=1)<-.001).sum())
    if group=='clean':
        assert not report['collision_samples'] and not penetrations
        assert np.all(abs(np.diff(m['dof_pos'].astype(float),axis=0))*fps<=robot.speed_limits+1e-4)
    else:
        assert report['collision_samples']>0 and penetrations>0
        assert lo+report['worst']['frame'] in expected, report['worst']
    return dict(collision_samples=report['collision_samples'],floor_samples_over_1mm=penetrations)


def run(pack, selecting):
    entries = json.loads((pack/'windows.json').read_text())
    known = {e['name'] for e in entries} | {e['group'] for e in entries}
    if unknown := set(selecting)-known:
        raise ValueError(f'Unknown case or group: {sorted(unknown)}')
    results = []
    start = time.monotonic()
    for entry in entries:
        name = entry['name']
        if selecting and name not in selecting and entry['group'] not in selecting:
            continue
        path = pack/'windows'/(name+'.pkl')
        began = time.monotonic()
        result = dict(case=name,group=entry['group'],frames=entry['frames'])
        try:
            assert sha(path)==entry['sha256'], 'Fixture checksum changed'
            assert sha(pack/entry['clip'])==entry['clip_sha256'], 'Motion clip checksum changed'
            # Fixtures are created locally by build from the trusted project archives.
            with path.open('rb') as stream:
                detail = verify(entry,pickle.load(stream))
            result.update(status='pass',detail=detail)
        except Exception as error:
            result.update(status='fail',error=f'{type(error).__name__}: {error}')
        result['seconds'] = round(time.monotonic()-began,3)
        results.append(result)
        print(f"{result['status']:4} {name} {result['seconds']:.3f}s {result.get('error','')}",flush=True)
    passed = sum(r['status']=='pass' for r in results)
    report = dict(passed=passed,total=len(results),seconds=round(time.monotonic()-start,3),cases=results)
    filename = 'latest-selection.json' if selecting else 'latest.json'
    (pack/filename).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"{passed}/{len(results)} passed in {report['seconds']:.3f}s")
    return passed==len(results)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('build','run'))
    parser.add_argument('--pack',type=Path,default=PACK)
    parser.add_argument('--history-root',type=Path,default=REGRESSION/'slow',
                        help='Restored historical payload for build')
    parser.add_argument('--select',action='append',default=[],help='Case name or group, repeatable')
    args = parser.parse_args()
    if args.command=='build':
        build(args.pack,args.history_root)
    else:
        raise SystemExit(0 if run(args.pack,args.select) else 1)
