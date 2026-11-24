"""Run A/B/C on the prepared LaFAN quick-regression windows."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import json
import math
import os
import pickle
import shutil
import tempfile
import time
from types import SimpleNamespace

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')

import numpy as np

from dex_frax.airborne import protected_height
from dex_frax.cli import atomic_json, sha
from dex_frax.initial_reuse import target_digest
from dex_frax.refine_pipeline import repair_output
from dex_frax.v8.convert import configure
from dex_frax.v8.robot import Robot


EXAMPLES = Path(__file__).resolve().parents[1]/'examples'
PACK = EXAMPLES/'regression/quick/lafan50'


def read(path):
    return json.loads(path.read_text())


def verify(pack, cases):
    sources = {}
    for case in cases:
        bundle = pack/'fixtures'/case['name']
        for filename, checksum in case['files'].items():
            if sha(bundle/filename) != checksum:
                raise ValueError(f"{case['name']}: changed fixture {filename}")
        path = pack/case['source_file']
        if path not in sources:
            sources[path] = sha(path)
        if sources[path] != case['source_sha256']:
            raise ValueError(f'Changed source BVH: {path.name}')


def metrics(q):
    return dict(
        status=q['status'], jitter=len(q['temporal_quality']['violating_frames']),
        collision_samples=q['full_self_collision']['collision_samples'],
        floor_samples=q['interpolated_floor']['samples_below_minus_1mm'],
        floor_depth_mm=max(0.,-10.*q['interpolated_floor']['minimum_body_visual_cm']),
        body_support=len(q['body_support']['violating_frames']),
        arm_pose=q['adaptive_pose_gate']['violating_frames'],
        motor_overspeed=q['step_velocity']['violating_intervals'],
    )


def convert_window(pack, output, case, flow, budget):
    name = case['name']
    baseline = pack/'fixtures'/name
    folder = output/name/flow
    folder.mkdir(parents=True)
    for suffix in ('.pkl','.json','.npz','.diagnostics.npz'):
        shutil.copyfile(baseline/('motion'+suffix),folder/('motion'+suffix))
    path = folder/'motion.pkl'
    meta = read(path.with_suffix('.json'))
    payload = pickle.loads(path.read_bytes())
    targets = pickle.loads((baseline/'targets.pkl').read_bytes())
    if target_digest(targets) != meta['prepared_targets_sha256']:
        raise ValueError('Prepared targets changed')
    robot = Robot()
    if sha(Path(robot.path)) != meta['model_sha256']:
        raise ValueError('Robot model differs from the fixture')
    configure(robot,meta['configuration']['anatomy'])
    meta.update(input=str(pack/case['source_file']),model=robot.path)
    atomic_json(path.with_suffix('.json'),meta)
    result = repair_output(robot,targets,None,meta,payload,path,
                           SimpleNamespace(report=meta['frax']),repair_budget=budget,
                           segment_cascade=True,refine_flow=flow)
    audit = result['_unified_quality']
    report = read(folder/'repair.json')
    if (report['baseline_sha256'] != case['baseline_sha256'] or report['refine_flow'] != flow
            or report['budget_s'] != budget or not report['final_fresh_audit_equal']
            or report['final_sha256'] != sha(path)):
        raise ValueError('Audit, baseline, flow or budget differs from the test contract')
    fixed = protected_height(targets,len(payload['dof_pos']))
    if int(fixed.sum()) != case['protected_frames']:
        raise ValueError('Source protection mask changed')
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as saved:
        for key in ('source_time_s','source_frame_float','joint_names','fps'):
            np.testing.assert_array_equal(saved[key],payload[key])
        for key in ('dof_pos','root_pos','root_rot'):
            np.testing.assert_array_equal(saved[key][fixed],payload[key][fixed])
    atomic_json(folder/'quality.json',audit)
    return dict(quality=metrics(audit),initial=metrics(report['before']),attempts=len(report['attempts']),
                kept=sum(row['status']=='kept' for row in report['attempts']),
                budget_exhausted=report['shared_budget_exhausted'])


def run_one(pack, output, case, flow, budget):
    name = case['name']
    start = time.monotonic()
    with (output/'logs'/f'{name}.{flow}.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
        try:
            result = convert_window(pack,output,case,flow,budget)
            result['status'] = 'complete'
        except Exception as error:
            import traceback
            traceback.print_exc()
            result = dict(status='failed',error=f'{type(error).__name__}: {error}')
    result['elapsed_s'] = time.monotonic()-start
    return result


def write_summary(output, cases, state):
    lines = [
        '# LaFAN 专项短窗 A/B/C 对照',
        '',
        f"每窗共用 Frax 初解、人体目标和诊断。每条线路共享精修时限 {state['budget_s']:g} 秒。",
        '窗口帧号为从 0 开始的 50 Hz 全片帧号，表中区间包含首尾。抖动和身体支撑计帧，自碰撞和穿地计含帧间的采样。',
        '',
        f"完成 {sum(r['status']=='complete' for r in state['jobs'].values())}/{state['expected']} 个分支。"
        f"整批墙钟 {state['elapsed_s']:.1f} 秒。",
        '',
        '| 片段 | 原始窗口 | 流程 | 抖动帧 | 自碰撞采样 | 穿地采样 | 最深穿地 mm | 手臂超标帧 | 身体支撑异常帧 | 质量 | 候选调用 / 采纳 | 文件 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|',
    ]
    for case in cases:
        name = case['name']
        lo,hi = case['window']
        finished = []
        for flow in 'ABC':
            result = state['jobs'][f'{name}/{flow}']
            if result['status'] != 'complete':
                lines.append(f'| {name} | {lo}–{hi-1} | {flow} | — | — | — | — | — | — | 运行失败 | — | [日志](logs/{name}.{flow}.log) |')
                continue
            finished.append(result['initial'])
            q = result['quality']
            lines.append(f"| {name} | {lo}–{hi-1} | {flow} | {q['jitter']} | {q['collision_samples']} | "
                         f"{q['floor_samples']} | {q['floor_depth_mm']:.2f} | {q['arm_pose']} | {q['body_support']} | "
                         f"{q['status']} | {result['attempts']} / {result['kept']} | "
                         f"[NPZ]({name}/{flow}/motion.npz) · [候选记录]({name}/{flow}/repair.json) |")
        if any(item != finished[0] for item in finished[1:]):
            raise ValueError(f'{name}: the initial audits differed across flows')
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')


def run(pack, output, workers, budget, selected):
    start = time.monotonic()
    cases = read(pack/'selection.json')
    if selected:
        valid = {row[key] for row in cases for key in ('name','category','source_case')}
        if unknown := set(selected)-valid:
            raise ValueError(f'Unknown window, source or category: {sorted(unknown)}')
        cases = [c for c in cases if set(selected).intersection(c[k] for k in ('name','category','source_case'))]
    if not cases:
        raise ValueError('No test windows matched')
    verify(pack,cases)
    (output/'logs').mkdir()
    atomic_json(output/'selection.json',cases)
    state = dict(status='running', expected=len(cases)*3,budget_s=budget,workers=workers,jobs={})
    atomic_json(output/'progress.json',state)
    print(output,flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(run_one,pack,output,c,flow,budget):(c['name'],flow)
                 for c in cases for flow in 'ABC'}
        for future in as_completed(tasks):
            name,flow = tasks[future]
            result = future.result()
            state['jobs'][f'{name}/{flow}'] = result
            state['elapsed_s'] = time.monotonic()-start
            atomic_json(output/'progress.json',state)
            print(len(state['jobs']),state['expected'],name,flow,result['status'],
                  round(result['elapsed_s'],1),result.get('error',''),flush=True)
    state['status'] = 'complete' if all(v['status']=='complete' for v in state['jobs'].values()) else 'failed'
    write_summary(output,cases,state)
    atomic_json(output/'progress.json',state)
    return state['status']=='complete'


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack',type=Path,default=PACK)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--budget-s',type=float,default=20.)
    parser.add_argument('--select',action='append',default=[],help='Case, source or category, repeatable')
    args = parser.parse_args()
    if args.workers<1 or not math.isfinite(args.budget_s) or args.budget_s<=0:
        parser.error('Workers and budget must be positive')
    pack = args.pack.resolve()
    if not (pack/'selection.json').is_file():
        parser.error(f'LaFAN development fixtures are required. Set --pack to a directory containing selection.json: {pack}')
    if args.output is None:
        directory = EXAMPLES/'tmp'
        directory.mkdir(exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix='quick_abc_',dir=directory))
    else:
        destination = args.output.resolve()
        destination.mkdir(parents=True)
    raise SystemExit(not run(pack,destination,args.workers,args.budget_s,args.select))
