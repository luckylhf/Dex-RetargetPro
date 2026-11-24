"""Recover the stopped 1.0.3 capacity run without replacing its artifacts."""
import json
from pathlib import Path
import shutil

from . import segments
from .cli import sha

# The stopped release's exact implementation. No other code, inputs, assets,
# Frax implementation, runtime or conversion settings may change on this path.
_ORIGINAL = {
    "capacity_scheduler.py": "b05f70c8655e657cc70c75c66df2c71de44562e36849d3e77dad88e5cd1de8fe",
    "cli.py": "b248a4419a8d9da80972924aaf7531e71563a4f26760b113342b5fb72989f9ba",
    "refine_pipeline.py": "1561b632a857e99563fa6e2730786f3c1bba434f35657182d3201c5c72e8a604",
    "retarget_batch.py": "2579b0a4093b101f647e80fc531109c12a2caef2d7337ad5195af004509be684",
    "segments.py": "163f9a05e1e1221e1824c44191de1e61492b5df5260c9632ed48ba1a44409795",
    "v8/convert.py": "5de820710785b22e09c9dd9ed14d116eb6f128644a456f81e9b7fbff92c67353",
}


def read(path):
    return json.loads(path.read_text())


def check_identity(old, new):
    if {k:v for k,v in old.items() if k!='implementation'} != {k:v for k,v in new.items() if k!='implementation'}:
        raise ValueError('Recovery requires the original runtime, assets and Frax implementation')
    before,after = old['implementation'],new['implementation']
    changed = {key for key in before.keys()|after.keys() if before.get(key)!=after.get(key)}
    if changed != _ORIGINAL.keys()|{'capacity_recovery.py'} or any(before.get(key)!=digest for key,digest in _ORIGINAL.items()):
        raise ValueError('Recovery implementation differs from the expected 1.0.3 fix')


def inspect(previous, cases, conversion, rounds):
    saved = read(previous/'capacity_config.json')
    if read(previous/'batch.json')['status']!='interrupted':
        raise ValueError('Recovery requires an interrupted source batch')
    if cases!=saved['cases'] or conversion!=saved['conversion'] or rounds!=saved['repair_rounds']:
        raise ValueError('Recovery requires the exact original inputs, conversion settings and round count')
    snapshots={}
    for motion,case in cases.items():
        directory=previous/'motions'/motion
        initial=directory/'frax_initial'
        if not (initial/'initial_ready.json').exists():
            continue
        identity=read(initial/'identity.json')
        if any(identity[k]!=value for k,value in saved['identity'].items()) or identity['input_sha256']!=case['input_sha256']:
            raise ValueError(f'Initial identity mismatch: {motion}')
        segments.verify(initial,identity)
        frames=read(initial/'run.json')['frames']
        result=dict(initial=str(initial),source=str(initial),kind='initial',frames=frames)
        diagnosis=directory/'diagnose_01'/'run.json'
        if diagnosis.exists():
            report=read(diagnosis)
            if report['status']=='repair_planned' and report['frames']==frames:
                result['workload']=report['workload']
        repair=directory/'repair_01'
        if (repair/'refine_state.json').is_file():
            if read(repair/'identity.json')!=identity:
                raise ValueError(f'Repair identity mismatch: {motion}')
            state=read(repair/'refine_state.json')
            required=segments._artifacts(repair,True)
            if (all((repair/name).is_file() for name in required)
                    and state.get('status')=='running'
                    and state['output_sha256']==sha(repair/'motion.pkl')
                    and state['baseline_sha256']==sha(repair/'frax_baseline/motion.pkl')
                    and state['avoidance_reference_sha256']==sha(repair/'avoidance_reference.npy')):
                result.update(source=str(repair),kind='repair',elapsed_s=state['elapsed_s'],
                              active_phase=state['active_phase'])
            else:
                result['fallback']='uncommitted repair state; verified Frax initial retained'
        snapshots[motion]=result
    return saved['identity'],snapshots


def stage_checkpoints(output,snapshots):
    sources={}
    for motion,record in snapshots.items():
        origin=Path(record['source'])
        if record['kind']=='initial':
            sources[motion]=str(origin)
            continue
        staged=output/'recovered_checkpoints'/motion
        staged.mkdir(parents=True)
        for name in sorted(segments._artifacts(origin,True)):
            dest=staged/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(origin/name,dest)
        identity=read(staged/'identity.json')
        segments.seal(staged,identity,continuation=True)
        segments.verify(staged,identity)
        sources[motion]=str(staged)
    return sources
