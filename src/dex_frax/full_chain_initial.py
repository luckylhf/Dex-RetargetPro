"""Full-clip source imitation, support landing and conditional collision repair."""
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .cli import atomic_json, sha
from .ik import FraxIK
from .identity import FULL_CLIP_INITIAL_POLICY
from .initial_landing import solve_landing
from .pelvis_priority import solve_protected
from .quality_v8 import adjacent_speed_report
from .temporal_constraints import diagnostics
from .temporal_quality import assess
from .v8.self_collision import CollisionModel, scan


def collision_trigger(impacted, length, fps):
    mask = np.zeros(length, bool)
    mask[list(impacted)] = True
    bounds = np.flatnonzero(np.diff(np.r_[False,mask,False])).reshape(-1,2)
    count = int(mask.sum())
    longest = int((bounds[:,1]-bounds[:,0]).max(initial=0))
    floor = max(math.ceil(.8*fps),math.ceil(.05*length))
    minimum_run = math.ceil(.4*fps)
    return dict(triggered=count>floor and (longest>=minimum_run or len(bounds)>=5),
                total_frames=length, native_frames=count, minimum_frames_exclusive=floor,
                minimum_contiguous_frames=minimum_run, alternative_minimum_episodes=5,
                episodes=len(bounds), longest_episode_frames=longest)


class FullClipInitial:
    improved = False
    defer_joint_retiming = True
    policy = FULL_CLIP_INITIAL_POLICY

    def __init__(self, output, raw_solver, iterations, initial_iterations, assets, *, smooth_lifts=False,
                 artifact_prefix='motion'):
        self.output = Path(output)
        self.artifact_prefix = artifact_prefix
        self.raw_solver = raw_solver
        self.settings = FraxIK(iterations, initial_iterations, improved=False, assets=assets)
        self.initial_collision_cfg = None
        self.source_time_s = None
        self.smooth_lifts = smooth_lifts
        self.report = {}
        self.ledger = dict(policy=self.policy, status='running', stages=[])

    def artifact(self, name):
        return self.output / f'{self.artifact_prefix}.{name}'

    def save_stage(self, name, rows, robot, targets, elapsed, diagnostics):
        dest = self.artifact(f'stage_{name}.npz')
        np.savez_compressed(dest, x=rows,root_r=targets['root_r'],joint_names=robot.names,
                            fps=targets['fps'])
        self.ledger['stages'].append(dict(stage=name, file=dest.name, sha256=sha(dest),
            frames=len(rows), wall_s=elapsed, solver=diagnostics))
        atomic_json(self.artifact('full_chain.json'),self.ledger)

    def __call__(self, robot, targets, cfg, progress=None):
        start = time.monotonic()
        print('initial: unrestricted source pose',flush=True)
        pose_targets = dict(targets,pose_only=True,sides={
            side:dict(data,foot_target_r=data['raw_foot_r']) for side,data in targets['sides'].items()})
        raw, records = self.raw_solver(robot,pose_targets,cfg,progress)
        self.baseline = raw.copy()
        self.save_stage('raw',raw,robot,targets,time.monotonic()-start,self.raw_solver.report)
        from .discontinuities import root_export_impulses
        temporal = assess(raw[:,3:],targets['fps'],robot.names,diagnostics(pose_targets))
        self.ledger['raw_temporal_audit'] = dict(joints=temporal, stage_snapshot=self.artifact('stage_raw.npz').name,
            motor=adjacent_speed_report(raw[:,3:],targets['fps'],robot.names,robot.speed_limits),
            root_impulses=root_export_impulses(raw[:,:3],targets['root_r'],targets['root_p'],targets['root_r'],targets['fps']))
        source_time = self.source_time_s
        if source_time is None:
            source_time = np.arange(len(raw))/targets['fps']
        common = dict(source_time_s=source_time,root_p=targets['root_p'],torso_r=targets['torso_r'])
        for side, foot in targets['sides'].items():
            for key in ('contact','phase','pivots','anchors'):
                common[side+'_'+key] = foot[key]
        path = self.artifact('stage_targets.npz')
        np.savez_compressed(path,**common)
        self.ledger.update(stage_targets=path.name,stage_targets_sha256=sha(path))

        landing_start = time.monotonic()
        print('initial: episode lift and anchor landing',flush=True)
        landing_solver = FraxIK(self.settings.iterations,self.settings.initial_iterations,
                               improved=False,assets=self.settings.assets)
        lifted,landed,records,landing = solve_landing(robot,targets,cfg,raw,landing_solver,progress)
        writable = lifted[:, 2] != raw[:, 2]
        if self.smooth_lifts and writable.any():
            from .lift_continuity import smooth_landing
            landed, landing['continuity'] = smooth_landing(robot,targets,raw,landed,writable)
        self.save_stage('lifted',lifted,robot,targets,0.,{'vertical_translation_only':True})
        self.save_stage('landed',landed,robot,targets,time.monotonic()-landing_start,landing)
        model = CollisionModel(robot,dict(self.initial_collision_cfg or {},coverage='full'))
        # Count the same native geometry as the exported trajectory. Inter-frame
        # samples are a separate metric and must not inflate the trigger.
        rotations = Rotation.from_quat(Rotation.from_matrix(targets['root_r']).as_quat().astype(np.float32)).as_matrix()
        checked = time.monotonic()
        audit, _, impacted = scan(model,landed.astype(np.float32).astype(float),rotations,1)
        trigger = collision_trigger(impacted,len(raw),targets['fps'])
        self.ledger.update(trigger=trigger, trigger_stage='landed', native_collision=audit, trigger_wall_s=time.monotonic()-checked)
        atomic_json(self.artifact('full_chain.json'),self.ledger)
        print('initial: full_clip_frax',trigger, flush=True)
        self.report = dict(landing['solver'],landing=landing['policy'],raw_frax_report=self.raw_solver.report)
        if trigger['triggered']:
            rows, records, detail = solve_protected(robot,targets,cfg,landed,self.settings,'feasible',
                progress,auto_lower=False,collision_cfg=self.initial_collision_cfg,full_chain=self)
            detail['lower_routing'] = dict(selected='full_feet',reason='widespread_whole_body_collision',
                                           trigger=trigger)
            detail['contact_windows'] = detail['ordered_limb_solvers']['feet']['post_leg_contact']
            self.report = dict(detail['limb_solver'], body_priority=detail, landing=landing['policy'],
                               landing_report=landing['solver'],raw_frax_report=self.raw_solver.report)
            self.report['qp_nonconverged_steps'] = (int(self.raw_solver.report['qp_nonconverged_steps'])+
                int(landing['solver']['qp_nonconverged_steps'])+
                sum(int(v['qp_nonconverged_steps']) for v in detail['ordered_limb_solvers'].values()))
        else:
            rows = landed
            self.report['qp_nonconverged_steps'] += int(self.raw_solver.report['qp_nonconverged_steps'])
        self.initial = rows.copy()
        self.ledger.update(status='completed',initial_wall_s=time.monotonic()-start,
             final_stage='arms' if trigger['triggered'] else 'landed')
        self.report['full_chain'] = dict(report_file=self.artifact('full_chain.json').name,trigger=trigger,
                                         selected=self.ledger['final_stage'])
        atomic_json(self.artifact('full_chain.json'),self.ledger)
        return rows, records
