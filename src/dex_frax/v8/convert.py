#!/usr/bin/env python3
"""Convert SOMA/LAFAN/mocap BVH to a named-joint Dex PKL, entirely on CPU."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import argparse,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..contract import DEFAULT_FOOT_CONTACT, DEFAULT_REFINE_FLOW
from .. import robot_profile
from .bvh import load
from .source import adapt, head_relative_target
from .robot import Robot,DEFAULT_MODEL,DEFAULT_URDF
from .contacts import prepare
from .solver import solve,constrain_support as legacy_support
from .support_fast import constrain_support as hybrid_support
from .arms import plan_and_refine
from .contact_solver import refine as refine_body_contacts
from .anatomy import configure
from .self_collision import refine as refine_self_collision, audit_export as audit_self_collision

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def support_solver(cfg):
    mode=cfg.get('support_refinement','hybrid_cpu_v1')
    if mode=='hybrid_cpu_v1':return hybrid_support
    if mode=='legacy_slsqp':return legacy_support
    raise ValueError('Unknown support_refinement: '+str(mode))

def convert(bvh,output,source='auto',model=DEFAULT_MODEL,fps=None,config=None,max_frames=None,no_contacts=False,initial_solver=None,refine=True,prepared_input=None,unified_repair=False,repair_similarity='soft',profile=None,repair_budget=None,continue_from=None,resume_recheck=False,allow_local_slowdown=False,bvh_units='cm',bvh_translations='absolute',foot_contact=DEFAULT_FOOT_CONTACT,initial_only=False,initial_from=None,initial_marker=None,scene=None,scene_hash=None,repair_modules=None,repair_plan_only=False, ground_running=False, *, refine_flow=DEFAULT_REFINE_FLOW):
    from ..resource_events import phase
    phase('preprocessing',compile_possible=initial_solver is not None and not getattr(initial_solver,'strict',False))
    start=time.monotonic();output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError('Refusing to overwrite '+str(output))
    cfg=json.loads((ROOT/'config.json').read_text()) if config is None else json.loads(Path(config).read_text())
    cfg['anatomy']=robot_profile.wrist_profile(cfg['anatomy'])
    if foot_contact == 'v2':
        from .foot_contact_v2 import DEFAULTS
        cfg['contact'].update(policy='v2', v2=dict(DEFAULTS))
    elif foot_contact != 'v1':
        raise ValueError(f'Unknown foot contact policy: {foot_contact}')
    from ..arm_policy import POLICY as ARM_POLICY
    if unified_repair and repair_similarity=='soft':cfg['solver']['arm_repair_policy']=ARM_POLICY
    constrain_support=support_solver(cfg['solver'])
    from ..input_errors import MotionInputError
    source_review=None
    try:
        if source == 'smplx':
            from ..smplx_source import load as load_smplx, adapt as adapt_smplx
            if prepared_input is None: raise ValueError('SMPL-X requires the dex-frax convert preparation entry')
            motion=load_smplx(prepared_input,fps=fps)
            source_targets=adapt_smplx(motion)
            if 'source_review_json' in motion.data:
                source_review=json.loads(str(motion.data['source_review_json']))
            from ..source_integrity import POLICY as SOURCE_POLICY
            if source_review is None or source_review.get('policy') != SOURCE_POLICY:
                raise ValueError('SMPL-X prepared input lacks the current native-frame audit. Regenerate from source parameters.')
            if source_review.get('ground_running_declared', False) != ground_running:
                raise ValueError('SMPL-X prepared input uses a different declared human locomotion scope')
            source_origin = float(motion.data['times'][0])
        else:
            from ..source_integrity import inspect, recheck_bvh, record_resampled
            from dataclasses import replace
            from .bvh import resample
            motion=load(bvh,units=bvh_units,translations=bvh_translations)
            motion,source_review=inspect(motion,source,ground_running=ground_running)
            checked_native=motion
            source_origin = source_review['kept_native_interval'][0] / motion.input_fps
            from ..cli import atomic_json
            source_review['source_sha256']=sha(bvh)
            atomic_json(output.with_suffix('.source_review.json'),source_review)
            if fps is not None:
                if max_frames is not None:
                    needed = int(np.ceil((max_frames-1)*motion.fps/fps))+1
                    motion=replace(motion,local_p=motion.local_p[:needed],local_r=motion.local_r[:needed])
                motion=resample(motion,fps)
            if max_frames:motion=replace(motion,local_p=motion.local_p[:max_frames],local_r=motion.local_r[:max_frames])
            if len(motion.local_p)<3:raise ValueError('Conversion needs at least 3 frames; one-frame poses are calibration-only')
            record_resampled(source_review,recheck_bvh(motion,source_review['source_type'],source_review,
                                                       ground_running=ground_running,native=checked_native))
            source_targets=adapt(motion,source)
    except (ValueError,KeyError,IndexError,StopIteration) as error:
        raise MotionInputError(str(error)) from error
    robot=Robot(model);configure(robot,cfg['anatomy']);t=prepare(source_targets,robot,cfg['contact'],enabled=not no_contacts)
    boxes = []
    scene_digest = None
    if scene is not None:
        from ..scene_support import scene_boxes
        try:
            scene_digest = sha(scene)
            boxes = scene_boxes(scene)
            if sha(scene)!=scene_digest or (scene_hash is not None and scene_digest!=scene_hash):
                raise ValueError('Scene file changed during conversion setup')
        except (ValueError,OSError,KeyError,TypeError) as error:
            raise MotionInputError(str(error)) from error
    if source_review is not None or boxes:
        from ..source_support import apply as assign_support
        support_review = assign_support(t,robot,boxes)
    else:
        support_review = None
    if source_review is not None:
        source_review['source_sha256']=sha(bvh)
        source_review['support']=support_review
        from ..cli import atomic_json
        atomic_json(output.with_suffix('.source_review.json'),source_review)
    timing=None;source_time=None;joint_indices=None
    timing_archive=None
    if allow_local_slowdown:
        if profile!='refine':raise ValueError('Local slowdown requires explicit refine mode')
        from ..retiming import apply as retime, resample_targets, time_map, motor_causes
        from ..source_integrity import eligible_timing_intervals
        angular_limit=robot_profile.ROOT_ANGULAR_LIMIT_RAD_S
        linear_limit=robot_profile.ROOT_LINEAR_LIMIT_M_S
        if angular_limit is not None:angular_limit*=.975
        if linear_limit is not None:linear_limit*=.975
        eligible=eligible_timing_intervals(source_review,source_origin+np.arange(len(t['root_p']))/t['fps'],root_only=True)
        t,source_time,timing=retime(t,robot,angular_limit,linear_limit,eligible=eligible)
        source_time += source_origin
        timing_archive=Path(initial_from or continue_from) if (initial_from or continue_from) else None
        if timing_archive is not None:
            if (timing_archive/'frax_baseline').is_dir():timing_archive=timing_archive/'frax_baseline'
            previous=timing_archive/'motion.time_map.npz'
            restored_meta=json.loads((timing_archive/'motion.json').read_text())
            info=restored_meta.get('retiming',{})
            if restored_meta.get('profile')=='fast':
                timing_archive=None
            if info.get('joint_adaptation',{}).get('inserted_ik_frames'):
                if sha(previous)!=info.get('source_time_sha256'):
                    raise ValueError('Continuation timing map changed')
                with np.load(previous,allow_pickle=False) as saved:
                    joint_indices=saved['joint_prepared_index']
                    source_time=saved['source_time_s']
                t=resample_targets(t,robot,joint_indices)
                timing={k:v for k,v in info.items() if k not in ('time_map','source_time_sha256')}
    if getattr(initial_solver, 'improved', False):
        from ..contact_transition import refine as refine_transitions
        initial_solver.report['contact_transitions']=refine_transitions(t,robot)
    from ..initial_reuse import target_digest
    prepared_targets_sha256=target_digest(t)
    def finish_refine(metadata):
        if allow_local_slowdown and not repair_plan_only:
            metadata['_retiming_ik_context'] = (robot, t, cfg['solver'])
        return metadata
    if initial_from is not None:
        from ..segments import restore
        from ..refine_pipeline import repair_output
        from types import SimpleNamespace

        phase('local_repair')
        meta = restore(initial_from, output, prepared_targets_sha256, initial_marker)
        payload = pickle.loads(output.read_bytes())
        restored_solver = SimpleNamespace(report=meta['frax'])
        continuation = initial_from if initial_marker.get('continuation') else None
        initial_solver = restored_solver
        metadata = repair_output(robot, t, None, meta, payload, output, initial_solver, repair_budget=repair_budget,
                                 continue_from=continuation, repair_modules=repair_modules,
                                 plan_only=repair_plan_only, segment_cascade=True,
                                 resume_same_round=continuation is not None, refine_flow=refine_flow)
        return finish_refine(metadata)
    stage_times={};stage_start=time.monotonic()
    if hasattr(initial_solver,'source_time_s'):
        initial_solver.source_time_s = (source_origin + np.arange(len(t['root_p']))/t['fps']
                                        if source_time is None else source_time)
    if hasattr(initial_solver,'initial_collision_cfg'):
        initial_solver.initial_collision_cfg=cfg.get('self_collision')
    phase('initial',compile_possible=initial_solver is not None and not getattr(initial_solver,'strict',False))
    if initial_solver is None:phase('v8_ik')
    x,records=(solve if initial_solver is None else initial_solver)(robot,t,cfg['solver'],lambda i,n:print(f'{output.stem}: {i}/{n} {time.monotonic()-start:.1f}s',flush=True))
    robot.clip_head_angles(x[:, 3:])
    if boxes:
        from ..scene_support import fit as fit_scene
        x = fit_scene(robot,t,x)
    stage_times['initial_ik']=time.monotonic()-stage_start
    raw_frax=getattr(initial_solver,'baseline',None)
    if source_time is not None and timing_archive is None and not getattr(initial_solver,'defer_joint_retiming',False):
        # Use the actually solved hinges to find robot-only timing pressure.
        # Preserve existing sample poses and solve new samples against the same prepared targets.
        eligible=eligible_timing_intervals(source_review,source_time)
        ix,joints=time_map(t['root_r'],x[:,:3],t['fps'],angular_limit,linear_limit,
                           eligible=eligible,joint_positions=x[:,3:],joint_limits=robot.speed_limits*.999,
                           keep_original_frames=True)
        if joints.get('joint_stretched_intervals'):
            joints['motor_causes']=motor_causes(x[:,3:],t['fps'],robot.names,robot.speed_limits,joints['joint_stretched_intervals'],source_time)
            grid=np.arange(len(x))
            from ..retiming import fit_inserted_frames
            sampled=resample_targets(t,robot,ix)
            try:
                fitted=fit_inserted_frames(robot,sampled,cfg['solver'],x,ix)
            except (ValueError,RuntimeError) as error:
                timing['joint_adaptation']=dict(joints,decision='retiming_ik_rejected',reason=str(error))
            else:
                joint_indices=ix
                t,x=sampled,fitted
                source_time=np.interp(ix,grid,source_time)
                if raw_frax is not None:
                    raw_frax=np.column_stack([np.interp(ix,grid,column) for column in raw_frax.T])
                records=np.asarray(records)[np.minimum(np.floor(ix+.5).astype(int),len(grid)-1)]
                priority=getattr(initial_solver,'report',{}).get('body_priority')
                if priority is not None and priority.get('torso_reference_error_deg') is not None:
                    from ..local_refine import torso_errors
                    priority['torso_reference_error_deg']=torso_errors(robot,t,x).tolist()
                joints.update(policy='fixed_original_samples_inserted_ik_v1',inserted_ik_frames=len(ix)-len(grid),
                              inserted_root_xy='linear_between_original_poses')
                timing.update(changed=True,joint_adaptation=joints,output_frames=len(ix),
                              output_duration_s=(len(ix)-1)/t['fps'])
                prepared_targets_sha256=target_digest(t)
        elif joints['blocked_source_review_intervals']:
            timing['joint_review_blocked_intervals']=joints['blocked_source_review_intervals']
    if source_time is not None:
        saved=dict(output_time_s=np.arange(len(source_time))/motion.fps,source_time_s=source_time,
                   source_frame_float=source_time*motion.input_fps,source_fps=motion.input_fps,
                   source_sha256=sha(bvh),frame_convention='zero_based_input',policy=timing['policy'])
        if joint_indices is not None:saved['joint_prepared_index']=joint_indices
        np.savez_compressed(output.with_suffix('.time_map.npz'),**saved)
        timing.update(added_frames=len(source_time)-len(motion.local_p),
                      limit_basis=dict(root='robot_profile',
                                       root_linear_m_s=robot_profile.ROOT_LINEAR_LIMIT_M_S,
                                       root_angular_rad_s=robot_profile.ROOT_ANGULAR_LIMIT_RAD_S,
                                       joints='per_axis_robot_model_velocity'),
                      acceptance='adapted_timing_is_informational; standard independent audit')
    if raw_frax is not None:
        np.savez_compressed(output.with_suffix('.raw_frax.npz'),x=raw_frax,root_r=t['root_r'],fps=motion.fps)
    initial_x=x.copy();collision_reference=None
    legacy_seed = getattr(initial_solver, "legacy_v8_seed", False)
    result_root_r = initial_solver.root_matrices if legacy_seed else t['root_r']
    v8_refine = profile == 'refine' and unified_repair and initial_solver is None
    v8_resume = getattr(initial_solver, 'v8_refine_checkpoint', False)
    if v8_refine:
        phase('support')
        stage_start = time.monotonic()
        x, support_reports = constrain_support(robot, t, x, cfg['solver'])
        stage_times['support'] = time.monotonic() - stage_start
        arm_report = collision_report = {'executed': False}
        robot.body_contact_qp_summary = {'executed': False}
    elif refine:
        phase('support')
        stage_start=time.monotonic()
        x,support_reports=constrain_support(robot,t,x,cfg['solver'],lambda i,n:print(f'{output.stem}: support {i}/{n} {time.monotonic()-start:.1f}s',flush=True))
        stage_times['support']=time.monotonic()-stage_start
        phase('arms')
        stage_start=time.monotonic()
        x,arm_report=plan_and_refine(robot,t,x,cfg['solver'],lambda side,phase,i,n:print(f'{output.stem}: {side} {phase} {i}/{n} {time.monotonic()-start:.1f}s',flush=True))
        stage_times['arms']=time.monotonic()-stage_start
        phase('body_contacts')
        stage_start=time.monotonic()
        x,body_contact_reports=refine_body_contacts(robot,t,x,cfg['solver'],lambda i,n:print(f'{output.stem}: body_contacts {i}/{n} {time.monotonic()-start:.1f}s',flush=True))
        stage_times['body_contacts']=time.monotonic()-stage_start
        support_reports.extend(body_contact_reports)
        phase('self_collision')
        stage_start=time.monotonic()
        x,collision_report=refine_self_collision(robot,t,x,cfg['solver'],cfg.get('self_collision'),
            lambda i,n:print(f'{output.stem}: self_collision {i}/{n}',flush=True))
        from ..graduated_collision import recover
        x,collision_report=recover(robot,t,x,cfg['solver'],cfg.get('self_collision'),collision_report)
        stage_times['self_collision']=time.monotonic()-stage_start
    else:
        phase('fast_refinement')
        support_reports=[];arm_report={'executed':False};collision_report={'executed':False}
        robot.support_branch_relaxations=[];robot.body_contact_qp_summary={'executed':False}
        if getattr(initial_solver, 'improved', False):
            from ..local_refine import refine as refine_local
            stage_start=time.monotonic()
            x,local_report=refine_local(robot,t,x,cfg['solver'])
            initial_solver.report['local_refinement']=local_report
            stage_times['local_pose']=time.monotonic()-stage_start
            if unified_repair and repair_similarity=='soft':
                from ..torso_branch import recover as recover_torso_branch
                stage_start=time.monotonic()
                x,branch_report=recover_torso_branch(robot,t,x,cfg['solver'],cfg.get('self_collision'))
                initial_solver.report['torso_branch_recovery']=branch_report
                stage_times['torso_branch']=time.monotonic()-stage_start
                np.savez_compressed(output.with_suffix('.branch.npz'),x=x,root_r=t['root_r'],fps=motion.fps)
            from ..adaptive_refine import refine as refine_adaptive
            stage_start=time.monotonic()
            x,adaptive_report=refine_adaptive(robot,t,x,cfg['solver'])
            initial_solver.report['adaptive_refinement']=adaptive_report
            arm_report={'executed':any(r['stage']=='arms' and r['status']=='accepted' for r in adaptive_report['repairs']),
                        'scope':'adaptive','details_location':'frax.adaptive_refinement.repairs'}
            robot.body_contact_qp_summary={'executed':any(r['stage']=='body' and r['status']=='accepted' for r in adaptive_report['repairs']),
                                          'scope':'adaptive','details_location':'frax.adaptive_refinement.repairs'}
            stage_times['adaptive_v8']=time.monotonic()-stage_start
            stage_start=time.monotonic()
            from ..fast_collision import refine as refine_fast_collision
            collision_reference=x.copy()
            x,collision_report=refine_fast_collision(robot,t,x,cfg['solver'],cfg.get('self_collision'))
            from ..graduated_collision import recover
            x,collision_report=recover(robot,t,x,cfg['solver'],cfg.get('self_collision'),collision_report,fast=True,reference=collision_reference)
            stage_times['self_collision']=time.monotonic()-stage_start
    import os
    initial_cache_reclaim=None
    if initial_solver is not None and os.environ.get('DEX_FRAX_RELEASE_INITIAL_CACHE')=='1':
        from ..ik import release_initial_cache
        initial_cache_reclaim=release_initial_cache()
    phase('export_check')
    np.savez_compressed(output.with_suffix('.initial.npz'),x=initial_x,root_r=result_root_r,fps=motion.fps,joint_names=robot.names,solver_records=np.asarray(records))
    quat=Rotation.from_matrix(result_root_r).as_quat()
    for i in range(1,len(quat)):
        if quat[i]@quat[i-1]<0:quat[i]*=-1
    stored = robot.export_rows(x)
    payload={'fps':motion.fps,'root_pos':stored[:,:3].copy(),'root_rot':quat.astype(np.float32),
        'dof_pos':stored[:,3:].copy(),'joint_names':robot.names,'quat_order':'xyzw',
        'local_body_pos':None,'link_body_list':None,'model_sha256':sha(robot.path),'robot':robot_profile.ROBOT_ID}
    sample_times = source_origin + np.arange(len(x), dtype=float)/motion.fps if source_time is None else source_time
    payload.update(source_time_s=sample_times,source_frame_float=sample_times*motion.input_fps,source_fps=motion.input_fps)
    collision_report['export_check']=audit_self_collision(robot,payload,cfg.get('self_collision'))
    from ..quality_v8 import adjacent_speed_report
    velocity_check = adjacent_speed_report(payload['dof_pos'], motion.fps, robot.names, robot.speed_limits)
    # Adjacent speed failures are retained in quality flags; no training export.
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(pickle.dumps(payload,protocol=4))
    np.savez_compressed(output.with_suffix('.npz'),**{k:v for k,v in payload.items() if v is not None})
    diagnostics={'root_target':t['root_p'],'root_r':t['root_r'],'torso_target':t['torso_r'],'solver_records':records}
    diagnostics['head_relative_target'] = head_relative_target(t['torso_r'], t['head_r'])
    diagnostics['source_time_s']=sample_times
    support_stats=getattr(robot,'support_fast_report',None)
    if support_stats is not None:
        diagnostics['support_route']=np.array([{'bypass':0,'light':1,'fallback':2}[v['route']] for v in support_stats['frames']],dtype=np.uint8)
        diagnostics['support_qp_steps']=np.array([v['qp_steps'] for v in support_stats['frames']],dtype=np.int32)
        diagnostics['support_light_stop_reason']=np.array([v['light_stop_reason'] for v in support_stats['frames']])
        diagnostics['support_fallback_triggers']=np.array(['|'.join(v['fallback_triggers']) for v in support_stats['frames']])
        diagnostics['support_final_feasible']=np.array([v['final_feasible'] for v in support_stats['frames']],dtype=bool)
        # Detailed diagnostics remain JSON-readable without pickle-enabled NPZ.
        output.with_suffix('.support.json').write_text(json.dumps(support_stats,ensure_ascii=False,indent=2))
    for s,v in t['sides'].items():
        for k in ('foot_target_p','foot_target_r','raw_foot_p','raw_foot_r','contact','phase','anchors','pivots','heel','toe','heel_speed','toe_speed','thigh','shin','upper','lower','elbow_axis','signed_elbow_bend','hand_forward','hand_width','forearm_width','ground_contact_confidence'):
            diagnostics[s+'_'+k]=v[k]
    diagnostics['source_expected_contact']=t['source_expected_contact'];diagnostics['source_airborne']=t['source_airborne']
    if support_review is not None:
        for key in ('source_support_uncertain','source_scene_support'):
            diagnostics[key]=t[key]
        for side,values in t['sides'].items():
            for key in ('platform_index','platform_anchor'):
                diagnostics[side+'_'+key]=values[key]
    for contact in t['body_contacts']:
        for k in ['mode','confidence','anchors','source_surface_height','source_surface_point']:diagnostics['body_'+contact['body_name']+'_'+k]=contact[k]
    if foot_contact == 'v2':
        for side, values in t['sides'].items():
            for name in ('foot_flat_confidence', 'foot_side_evidence', 'foot_roll_level_weight'):
                diagnostics[f'{side}_{name}'] = values[name]
    np.savez_compressed(output.with_suffix('.diagnostics.npz'),**diagnostics)
    from ..identity import CONVERTER_RELEASE,SOLVER_POLICY
    meta={'profile':profile,'converter_version':8,'generation_mode':'frax_v8_from_original_bvh' if initial_solver is not None else 'v8_reference_from_original_bvh','pose_solver':'frax_diff_ik' if initial_solver is not None else 'v8_trf','converter_release':CONVERTER_RELEASE,'solver_policy':SOLVER_POLICY,
        'input':str(Path(bvh).resolve()),'input_sha256':sha(bvh),'source_type':t['source_type'],
        'bvh_input_convention':None if source=='smplx' else dict(units=bvh_units,translations=bvh_translations),
        'support_solver_execution':dict(executed=bool(refine or v8_refine),scope='v8_support_postprocess',status='executed' if refine or v8_refine else 'not_run',window_repairs='repair.json' if unified_repair else None),
        'input_frames':motion.input_frames,'input_fps':motion.input_fps,'output_frames':len(x),'output_fps':motion.fps,
        'model':robot.path,'model_sha256':sha(robot.path),'viewer_urdf':DEFAULT_URDF,'viewer_urdf_sha256':sha(DEFAULT_URDF),
        'joint_names':robot.names,'quat_order':'xyzw','units':'metres/radians','cpu_only':True,
        'official_reference_commit':'cbdb2b494a2af82137bf9b6f15122d60167281c4','official_runtime':False,
        'prepared_targets_sha256':prepared_targets_sha256,'configuration':cfg,'contacts_enabled':not no_contacts,'source_floor_m':t['floor'],'leg_scale':t['scale'],
        'head_calibration':t['head_calibration'],
        'ground_calibration':t['ground_calibration'],'multicontact':t['multicontact_record'],
        'body_contact_solver':robot.body_contact_qp_summary,
        'body_contact_task_policy':'bounded extra-body normal cues with existing foot support, linear body support without foot plants; independent hard mesh clearance and foot XYZ constraints',
        'body_contact_models':[{k:v for k,v in c.items() if k not in ('mode','confidence','anchors','source_surface_height','source_surface_point')} for c in t['body_contacts']],
        'world_origin':t['world_origin'].tolist(),'source_calibration':{s:{k:v[k] for k in ('foot_calibration','hand_method','elbow_calibration')} for s,v in t['sides'].items()},
        'wrist_limits':robot.wrist_limit_record,
        'estimated_contact_segments':{s:v['segments'] for s,v in t['sides'].items()},
        'arm_branch_planning':arm_report,
        'self_collision':collision_report,
        'support_constraint_failures':support_reports,
        'support_branch_relaxations':robot.support_branch_relaxations,
        'support_refinement':cfg['solver'].get('support_refinement','hybrid_cpu_v1'),
        'support_refinement_statistics':{k:v for k,v in support_stats.items() if k!='frames'} if support_stats is not None else None,
        'joint_velocity_limits_rad_s':dict(zip(robot.names,robot.speed_limits.tolist())),
        'robot_motion_profile':dict(
            velocity_source=robot_profile.VELOCITY_SOURCE,
            linear_speed_basis=robot_profile.LINEAR_SPEED_BASIS,
            joint_limits={name:robot_profile.JOINT_SPEED_LIMITS[name]._asdict() for name in robot.names}),
        'solver_velocity_check':velocity_check,
        'stage_times_s':stage_times,'frax':getattr(initial_solver,'report',None),'v8_refinement':refine,'elapsed_s':time.monotonic()-start,'output_sha256':sha(output),
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(list(ROOT.glob('*.py'))+[ROOT/'config.json'])},
        'limits':'Kinematic reference only; contact inferred; flat floor assumed; no dynamics/self-collision certification.'}
    if support_review is not None:
        meta['source_support_review']=support_review
    if boxes:
        meta['scene_support']=dict(boxes=boxes,sha256=scene_digest)
    if source_review is not None:
        meta['source_review']=dict(policy=source_review['policy'], report='motion.source_review.json',
                                   native_frame_count=source_review['frame_count'],
                                   kept_native_interval=source_review.get('kept_native_interval'),
                                   repaired_frames=source_review['repaired_frames'],
                                   discarded_frames=source_review.get('discarded_frames',[]),
                                   anomaly_segment_count=len(source_review.get('anomaly_segments', [])),
                                   events=len(source_review['events']), blocking_events=source_review.get('blocking_events',[]))
    if timing is not None:
        meta['retiming']=dict(timing,time_map='motion.time_map.npz',
                              source_time_sha256=sha(output.with_suffix('.time_map.npz')))
    if source == 'smplx':
        meta.update(generation_mode='smplx_parameters_to_dex',input_adapter=motion.metadata)
    if v8_resume:
        meta.update(generation_mode='v8_window_refine_continuation', pose_solver='v8_trf',
                    support_solver_execution=initial_solver.meta['support_solver_execution'])
    if legacy_seed:
        meta.update(generation_mode="legacy_v8_checkpoint_with_modern_window_repair",
                    pose_solver="v8_trf", v8_seed=initial_solver.report)
    if initial_cache_reclaim is not None:meta['initial_cache_reclaim']=initial_cache_reclaim
    output.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    if initial_only:return meta
    if unified_repair:
        if refine or (initial_solver is None and not v8_refine):
            raise ValueError('Unified repair requires a completed initial trajectory')
        if profile in ('fast','refine'):
            from ..refine_pipeline import repair_output
            legacy_v8 = v8_refine or legacy_seed or v8_resume
            meta=repair_output(robot,t,x,meta,payload,output,initial_solver,repair_budget=repair_budget,continue_from=continue_from,resume_recheck=resume_recheck,repair_modules=repair_modules,plan_only=repair_plan_only,v8_candidates=legacy_v8,segment_cascade=not legacy_v8,refine_flow=refine_flow)
        else:
            from ..unified_repair import repair_output
            meta=repair_output(robot,t,x,meta,payload,output,initial_solver,repair_similarity,collision_reference=collision_reference)
    return finish_refine(meta)

if __name__=='__main__':
    raise SystemExit('Use dex-frax convert --mode fast or --mode refine; standalone V8 conversion entry removed')
