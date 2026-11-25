"""Short free-base PD tracking and support-surface diagnostics for exported data."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .v8.robot import Robot, DEFAULT_MODEL
from .contact_phase import surface_patch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(folder):
    folder=Path(folder)
    meta=json.loads((folder/'motion.json').read_text())
    with np.load(folder/'motion.npz',allow_pickle=False) as f:
        if f['quat_order'].item()!='xyzw':raise ValueError('Expected XYZW motion export')
        if f['model_sha256'].item()!=digest(Path(DEFAULT_MODEL)):raise ValueError('Model fingerprint mismatch')
        return meta,{k:f[k] for k in ('root_pos','root_rot','dof_pos','joint_names','fps')}


def scene_boxes(path):
    if path is None:return []
    scene=json.loads(Path(path).read_text())
    if scene.get('frame')!='robot_export_world_z_up_metres' or not scene.get('provenance'):
        raise ValueError('Scene needs export-world coordinates and a provenance description')
    boxes=scene['boxes']
    for box in boxes:
        xyz=np.asarray(box['center_m'],float)
        size=np.asarray(box['half_size_m'],float)
        if not box['id'] or xyz.shape!=(3,) or size.shape!=(3,) or not np.isfinite([xyz,size]).all() or (size<=0).any():
            raise ValueError('Invalid scene box')
    if len({b['id'] for b in boxes})!=len(boxes):raise ValueError('Duplicate scene surface')
    return boxes


def surface_at(point,boxes):
    z,name=0.,'floor'
    for box in boxes:
        xyz,size=np.asarray(box['center_m']),np.asarray(box['half_size_m'])
        if np.all(np.abs(point[:2]-xyz[:2])<=size[:2]) and xyz[2]+size[2]>z:
            z,name=xyz[2]+size[2],box['id']
    return z,name


def probe(folder,start,frames,initial_from=None,scene=None):
    folder=Path(folder)
    meta,ref=reference(folder)
    start_meta,initial=reference(initial_from or folder)
    robot=Robot()
    if not 0<=start or frames<2 or start+frames>len(ref['root_pos']):raise ValueError('Window outside trajectory')
    if (meta['input_sha256']!=start_meta['input_sha256'] or float(initial['fps'])!=float(ref['fps'])
            or len(initial['root_pos'])!=len(ref['root_pos']) or list(ref['joint_names'])!=robot.names
            or list(initial['joint_names'])!=robot.names):raise ValueError('References must have the same source, axes and timeline')
    boxes=scene_boxes(scene)
    from .model_cache import model_spec
    spec=model_spec(DEFAULT_MODEL)
    for box in boxes:
        spec.worldbody.add_geom(name='scene_'+box['id'],type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=box['center_m'],size=box['half_size_m'],contype=1,conaffinity=1)
    m=spec.compile();d=mujoco.MjData(m)
    fps=float(ref['fps']); substeps=max(1,round(1./(m.opt.timestep*fps)))
    m.opt.timestep=1./(substeps*fps)
    act=m.actuator_trnid[:,0]
    if (m.nu!=len(robot.names) or set(act)!=set(robot.jids) or np.any(m.actuator_gear[:,0]!=1.)
            or np.any(~m.actuator_ctrllimited.astype(bool))):raise ValueError('Probe requires one limited unit-gear joint motor per hinge')
    joint_order=np.array([robot.names.index(m.joint(int(j)).name) for j in act])
    axis=m.jnt_qposadr[act];vel=m.jnt_dofadr[act]
    def qpose(data,i):
        q=np.zeros(m.nq)
        q[:3]=data['root_pos'][i];q[3:7]=data['root_rot'][i,[3,0,1,2]]
        q[axis]=data['dof_pos'][i,joint_order]
        return q
    d.qpos[:]=qpose(initial,start)
    # Use the incoming velocity, identical for both paired trajectories.
    i0=max(start-1,0);i1=i0+1
    mujoco.mj_differentiatePos(m,d.qvel,1/fps,qpose(initial,i0),qpose(initial,i1))
    mujoco.mj_forward(m,d)
    maximum=np.max(np.abs(m.actuator_ctrlrange),axis=1)
    kp=np.clip(maximum,12.,180.);kd=.065*kp
    root,joint,slip,tilt=[],[],[],[]
    hits=0; pivots=[]
    env={m.geom('floor').id,*[m.geom('scene_'+b['id']).id for b in boxes]}
    foot_bodies={g['foot'] for g in robot.sides.values()}
    with np.load(folder/'motion.diagnostics.npz',allow_pickle=False) as dg:
        for k in range(start,start+frames-1):
            robot.set(np.r_[ref['root_pos'][k],ref['dof_pos'][k]],Rotation.from_quat(ref['root_rot'][k]).as_matrix())
            for side,g in robot.sides.items():
                phase=int(dg[side+'_phase'][k])
                if phase:
                    points=surface_patch(g,phase)@robot.rot(g['foot']).T+robot.point(g['foot'])
                    gaps=[(p[2]-height,surface) for p in points for height,surface in [surface_at(p,boxes)]]
                    gap,surf=min(gaps)
                    pivots.append(dict(frame=k,foot=side,phase=phase,surface=surf,gap_mm=1000.*gap))
            qa=ref['dof_pos'][k,joint_order];qb=ref['dof_pos'][k+1,joint_order]
            for s in range(substeps):
                u=s/substeps
                raw=kp*((1-u)*qa+u*qb-d.qpos[axis])+kd*((qb-qa)*fps-d.qvel[vel])
                hits+=int((np.abs(raw)>maximum).sum())
                d.ctrl[:]=np.clip(raw,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
                mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d)
            for contact in d.contact:
                g1,g2=contact.geom
                body=m.geom_bodyid[g2] if g1 in env else m.geom_bodyid[g1] if g2 in env else -1
                if body in foot_bodies:
                    jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,contact.pos,body)
                    slip.append(float(np.linalg.norm((jac@d.qvel)[:2])))
            root.append(float(np.linalg.norm(d.qpos[:3]-ref['root_pos'][k+1])))
            joint.extend(np.rad2deg(d.qpos[axis]-qb))
            actual=Rotation.from_quat(d.qpos[[4,5,6,3]])
            target=Rotation.from_quat(ref['root_rot'][k+1])
            tilt.append(np.rad2deg((actual*target.inv()).magnitude()))
    rms=lambda v:float(np.sqrt(np.mean(np.square(v))))
    return dict(scope='unassisted_free_base_fixed_joint_PD_short_simulation',source_sha256=meta['input_sha256'],
        model_sha256=meta['model_sha256'],motion_npz_sha256=digest(folder/'motion.npz'),
        initial_reference=str(Path(initial_from or folder).resolve()),initial_frame=start,frames=frames,
        duration_s=(frames-1)/fps,substeps=substeps,controller='kp=clip(motor_ctrl_limit,12,180);kd=.065*kp',
        scene='supplied_boxes_and_model_floor' if scene else 'model_floor_only_unmeasured_scene',
        scene_sha256=digest(Path(scene)) if scene else None,root_rmse_m=rms(root),joint_rmse_deg=rms(joint),
        root_rotation_rmse_deg=rms(tilt),saturation_fraction=hits/((frames-1)*substeps*m.nu),
        foot_contact_slip_rms_m_s=rms(slip) if slip else None,foot_contact_samples=len(slip),
        mujoco_warnings={str(k):int(w.number) for k,w in enumerate(d.warning) if w.number},
        reference_patch_samples=pivots)
