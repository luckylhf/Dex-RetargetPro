"""SMPL-X anatomy adapter for the shared Dex target contract."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from dex_frax.v8.math3d import unit,frame_yz,frame_xz
from . import robot_profile as profile

def load(path, fps=None):
 import json
 with np.load(path,allow_pickle=False) as z: data={k:z[k] for k in z.files}
 meta=json.loads(str(data['metadata_json']))
 if meta.get('schema')!='smplx_parameters_v1' or float(data['fps'])!=50. or fps not in (None,50.):
  raise ValueError('Not a prepared SMPL-X geometry file at 50 Hz')
 n=len(data['positions']); joints=len(data['names'])
 if n<3 or data['positions'].shape!=(n,joints,3) or data['rotations'].shape!=(n,joints,3,3):
  raise ValueError('Invalid prepared SMPL-X geometry shape')
 landmarks=tuple(side+profile.SMPLX_SOLE_SUFFIXES[key] for side in profile.SIDE_LAYOUT for key in ('heel','toe'))
 for key in ('positions','rotations','neutral_positions',*landmarks):
  if not np.isfinite(data[key]).all():raise ValueError('Nonfinite prepared SMPL-X geometry: '+key)
 for side in profile.SIDE_LAYOUT:
  corners = data.get(side+profile.SMPLX_SOLE_SUFFIXES['corners'])
  if corners is not None and (corners.shape != (n, 4, 3) or not np.isfinite(corners).all()):
   raise ValueError('Invalid SMPL-X sole corner geometry: '+side)
 return SimpleNamespace(data=data,metadata=meta,fps=50.,input_fps=meta['fps'],input_frames=meta['frames'],local_p=data['positions'],local_r=data['rotations'])

def adapt(motion,source=None):
 z=motion.data;names=z['names'].tolist();P=z['positions'];R=z['rotations'];N=z['neutral_positions']
 idx=lambda name:names.index(name)
 point=lambda name:P[:,idx(name)]
 rot=lambda name:R[:,idx(name)]
 mapping=dict(profile.SMPLX_BODY_JOINTS)
 for s in profile.SIDE_LAYOUT:
  mapping.update({s+k:s.lower()+'_'+v for k,v in profile.SMPLX_LIMB_JOINTS.items()})
 p={k:point(v) for k,v in mapping.items()};r={k:rot(v) for k,v in mapping.items()}
 root=frame_yz(p['LeftUpLeg']-p['RightUpLeg'],r['Hips']@profile.SMPLX_PELVIS_LOCAL_UP)
 torso=frame_yz(p['LeftArm']-p['RightArm'],p['Neck']-p['Spine'])
 t=dict(root_r=root,torso_r=torso,hips=p['Hips'],source_type='smplx',fps=50.,sides={},p=p,r=r)
 if 'vertical_drift' in z:
  t['scene_ground']=dict(height_m=0.,normal=[0,0,1],method='stationarity_weighted_common_vertical_drift_v1',measured=False)
 for s in profile.SIDE_LAYOUT:
  low=s.lower();upper=unit(p[s+'ForeArm']-p[s+'Arm']);lower=unit(p[s+'Hand']-p[s+'ForeArm'])
  angle=np.arccos(np.clip((upper*lower).sum(1),-1,1));normal=unit(np.cross(upper,lower));minimum,maximum=profile.SMPLX_ELBOW_CALIBRATION_RANGE_RAD;valid=(angle>minimum)&(angle<maximum)
  local=np.einsum('nji,nj->ni',r[s+'Arm'],normal)
  axis=unit(np.median(local[valid],axis=0)) if valid.any() else np.array(profile.SMPLX_ELBOW_FALLBACK[s])
  elbow=r[s+'Arm']@axis;elbow=unit(elbow-upper*(elbow*upper).sum(1,keepdims=True))
  bend=np.arctan2((elbow*np.cross(upper,lower)).sum(1),(upper*lower).sum(1))
  fingers={role:low+'_'+name for role,name in profile.SMPLX_FINGER_JOINTS.items()}
  forward=unit(point(fingers['distal'])-p[s+'Hand']);width=point(fingers['thumb_side'])-point(fingers['little_side']);width=unit(width-forward*(width*forward).sum(1,keepdims=True))
  # SMPL-X neutral joint axes are identity; neutral finger width is in Y-up.
  C=np.array(profile.SMPLX_Y_UP_TO_Z_UP)
  fwlocal=unit(C.T@(N[idx(fingers['thumb_side'])]-N[idx(fingers['little_side'])]))
  fw=r[s+'ForeArm']@fwlocal;fw=unit(fw-lower*(fw*lower).sum(1,keepdims=True))
  heel=z[s+profile.SMPLX_SOLE_SUFFIXES['heel']];toe=z[s+profile.SMPLX_SOLE_SUFFIXES['toe']];foot=frame_xz(toe-heel,r[s+'Foot']@profile.SMPLX_ANKLE_LOCAL_UP)
  t['sides'][s]=dict(hip=p[s+'UpLeg'],knee=p[s+'Leg'],ankle=p[s+'Foot'],thigh=unit(p[s+'Leg']-p[s+'UpLeg']),shin=unit(p[s+'Foot']-p[s+'Leg']),upper=upper,lower=lower,elbow_axis=elbow,signed_elbow_bend=bend,elbow_calibration=dict(method='SMPL-X upper-arm-frame median axis from bent frames',local_axis=axis.tolist(),check_frames=int(valid.sum())),hand_forward=forward,hand_width=width,forearm_width=fw,foot_r=foot,heel=heel,toe=toe,foot_calibration='SMPL-X skin sole regions from same-beta neutral geometry; global ankle up; no per-frame grounding',hand_method=('SMPL-X explicit hand poses' if motion.metadata['hands_observed'] else 'SMPL-X relaxed default hands; not measured'))
  corners=s+profile.SMPLX_SOLE_SUFFIXES['corners']
  if corners in z:
   t['sides'][s]['sole_corners'] = z[corners]
 return t
