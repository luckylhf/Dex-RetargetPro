from pathlib import Path
import json
import mujoco
import numpy as np
from .math3d import skew

from dex_frax.paths import ASSETS

from .. import robot_profile as profile

DEFAULT_MODEL=str(ASSETS/profile.MODEL_XML)
DEFAULT_URDF=str(ASSETS/profile.MODEL_URDF)

class Robot:
    def __init__(self,path=DEFAULT_MODEL):
        from dex_frax.model_cache import load
        self.path=str(Path(path).resolve());self.m=load(self.path);self.d=mujoco.MjData(self.m)
        m=self.m
        self.jids=np.array([i for i in range(m.njnt) if m.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE])
        if (len(self.jids), m.nq, m.nv, m.nu) != (
                profile.JOINT_COUNT, profile.JOINT_COUNT + 7, profile.JOINT_COUNT + 6, profile.JOINT_COUNT):
            raise ValueError('Expected Dex 31 revolute joints and actuators plus free pelvis')
        hinges = {m.joint(int(i)).name: i for i in self.jids}
        self.names = list(profile.JOINT_ORDER)
        if set(self.names) != hinges.keys():
            raise ValueError('Robot profile joints differ from the MJCF hinge joints')
        self.jids = np.array([hinges[name] for name in self.names])
        if set(m.actuator_trnid[:, 0]) != set(self.jids):
            raise ValueError('Dex actuator joints differ from the named hinges')
        manifest = json.loads((ASSETS / profile.MODEL_MANIFEST).read_text())
        if self.names != manifest['joint_names']:
            raise ValueError('Robot profile order differs from the asset manifest')
        self.speed_limits = np.array(profile.joint_angular_limits(self.names, manifest))
        self.qadr=m.jnt_qposadr[self.jids];self.dadr=m.jnt_dofadr[self.jids]
        self.lower=np.r_[[-np.inf]*3,m.jnt_range[self.jids,0]]
        self.upper=np.r_[[np.inf]*3,m.jnt_range[self.jids,1]]
        self.cols=np.r_[0,1,2,self.dadr]
        self._root_matrix=None;self._root_quat=None;self._fk_qpos=None;self._jac_cache={}
        self.cache_stats=dict(set_calls=0,fk_calls=0,jac_calls=0,jac_hits=0,rotation_conversions=0)
        self.set(np.r_[0.,0.,1.,np.zeros(profile.JOINT_COUNT)],np.eye(3))
        self.sides={};self.torso=self.bid(profile.TORSO_BODY);self.head=self.bid(profile.HEAD_BODY)
        for side,layout in profile.SIDE_LAYOUT.items():
            g={role:self.bid(name) for role,name in layout['bodies'].items()}
            for chain in ('arm', 'leg'):
                g[chain]=[3+self.names.index(name) for name in layout[chain]]
            gid=m.geom(layout['sole_geom']).id;mid=m.geom_dataid[gid]
            v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].copy()
            world=v@self.d.geom_xmat[gid].reshape(3,3).T+self.d.geom_xpos[gid]
            local=(world-self.d.xpos[g['foot']])@self.rot(g['foot'])
            g['vertices']=local
            zmin=float(local[:,2].min());sole=local[local[:,2]<zmin+profile.SOLE_BAND_M]
            # Ground plane support polygon in the visual foot's own frame.
            # Sampling the actual mesh avoids using ankle origin as the sole.
            from scipy.spatial import ConvexHull
            hull=ConvexHull(local)
            g['hull']=local[hull.vertices]
            g['toe']=np.r_[np.percentile(sole[:,0],profile.SOLE_LENGTH_PERCENTILES[1]),np.median(sole[:,1]),zmin]
            g['heel']=np.r_[np.percentile(sole[:,0],profile.SOLE_LENGTH_PERCENTILES[0]),np.median(sole[:,1]),zmin]
            g['sole']=(g['toe']+g['heel'])/2
            g['thigh_length']=float(np.linalg.norm(self.d.xpos[g['knee']]-self.d.xpos[g['hip']]))
            g['shin_length']=float(np.linalg.norm(self.d.xpos[g['foot']]-self.d.xpos[g['knee']]))
            g['upper_length']=float(np.linalg.norm(self.d.xpos[g['elbow']]-self.d.xpos[g['shoulder']]))
            g['lower_length']=float(np.linalg.norm(self.d.xpos[g['wrist']]-self.d.xpos[g['elbow']]))
            u=self.point(g['elbow'])-self.point(g['shoulder']);u/=np.linalg.norm(u)
            v=self.point(g['wrist'])-self.point(g['elbow']);v/=np.linalg.norm(v)
            # Actual joint-origin geometry, not a hard-coded 12.9 degree offset.
            flex_axis=self.rot(g['shoulder'])@profile.ELBOW_FLEXION_AXIS
            g['elbow_geometric_zero']=float(np.arctan2(flex_axis@np.cross(u,v),u@v))
            self.sides[side]=g
        self.neutral_pos=self.d.xpos.copy()
        from .multicontact import surfaces,region_hull
        self.surface_models=surfaces(self)
        for side,g in self.sides.items():
            g['ground_hulls']={b:self.surface_models[b]['hull'] for b in [g['shoulder'],g['forearm'],g['tcp']]}
            g['ground_hulls'][g['tcp']]=region_hull(self,side+'_hand',g['tcp'])

    def joint_step(self, fps, cfg):
        # Array is in joint_names order, without the three root translations.
        numerical_limit = np.deg2rad(cfg['numerical_joint_speed_deg_s'])
        return np.minimum(self.speed_limits, numerical_limit) / fps

    def bid(self,name):
        i=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,name)
        if i<0: raise ValueError(name)
        return i
    def rot(self,b):return self.d.xmat[b].reshape(3,3)
    def set(self,x,root_r):
        from scipy.spatial.transform import Rotation
        self.cache_stats['set_calls']+=1
        if self._root_matrix is None or not np.array_equal(root_r,self._root_matrix):
            self._root_matrix=np.array(root_r,copy=True)
            self._root_quat=Rotation.from_matrix(root_r).as_quat()[[3,0,1,2]]
            self.cache_stats['rotation_conversions']+=1
        self.d.qpos[:3]=x[:3];self.d.qpos[3:7]=self._root_quat;self.d.qpos[self.qadr]=x[3:]
        if self._fk_qpos is not None and np.array_equal(self.d.qpos,self._fk_qpos):return
        mujoco.mj_kinematics(self.m,self.d);mujoco.mj_comPos(self.m,self.d)
        self._fk_qpos=self.d.qpos.copy();self._jac_cache.clear();self.cache_stats['fk_calls']+=1
    def point(self,b,local=None):
        return self.d.xpos[b].copy() if local is None else self.d.xpos[b]+self.rot(b)@local
    def jac(self,b,local=None):
        self.cache_stats['jac_calls']+=1
        key=(int(b),None if local is None else tuple(local))
        if key in self._jac_cache:
            self.cache_stats['jac_hits']+=1
            return tuple(a.copy() for a in self._jac_cache[key])
        jp=np.zeros((3,self.m.nv));jr=np.zeros_like(jp)
        mujoco.mj_jac(self.m,self.d,jp,jr,self.point(b,local),b)
        result=jp[:,self.cols],jr[:,self.cols]
        self._jac_cache[key]=tuple(a.copy() for a in result)
        return result
    def lowest(self,side):
        g=self.sides[side];v=g['hull'];return v[np.argmin(v@self.rot(g['foot'])[2,:])]
