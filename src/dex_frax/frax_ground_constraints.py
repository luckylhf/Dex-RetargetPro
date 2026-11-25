"""Support, swept floor and root-velocity constraints for whole-motion limb IK."""
from .robot_profile import POSE_SIZE
import threading

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from frax.ik.constraints import InequalityConstraint
from scipy.spatial.transform import Rotation

from .airborne import protected_height
from .refine_routing import descendant_bodies

SUBDIVISIONS = 8
CLEARANCE_M = .0001
ROOT_CORRECTION_SPEED_M_S = .20
STANCE_STEP_PER_AXIS_M = .00035
ROOT_REFERENCE_BAND_M = .05


def stage_targets(t):
    """Rotations use exactly the shortest-arc interpolation of the export audit."""
    reference = np.asarray(t['full_chain_reference'])
    rotations = np.asarray(t['root_r'])
    if reference.shape != (len(rotations), POSE_SIZE) or not np.isfinite(reference).all():
        raise ValueError('Invalid whole-motion Frax reference')
    previous_r = np.concatenate([rotations[:1],rotations[:-1]])
    delta = Rotation.from_matrix(previous_r.transpose(0,2,1) @ rotations).as_rotvec()
    fractions = np.arange(1, SUBDIVISIONS+1) / SUBDIVISIONS
    between = Rotation.from_rotvec((delta[:,None,:]*fractions[None,:,None]).reshape(-1,3))
    sampled = previous_r[:,None] @ between.as_matrix().reshape(-1,SUBDIVISIONS,3,3)
    sampled[:,-1] = rotations
    # Quaternions are prepared only once for the MuJoCo callbacks.
    quat = Rotation.from_matrix(sampled.reshape(-1,3,3)).as_quat().reshape(-1,SUBDIVISIONS,4)
    return dict(floor_quaternions=quat[:,:,[3,0,1,2]], previous_r=previous_r,
                protected_z=protected_height(t,len(rotations)), reference_z=reference[:,2])


def root_tube(robot,t):
    """Feasible whole-film Z correction corridor about a smoothed Frax reference.

    For the leg solve every upper hull is fixed relative to the pelvis. Its
    required translation gives an exact independent lower bound on root Z. The
    backward/forward envelope admits the permitted source-relative root rate.
    """
    pose=np.asarray(t['priority_pose'])
    source=np.asarray(t['root_p'])[:,2]
    legs=[i for side in robot.sides.values() for i in side['leg']]
    fixed=sorted(set(robot.surface_models)-descendant_bodies(robot,legs))
    lower=np.empty(len(source))
    for i,(row,rotation) in enumerate(zip(pose,t['root_r'])):
        robot.set(row,rotation)
        low=min(robot.d.xpos[b,2]+np.min(robot.surface_models[b]['hull']@robot.rot(b)[2]) for b in fixed)
        lower[i]=row[2]-low+CLEARANCE_M-source[i]
    step=ROOT_CORRECTION_SPEED_M_S/float(t['fps'])
    # The smallest rate-bounded upper envelope of the mandatory heights.
    for i in range(1,len(lower)):
        lower[i]=max(lower[i],lower[i-1]-step)
    for i in range(len(lower)-2,-1,-1):
        lower[i]=max(lower[i],lower[i+1]-step)
    # Preserve the stable, data-driven height supplied by raw Frax where it
    # permits floor clearance. Fit its offsets with bounded differences.
    from scipy import sparse
    import osqp
    n=len(source)
    eye=sparse.eye(n,format='csc')
    diff=sparse.diags([-1.,1.],[0,1],shape=(n-1,n),format='csc')
    A=sparse.vstack([eye,diff],format='csc')
    s=osqp.OSQP()
    s.setup(P=(eye+5.*diff.T@diff).tocsc(),q=source-pose[:,2],A=A,
            l=np.r_[lower,np.full(n-1,-step)],u=np.r_[np.full(n,np.inf),np.full(n-1,step)],
            verbose=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=30000)
    v=s.solve()
    if v.x is None or v.info.status_val not in (1,2):
        raise RuntimeError('Infeasible whole-film reference root-height corridor')
    return np.stack([source+np.maximum(lower,v.x-ROOT_REFERENCE_BAND_M),
                     source+v.x+ROOT_REFERENCE_BAND_M],axis=-1)


def support_targets(values, sole):
    count = len(values['contact'])
    platform = np.asarray(values.get('platform_index',np.full(count,-1))) >= 0
    pivot = np.where(platform[:,None],sole,values['pivots'])
    anchor = np.where(platform[:,None],values.get('platform_anchor',np.zeros((count,3))),
                      values['anchors'])
    active = (np.asarray(values['contact']) > 0) | platform
    phase = np.asarray(values.get('phase',active))
    stable = np.r_[False, active[1:] & active[:-1] & (phase[1:]==phase[:-1])
                   & (platform[1:] == platform[:-1])
                   & np.all(np.isclose(pivot[1:],pivot[:-1],atol=1e-8,rtol=0),axis=1)
                   & np.all(np.isclose(anchor[1:,:2],anchor[:-1,:2],atol=1e-8,rtol=0),axis=1)]
    return dict(stance_active=active, stance_pivot=pivot, stance_anchor=anchor, stance_stable=stable)


@jax.tree_util.register_static
class RootContinuity(InequalityConstraint):
    """Allow a bounded Z correction relative to the incoming source step."""
    def __init__(self, space, fps, column):
        self.robot=space
        self.column=int(column)
        self.gain=float(fps)
        self.num_rows=2
        self.penalty=1e9
        axis=np.eye(space.nv)[self.column]
        self.matrix=np.stack([axis,-axis])

    def bounds(self, ref):
        source_goal=ref['root_velocity_goal'][2]
        radius=ROOT_CORRECTION_SPEED_M_S/self.gain
        lo=jnp.where(ref['first'], -jnp.inf, source_goal-radius)
        hi=jnp.where(ref['first'], jnp.inf, source_goal+radius)
        if 'root_tube' in ref:
            lo=jnp.maximum(lo,ref['root_tube'][0]);hi=jnp.maximum(lo,jnp.minimum(hi,ref['root_tube'][1]))
        # Smoothly approach the incoming protected height if a previous ground
        # correction makes exact recovery on the first flight step infeasible.
        protected=jnp.clip(ref['reference_z'],lo,hi)
        return (jnp.where(ref['protected_z'],protected,lo),
                jnp.where(ref['protected_z'],protected,hi))

    def rows(self, q, ref):
        low,high=self.bounds(ref)
        # Inactive bounds are represented by finite slack for elastiqp.
        return self.matrix,self.gain*jnp.minimum(1e3,jnp.array([high-q[self.column],q[self.column]-low]))


@jax.tree_util.register_static
class SweptFloor(InequalityConstraint):
    """Linearize the lowest incoming eighth-step for each movable visual hull."""
    def __init__(self, space, robot, fps, columns):
        self.robot=space
        self.model=robot.m
        self.data=mujoco.MjData(self.model)
        self.columns=np.asarray(columns,int)
        self.qadr=robot.qadr
        self.velocity_cols=robot.cols[self.columns]
        self.gain=float(fps)
        self.penalty=1e8
        bodies=set(robot.surface_models)
        if 2 not in self.columns:
            bodies &= descendant_bodies(robot,self.columns[self.columns >= 3])
        self.bodies=np.array(sorted(bodies),int)
        self.num_rows=len(self.bodies)
        self.hulls=[robot.surface_models[b]['hull'] for b in self.bodies]
        self.centers=np.array([(v.min(axis=0)+v.max(axis=0))/2 for v in self.hulls])
        self.radii=np.array([np.linalg.norm(v-c,axis=1).max() for v,c in zip(self.hulls,self.centers)])
        self.jacp=np.empty((3,self.model.nv))
        self.lock=threading.Lock()

    def _set(self, row, quat):
        self.data.qpos[:3]=row[:3]
        self.data.qpos[3:7]=quat
        self.data.qpos[self.qadr]=row[3:]
        mujoco.mj_kinematics(self.model,self.data)
        mujoco.mj_comPos(self.model,self.data)

    def measure(self, q, previous, quats, first):
        q=np.asarray(q);previous=np.asarray(previous);quats=np.asarray(quats)
        with self.lock:
            shape=(self.num_rows,len(self.columns))
            jac=np.zeros(shape)
            gap=np.ones(self.num_rows)
            steps=(SUBDIVISIONS-1,) if bool(first) else range(SUBDIVISIONS)
            for k in steps:
                alpha=(k+1)/SUBDIVISIONS
                self._set(previous+alpha*(q-previous),quats[k])
                vz=self.data.xmat[self.bodies,6:9]
                origin_z=self.data.xpos[self.bodies,2]
                lower=origin_z+np.einsum('ij,ij->i',self.centers,vz)-self.radii
                # The enclosing sphere only eliminates hulls strictly above
                # both the activation height and the best sampled clearance.
                for j in np.flatnonzero(lower < np.minimum(.015,gap+CLEARANCE_M)):
                    hull=self.hulls[j]
                    z=hull @ vz[j]
                    index=int(z.argmin())
                    clearance=origin_z[j]+z[index]-CLEARANCE_M
                    if clearance >= gap[j]:
                        continue
                    b=self.bodies[j]
                    point=self.data.xpos[b]+self.data.xmat[b].reshape(3,3) @ hull[index]
                    mujoco.mj_jac(self.model,self.data,self.jacp,None,point,b)
                    jac[j]=-alpha*self.jacp[2,self.velocity_cols]
                    gap[j]=clearance
            return jac,gap

    def rows(self, q, ref):
        full=ref['priority_pose'].at[self.columns].set(q)
        G,h=jax.pure_callback(self.measure,
            (jax.ShapeDtypeStruct((self.num_rows,len(self.columns)),q.dtype),
             jax.ShapeDtypeStruct((self.num_rows,),q.dtype)),
            full,ref['previous'],ref['floor_quaternions'],ref['first'],vmap_method='sequential')
        return G,self.gain*h
