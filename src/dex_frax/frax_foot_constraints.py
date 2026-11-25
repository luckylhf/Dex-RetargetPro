"""Exact declared foot geometry rows in the refine foot-stage Frax QP.

CPU callbacks use the same MuJoCo/Mink geometry as the independent output audit.
This is an initial solve constraint, not a rebase of a repair's fixed reference.
"""
from .robot_profile import POSE_SIZE
import threading
import numpy as np
import jax
import jax.numpy as jnp
from frax.ik.constraints import InequalityConstraint


from .frax_arm_constraints import ArmGeometry


@jax.tree_util.register_static
class FootGeometry(ArmGeometry):
    """The complete owned leg class, including the incoming swept segment."""
    def __init__(self,kinematics,robot,gain,columns=None):
        super().__init__(kinematics,robot,gain,columns=columns,pair_group='leg_collision')
        self.margin=.0001
        self.report.update(policy='declared_complete_leg_geometry_in_initial_frax_swept_v2',
            clearance_target_m=self.margin,scope=self.geometry.names)


@jax.tree_util.register_static
class FootSupport(InequalityConstraint):
    """Actual sole floor and the existing 0.5 mm stance support box."""
    def __init__(self,kinematics,robot,gain,columns=None,space=None,full_chain=False):
        self.robot=kinematics if space is None else space
        self.kinematics=kinematics;self.sides=robot.sides;self.gain=float(gain)
        self.columns=np.arange(POSE_SIZE) if columns is None else np.asarray(columns,int)
        self.full_chain=full_chain
        self.num_rows=22 if full_chain else 14
        self.penalty=(np.r_[np.repeat(4e8,2),np.repeat(4e8,12),np.repeat(1e8,8)]
                      if full_chain else 1e8)

    def rows(self,q,ref):
        def values(x):
            if len(self.columns)!=POSE_SIZE:x=ref['priority_pose'].at[self.columns].set(x)
            if 'priority_mask' in ref:x=jnp.where(ref['priority_mask'],ref['priority_pose'],x)
            tf=self.kinematics.transforms(x,ref['root_r']);p=tf[:,:3,3];r=tf[:,:3,:3];out=[]
            for side,g in self.sides.items():
                v=ref['sides'][side];body=g['foot']
                pivot=v['stance_pivot'] if self.full_chain else v['pivots']
                point=p[body]+r[body]@pivot
                height=p[body,2]+jnp.min(jnp.asarray(g['hull'])@r[body,2])
                out.extend([height,point[0],point[1]])
            return jnp.stack(out)
        value=values(q);J=jax.jacfwd(values)(q);floor=jnp.array([0,3])
        foot_floor=value[floor]
        active=jnp.repeat(jnp.array([ref['sides'][s]['stance_active'] if self.full_chain else ref['sides'][s]['contact']>0 for s in self.sides]),3)
        anchors=jnp.stack([ref['sides'][s]['stance_anchor' if self.full_chain else 'anchors'] for s in self.sides])
        target=jnp.stack([anchors[:,2] if self.full_chain else jnp.zeros(2),anchors[:,0],anchors[:,1]],axis=1).ravel()
        error=value-target
        G=jnp.concatenate([-J[floor],jnp.where(active[:,None],J,0.),jnp.where(active[:,None],-J,0.)])
        h=jnp.concatenate([foot_floor-.0001,jnp.where(active,.0005-error,1.),jnp.where(active,.0005+error,1.)])
        if self.full_chain:
            from .frax_ground_constraints import STANCE_STEP_PER_AXIS_M
            tf=self.kinematics.transforms(ref['previous'],ref['previous_r'])
            old=jnp.stack([(tf[g['foot'],:3,3]+tf[g['foot'],:3,:3]@ref['sides'][s]['stance_pivot'])[:2]
                           for s,g in self.sides.items()]).ravel()
            xy=jnp.array([1,2,4,5])
            moving=value[xy]-old
            # If a contact starts outside its anchor box, allow the foot to
            # recover toward the marked world point without penalizing that
            # motion a second time as slip. The opposite direction stays bound.
            recovery=target[xy]-old
            step_low=jnp.minimum(-STANCE_STEP_PER_AXIS_M,recovery)
            step_high=jnp.maximum(STANCE_STEP_PER_AXIS_M,recovery)
            stable=jnp.repeat(jnp.array([ref['sides'][s]['stance_stable'] for s in self.sides]),2)
            link=jnp.where(stable[:,None],J[xy],0.)
            G=jnp.concatenate([G,link,-link])
            h=jnp.concatenate([h,jnp.where(stable,step_high-moving,1.),
                               jnp.where(stable,moving-step_low,1.)])
        return G,self.gain*h
