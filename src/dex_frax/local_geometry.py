"""Native/support and swept geometry rows for the existing sparse local fit."""
import numpy as np
import mujoco
from scipy import sparse
from scipy.spatial.transform import Rotation
from mink.limits.collision_avoidance_limit import compute_contact_normal_jacobian
from .support_quality import preserve_contact, same_support_step
from .v8.support_acceptance import SUPPORT_ACCEPTANCE_M


def platform_contact(robot, group, anchor):
    body=group['foot']
    hull=group['hull']
    vertex=hull[np.argmin(hull@robot.rot(body)[2])]
    error=np.r_[robot.point(body,group['sole'])[:2]-anchor[:2],
                robot.point(body,vertex)[2]-anchor[2]]
    return error,vertex


class Geometry:
    def __init__(self,robot,t,xs,a,b,ids,moves,restore=False,fixed_frames=(),*,initial_leg_repair=False,collision_cfg=None,frame_reconstruction=False,coupled_support=False,preserve_support_step=False,coupled_leg_clearance=False,restore_violations_only=False):
        self.r,self.t,self.xs,self.a,self.b,self.ids=robot,t,xs,a,b,np.asarray(ids)
        self.restore=restore;self.fixed=set(fixed_frames);self.n=b-a;self.width=len(ids)
        self.initial_leg_repair=initial_leg_repair
        self.frame_reconstruction=frame_reconstruction
        self.coupled_support=coupled_support
        self.coupled_leg_clearance=coupled_leg_clearance
        self.preserve_support_step = preserve_support_step
        self.support_step_bounds = {}
        self.elastic_rows=[]
        self.bodies=[body for body in robot.surface_models if moves(body)]
        self.collision=None;self.palms=None
        joints=[j for j in ids if j>=3]
        sides=[side for side,g in robot.sides.items() if set(ids)&set(g['arm'])]
        if joints:
            from .v8.self_collision import CollisionModel
            from .refine_routing import movable_submodel,collision_groups
            self.collision=movable_submodel(CollisionModel(robot,dict(collision_cfg or {},coverage='full')),joints)
            passive=np.zeros(len(self.collision.pairs),bool)
            if not sides:passive[collision_groups(self.collision)['collision']]=True
            if coupled_leg_clearance:
                self.leg_pairs = np.zeros(len(self.collision.pairs),bool)
                self.leg_pairs[collision_groups(self.collision)['leg_collision']] = True
            if coupled_support:
                sides=[side for side in sides if all(k in t['sides'][side] for k in ('hand_forward','hand_width'))]
            if sides:
                from .palm_constraints import SourcePalms
                self.palms=SourcePalms(robot,t,xs,a,b,sides,ids,interior_deg=0. if coupled_support else .1,fixed_frames=sorted(self.fixed))
        deltas=Rotation.from_matrix(t['root_r'][a:b-1].transpose(0,2,1)@t['root_r'][a+1:b]).as_rotvec()
        self.samples=[];self.native=[]
        self.platform_rows={}
        platform=t.get('_inferred_platform_repair')
        previous_points = {}
        for j in range(self.n):
            for part in range(8 if j+1<self.n else 1):
                u=part/8.;rotation=t['root_r'][a+j] if not part else t['root_r'][a+j]@Rotation.from_rotvec(deltas[j]*u).as_matrix()
                row=xs[a+j].copy() if not part else xs[a+j]*(1-u)+xs[a+j+1]*u
                robot.set(row,rotation)
                immutable=j in self.fixed and (not part or j+1 in self.fixed)
                before_floor=np.array([float((robot.surface_models[body]['hull']@robot.rot(body)[2]+robot.point(body)[2]).min()) for body in self.bodies])
                floor_target=np.minimum(-.00099,before_floor-1e-7)
                if restore and not immutable:
                    needs_clearance = before_floor < -SUPPORT_ACCEPTANCE_M if restore_violations_only else slice(None)
                    floor_target[needs_clearance] = -.0009 if restore_violations_only else -.0008
                target=None;before_distance=None
                if self.collision is not None:
                    self.collision.update(row,rotation);before_distance=self.collision.distances().copy()
                    target=np.minimum(.0001,before_distance-1e-7)
                    if restore and not immutable:
                        owned = ~passive
                        if restore_violations_only:
                            owned &= before_distance < -self.collision.cfg['penetration_tolerance_m']
                        target[owned]=.0001
                self.samples.append((j,u,rotation,row,floor_target,target,before_floor,before_distance))
                if platform is not None and not immutable:
                    for side, group in robot.sides.items():
                        if not moves(group['foot']):continue
                        foot=t['sides'][side]
                        i=a+j
                        owner=foot['platform_index'][i]
                        if owner<platform or (part and foot['platform_index'][i+1]!=owner):
                            continue
                        anchor=foot['platform_anchor'][i]
                        error,_=platform_contact(robot,group,anchor)
                        # Use the existing platform-phase interior target and
                        # taper toward exact, protected boundary samples.
                        steps=min((abs(j-fixed) for fixed in self.fixed),default=8)
                        if part:steps=min(steps,min((abs(j+1-fixed) for fixed in self.fixed),default=8))
                        weight=.85*np.sin(np.pi/2.*min(steps/8.,1.))**2
                        radius=max(.00097,float(np.linalg.norm(error)))
                        floor=min(-.00097,float(error[2]))
                        radius+=weight*(.00070-radius)
                        floor+=weight*(max(floor,-.00075)-floor)
                        self.platform_rows.setdefault(len(self.samples)-1,[]).append((group,anchor,radius,floor))
            robot.set(xs[a+j],t['root_r'][a+j]);feet=[];body_contacts=[]
            for side,g in robot.sides.items():
                if not moves(g['foot']) or t['sides'][side]['contact'][a+j]<=0:continue
                v=t['sides'][side];point=robot.point(g['foot'],v['pivots'][a+j]);low=(g['hull']@robot.rot(g['foot'])[2]+robot.point(g['foot'])[2]).min()
                error=np.r_[point[:2]-v['anchors'][a+j,:2],low]
                # Leave an interior corridor below the unchanged 1 mm gate.
                # Pinning a passing contact to its current norm can stall SQP
                # at a curved boundary during otherwise feasible smoothing.
                norm = float(np.linalg.norm(error))
                bound = max(.00099,norm)
                if restore and j not in self.fixed and (not restore_violations_only or norm > SUPPORT_ACCEPTANCE_M):
                    bound = .00092 if restore_violations_only else .0008
                tolerance=np.full(3,bound)
                feet.append((side,tolerance,bound,max(.001,norm)))
                if preserve_support_step:
                    if j and same_support_step(v, a+j) and not {j-1,j}.issubset(self.fixed):
                        previous_step = np.linalg.norm(point[:2] - previous_points[side])
                        # Leave a numerical margin below the acceptance limit,
                        # while allowing existing larger steps to improve in place.
                        self.support_step_bounds[j,side] = max(SUPPORT_ACCEPTANCE_M - 1e-5, previous_step)
                    previous_points[side] = point[:2]
            for c in t.get('body_contacts',[]):
                if c['body'] not in self.bodies or not preserve_contact(c,a+j):continue
                body=c['body'];hull=np.asarray(c['robot_hull']);local=hull[np.argmin(hull@robot.rot(body)[2])]
                point = robot.point(body,local)
                if self.frame_reconstruction or self.coupled_support:
                    anchor = c['anchors'][a+j,:2]
                    z_limit = t['body_contact_height_tolerance_m']
                    xy_limit = t.get('body_contact_anchor_tolerance_m',z_limit)
                    z_error=abs(point[2])
                    xy_error=np.linalg.norm(point[:2]-anchor)
                    if self.coupled_support:
                        # The refine adoption check also protects each trusted
                        # component below its global acceptance threshold.
                        z_bound=min(max(.995*z_limit,z_error),max(.001,z_error)+.0004)
                        xy_bound=min(max(.995*xy_limit,xy_error),max(.001,xy_error)+.0004)
                    else:
                        z_bound=max(.995*z_limit,z_error-1e-6)
                        xy_bound=max(.995*xy_limit,xy_error-1e-6)
                    body_contacts.append((body,hull,anchor,z_bound,xy_bound,int(c['mode'][a+j])))
                else:
                    body_contacts.append((body,hull,local.copy(),point,int(c['mode'][a+j])))
            self.native.append((feet,body_contacts))
        # Unfixed boundary poses also constrain their incoming/outgoing seams.
        for j,frame,neighbor in ((-1,a-1,a),(self.n-1,b-1,b)):
            if not (0<=frame<len(xs) and 0<=neighbor<len(xs)):continue
            delta=Rotation.from_matrix(t['root_r'][frame].T@t['root_r'][neighbor]).as_rotvec()
            for part in range(1,8):
                u=part/8.;rotation=t['root_r'][frame]@Rotation.from_rotvec(delta*u).as_matrix();row=xs[frame]*(1-u)+xs[neighbor]*u
                robot.set(row,rotation);bf=np.array([float((robot.surface_models[body]['hull']@robot.rot(body)[2]+robot.point(body)[2]).min()) for body in self.bodies]);ft=np.minimum(-.00099,bf-1e-7)
                bd=dt=None
                if self.collision is not None:self.collision.update(row,rotation);bd=self.collision.distances().copy();dt=np.minimum(.0001,bd-1e-7)
                self.samples.append((j,u,rotation,row,ft,dt,bf,bd))
        if self.collision is not None:
            self._normal=np.empty(3);self._jac1=np.empty((3,self.collision.m.nv));self._jac2=np.empty_like(self._jac1);self._fromto=np.empty(6)
        self.last={}

    def evaluate(self,z,linear=False):
        r=self.r;qs=z.reshape(self.n,self.width);rows=[];columns=[];values=[];lower=[];upper=[];penalty=0.;worst_floor={};worst_pair={}
        floor_cost=0.;floor_jac=[];floor_residual=[]
        collision_cost=0.;collision_jac=[];collision_residual=[]
        body_cost=0.;body_jac=[];body_residual=[]
        nonreg=0.;floor_reg=0.;contact_reg=0.;floor_min=float('inf');contact_max=0.;collision_min=float('inf');interior_floor=float('inf');interior_collision=float('inf')
        previous_feet = {}

        def append_frame(index, j, jacobian):
            nz = np.flatnonzero(abs(jacobian) > 1e-14)
            rows.extend([index]*len(nz))
            columns.extend((j*self.width+nz).tolist())
            values.extend(jacobian[nz].tolist())

        def add(J,lo,hi,j,u=0.,*,next_jacobian=None):
            J=np.atleast_2d(J);lo=np.broadcast_to(np.atleast_1d(lo),(len(J),));hi=np.broadcast_to(np.atleast_1d(hi),(len(J),))
            if next_jacobian is not None:
                next_jacobian = np.atleast_2d(next_jacobian)
            for k,row in enumerate(J):
                index=len(lower)
                if j>=0:
                    append_frame(index,j,row*(1-u))
                if u and j+1<self.n:
                    append_frame(index,j+1,row*u)
                if next_jacobian is not None:
                    append_frame(index,j+1,next_jacobian[k])
                lower.append(float(lo[k]));upper.append(float(hi[k]))
        def pose(j,u,base):
            x=base.copy()
            if j<0:x[self.ids]=self.xs[self.a-1,self.ids]*(1-u)+qs[0]*u
            elif j+1>=self.n and u:x[self.ids]=qs[-1]*(1-u)+self.xs[self.b,self.ids]*u
            else:x[self.ids]=qs[j] if not u else qs[j]*(1-u)+qs[j+1]*u
            return x
        for index,(j,u,rotation,base,ft,dt,bf,bd) in enumerate(self.samples):
            x=pose(j,u,base);r.set(x,rotation)
            movable=(0<=j<self.n and j not in self.fixed) or (u and 0<=j+1<self.n and j+1 not in self.fixed)
            for group,anchor,radius,minimum in self.platform_rows.get(index,()):
                error,vertex=platform_contact(r,group,anchor)
                height=error[2]
                penalty+=max(minimum-height,0.)**2
                magnitude=np.linalg.norm(error)
                if not u:penalty+=max(magnitude-radius,0.)**2
                if linear:
                    vertical,_=r.jac(group['foot'],vertex)
                    add(vertical[2,self.ids],minimum-height,np.inf,j,u)
                    if not u:
                        xy,_=r.jac(group['foot'],group['sole'])
                        matrix=np.vstack([xy[:2],vertical[2]])[:,self.ids]
                        add(matrix,-radius-error,radius-error,j)
                        if magnitude>1e-10:add((error/magnitude)@matrix,-np.inf,radius-magnitude,j)
            for k,body in enumerate(self.bodies):
                hull=r.surface_models[body]['hull'];heights=hull@r.rot(body)[2]+r.point(body)[2];v=int(np.argmin(heights));height=float(heights[v]);gap=height-ft[k]
                floor_min=min(floor_min,height)
                if 0<=j<self.n and not (j==self.n-1 and u):interior_floor=min(interior_floor,height)
                penalty+=max(-gap,0.)**2;floor_reg=max(floor_reg,min(bf[k],-.001)-height);nonreg=max(nonreg,floor_reg)
                key=(j,body)
                if self.coupled_support:
                    # Penalize remaining penetration only after setting the
                    # per-body, per-sample non-regression bounds.
                    if height<-.001 and not (j in self.fixed and (not u or j+1 in self.fixed)):
                        floor_cost+=float((height+.001)**2)
                    if linear and height<.012:
                        worst_floor[(j,body,index)]=(gap,index,hull[v].copy())
                elif linear and height<.02:
                    if not u:
                        for vertex in np.argsort(heights)[:4]:worst_floor[(j,body,int(vertex))]=(float(heights[vertex]-ft[k]),index,hull[vertex].copy())
                    elif key not in worst_floor or gap<worst_floor[key][0]:worst_floor[key]=(gap,index,hull[v].copy())
            if self.collision is not None:
                self.collision.update(x,rotation);distance=self.collision.distances();deficit=np.maximum(dt-distance,0.);penalty+=float(deficit@deficit)
                if self.coupled_leg_clearance and movable:
                    owned=self.leg_pairs & (bd < -self.collision.cfg['penetration_tolerance_m'])
                    remaining=np.maximum(.0001-distance[owned],0.)
                    collision_cost+=float(remaining@remaining)
                collision_min=min(collision_min,float(distance.min()))
                if 0<=j<self.n and not (j==self.n-1 and u):interior_collision=min(interior_collision,float(distance.min()))
                nonreg=max(nonreg,float((np.minimum(bd,0.)-distance).max()))
                if linear:
                    for k in np.flatnonzero(distance<.02):
                        key=(j,int(k),index);gap=distance[k]-dt[k]
                        if key not in worst_pair or gap<worst_pair[key][0]:worst_pair[key]=(float(gap),index,float(distance[k]))
            if u:continue
            feet,contacts=self.native[j]
            for side,tolerance,bound,original_bound in feet:
                g=r.sides[side];v=self.t['sides'][side];point=r.point(g['foot'],v['pivots'][self.a+j]);hull=g['hull'];low=hull[np.argmin(hull@r.rot(g['foot'])[2])]
                value=np.r_[point[:2]-v['anchors'][self.a+j,:2],r.point(g['foot'],low)[2]];norm=float(np.linalg.norm(value));contact_max=max(contact_max,norm)
                penalty+=max(norm-bound,0.)**2;contact_reg=max(contact_reg,norm-original_bound);nonreg=max(nonreg,contact_reg)
                if linear:
                    P,_=r.jac(g['foot'],v['pivots'][self.a+j]);G,_=r.jac(g['foot'],low);J=np.vstack([P[:2],G[2]])[:,self.ids]
                    search_bound=bound
                    if self.coupled_leg_clearance and movable:
                        # Give the nonlinear anchor check room for the foot
                        # curvature omitted by a single Jacobian.
                        search_bound=max(.0009,min(bound,original_bound-.0001))
                    add(J,-tolerance-value,tolerance-value,j)
                    if norm>1e-10:add((value/norm)@J,-np.inf,search_bound-norm,j)
                if self.preserve_support_step:
                    step_bound = self.support_step_bounds.get((j,side))
                    foot_jacobian = J[:2] if linear else None
                    if step_bound is not None:
                        previous_point, previous_jacobian = previous_feet[side]
                        step = point[:2] - previous_point
                        length = np.linalg.norm(step)
                        # The line search checks the exact norm after updating both poses.
                        penalty += max(0.,length-step_bound)**2
                        contact_reg = max(contact_reg,length-step_bound)
                        nonreg = max(nonreg,contact_reg)
                        if linear:
                            search_bound=step_bound-.00008 if self.coupled_leg_clearance else step_bound
                            add(-previous_jacobian,-search_bound-step,search_bound-step,j-1,
                                next_jacobian=foot_jacobian)
                            if length>1e-10:
                                direction = step/length
                                add(-(direction@previous_jacobian),-np.inf,search_bound-length,j-1,
                                    next_jacobian=direction@foot_jacobian)
                    previous_feet[side] = (point[:2],foot_jacobian)
            if self.frame_reconstruction or self.coupled_support:
                for body,hull,anchor,z_bound,xy_bound,mode in contacts:
                    local=hull[np.argmin(hull@r.rot(body)[2])]
                    point=r.point(body,local)
                    xy=point[:2]-anchor
                    norm=float(np.linalg.norm(xy))
                    z_excess=max(0.,abs(point[2])-z_bound)
                    xy_excess=max(0.,norm-xy_bound) if mode==1 else 0.
                    penalty+=z_excess**2+xy_excess**2
                    nonreg=max(nonreg,z_excess,xy_excess)
                    height_limit=self.t['body_contact_height_tolerance_m']
                    height_residual=np.copysign(max(0.,abs(point[2])-height_limit),point[2])
                    if self.coupled_leg_clearance and movable:
                        body_cost+=height_residual**2
                    if linear:
                        P,_=r.jac(body,local);J=P[:,self.ids]
                        if self.coupled_leg_clearance and movable and height_residual:
                            body_jac.append(len(lower))
                            body_residual.append(height_residual)
                        add(J[2],-z_bound-point[2],z_bound-point[2],j)
                        if mode==1 and norm>1e-10:
                            add((xy/norm)@J[:2],-np.inf,xy_bound-norm,j)
            else:
                for body,hull,local,point,mode in contacts:
                    if mode!=1:local=hull[np.argmin(hull@r.rot(body)[2])]
                    value=r.point(body,local)-point
                    if mode!=1:value=value[2:]
                    excess=np.maximum(abs(value)-.00049,0.);penalty+=float(excess@excess);nonreg=max(nonreg,float(abs(value).max())-.0005)
                    if linear:
                        P,_=r.jac(body,local);P=P if mode==1 else P[2:];add(P[:,self.ids],-.00049-value,.00049-value,j)
            if self.palms is not None:
                C,h,excess=self.palms.evaluate(x,j);penalty+=excess**2
                nonreg=max(nonreg,excess-np.deg2rad(.001))
                if linear:add(C,-np.inf,h,j)
        if linear:
            self.elastic_rows=[]
            # Keep every near collision witness from the eight audit samples.
            # One representative per pair can trade penetration between
            # fractions, stalling or failing the independent dense guard.
            # Floor vertices keep their per-body relinearized witness policy.
            for key,(gap,index,local) in worst_floor.items():
                body=key[1];j,u,rotation,base,*_=self.samples[index];x=pose(j,u,base);r.set(x,rotation)
                P,_=r.jac(body,local);gradient=P[2,self.ids];add(gradient,-gap,np.inf,j,u)
                if self.coupled_support:
                    height=r.point(body,local)[2]
                    if height<-.001 and not (j in self.fixed and (not u or j+1 in self.fixed)):
                        floor_jac.append(len(lower)-1)
                        floor_residual.append(float(height+.001))
            for (_,k,_),(gap,index,distance) in worst_pair.items():
                j,u,rotation,base,*_=self.samples[index];x=pose(j,u,base)
                model=self.collision;model.update(x,rotation);a,b=map(int,model.pairs[k]);value=mujoco.mj_geomDistance(model.m,model.configuration.data,a,b,model.cfg['detection_m'],self._fromto)
                derivative=compute_contact_normal_jacobian(model.m,model.configuration.data,a,b,self._fromto,self._normal,self._jac1,self._jac2)
                G=(-1. if value>=0. else 1.)*derivative[r.cols[self.ids]]
                if self.initial_leg_repair:
                    _,_,_,_,_,target,_,baseline=self.samples[index]
                    # An owned row has a positive restoration target. Only a
                    # pair already penetrating at this sample can take slack.
                    if target[k]>0. and baseline[k]<-model.cfg['penetration_tolerance_m']:
                        self.elastic_rows.append(len(lower))
                if self.coupled_leg_clearance:
                    baseline=self.samples[index][-1]
                    movable=(0<=j<self.n and j not in self.fixed) or (u and 0<=j+1<self.n and j+1 not in self.fixed)
                    if movable and self.leg_pairs[k] and baseline[k] < -model.cfg['penetration_tolerance_m'] and value < .0001:
                        collision_jac.append(len(lower))
                        collision_residual.append(.0001-value)
                add(G,-np.inf,max(gap,-.008) if self.restore else gap,j,u)
            matrix=sparse.csc_matrix((values,(rows,columns)),shape=(len(lower),len(z)))
            if self.coupled_support:
                self.floor_objective=(matrix[floor_jac],np.asarray(floor_residual))
                self.swept_constraint_rows=len(worst_floor)
            if self.coupled_leg_clearance:
                self.collision_objective=(matrix[collision_jac],np.asarray(collision_residual))
                self.body_objective=(matrix[body_jac],np.asarray(body_residual))
            return matrix,np.asarray(lower),np.asarray(upper)
        self.last=dict(nonregression_excess=max(nonreg,0.),minimum_floor_m=floor_min,maximum_contact_m=contact_max,minimum_collision_m=collision_min,
            interior_floor_m=interior_floor,interior_collision_m=interior_collision,
            safety_goal_met=(interior_floor>=-.00098 and interior_collision>=-1e-5 and contact_max<=.00098 and nonreg<=2e-6))
        if self.initial_leg_repair or self.coupled_support:
            self.last.update(floor_regression_m=max(0.,floor_reg),contact_regression_m=max(0.,contact_reg))
        if self.coupled_support:
            self.last.update(swept_constraint_rows=getattr(self,'swept_constraint_rows',0),
                             swept_floor_deficit_m2=floor_cost)
        if self.coupled_leg_clearance:
            self.last.update(leg_collision_deficit_m2=collision_cost,body_support_height_deficit_m2=body_cost)
        return (1e8*penalty + (2500.*floor_cost if self.coupled_support else 0.)
                + 40000.*collision_cost + 2500.*body_cost)
def project_root_height(robot, targets, rows, start, end, *, protected_frames=None):
    """Solve the vertical translation subproblem without changing any hinge.

    A shared upward translation cannot introduce self-collision, change limb
    directions, or add joint jitter. Each planted foot retains the existing
    1 mm envelope (or its already-failing measured error). Other support points
    keep the original 0.5 mm preservation corridor. Common adoption still checks
    every dense seam and all final gates.
    """
    from scipy import sparse
    from osqp import OSQP
    n=end-start; upper=np.zeros(n); goal=np.zeros(n)
    for j,i in enumerate(range(start,end)):
        robot.set(rows[i],targets['root_r'][i])
        floor=min(float((shape['hull']@robot.rot(body)[2]+robot.point(body)[2]).min())
                  for body,shape in robot.surface_models.items())
        goal[j]=max(0.,.0001-floor)
        cap=goal[j]
        for side,g in robot.sides.items():
            values=targets['sides'][side]
            if values['contact'][i]<=0:continue
            point=robot.point(g['foot'],values['pivots'][i]);xy=point[:2]-values['anchors'][i,:2]
            height=float((g['hull']@robot.rot(g['foot'])[2]+robot.point(g['foot'])[2]).min())
            original=np.sqrt(float(xy@xy)+height*height)
            radius=np.sqrt(max(max(.00099,original)**2-float(xy@xy),0.))
            cap=min(cap,max(0.,radius-height))
        from .support_quality import preserve_contact
        if any(preserve_contact(contact,i) for contact in targets.get('body_contacts',[])):
            cap=min(cap,.0005)
        upper[j]=cap
    if start:upper[:2]=0.
    if end<len(rows):upper[-2:]=0.
    if protected_frames is not None:
        upper[protected_frames]=0.
    if upper.max()<2e-6:return None
    I=sparse.eye(n,format='csc')
    D=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc')
    problem=OSQP();problem.setup(P=sparse.triu(I+2.*D.T@D,format='csc'),q=-np.minimum(goal,upper),
        A=I,l=np.zeros(n),u=upper,verbose=False,eps_abs=1e-9,eps_rel=1e-9,max_iter=10000,polishing=True)
    result=problem.solve()
    if result.info.status_val not in (1,2):return None
    delta=np.clip(result.x,0.,upper)
    if delta.max()<2e-6:return None
    candidate=rows.copy();candidate[start:end,2]+=delta
    return candidate,dict(vertical_support_subproblem=True,maximum_root_lift_m=float(delta.max()),
                           changed_frames=int(np.count_nonzero(delta>1e-6)))

