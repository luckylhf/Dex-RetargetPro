"""Reconstruct linked foot contacts before asking the export guard to adopt them."""
import time

import mujoco
import numpy as np
from mink.limits.collision_avoidance_limit import compute_contact_normal_jacobian
from scipy import sparse
from scipy.interpolate import PchipInterpolator
from scipy.linalg import lstsq
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .airborne import protected_height
from .adaptive_refine import MinimumChangeObjective
from .local_refine import _qp_with_geometry_slack,fit_window
from .protected_collision import _Rows, _difference
from .refine_routing import submodel
from .support_quality import same_support_step
from .unified_repair import dense_geometry
from .v8.support_acceptance import SUPPORT_ACCEPTANCE_M
from .v8.trajectory import _qp

RECIPE = 'cross_foot_contact_reconstruction'


def jobs(model, targets, samples):
    """Group cross-foot hits across a foot exchange and include both joins."""
    feet = {group['foot'] for group in model.r.sides.values()}
    pairs = {i for i, (a, b) in enumerate(model.pairs)
             if {model.m.geom_bodyid[a], model.m.geom_bodyid[b]} == feet}
    if not pairs:
        return []
    hits = np.array([i//8 for i, (_, _, owned) in enumerate(samples) if pairs.intersection(owned)], int)
    if not len(hits):
        return []
    fps = targets['fps']
    n = len(targets['root_r'])
    double_support = np.logical_and.reduce([foot['contact'] > 0 for foot in targets['sides'].values()])
    scene_support = np.broadcast_to(targets.get('source_scene_support',False),(n,)).astype(bool)
    for foot in targets['sides'].values():
        if 'platform_index' in foot:
            scene_support |= foot['platform_index'] >= 0
    # Keep related native and swept hits within a single candidate, without
    # joining motion across separate step transitions.
    def separate_phases(left,right):
        gap = right-left
        if gap<=round(.25*fps):
            return False
        if gap>round(.5*fps) or not double_support[left:right+1].all():
            return True
        return any(np.any(foot['anchors'][left:right+1]!=foot['anchors'][left])
                   for foot in targets['sides'].values())

    groups = np.split(hits,[i for i,(a,b) in enumerate(zip(hits[:-1],hits[1:]),1)
                            if separate_phases(a,b)])
    padding = max(3, round(.42 * fps))
    result = []
    for hit in groups:
        core_start, core_end = int(hit[0]), min(n, int(hit[-1])+2)
        a, b = max(0, core_start-padding), min(n, core_end+padding)
        if not double_support[core_start:core_end].any() or b-a > round(2*fps) or b-a < 5:
            continue
        if scene_support[a:b].any():
            continue
        result.append(dict(module='leg_collision', side='cross_foot', start=a, end=b,
                           core_start=core_start, core_end=core_end,
                           severity=len(hit), cross_foot_window=[a,b],
                           cross_foot_pairs=sorted(pairs), candidate_order=[RECIPE]))
    return result


class ContactWindow:
    def __init__(self, robot, targets, rows, model, job, deadline):
        self.robot, self.targets, self.rows, self.model = robot, targets, rows, model
        self.a, self.b = job['cross_foot_window']
        self.deadline = deadline
        self.n = self.b-self.a
        self.legs = np.array([i for g in robot.sides.values() for i in g['leg']])
        self.ids = np.r_[2,self.legs]
        self.fixed = protected_height(targets,len(rows))[self.a:self.b].copy()
        self.fixed[[0,1,-2,-1]] = True
        self.pairs = submodel(model,job['cross_foot_pairs'])
        rotates = targets['root_r'][self.a:self.b]
        self.rotations = rotates
        self.deltas = Rotation.from_matrix(rotates[:-1].transpose(0,2,1) @ rotates[1:]).as_rotvec()
        self.normal = np.empty(3)
        self.ja = np.empty((3,model.m.nv))
        self.jb = np.empty_like(self.ja)
        self.witness = np.empty(6)

    def tick(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('cross_foot_reconstruction_deadline')

    def samples(self, rows):
        for j, rotation in enumerate(self.rotations):
            self.tick()
            f = self.a+j
            for part in range(8 if j+1<self.n else 1):
                u = part/8
                rot = rotation if not part else rotation @ Rotation.from_rotvec(u*self.deltas[j]).as_matrix()
                pose = rows[f] if not part else (1-u)*rows[f]+u*rows[f+1]
                self.pairs.update(pose,rot)
                yield j,u,self.pairs.distances()

    def clearance(self, rows):
        return min(float(d.min()) for _,_,d in self.samples(rows))

    def score(self, rows):
        count = 0
        worst = np.inf
        tolerance = self.pairs.cfg['penetration_tolerance_m']
        for _,_,dist in self.samples(rows):
            minimum = float(dist.min())
            worst = min(worst,minimum)
            count += minimum < -tolerance
        return count,-worst

    def point(self, row, frame, side, jacobian=False):
        robot = self.robot
        robot.set(row,self.targets['root_r'][frame])
        group = robot.sides[side]
        body = group['foot']
        pivot = self.targets['sides'][side]['pivots'][frame]
        low = robot.lowest(side)
        value = np.r_[robot.point(body,pivot)[:2],robot.point(body,low)[2]]
        if not jacobian:
            return value
        ids = group['leg']
        jac = np.vstack((robot.jac(body,pivot)[0][:2,ids],robot.jac(body,low)[0][2:,ids]))
        return value,jac

    def active(self, frame):
        return (side for side,foot in self.targets['sides'].items() if foot['contact'][frame]>0)

    def anchor(self, frame, side):
        return np.r_[self.targets['sides'][side]['anchors'][frame,:2],0.]

    def separation_seed(self, margin, anchored=False):
        """Use existing swept clearance rows and the elastic QP as a guide."""
        ids = self.legs
        width = len(ids)
        original = self.rows[self.a:self.b,ids].copy()
        poses = self.rows.copy()
        identity = sparse.eye(self.n*width,format='csc')
        first,second = _difference(self.n,width)
        base = sparse.vstack([identity,first],format='csc')
        H = (identity+3*first.T@first+70*second.T@second).tocsc()
        limit = np.maximum(self.robot.speed_limits[ids-3]/self.targets['fps'],
                           abs(np.diff(original,axis=0)))
        anchors = {}
        if anchored:
            for frame in range(self.a,self.b):
                if self.fixed[frame-self.a]:
                    continue
                for side in self.active(frame):
                    position = self.point(self.rows[frame],frame,side)
                    offset = position-self.anchor(frame,side)
                    error = np.linalg.norm(offset)
                    if error>SUPPORT_ACCEPTANCE_M:
                        position -= offset*min(.002,.000012/error)
                    anchors[frame,side] = position
        old_score = self.score(poses)
        for _ in range(12 if anchored else 9):
            constraints = _Rows(self.n,width)
            for j,u,dist in self.samples(poses):
                if self.fixed[j] and (not u or self.fixed[j+1]):
                    continue
                data = self.pairs.configuration.data
                for index in np.flatnonzero(dist<margin):
                    ga,gb = self.pairs.pairs[index]
                    d = mujoco.mj_geomDistance(self.model.m,data,int(ga),int(gb),
                                                self.pairs.cfg['detection_m'],self.witness)
                    jac = compute_contact_normal_jacobian(self.model.m,data,int(ga),int(gb),self.witness,
                                                          self.normal,self.ja,self.jb)[self.robot.cols[ids]]
                    constraints.add(jac if d>=0 else -jac,margin-d,np.inf,j,u)
            if not constraints.low or (old_score[0]==0 and -old_score[1]>=.00005):
                break
            q = poses[self.a:self.b,ids]
            lo = np.maximum(-.04,self.robot.lower[ids]-q)
            hi = np.minimum(.04,self.robot.upper[ids]-q)
            lo[self.fixed] = hi[self.fixed] = 0.
            steps = np.diff(q,axis=0)
            low,high = np.r_[lo.ravel(),(-limit-steps).ravel()],np.r_[hi.ravel(),(limit-steps).ravel()]
            A = base
            if anchors:
                planted = _Rows(self.n,width)
                for (frame,side),expected in anchors.items():
                    actual,jac = self.point(poses[frame],frame,side,jacobian=True)
                    ids_for_side = self.robot.sides[side]['leg']
                    columns = np.isin(ids,ids_for_side)
                    for derivative,residual in zip(jac,expected-actual):
                        row = np.zeros(width)
                        row[columns] = derivative
                        planted.add(row,residual-1e-5,residual+1e-5,frame-self.a)
                A = sparse.vstack([base,planted.matrix()],format='csc')
                low,high = np.r_[low,planted.low],np.r_[high,planted.high]
            delta,_ = _qp_with_geometry_slack(H,H@(q-original).ravel(),A,low,high,
                constraints.matrix(),np.asarray(constraints.low),np.asarray(constraints.high),
                time_limit_s=max(.001,min(3.,self.deadline-time.monotonic())))
            for alpha in (1.,.5,.25,.125):
                trial = poses.copy()
                trial[self.a:self.b,ids] += alpha*delta.reshape(self.n,width)
                trial[self.a:self.b][self.fixed] = self.rows[self.a:self.b][self.fixed]
                score = self.score(trial)
                if score<old_score:
                    poses,old_score = trial,score
                    break
            else:
                break
        return poses

    def solve_frame(self, frame, guides, previous, margin):
        origin = self.rows[frame]
        ids = self.ids
        scale = np.r_[.1,np.ones(len(self.legs))]
        lo = np.maximum((self.robot.lower[ids]-origin[ids])/scale,-.55)
        hi = np.minimum((self.robot.upper[ids]-origin[ids])/scale,.55)
        lo[0],hi[0] = max(lo[0],-.25),min(hi[0],.25)
        for group in self.robot.sides.values():
            knee = group['leg'][3]
            if origin[knee]>=0:
                col = np.flatnonzero(ids==knee).item()
                lo[col] = max(lo[col],-origin[knee])
        active = []
        for side in self.active(frame):
            target = self.anchor(frame,side)
            point = self.point(origin,frame,side)
            error = np.linalg.norm(point-target)
            active.append((side,target,max(SUPPORT_ACCEPTANCE_M,error)-2.5e-6,
                           min(-SUPPORT_ACCEPTANCE_M,point[2])-1e-6))

        def pose(d):
            row = origin.copy()
            row[ids] += d*scale
            return row

        def feasible(d):
            self.tick()
            row = pose(d)
            self.pairs.update(row,self.targets['root_r'][frame])
            bounds = list(self.pairs.distances()-margin)
            for side,target,limit,floor in active:
                point = self.point(row,frame,side)
                bounds.extend((limit-np.linalg.norm(point-target),point[2]-floor))
            return 1000*np.asarray(bounds)

        ref = np.zeros(len(ids)) if previous is None else (previous[ids]-origin[ids])/scale

        def objective(d):
            return float(d@d+.35*((d-ref)@(d-ref)))

        best,cost = None,float('inf')
        for guide in (*guides,origin if previous is None else previous):
            initial = np.clip((guide[ids]-origin[ids])/scale,lo,hi)
            fitted = minimize(objective,initial,method='SLSQP',bounds=list(zip(lo,hi)),
                              constraints=dict(type='ineq',fun=feasible),
                              options=dict(maxiter=50,ftol=2e-10))
            if feasible(fitted.x).min()>=-.0005 and objective(fitted.x)<cost:
                best,cost = pose(fitted.x),objective(fitted.x)
                if fitted.success:
                    break
        return best

    def polish(self, seed, margin):
        """Restore the measured stance envelope without moving the fixed joins."""
        rows = seed.copy()
        points = {(f,s):self.point(self.rows[f],f,s) for f in range(self.a,self.b) for s in self.active(f)}

        def nudge(f,side,desired):
            value,jac = self.point(rows[f],f,side,jacobian=True)
            delta = lstsq(jac,desired-value,lapack_driver='gelsy')[0]
            rows[f,self.robot.sides[side]['leg']] += np.clip(delta,-.002,.002)

        for sweep in range(6):
            forward = sweep%2==0
            span = range(self.a+2,self.b-2) if forward else range(self.b-3,self.a+1,-1)
            for f in span:
                self.tick()
                if self.fixed[f-self.a]:
                    continue
                for side in self.active(f):
                    target = self.anchor(f,side)
                    point = self.point(rows[f],f,side)
                    relative = point-target
                    norm = np.linalg.norm(relative)
                    bound = max(SUPPORT_ACCEPTANCE_M,np.linalg.norm(points[f,side]-target))-margin
                    if norm>bound and np.any(rows[f]!=self.rows[f]):
                        nudge(f,side,target+relative*(bound/norm))
                    other = f-1 if forward else f+1
                    if (other,side) not in points or not same_support_step(
                            self.targets['sides'][side],max(f,other)):
                        continue
                    point = self.point(rows[f],f,side)
                    previous = self.point(rows[other],other,side)
                    delta = point[:2]-previous[:2]
                    norm = np.linalg.norm(delta)
                    cap = max(SUPPORT_ACCEPTANCE_M,np.linalg.norm(points[f,side][:2]-points[other,side][:2]))-margin
                    if norm>cap:
                        nudge(f,side,np.r_[previous[:2]+delta*cap/norm,point[2]])
        return rows,points

    def lift(self, rows, points, old_floor, rotations, start, end):
        _,floor = dense_geometry(self.robot,self.model,{'root_r':rotations},rows[start:end])
        required = np.max(np.minimum(-SUPPORT_ACCEPTANCE_M,old_floor)-floor,axis=1)
        samples = np.flatnonzero(required>5e-7)
        if not len(samples):
            return rows,0.
        cap = np.full(self.n,4.)
        cap[self.fixed] = 0.
        for f in range(self.a,self.b):
            limits = [np.linalg.norm(points[f,s]-self.anchor(f,s)) for s in self.active(f)]
            bound = max(SUPPORT_ACCEPTANCE_M,*limits) if limits else SUPPORT_ACCEPTANCE_M
            for side in self.active(f):
                p = self.point(rows[f],f,side)-self.anchor(f,side)
                dz = np.sqrt(max(0.,bound**2-p[:2]@p[:2]))-p[2]
                cap[f-self.a] = min(cap[f-self.a],max(0.,1000*dz))
        columns,values,indices = [],[],[]
        for i,sample in enumerate(samples):
            f,part = divmod(int(sample),8)
            k = start+f-self.a
            for j,w in ((k,1.-part/8),(k+1,part/8)):
                if 0<=j<self.n and w:
                    columns.append(j)
                    values.append(w)
                    indices.append(i)
        I = sparse.eye(self.n,format='csc')
        _,second = _difference(self.n,1)
        G = sparse.csc_matrix((values,(indices,columns)),shape=(len(samples),self.n))
        limit = 1000*(required[samples]+.5e-6)
        self.tick()
        delta,_ = _qp((I+12*second.T@second).tocsc(),np.zeros(self.n),sparse.vstack([I,G],format='csc'),
                     np.r_[np.zeros(self.n),limit],np.r_[cap,np.full(len(limit),np.inf)],np.zeros(self.n),
                     scaling=5,time_limit_s=max(.001,min(5.,self.deadline-time.monotonic())))
        result = rows.copy()
        result[self.a:self.b,2] += np.clip(delta,0.,cap)/1000.
        return result,float(delta.max())

    def run(self, weights):
        bad = {j for j,_,dist in self.samples(self.rows) if np.any(dist<-self.pairs.cfg['penetration_tolerance_m'])}
        active = sorted({self.a+f for j in bad for f in range(max(2,j-1),min(self.n-2,j+3))
                         if not self.fixed[f]})
        if not active:
            raise ValueError('cross_foot_contacts_are_protected')
        if self.clearance(self.rows)>-.001:
            try:
                seed = self.separation_seed(.001,anchored=True)
                cfg = dict(weights,_pelvis_height_only=True,_coupled_swept_support=True,
                           _coupled_leg_clearance=True,_preserve_support_step=True,_phase_elastic_search=True,
                           _initial_leg_collision_cfg=self.model.cfg,_repair_deadline=self.deadline-2.5)
                fitted,_ = fit_window(self.robot,self.targets,self.rows,cfg,self.a,self.b,15,
                    ids_override=self.ids,objective_factory=lambda f:MinimumChangeObjective(seed[f]),
                    local_checks=True,allow_support=True,seed=seed,protected_frames=self.fixed)
                if self.clearance(fitted)>=0.:
                    return fitted,dict(policy=RECIPE,window=[self.a,self.b],seed='anchored_clearance')
            except (ValueError,RuntimeError):
                self.tick()
        guides = [self.separation_seed(margin) for margin in (.0003,.001)]
        start,end = max(0,self.a-1),min(len(self.rows),self.b+1)
        rotations = self.targets['root_r'][start:end]
        _,old_floor = dense_geometry(self.robot,self.model,{'root_r':rotations},self.rows[start:end])
        for margin,order in ((.001,active),(.00035,active),(.00035,active[::-1])):
            samples = {}
            previous = None
            for f in order:
                self.tick()
                pose = self.solve_frame(f,[guide[f] for guide in guides],previous,margin)
                if pose is not None:
                    samples[f] = pose
                    previous = pose
            if not samples:
                continue
            grid = np.arange(self.a,self.b)
            cut = sorted({self.a,self.a+1,self.b-2,self.b-1,*samples})
            delta = np.asarray([(samples[f]-self.rows[f])[self.ids] if f in samples
                                else np.zeros(len(self.ids)) for f in cut])
            seed = self.rows.copy()
            seed[self.a:self.b,self.ids] += PchipInterpolator(cut,delta,axis=0)(grid)
            seed[self.a:self.b][self.fixed] = self.rows[self.a:self.b][self.fixed]
            for inset in (4e-6,4e-5,1e-4,2.2e-4):
                polished,points = self.polish(seed,inset)
                if self.clearance(polished)<0.:
                    continue
                try:
                    candidate,lift = self.lift(polished,points,old_floor,rotations,start,end)
                except (ValueError,RuntimeError):
                    continue
                return candidate,dict(policy=RECIPE,window=[self.a,self.b],native_frames=len(samples),
                                      contact_inset_m=inset,clearance_seed_m=margin,root_lift_mm=lift)
        raise ValueError('cross_foot_contact_reconstruction_did_not_clear')
