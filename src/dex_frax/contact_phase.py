"""Event-triggered whole-body IK over a short, immutable-source contact interval."""
import time
import numpy as np
from .airborne import protected_height
from scipy import sparse
from scipy.optimize import least_squares

from .support_quality import preserve_contact
from .temporal_constraints import assess, diagnostics, regressions, root_speed_bounds
from .unified_repair import dense_geometry
from .v8.self_collision import CollisionModel

POLICY = 'contact_surface_window_v1'


def transition_windows(rows, targets):
    fps = targets['fps']
    speed = np.linalg.norm(np.diff(rows[:, :3], axis=0), axis=1) * fps
    switch = np.zeros(len(rows), bool)
    for foot in targets['sides'].values():
        phase = foot['phase']
        switch[1:] |= phase[1:] != phase[:-1]
    margin = max(2, round(.08 * fps))
    radius = max(5, round(.20 * fps))
    target = targets.get('root_p', np.zeros_like(rows[:, :3]))
    excess = speed - root_speed_bounds(target, fps)
    peaks = sorted(np.flatnonzero(excess > 0), key=lambda i: -excess[i])
    windows = []
    for edge in peaks:
        frame = int(edge + 1)
        if not switch[max(0,frame-margin):frame+margin+1].any():
            continue
        if any(start+2 <= frame < stop-2 for start,stop,_ in windows):
            continue
        start,stop = max(0,frame-radius), min(len(rows),frame+radius+1)
        if stop-start >= 6:
            windows.append((start,stop,frame))
    return windows


def surface_patch(foot, phase):
    """Use the sole footprint, including the lateral edges of either end."""
    hull = foot['hull']
    length = foot['toe'][0] - foot['heel'][0]
    if phase == 2:
        return hull[hull[:,0] >= foot['toe'][0] - .28*length]
    if phase == 3:
        return hull[hull[:,0] <= foot['heel'][0] + .28*length]
    return hull


def lowest(robot, body, hull):
    local = hull[np.argmin(hull @ robot.rot(body)[2])]
    return local, robot.point(body,local)


class PhaseFit:
    def __init__(self, robot, targets, old, start, stop, use_patch, use_time, pose_side=None, deadline=None, hold_arms=False, restore_support=False):
        self.r, self.t, self.old = robot, targets, old[start:stop].copy()
        self.start, self.stop = start, stop
        self.use_patch, self.use_time = use_patch, use_time
        self.pose_side, self.deadline = pose_side, deadline
        self.restore_support = restore_support
        n = len(self.old)
        self.fps = targets['fps']
        root_ids = {2} if restore_support else {0,1,2}
        self.ids = sorted(root_ids.union(*(g['leg'] for g in robot.sides.values())))
        for side,g in robot.sides.items():
            region = {g['shoulder'],g['forearm'],g['tcp']}
            if (not hold_arms) and (side == pose_side or (pose_side is None and any(c['body'] in region and any(preserve_contact(c,i) for i in range(start,stop))
                   for c in targets.get('body_contacts',())))):
                self.ids.extend(g['arm'])
        self.ids = np.asarray(self.ids)
        self.width = len(self.ids)
        fixed = protected_height(targets,len(targets['root_p']))[start:stop].copy()
        fixed[[0,1,-2,-1]] = True
        self.free = np.broadcast_to(~fixed[:,None],(n,self.width)).ravel()
        self.feet, self.supports, self.floors = [], [], []
        for j,row in enumerate(self.old):
            i = start+j
            robot.set(row,targets['root_r'][i])
            feet, supports = [], []
            for name,g in robot.sides.items():
                f = targets['sides'][name]
                if f['contact'][i] <= 0: continue
                pivot,anchor = f['pivots'][i],f['anchors'][i]
                low = lowest(robot,g['foot'],g['hull'])[1][2]
                previous_error = np.linalg.norm(np.r_[robot.point(g['foot'],pivot)[:2]-anchor[:2],low])
                feet.append((g,pivot,anchor,surface_patch(g,int(f['phase'][i])),max(.001,previous_error)))
            for c in targets.get('body_contacts',()):
                if preserve_contact(c,i):
                    hull = np.asarray(c['robot_hull'])
                    _,p = lowest(robot,c['body'],hull)
                    normal = targets['body_contact_height_tolerance_m']*.985
                    if not restore_support: normal = max(normal,abs(p[2])-.0002)
                    anchor = None
                    if c['mode'][i] == 1:
                        target = c['anchors'][i,:2]
                        allowance = max(targets.get('body_contact_anchor_tolerance_m',targets['body_contact_height_tolerance_m'])*.985,np.linalg.norm(p[:2]-target)-.0002)
                        anchor = (target,allowance)
                        if restore_support: anchor = (target,targets.get('body_contact_anchor_tolerance_m',normal)*.985)
                    supports.append((c['body'],hull,normal,anchor))
            floors = []
            for body,shape in robot.surface_models.items():
                low = lowest(robot,body,shape['hull'])[1][2]
                if low < .07:
                    floors.append((body,shape['hull'],min(low,-.001) if restore_support else -.0006))
            self.feet.append(feet)
            self.supports.append(supports)
            self.floors.append(floors)
        self.palms = None
        if restore_support:
            from .palm_constraints import SourcePalms
            sides = [side for side,g in robot.sides.items()
                     if np.intersect1d(self.ids,g['arm']).size
                     and all(key in targets['sides'][side] for key in ('hand_forward','hand_width'))]
            if sides:
                self.palms = SourcePalms(robot,targets,old,start,stop,sides,self.ids)
        eye = sparse.eye(n,format='csr')
        D = eye[1:] - eye[:-1]
        self.D = sparse.kron(D,sparse.eye(self.width),format='csr')
        self.D2 = sparse.kron(D[1:]-D[:-1],sparse.eye(self.width),format='csr')
        self.axes = np.eye(self.width)
        self.last = None

    def evaluate(self, values):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError('contact_window_search_budget')
        if self.last is not None and np.array_equal(self.last,values):
            return self.cached
        delta = np.zeros(self.free.size)
        delta[self.free] = values
        arr = self.old.copy()
        arr[:,self.ids] += delta.reshape(-1,self.width)
        blocks, vals = [], []
        for j,row in enumerate(arr):
            i = self.start+j
            self.r.set(row,self.t['root_r'][i])
            J, v = [], []
            def add(res, jac, weight):
                v.extend(np.atleast_1d(res)*weight)
                J.extend(np.atleast_2d(jac)*weight)
            def point(body,local):
                return self.r.jac(body,local)[0][:,self.ids]
            def outside(value, jac, bound, weight):
                size = np.linalg.norm(value)
                if size > bound:
                    add(size-bound,value @ jac/size,weight)
                else:
                    add(0.,np.zeros(self.width),weight)
            for g,pivot,anchor,patch,allowance in self.feet[j]:
                body = g['foot']
                lo,p = lowest(self.r,body,g['hull'])
                err = np.r_[self.r.point(body,pivot)[:2]-anchor[:2],p[2]]
                jac = np.vstack([point(body,pivot)[:2],point(body,lo)[2]])
                add(err,jac,850.)
                if self.restore_support:
                    outside(err,jac,allowance,6000.)
                if self.use_patch:
                    mat,position = lowest(self.r,body,patch)
                    add(position[2],point(body,mat)[2],90.)
            for side,g in self.r.sides.items():
                if self.pose_side != side: continue
                source = self.t['sides'][side]
                for base,tip,signal in (('shoulder','elbow','upper'),('elbow','wrist','lower')):
                    diff = self.r.point(g[tip])-self.r.point(g[base])
                    length = np.linalg.norm(diff)
                    direction = diff/length
                    jac = ((np.eye(3)-np.outer(direction,direction))/length) @ (point(g[tip],None)-point(g[base],None))
                    add(direction-source[signal][i],jac,12. if self.pose_side==side else 1.5)
            for body,hull,normal,anchor in self.supports[j]:
                mat,p = lowest(self.r,body,hull)
                jac = point(body,mat)
                outside(p[2:3],jac[2:3],normal,530.)
                if anchor is not None:
                    outside(p[:2]-anchor[0],jac[:2],anchor[1],530.)
            for body,hull,limit in self.floors[j]:
                mat,p = lowest(self.r,body,hull)
                active = p[2] < limit
                add(min(0.,p[2]-limit),point(body,mat)[2] if active else np.zeros(self.width),
                    6000. if self.restore_support else 100.)
            if self.palms is not None:
                jac,margin,_ = self.palms.evaluate(row,j)
                add(np.maximum(-margin,0.),jac*(margin<0.)[:,None],3000.)
            blocks.append(sparse.csr_matrix(np.asarray(J).reshape(-1,self.width)))
            vals.extend(v)
        matrix = [sparse.block_diag(blocks,format='csr')]
        weight = np.where(self.ids<3,[40 if k<2 else 4 for k in self.ids],1.8)
        n = len(arr)
        vals.extend(delta*np.tile(weight,n))
        matrix.append(sparse.diags(np.tile(weight,n)))
        if self.use_time:
            v = arr[:,self.ids]
            acc = np.diff(v,n=2,axis=0)
            acc_weight = np.where(self.ids<3,42.,3.)
            vals.extend((acc*acc_weight).ravel())
            matrix.append(self.D2.multiply(np.tile(acc_weight,n-2)[:,None]))
            step = np.diff(v,axis=0)
            limit = np.r_[np.full(3,np.inf),self.r.speed_limits/ self.fps][self.ids]
            excess = step-np.clip(step,-limit,limit)
            vals.extend((excess*100.).ravel())
            matrix.append(self.D.multiply((np.abs(excess)>0).ravel()[:,None]*100.))
            # Compare the Euclidean step against the per-interval root bound.
            root_columns = self.ids<3
            linear = step[:,root_columns]
            norm = np.linalg.norm(linear,axis=1)
            limit = root_speed_bounds(self.t['root_p'][self.start:self.stop], self.fps)/self.fps
            vals.extend(230.*np.maximum(0.,norm-limit))
            coef = np.zeros_like(step)
            active = norm>limit
            coef[np.ix_(active,root_columns)] = 230.*linear[active]/norm[active,None]
            speed_J = sparse.block_diag([c[None,:] for c in coef],format='csr') @ self.D
            matrix.append(speed_J)
        self.cached = np.asarray(vals),sparse.vstack(matrix,format='csr')[:,self.free]
        self.last = values.copy()
        return self.cached

    def solve(self,seed=None):
        if not self.free.any(): return self.old.copy()
        ids = self.ids
        cap = np.where(ids<2,.065,np.where(ids==2,.38,.58))
        lo = np.maximum(-cap,self.r.lower[ids]-self.old[:,ids])
        hi = np.minimum(cap,self.r.upper[ids]-self.old[:,ids])
        lo = np.minimum(lo,-1e-10).ravel()[self.free]
        hi = np.maximum(hi,1e-10).ravel()[self.free]
        x0 = np.zeros_like(lo) if seed is None else np.clip((seed[:,ids]-self.old[:,ids]).ravel()[self.free],lo,hi)
        fit = least_squares(lambda v:self.evaluate(v)[0], x0,
                            jac=lambda v:self.evaluate(v)[1], bounds=(lo,hi), max_nfev=30,
                            ftol=2e-5,xtol=2e-6,gtol=1e-5)
        delta = np.zeros(self.free.size)
        delta[self.free] = fit.x
        new = self.old.copy()
        new[:,ids] += delta.reshape(-1,len(ids))
        return new


def support_errors(robot, t, rows, start):
    values,limits = [],[]
    for j,row in enumerate(rows):
        i = start+j
        robot.set(row,t['root_r'][i])
        for side,g in robot.sides.items():
            v = t['sides'][side]
            if v['contact'][i]>0:
                p = robot.point(g['foot'],v['pivots'][i])
                low = lowest(robot,g['foot'],g['hull'])[1][2]
                values.append(np.linalg.norm(np.r_[p[:2]-v['anchors'][i,:2],low]))
                limits.append(.001)
        for c in t.get('body_contacts',()):
            if not preserve_contact(c,i): continue
            p = lowest(robot,c['body'],np.asarray(c['robot_hull']))[1]
            values.append(abs(p[2]));limits.append(t['body_contact_height_tolerance_m'])
            if c['mode'][i]==1:
                values.append(np.linalg.norm(p[:2]-c['anchors'][i,:2]))
                limits.append(t.get('body_contact_anchor_tolerance_m',t['body_contact_height_tolerance_m']))
    return np.array(values),np.array(limits)


def apply(robot,t,rows,collision_cfg=None,*,raw=None,use_patch=True,use_time=True):
    result = rows.copy()
    report = dict(policy=POLICY,events=[],kept=0)
    jobs = transition_windows(rows,t)
    if not jobs: return result,report
    model = CollisionModel(robot,dict(collision_cfg or {},coverage='full'))
    dg = diagnostics(t)
    for a,b,frame in jobs:
        original = result[a:b].copy()
        rotation = t['root_r'][a:b]
        original_f32 = original.astype(np.float32).astype(float)
        local = dict(root_r=rotation)
        fps = t['fps']
        l,h = max(0,a-20),min(len(result),b+20)
        signals = {k:v[l:h] for k,v in dg.items()}
        before_j = assess(result[l:h,3:],fps,robot.names,signals)
        d0,f0 = dense_geometry(robot,model,local,original_f32)
        speed = lambda x: np.linalg.norm(np.diff(x[:,:3],axis=0),axis=1)*fps
        v0 = speed(original_f32)
        fit = PhaseFit(robot,t,result,a,b,use_patch,use_time)
        candidate = fit.solve()
        event = dict(frame=frame,window=[a,b],peak_before_m_s=float(v0.max()),rejections=[])
        def added_jitter(window):
            segment = result[l:h,3:].copy()
            segment[a-l:b-l] = window[:,3:]
            added = regressions(before_j,assess(segment,fps,robot.names,signals))
            return sorted({l+int(i) for indices in added.values() for i in indices})
        from .phase_path import select
        from .phase_height import project
        translation = project(robot,t,original_f32,a,f0,candidate,use_patch)
        alternatives=[candidate,translation]
        if raw is not None:
            alternatives.append(fit.solve(raw[a:b]))
        if any(axis in fit.ids for g in robot.sides.values() for axis in g['arm']):
            legs=PhaseFit(robot,t,result,a,b,use_patch,use_time,hold_arms=True)
            alternatives.append(legs.solve())
            arm_ids=sum((g['arm'] for g in robot.sides.values()),[])
            held=candidate.copy()
            held[:,arm_ids]=original[:,arm_ids]
            alternatives.append(held)
        path = select(fit,alternatives,model,d0,f0,support_errors,added_jitter)
        if path is None:
            event['rejections'].append('no_admissible_surface_path')
        else:
            v1 = speed(path)
            if v1.max() < v0.max()-.005:
                # Preserve exact double values for locked poses and unmodified axes.
                result[a:b] = np.where(path==original_f32,original,path)
                _,f1 = dense_geometry(robot,model,local,path)
                event.update(peak_after_m_s=float(v1.max()),floor_before_mm=-1000.*float(f0.min()),
                             floor_after_mm=-1000.*float(f1.min()))
                report['kept']+=1
            else:
                event['rejections'].append('no_peak_improvement')
        report['events'].append(event)
    return result,report


def supported_arm_span(robot,t,side,a,b):
    g=robot.sides[side]
    active=np.zeros(b-a,bool)
    for c in t.get('body_contacts',()):
        if c['body'] not in (g['tcp'],g['forearm'],g['shoulder']):continue
        active |= np.asarray(c['mode'][a:b])>0 if 'confidence' not in c else (
            (np.asarray(c['mode'][a:b])>0) & (np.asarray(c['confidence'][a:b])*c.get('source_geometry_confidence',0.)>=.8))
    return np.flatnonzero(active)


def arm_proposal(robot,t,out,job,deadline):
    a,b=job['start'],job['end']
    active=supported_arm_span(robot,t,job['side'],job['core_start'],job['core_end'])
    if not len(active):return out.copy(),{'reason':'no_trusted_arm_contact'}
    center=job['core_start']+int(active[len(active)//2])
    a,b=max(a,center-10),min(b,center+11)
    if b-a<6:return out.copy(),{'reason':'short_contact_window'}
    fit=PhaseFit(robot,t,out,a,b,True,True,pose_side=job['side'],deadline=deadline)
    new=out.copy()
    new[a:b]=fit.solve()
    return new,dict(scope='trusted_upper_support',side=job['side'],window=[a,b])



def body_proposal(robot, targets, rows, job, deadline, cfg=None, progress=None):
    """Re-use the coupled contact target and solve on the feasible swept grid."""
    from .adaptive_refine import MinimumChangeObjective
    from .local_refine import fit_window

    a,b = job['start'],job['end']
    if job.get('restore_support_deficit'):
        from .pelvis_contact import _window
        a,b = _window(robot,targets,rows,job)
        fit = PhaseFit(robot,targets,rows,a,b,True,True,deadline=deadline,restore_support=True)
        options = dict(cfg or {},_coupled_swept_support=True,_coupled_leg_clearance=True,
                       _preserve_support_step=True,_repair_deadline=deadline-2.5,
                       _local_fit_progress=progress)
        result, detail = fit_window(robot,targets,rows,options,a,b,iterations=5,
            ids_override=fit.ids,allow_support=True,local_checks=True,
            objective_factory=lambda i:MinimumChangeObjective(rows[i]),
            protected_frames=protected_height(targets,len(rows))[a:b])
        return result,dict(scope='trusted_body_support_with_swept_floor_and_collision',
                           window=[a,b],movable_joint_columns=fit.ids.tolist(),fit=detail)
    fit = PhaseFit(robot,targets,rows,a,b,True,True,deadline=deadline,restore_support=True)
    wanted=fit.solve()
    options=dict(cfg or {},_coupled_swept_support=True,_repair_deadline=deadline,
                 _local_fit_progress=progress)
    options.setdefault('numerical_joint_speed_deg_s',720.)
    result, detail=fit_window(robot,targets,rows,options,a,b,iterations=12,
        ids_override=fit.ids,allow_support=True,local_checks=True,
        objective_factory=lambda i:MinimumChangeObjective(wanted[i-a]))
    moving = [side for side,g in robot.sides.items() if np.intersect1d(fit.ids,g['arm']).size]
    return result,dict(scope='trusted_body_support_with_swept_floor',window=[a,b],
                       arms=moving,fit=detail,interpolation_subdivisions=8)


def has_joint_support(targets,a,b):
    """Require overlapping, multi-frame trusted support and foot contact."""
    feet = np.logical_or.reduce([v['contact'][a:b]>0 for v in targets['sides'].values()])
    trusted = np.zeros(b-a,bool)
    for contact in targets.get('body_contacts',()):
        for i in np.flatnonzero(contact['mode'][a:b]):
            trusted[i] |= preserve_contact(contact,a+int(i))
    return bool(np.any((feet & trusted)[1:] & (feet & trusted)[:-1]))
