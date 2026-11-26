"""Bounded temporal repair for existing refine trajectories."""
import time

import numpy as np
from .airborne import airborne_mask, protected_height
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

from .support_quality import preserve_contact, same_support_step
from .temporal_constraints import regressions, root_speed_bounds
from .robot_profile import ROOT_ANGULAR_LIMIT_RAD_S
from .temporal_quality import assess
from .torso_branch import floor_samples
from .v8.multicontact import contact_local
from .v8.self_collision import scan, windows
from .v8.support_acceptance import SUPPORT_ACCEPTANCE_M

JITTER = 'conversion_added_jitter'
ANGULAR = 'root_angular_speed_exceeds_profile_limit'
FLAGS = frozenset((JITTER, ANGULAR))
POSE_BAND = np.deg2rad(4.975)


def angular_speeds(rotation, fps):
    return (rotation[1:] * rotation[:-1].inv()).magnitude() * fps


def bounded_rotation(reference, current, fps, fixed, deadline):
    """Keep source-relative orientation and the two window joins bounded."""
    n = len(reference)
    seed = (current * reference.inv()).as_rotvec()
    cap = np.full((n, 3), POSE_BAND)
    lower = -cap.copy()
    cap[fixed] = lower[fixed] = seed[fixed]
    step = (ROOT_ANGULAR_LIMIT_RAD_S - .005) / fps

    def cost(values):
        rows = values.reshape(n, 3)
        return float(np.sum(rows**2) + .5*np.sum(np.diff(rows, n=2, axis=0)**2))

    def constraints(values):
        rows = values.reshape(n, 3)
        rotation = Rotation.from_rotvec(rows) * reference
        return np.r_[POSE_BAND**2 - np.sum(rows**2, axis=1),
                     step**2 - (angular_speeds(rotation, fps)/fps)**2]

    def check_time(_):
        if time.monotonic() >= deadline:
            raise TimeoutError('temporal_rotation_budget')

    fit = minimize(cost, seed.ravel(), method='SLSQP', bounds=list(zip(lower.ravel(), cap.ravel())),
                   constraints=[dict(type='ineq', fun=constraints)], callback=check_time,
                   options=dict(maxiter=120, ftol=2e-11))
    rows = fit.x.reshape(n, 3)
    return Rotation.from_rotvec(rows) * reference


def regress_at(before, after, tolerance, margin=1e-7):
    return np.any(after > np.maximum(before, tolerance) + margin)


class Repair:
    def __init__(self, robot, targets, diagnostics, rows, rotations, model, deadline,
                 *, allow_unconfirmed_arm_jitter=False):
        self.robot = robot
        self.targets = targets
        self.diagnostics = diagnostics
        self.rows = rows
        self.quaternions = rotations
        self.rotations = Rotation.from_quat(rotations).as_matrix()
        self.model = model
        self.deadline = deadline
        self.fps = targets['fps']
        self.air = protected_height(targets,len(rows))
        self.arm_only = np.zeros(len(rows),bool)
        if allow_unconfirmed_arm_jitter:
            uncertain = np.broadcast_to(targets.get('source_support_uncertain',False), (len(rows),))
            scene = np.broadcast_to(targets.get('source_scene_support',False), (len(rows),))
            inferred = np.broadcast_to(targets.get('source_inferred_platform',False), (len(rows),))
            self.arm_only = ((uncertain & ~scene) | inferred) & ~airborne_mask(targets,len(rows))
            for contact in targets.get('body_contacts',()):
                for i in np.flatnonzero(self.arm_only & (np.asarray(contact['mode']) > 0)):
                    if preserve_contact(contact,i):
                        self.arm_only[i] = False
        arm_columns = [column for group in robot.sides.values() for column in group['arm']]
        self.non_arm_columns = np.setdiff1d(np.arange(rows.shape[1]),arm_columns)
        self.events = []

    def time_left(self):
        return time.monotonic() < self.deadline

    def temporal(self, rows, a, b):
        targets = {k: v[a:b] for k, v in self.diagnostics.items() if v.ndim and len(v)==len(rows)}
        return assess(rows[a:b, 3:], self.fps, self.robot.names, targets)

    def protected_pose_changed(self, before, after, a, b):
        arm_only = self.arm_only[a:b]
        locked = self.air[a:b] & ~arm_only
        return (not np.array_equal(before[a:b][locked],after[a:b][locked]) or
                not np.array_equal(before[a:b][np.ix_(arm_only,self.non_arm_columns)],
                                   after[a:b][np.ix_(arm_only,self.non_arm_columns)]))

    def supports(self, rows, rotations, a, b):
        r = self.robot
        measured = []
        previous = {}
        for i in range(a,b):
            r.set(rows[i], rotations[i])
            for side, g in r.sides.items():
                foot = self.targets['sides'][side]
                if foot['contact'][i] > 0:
                    anchor = foot['anchors'][i]
                    pivot = foot['pivots'][i]
                    p = r.point(g['foot'],pivot) - anchor
                    measured.append((float(np.linalg.norm(p[:2])),SUPPORT_ACCEPTANCE_M))
                    if i > a and same_support_step(foot, i):
                        measured.append((float(np.linalg.norm(p[:2]-previous[side])),SUPPORT_ACCEPTANCE_M))
                    previous[side]=p[:2].copy()
                    p[2] = r.point(g['foot'],r.lowest(side))[2]
                    measured.append((float(np.linalg.norm(p)),SUPPORT_ACCEPTANCE_M))
            for contact in self.targets['body_contacts']:
                if not preserve_contact(contact,i):
                    continue
                point = r.point(contact['body'],contact_local(r,contact))
                height = self.targets['body_contact_height_tolerance_m']
                anchor = self.targets.get('body_contact_anchor_tolerance_m',height)
                measured.append((abs(point[2]),height))
                if contact['mode'][i] == 1:
                    measured.append((float(np.linalg.norm(point[:2]-contact['anchors'][i,:2])),anchor))
        return np.asarray(measured,float).reshape(-1,2)

    def check(self, rows, rotation, a, b):
        if not np.isfinite(rows[a:b]).all() or not np.isfinite(rotation[a:b]).all():
            return 'nonfinite'
        if self.protected_pose_changed(self.rows,rows,a,b):
            return 'source_airborne_changed'
        lo, hi = max(0,a-2), min(len(rows),b+2)
        knees = [g['leg'][3] for g in self.robot.sides.values()]
        if np.any((self.rows[a:b,knees]>=0) & (rows[a:b,knees]<-1e-7)):
            return 'new_knee_hyperextension'
        j0 = np.abs(np.diff(self.rows[lo:hi,3:],axis=0))*self.fps
        j1 = np.abs(np.diff(rows[lo:hi,3:],axis=0))*self.fps
        v0 = np.linalg.norm(np.diff(self.rows[lo:hi,:3],axis=0),axis=1)*self.fps
        v1 = np.linalg.norm(np.diff(rows[lo:hi,:3],axis=0),axis=1)*self.fps
        root_bound = root_speed_bounds(self.targets['root_p'][lo:hi], self.fps)
        if regress_at(j0,j1,self.robot.speed_limits+1e-4) or regress_at(v0,v1,root_bound,1e-5):
            return 'speed_or_seam'
        l,h = max(0,a-18),min(len(rows),b+18)
        if regressions(self.temporal(self.rows,l,h),self.temporal(rows,l,h)):
            return 'new_joint_jitter'
        old = self.supports(self.rows,self.rotations,a,b)
        new = self.supports(rows,rotation,a,b)
        if regress_at(old[:,0],new[:,0],old[:,1]):
            return 'support_regression'
        old_floor = floor_samples(self.robot,dict(root_r=self.rotations[lo:hi]),self.rows[lo:hi])
        new_floor = floor_samples(self.robot,dict(root_r=rotation[lo:hi]),rows[lo:hi])
        if np.any(new_floor<np.minimum(old_floor,-SUPPORT_ACCEPTANCE_M)-2e-6):
            return 'swept_floor_regression'
        old_col = scan(self.model,self.rows[lo:hi],self.rotations[lo:hi],8)[0]
        new_col = scan(self.model,rows[lo:hi],rotation[lo:hi],8)[0]
        if (new_col['collision_samples']>old_col['collision_samples'] or
                new_col['minimum_distance_m']<min(old_col['minimum_distance_m'],0.)-2e-6):
            return 'swept_collision_regression'
        if new_col['collision_samples']:
            from .unified_repair import dense_geometry
            old_distance,_ = dense_geometry(self.robot,self.model,{'root_r':self.rotations[lo:hi]},self.rows[lo:hi])
            new_distance,_ = dense_geometry(self.robot,self.model,{'root_r':rotation[lo:hi]},rows[lo:hi])
            tolerance = self.model.cfg['penetration_tolerance_m']
            if np.any((new_distance < -tolerance) & ((old_distance >= -tolerance) | (new_distance < old_distance-2e-6))):
                return 'swept_collision_pair_regression'
        return None

    def compensate_jitter_foot(self, rows, joint, lo, hi):
        """Hold the jittered hinge and recover the planted sole with other hinges."""
        robot = self.robot
        for side, group in robot.sides.items():
            if joint in group['leg']:
                break
        else:
            return False
        foot = self.targets['sides'][side]
        indices = np.flatnonzero((foot['contact'][lo+1:hi]>0) &
                                 (rows[lo+1:hi,joint]!=self.rows[lo+1:hi,joint])) + lo + 1
        if not len(indices) or self.air[indices].any():
            return False
        columns = np.asarray([j for j in group['leg'] if j!=joint])
        body = group['foot']
        hull = group['hull']
        for i in indices:
            if not self.time_left():
                return False
            current = self.rows[i]
            pivot = foot['pivots'][i]
            robot.set(current,self.rotations[i])
            point = robot.point(body,pivot)
            lowest = hull[np.argmin(hull @ robot.rot(body)[2])]
            goal = np.r_[point[:2],robot.point(body,lowest)[2]]
            row = rows[i].copy()
            seed = row[columns].copy()
            lower = np.maximum(robot.lower[columns],seed-np.deg2rad(8.))
            upper = np.minimum(robot.upper[columns],seed+np.deg2rad(8.))

            def residual(values, jacobian=False):
                row[columns] = values
                robot.set(row,self.rotations[i])
                lowest = hull[np.argmin(hull @ robot.rot(body)[2])]
                if jacobian:
                    xy = robot.jac(body,pivot)[0][:2,columns]
                    z = robot.jac(body,lowest)[0][2,columns]
                    return np.vstack((1000.*np.vstack((xy,z)),np.eye(len(columns))))
                actual = np.r_[robot.point(body,pivot)[:2],robot.point(body,lowest)[2]]
                return np.r_[1000.*(actual-goal),values-seed]

            fit = least_squares(residual,np.clip(seed,lower,upper),bounds=(lower,upper),
                                jac=lambda x: residual(x,True),max_nfev=22)
            rows[i,columns] = fit.x.astype(np.float32)
        return True

    def jitter(self, *, preserve_contact_edges=False):
        quality = self.temporal(self.rows,0,len(self.rows))
        tasks = []
        for j,name in enumerate(self.robot.names,3):
            frames = quality['joints'][name]['violating_frames']
            tasks.extend((b-a,a,b,j,name) for a,b in windows(frames,len(self.rows),0))
        for _,a,b,j,name in sorted(tasks):
            if not self.time_left():
                return
            options = []
            bounds = []
            for radius in (1,2,3,5):
                lo,hi = max(1,a-radius),min(len(self.rows)-2,b+radius)
                bounds.append((False,lo,hi))
                if not preserve_contact_edges:
                    continue
                for side, group in self.robot.sides.items():
                    if j not in group['leg']:
                        continue
                    contact = np.asarray(self.targets['sides'][side]['contact']) > 0
                    for edge in range(max(2,a-1),min(len(contact)-2,b+1)):
                        if contact[edge] and not contact[edge-1]:
                            bounds.append((True,lo,edge))
                        elif contact[edge-1] and not contact[edge]:
                            bounds.append((True,edge-1,hi))
            for anchored,lo,hi in dict.fromkeys(bounds):
                if hi-lo<3:
                    continue
                ds = slice(lo+1,hi)
                original = self.rows[:,j]
                curve = CubicHermiteSpline([lo,hi],original[[lo,hi]],
                          [original[lo]-original[lo-1],original[hi+1]-original[hi]])(np.arange(lo+1,hi))
                start,end = max(0,lo-18),min(len(self.rows),hi+18)
                before = self.temporal(self.rows,start,end)
                old = set(before['joints'][name]['violating_frames'])
                for strength in (.25,.5,.75,1.):
                    if not self.time_left():
                        return
                    rows = self.rows.copy()
                    rows[ds,j] = (original[ds]+strength*(curve-original[ds])).astype(np.float32)
                    if j-3 in self.robot.head_joints:
                        rows[ds] = self.robot.export_rows(rows[ds])
                    if np.any(rows[ds,j] < self.robot.lower[j]) or np.any(rows[ds,j] > self.robot.upper[j]):
                        continue
                    after = self.temporal(rows,start,end)
                    new = set(after['joints'][name]['violating_frames'])
                    if old-new and not regressions(before,after):
                        options.append((anchored,len(new),float(np.max(abs(rows[ds,j]-original[ds]))),lo,hi,rows[ds,j].copy()))
            kept = False
            for anchored in (False,True):
                candidates = sorted((x for x in options if x[0]==anchored),key=lambda x:x[1:3])[:4]
                for _,_,_,lo,hi,values in candidates:
                    if not self.time_left():
                        return
                    rows = self.rows.copy()
                    rows[lo+1:hi,j] = values
                    reason = self.check(rows,self.rotations,lo,hi+1)
                    self.events.append(dict(kind='jitter',window=[lo,hi+1],joint=name,reason=reason,
                                            contact_boundary=anchored))
                    if (preserve_contact_edges and reason in ('support_regression','swept_floor_regression')
                            and self.compensate_jitter_foot(rows,j,lo,hi)):
                        reason = self.check(rows,self.rotations,lo,hi+1)
                        self.events.append(dict(kind='jitter',window=[lo,hi+1],joint=name,reason=reason,
                                                contact_boundary=anchored,foot_compensation=True))
                    if reason is None:
                        self.rows = rows
                        kept = True
                        break
                if kept:
                    break

    def compensate_feet(self, rotation, changed, seed=None):
        r = self.robot
        joint_columns = np.array([i for g in r.sides.values() for i in g['leg']])
        leg = np.r_[np.arange(3), joint_columns] if seed is None else joint_columns
        caps = np.where(leg < 3, .008, np.deg2rad(4. if seed is None else 35.))
        weights = np.where(leg < 3, 1., .03)
        rows = self.rows.copy()
        for frame in changed:
            if not self.time_left():
                raise TimeoutError('temporal_foot_ik_budget')
            original = rows[frame].copy()
            root_goal = None if seed is None else seed[frame, :3]
            r.set(original,self.rotations[frame])
            expected = [(g['foot'],r.point(g['foot']).copy(),r.rot(g['foot']).copy()) for g in r.sides.values()]
            low = np.maximum(r.lower[leg]+1e-8,original[leg]-caps)
            high = np.minimum(r.upper[leg]-1e-8,original[leg]+caps)
            if np.any(high<=low):
                raise ValueError('temporal_foot_ik_joint_bounds')
            margin = np.minimum((high-low)/3,1e-9)
            guess = np.clip(original[leg],low+margin,high-margin)

            def error(values):
                q = original.copy()
                if root_goal is not None:
                    q[:3] = root_goal
                q[leg] = values
                r.set(q,rotation[frame])
                residual = []
                for body,point,orientation in expected:
                    residual.extend((r.point(body)-point)*100.)
                    residual.extend(Rotation.from_matrix(r.rot(body)@orientation.T).as_rotvec()*3.)
                residual.extend((values-original[leg])*weights)
                return residual

            q = least_squares(error,guess,bounds=(low,high),max_nfev=35,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            rows[frame,leg] = q.x
            if root_goal is not None:
                rows[frame,:3] = root_goal
        return rows.astype(np.float32).astype(float)

    def angular(self):
        if ROOT_ANGULAR_LIMIT_RAD_S is None:
            return
        reference = Rotation.from_matrix(self.diagnostics['root_r'])
        initial = angular_speeds(Rotation.from_quat(self.quaternions),self.fps)
        frozen = self.air[:-1] & self.air[1:]
        self.events.append(dict(kind='immutable_airborne_intervals', count=int(np.count_nonzero((initial>ROOT_ANGULAR_LIMIT_RAD_S)&frozen))))
        for start,end in windows(np.flatnonzero(initial>ROOT_ANGULAR_LIMIT_RAD_S).tolist(),len(initial),4):
            if np.all(frozen[start:end]):
                continue
            for padding in (5,8,12,16):
                if not self.time_left():
                    return
                a,b = max(0,start-padding),min(len(self.rows),end+padding+2)
                fixed = self.air[a:b].copy()
                fixed[[0,1,-2,-1]] = True
                if b-a>90 or fixed.all():
                    self.events.append(dict(kind='root_angular',window=[a,b],reason='window_size_or_immutable'))
                    break
                old = Rotation.from_quat(self.quaternions[a:b])
                try:
                    fitted = bounded_rotation(reference[a:b],old,self.fps,fixed,self.deadline)
                except TimeoutError:
                    return
                quats = self.quaternions.copy()
                values = fitted.as_quat().astype(quats.dtype)
                values[np.einsum('ij,ij->i', values, quats[a:b])<0] *= -1
                values[fixed] = quats[a:b][fixed]
                quats[a:b] = values
                rotations = Rotation.from_quat(quats).as_matrix()
                lo,hi = max(0,a-1),min(len(quats),b+1)
                v0 = angular_speeds(Rotation.from_quat(self.quaternions[lo:hi]),self.fps)
                v1 = angular_speeds(Rotation.from_quat(quats[lo:hi]),self.fps)
                absolute = (Rotation.from_quat(values[~fixed])*reference[a:b][~fixed].inv()).magnitude()
                if (absolute.max()>np.deg2rad(5.) or regress_at(v0,v1,ROOT_ANGULAR_LIMIT_RAD_S,1e-6) or
                        np.count_nonzero(v1>ROOT_ANGULAR_LIMIT_RAD_S)>=np.count_nonzero(v0>ROOT_ANGULAR_LIMIT_RAD_S)):
                    self.events.append(dict(kind='root_angular',window=[a,b],reason='angular_constraints'))
                    continue
                changed = np.flatnonzero(np.any(quats!=self.quaternions,axis=1))
                for method in ('rotation_only','preserve_feet'):
                    try:
                        rows = self.rows if method=='rotation_only' else self.compensate_feet(rotations,changed)
                    except TimeoutError:
                        return
                    except ValueError as error:
                        self.events.append(dict(kind='root_angular',window=[a,b],reason=str(error)))
                        continue
                    reason = self.check(rows,rotations,a,b)
                    self.events.append(dict(kind='root_angular',window=[a,b],method=method,reason=reason,
                                            source_deviation_deg=float(np.rad2deg(absolute.max()))))
                    if reason is None:
                        self.rows,self.quaternions,self.rotations = rows,quats,rotations
                        break
                if reason is None:
                    break


def final_guard(before, after, targets, old_cache, new_cache):
    """Reject whole-export regressions that can escape individual windows."""
    added = set(after['flags']) - set(before['flags'])
    if added:
        return 'new_quality_flags:' + ','.join(sorted(added))
    if 'inferred_platform_support' in before or 'inferred_platform_support' in after:
        from .scene_support import gravity_regression
        if gravity_regression(before,after):
            return 'inferred_platform_support_regression'
    if regressions(before['temporal_quality'],after['temporal_quality']):
        return 'new_joint_jitter'
    for model in ('self_collision','full_self_collision'):
        for key in ('collision_samples','native_colliding_frames','interpolated_colliding_intervals'):
            if after[model][key] > before[model][key]:
                return 'collision_count_regression'
    old = old_cache['frames']['body_errors']
    new = new_cache['frames']['body_errors']
    height = targets['body_contact_height_tolerance_m']
    anchor = targets.get('body_contact_anchor_tolerance_m',height)
    for c in targets['body_contacts']:
        part = c['body_name']
        for key,tolerance in (('normal_cm',height),('fixed_xy_cm',anchor)):
            for i,(a,b) in enumerate(zip(old[part][key],new[part][key])):
                if a is not None and b is not None and preserve_contact(c,i) and b > max(a,tolerance*100)+1e-5:
                    return 'body_support_regression:' + part
    if (after['adaptive_pose_gate']['violating_frames'] > before['adaptive_pose_gate']['violating_frames'] and
            'sustained_arm_pose_mismatch' in before['flags']):
        return 'more_sustained_pose_frames'
    return None
