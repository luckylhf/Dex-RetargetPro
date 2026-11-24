"""Select compatible neighboring poses; verify expensive swept edges on demand."""
import numpy as np
from . import robot_profile as target_profile

from .temporal_constraints import root_speed_bounds
from .unified_repair import dense_geometry


def upper_errors(robot,t,frame):
    result=[]
    for side,g in robot.sides.items():
        refs=t['sides'][side]
        for base,tip,key in (('shoulder','elbow','upper'),('elbow','wrist','lower')):
            direction=robot.point(g[tip])-robot.point(g[base])
            result.append((direction/np.linalg.norm(direction))@refs[key][frame])
        R=robot.rot(g['tcp'])
        result.extend(((R@target_profile.TCP_FORWARD)@refs['hand_forward'][frame],(R@target_profile.TCP_WIDTH)@refs['hand_width'][frame],
                       (robot.rot(g['shoulder'])@target_profile.ELBOW_FLEXION_AXIS)@refs['elbow_axis'][frame]))
    return np.rad2deg(np.arccos(np.clip(result,-1.,1.)))


def select(fit, candidates, model, before_distances, before_floor, support_errors, check_temporal):
    old = fit.old.astype(np.float32).astype(float)
    r,t,start = fit.r,fit.t,fit.start
    rotations = t['root_r'][start:fit.stop]
    n = len(old)
    from .adaptive_refine import policy
    gates=policy()
    thresholds=np.tile([gates['arm_direction_deg']]*2+[gates['hand_axis_deg']]*2+[gates['elbow_axis_deg']],len(r.sides))
    nodes = []
    for j,row in enumerate(old):
        layers = [row]
        previous,bounds = support_errors(r,t,row[None],start+j)
        old_pose=upper_errors(r,t,start+j)
        for candidate in candidates:
            for f in (1.,.8,.6,.4,.25,.12,.06,.025):
                q = (row+f*(candidate[j]-row)).astype(np.float32).astype(float)
                if any(np.array_equal(q,prev) for prev in layers): continue
                current,_ = support_errors(r,t,q[None],start+j)
                if np.any(current > np.maximum(bounds,previous)+1e-6): continue
                pose=upper_errors(r,t,start+j)
                if np.any(pose>np.maximum(thresholds,old_pose)+.02):continue
                knees=[g['leg'][3] for g in r.sides.values()]
                if np.any((row[knees]>=0.) & (q[knees]<-1e-7)):continue
                model.update(q,rotations[j])
                if np.any(model.distances()<np.minimum(before_distances[8*j],0.)-1e-6):continue
                z = np.array([r.point(b)[2]+np.min(shape['hull']@r.rot(b)[2]) for b,shape in r.surface_models.items()])
                if np.any(z<np.minimum(before_floor[8*j],-.001)-1e-6):continue
                layers.append(q)
        nodes.append(np.array(layers))
    costs = []
    root_bounds = root_speed_bounds(t["root_p"][start:fit.stop],fit.fps)
    for j,(left,right) in enumerate(zip(nodes,nodes[1:])):
        velocity = (right[None,:,:]-left[:,None,:])*fit.fps
        initial = (old[j+1]-old[j])*fit.fps
        speed = np.linalg.norm(velocity[:,:,:3],axis=2)
        allowed = (speed<=max(root_bounds[j],np.linalg.norm(initial[:3]))+1e-5)
        allowed &= (abs(velocity[:,:,3:])<=np.maximum(abs(initial[3:]),r.speed_limits)+1e-4).all(axis=2)
        effort = .0001*np.sum((right-old[j+1])**2,axis=1)
        cost = np.maximum(speed-root_bounds[j],0.)**2 + effort[None,:]
        costs.append(np.where(allowed,cost,np.inf))
    initial_cost = sum(m[0,0] for m in costs)
    verified = set()
    for _ in range(90):
        distance = np.zeros(len(nodes[0]))
        parents = []
        for cost in costs:
            values = distance[:,None]+cost
            parents.append(values.argmin(axis=0))
            distance=values.min(axis=0)
        idx=int(np.argmin(distance))
        if distance[idx]>=initial_cost-1e-5:return None
        labels=[idx]
        for back in reversed(parents):labels.append(int(back[labels[-1]]))
        labels.reverse()
        rows=np.array([layer[k] for layer,k in zip(nodes,labels)])
        valid=True
        for j,(left,right) in enumerate(zip(labels,labels[1:])):
            key=(j,left,right)
            if (not left and not right) or key in verified:continue
            ds,fs=dense_geometry(r,model,{'root_r':rotations[j:j+2]},rows[j:j+2])
            section=slice(j*8,j*8+9)
            if (np.any(ds<np.minimum(before_distances[section],0.)-2e-6)
                    or np.any(fs<np.minimum(before_floor[section],-.001)-2e-6)):
                costs[j][left,right]=np.inf
                valid=False
            else: verified.add(key)
        if not valid:continue
        jitter = check_temporal(rows)
        if not jitter:return rows
        # Eliminate an implicated edge, preserving the unchanged zero path.
        for absolute in jitter:
            i=max(0,min(n-2,absolute-start))
            neighborhood=range(max(0,i-2),min(n-1,i+3))
            moved=[k for k in neighborhood if labels[k] or labels[k+1]]
            if moved:
                k=min(moved,key=lambda k:abs(k-i))
                costs[k][labels[k],labels[k+1]]=np.inf
                valid=False
        if valid:return None
    return None
