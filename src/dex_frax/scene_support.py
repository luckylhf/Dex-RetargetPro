"""Measured horizontal box tops and independently checked export geometry."""

import json
from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares

from .contact_probe import scene_boxes
from .v8.support_acceptance import SUPPORT_ACCEPTANCE_M

NONE = -1
TARGET_CLEARANCE_M = .00065


def footprint(geometry, rotation, origin):
    return geometry['hull'] @ rotation.T + origin


def within(points, box):
    center = np.asarray(box['center_m'])
    half = np.asarray(box['half_size_m'])
    return bool(np.all(np.abs(points[:, :2]-center[:2]) <= half[:2]+1e-8))


def top(box):
    return box['center_m'][2]+box['half_size_m'][2]


def assign(targets, robot, observation, boxes):
    """Associate stationary source evidence with a bounded, supplied top."""
    count = len(targets['root_p'])
    for values in targets['sides'].values():
        values['platform_index'] = np.full(count,NONE,dtype=int)
        values['platform_anchor'] = np.zeros((count,3))
    for segment in observation['elevated_foot_evidence']:
        side = segment['side']
        values = targets['sides'][side]
        geometry = robot.sides[side]
        a,b = segment['start'],segment['end']
        poses = values['raw_foot_p'][a:b]
        rotations = values['raw_foot_r'][a:b]
        soles = [footprint(geometry,r,p) for r,p in zip(rotations,poses)]
        possible = [i for i,box in enumerate(boxes) if all(
            within(sole,box) and abs(sole[:,2].min()-top(box)) <= .08 for sole in soles)]
        if not possible:
            continue
        index = min(possible,key=lambda i: np.mean([abs(sole[:,2].min()-top(boxes[i])) for sole in soles]))
        points = poses + rotations @ geometry['sole']
        values['platform_index'][a:b] = index
        values['platform_anchor'][a:b] = [*np.median(points[:,:2],axis=0),top(boxes[index])]
        segment['measured_surface'] = boxes[index]['id']
    observation['surface_geometry_supplied'] = bool(boxes)
    observation['surface_geometry_measured'] = bool(boxes)


def fit(robot, targets, rows):
    """Dedicated contact fit leaves the generic floor/flight solver intact."""
    active = np.logical_or.reduce([v['platform_index']>=0 for v in targets['sides'].values()])
    if not active.any():
        return rows
    result = rows.copy()
    ids = np.array([0,1,2,*[j for g in robot.sides.values() for j in g['leg']]])
    movement = np.array([.08,.08,.13,*([.45]*(len(ids)-3))])
    for i in np.flatnonzero(active):
        row = rows[i]
        rotation = targets['root_r'][i]
        robot.set(row,rotation)
        goals = []
        for side,g in robot.sides.items():
            values = targets['sides'][side]
            if values['platform_index'][i]>=0:
                anchor = values['platform_anchor'][i].copy()
                # Plan just inside the support tolerance, allowing clearance
                # for small differences between the contact and collision meshes.
                anchor[2] += TARGET_CLEARANCE_M
                goals.append((g,g['sole'],anchor,values['raw_foot_r'][i]))
            elif values['contact'][i]>0:
                goals.append((g,values['pivots'][i],values['anchors'][i],values['foot_target_r'][i]))
        lo = np.maximum(row[ids]-movement,robot.lower[ids])
        hi = np.minimum(row[ids]+movement,robot.upper[ids])
        if np.any(hi<=lo):
            continue

        def residual(q):
            candidate = row.copy()
            candidate[ids] = q
            robot.set(candidate,rotation)
            terms = [(q-row[ids])*.15]
            for g,pivot,anchor,orientation in goals:
                r = robot.rot(g['foot'])
                cloud = footprint(g,r,robot.point(g['foot']))
                point = robot.point(g['foot'],pivot)
                terms.extend([(point[:2]-anchor[:2])*250,[(cloud[:,2].min()-anchor[2])*400],
                              (r[:,2]-orientation[:,2])*12,(r[:,0]-orientation[:,0])*8])
            return np.concatenate(terms)

        seed = np.clip(result[i-1,ids] if i and active[i-1] else row[ids],lo+1e-10,hi-1e-10)
        result[i,ids] = least_squares(residual,seed,bounds=(lo,hi),max_nfev=22,ftol=1e-8,gtol=1e-8).x
    return result


@lru_cache(maxsize=2)
def scene_model(path, scene_json):
    import mujoco
    from .model_cache import model_spec
    spec = model_spec(path)
    for i,box in enumerate(json.loads(scene_json)):
        spec.worldbody.add_geom(name=f'retarget_scene_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=box['center_m'],size=box['half_size_m'])
    return spec.compile()


def audit_scene(robot, rows, rotations, diagnostics, boxes):
    """Check declared robot collision meshes against box volume at 8 substeps."""
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp
    if not boxes:
        return None
    model = scene_model(robot.path,json.dumps(boxes,sort_keys=True))
    data = mujoco.MjData(model)
    box_ids = [model.geom(f'retarget_scene_{i}').id for i in range(len(boxes))]
    from .foot_collision import offline_collision_geoms
    from .robot_profile import SIDE_LAYOUT
    foot_hulls = [model.geom(layout['sole_geom']).id for layout in SIDE_LAYOUT.values()]
    geoms = np.r_[offline_collision_geoms(model), foot_hulls]
    geoms = geoms[model.geom_bodyid[geoms] != 0]
    radii = model.geom_rbound[geoms]+SUPPORT_ACCEPTANCE_M
    bounds = [(j,np.asarray(b['center_m']),np.asarray(b['half_size_m'])) for j,b in zip(box_ids,boxes)]
    samples = np.arange((len(rows)-1)*8+1)/8.
    source_grid = np.arange(len(rows))
    angles = Slerp(source_grid,Rotation.from_matrix(rotations))(samples).as_quat()
    poses = np.column_stack([np.interp(samples,source_grid,col) for col in rows.T])
    penetration = []
    witness = np.empty(6)
    worst = 0.
    # Conservative bounding sphere rejection before exact convex distance.
    for s,(pose,quat) in enumerate(zip(poses,angles)):
        data.qpos[:3]=pose[:3]
        data.qpos[3:7]=quat[[3,0,1,2]]
        data.qpos[robot.qadr]=pose[3:]
        mujoco.mj_kinematics(model,data)
        centers = data.geom_xpos[geoms]
        for j,center,half in bounds:
            outside = np.maximum(np.abs(centers-center)-half,0.)
            near = np.linalg.norm(outside,axis=1) <= radii
            for g in geoms[near]:
                distance = mujoco.mj_geomDistance(model,data,int(g),j,.01,witness)
                worst = min(worst,float(distance))
                if distance < -SUPPORT_ACCEPTANCE_M:
                    penetration.append(s)
                    break
    errors = []
    for side,g in robot.sides.items():
        index = diagnostics.get(side+'_platform_index',np.full(len(rows),NONE))
        anchors = diagnostics.get(side+'_platform_anchor')
        for i in np.flatnonzero(index>=0):
            robot.set(rows[i],rotations[i])
            cloud = footprint(g,robot.rot(g['foot']),robot.point(g['foot']))
            point = robot.point(g['foot'],g['sole'])
            error = float(np.linalg.norm(np.r_[point[:2]-anchors[i,:2],cloud[:,2].min()-anchors[i,2]]))
            inside = within(cloud,boxes[index[i]])
            if error>SUPPORT_ACCEPTANCE_M or not inside:
                errors.append(dict(frame=int(i),side=side,error_m=error,within_platform=inside))
    return dict(checked=True,support_targets=int(sum(np.count_nonzero(
                    np.asarray(diagnostics.get(side+'_platform_index',[]))>=0) for side in robot.sides)),
                support_violations=errors,minimum_convex_distance_m=worst,
                penetrating_samples=sorted(set(penetration)),substeps=8)
