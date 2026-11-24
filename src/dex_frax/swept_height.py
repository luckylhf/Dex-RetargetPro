"""Batch small swept-floor defects through the exact root-translation identity."""
import time
import numpy as np
from .airborne import protected_height
from .support_quality import preserve_contact
from .v8.multicontact import contact_local

FLAG = 'interpolated_full_body_mesh_penetration_over_1mm'


def distribute(height, capacity, weights, required):
    """Add the minimum common lift, respecting both endpoint capacities."""
    gap = required-float(weights@height)
    if gap<=0:
        return height
    room = capacity-height
    useful=weights>0
    for _ in range(2):
        active=useful & (room>1e-12)
        if not active.any() or gap<=1e-12:break
        change=np.zeros(2)
        change[active]=np.minimum(room[active],gap/weights[active].sum())
        height=height+change
        room=room-change
        gap-=float(weights@change)
    return height


def project(robot, targets, original, proposal, quats, cache, deadline):
    floor=np.min(cache['floor'],axis=1)
    if floor.min()<-.003 or len(floor)!=8*(len(original)-1)+1:
        return dict(changed_frames=0)
    bad=np.flatnonzero(floor<-.001)
    if not len(bad):return dict(changed_frames=0)
    changed=np.any(proposal!=original,axis=1)|np.any(quats!=cache['data']['root_rot'],axis=1)
    blocked=protected_height(targets,len(original)).copy()
    n=len(original)
    capacity={};offset={}
    def bound(i):
        if i in capacity:return capacity[i]
        robot.set(original[i],targets['root_r'][i])
        top=.002
        for side in robot.sides:
            err=cache['frames']['contact_xyz'][side][i]
            if err is not None:
                x,y,z=np.asarray(err)*.01
                radius=max(.0009995,np.linalg.norm([x,y,z]))
                top=min(top,np.sqrt(max(0.,radius**2-x*x-y*y))-z)
        for contact in targets.get('body_contacts',()):
            if preserve_contact(contact,i):
                z=robot.point(contact['body'],contact_local(robot,contact))[2]
                top=min(top,max(abs(z),targets['body_contact_height_tolerance_m']-.000001)-z)
        capacity[i]=max(0.,top-2e-7) if not blocked[i] else 0.
        return capacity[i]
    # The caller supplies the actual changed states. Do not use a cached
    # floor sample if the temporal candidate touched either endpoint.
    for k in bad:
        if time.monotonic()>=deadline:break
        i=min(k//8,n-2);ids=[i,i+1];u=k/8-i
        if changed[ids].any():continue
        weights=np.array([1.-u,u])
        h=np.array([offset.get(j,0.) for j in ids])
        caps=np.array([bound(j) for j in ids])
        new=distribute(h,caps,weights,-.00098-floor[k])
        for j,value in zip(ids,new):
            if value>0:offset[j]=float(value)
    for j,dz in offset.items():proposal[j,2]=np.float32(proposal[j,2]+dz)
    return dict(changed_frames=len(offset),bad_samples_before=len(bad),
                max_lift_mm=1000*max(offset.values(),default=0.))
