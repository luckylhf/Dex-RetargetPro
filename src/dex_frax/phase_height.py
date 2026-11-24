"""Exact one-axis feasibility fallback for an already colliding mixed-contact pose."""
import numpy as np
from .airborne import protected_height
from scipy.optimize import Bounds, LinearConstraint, minimize
from .support_quality import preserve_contact
from .v8.multicontact import contact_local


def project(robot,t,old,a,floor,goal,use_patch):
    n=len(old)
    lower=np.full(n,-.30)
    upper=np.full(n,.30)
    for j,row in enumerate(old):
        i=a+j
        robot.set(row,t['root_r'][i])
        for side,g in robot.sides.items():
            f=t['sides'][side]
            if f['contact'][i]<=0:continue
            xy=robot.point(g['foot'],f['pivots'][i])[:2]-f['anchors'][i,:2]
            low=robot.point(g['foot'],robot.lowest(side))[2]
            radius=max(.000999,float(np.hypot(np.linalg.norm(xy),low)))
            dz=np.sqrt(max(0.,radius**2-xy@xy))
            lower[j]=max(lower[j],-dz-low)
            upper[j]=min(upper[j],dz-low)
            if use_patch:
                from .contact_phase import surface_patch,lowest
                patch=surface_patch(g,int(f['phase'][i]))
                patch_z=lowest(robot,g['foot'],patch)[1][2]
                allowance=max(abs(patch_z),.001)
                lower[j]=max(lower[j],-allowance-patch_z)
                upper[j]=min(upper[j],allowance-patch_z)
        for c in t.get('body_contacts',()):
            if not preserve_contact(c,i):continue
            z=robot.point(c['body'],contact_local(robot,c))[2]
            radius=max(abs(z),t['body_contact_height_tolerance_m']-.00001)
            lower[j]=max(lower[j],-radius-z)
            upper[j]=min(upper[j],radius-z)
    fixed=protected_height(t,len(t['root_p']))[a:a+n].copy()
    fixed[[0,1,-2,-1]]=True
    lower=np.minimum(0.,lower);upper=np.maximum(0.,upper)
    lower[fixed]=upper[fixed]=0.
    if np.all(upper-lower<1e-10):return old
    A=np.zeros((len(floor),n))
    for k in range(len(floor)):
        j,u=divmod(k,8)
        A[k,j]=1-u/8
        if u:A[k,j+1]=u/8
    bound=(np.minimum(floor,-.000999)-floor).max(axis=1)
    desired=goal[:,2]-old[:,2]
    def value(z):
        return .5*np.sum((z-desired)**2)
    sol=minimize(value,np.zeros(n),jac=lambda z:z-desired,bounds=Bounds(lower,upper),
                 constraints=[LinearConstraint(A,bound,np.inf)],method='SLSQP',
                 options={'maxiter':65,'ftol':1e-12})
    rows=old.copy()
    if np.isfinite(sol.x).all():rows[:,2]+=sol.x
    return rows
