"""Report candidate support surfaces without promoting inferred planes to facts."""

import numpy as np

from .source_integrity import stable_segments


def summarize(targets):
    fps = targets['fps']
    floor = targets['floor']
    contact = targets['source_expected_contact']
    hips = targets['hips'][:,2]
    feet = list(targets['sides'].items())
    sole_z = np.column_stack([np.minimum(f['heel'][:,2],f['toe'][:,2])-floor for _,f in feet])
    elevated = np.min(sole_z,axis=1)>.08
    raised = np.zeros(len(hips),dtype=bool)
    plateaus = []
    for j,(side,foot) in enumerate(feet):
        center=(foot['heel']+foot['toe'])/2
        speed=np.linalg.norm(np.diff(center,axis=0,prepend=center[:1]),axis=1)*fps
        flat=foot['foot_r'][:,2,2]>np.cos(np.deg2rad(18))
        for a,b in stable_segments((sole_z[:,j]>.08) & flat & (speed<.14)):
            if b-a<max(6,round(fps*.20)) or np.ptp(sole_z[a:b,j])>.035:
                continue
            raised[a:b] |= elevated[a:b]
            confirmed=targets['source_airborne'][a:b]
            plateaus.append(dict(start=a,end=b,side=side,height_above_floor_m=float(np.median(sole_z[a:b,j])),
                                 legacy_airborne_frames=int(confirmed.sum())))
    airborne=np.zeros(len(hips),dtype=bool)
    for a,b in stable_segments(targets['source_airborne'] & ~raised):
        if a==0 or b==len(hips) or not .12<=(b-a)/fps<=.85:
            continue
        apex=int(np.argmax(hips[a:b]))+a
        if (a<apex<b-1 and min(hips[apex]-hips[a],hips[apex]-hips[b-1])>.035
                and min(sole_z[apex])>.10):
            airborne[a:b]=True
    states={'ground_or_body_contact': contact & ~raised, 'elevated_support_candidate': raised,
            'airborne_candidate': airborne, 'unknown': ~(contact | raised | airborne)}
    return dict(fps=fps,reference_floor_m=floor,frame_convention='zero_based_prepared_end_exclusive',
                segments={key:list(stable_segments(mask)) for key,mask in states.items()},
                elevated_foot_evidence=plateaus,
                informational_only=True,surface_geometry_measured=False,surface_geometry_supplied=False)



def apply(targets, robot, boxes):
    """Keep uncertain altitude separate from measured scene support."""
    observation = summarize(targets)
    from .scene_support import assign
    assign(targets,robot,observation,boxes)
    unresolved = [item for item in observation['elevated_foot_evidence']
                  if item['legacy_airborne_frames'] >= (item['end']-item['start'])*.5
                  and not item.get('measured_surface')]
    targets['source_support_uncertain'] = np.zeros(len(targets['root_p']),dtype=bool)
    for item in unresolved:
        targets['source_support_uncertain'][item['start']:item['end']] = True
    known = np.logical_or.reduce([v['platform_index']>=0 for v in targets['sides'].values()])
    targets['source_scene_support'] = known
    targets['source_airborne'] &= ~(known | targets['source_support_uncertain'])
    observation['segments']['measured_platform'] = list(stable_segments(known))
    observation['segments']['unconfirmed_platform'] = list(stable_segments(targets['source_support_uncertain']))
    observation['unresolved_platforms'] = unresolved
    observation['informational_only'] = not known.any()
    return observation
