"""Re-use the window solver on temporal outliers before the shared sparse-floor audit."""
import time
import numpy as np
from osqp import OSQPException
from .refine_queue import jobs_from_masks
from .temporal_constraints import regressions


def finish(engine, weights, nominal_deadline, *, candidate_seconds):
    from .refine_pipeline import propose
    n=len(engine.rows)
    before=engine.temporal(engine.rows,0,n)
    masks=[]
    for joint,measure in before['joints'].items():
        bad=measure['violating_frames']
        if bad:
            mask=np.zeros(n,bool)
            mask[np.asarray(bad,int)]=True
            masks.append(('jitter',joint,mask))
    events=[]
    targets=dict(engine.targets,root_r=engine.rotations)
    for job in jobs_from_masks(masks,n):
        if time.monotonic()>=nominal_deadline:break
        a,b,side=job['start'],job['end'],job['side']
        lo,hi=max(0,a-20),min(n,b+20)
        old=engine.temporal(engine.rows,lo,hi)
        if not old['joints'][side]['violating_frames']:continue
        try:
            local_deadline = min(nominal_deadline, time.monotonic() + candidate_seconds)
            proposal,_=propose(engine.robot,targets,engine.rows,engine.rows,weights,engine.model,
                               job,'temporal_more',local_deadline,{})
            proposal=engine.robot.export_rows(proposal).astype(float)
            proposal[engine.air]=engine.rows[engine.air]
            new=engine.temporal(proposal,lo,hi)
            gain=len(new['joints'][side]['violating_frames'])<len(old['joints'][side]['violating_frames'])
            if not gain or regressions(old,new):
                reason='no_temporal_gain'
            else:
                reason=engine.check(proposal,engine.rotations,a,b)
            if reason is None:
                engine.rows=proposal
        except (RuntimeError,ValueError,TimeoutError,OSQPException) as exc:
            reason=str(exc)
        events.append(dict(joint=side,start=a,end=b,reason=reason))
    return events
