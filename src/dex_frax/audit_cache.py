"""Exact measurement reuse, never percentile-of-percentiles or stale acceptance.

Caches belong to one immutable source/model/configuration. A candidate must
prove all rows outside its declared window unchanged. Trial caches are forked;
a rejected candidate cannot poison the committed cache. Final audit bypasses it.
"""
import copy
import hashlib
import json
import numpy as np


def digest(meta, diagnostics, model_path):
    from .cli import sha
    h=hashlib.sha256()
    static={k:meta.get(k) for k in ('configuration','estimated_contact_segments','body_contact_models','baseline','source_type','scene_support')}
    h.update(json.dumps(static,sort_keys=True).encode());h.update(sha(model_path).encode())
    for key,value in sorted(diagnostics.items()):
        array=np.asarray(value);h.update(key.encode());h.update(str(array.shape).encode());h.update(str(array.dtype).encode());h.update(array.tobytes())
    return h.hexdigest()


def valid_rows(cache, data, changed, key=None):
    if not cache or changed is None or (key is not None and cache.get('key')!=key):return False
    a,b=changed;n=len(data['dof_pos'])
    if not 0<=a<b<=n:return False
    for name in ('dof_pos','root_pos','root_rot'):
        old=cache.get('data',{}).get(name);new=data[name]
        if old is None or old.shape!=new.shape:return False
        if not np.array_equal(old[:a],new[:a]) or not np.array_equal(old[b:],new[b:]):return False
    return cache.get('fps')==data['fps'] and cache.get('names')==list(data['joint_names'])


def remember(cache,data,key=None):
    cache.update(data={k:np.array(data[k],copy=True) for k in ('dof_pos','root_pos','root_rot')},
                 fps=data['fps'],names=list(data['joint_names']))
    if key is not None:cache['key']=key


def fork(cache, changed=None):
    """Copy writable frame containers and the proposed window only.

    Dense floor storage is copied by dense_floor before writes. Collision scan
    copies its sample list before replacing the affected halo. Saved trajectory
    and metric arrays are read-only until remember/audit replaces them.
    """
    if changed is None:return copy.deepcopy(cache)
    a,b=changed;n=len(cache.get('data',{}).get('dof_pos',[]))
    if not 0<=a<b<=n:raise ValueError('Invalid audit cache fork window')
    def readonly(value):
        if isinstance(value,np.ndarray):
            view=value.view();view.flags.writeable=False;return view
        if isinstance(value,dict):return {k:readonly(v) for k,v in value.items()}
        return value
    def frame_values(value):
        if isinstance(value,dict):return {k:frame_values(v) for k,v in value.items()}
        if isinstance(value,list):
            out=list(value)
            if len(value)==n:out[a:b]=copy.deepcopy(value[a:b])
            else:return copy.deepcopy(value)
            return out
        return readonly(value)
    out={k:readonly(v) for k,v in cache.items()}
    out['frames']=frame_values(cache['frames'])
    for name in ('arm_collision','full_collision'):
        if name not in cache:continue
        original=cache[name];part={k:readonly(v) for k,v in original.items()}
        if 'samples' in original:
            steps=original.get('key',(8,))[0];_,_,lo,hi=dense_range(changed,n,steps)
            samples=list(original['samples']);samples[lo:hi]=copy.deepcopy(samples[lo:hi]);part['samples']=samples
        out[name]=part
    return out


def dense_range(changed,n,steps=8):
    a,b=changed;lo=max(0,a-1);hi=min(n,b+1)
    return lo,hi,lo*steps,(hi-1)*steps+1


def dense_floor(robot,rr,rows,cache,changed,valid):
    from .torso_branch import floor_samples
    if valid and 'floor' in cache:
        lo,hi,a,b=dense_range(changed,len(rows))
        values=cache['floor'].copy()
        values[a:b]=floor_samples(robot,{'root_r':rr[lo:hi]},rows[lo:hi])
    else:values=floor_samples(robot,{'root_r':rr},rows)
    cache['floor']=values
    return values


def clear_dense_window(cache,start,end,n):
    """A cached nonnegative minimum proves every declared pair nonnegative.

    Only exact float32 export-audit caches are supplied by the caller. Missing
    coverage is never treated as clear. Existing penetrations still need their
    per-pair comparison; this shortcut applies only to completely clear input
    and candidate windows.
    """
    samples=cache.get('full_collision',{}).get('samples')
    floor=cache.get('floor')
    if samples is None or floor is None or len(samples)!=(n-1)*8+1 or len(floor)!=len(samples):return False
    low=start*8;high=(end-1)*8+1
    return all(sample is not None and sample[0]>=0. for sample in samples[low:high])
