"""Explicit BVH channel conventions; retain End Sites as kinematic landmarks."""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

@dataclass
class Motion:
    names: list
    parents: np.ndarray
    offsets: np.ndarray
    local_p: np.ndarray
    local_r: np.ndarray
    fps: float
    input_frames: int
    input_fps: float
    source: str

    def fk(self):
        p, r = self.local_p.copy(), self.local_r.copy()
        for j, parent in enumerate(self.parents):
            if parent >= 0:
                p[:, j] = p[:, parent] + np.einsum('nij,nj->ni', r[:, parent], p[:, j])
                r[:, j] = r[:, parent] @ r[:, j]
        # Proper rotation (det +1): BVH Y up -> robot Z up.
        c = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        return ({k: p[:, i] @ c.T for i, k in enumerate(self.names)},
                {k: c @ r[:, i] for i, k in enumerate(self.names)})

def load(path, fps=None, units='cm', translations='absolute'):
    """Absolute channels match SOMA's official reader and supplied LAFAN/mocap.

    For BVH producers that encode channels as deltas from OFFSET, explicitly
    select translations='offset'. Do not guess from a file's first pose.
    """
    if units not in ('cm','m','mm'):raise ValueError('Unsupported BVH units: '+str(units))
    if translations not in ('absolute','offset'):raise ValueError('Unsupported BVH translation convention: '+str(translations))
    text = Path(path).read_text()
    head, body = text.split('MOTION', 1)
    tok = re.findall(r'[^\s{}]+|[{}]', head)
    names, parents, offsets, channels = [], [], [], []
    stack, current, i, cursor = [], None, 0, 0
    factor = {'cm': .01, 'm': 1., 'mm': .001}[units]
    while i < len(tok):
        t = tok[i]; i += 1
        if t in ('ROOT', 'JOINT', 'End'):
            name = tok[i]; i += 1
            if t == 'End': name = names[stack[-1]] + '__EndSite'
            if name in names: raise ValueError('Duplicate joint: ' + name)
            current = len(names)
            names.append(name); parents.append(stack[-1] if stack else -1)
            offsets.append(np.zeros(3)); channels.append([])
        elif t == '{': stack.append(current)
        elif t == '}': stack.pop()
        elif t == 'OFFSET':
            offsets[stack[-1]] = np.array(tok[i:i+3], float) * factor; i += 3
        elif t == 'CHANNELS':
            count = int(tok[i]); i += 1
            channels[stack[-1]] = [(cursor+k, v) for k,v in enumerate(tok[i:i+count])]
            cursor += count; i += count
    lines = body.strip().splitlines()
    count = int(lines[0].split(':')[1]); dt = float(lines[1].split(':')[1])
    if count < 1 or not np.isfinite(dt) or dt <= 0: raise ValueError('Need at least one frame and positive frame time')
    values = np.fromstring(' '.join(lines[2:]), sep=' ')
    if values.size != count*cursor or not np.isfinite(values).all(): raise ValueError('Malformed/nonfinite BVH motion')
    values = values.reshape(count, cursor)
    lp = np.broadcast_to(offsets, (count,len(names),3)).copy()
    lr = np.broadcast_to(np.eye(3), (count,len(names),3,3)).copy()
    for j, ch in enumerate(channels):
        rc = [(col,name[0].upper()) for col,name in ch if name.lower().endswith('rotation')]
        pc = [(col,'XYZ'.index(name[0].upper())) for col,name in ch if name.lower().endswith('position')]
        for col,axis in pc:
            lp[:,j,axis] = values[:,col]*factor + (offsets[j][axis] if translations == 'offset' else 0.)
        if rc: lr[:,j] = Rotation.from_euler(''.join(x[1] for x in rc), values[:,[x[0] for x in rc]], degrees=True).as_matrix()
    motion=Motion(names,np.array(parents),offsets,lp,lr,1./dt,count,1./dt,str(Path(path).resolve()))
    return resample(motion,fps) if fps is not None else motion


def resample(motion, fps):
    """Resample edited native arrays once while retaining original time data."""
    from dataclasses import replace
    output_fps=float(fps)
    if not np.isfinite(output_fps) or output_fps<=0:
        raise ValueError('FPS must be positive')
    count=len(motion.local_p)
    if count<2 or abs(output_fps-motion.fps)<=1e-5:
        return replace(motion,fps=output_fps)
    old_t=np.arange(count)*(1./motion.fps)
    new_t=np.arange(int(np.floor(old_t[-1]*output_fps+1e-8))+1)/output_fps
    positions=np.empty((len(new_t),len(motion.names),3))
    rotations=np.empty((len(new_t),len(motion.names),3,3))
    for j in range(len(motion.names)):
        for axis in range(3):
            positions[:,j,axis]=np.interp(new_t,old_t,motion.local_p[:,j,axis])
        rotations[:,j]=Slerp(old_t,Rotation.from_matrix(motion.local_r[:,j]))(new_t).as_matrix()
    return replace(motion,local_p=positions,local_r=rotations,fps=output_fps)
