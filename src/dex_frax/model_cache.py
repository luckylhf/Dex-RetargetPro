"""Verified binary model cache; spawn workers reuse models, never fork JAX."""
import hashlib
import json
from pathlib import Path
from dex_frax.robot_profile import MODEL_XML, PELVIS_BODY

_model=None
_xml=None


def model_spec(path):
    """Map the original Dex MJCF to the retargeter's visual and world frames."""
    import mujoco

    path = Path(path).resolve()
    spec = mujoco.MjSpec.from_file(str(path))
    original = spec.compile()
    # The upstream pelvis has inferred inertia. Preserve it before changing
    # visual meshes so that loading the retarget geometry does not change mass.
    for body in spec.bodies:
        if body.name != 'world' and not body.explicitinertial:
            compiled = original.body(body.name)
            body.explicitinertial = True
            body.mass = compiled.mass.item()
            body.ipos = compiled.ipos
            body.iquat = compiled.iquat
            body.inertia = compiled.inertia

    visual_meshes = {}
    for geom in spec.geoms:
        if not geom.name.endswith('_visual') or not geom.meshname:
            continue
        name = geom.meshname
        if name not in visual_meshes:
            filename = Path(spec.mesh(name).file).name
            visual = path.parent / 'meshes_simplify' / filename
            visual_meshes[name] = spec.add_mesh(name=f'{name}_retarget_visual', file=str(visual)).name
        geom.meshname = visual_meshes[name]

    floor = spec.geom('floor')
    pelvis = spec.body(PELVIS_BODY)
    pelvis.pos[2] -= floor.pos[2]
    floor.pos[2] = 0.
    return spec


def prepare(assets,cache):
    import mujoco
    import numpy as np
    from .paths import ASSETS
    mapping=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    key=hashlib.sha256(json.dumps(dict(assets=assets,mujoco=mujoco.__version__,mapping=mapping),sort_keys=True).encode()).hexdigest()
    folder=Path(cache)/key;folder.mkdir(parents=True,exist_ok=True)
    binary=folder/'model.mjb';receipt=folder/'receipt.json';xml=ASSETS/MODEL_XML
    if receipt.exists():
        info=json.loads(receipt.read_text())
        if hashlib.sha256(binary.read_bytes()).hexdigest()==info['sha256']:
            return dict(binary=str(binary),xml=str(xml),sha256=info['sha256'])
        raise ValueError('Cached MJB hash mismatch')
    original=model_spec(xml).compile()
    temporary=folder/'model.tmp.mjb';mujoco.mj_saveModel(original,str(temporary))
    restored=mujoco.MjModel.from_binary_path(str(temporary));compared=[]
    for name in dir(original):
        a=getattr(original,name)
        if isinstance(a,np.ndarray):
            if not np.array_equal(a,getattr(restored,name)):raise ValueError('XML/MJB mismatch: '+name)
            compared.append(name)
    temporary.replace(binary)
    info=dict(sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),arrays_compared=compared,assets=assets,mujoco=mujoco.__version__)
    receipt.write_text(json.dumps(info,indent=2))
    return dict(binary=str(binary),xml=str(xml),sha256=info['sha256'])

def enable(info):
    import mujoco
    global _model,_xml
    if hashlib.sha256(Path(info['binary']).read_bytes()).hexdigest()!=info['sha256']:raise ValueError('MJB changed before worker load')
    _model=mujoco.MjModel.from_binary_path(info['binary']);_xml=str(Path(info['xml']).resolve())

def load(path):
    if _model is not None and str(Path(path).resolve())==_xml:return _model
    return model_spec(path).compile()

# Experimental worker-local, read-only collision variants. MjData is never shared.
_collision_models={}

def collision_model(original):
    import copy
    import numpy as np
    key=id(original)
    cached=_collision_models.get(key)
    if cached is not None and cached[0] is original:return cached[1]
    model=copy.copy(original)
    ids=np.flatnonzero(model.geom_group==1)
    model.geom_contype[ids]=1;model.geom_conaffinity[ids]=1
    _collision_models[key]=(original,model)
    while len(_collision_models)>1:_collision_models.pop(next(iter(_collision_models)))
    return model
