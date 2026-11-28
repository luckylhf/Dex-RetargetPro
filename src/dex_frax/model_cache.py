"""Verified binary model cache; spawn workers reuse models, never fork JAX."""
import hashlib
import json
import fcntl
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from dex_frax.robot_profile import MODEL_XML, PELVIS_BODY

_model=None
_xml=None
_audit_models=None


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


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _checked_binary(info):
    binary = Path(info['binary'])
    if _digest(binary) != info['sha256']:
        raise ValueError('Cached MJB hash mismatch: '+str(binary))
    if _key(info['metadata']) != info['metadata_sha256']:
        raise ValueError('Cached MJB metadata hash mismatch: '+str(binary))
    return str(binary)


def _prepare_binary(folder, build):
    import mujoco
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    binary, receipt = folder/'model.mjb', folder/'receipt.json'
    with (folder/'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.is_file():
            info = dict(json.loads(receipt.read_text()), binary=str(binary))
            _checked_binary(info)
            return dict(info, reused=True)
        original, metadata = build()
        compared = []
        with TemporaryDirectory(dir=folder) as directory:
            temporary = Path(directory)/binary.name
            mujoco.mj_saveModel(original, str(temporary))
            restored = mujoco.MjModel.from_binary_path(str(temporary))
            if original.names != restored.names:
                raise ValueError('XML/MJB object name mismatch')
            for name in dir(original):
                value = getattr(original, name)
                if isinstance(value, np.ndarray):
                    if not np.array_equal(value, getattr(restored, name)):
                        raise ValueError('XML/MJB mismatch: '+name)
                    compared.append(name)
            info = dict(sha256=_digest(temporary), arrays_compared=compared,
                        metadata=metadata, metadata_sha256=_key(metadata))
            staged_receipt = Path(directory)/receipt.name
            staged_receipt.write_text(json.dumps(info, indent=2)+'\n')
            temporary.replace(binary)
            staged_receipt.replace(receipt)
        return dict(info, binary=str(binary), reused=False)


def prepare(assets, cache):
    import mujoco
    from .paths import ASSETS

    xml = ASSETS/MODEL_XML
    key = _key(dict(assets=assets, mujoco=mujoco.__version__, mapping=_digest(__file__),
                    platform=platform.system(), machine=platform.machine()))
    info = _prepare_binary(Path(cache)/key, lambda: (model_spec(xml).compile(), {}))
    return dict(info, xml=str(xml))


def prepare_audit_models(assets, cache):
    import numpy as np
    import scipy
    from .foot_collision import build_augmented
    from .v8.robot import Robot

    base = prepare(assets, cache)
    root = Path(__file__).parent
    code = {name: _digest(root/name) for name in ('foot_collision.py', 'robot_profile.py', 'v8/robot.py')}
    key = _key(dict(assets=assets, base_key=Path(base['binary']).parent.name, base_sha256=base['sha256'],
                    code=code, numpy=np.__version__, scipy=scipy.__version__))

    def build():
        global _model, _xml, _audit_models
        previous = _model, _xml, _audit_models
        try:
            enable(base)
            return build_augmented(Robot(base['xml']))
        finally:
            _model, _xml, _audit_models = previous

    return dict(base=base, collision=_prepare_binary(Path(cache)/key, build))

def enable(info):
    import mujoco
    global _model, _xml, _audit_models
    _model = mujoco.MjModel.from_binary_path(_checked_binary(info))
    _xml = str(Path(info['xml']).resolve())
    _audit_models = None


def enable_audit_models(models):
    global _audit_models
    enable(models['base'])
    _audit_models = models


def audit_models():
    return _audit_models


def load_full_collision(robot):
    if _audit_models is None or robot.m is not _model:
        return None
    import mujoco

    info = _audit_models['collision']
    return mujoco.MjModel.from_binary_path(_checked_binary(info)), info['metadata']

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
