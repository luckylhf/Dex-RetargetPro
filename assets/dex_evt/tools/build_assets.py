"""Import the upstream Dex models and meshes without rewriting their contents."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

URDF = Path('urdf/tiangong3.urdf')
MJCF = Path('mujoco/tiangong3_torq.xml')
VISUAL_MJCF = Path('mujoco/tiangong3_simplify.xml')
HEAD_JOINTS = ('head_yaw_joint', 'head_pitch_joint')
MESH_DIRS = ('meshes', 'mujoco/meshes_convex', 'mujoco/meshes_simplify')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_model(source, target, revision):
    urdf = ET.parse(source / URDF)
    mjcf = ET.parse(source / MJCF)
    joints = {joint.attrib['name']: joint.find('limit').attrib
              for joint in urdf.findall('joint') if joint.get('type') == 'revolute'}
    hinges = {joint.attrib['name'] for joint in mjcf.findall('.//worldbody//joint')
              if joint.get('type') == 'hinge'}
    motors = [motor.attrib['joint'] for motor in mjcf.findall('./actuator/motor')]
    if len(joints) != 31 or joints.keys() != hinges or len(motors) != 31 or set(motors) != hinges:
        raise ValueError('The original URDF, MJCF and actuators must agree on the 31 axes')

    for relative in (URDF, MJCF, VISUAL_MJCF, Path('LICENSE')):
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, path)
    for directory in MESH_DIRS:
        for mesh in sorted((source / directory).glob('*.STL')):
            path = target / directory / mesh.name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.is_file() or sha256(path) != sha256(mesh):
                shutil.copy2(mesh, path)

    order = [name for name in joints if name not in HEAD_JOINTS] + list(HEAD_JOINTS)
    manifest = dict(robot='tienkung_dex_31', head='actuated_yaw_pitch', hands=False,
                    source='Tienkung_Dex_Model_All', source_revision=revision,
                    model_sha256=sha256(target / MJCF), urdf_sha256=sha256(target / URDF),
                    joint_names=order, joint_limits={name: joints[name] for name in order})
    (target / 'model.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--revision', help='Upstream revision when importing from an archive')
    args = parser.parse_args()
    revision = args.revision or subprocess.check_output(
        ['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    import_model(args.source.resolve(), args.target.resolve(), revision)
