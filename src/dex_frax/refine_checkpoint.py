"""Keep the motion and reference behind the last committed queue state."""
from pathlib import Path
import shutil
import tempfile

from .cli import atomic_json, sha


ARTIFACTS = {
    'motion.pkl': 'output_sha256',
    'avoidance_reference.npy': 'avoidance_reference_sha256',
}


def write(folder, state):
    root = folder / '.refine-checkpoints'
    root.mkdir(exist_ok=True)
    generation = state['output_sha256'] + '-' + state['avoidance_reference_sha256']
    saved = root / generation
    if not saved.is_dir():
        with tempfile.TemporaryDirectory(dir=root) as scratch:
            staging = Path(scratch) / 'snapshot'
            staging.mkdir()
            for name, key in ARTIFACTS.items():
                shutil.copyfile(folder / name, staging / name)
                if sha(staging / name) != state[key]:
                    raise ValueError('Checkpoint artifact changed while publishing: ' + name)
            staging.rename(saved)
    atomic_json(folder / 'refine_state.json', dict(state, checkpoint_generation=generation))
    for previous in root.iterdir():
        if previous != saved and previous.is_dir() and len(previous.name) == len(generation):
            shutil.rmtree(previous, ignore_errors=True)


def artifact(folder, state, name):
    """Restore from a published snapshot if an interrupted candidate changed the live file."""
    expected = state[ARTIFACTS[name]]
    path = folder / name
    if path.is_file() and sha(path) == expected:
        return path
    generation = state.get('checkpoint_generation')
    if state.get('status') == 'running' and generation is not None:
        expected_generation = state['output_sha256'] + '-' + state['avoidance_reference_sha256']
        if generation == expected_generation and all(c in '0123456789abcdef-' for c in generation):
            saved = folder / '.refine-checkpoints' / generation / name
            if saved.is_file() and sha(saved) == expected:
                return saved
    raise ValueError('Committed checkpoint artifact changed: ' + name)
