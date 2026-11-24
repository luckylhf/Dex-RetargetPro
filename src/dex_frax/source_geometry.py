"""Resolve user-supplied reference surfaces and identify the bytes in use."""
from pathlib import Path
import json
import os

from .input_errors import MotionInputError
from .paths import ROOT


REFERENCE = Path(__file__).parent / 'v8' / 'reference'
SURFACES = {'lafan': 'lafan1_surfaces.json', 'soma': 'soma_surfaces.json'}


def surface_path(source):
    override = os.environ.get('DEX_FRAX_SOURCE_GEOMETRY')
    name = SURFACES[source]
    if override:
        return Path(override).expanduser() / name
    local = ROOT / '.local-data' / 'source_geometry' / name
    return local if local.is_file() else REFERENCE / name


def load_surface(source):
    path = surface_path(source)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise MotionInputError(
            f'{source} reference geometry could not be read from {path}. '
            f'Set DEX_FRAX_SOURCE_GEOMETRY to the directory containing {path.name}.'
        ) from error
