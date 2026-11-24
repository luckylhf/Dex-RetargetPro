"""Disk handoff from the Frax backbone to a separate repair process."""
import json
import shutil
from pathlib import Path

_REQUIRED = {"identity.json", "motion.pkl", "motion.json", "motion.initial.npz",
             "motion.diagnostics.npz", "motion.npz"}


def _artifacts(folder, continuation):
    files = _REQUIRED | {path.name for path in folder.glob("motion.*") if path.is_file()}
    if continuation:
        base = folder/"frax_baseline"
        for name in ("specialist_report.json","ground_settle.json","ground_settle_evidence.npz","cascade_segments.json"):
            if (folder/name).is_file():
                files.add(name)
        files |= {"refine_state.json", "avoidance_reference.npy", "frax_baseline/quality.json"}
        files |= {"frax_baseline/"+name for name in _REQUIRED if name.startswith("motion.")}
        files |= {"frax_baseline/"+path.name for path in base.glob("motion.*") if path.is_file()}
    return files


def _verify_committed_repair(folder, files):
    state = json.loads((folder/"refine_state.json").read_text())
    if (state.get("schema") != 3 or state.get("status") != "running"
            or state["output_sha256"] != files["motion.pkl"]
            or state["baseline_sha256"] != files["frax_baseline/motion.pkl"]
            or state["avoidance_reference_sha256"] != files["avoidance_reference.npy"]):
        raise ValueError("Repair checkpoint does not match its committed artifacts")
    metadata = json.loads((folder/"frax_baseline/motion.json").read_text())
    if (metadata["output_sha256"] != state["baseline_sha256"]
            or metadata["prepared_targets_sha256"] != state["prepared_targets_sha256"]):
        raise ValueError("Repair checkpoint baseline changed")
    return state


def seal(folder, identity, continuation=False):
    from .cli import atomic_json, sha
    files = {name: sha(folder/name) for name in sorted(_artifacts(folder, continuation))}
    if continuation:
        _verify_committed_repair(folder, files)
    marker = dict(schema=1, fingerprint=identity["fingerprint"], files=files)
    if continuation:
        marker["continuation"] = True
    atomic_json(folder/"initial_ready.json", marker)


def verify(folder, identity):
    from .cli import sha
    folder = Path(folder).resolve()
    marker = json.loads((folder/"initial_ready.json").read_text())
    stored = json.loads((folder/"identity.json").read_text())
    continuation = marker.get("continuation", False)
    if marker["schema"] != 1 or marker["fingerprint"] != stored["fingerprint"]:
        raise ValueError("Frax checkpoint identity changed")
    if stored != identity:
        if not continuation:
            # A sealed initial trajectory has no repair state and may seed either flow.
            def initial_contract(value):
                contract = dict(value['conversion_contract'])
                contract.pop('refine_flow', None)
                return {k: v for k, v in value.items() if k not in ('conversion_contract', 'fingerprint')}, contract

            if initial_contract(stored) != initial_contract(identity):
                raise ValueError("Frax checkpoint identity changed")
        else:
            from .managed_batch import scheduler_update
            fixed = ("converter", "version", "implementation", "frax_implementation", "assets", "runtime")
            scheduler_update({key: stored[key] for key in fixed}, {key: identity[key] for key in fixed})
            ignored = {"implementation", "fingerprint"}
            if {k:v for k,v in stored.items() if k not in ignored} != {k:v for k,v in identity.items() if k not in ignored}:
                raise ValueError("Repair checkpoint conversion parameters changed")
    if set(marker["files"]) != _artifacts(folder, continuation):
        raise ValueError("Missing Frax checkpoint artifact")
    for name, expected in marker["files"].items():
        path = folder/name
        if path.resolve() != path or not path.is_file() or sha(path) != expected:
            raise ValueError("Changed Frax checkpoint artifact: " + name)
    if continuation:
        _verify_committed_repair(folder, marker["files"])
    return marker


def restore(source, destination, target_hash, marker):
    directory = source/"frax_baseline" if marker.get("continuation") else source
    metadata = json.loads((directory/"motion.json").read_text())
    if metadata["prepared_targets_sha256"] != target_hash:
        raise ValueError("Frax checkpoint targets changed")
    for name in marker["files"]:
        file = source/name
        if file.parent == directory and file.name.startswith("motion."):
            shutil.copyfile(file, destination.parent/file.name)
    return metadata
