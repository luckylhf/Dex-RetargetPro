"""Apply current window repairs to a verified independent V8 checkpoint."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .cli import convert, sha
from .robot_profile import MODEL_XML


class V8Seed:
    strict = True
    improved = False
    legacy_v8_seed = True

    def __init__(self, path, identity):
        self.path = Path(path)
        self.identity = identity
        self.report = {"solver": "v8_trf", "checkpoint_sha256": identity["v8_seed_sha256"]}

    def __call__(self, robot, targets, cfg, progress):
        if sha(self.path) != self.report["checkpoint_sha256"]:
            raise ValueError("V8 checkpoint changed after identity was calculated")
        with np.load(self.path, allow_pickle=False) as z:
            if str(z["input_sha256"]) != self.identity["input_sha256"]:
                raise ValueError("V8 checkpoint input differs")
            if str(z["model_sha256"]) != self.identity["assets"][MODEL_XML]:
                raise ValueError("V8 checkpoint model differs")
            if list(z["joint_names"]) != robot.names or str(z["quat_order"]) != "xyzw":
                raise ValueError("V8 checkpoint joint or quaternion convention differs")
            n = len(targets["root_p"])
            rows = np.c_[z["root_pos"], z["dof_pos"]]
            quat = z["root_rot"]
            if float(z["fps"]) != float(targets["fps"]) or rows.shape != (n, 32) or quat.shape != (n, 4):
                raise ValueError("V8 checkpoint timeline or shape differs")
            if not np.isfinite(rows).all() or not np.isfinite(quat).all():
                raise ValueError("V8 checkpoint contains nonfinite data")
            self.root_matrices = Rotation.from_quat(quat).as_matrix()
        # TRF does not produce Frax QP records. No steps were replayed by Frax.
        return rows, np.zeros((n, 3), dtype=np.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("seed", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--repair-budget", type=float, default=180.)
    p.add_argument("--foot-contact", choices=("v1", "v2"), default="v1")
    opts = p.parse_args()
    if not np.isfinite(opts.repair_budget) or opts.repair_budget <= 0:
        p.error("--repair-budget must be positive and finite")
    convert(argparse.Namespace(input=opts.input, output=opts.output, v8_seed=opts.seed,
                               source="lafan", mode="v8", profile="refine",
                               max_frames=None, iterations=8, initial_iterations=100,
                               foot_contact=opts.foot_contact, pipeline="unified",
                               unified_repair=True, repair_budget=opts.repair_budget,
                               allow_local_slowdown=False))


if __name__ == "__main__":
    main()
