"""Independent BVH-to-Dex V8 route using the frozen cd1f9c8 implementation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", choices=("lafan", "soma", "mocap", "auto"), default="auto")
    parser.add_argument("--stop-after-support", action="store_true")
    args = parser.parse_args()
    release = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(release / "legacy" / "src"), PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, str(release / "legacy" / "run_v8.py"), str(args.input.resolve()),
               str(args.output.resolve()), "--source", args.source]
    if args.stop_after_support:
        command.append("--stop-after-support")
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
