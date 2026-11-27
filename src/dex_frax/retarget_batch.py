"""Run the 0.9.0 Frax/repair converter with capacity-based dispatch."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

from . import capacity_scheduler as scheduler, resources
from .cli import atomic_json, sha
from .contract import UNLIMITED_REPAIR_BUDGET
from .resource_events import set_gate

FRAX_PROFILE = dict(stage="frax", cores=1, fixed_bytes=math.ceil(2.57 * resources.GIB),
                    bytes_per_frame={"motion": math.ceil(13.4 * resources.MIB / 1024)})
DIAGNOSIS_BASE = math.ceil(2.25 * resources.GIB)
DIAGNOSIS_PER_FRAME = math.ceil(35.5 * 1024)
REPAIR_EXTRA = math.ceil(0.43 * resources.GIB)

REPAIR_DIMENSIONS = (
    "motion", "discontinuities", "root_angular", "jitter", "support", "body",
    "dense_floor", "torso", "body_collision", "leg_collision", "collision", "arms",
    "leg_pose", "segment_solvers", "ground_settle",
)


def host_growth_headroom(reserve_mib):
    snapshot = resources.memory()
    return min(snapshot["available"] - reserve_mib * resources.MIB, snapshot["cgroup_headroom"])


def worker_command(stage, motion, config, round_number=0, modules=()):
    return [sys.executable, "-m", "dex_frax.retarget_batch", "_worker",
            stage, motion, str(config), str(round_number), *modules]


def run_worker(stage, motion, config_path, round_number, modules):
    from . import cli, model_cache

    config = json.loads(config_path.read_text())
    set_gate(scheduler.worker_memory_gate)
    case = config["cases"][motion]
    if sha(case["input"]) != case["input_sha256"]:
        raise ValueError("Input changed after manifest validation")
    cli._FROZEN_IDENTITY = config["identity"]
    model_cache.enable(config["model"])
    folder = Path(config["motions_root"]) / motion
    initial = folder / "frax_initial"
    output = initial if stage == "frax" else folder / f"{stage}_{round_number:02d}"
    previous = folder / f"repair_{round_number - 1:02d}"
    recovered = config.get("recovered", {}).get(motion) if round_number == 1 else None
    args = argparse.Namespace(**config["conversion"], input=case["input"], source=case["source"],
                              mode="frax", profile="refine", unified_repair=True, max_frames=None,
                              ground_running=case.get("ground_running", False),
                              output=output, initial_only=stage == "frax",
                              initial_from=Path(recovered or initial) if round_number == 1 else None,
                              checkpoint_implementation=config["original_implementation"] if recovered else None,
                              continue_from=previous if round_number > 1 else None,
                              resume_same_round=round_number > 1,
                              repair_plan_only=stage == "diagnose",
                              repair_modules=frozenset(modules) if stage == "repair" else None)
    record = cli.convert(args)
    successor = None
    if stage == "frax":
        if record["status"] != "initial_ready":
            raise ValueError("Frax did not commit an initial checkpoint")
        successor = dict(profile="diagnose", frames={"motion": record["frames"]},
                         command=worker_command("diagnose", motion, config_path, 1))
    elif stage == "diagnose":
        if record["status"] != "repair_planned":
            raise ValueError("Repair diagnosis did not produce a workload")
        workload = record["workload"]
        if set(workload) - set(REPAIR_DIMENSIONS):
            raise ValueError("Repair plan contains an unbudgeted module")
        frames = {name: workload.get(name, 0) for name in REPAIR_DIMENSIONS}
        selected = [name for name, count in frames.items() if count and name != "motion"]
        successor = dict(profile="repair", frames=frames,
                         command=worker_command("repair", motion, config_path, round_number, selected))
    else:
        if record["status"] != "complete":
            raise ValueError("Repair did not finish its quality check")
        round_limit = config["repair_rounds"]
        if (record.get("quality") in {"review", "accepted_with_notes"}
                and (round_limit is None or round_number < round_limit)
                and not (output / "pre_final_retime").exists()):
            report = json.loads((output / "repair.json").read_text())
            retry_limited = (round_limit is not None and report["status"] in
                             {"pending_budget", "time_slice_review"})
            untried_module = any(job["module"] not in modules and job["next_recipe"]
                                 and not job.get("dependency") for job in report["pending"])
            if not report.get("shared_budget_exhausted") and (retry_limited or untried_module):
                successor = dict(profile="diagnose", frames={"motion": record["frames"]},
                                 command=worker_command("diagnose", motion, config_path, round_number + 1))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    atomic_json(Path(os.environ["DEX_SCHEDULE_RESULT"]), {
        "next": successor, "output": str(output), "record_status": record["status"],
        "quality": record.get("quality"), "frames": record["frames"], "workload": record.get("workload"),
        "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
    })


def add_options(parser):
    parser.add_argument("--system-reserve-mib", type=int, default=1024)
    parser.add_argument("--repair-rounds", type=int, help="Optional repair pass limit for capacity scheduling")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--recover-from", type=Path, help="Recover the stopped 1.0.3 capacity run")
    parser.add_argument("--recovery-pilot", help="Hold other recovery jobs until the pilot is verified")


def batch(args, parser):
    if args.profile != "refine" or args.resume:
        parser.error("Capacity scheduling requires --mode refine and a new output directory")
    if args.resource_policy or args.memory_limit_gib is not None:
        parser.error("Capacity scheduling uses fixed resource profiles and the current cgroup limit")
    if args.repair_rounds is None and (
        args.recover_from or args.repair_budget not in (None, UNLIMITED_REPAIR_BUDGET)
    ):
        args.repair_rounds = 2
    numbers = [args.repair_rounds, args.iterations, args.initial_iterations, args.ik_block_frames,
               args.budget]
    if args.repair_budget not in (None, UNLIMITED_REPAIR_BUDGET):
        numbers.append(args.repair_budget)
    if any(not math.isfinite(value) or value <= 0 for value in numbers if value is not None):
        parser.error("budgets, round limits, and iteration counts must be positive and finite")
    if args.recovery_pilot and not args.recover_from:
        parser.error("--recovery-pilot requires --recover-from")
    if args.plan_only:
        return run(args, parser)
    with resources.batch_lock():
        return run(args, parser)


def run(args, parser):
    from .cascade import load_manifest

    began = time.monotonic()
    manifest = args.manifest.resolve()
    contents = manifest.read_bytes()
    cases = {case["id"]: case for case in load_manifest(manifest, contents)}
    output = args.output.resolve()
    config_path = output / "capacity_config.json"
    profiles = {
        "frax": FRAX_PROFILE,
        "diagnose": dict(stage="repair", cores=1, fixed_bytes=DIAGNOSIS_BASE,
                         bytes_per_frame={"motion": DIAGNOSIS_PER_FRAME}),
        "repair": dict(stage="repair", cores=1, fixed_bytes=DIAGNOSIS_BASE + REPAIR_EXTRA,
                       bytes_per_frame={**dict.fromkeys(REPAIR_DIMENSIONS, 0),
                                        "motion": DIAGNOSIS_PER_FRAME}),
    }
    conversion = {key: getattr(args, key) for key in (
        "iterations", "initial_iterations", "ik_block_frames", "bvh_units", "bvh_translations",
        "foot_contact", "allow_local_slowdown", "repair_budget", "refine_flow",
    )}
    frame_counts = {motion: resources.output_frames(case["input"]) for motion, case in cases.items()}
    snapshots = {}
    if args.recover_from:
        from . import capacity_recovery
        original_identity,snapshots = capacity_recovery.inspect(
            args.recover_from.resolve(), cases, conversion, args.repair_rounds)
    if args.recovery_pilot and snapshots.get(args.recovery_pilot, {}).get("kind") != "repair":
        parser.error("The recovery pilot needs a verified repair checkpoint")
    jobs = []
    for motion,frames in sorted(frame_counts.items(), key=lambda item: item[1], reverse=True):
        snapshot = snapshots.get(motion)
        if snapshot is None:
            job = dict(profile="frax",frames={"motion":frames},
                       command=worker_command("frax",motion,config_path))
            jobs.append(scheduler.Job.parse(job,profiles,motion,"frax",0))
        elif "workload" in snapshot:
            workload = snapshot["workload"]
            modules = [name for name in REPAIR_DIMENSIONS if name != "motion" and workload.get(name,0)]
            job = dict(profile="repair",frames={name:workload.get(name,0) for name in REPAIR_DIMENSIONS},
                       command=worker_command("repair",motion,config_path,1,modules))
            jobs.append(scheduler.Job.parse(job,profiles,motion,"repair",2))
        else:
            job=dict(profile="diagnose",frames={"motion":snapshot["frames"]},
                     command=worker_command("diagnose",motion,config_path,1))
            jobs.append(scheduler.Job.parse(job,profiles,motion,"repair",1))
    jobs.sort(key=lambda job: job.round,reverse=True)
    reserve = args.system_reserve_mib
    if args.memory_reserve_gib is not None:
        reserve = max(reserve, math.ceil(args.memory_reserve_gib * 1024))
    capacity, snapshot = scheduler.host_capacity(reserve, resources.workers(args.workers))
    plan = dict(host=snapshot, capacity=vars(capacity), actions=len(jobs), frames=sum(frame_counts.values()),
                profiles=profiles, initial=[dict(motion=job.motion, frames=frame_counts[job.motion],
                                                stage=job.stage, dispatch_round=job.round,
                                                **vars(job.budget)) for job in jobs])
    if args.recover_from:
        plan["recovery"] = dict(previous=str(args.recover_from.resolve()),
                                pilot=args.recovery_pilot, snapshots=snapshots)
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if any(job.budget.memory_bytes > capacity.memory_bytes for job in jobs):
        parser.error("At least one Frax budget exceeds host capacity")
    if shutil.disk_usage(output.parent).free < args.disk_reserve_gib * resources.GIB:
        parser.error("Available output disk space is below --disk-reserve-gib")
    from .cli import static_identity
    from .model_cache import prepare

    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "capacity_plan.json", plan)
    state = resources.state_root()
    cache = state / "jax_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
                      DEX_FRAX_KERNEL_CACHE_SIZE="2", DEX_FRAX_COMPILE_LOCK_DIR=str(state / "compiler"),
                      JAX_COMPILATION_CACHE_DIR=str(cache),
                      JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0")
    os.environ["DEX_FRAX_BLOCK_MEMORY"] = "1"
    identity = static_identity()
    model = prepare(identity["assets"], state / "models")
    resources.release_memory()
    config=dict(cases=cases, conversion=conversion, model=model, identity=identity,
                repair_rounds=args.repair_rounds, motions_root=str(output / "motions"))
    if args.recover_from:
        capacity_recovery.check_identity(original_identity,identity)
        config.update(recovered=capacity_recovery.stage_checkpoints(output,snapshots),
                      original_implementation=original_identity["implementation"])
    atomic_json(config_path,config)
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    attempts = output / "scheduler"
    deadline = math.inf if args.budget is None else began + args.budget
    max_dispatch_rounds = None if args.repair_rounds is None else 2 * args.repair_rounds
    first = [job for job in jobs if job.motion==args.recovery_pilot] if args.recovery_pilot else jobs
    held = [job for job in jobs if job.motion!=args.recovery_pilot] if args.recovery_pilot else ()
    try:
        outcomes = scheduler.schedule(first, profiles, capacity, attempts, max_dispatch_rounds,
                                      stop_requested=lambda: stopping or time.monotonic() >= deadline,
                                      job_timeout=args.frax_timeout,
                                      growth_reserve=256 * resources.MIB,
                                      growth_headroom=lambda: host_growth_headroom(reserve),
                                      prioritize_repair=True,
                                      deferred=held, release_gate=(output/"release_remaining.json").is_file)
    except KeyboardInterrupt:
        atomic_json(output / "batch.json", dict(status="interrupted" if stopping else "budget_exhausted",
                                                actions=len(jobs)))
        return 130
    results = {}
    for receipt in sorted(attempts.glob("*/result.json")):
        result = json.loads(receipt.read_text())
        if result["record_status"] == "complete":
            results[Path(result["output"]).parent.name] = result
    records = [
        dict(motion=motion, status=results[motion]["quality"], output=results[motion]["output"])
        if status == "complete" else dict(motion=motion, status="failed", error=status)
        for motion, status in outcomes.items()
    ]
    complete = len(outcomes)==len(jobs) and all(status == "complete" for status in outcomes.values())
    atomic_json(output / "batch.json", dict(status="complete" if complete else "failed",
                                            actions=len(jobs), elapsed_s=time.monotonic() - began,
                                            records=records))
    return int(not complete)


def main():
    _, entry, stage, motion, config, round_number, *modules = sys.argv
    if entry != "_worker":
        raise ValueError("Use dex_frax batch --capacity-scheduler")
    run_worker(stage, motion, Path(config), int(round_number), modules)


if __name__ == "__main__":
    raise SystemExit(main())
