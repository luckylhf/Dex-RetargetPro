"""Experimental batch scheduler for isolated, budgeted Retarget commands."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from itertools import islice
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from . import resources


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"expected integer >= {minimum}, got {value!r}")
    return value


@dataclass(frozen=True)
class Budget:
    cores: int
    memory_bytes: int


@dataclass(frozen=True)
class Job:
    motion: str
    stage: str
    round: int
    command: tuple
    budget: Budget

    @classmethod
    def parse(cls, spec, profiles, motion, stage, round_number):
        profile = profiles[spec["profile"]]
        if profile["stage"] != stage:
            raise ValueError(f"profile does not match stage {stage}")
        frames = spec["frames"]
        fixed = integer(profile["fixed_bytes"], 1)
        slopes = profile["bytes_per_frame"]
        if not slopes or frames.keys() != slopes.keys():
            raise ValueError("frame counts must match the profile's frame dimensions")
        peak = fixed + sum(integer(frames[key]) * integer(slope) for key, slope in slopes.items())
        command = spec["command"]
        if not isinstance(command, list) or not command or not all(isinstance(arg, str) for arg in command):
            raise ValueError("command must be a nonempty argument list")
        return cls(motion, stage, round_number, tuple(command), Budget(integer(profile["cores"], 1), peak))


@dataclass
class Running:
    job: Job
    process: subprocess.Popen
    cores: list
    result: Path
    started: float
    extra: int = 0


GROWTH_STEP = 128 * resources.MIB


def publish_memory(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(str(value))
    temp.replace(path)


def worker_memory_gate(event):
    """Request an increase before entering a block or phase with low headroom."""
    budget = int(os.environ["DEX_SCHEDULE_MEMORY_BYTES"])
    required = max(event["pss_bytes"] + GROWTH_STEP // 2,
                   event.get("required_peak_bytes", 0))
    if required <= budget:
        return
    folder = Path(os.environ["DEX_SCHEDULE_RESULT"]).parent
    amount = ((required - budget + GROWTH_STEP - 1) // GROWTH_STEP) * GROWTH_STEP
    publish_memory(folder / "memory_request", amount)
    grant = folder / "memory_grant"
    while not grant.exists():
        time.sleep(0.02)
    granted = int(grant.read_text())
    grant.unlink()
    if granted != amount:
        raise MemoryError("Scheduler cannot grant the requested phase memory")
    os.environ["DEX_SCHEDULE_MEMORY_BYTES"] = str(budget + granted)


def host_capacity(reserve_mib, cpu_limit=None):
    snapshot = resources.memory()
    cores = resources.cpu_count()
    if cpu_limit is not None:
        cores = min(cores, integer(cpu_limit, 1))
    usable = min(snapshot["available"] - integer(reserve_mib) * resources.MIB,
                 snapshot["cgroup_headroom"])
    return Budget(cores, max(0, usable)), snapshot


def launch(job, cpus, folder):
    folder.mkdir()
    result = (folder / "result.json").resolve()
    env = dict(os.environ, DEX_SCHEDULE_RESULT=str(result),
               DEX_SCHEDULE_MEMORY_BYTES=str(job.budget.memory_bytes))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = str(job.budget.cores)
    command = job.command
    if hasattr(os, "sched_getaffinity"):
        taskset = shutil.which("taskset")
        if not taskset:
            raise RuntimeError("taskset is required for Linux CPU affinity")
        command = (taskset, "-c", ",".join(map(str, cpus)), *command)
    with (folder / "worker.log").open("wb") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    return Running(job, process, cpus, result, time.monotonic())


def kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def schedule(jobs, profiles, capacity, output, max_repair_rounds=4, stop_requested=lambda: False,
             job_timeout=None, prioritize_repair=False, growth_reserve=0, growth_headroom=None,
             deferred=(), release_gate=None):
    """Run commands to exit before returning their tokens and queuing successors."""
    integer(max_repair_rounds, 1)
    pending = deque(jobs)
    deferred = tuple(deferred)
    if deferred and release_gate is None:
        raise ValueError('Deferred jobs require a release gate')
    cores = resources.cpu_order()[:capacity.cores]
    if not cores or len(cores) != capacity.cores:
        raise ValueError("invalid CPU capacity")
    output.mkdir(parents=True, exist_ok=False)
    active = {}
    used = 0
    attempts = 0
    skipped = 0
    outcomes = {}
    with (output / "events.jsonl").open("w") as log:
        def event(kind, job, **details):
            row = dict(event=kind, motion=job.motion, stage=job.stage, round=job.round,
                       cores=job.budget.cores, memory_bytes=job.budget.memory_bytes,
                       reserved_bytes=used, reserved_cores=capacity.cores - len(cores), **details)
            print(json.dumps(row, ensure_ascii=False), file=log, flush=True)

        try:
            while pending or active or deferred:
                if stop_requested():
                    raise KeyboardInterrupt
                if deferred and release_gate():
                    pending.extend(deferred)
                    deferred = ()
                if not pending and not active:
                    if any(status != 'complete' for status in outcomes.values()):
                        break
                    time.sleep(.2)
                for pid, worker in list(active.items()):
                    code = worker.process.poll()
                    elapsed = time.monotonic() - worker.started
                    timed_out = code is None and job_timeout is not None and elapsed >= job_timeout
                    if code is None and not timed_out:
                        continue
                    kill_group(worker.process)
                    del active[pid]
                    job = worker.job
                    cores.extend(worker.cores)
                    used -= job.budget.memory_bytes + worker.extra
                    event("released", job, pid=pid, exit_code=worker.process.returncode,
                          elapsed_s=elapsed)
                    try:
                        if timed_out:
                            raise ValueError("worker exceeded configured timeout")
                        if code:
                            raise ValueError(f"worker exited with code {code}")
                        next_spec = json.loads(worker.result.read_text())["next"]
                        if next_spec is None:
                            outcomes[job.motion] = "complete"
                            continue
                        next_round = job.round + 1
                        if next_round > max_repair_rounds:
                            raise ValueError("repair round limit reached")
                        successor = Job.parse(next_spec, profiles, job.motion, "repair", next_round)
                        if prioritize_repair:
                            position = next((i for i, item in enumerate(pending)
                                             if item.stage == "frax"), len(pending))
                            pending.insert(position, successor)
                            skipped = 0
                        else:
                            pending.append(successor)
                        event("queued", successor)
                    except (OSError, ValueError, KeyError, TypeError, AttributeError) as err:
                        outcomes[job.motion] = str(err)
                        event("failed", job, reason=str(err))

                waiting = {}
                if growth_reserve:
                    for pid, worker in active.items():
                        request = worker.result.parent / "memory_request"
                        if request.exists():
                            waiting[pid] = (request, integer(int(request.read_text()), 1))
                    if waiting:
                        headroom = min(capacity.memory_bytes - used, growth_headroom())
                        for pid, (request, amount) in list(waiting.items()):
                            if amount > headroom:
                                continue
                            worker = active[pid]
                            request.unlink()
                            used += amount
                            headroom -= amount
                            worker.extra += amount
                            publish_memory(request.with_name("memory_grant"), amount)
                            event("memory_granted", worker.job, pid=pid, extra_bytes=amount)
                            del waiting[pid]
                        if waiting and len(waiting) == len(active):
                            pid = max(waiting, key=lambda key: waiting[key][1])
                            request, amount = waiting[pid]
                            request.unlink()
                            publish_memory(request.with_name("memory_grant"), 0)
                            event("memory_denied", active[pid].job, pid=pid, requested_bytes=amount)

                while pending and not waiting and not stop_requested():
                    head = pending[0]
                    head_budget = head.budget
                    if head_budget.cores > capacity.cores or head_budget.memory_bytes > capacity.memory_bytes:
                        pending.popleft()
                        skipped = 0
                        outcomes[head.motion] = "budget exceeds host capacity"
                        event("failed", head, reason=outcomes[head.motion])
                        continue
                    # Backfill a limited number of times before reserving capacity for the head.
                    candidates = islice(pending, 8) if skipped < 8 else [head]
                    room = capacity.memory_bytes - used - (growth_reserve if active else 0)
                    job = next((item for item in candidates if item.budget.cores <= len(cores)
                                and item.budget.memory_bytes <= room), None)
                    if job is None:
                        break
                    pending.remove(job)
                    skipped = 0 if job is head else skipped + 1
                    cpus = cores[:job.budget.cores]
                    attempts += 1
                    try:
                        worker = launch(job, cpus, output / f"{attempts:06d}")
                    except (OSError, RuntimeError, ValueError) as err:
                        outcomes[job.motion] = str(err)
                        event("failed", job, reason=str(err))
                        continue
                    del cores[:job.budget.cores]
                    used += job.budget.memory_bytes
                    active[worker.process.pid] = worker
                    event("started", job, pid=worker.process.pid, cpu_ids=cpus)
                if active:
                    time.sleep(0.02)
        finally:
            for worker in active.values():
                kill_group(worker.process)
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--system-reserve-mib", type=int, default=1024)
    parser.add_argument("--cpu-limit", type=int)
    parser.add_argument("--max-repair-rounds", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    profiles = plan["profiles"]
    jobs = [Job.parse(spec, profiles, motion, "frax", 0) for motion, spec in plan["motions"].items()]
    capacity, snapshot = host_capacity(args.system_reserve_mib, args.cpu_limit)
    inventory = dict(host=snapshot, capacity=vars(capacity), motion_count=len(jobs),
                     motions=[
        dict(motion=job.motion, frames=plan["motions"][job.motion]["frames"], **vars(job.budget))
        for job in jobs
    ])
    print(json.dumps(inventory, ensure_ascii=False, indent=2), flush=True)
    if args.plan_only:
        return 0
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        outcomes = schedule(jobs, profiles, capacity, args.output, args.max_repair_rounds,
                            stop_requested=lambda: stopping)
    except KeyboardInterrupt:
        return 130
    (args.output / "outcomes.json").write_text(json.dumps(outcomes, ensure_ascii=False, indent=2))
    return int(any(status != "complete" for status in outcomes.values()))


if __name__ == "__main__":
    raise SystemExit(main())
