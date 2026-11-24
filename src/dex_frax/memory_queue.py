"""Queue and reduce owned work under pressure without changing conversion semantics."""
import time
from .resources import GIB, MIB, cgroup_floor


class MemoryQueue:
    def __init__(self, ceiling, previous=None):
        previous = previous or {}
        self.ceiling = ceiling
        self.limit = min(ceiling, max(1, previous.get('dispatch_limit', ceiling)))
        self.floors = dict(previous.get('base_floor_bytes', {}))
        self.retries = previous.get('requeued_attempts', 0)
        self.recovered = 0
        self.stable_since = None
        self.timed_recoveries = 0
        self.stage_floors = dict(previous.get('stage_floor_bytes', {}))

    @staticmethod
    def key(mode, source):
        return mode + '|' + source

    def peak(self, mode, source, frames, per_frame, estimate, stage=None):
        # A whole-job backoff does not identify which safe stage needs RAM.
        # Apply it at job admission; stage gates use their own measured envelope.
        key = self.key(mode, source)
        floor = (self.floors.get(key, 0) if stage is None else
                 self.stage_floors.get(key+'|'+stage, 0))
        return max(estimate, floor + frames * per_frame if floor else 0)

    def defer(self, mode, source, frames, per_frame, peak, active_count, stage=None, underestimated=True):
        # A failed estimate must not immediately admit the same work again.
        guarded = int(peak * 1.25) + 128 * MIB
        key = self.key(mode, source)
        if underestimated:
            self.floors[key] = max(self.floors.get(key, 0), guarded - frames * per_frame)
            if stage is not None:
                stage_key = key+'|'+stage
                self.stage_floors[stage_key] = max(self.stage_floors.get(stage_key, 0), guarded-frames*per_frame)
        self.stable_since = None
        self.limit = max(1, min(self.limit, active_count - 1))
        self.retries += 1
        self.recovered = 0

    def complete(self):
        self.recovered += 1
        if self.recovered >= 2:
            self.limit = min(self.ceiling, self.limit + 1)
            self.recovered = 0

    def recover(self, snapshot, reserve, active_count, waiting=False,
                running=(), shared_reserve=0, now=None):
        """Recover one slot per ten stable seconds; every start is still admitted."""
        now = time.monotonic() if now is None else now
        headroom = min(snapshot['available']-reserve, snapshot['cgroup_headroom'])
        unspent = sum(max(0, peak-used) for peak, used in running)
        safe = (not waiting and not pressure(snapshot, reserve)[0]
                and headroom-unspent-shared_reserve >= GIB)
        if not safe or active_count < self.limit or self.limit >= self.ceiling:
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = now
        if now-self.stable_since < 10.:
            return False
        self.limit = min(self.ceiling, self.limit+1)
        self.stable_since = now
        self.timed_recoveries += 1
        return True

    def report(self):
        return dict(policy='queue_reclaim_requeue_v2', dispatch_limit=self.limit,
                    requeued_attempts=self.retries, base_floor_bytes=dict(self.floors),
                    stage_floor_bytes=dict(self.stage_floors),
                    recovery_interval_s=10., timed_recoveries=self.timed_recoveries)


def pressure(snapshot, reserve):
    """Soft pressure stops dispatch; critical pressure requires releasing memory."""
    critical = (snapshot['available'] < reserve // 2
                or snapshot['cgroup_headroom'] < cgroup_floor(snapshot))
    return critical, snapshot['available'] < reserve or critical


def victim(active, usage, protected=None):
    # Drain results first. Prefer restarting early work over discarding a nearly
    # exported motion; within a stage release the largest actual allocation.
    early = {'loading', 'preprocessing', 'initial'}
    candidates=[s for s in active if s is not protected] or list(active)
    return max(candidates, key=lambda slot: (
        getattr(slot, 'telemetry', {}).get('stage', 'initial') in early,
        usage.get(slot, 0), getattr(slot, 'started', 0)), default=None)
