"""Joint CPU and memory admission for one host's Retarget Pro workers."""
from contextlib import contextmanager
from dataclasses import dataclass

from .resources import GIB, MIB

BASE = 2816 * MIB
DANGER, RECOVER, START = 8 * GIB, 10 * GIB, 12 * GIB


@dataclass
class Lease:
    grant: int = BASE
    resident: int = 0
    cpu: int | None = None
    borrowed_cpus: tuple[int, ...] = ()
    phase_grant: int | None = None

    @property
    def unused(self):
        return max(0, self.grant - self.resident)

    def return_unused(self):
        before = self.grant
        self.grant = min(before, max(BASE, self.resident))
        return before - self.grant

    def borrow(self, cpus, workspace_bytes):
        self.phase_grant = self.grant
        self.grant = max(self.grant, self.resident) + workspace_bytes
        self.borrowed_cpus = cpus

    def return_workspace(self):
        unused_before = self.unused
        self.grant = self.phase_grant
        self.phase_grant = None
        self.borrowed_cpus = ()
        return unused_before - self.unused


class Pool:
    def __init__(self, watermarks=(DANGER, RECOVER, START)):
        self.danger, self.recover, self.start = watermarks
        self.frozen = False
        self.recovery_samples = 0

    @staticmethod
    def remaining(snapshot, leases):
        committed = sum(lease.unused for lease in leases)
        return snapshot['available'] - committed, snapshot['cgroup_headroom'] - committed

    def update(self, snapshot, leases, pressure=False):
        host, group = self.remaining(snapshot, leases)
        if pressure or host < self.danger or group < 256 * MIB:
            self.frozen = True
            self.recovery_samples = 0
        elif self.frozen:
            self.recovery_samples = self.recovery_samples + 1 if host >= self.recover and group >= 768 * MIB else 0
            if self.recovery_samples >= 3:
                self.frozen = False
                self.recovery_samples = 0
        return host, group

    def allows(self, snapshot, leases, amount, *, start=False):
        host, group = self.remaining(snapshot, leases)
        host_floor = self.start if start else self.danger
        group_floor = GIB if start else 256 * MIB
        return not self.frozen and host - amount >= host_floor and group - amount >= group_floor


class JointPool(Pool):
    def __init__(self, cpu_groups, watermarks=(DANGER, RECOVER, START)):
        super().__init__(watermarks)
        self.cpu_groups = cpu_groups

    def free_cpus(self, leases, limit, *, physical_only=False):
        occupied = {lease.cpu for lease in leases if lease.cpu is not None}
        occupied.update(cpu for lease in leases for cpu in lease.borrowed_cpus)
        free = [core[0] for core in self.cpu_groups if occupied.isdisjoint(core)]
        if not physical_only:
            placed = occupied | set(free)
            free.extend(cpu for core in self.cpu_groups for cpu in core if cpu not in placed)
        return free[:max(0, limit - len(occupied))]

    def startup_cpu(self, snapshot, leases, limit):
        if self.allows(snapshot, leases, BASE, start=True):
            return next(iter(self.free_cpus(leases, limit)), None)
        return None

    def workspace_cpus(self, snapshot, leases, limit, lease, *, cores, workspace_bytes, cgroup_headroom):
        # Optional work uses the stricter startup reserve and only whole idle physical cores.
        if (lease.borrowed_cpus or cores < 1 or workspace_bytes <= 0
                or cgroup_headroom - lease.unused - workspace_bytes < GIB
                or not self.allows(snapshot, leases, workspace_bytes, start=True)):
            return ()
        free = self.free_cpus(leases, limit, physical_only=True)
        return tuple(free[:cores]) if len(free) >= cores else ()


_workspace_gate = None


def set_workspace_gate(gate):
    global _workspace_gate
    _workspace_gate = gate


@contextmanager
def operator_workspace(cores, workspace_bytes):
    """Get CPU IDs for helpers or an empty tuple for the serial path.

    Call from the worker's main thread at a safe boundary. Keep helpers in
    its process group, bind where supported, and join before leaving this scope.
    """
    gate = _workspace_gate
    cpus = gate(cores, workspace_bytes) if gate is not None else ()
    try:
        yield cpus
    finally:
        if cpus:
            gate()


def requested_total(event, frames):
    resident = event['pss_bytes']
    if event.get('required_peak_bytes') is not None:
        return max(BASE, event['required_peak_bytes'])
    phase = event['phase']
    if phase == 'loading':
        return BASE
    if phase == 'waiting_compile' and event.get('compile_possible'):
        return max(BASE, 3840 * MIB, resident + 512 * MIB)
    # These checkpoints precede native calls with no workspace query. The
    # allowance is for the upcoming phase, and is dropped at the next gate.
    workspace = 96 * MIB
    if phase in ('preprocessing', 'initial', 'export_check', 'local_repair', 'audit'):
        workspace += frames * 32 * 1024
    return max(BASE, resident + workspace)
