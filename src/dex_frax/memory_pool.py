"""Admission for one host's active Retarget Pro workers."""
from dataclasses import dataclass

from .resources import GIB, MIB

BASE = 2816 * MIB
DANGER, RECOVER, START = 8 * GIB, 10 * GIB, 12 * GIB


@dataclass
class Lease:
    grant: int = BASE
    resident: int = 0

    @property
    def unused(self):
        return max(0, self.grant - self.resident)

    def return_unused(self):
        before = self.grant
        self.grant = min(before, max(BASE, self.resident))
        return before - self.grant


class Pool:
    def __init__(self):
        self.frozen = False
        self.recovery_samples = 0

    @staticmethod
    def remaining(snapshot, leases):
        committed = sum(lease.unused for lease in leases)
        return snapshot['available'] - committed, snapshot['cgroup_headroom'] - committed

    def update(self, snapshot, leases, pressure=False):
        host, group = self.remaining(snapshot, leases)
        if pressure or host < DANGER or group < 256 * MIB:
            self.frozen = True
            self.recovery_samples = 0
        elif self.frozen:
            self.recovery_samples = self.recovery_samples + 1 if host >= RECOVER and group >= 768 * MIB else 0
            if self.recovery_samples >= 3:
                self.frozen = False
                self.recovery_samples = 0
        return host, group

    def allows(self, snapshot, leases, amount, *, start=False):
        host, group = self.remaining(snapshot, leases)
        host_floor = START if start else DANGER
        group_floor = GIB if start else 256 * MIB
        return not self.frozen and host - amount >= host_floor and group - amount >= group_floor


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
