"""Measured resident admission and incremental permits for Frax workers."""
import time

from . import resources as res


class InitialMemory:
    MIN_RESIDENT_BYTES = res.GIB
    PROGRESS_MARGIN = 512 * res.MIB
    COMPILE_PHASES = {'waiting_compile', 'ik_prepare', 'ik_compile'}

    def __init__(self):
        self.owner = None
        self.compiler = None
        self.private_bytes = 0
        self.compiler_bytes = 0
        self.measured_base = 0
        self.launch_estimate = 0
        self.launch_margin = 0
        self.unsampled_workers = 0
        self.permits = 0
        self.waiting = set()
        self.memory_waiting = set()
        self.stalled_since = None

    def elect(self, initial, usage):
        if self.owner not in initial:
            # Advance the oldest unfinished job whenever the current owner exits.
            self.owner = next(iter(initial), None)
            self.stalled_since = None
        if (self.compiler not in initial or
                self.compiler.telemetry.get('phase') not in self.COMPILE_PHASES):
            self.compiler = None
        for slot in initial:
            if slot is not self.compiler:
                self.private_bytes = max(self.private_bytes, usage.get(slot, 0) + 128*res.MIB)
        self.waiting.clear()
        self.memory_waiting.clear()

    def reservations(self, initial, usage, exclude=None, *, compiling=False):
        """Charge only live permits and the one completion allowance."""
        running = []
        for slot, item in initial.items():
            if slot is exclude:
                continue
            peak = max(item['grant'], self.private_bytes) if slot is self.owner else item['grant']
            running.append((peak, usage.get(slot, 0)))
        if self.compiler in initial:
            scratch_used = max(0, usage.get(self.compiler, 0) - initial[self.compiler]['grant'])
            running.append((self.compiler_bytes, scratch_used))
        elif compiling:
            running.append((self.compiler_bytes, 0))
        return running

    @property
    def bootstrap_limit(self):
        return 4 if self.measured_base else 2

    def measure_residency(self, initial, usage, lengths):
        for slot, item in initial.items():
            resident = usage.get(slot, 0)
            if slot is self.compiler or not resident or not getattr(slot, 'frames_completed', 0):
                continue
            item['resident_sampled'] = True
            sample = int(resident*1.15) + 128*res.MIB - lengths[item['index']]
            self.measured_base = max(self.measured_base, self.MIN_RESIDENT_BYTES, sample)
        self.unsampled_workers = sum(not item.get('resident_sampled') for item in initial.values())

    def launch_budget(self, initial, usage, lengths, index, fallback):
        base = self.measured_base or max(fallback,self.MIN_RESIDENT_BYTES)
        running = self.reservations(initial,usage)
        for pos, (slot, item) in enumerate(initial.items()):
            peak, used = running[pos]
            running[pos] = max(peak,base+lengths[item['index']]), used
        self.launch_estimate = base+lengths[index]
        if self.owner is None:
            self.launch_estimate = max(self.launch_estimate,self.private_bytes)
        self.launch_margin = self.PROGRESS_MARGIN
        if self.compiler is None and any(
                self.needs_compile(getattr(slot,'memory_request',None) or {}) for slot in initial):
            # Jobs are already at the compiler gate. Keep room for one ticket
            # before adding another resident process. Running permits still
            # charge the scratch allowance only when the ticket is issued.
            self.launch_margin = max(self.launch_margin,self.compiler_bytes)
        return self.launch_estimate, running, self.launch_margin

    @staticmethod
    def request_peak(request, used, length_bytes):
        resident = max(used, request.get('pss_bytes', 0))
        if 'required_peak_bytes' in request:
            extra = request['required_peak_bytes'] - request.get('pss_bytes', 0)
            return resident + max(128*res.MIB, extra)
        phase = request.get('phase')
        if phase == 'loading':
            return resident + 768*res.MIB
        # Parsing, initial target construction and export allocate length-
        # dependent arrays before reaching the next checkpoint.
        return resident + 256*res.MIB + length_bytes

    @staticmethod
    def needs_compile(request):
        return request.get('phase') == 'waiting_compile' and request.get('compile_possible')

    def grant_requests(self, initial, usage, other, snapshot, reserve, monitor, lengths, *, pressured=False):
        requests = sorted((slot for slot in initial if getattr(slot, 'memory_request', None) is not None),
                          key=lambda s: s.wait_started)
        for slot in requests:
            item = initial[slot]
            request = slot.memory_request
            resident = usage.get(slot, 0)
            # Both the actual resident pages and the unspent allowance for
            # runnable workers remain in the ledger. A paused worker returns
            # only the unused part of its current permit.
            item['grant'] = resident
            item['requested_peak'] = self.request_peak(request, resident, lengths[item['index']])
        if self.owner in requests:
            self.private_bytes = max(self.private_bytes, initial[self.owner]['requested_peak'])
        def grant(slot):
            if pressured:
                return 'memory'
            item = initial[slot]
            compile_needed = self.needs_compile(slot.memory_request)
            if compile_needed and self.compiler is not None and self.compiler is not slot:
                return 'compiler'
            peak = item['requested_peak']
            # The private completion allowance protects the owner from others.
            # The owner itself only needs room for its next safe step.
            running = other + self.reservations(initial, usage, exclude=slot, compiling=compile_needed)
            margin = self.PROGRESS_MARGIN if slot is not self.owner else 0
            if not res.can_admit(snapshot, reserve, 1, 0, running, peak, usage.get(slot, 0),margin)[0]:
                return 'memory'
            item['grant'] = item.pop('requested_peak')
            if compile_needed:
                self.compiler = slot
            slot.grant_memory()
            self.permits += 1
            fresh = monitor()
            for key in ('available', 'cgroup_headroom'):
                snapshot[key] = min(snapshot[key], fresh[key])
            return None

        grants_before = self.permits
        owner_blocked = self.owner in requests and grant(self.owner) == 'memory'
        for slot in requests:
            if slot is not self.owner:
                grant(slot)

        if (owner_blocked and self.permits == grants_before and
                self.compiler is None and len(requests) == len(initial)):
            # If the owner cannot run its next step, let an already advanced
            # worker with a feasible non-compiler step release its process.
            original = self.owner
            relief = [slot for slot in requests if slot is not original
                      and not self.needs_compile(slot.memory_request)]
            for slot in sorted(relief, key=lambda s: (
                    s.memory_request.get('phase') not in ('export_check', 'ik_finalize'),
                    -s.memory_request.get('frames_completed', 0), s.wait_started)):
                self.owner = slot
                if grant(slot) is None:
                    break
                self.owner = original

        for slot in requests:
            if slot.memory_request is None:
                continue
            self.waiting.add(slot)
            if not (self.needs_compile(slot.memory_request) and self.compiler is not None):
                self.memory_waiting.add(slot)
        if len(self.waiting) == len(initial) and self.permits == grants_before:
            if self.stalled_since is None:
                self.stalled_since = time.monotonic()
        else:
            self.stalled_since = None

    def stalled_victim(self, initial, usage, now=None):
        """Fall back only after all safe work has stalled with pages pinned."""
        now = time.monotonic() if now is None else now
        if (self.stalled_since is None or now-self.stalled_since < 3 or self.compiler is not None
                or self.owner not in self.memory_waiting):
            return None
        candidates = [slot for slot in self.memory_waiting if slot is not self.owner
                      and slot in initial and usage.get(slot,0)>0]
        return min(candidates,key=lambda slot: (getattr(slot,'frames_completed',0),-usage[slot]),default=None)

    def report(self, initial):
        return dict(policy='measured_resident_admission',
                    protected_case=initial.get(self.owner, {}).get('case'),
                    compiler_case=initial.get(self.compiler, {}).get('case'),
                    protected_bytes=self.private_bytes,
                    compiler_budget_bytes=self.compiler_bytes,
                    compiler_reserved_bytes=self.compiler_bytes if self.compiler in initial else 0,
                    measured_resident_base_bytes=self.measured_base,
                    next_launch_estimate_bytes=self.launch_estimate,
                    launch_margin_bytes=self.launch_margin,
                    unsampled_workers=self.unsampled_workers,
                    unsampled_worker_limit=self.bootstrap_limit,
                    permits_granted=self.permits, waiting_memory=len(self.memory_waiting),
                    waiting_compiler=len(self.waiting-self.memory_waiting))
