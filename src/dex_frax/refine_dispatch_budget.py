"""Prioritize existing Refine A candidates within a finite middle-stage budget."""

from collections import Counter, defaultdict, deque
from math import ceil

from .acceptance import POSE_NOTES
from .refine_routing import WINDOW_GEOMETRY, WINDOW_POSE


def affected_span(job):
    start, end = job['core_start'], job['core_end']
    for name in ('phase_leg_window', 'coupled_leg_window', 'platform_window', 'cross_foot_window'):
        if name in job:
            lo, hi = job[name]
            start, end = min(start, lo), max(end, hi)
    return start, end


def work_groups(queue, jobs):
    """Group overlapping geometry by its actual solve context and keep pose separate."""
    grouped = {}
    for pose in (False, True):
        spans = sorted((affected_span(job), queue.key(job), job['module'])
                       for job in jobs if (job['module'] in WINDOW_POSE) == pose)
        component, end = None, -1
        for (start, stop), key, module in spans:
            if start >= end:
                component = dict(keys=set(), modules=set())
                end = stop
            end = max(end, stop)
            component['keys'].add(key)
            component['modules'].add(module)
            grouped[key] = component
    return grouped


def upper_quartile(samples, fallback):
    return sorted(samples)[ceil(.75 * len(samples)) - 1] if samples else fallback


class FiniteDispatch:
    def __init__(self, queue, attempts, same_round=False):
        self.queue = queue
        self.costs = defaultdict(lambda: deque(maxlen=12))
        self.check_costs = defaultdict(lambda: deque(maxlen=12))
        self.outcomes = defaultdict(lambda: deque(maxlen=12))
        self.visited = set()
        self.budget_blocked = False
        for entry in attempts:
            self.observe(entry, in_round=same_round and 'budget_dispatch' in entry)

    def observe(self, entry, *, in_round=True):
        if 'recipe' not in entry or 'core_start' not in entry:
            return
        family = entry['module'], entry['recipe']
        if entry.get('elapsed_s', 0) > 0:
            self.costs[family].append(entry['elapsed_s'])
        if entry.get('validation_and_commit_elapsed_s', 0) > 0:
            self.check_costs[family].append(entry['validation_and_commit_elapsed_s'])
        self.outcomes[family].append(
            1 + len(entry.get('cleared_flags', ())) if entry['status'] == 'kept'
            else .5 if entry.get('resumable_progress') else 0)
        if in_round:
            self.visited.add(self.queue.key(entry))

    def choose(self, pending, ready, flags, remaining, slice_seconds, combined_recipes,
               *, nominal_frames):
        groups = work_groups(self.queue, pending)
        unique_groups = {id(group): group for group in groups.values()}
        occurrences = Counter(module for group in unique_groups.values()
                              for module in group['modules'])
        pose_is_last_gate = bool(flags) and set(flags) <= POSE_NOTES
        geometry_pending = any(module in occurrences for module in (*WINDOW_GEOMETRY, 'body'))
        choices = []
        self.budget_blocked = False
        for job in ready:
            key = self.queue.key(job)
            group = groups[key]
            module = job['module']
            recipe = self.queue.recipe(job)
            family = module, recipe
            limit = slice_seconds(job, recipe)
            lo, hi = affected_span(job)
            width = max(hi - lo, job['end'] - job['start'])
            estimate = upper_quartile(self.costs[family], limit * min(1., width / nominal_frames))
            verification = upper_quartile(self.check_costs[family], 0.)
            if estimate > remaining or verification >= remaining:
                self.budget_blocked = True
                continue
            coverage = group['modules'] if recipe in combined_recipes else {module}
            impact = sum(1 + 1 / occurrences[gate] for gate in coverage)
            recent = self.outcomes[family]
            progress = (1 + sum(recent)) / (2 + len(recent))
            rate = impact * progress / max(estimate, .001)
            tier = (2 if module in WINDOW_POSE and not pose_is_last_gate
                    else 1 if module == 'jitter' and geometry_pending else 0)
            visited = bool(self.visited & group['keys'])
            priority = (tier, visited, -rate, -job['severity'], job['core_start'], key)
            details = dict(estimated_seconds=estimate, verification_seconds=verification,
                           group_modules=sorted(group['modules']), estimated_progress_per_second=rate,
                           first_attempt_in_group=not visited, pose_is_last_gate=pose_is_last_gate)
            choices.append((priority, job, details))
        if choices:
            _, job, details = min(choices, key=lambda item: item[0])
            self.budget_blocked = False
            return job, details
        return None, None
