"""Partition residual action cores before adding mutable seam ramps and read context."""
import numpy as np


FAMILIES = ('body_support', 'root_drop', 'geometry', 'timing', 'pose')


def plan_segments(evidence, contacts, fps, source_time):
    """Assign every anomalous frame once; contextual overlap never merges two cores.

    Each evidence channel is (family, nonnegative excess per frame). A trusted
    support episode owns its entire core. Longer secondary symptoms are attached
    to that episode and their unowned residual intervals remain independent.
    """
    n = len(source_time)
    claimed = np.zeros(n, bool)
    stable_gap = max(2, round(.24 * fps))
    ramp = max(2, round(.14 * fps))
    context = max(2, round(.12 * fps))
    changes = np.flatnonzero(np.any(contacts[1:] != contacts[:-1], axis=1)) + 1
    transition = np.zeros(n, bool)
    transition[changes] = True
    jobs = []
    for family in FAMILIES:
        channels = [values for group, values in evidence.values() if group == family]
        if not channels:
            continue
        severity = np.maximum.reduce(channels)
        active = np.flatnonzero((severity > 0) & ~claimed)
        if not active.size:
            continue
        # Never join a secondary symptom through a primary-owned frame.
        owners = np.cumsum(claimed)
        split = (np.diff(active) > stable_gap) | (np.diff(owners[active]) != 0)
        for run in np.split(active, np.flatnonzero(split) + 1):
            a, b = int(run[0]), int(run[-1]) + 1
            claimed[a:b] = True
            lo, hi = max(0, a-ramp), min(n, b+ramp)
            # Move the writable boundaries away from contact switches. Context
            # outside these ramps is read for the solver and never copied back.
            while lo > 0 and transition[max(0,lo-1):min(n,lo+2)].any():
                lo -= 1
            while hi < n and transition[max(0,hi-1):min(n,hi+2)].any():
                hi += 1
            symptoms = sorted(name for name, (_, values) in evidence.items() if np.any(values[a:b] > 0))
            jobs.append(dict(segment_id=f'{family}-{a}-{b}', primary=family,
                             start=lo, end=hi, core_start=a, core_end=b,
                             read_start=max(0,lo-context), read_end=min(n,hi+context),
                             source_start_s=float(source_time[a]), source_last_s=float(source_time[b-1]),
                             peak_excess=float(severity[a:b].max()), affected_frames=len(run),
                             symptoms=symptoms))
    return jobs


def next_segment(jobs, attempted, retry_ids=()):
    """One independent first try per family before revisiting a family.

    Changed residuals which overlap a previously tried core in the same
    family inherit its visit. Context-only overlap does not count as coverage.
    """
    seen = {entry['segment_id'] for entry in attempted}
    remaining = [j for j in jobs if j['segment_id'] not in seen or j['segment_id'] in retry_ids]
    if not remaining:
        return None
    counts = {family: sum(a['primary'] == family for a in attempted) for family in FAMILIES}

    def priority(j):
        repeated = any(a['primary'] == j['primary'] and a['core_start'] < j['core_end']
                       and a['core_end'] > j['core_start'] for a in attempted)
        return (repeated, counts[j['primary']], FAMILIES.index(j['primary']),
                -j['peak_excess'], -j['affected_frames'], j['core_start'])
    return min(remaining, key=priority)
