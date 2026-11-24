"""Detect discontinuities and test local reconstructions during refine."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .frame_repair import contact_phases, coupled_contact, isolated_frames, reconstruction_candidates
from .robot_profile import (
    ROOT_ANGULAR_LIMIT_RAD_S, ROOT_EXTRA_DROP_TOLERANCE_M, ROOT_LINEAR_LIMIT_M_S, ROOT_NEIGHBOR_JUMP_FLOOR_M,
)
from .temporal_repair import regress_at


@dataclass(frozen=True)
class Signal:
    key: str
    floor: float
    unit: str
    limit: float = np.inf
    columns: tuple = ()
    side: str = ''
    part: str = ''

    @property
    def angular(self):
        return self.key == 'root_rotation' or self.key.startswith('source_joint:')


def steps(signal, values):
    if signal.angular:
        rotation = Rotation.from_matrix(values)
        return (rotation[1:] * rotation[:-1].inv()).as_rotvec()
    return np.diff(values, axis=0)


def detect(signal, values, fps, expected_height=None):
    """One mark per adjacent group of anomalous intervals, indexed from zero."""
    delta = steps(signal, values)
    speed = np.linalg.norm(delta, axis=1) * fps
    fast = speed > signal.limit + 1e-4
    pulse = np.zeros(len(speed), dtype=bool)
    jump = np.zeros(len(speed), dtype=bool)
    # Both sides of an out-and-back move are tested. Scales use displacement
    # as well as rate so a high frame rate does not amplify tiny input noise.
    floor = max(signal.floor, signal.floor * 30 / fps)
    if len(speed) >= 4:
        magnitude = speed / fps
        before, first, second, after = (magnitude[i:i+len(speed)-3] for i in range(4))
        outer = np.maximum(before, after)
        comparable_pair = np.minimum(first, second) >= np.maximum(floor, 3 * outer)
        residual = np.linalg.norm(delta[1:-2] + delta[2:-1], axis=1)
        returns = residual <= .4 * np.maximum(first, second)
        pulse[1:-2] |= comparable_pair & returns
        pulse[2:-1] |= comparable_pair & returns
    if len(speed) >= 3:
        outer = np.maximum(speed[:-2], speed[2:])
        jump[1:-1] = (speed[1:-1] >= floor * fps) & (speed[1:-1] >= 4 * outer)
    # A retargeting solver can introduce a supported downward step without the
    # immediate reversal or quiet neighboring steps required by the pulse screen.
    added_drop = np.zeros(len(speed), dtype=bool)
    if signal.key == 'pelvis_height' and expected_height is not None:
        source_step = np.diff(np.asarray(expected_height).reshape(-1))
        if source_step.shape != speed.shape:
            raise ValueError('Source height must be aligned to the output time grid')
        drop = -delta[:,0]
        added_drop = drop - np.maximum(0.,-source_step) > ROOT_EXTRA_DROP_TOLERANCE_M
    hit = np.flatnonzero(fast | pulse | jump | added_drop)
    if not len(hit):
        return []
    groups = np.split(hit, np.flatnonzero(np.diff(hit) > 1) + 1)
    return [dict(signal=signal.key, intervals=[int(g[0]), int(g[-1]+1)],
                 peak_interval=int(g[np.argmax(speed[g])]), peak_rate=float(speed[g].max()), unit=signal.unit,
                 reasons=[name for name, mask in (('rate_limit', fast), ('reversal', pulse), ('isolated_step', jump),
                                                  ('source_relative_drop',added_drop))
                          if mask[g].any()]) for g in groups]


def signals(robot):
    yield Signal('root_translation', ROOT_NEIGHBOR_JUMP_FLOOR_M, 'm/s', ROOT_LINEAR_LIMIT_M_S or np.inf, (0, 1, 2))
    yield Signal('pelvis_height', .025, 'm/s', columns=(2,))
    yield Signal('root_rotation', np.deg2rad(4), 'rad/s', ROOT_ANGULAR_LIMIT_RAD_S or np.inf)
    for index, name in enumerate(robot.names):
        yield Signal('joint:'+name, np.deg2rad(4), 'rad/s', robot.speed_limits[index], (index+3,))
    for side, group in robot.sides.items():
        for part in ('hand', 'foot'):
            chain = group['arm' if part == 'hand' else 'leg']
            yield Signal(side+'_'+part, .035, 'm/s', columns=tuple(chain), side=side, part=part)


def read_signals(robot, definitions, rows, rotations):
    values = {}
    ends = [s for s in definitions if s.part]
    for signal in definitions:
        if signal.part:
            values[signal.key] = np.empty((len(rows), 3))
        else:
            values[signal.key] = rotations if signal.angular else rows[:, signal.columns]
    if ends:
        for i, (row, rotation) in enumerate(zip(rows, rotations)):
            robot.set(row, rotation)
            for signal in ends:
                group = robot.sides[signal.side]
                point = (robot.point(group['tcp']) if signal.part == 'hand'
                         else robot.point(group['foot'], group['sole']))
                values[signal.key][i] = point
    return values


def mark(definitions, observations, fps, expected_height=None):
    return {s.key: detect(s, observations[s.key], fps,expected_height) for s in definitions if s.key in observations}


def iter_parents(parents, joint):
    parent = parents[joint]
    while parent >= 0:
        yield parent
        parent = parents[parent]


def source_evidence(meta, definitions, duration):
    """Read a verified native BVH, comparing events by time at its native rate."""
    from .cli import sha
    from .v8.bvh import load

    convention = meta.get('bvh_input_convention')
    source = Path(meta.get('input') or '.')
    if not convention:
        return {}, None, 'unsupported_source'
    if not source.is_file():
        return {}, None, 'source_unavailable'
    try:
        if sha(source) != meta.get('input_sha256'):
            return {}, None, 'source_hash_mismatch'
        motion = load(source, **convention)
    except (OSError, ValueError, IndexError, KeyError):
        return {}, None, 'source_read_error'
    if not len(motion.names) or motion.input_frames < 2:
        return {}, motion.fps, 'source_too_short'
    # Native BVH Y-up to Z-up, preserving raw sampling and original timing.
    count = min(motion.input_frames, int(np.ceil(duration*motion.fps)) + 2)
    root = int(np.flatnonzero(motion.parents < 0)[0])
    pos = motion.local_p[:count, root][:, [0, 2, 1]].copy()
    pos[:, 1] *= -1
    source_signals = {'root_translation': pos, 'pelvis_height': pos[:, 2:3],
                      'root_rotation': motion.local_r[:count, root]}
    if any(s.part for s in definitions):
        by_name = {name.lower(): i for i, name in enumerate(motion.names)}
        for signal in definitions:
            if not signal.part:
                continue
            joint = by_name.get((signal.side+signal.part).lower())
            if joint is None:
                continue
            # Propagate only the two hands and feet; avoid allocating a
            # second full BVH skeleton alongside the prepared targets.
            p = motion.local_p[:count, joint].copy()
            for parent in iter_parents(motion.parents, joint):
                p = motion.local_p[:count, parent] + np.einsum('nij,nj->ni', motion.local_r[:count, parent], p)
            source_signals[signal.key] = p
    native = [Signal(s.key, s.floor, s.unit) for s in definitions if s.key in source_signals]
    for j, name in enumerate(motion.names):
        if j != root and not name.endswith('__EndSite'):
            signal = Signal('source_joint:'+name, np.deg2rad(4), 'rad/s')
            native.append(signal)
            source_signals[signal.key] = motion.local_r[:count, j]
    return mark(native, source_signals, motion.fps), motion.fps, 'verified_native_bvh'


def saved_observations(robot, definitions, path, filename, n, fps, expected_height=None):
    checkpoint = path.parent / filename
    if not checkpoint.is_file():
        checkpoint = path.parent / 'frax_baseline' / filename
    if not checkpoint.is_file():
        return {}
    try:
        with np.load(checkpoint, allow_pickle=False) as archive:
            rows, rotations = archive['x'], archive['root_r']
            if (rows.shape != (n, len(robot.names)+3) or rotations.shape != (n, 3, 3)
                    or not np.isclose(float(archive['fps']), fps)
                    or not np.isfinite(rows).all() or not np.isfinite(rotations).all()):
                return {}
    except (OSError, ValueError, KeyError):
        return {}
    return mark(definitions, read_signals(robot, definitions, rows, rotations), fps,expected_height)


def matches(event, candidates, fps, comparison_fps):
    """Compare time coverage rather than equality between frame indices."""
    a, b = np.asarray(event['intervals'], dtype=float) / fps
    slack = max(1/fps, 1/comparison_fps)
    return any(start/comparison_fps < b+slack and end/comparison_fps > a-slack
               for start, end in (other['intervals'] for other in candidates))


class Discontinuities:
    def __init__(self, repair, path, meta, diagnostics, modules=None):
        self.repair = repair
        root_target = diagnostics.get('root_target')
        self.source_height = None if root_target is None else np.asarray(root_target)[:,2]
        self.definitions = list(signals(repair.robot))
        self.by_key = {s.key: s for s in self.definitions}
        self.before = self.measure()
        self.modules = modules
        self.weights = meta.get('configuration', {}).get('solver', {})
        self.phases = contact_phases(repair.targets)
        fps = repair.fps
        current = [event for events in self.before.values() for event in events]
        source_time = diagnostics.get('source_time_s')
        if source_time is not None and len(source_time) != len(repair.rows):
            source_time = None
        aligned = not meta.get('retiming') or source_time is not None
        duration = float(source_time[-1]) if source_time is not None else (len(repair.rows)-1)/fps
        self.source, native_fps, self.source_status = source_evidence(meta, self.definitions, duration)
        if not aligned:
            self.source_status += '_time_alignment_unavailable'
        self.stage_names = {}
        self.attempts = []
        self.native_fps = native_fps
        if current:
            prepared = {k: diagnostics[v] for k, v in (
                ('root_translation', 'root_target'), ('root_rotation', 'root_r'),
                ('Left_foot', 'Left_foot_target_p'), ('Right_foot', 'Right_foot_target_p')) if v in diagnostics}
            if 'root_translation' in prepared:
                prepared['pelvis_height'] = prepared['root_translation'][:, 2:3]
            prepared_marks = mark(self.definitions, prepared, fps)
            raw = saved_observations(repair.robot, self.definitions, path, 'motion.raw_frax.npz', len(repair.rows), fps,self.source_height)
            initial = saved_observations(repair.robot, self.definitions, path, 'motion.initial.npz', len(repair.rows), fps,self.source_height)
            available = [('native_bvh', self.source, native_fps), ('prepared_targets', prepared_marks, fps),
                         ('raw_frax', raw, fps), ('protected_initial', initial, fps)]
            self.stage_names = {name: list(stage) for name, stage, _ in available}
            for event in current:
                key = event['signal']
                evidence = {}
                for name, stage, rate in available:
                    if not rate or (name == 'native_bvh' and not aligned):
                        continue
                    start, end = event['intervals']
                    if name == 'native_bvh' and source_time is not None:
                        start, end = source_time[[start, end]] * fps
                    nearby = [candidate for candidate in stage.get(key, ()) if
                              matches(dict(intervals=[start,end]), [candidate], fps, rate)]
                    if nearby:
                        evidence[name] = max(nearby, key=lambda item: item['peak_rate'])
                event['matching_stages'] = list(evidence)
                event['stage_evidence'] = evidence
                event['earliest_observed'] = next(iter(event['matching_stages']), 'refine_input')
                a, b = event['intervals']
                event['airborne_frames'] = int(repair.air[a:b+1].sum())
                event['contact_switch'] = bool(np.any(np.diff(self.phases[max(0,a-2):b+3],axis=0)))
                if 'native_bvh' in evidence or 'prepared_targets' in evidence:
                    event['priority'] = 'source_evidence_present'
                elif key in prepared_marks and event['earliest_observed'] in ('raw_frax','protected_initial'):
                    event['priority'] = 'introduced_after_preparation'
                else:
                    event['priority'] = 'origin_uncertain'

    def measure(self, rows=None, rotations=None):
        r = self.repair
        values = read_signals(r.robot, self.definitions, r.rows if rows is None else rows,
                              r.rotations if rotations is None else rotations)
        return mark(self.definitions, values, r.fps,self.source_height)

    def height_window(self,lo,hi):
        return None if self.source_height is None else self.source_height[lo:hi]

    def candidates(self, signal, a, b):
        r = self.repair
        frames = np.arange(a+1, b-1)
        frames = frames[~r.air[frames]]
        if not frames.size:
            return
        if signal.angular:
            quats = r.quaternions.copy()
            proposed = Slerp([a, b-1], Rotation.from_quat(quats[[a, b-1]]))(frames).as_quat()
            proposed[np.einsum('ij,ij->i', proposed, quats[frames]) < 0] *= -1
            quats[frames] = proposed
            rotations = Rotation.from_quat(quats).as_matrix()
            yield 'rotation_interpolation', r.rows, quats, rotations
            if r.time_left():
                yield 'rotation_with_foot_ik', r.compensate_feet(rotations, frames), quats, rotations
            return
        rows = r.rows.copy()
        columns = np.asarray(signal.columns)
        endpoints = rows[[a,b-1]][:, columns]
        weight = (frames-a)[:, None] / (b-1-a)
        rows[np.ix_(frames, columns)] = ((1-weight)*endpoints[0] + weight*endpoints[1]).astype(np.float32)
        yield 'trajectory_interpolation', rows, r.quaternions, r.rotations
        if columns.min() < 3 and r.time_left():
            yield ('translation_with_foot_ik', r.compensate_feet(r.rotations, frames, seed=rows),
                   r.quaternions, r.rotations)

    def improve(self, signal, event, rows, rotations, a, b):
        r = self.repair
        lo, hi = max(0, a-4), min(len(rows), b+4)
        old = read_signals(r.robot, [signal], r.rows[lo:hi], r.rotations[lo:hi])[signal.key]
        new = read_signals(r.robot, [signal], rows[lo:hi], rotations[lo:hi])[signal.key]
        before = np.linalg.norm(steps(signal, old), axis=1) * r.fps
        after = np.linalg.norm(steps(signal, new), axis=1) * r.fps
        start, end = event['intervals']
        focus = slice(start-lo, end-lo)
        old_peak, new_peak = float(before[focus].max()), float(after[focus].max())
        reference=self.height_window(lo,hi)
        remaining = detect(signal, new, r.fps,reference)
        existing = detect(signal, old, r.fps,reference)
        new_marks = any(not matches(e, existing, r.fps, r.fps) for e in remaining)
        unresolved = any(e['intervals'][0] < end-lo and e['intervals'][1] > start-lo for e in remaining)
        improved = (new_peak < old_peak*.8 and not new_marks and
                    after.max() <= max(before.max(), signal.limit if np.isfinite(signal.limit) else 0.)+1e-4 and
                    not regress_at(before, after, signal.limit, 1e-4))
        return improved, unresolved, old_peak, new_peak

    def try_candidate(self, signal, event, method, rows, quats, rotations, a, b, **detail):
        r = self.repair
        target = dict(event, intervals=[max(a,event['intervals'][0]), min(b-1,event['intervals'][1])])
        better, unresolved, old, new = self.improve(signal, target, rows, rotations, a, b)
        reason = r.check(rows, rotations, a, b) if better else 'discontinuity_or_seam_not_improved'
        self.attempts.append(dict(signal=signal.key, interval=event['peak_interval'], window=[a,b], method=method,
                                  before=old, after=new, reason=reason, **detail))
        if reason is not None:
            return False
        r.rows, r.quaternions, r.rotations = rows, quats, rotations
        event['status'] = 'candidate_improved' if unresolved else 'candidate_cleared'
        return True

    def rebuild_frames(self, signal, event):
        frames = isolated_frames(event)
        start, end = event['intervals']
        if not len(frames) and end-start == 2 and 'rate_limit' in event['reasons']:
            r = self.repair
            data = read_signals(r.robot,[signal],r.rows[start:end+1],r.rotations[start:end+1])
            delta = steps(signal,data[signal.key])
            # A pair already beyond a declared speed limit can be a corrupt
            # sample even against a fast-moving neighborhood.
            dominant = np.argmax(np.abs(delta),axis=1)
            axis = dominant[0]
            if (dominant[1] == axis and
                    abs(delta[:,axis].sum()) <= .4*np.abs(delta[:,axis]).max() and
                    abs(delta[:,axis]).min() > .5*np.abs(delta[:,axis]).max()):
                return np.array([start+1])
        return frames

    def reconstruct(self, signal, event):
        r = self.repair
        frames = self.rebuild_frames(signal,event)
        if not len(frames) or np.any(r.air[frames]):
            return False
        available = np.ones(len(r.rows),bool)
        for related in self.before[signal.key]:
            rejected = self.rebuild_frames(signal,related)
            available[rejected] = False
        a, b = max(0,int(frames[0])-2), min(len(r.rows),int(frames[-1])+3)
        seeds = []
        for method, rows, quats, rotations, context in reconstruction_candidates(
                r, signal, frames, available, self.phases):
            if not r.time_left():
                break
            detail = dict(rebuilt_frames=frames.tolist(), context_frames=context.tolist())
            if self.try_candidate(signal,event,method,rows,quats,rotations,a,b,**detail):
                return True
            seeds.append((method,rows,detail))
        if signal.key not in ('root_translation','pelvis_height') or not event['contact_switch']:
            return False
        # Horizontal and vertical events at the same frames have the same
        # coupled variables. Charge only one joint optimization to that burst.
        event_key = tuple(frames)
        if event_key in self.coupled_spans:
            return False
        self.coupled_spans.add(event_key)
        for padding in (4, 8):
            a, b = max(0,int(frames[0])-padding), min(len(r.rows),int(frames[-1])+padding+1)
            if a > frames[0]-2 or b < frames[-1]+3:
                continue
            for method, rows, detail in reversed(seeds):
                if not r.time_left():
                    return False
                try:
                    coupled, fit = coupled_contact(r,rows,a,b,frames,self.weights,seconds=8.)
                except (ValueError, RuntimeError, TimeoutError) as exc:
                    self.attempts.append(dict(signal=signal.key, interval=event['peak_interval'], window=[a,b],
                                              method='coupled_contact_reconstruction', seed=method, reason=str(exc), **detail))
                    continue
                if self.try_candidate(signal,event,'coupled_contact_reconstruction',coupled,r.quaternions,r.rotations,a,b,
                                      seed=method, fit_iterations=fit['iterations'], **detail):
                    return True
        return False

    def run(self):
        r = self.repair
        root = ('root_translation', 'pelvis_height', 'root_rotation')
        events = [event for group in self.before.values() for event in group]
        # Root anomalies may also mark every end effector. Correct the shared
        # driver first, then recheck each endpoint before proposing any edits.
        priority = {key: i for i, key in enumerate(root)}
        origin = {'introduced_after_preparation': 0, 'origin_uncertain': 1, 'source_evidence_present': 2}
        events.sort(key=lambda e: ('source_relative_drop' not in e['reasons'] or e['airborne_frames']>0,
                                   origin[e['priority']],
                                   not (e['signal'] in root and e['contact_switch']),
                                   priority.get(e['signal'], 4 if self.by_key[e['signal']].part else 3),
                                   'rate_limit' not in e['reasons'], -e['peak_rate'], e['peak_interval']))
        self.coupled_spans = set()
        for event in events:
            key = event['signal']
            if self.modules is not None and 'discontinuities' not in self.modules:
                route = 'root_angular' if key == 'root_rotation' else 'jitter'
                if route not in self.modules:
                    event['status'] = 'outside_selected_modules'
                    continue
            if not r.time_left():
                event['status'] = 'pending_budget'
                continue
            signal = self.by_key[key]
            start, end = event['intervals']
            focus = event['peak_interval']
            lo, hi = max(0,start-4), min(len(r.rows),end+5)
            values = read_signals(r.robot, [signal], r.rows[lo:hi], r.rotations[lo:hi])[key]
            if not matches(dict(intervals=[start-lo,end-lo]), detect(signal, values, r.fps,self.height_window(lo,hi)), r.fps, r.fps):
                event['status'] = 'cleared_by_related_repair'
                continue
            if np.all(r.air[start:end+1]):
                event['status'] = 'protected_airborne'
                continue
            event['status'] = 'no_safe_candidate'
            # Use a bounded local window even for sustained rate violations.
            # A surviving mark stays visible in the report.
            for padding in (2, 4, 8):
                a, b = max(0, max(start, focus-8)-padding), min(len(r.rows), min(end, focus+9)+padding+1)
                try:
                    for method, rows, quats, rotations in self.candidates(signal, a, b):
                        if not r.time_left() or self.try_candidate(signal,event,method,rows,quats,rotations,a,b):
                            break
                        if ('source_relative_drop' in event['reasons'] and method=='translation_with_foot_ik'
                                and r.time_left()):
                            adjusted=np.arange(a+1,b-1)
                            adjusted=adjusted[~r.air[adjusted]]
                            for amplitude in (.65,.4):
                                if not r.time_left():
                                    break
                                seed=r.rows.copy()
                                seed[adjusted,2]+=amplitude*(rows[adjusted,2]-seed[adjusted,2])
                                smaller=r.compensate_feet(rotations,adjusted,seed=seed)
                                if self.try_candidate(signal,event,'partial_translation_with_foot_ik',smaller,quats,rotations,a,b,amplitude=amplitude):
                                    break
                            if event['status'].startswith('candidate'):
                                break
                            if r.time_left():
                                fitted,detail=coupled_contact(r,rows,a,b,adjusted,self.weights,seconds=2.)
                                if self.try_candidate(signal,event,'supported_drop_window_ik',fitted,quats,rotations,a,b,
                                                      fit_iterations=detail['iterations']):
                                    break
                    if padding == 2 and r.time_left() and not event['status'].startswith('candidate'):
                        self.reconstruct(signal,event)
                except (ValueError, RuntimeError, TimeoutError) as error:
                    self.attempts.append(dict(signal=key, interval=focus, window=[a,b], reason=str(error)))
                if event['status'].startswith('candidate') or not r.time_left():
                    break
            if event['status'] == 'no_safe_candidate' and not r.time_left():
                event['status'] = 'time_slice'

    def report(self, rows, rotations, kept):
        after = self.measure(rows, rotations)
        return dict(schema=1, frame_index='zero_based_step_start_end_exclusive', fps=self.repair.fps,
                    source_status=self.source_status, native_fps=self.native_fps,
                    stage_signals=self.stage_names, source_events=self.source,
                    input_events=self.before, output_events=after, attempts=self.attempts,
                    local_candidates_adopted=kept, thresholds_are='experimental continuity heuristics and existing robot limits')


def root_export_impulses(rows, rotations, source_rows, source_rotations, fps):
    """High-confidence output-only impulses, with source evidence per join."""
    result = []
    for signal, actual, source in (
        (Signal('root_translation', .07, 'm/s'), rows, source_rows),
        (Signal('root_rotation', np.deg2rad(10.), 'rad/s'), rotations, source_rotations),
    ):
        ref = np.linalg.norm(steps(signal,source),axis=1)*fps
        for event in detect(signal,actual,fps):
            start,end = event['intervals']
            local = ref[max(0,start-1):min(len(ref),end+1)]
            if event['peak_rate'] > 4*float(local.max(initial=0.)):
                result.append(event)
    return result
