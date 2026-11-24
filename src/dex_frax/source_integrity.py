"""Native-time evidence and conservative repair for BVH translation and rotation."""

from dataclasses import replace
from itertools import groupby

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .human_motion_constants import source_reference_scope


POLICY = 'native_source_jump_v11'

# Engineering noise floors and local ratio from the 0.8.8 source audit.
# Speeds here reject small neighbor differences; human speed references are
# evaluated independently and do not take part in any jump or repair decision.
MIN_TRANSIENT_M_S = .65
MIN_TRANSIENT_RAD_S = np.deg2rad(120.)
ISOLATED_STEP_RATIO = 4.
FIXED_LAFAN_TOES = frozenset(('LeftToe', 'RightToe', 'LeftToeBase', 'RightToeBase'))

# Project repair policy: cap the short reconstruction window at about 0.2 s,
# with a minimum allowance of three samples at low native frame rates.
BRIDGE_WINDOW_S = .2
ADVISORY_STEPS = frozenset(('isolated_step', 'isolated_reversal'))


def resolve_source(motion, source):
    if source != 'auto':
        return source
    if 'LeftShin' in motion.names:
        return 'soma'
    return 'mocap' if 'LeftHandIndex1' in motion.names else 'lafan'


def step_vectors(values, rotation):
    if rotation:
        return Rotation.from_matrix(values[1:] @ values[:-1].transpose(0,2,1)).as_rotvec()
    return np.diff(values,axis=0)


def isolated_pairs(steps,input_fps,minimum):
    """Detect an isolated out-and-back sample using local time context."""
    norm = np.linalg.norm(steps,axis=1)*input_fps
    if len(norm)<4:
        return np.empty(0,dtype=int)
    outer = np.maximum(norm[:-3],norm[3:])
    pair = np.minimum(norm[1:-2],norm[2:-1])
    maximum = np.maximum(norm[1:-2],norm[2:-1])
    residual = np.linalg.norm(steps[1:-2]+steps[2:-1],axis=1)*input_fps
    # Legacy 0.8.8 shape heuristics: a 3x local rise and up to 35% residual.
    return np.flatnonzero((pair>=np.maximum(minimum,3*outer)) & (residual<=.35*maximum))+2


def stable_segments(mask):
    for is_set, indices in groupby(enumerate(mask),key=lambda item:item[1]):
        if not is_set:
            continue
        frames=[i for i,_ in indices]
        yield frames[0],frames[-1]+1


def _bridge_frame_limit(fps):
    return max(3, round(BRIDGE_WINDOW_S * fps))


def _interior_samples(values, rotation, left, right):
    t = np.arange(1, right - left)
    anchors = values[[left, right]]
    if rotation:
        return Slerp([0, right - left], Rotation.from_matrix(anchors))(t).as_matrix()
    return anchors[0] + t[:, None] / (right - left) * (anchors[1] - anchors[0])


def review_channel(name, values, rotation, fps, *, fixed=False):
    steps=step_vectors(values,rotation)
    speed=np.linalg.norm(steps,axis=1)*fps
    minimum=MIN_TRANSIENT_RAD_S if rotation else MIN_TRANSIENT_M_S
    pulses=isolated_pairs(steps,fps,minimum)
    repaired=values.copy() if len(pulses) else values
    events=[]
    for frame in pulses:
        evidence = None
        # Require agreement across the clip to distinguish authored translation
        # from a one-sample violation of a fixed segment coordinate.
        if fixed:
            others=np.delete(values,frame,axis=0)
            if np.max(abs(others-values[frame-1])) < (5e-5 if rotation else 2e-4):
                evidence='isolated_violation_of_fixed_lafan_channel'
        method = None
        if evidence:
            # Keep the tested adjacent timestamps and use the geometry of the
            # channel for interpolation.
            triplet = values[frame-1:frame+2]
            if rotation:
                reference = Slerp([0,2],Rotation.from_matrix(triplet[[0,2]]))([1])[0]
                replacement = reference.as_matrix()
            else:
                replacement = (triplet[0]+triplet[2])/2
            repaired[frame], method = replacement, 'neighbor_reconstruction'
        events.append(dict(channel=name,frame=int(frame),kind='isolated_reversal',
                           peak_rate=float(speed[frame-1:frame+1].max()),
                           unit='rad/s' if rotation else 'm/s',
                           decision='repaired' if method else 'review',evidence=evidence,method=method))
    if len(speed)>=3:
        isolated = speed[1:-1]>=np.maximum(ISOLATED_STEP_RATIO*np.maximum(speed[:-2],speed[2:]),minimum)
        for edge in np.flatnonzero(isolated)+1:
            if np.any((pulses==edge)|(pulses==edge+1)):
                continue
            events.append(dict(channel=name,interval=[int(edge),int(edge+1)],kind='isolated_step',
                               peak_rate=float(speed[edge]),unit='rad/s' if rotation else 'm/s',
                               decision='review',evidence=None))
    if fixed:
        # A short loss of a constant structural channel need not form a
        # one-frame reversal. Require clean observations on both sides.
        median=np.median(values,axis=0)
        tolerance=5e-5 if rotation else 2e-4
        err=np.max(np.abs(repaired-median),axis=tuple(range(1,values.ndim)))
        if np.mean(err<=tolerance)>=.9:
            for a,b in stable_segments(err>tolerance):
                can_reconstruct = b-a<=_bridge_frame_limit(fps) and a>=2 and b+2<=len(values) and (
                    np.max(err[a-2:a])<=tolerance and np.max(err[b:b+2])<=tolerance)
                if can_reconstruct:
                    if repaired is values: repaired = values.copy()
                    reference = values[a-1]
                    repaired[a:b] = reference
                    events.extend(dict(channel=name,frame=frame,kind='fixed_channel_run',
                                       decision='repaired',evidence='verified_constant_lafan_channel',
                                       method='same_channel_neighbor_reconstruction') for frame in range(a,b))
                else:
                    events.append(dict(channel=name,interval=[max(0,a-1),min(len(values)-1,b)],
                                       kind='unresolved_fixed_channel',decision='review',
                                       evidence='verified_constant_lafan_channel',repair_blocking=True))
    return repaired,events


def _summary(steps, fps):
    speed=np.abs(steps)*fps
    if not len(speed):
        return dict(absolute_peak_xyz=[0.,0.,0.],peak_interval_start_xyz=[None,None,None],peak_norm=0.)
    return dict(absolute_peak_xyz=speed.max(axis=0).tolist(),
                peak_interval_start_xyz=speed.argmax(axis=0).tolist(),
                peak_norm=float(np.linalg.norm(speed,axis=1).max()))


def _held_steps(values):
    return np.max(np.abs(np.diff(values, axis=0)), axis=tuple(range(1, values.ndim))) < 1e-8


def _hold_bridges(values, rotation, fps, first, stop):
    left, gap, right = step_vectors(values[[first-1, first, stop+1, stop+2]], rotation)
    middle = gap / (stop + 1 - first)
    length = np.linalg.norm(middle)
    # Project dropout heuristics: require measurable motion and agreement
    # within 28% of the bridge step with both neighboring motion vectors.
    return bool(length*fps >= (.2 if rotation else .04)
                and np.linalg.norm(left-middle) <= .28*length
                and np.linalg.norm(right-middle) <= .28*length)


def packet_holds(channels, fps):
    """Recover repeated capture packets with evidence on both sides.

    A true pause can also repeat the entire pose. Require the subsequent step
    to account for the missing time in at least two independently moving
    channels, with matching direction and speed immediately before and after.
    """
    items=list(channels.items())
    n=len(items[0][1][0])
    same=np.ones(n-1,dtype=bool)
    for values, _ in channels.values():
        same &= _held_steps(values)
    events=[]
    for a,b in stable_segments(same):
        if a < 1 or b + 2 >= n or b - a > _bridge_frame_limit(fps):
            continue
        support = [name for name, (values, rot) in items if _hold_bridges(values, rot, fps, a, b)]
        joints={name.split('.')[0] for name in support}
        if len(joints)<2:
            if support:
                events.append(dict(kind='possible_repeated_packet',interval=[a,b+1],channel='whole_pose',
                    decision='review',evidence_channels=support,repair_blocking=False))
            continue
        for name,(v,rot) in items:
            if np.array_equal(v[a],v[b+1]):
                continue
            v[a+1:b+1] = _interior_samples(v, rot, a, b+1)
        for frame in range(a+1,b+1):
            events.append(dict(channel='whole_pose',frame=frame,interval=[a,b+1],
                kind='repeated_capture_packet',decision='repaired',evidence_channels=support,
                evidence='whole_pose_held_then_caught_up_at_both_neighbor_velocities',
                method='translation_linear_rotation_slerp'))
    return events


def _channel_holds(name, values, rotation, fps):
    events = []
    for a, b in stable_segments(_held_steps(values)):
        if a < 1 or b + 2 >= len(values) or b - a > _bridge_frame_limit(fps):
            continue
        if not _hold_bridges(values, rotation, fps, a, b):
            continue
        replacement = _interior_samples(values, rotation, a, b + 1)
        values[a+1:b+1] = replacement
        for frame in range(a + 1, b + 1):
            events.append(dict(channel=name, frame=frame, interval=[a, b+1], kind='repeated_joint_samples',
                decision='repaired', evidence='catch_up_matches_velocity_on_both_sides',
                method='rotation_slerp' if rotation else 'translation_linear'))
    return events


def _audit_channels(channels, fps, *, fixed=()):
    events=packet_holds(channels,fps)
    for name,(values,rot) in channels.items():
        if name not in fixed:
            events.extend(_channel_holds(name, values, rot, fps))
        result,found=review_channel(name,values,rot,fps,fixed=name in fixed)
        for event in found:
            if event['decision']=='repaired':
                frame=event['frame']
                lo,hi=max(0,frame-1),min(len(values),frame+2)
                event['before_local_rate']=_summary(step_vectors(values[lo:hi],rot),fps)
                event['after_local_rate']=_summary(step_vectors(result[lo:hi],rot),fps)
        values[:]=result
        events.extend(found)
    return events


def _large_steps(values, rotation, fps):
    magnitude = np.linalg.norm(step_vectors(values, rotation), axis=1)
    if not len(magnitude):
        return magnitude, np.zeros(0, dtype=bool)
    # Compare with nearby motion, excluding both immediate neighbors so an
    # out-and-back spike cannot set its own reference.
    padded = np.pad(magnitude, 5, constant_values=np.inf)
    neighbors = np.sort([padded[5+delta:5+delta+len(magnitude)]
                         for delta in (-5,-4,-3,-2,2,3,4,5)], axis=0)
    count = np.isfinite(neighbors).sum(axis=0)
    context = neighbors[np.maximum(0,(count-1)//2), np.arange(len(magnitude))]
    context[count==0] = 0.
    minimum = MIN_TRANSIENT_RAD_S if rotation else MIN_TRANSIENT_M_S
    large = magnitude >= np.maximum(minimum / fps, ISOLATED_STEP_RATIO * context)
    return magnitude, large


def _bridge_large_gaps(name, values, rotation, fps):
    """Reconstruct a short span bracketed by two abrupt, opposite changes."""
    magnitude, large = _large_steps(values, rotation, fps)
    edges = np.flatnonzero(large).tolist()
    vectors = step_vectors(values, rotation)
    limit = _bridge_frame_limit(fps)
    events = []
    consumed_through = -1
    for index, left in enumerate(edges):
        if left <= consumed_through or (left > 0 and large[left-1]):
            continue
        for right in edges[index+1:index+limit+1]:
            if right-left > limit:
                break
            stop = right+1
            if stop < len(magnitude) and large[stop]:
                continue
            neighboring = np.r_[magnitude[max(0,left-2):left], magnitude[stop:stop+2]]
            if not len(neighboring):
                continue
            base_rate = MIN_TRANSIENT_RAD_S if rotation else MIN_TRANSIENT_M_S
            cap = min(max(base_rate/fps, 2.*neighboring.max()), .5*min(magnitude[left],magnitude[right]))
            anchors = values[[left, stop]]
            if vectors[left] @ vectors[right] >= 0 or np.any(magnitude[left+1:right] > cap):
                continue
            if np.linalg.norm(step_vectors(anchors,rotation)[0]) > cap*(stop-left):
                continue
            values[left+1:stop] = _interior_samples(values, rotation, left, stop)
            for frame in range(left+1,stop):
                events.append(dict(channel=name, frame=frame, interval=[left,stop],
                    kind='large_source_gap', decision='repaired',
                    evidence='two_large_steps_with_continuous_neighbor_anchors',
                    method='rotation_slerp' if rotation else 'translation_linear',
                    peak_rate=float(max(magnitude[left],magnitude[right])*fps),
                    unit='rad/s' if rotation else 'm/s'))
            consumed_through = right
            break
    return events


def _continues_in_same_direction(vectors, start, stop):
    """Check the candidate steps against steps immediately before and after."""
    if start == 0 or stop >= len(vectors):
        return False
    before, after = vectors[start - 1], vectors[stop]
    steps = vectors[start:stop]
    # Use direction agreement only. A separate threshold would turn a
    # continuous speed change back into a presumed tracking defect.
    return bool(before @ after > 0 and np.all(steps @ before > 0)
                and np.all(steps @ after > 0))


def _smooth_step_candidate(values, rotation, fps, start, stop, *, reversal=False):
    """Evaluate a short transition while holding the poses on both sides fixed."""
    left, right = (start, stop) if reversal else (start - 1, stop + 1)
    if left < 0 or right >= len(values):
        return None, dict(reason='missing_neighbor_anchors')
    if right - left - 1 > _bridge_frame_limit(fps):
        return None, dict(reason='run_exceeds_reconstruction_window')

    # The large-step screen consults five steps to either side. Keep enough
    # extra context to recheck every step whose threshold could change.
    lo, hi = max(0, left - 10), min(len(values), right + 11)
    before = values[lo:hi]
    before_steps = step_vectors(before, rotation)
    if not reversal and _continues_in_same_direction(before_steps, start-lo, stop-lo):
        return None, dict(reason='same_direction_speed_change_preserved')
    after = before.copy()
    replacement = _interior_samples(values, rotation, left, right)
    after[left + 1 - lo:right - lo] = replacement

    original, original_large = _large_steps(before, rotation, fps)
    smoothed, smoothed_large = _large_steps(after, rotation, fps)
    core = slice(left - lo, right - lo)
    info = dict(anchor_frames=[left, right], before_peak_rate=float(original[core].max() * fps),
                candidate_peak_rate=float(smoothed[core].max() * fps))
    after_steps = step_vectors(after, rotation)
    minimum = MIN_TRANSIENT_RAD_S if rotation else MIN_TRANSIENT_M_S
    if reversal:
        # Both changes belong to the suspected frame. Compare the pair with
        # its outside neighbors instead of testing either half in isolation.
        if start + 1 - lo not in isolated_pairs(before_steps, fps, minimum):
            return None, dict(info, reason='change_does_not_confirm_isolated_reversal')
    else:
        impulse = original[start - lo:stop - lo].max()
        adjacent = max(original[left - lo], original[right - 1 - lo])
        if impulse < ISOLATED_STEP_RATIO * adjacent:
            return None, dict(info, reason='change_not_isolated_from_neighbors')
    if rotation and before_steps[core].sum(axis=0) @ after_steps[core].sum(axis=0) < 0:
        return None, dict(info, reason='candidate_reverses_rotation_direction')
    if not reversal and not _corroborated_step(before_steps, fps, rotation, start-lo, stop-lo):
        return None, dict(info, reason='coherent_change_or_unconfirmed_transition')
    # A continuous bridge can still exceed the broad neighborhood candidate
    # screen. Reject it if the new ramp still has a strong step at either
    # anchor, or if the candidate introduces a flag outside its changed steps.
    outer = np.r_[original[max(0, core.start-1):core.start], original[core.stop:core.stop+1]]
    if (smoothed_large[core].any() and outer.size
            and smoothed[core].max() >= max(minimum / fps, ISOLATED_STEP_RATIO * outer.max())):
        return None, dict(info, reason='candidate_does_not_clear_step_screen')
    new_flags = smoothed_large & ~original_large
    new_flags[core] = False
    if new_flags.any():
        return None, dict(info, reason='candidate_introduces_neighbor_candidate')
    joins = slice(max(0, left - lo - 1), min(len(original), right - lo + 1))
    before_turn = np.linalg.norm(np.diff(before_steps[joins], axis=0), axis=1)
    after_turn = np.linalg.norm(np.diff(after_steps[joins], axis=0), axis=1)
    if smoothed[core].max() >= original[core].max() or after_turn.max() >= before_turn.max():
        return None, dict(info, reason='candidate_does_not_improve_step_and_joins')
    reason = ('isolated_reversal_smoothed_between_neighbor_frames' if reversal
              else 'isolated_transition_smoothed_between_fixed_anchors')
    return replacement, dict(info, reason=reason)


def _corroborated_step(vectors, fps, rotation, start, stop):
    """Distinguish a sharp single step or reversal from a broad speed peak."""
    lo, hi = max(0, start - 3), min(len(vectors), stop + 3)
    vectors = vectors[lo:hi]
    start, stop = start - lo, stop - lo
    magnitude = np.linalg.norm(vectors, axis=1)
    minimum = MIN_TRANSIENT_RAD_S if rotation else MIN_TRANSIENT_M_S
    indices = np.arange(max(1, start), min(len(vectors)-1, stop))
    neighbor = np.maximum(magnitude[indices-1], magnitude[indices+1])
    if np.any(magnitude[indices] >= np.maximum(minimum / fps, ISOLATED_STEP_RATIO * neighbor)):
        return True
    pulses = isolated_pairs(vectors, fps, minimum)
    return bool(np.any((pulses >= start + 1) & (pulses <= stop)))


def _review_middle_steps(name, values, rotation, fps, *, fixed):
    """Review every remaining large transition, including a single detected step."""
    _, large = _large_steps(values, rotation, fps)
    decisions = {}
    events = []
    reversal_frames = []
    if not fixed and large.any():
        minimum = MIN_TRANSIENT_RAD_S if rotation else MIN_TRANSIENT_M_S
        pulses = isolated_pairs(step_vectors(values, rotation), fps, minimum)
        for pulse in pulses:
            frame = int(pulse)
            left, right = frame - 1, frame + 1
            if not large[left:right].any():
                continue
            if reversal_frames and frame - reversal_frames[-1] < 2:
                continue
            reversal_frames.append(frame)
            replacement, info = _smooth_step_candidate(values, rotation, fps, left, right, reversal=True)
            if replacement is None:
                decisions[left] = dict(interval=[left, right], **info)
                continue
            values[frame] = replacement[0]
            events.append(dict(channel=name, frame=frame, interval=[left, right],
                kind='smoothed_source_reversal', decision='repaired',
                method='rotation_slerp' if rotation else 'translation_linear',
                unit='rad/s' if rotation else 'm/s', smoothing=info))
        if events:
            _, large = _large_steps(values, rotation, fps)
    last_anchor = -1
    for a, b in stable_segments(large):
        # The single-step candidate would move the identified reversal or
        # its anchors. Keep it on the pair-specific path.
        if any(a <= frame + 1 and b >= frame - 1 for frame in reversal_frames):
            continue
        if fixed:
            candidate, info = None, dict(reason='fixed_channel_requires_neighbor_reconstruction')
        elif a - 1 < last_anchor:
            candidate, info = None, dict(reason='overlaps_an_accepted_reconstruction')
        else:
            candidate, info = _smooth_step_candidate(values, rotation, fps, a, b)
        if candidate is None:
            decisions[a] = dict(interval=[a, b], **info)
            continue
        values[a:b + 1] = candidate
        last_anchor = b + 1
        for frame in range(a, b + 1):
            events.append(dict(channel=name, frame=frame, interval=[a, b], kind='smoothed_source_transition',
                decision='repaired', method='rotation_slerp' if rotation else 'translation_linear',
                unit='rad/s' if rotation else 'm/s', smoothing=dict(info)))

    magnitude, remaining = _large_steps(values, rotation, fps)
    vectors = step_vectors(values, rotation)
    for a, b in stable_segments(remaining):
        checks = [dict(check) for check in decisions.values() if check['interval'][0] < b and check['interval'][1] > a]
        # A speed peak alone is advisory when the steps continue in the
        # direction supported by both sides. Fixed rig channels retain their
        # structural checks.
        same_direction = not fixed and _continues_in_same_direction(vectors, a, b)
        corroborated = not same_direction and _corroborated_step(vectors, fps, rotation, a, b)
        if same_direction:
            kind, evidence = 'same_direction_speed_change', 'neighbor_steps_continue_in_same_direction'
        else:
            kind = 'unresolved_large_source_step' if corroborated else 'neighborhood_candidate'
            evidence = 'unbridged_native_step'
        events.append(dict(channel=name, interval=[a, b], kind=kind,
                           decision='review', repair_blocking=corroborated, peak_rate=float(magnitude[a:b].max()*fps),
                           unit='rad/s' if rotation else 'm/s', evidence=evidence, smoothing_checks=checks))
    return events


def _review_take(channels, fps, *, fixed=()):
    """Screen the complete native take before any resampling."""
    count = len(next(iter(channels.values()))[0])
    start, stop = 0, count
    gross = []
    boundary = {0: {}, count-1: {}}
    for name, (values, rotation) in channels.items():
        if name not in fixed:
            gross.extend(_bridge_large_gaps(name, values, rotation, fps))
        magnitude, large = _large_steps(values, rotation, fps)
        if len(large) < 2:
            continue
        for frame, edge, inward in ((0,0,1), (count-1,-1,-2)):
            if large[edge] and not large[inward] and magnitude[edge] >= ISOLATED_STEP_RATIO * magnitude[inward]:
                boundary[frame][name] = dict(rate=float(magnitude[edge]*fps),
                                             unit='rad/s' if rotation else 'm/s')
    for frame, evidence in boundary.items():
        if not evidence:
            continue
        leading = frame == 0
        start += int(leading)
        stop -= int(not leading)
        gross.append(dict(channel='whole_pose', frame=frame, kind='boundary_source_discontinuity',
                          interval=[0,1] if leading else [frame-1,frame], decision='discarded',
                          evidence=evidence, method='exclude_boundary_native_frame'))
    working = {name: (values[start:stop], rotation) for name, (values, rotation) in channels.items()}
    events = _audit_channels(working, fps, fixed=fixed)
    for name, (values, rotation) in working.items():
        steps = _review_middle_steps(name, values, rotation, fps,
                                     fixed=name in fixed)
        if any(e['decision'] == 'repaired' for e in steps):
            stale_kinds = ADVISORY_STEPS
            events = [e for e in events if not (e['channel'] == name and e['decision'] == 'review'
                                                and e['kind'] in stale_kinds)]
            _, current = review_channel(name, values, rotation, fps)
            events.extend(current)
        events.extend(steps)
    for event in events:
        if event['decision'] == 'review' and event['kind'] in ADVISORY_STEPS:
            name = event['channel']
            frame = event.get('frame')
            a, b = event['interval'] if frame is None else (frame - 1, frame + 1)
            values, rotation = working[name]
            candidate, info = _smooth_step_candidate(values, rotation, fps, a, b,
                                                     reversal=event['kind'] == 'isolated_reversal')
            if candidate is not None:
                info['reason'] = 'advisory_peak_needs_corroboration'
            event['smoothing'] = info
        if 'frame' in event:
            event['frame'] += start
        if 'interval' in event:
            event['interval'] = [frame + start for frame in event['interval']]
        checks = [event['smoothing']] if 'smoothing' in event else event.get('smoothing_checks', [])
        for check in checks:
            for key in ('interval', 'anchor_frames'):
                if key in check:
                    check[key] = [frame + start for frame in check[key]]
    return sorted(gross+events, key=lambda event: event['frame'] if 'frame' in event else event['interval'][0]), (start,stop)


def inspect(motion, source, *, repair=True, ground_running=False):
    """Scan native capture timestamps; mutate private arrays and preserve the source."""
    source=resolve_source(motion,source)
    pos,rot=motion.local_p.copy(),motion.local_r.copy()
    channels={}
    fixed=set()
    for j,name in enumerate(motion.names):
        if name.endswith('__EndSite'):
            continue
        root=motion.parents[j]<0
        for orientation,values in ((False,pos),(True,rot)):
            key=name+('.rotation' if orientation else '.translation')
            channels[key]=(values[:,j],orientation)
            if source=='lafan' and not root and (not orientation or name in FIXED_LAFAN_TOES):
                fixed.add(key)
    events, (start,stop) = _review_take(channels, motion.fps, fixed=fixed)
    cleaned=replace(motion,local_p=pos[start:stop],local_r=rot[start:stop])
    if not repair:
        for e in events:
            if e['decision'] in ('repaired','discarded'):
                e.update(decision='repairable',repair_blocking=True)
        cleaned=motion
        start, stop = 0, len(pos)
    report_data=report(source,motion.fps,len(pos),events,(start,stop))
    report_data['ground_running_declared'] = ground_running
    from .source_speed import inspect_speeds, record_remaining_speed_limits
    raw_positions, raw_rotations = motion.fk()
    report_data['human_speed'] = inspect_speeds(raw_positions, raw_rotations, source, motion.fps,
                                                ground_running=ground_running)
    world_positions, world_rotations = cleaned.fk()
    if report_data['human_speed']['hard_speed_limits']:
        corrected = inspect_speeds(world_positions,world_rotations,source,motion.fps,
                                   ground_running=ground_running,frame_origin=start)
        record_remaining_speed_limits(report_data,corrected,'native')
    report_data['axis_rates']={name:{
        'world_z_up_linear_m_s':_summary(np.diff(world_positions[name],axis=0),motion.fps),
        'parent_frame_angular_rad_s':_summary(step_vectors(cleaned.local_r[:,j],True),motion.fps)}
        for j,name in enumerate(motion.names) if not name.endswith('__EndSite')}
    if start:
        for rates in report_data['axis_rates'].values():
            for rate in rates.values():
                rate['peak_interval_start_xyz'] = [None if i is None else i+start for i in rate['peak_interval_start_xyz']]
    return cleaned,report_data


def _anomaly_segments(events):
    """Group overlapping or neighboring abnormal intervals on the native timeline."""
    intervals = []
    for event in events:
        frames = event.get('interval')
        if frames is None:
            frames = [event['frame'], event['frame']]
        intervals.append((frames[0], frames[1], event))
    intervals.sort(key=lambda item: item[:2])
    segments = []
    for start, stop, event in intervals:
        if not segments or start > segments[-1]['interval'][1] + 1:
            segments.append(dict(interval=[start, stop], channels=set(), repaired_frames=set(),
                                 discarded_frames=set(), review_channels=set(), blocking_channels=set()))
        segment = segments[-1]
        segment['interval'][1] = max(stop, segment['interval'][1])
        channels = set(event.get('evidence_channels') or [event['channel']])
        segment['channels'].update(channels)
        decision = event['decision']
        if decision == 'repaired':
            segment['repaired_frames'].add(event['frame'])
        elif decision == 'discarded':
            segment['discarded_frames'].add(event['frame'])
        else:
            segment['review_channels'].update(channels)
        if event.get('repair_blocking'):
            segment['blocking_channels'].update(channels)
    return [{key: sorted(value) if isinstance(value, set) else value
             for key, value in segment.items()} for segment in segments]


def report(source, fps, count, events, kept):
    return dict(policy=POLICY,native_fps=fps,frame_convention='zero_based_input',
                frame_count=count,source_type=source,events=events,anomaly_segments=_anomaly_segments(events),
                kept_native_interval=list(kept),
                repaired_frames=sorted({e['frame'] for e in events if e['decision']=='repaired' and kept[0]<=e['frame']<kept[1]}),
                discarded_frames=sorted({e['frame'] for e in events if e['decision']=='discarded'}),
                blocking_events=[e for e in events if e.get('repair_blocking')],
                review_event_count=sum(e['decision']=='review' for e in events),
                tracking_discontinuity_screen=dict(
                    min_transient_m_s=MIN_TRANSIENT_M_S, min_transient_rad_s=float(MIN_TRANSIENT_RAD_S),
                    neighborhood_ratio=ISOLATED_STEP_RATIO, provenance='0.8.8 engineering noise floors and local ratio',
                    max_reconstructed_frames=_bridge_frame_limit(fps), reconstruction_window_s=BRIDGE_WINDOW_S),
                references=source_reference_scope())


def smplx_channels(fields, count, joint_names):
    layout={'global_orient':joint_names[:1], 'body_pose':joint_names[1:22],
            'jaw_pose':joint_names[22:23], 'leye_pose':joint_names[23:24],
            'reye_pose':joint_names[24:25], 'left_hand_pose':joint_names[25:40],
            'right_hand_pose':joint_names[40:55]}
    channels={'pelvis.translation':(np.array(fields['transl'][:count],copy=True),False)}
    for field,names in layout.items():
        if field not in fields:
            continue
        vecs=fields[field][:count].reshape(count,-1,3)
        for name,vec in zip(names,vecs.transpose(1,0,2),strict=True):
            channels[name+'.rotation']=(Rotation.from_rotvec(vec).as_matrix(),True)
    return layout, channels


def prepare_smplx_channels(fields, fps, count, joint_names, up_axis='y'):
    """Audit native parameters and return private corrected fields for resampling."""
    result = dict(fields)
    layout, channels = smplx_channels(fields, count, joint_names)
    events, (start,stop) = _review_take(channels, fps)
    out=report('smplx',fps,count,events,(start,stop))
    # Preserve the original rotvec branch on joints we did not repair.
    changed = {}
    for event in events:
        if event['decision'] == 'repaired' and start <= event['frame'] < stop:
            changed.setdefault(event['channel'], set()).add(event['frame'])
    whole_pose = changed.get('whole_pose', set())
    translated = sorted(whole_pose | changed.get('pelvis.translation', set()))
    if translated:
        result['transl']=fields['transl'].copy()
        result['transl'][translated]=channels['pelvis.translation'][0][translated]
    for field, names in layout.items():
        if field not in fields:
            continue
        for j, name in enumerate(names):
            channel = name + '.rotation'
            frames = sorted(whole_pose | changed.get(channel, set()))
            if not frames:
                continue
            if result[field] is fields[field]:
                result[field] = fields[field].copy()
            view = result[field].reshape(len(result[field]), -1, 3)
            view[frames, j] = Rotation.from_matrix(channels[channel][0][frames]).as_rotvec()
    out['axis_rates']={name:_summary(step_vectors(values[start:stop],angle),fps) for name,(values,angle) in channels.items()}
    if start:
        for rate in out['axis_rates'].values():
            rate['peak_interval_start_xyz'] = [None if i is None else i+start for i in rate['peak_interval_start_xyz']]
    out['axis_rate_frames']=dict(translation='native world '+up_axis+'-up, m/s',rotation='parent frame, rad/s',
                                 per_joint_world_linear='not available from parameters without native model FK')
    return result,out


def recheck_resampled(channels, fps, times, native_fps, *, native_channels=None, native_times=None):
    """Replace confirmed interpolation defects using the checked native channel."""
    events = []
    source_frames = np.asarray(times) * native_fps
    for name, (values, rotation) in channels.items():
        magnitude, candidate = _large_steps(values, rotation, fps)
        if not candidate.any():
            continue
        vectors = step_vectors(values, rotation)
        for a, b in stable_segments(candidate):
            same_direction = _continues_in_same_direction(vectors, a, b)
            corroborated = not same_direction and _corroborated_step(vectors, fps, rotation, a, b)
            event = dict(channel=name, interval=[a, b], decision='review',
                         kind='same_direction_speed_change' if same_direction else 'neighborhood_candidate',
                         source_frame_float=source_frames[[a, b]].tolist(), corroborated=corroborated,
                         peak_rate=float(magnitude[a:b].max() * fps), unit='rad/s' if rotation else 'm/s')
            if corroborated and native_channels is not None and name in native_channels:
                native, _ = native_channels[name]
                span = slice(max(0, a-1), min(len(values), b+2))
                sample_times = np.clip(times[span], native_times[0], native_times[-1])
                if rotation:
                    lo = max(0, np.searchsorted(native_times,sample_times[0],side='left')-1)
                    hi = min(len(native), np.searchsorted(native_times,sample_times[-1],side='right')+1)
                    reference = Slerp(native_times[lo:hi],Rotation.from_matrix(native[lo:hi]))(sample_times).as_matrix()
                else:
                    reference = np.stack([np.interp(sample_times, native_times, col) for col in native.T], axis=1)
                current = values[span]
                different = np.max(abs(current-reference), axis=tuple(range(1,values.ndim))) > 1e-6
                if different.any():
                    event.update(kind='interpolation_jump', repair_blocking=True)
                    if different[0] or different[-1]:
                        event['reason'] = 'missing_matching_endpoint_anchors'
                    else:
                        before = np.linalg.norm(step_vectors(current, rotation), axis=1)
                        after = np.linalg.norm(step_vectors(reference, rotation), axis=1)
                        if after.max() + 1e-8 < before.max():
                            current[different] = reference[different]
                            changed = (span.start+np.flatnonzero(different)).tolist()
                            event.update(decision='repaired', repair_blocking=False, repaired_frames=changed,
                                         method='reconstruct_from_checked_native_samples', repaired_peak_rate=float(after.max()*fps))
                        else:
                            event['reason'] = 'native_reference_does_not_reduce_step'
                else:
                    event['reason'] = 'consistent_with_native_motion'
            events.append(event)
    return dict(fps=float(fps), frame_convention='zero_based_resampled', frame_count=len(times),
                events=events, blocking_events=[e for e in events if e.get('repair_blocking')])


def record_resampled(report, recheck):
    report['resampled'] = recheck
    for event in recheck['blocking_events']:
        report['blocking_events'].append(dict(event, frame_grid='resampled',
            interval=event['source_frame_float'], resampled_interval=event['interval']))


def recheck_bvh(motion, source, report, *, ground_running=False, native=None):
    from .source_speed import inspect_speeds, record_remaining_speed_limits

    def bvh_channels(sequence):
        channels = {}
        for j, name in enumerate(sequence.names):
            if not name.endswith('__EndSite'):
                channels[name + '.translation'] = (sequence.local_p[:, j], False)
                channels[name + '.rotation'] = (sequence.local_r[:, j], True)
        return channels

    start = report['kept_native_interval'][0] / motion.input_fps
    times = start + np.arange(len(motion.local_p)) / motion.fps
    if native is not None:
        native_times = start + np.arange(len(native.local_p)) / motion.input_fps
        recheck = recheck_resampled(bvh_channels(motion), motion.fps, times, motion.input_fps,
                                    native_channels=bvh_channels(native), native_times=native_times)
    else:
        recheck = recheck_resampled(bvh_channels(motion),motion.fps,times,motion.input_fps)
    positions, rotations = motion.fk()
    recheck['human_speed'] = inspect_speeds(positions, rotations, source, motion.fps,
                                             ground_running=ground_running)
    record_remaining_speed_limits(report,recheck['human_speed'],'resampled',
                                  source_frames=times*motion.input_fps)
    return recheck


def inspect_smplx(fields, fps, count, joint_names):
    return prepare_smplx_channels(fields,fps,count,joint_names)[1]


def eligible_timing_intervals(report, source_time, *, root_only=False):
    """Avoid adapting intervals with unresolved capture defects."""
    allowed=np.ones(len(source_time)-1,dtype=bool)
    if report is None:
        return allowed
    left=source_time[:-1]*report['native_fps']
    right=source_time[1:]*report['native_fps']
    for event in report['blocking_events'] if 'blocking_events' in report else report['events']:
        if not event.get('repair_blocking') or event['decision']!='review':
            continue
        if root_only and not (event['channel']=='whole_pose' or event['channel'].startswith(('Hips.','pelvis.'))):
            continue
        start,end = event['interval'] if 'interval' in event else (event['frame']-1,event['frame']+1)
        allowed &= ~((left<end-1e-7)&(right>start+1e-7))
    return allowed
