"""Opt-in local experiment for a deep-penetration contact transition."""
import numpy as np
from scipy.optimize import least_squares

from .torso_branch import floor_samples


def recover_window(robot, targets, rows, raw, start, stop, center):
    if not 0 < start <= center < center + 4 < stop < len(rows):
        raise ValueError("Recovery window needs both unchanged boundary frames")
    fps = targets["fps"]
    rotations = targets["root_r"]
    ids = np.r_[:3, [i for group in robot.sides.values() for i in group["leg"]]]
    step = robot.speed_limits[ids[3:] - 3] / fps
    candidate = rows.copy()

    def measure(row, frame, active):
        robot.set(row, rotations[frame])
        floor = np.array([robot.point(b)[2] + np.min(shape["hull"] @ robot.rot(b)[2])
                          for b, shape in robot.surface_models.items()])
        errors = []
        for side in active:
            g = robot.sides[side]
            foot = targets["sides"][side]
            offset = robot.point(g["foot"], foot["pivots"][frame])[:2] - foot["anchors"][frame, :2]
            sole = robot.point(g["foot"], robot.lowest(side))[2]
            errors.extend([*offset, sole - .00015])
        return floor, np.asarray(errors)

    def active_at(frame):
        return [side for side, foot in targets["sides"].items() if foot["contact"][frame] > 0]

    def sample(array, a, b):
        return floor_samples(robot, {"root_r": rotations[a:b]}, array[a:b])

    old_floor = sample(rows, center, center + 5)
    if old_floor.min() >= -.05:
        return rows, {"status": "no_deep_penetration", "floor_before_mm": float(old_floor.min()*1000)}
    dominant = set(active_at(center))
    exit_step = max(3.0 / fps, np.linalg.norm(rows[stop, :3] - rows[stop-1, :3])) / np.sqrt(3)
    adjacent = np.diff(rows[start-1:stop+1, :3], axis=0)
    per_axis_step = np.maximum(.08, abs(adjacent).max(axis=0))
    for frame in range(start, stop):
        active = active_at(frame)
        previous = candidate[frame - 1, ids]
        reference = raw[frame].copy()
        reference[2] = targets["root_p"][frame, 2]
        lo = robot.lower[ids].copy()
        hi = robot.upper[ids].copy()
        lo[:2] = np.minimum(reference[:2], rows[frame, :2]) - .30
        hi[:2] = np.maximum(reference[:2], rows[frame, :2]) + .30
        lo[2] = min(reference[2], raw[frame, 2]) - .15
        hi[2] = max(reference[2], raw[frame, 2]) + .25
        root_remaining = (stop - frame) * exit_step
        lo[:3] = np.maximum.reduce([lo[:3], previous[:3]-per_axis_step, rows[stop,:3]-root_remaining])
        hi[:3] = np.minimum.reduce([hi[:3], previous[:3]+per_axis_step, rows[stop,:3]+root_remaining])
        remaining = (stop - frame) * step
        end = rows[stop, ids[3:]]
        lo[3:] = np.maximum.reduce([lo[3:], previous[3:] - step, end - remaining])
        hi[3:] = np.minimum.reduce([hi[3:], previous[3:] + step, end + remaining])
        if np.any(lo > hi + 1e-7):
            return rows, {"status": "rejected", "reason": "unreachable_joint_boundary", "frame": frame}
        middle = (lo + hi) * .5
        lo = np.minimum(lo, middle - 2e-8)
        hi = np.maximum(hi, middle + 2e-8)
        _, old_contact = measure(rows[frame], frame, active)
        weights = np.repeat([12000. if side in dominant else 1500. for side in active], 3)
        allowed = np.maximum(.0005, abs(old_contact))
        allowed[weights == 12000.] = .00035

        def residual(q):
            row = rows[frame].copy()
            row[ids] = q
            floor, contact = measure(row, frame, active)
            error = np.sign(contact) * np.maximum(abs(contact) - allowed, 0)
            return np.r_[30000 * np.minimum(floor + .0002, 0), weights * error,
                         3 * (q - reference[ids]), 25 * (q - previous)]

        fits = [least_squares(residual, np.clip(seed, lo + 1e-8, hi - 1e-8),
                              bounds=(lo, hi), max_nfev=90, ftol=1e-6)
                for seed in (reference[ids], raw[frame, ids], previous)]
        candidate[frame, ids] = min(fits, key=lambda fit: fit.cost).x

    focus_floor = sample(candidate, center, center + 5)
    old_span = sample(rows, start - 1, stop + 1).min(axis=1)
    new_span = sample(candidate, start - 1, stop + 1).min(axis=1)
    contacts = []
    for frame in range(center, center + 5):
        active = [side for side in active_at(frame) if side in dominant]
        _, values = measure(candidate[frame], frame, active)
        for x, y, z in values.reshape(-1, 3):
            contacts.append((np.hypot(x, y), abs(z + .00015)))
    old_steps = np.diff(rows[start-1:stop+1, :3], axis=0) * fps
    new_steps = np.diff(candidate[start-1:stop+1, :3], axis=0) * fps
    seam = [0, len(new_steps)-1]
    joint_ratio = float(np.max(abs(np.diff(candidate[start-1:stop+1, ids[3:]], axis=0)) / step))
    info = dict(status="accepted", start=start, stop=stop, center=center,
                floor_before_mm=float(old_floor.min()*1000), floor_after_mm=float(focus_floor.min()*1000),
                anchor_max_mm=float(max((x for x,_ in contacts), default=np.inf)*1000),
                sole_max_mm=float(max((z for _,z in contacts), default=np.inf)*1000),
                joint_step_ratio=joint_ratio,
                new_seams_m_s=np.linalg.norm(new_steps[seam], axis=1).tolist(),
                old_seams_m_s=np.linalg.norm(old_steps[seam], axis=1).tolist(),
                max_new_penetration_mm=float(np.maximum(np.minimum(old_span,-.001)-new_span,0).max()*1000))
    for rejected, reason in (
        (info["floor_after_mm"] < -1., "focus_floor"),
        (max(info["anchor_max_mm"], info["sole_max_mm"]) > 1., "stance_anchor"),
        (joint_ratio > 1.001, "joint_speed"),
        (any(new > max(3.0, old + .05) for old, new in zip(info["old_seams_m_s"],info["new_seams_m_s"])), "root_seam"),
        (info["max_new_penetration_mm"] > 1., "window_floor_regression"),
    ):
        if rejected:
            return rows, dict(info, status="rejected", reason=reason)
    return candidate, info
