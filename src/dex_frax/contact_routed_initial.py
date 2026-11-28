"""Use the landing solver for trusted non-foot support and preserve flight."""

import time

import numpy as np
from osqp import OSQPException
from scipy.spatial.transform import Rotation

from .adaptive_refine import MinimumChangeObjective
from .airborne import airborne_mask
from .cli import atomic_json
from .full_chain_initial import FullClipInitial
from .ik import FraxIK
from .identity import REFINE_INITIAL_POLICY
from .local_refine import fit_window
from .pelvis_priority import solve_protected
from .post_leg_contact import contact_errors
from .support_quality import preserve_contact
from .temporal_constraints import root_speed_bounds
from .unified_repair import dense_geometry
from .v8.math3d import runs
from .v8.self_collision import CollisionModel


def contact_routes(robot, targets):
    count = len(targets['root_p'])
    flight = airborne_mask(targets, count)
    feet = np.zeros(count, dtype=bool)
    for foot in targets['sides'].values():
        feet |= (np.asarray(foot['contact']) > 0) | (np.asarray(foot.get('platform_index', -1)) >= 0)
    foot_bodies = {group['foot'] for group in robot.sides.values()}
    other_support = np.zeros(count, dtype=bool)
    evidence = {}
    for contact in targets['body_contacts']:
        if contact['body'] in foot_bodies:
            continue
        active = np.flatnonzero(contact['mode'])
        trusted = [int(i) for i in active if preserve_contact(contact, i)]
        if trusted:
            evidence[contact['body_name']] = trusted
            other_support[trusted] = True
    uncertain = np.asarray(targets.get('source_support_uncertain', False), dtype=bool)
    feet_only = feet & ~other_support & ~uncertain & ~flight
    return other_support & ~flight, dict(other_support=evidence,
                                        feet_only_frames=np.flatnonzero(feet_only).tolist(),
                                        airborne_frames=np.flatnonzero(flight).tolist())


def expansion_edge(protected, cut, padding, entering):
    """Find the furthest cut reachable without crossing a protected frame."""
    if entering:
        edge = max(0, cut - padding)
        blocked = np.flatnonzero(protected[edge:cut])
        return edge + int(blocked[-1]) + 1 if blocked.size else edge
    edge = min(len(protected), cut + padding)
    blocked = np.flatnonzero(protected[cut:edge])
    return cut + int(blocked[0]) if blocked.size else edge


def expanded_sections(active, padding, protected=None):
    """Grow and merge landing intervals within the non-flight sections."""
    if protected is None:
        protected = np.zeros(len(active), dtype=bool)
    sections = []
    for first, stop in runs(active):
        lo = expansion_edge(protected, int(first), padding, entering=True)
        hi = expansion_edge(protected, int(stop), padding, entering=False)
        if sections and lo <= sections[-1][1]:
            sections[-1] = (sections[-1][0], hi)
        else:
            sections.append((lo, hi))
    return sections


def choose_rows(base, modern, active):
    return np.where(active[:, None], modern, base)


class SectionJoins:
    """Certify discrete route changes and solve short constrained joins as needed."""

    def __init__(self, robot, targets, cfg, legacy, landing, collision_cfg):
        self.robot = robot
        self.targets = targets
        self.cfg = cfg
        self.legacy = legacy
        self.landing = landing
        self.legacy_measurement = legacy.astype(np.float32).astype(float)
        self.landing_measurement = landing.astype(np.float32).astype(float)
        self.model = CollisionModel(robot, dict(collision_cfg or {}, coverage='full'))
        self.motor_step = robot.speed_limits / targets['fps']
        self.root_step = root_speed_bounds(targets['root_p'], targets['fps']) / targets['fps']
        self.rotation = Rotation.from_quat(
            Rotation.from_matrix(targets['root_r']).as_quat().astype(np.float32)).as_matrix()
        self.direct = {}

    def reference(self, start, end):
        section = slice(start, end)
        targets = {'root_r': self.rotation[section]}
        left, right = self.legacy_measurement, self.landing_measurement
        old_d, old_floor = dense_geometry(self.robot, self.model, targets, left[section])
        new_d, new_floor = dense_geometry(self.robot, self.model, targets, right[section])
        old_contact = contact_errors(self.robot, self.targets, left, start, end)
        new_contact = contact_errors(self.robot, self.targets, right, start, end)
        support_allowance = {k: max(v[0], v[1], new_contact[k][0]) for k, v in old_contact.items()}
        root_allowance = np.maximum(
            self.root_step[start:end - 1],
            np.maximum(np.linalg.norm(np.diff(left[section, :3], axis=0), axis=1),
                       np.linalg.norm(np.diff(right[section, :3], axis=0), axis=1)))
        return np.minimum(old_d, new_d), np.minimum(old_floor, new_floor), support_allowance, root_allowance

    def check(self, rows, start, end):
        rounded = rows.astype(np.float32).astype(float)
        sample = rounded[start:end]
        steps = np.diff(sample, axis=0)
        if np.any(abs(steps[:, 3:]) > self.motor_step + 1e-4 / self.targets['fps']):
            return 'motor_speed'
        if (np.any(sample[:, 3:] < self.robot.lower[3:] - 2e-6) or
                np.any(sample[:, 3:] > self.robot.upper[3:] + 2e-6)):
            return 'joint_range'
        before_d, before_floor, support, root_allowance = self.reference(start, end)
        if np.any(np.linalg.norm(steps[:, :3], axis=1) > root_allowance + 2e-6):
            return 'root_step'
        local = {'root_r': self.rotation[start:end]}
        distance, floor = dense_geometry(self.robot, self.model, local, sample)
        tolerance = self.model.cfg['penetration_tolerance_m']
        if np.any(distance < np.minimum(before_d, -tolerance) - 2e-6):
            return 'swept_collision'
        if np.any(floor < np.minimum(before_floor, -.001) - 2e-6):
            return 'swept_floor'
        contacts = contact_errors(self.robot, self.targets, rounded, start, end)
        if any(value > support[key] + 2e-6 for key, (value, _) in contacts.items()):
            return 'contact_or_slip'
        return None

    def check_direct(self, cut, entering):
        key = (cut, entering)
        if key not in self.direct:
            rows = self.legacy.copy() if entering else self.landing.copy()
            rows[cut:] = self.landing[cut:] if entering else self.legacy[cut:]
            self.direct[key] = self.check(rows, cut - 1, cut + 1)
        return self.direct[key]

    def fit(self, rows, cut, entering, section, padding):
        # The two rows on each side of this fit are fixed by the window solver.
        lo, hi = section
        if entering:
            start, end = cut - 2, min(hi, cut + padding)
            mutable = range(cut, end - 2)
        else:
            start, end = max(lo, cut - padding), cut + 2
            mutable = range(start + 2, cut)
        if start < 0 or end > len(rows) or not mutable or end - start < 5:
            raise ValueError('insufficient_boundary_context')
        options = dict(self.cfg, _frame_reconstruction={'frames': list(mutable)},
                       _coupled_swept_support=True, _initial_leg_collision_cfg=self.model.cfg)
        candidate, detail = fit_window(
            self.robot, self.targets, rows, options, start, end, iterations=6,
            ids_override=np.arange(rows.shape[1]), allow_support=True, local_checks=True,
            objective_factory=lambda i: MinimumChangeObjective(rows[i]), seed=rows)
        reason = self.check(candidate, start, end)
        if reason:
            raise ValueError(reason)
        return candidate, dict(window=[start, end], iterations=detail['iterations'])

    def recover_section(self, rows, required, section, blocked, joins):
        """Try removing optional padding before dropping a rejected landing section."""
        lo, hi = section
        core = np.flatnonzero(required[lo:hi]) + lo
        start, end = lo, hi
        moved = []
        for failure in blocked:
            cut, entering = failure['frame'], failure['entering']
            alternatives = (range(cut + 1, int(core[0]) + 1) if entering
                            else range(cut - 1, int(core[-1]), -1))
            alternate = next((i for i in alternatives if self.check_direct(i, entering) is None), None)
            if alternate is None:
                return None, dict(reason='no_checked_cut_in_padding')
            if entering:
                start = alternate
            else:
                end = alternate
            moved.append(dict(frame=cut,new_frame=alternate))

        candidate = rows.copy()
        candidate[lo:start] = self.legacy[lo:start]
        candidate[end:hi] = self.legacy[end:hi]
        for cut, old in ((start,lo),(end,hi)):
            if cut in (0,len(rows)):
                continue
            # A fit at the other end may overlap the trimmed rows. Check the
            # assembled trajectory, including the entire good fit if present.
            window = joins.get(old, {}).get('window') if cut==old else None
            a,b = window or (cut-1,cut+1)
            reason = self.check(candidate,a,b)
            if reason:
                return None,dict(reason=reason,frame=cut)
        return candidate,dict(remaining_section=[start,end],moved_joins=moved)

    def assemble(self, required, padding):
        if padding < 1:
            raise ValueError('positive join padding required')
        flight = airborne_mask(self.targets, len(required))
        if np.any(required & flight):
            raise ValueError('Landing routing must preserve source flight')
        active = required.copy()
        for a, b in expanded_sections(active, padding, flight):
            active[a:b] = True
        attempts = []
        while True:
            sections = [(int(a), int(b)) for a, b in runs(active)]
            rows = choose_rows(self.legacy, self.landing, active)
            grow = []
            joins = []
            blocked_sections = []
            for lo, hi in sections:
                blocked = []
                checked = {}
                for cut, entering in ((lo, True), (hi, False)):
                    if cut in (0, len(rows)):
                        continue
                    reason = self.check_direct(cut, entering)
                    if reason is None:
                        joins.append(dict(frame=cut, status='checked_direct'))
                        continue
                    # Look outward before fitting a seam, up to the first flight frame.
                    direction = -1 if entering else 1
                    edge = expansion_edge(flight, cut, padding, entering)
                    for alternate in range(cut + direction, edge + direction, direction):
                        if (alternate <= 0 or alternate >= len(rows) or
                                active[alternate - 1 if entering else alternate] or
                                self.check_direct(alternate, entering) is None):
                            grow.append((min(cut, max(0, alternate)), max(cut, min(len(rows), alternate))))
                            joins.append(dict(frame=cut, status='moved_outward', new_frame=alternate, reason=reason))
                            break
                    else:
                        try:
                            rows, fit = self.fit(rows, cut, entering, (lo, hi), padding)
                        except (ValueError, RuntimeError, OSQPException) as error:
                            if edge == cut:
                                blocked.append(dict(frame=cut,entering=entering,reason=reason,fit_reason=str(error)))
                            else:
                                grow.append((min(cut, edge), max(cut, edge)))
                                joins.append(dict(frame=cut, status='expand', original_reason=reason, reason=str(error)))
                        else:
                            checked[cut] = fit
                            joins.append(dict(frame=cut, status='solved', **fit))
                if blocked:
                    blocked_sections.append(((lo,hi),blocked,checked))
            attempts.append(dict(sections=sections, joins=joins))
            if blocked_sections:
                attempts[-1]['blocked_sections'] = [
                    dict(section=list(section),joins=failures) for section,failures,_ in blocked_sections]
            if not grow:
                for (lo,hi), failures, checked in blocked_sections:
                    candidate, detail = self.recover_section(rows,required,(lo,hi),failures,checked)
                    if candidate is None:
                        rows[lo:hi] = self.legacy[lo:hi]
                        active[lo:hi] = False
                        status = 'kept_contact_first'
                    else:
                        rows = candidate
                        start,end = detail['remaining_section']
                        active[lo:start] = False
                        active[end:hi] = False
                        status = 'moved_inward'
                    joins.append(dict(section=[lo,hi],status=status,**detail))
                return rows, active, attempts
            for lo, hi in grow:
                active[lo:hi] = True


class ContactRoutedInitial(FullClipInitial):
    """Adopt expanded 1.0.3 sections with explicit, audited boundaries."""

    policy = REFINE_INITIAL_POLICY

    def contact_first(self, robot, targets, cfg, progress, started):
        raw, _ = self.raw_solver(robot, targets, cfg, progress)
        self.baseline = raw.copy()
        self.save_stage('contact_first_raw', raw, robot, targets,
                        time.monotonic() - started, self.raw_solver.report)
        legacy, records, detail = solve_protected(
            robot, targets, cfg, raw, self.settings, 'feasible', progress,
            auto_lower=True, collision_cfg=self.initial_collision_cfg, correct_transitions=False)
        self.report = dict(detail['limb_solver'], body_priority=detail,
                           raw_frax_report=dict(self.raw_solver.report))
        self.report['qp_nonconverged_steps'] += self.raw_solver.report['qp_nonconverged_steps']
        self.save_stage('contact_first', legacy, robot, targets, time.monotonic() - started, detail)
        return legacy, records

    def landing_reference(self, robot, targets, cfg, progress):
        raw_solver = FraxIK(self.settings.iterations, self.settings.initial_iterations,
                           improved=False, assets=self.settings.assets)
        branch = FullClipInitial(self.output, raw_solver, self.settings.iterations,
                                 self.settings.initial_iterations, self.settings.assets,
                                 smooth_lifts=True, artifact_prefix='motion.landing_reference')
        branch.initial_collision_cfg = self.initial_collision_cfg
        branch.source_time_s = self.source_time_s
        rows, records = branch(robot, targets, cfg, progress)
        return rows, records, branch.report

    def __call__(self, robot, targets, cfg, progress=None):
        started = time.monotonic()
        required, evidence = contact_routes(robot, targets)
        self.ledger.update(routing_evidence=evidence, required_103_frames=np.flatnonzero(required).tolist())
        if required.all():
            self.smooth_lifts = True
            self.ledger.update(applied_103_frames=np.flatnonzero(required).tolist(),
                               applied_sections=[(0, len(required))])
            return super().__call__(robot, targets, cfg, progress)

        print('initial: contact-first reference', flush=True)
        legacy, records = self.contact_first(robot, targets, cfg, progress, started)
        rows = legacy
        active = required
        if required.any():
            print('initial: landing reference and expanded sections', flush=True)
            modern, landing_records, landing_report = self.landing_reference(robot, targets, cfg, progress)
            joins = SectionJoins(robot, targets, cfg, legacy, modern, self.initial_collision_cfg)
            rows, active, attempts = joins.assemble(required, max(3, round(.25 * targets['fps'])))
            records = np.where(active[:, None], landing_records, records)
            self.report['qp_nonconverged_steps'] += landing_report['qp_nonconverged_steps']
            self.ledger.update(landing_reference=landing_report['full_chain']['report_file'],
                               join_attempts=attempts)
            self.save_stage('contact_routed', rows, robot, targets,
                            time.monotonic() - started, {'attempts': len(attempts)})

        assert np.array_equal(rows[~active], legacy[~active])
        unapplied = required & ~active
        self.initial = rows.copy()
        final_stage = 'contact_routed' if active.any() else 'contact_first'
        self.ledger.update(status='completed', final_stage=final_stage,
                           applied_103_frames=np.flatnonzero(active).tolist(),
                           unapplied_103_frames=np.flatnonzero(unapplied).tolist(),
                           applied_sections=[(int(a), int(b)) for a, b in runs(active)],
                           initial_wall_s=time.monotonic() - started)
        self.report['full_chain'] = dict(report_file='motion.full_chain.json', selected=final_stage,
                                         routed_ground_frames=int(active.sum()),
                                         unapplied_landing_required_frames=int(unapplied.sum()))
        atomic_json(self.artifact('full_chain.json'), self.ledger)
        return rows, records
