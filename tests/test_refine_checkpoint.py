import hashlib
import json

import pytest

from dex_frax.refine_checkpoint import artifact, write


def test_interrupted_candidate_recovers_last_committed_motion_and_reference(tmp_path):
    motion = tmp_path / 'motion.pkl'
    reference = tmp_path / 'avoidance_reference.npy'
    motion.write_bytes(b'accepted motion')
    reference.write_bytes(b'accepted reference')
    state = dict(schema=3, status='running',
                 output_sha256=hashlib.sha256(motion.read_bytes()).hexdigest(),
                 avoidance_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
    write(tmp_path, state)
    saved = json.loads((tmp_path / 'refine_state.json').read_text())
    motion.write_bytes(b'candidate not committed to the queue')
    reference.write_bytes(b'candidate reference')
    assert artifact(tmp_path, saved, motion.name).read_bytes() == b'accepted motion'
    assert artifact(tmp_path, saved, reference.name).read_bytes() == b'accepted reference'
    with pytest.raises(ValueError, match='Committed checkpoint artifact changed'):
        artifact(tmp_path, dict(saved, status='review'), motion.name)


def test_initial_checkpoint_can_seed_either_refine_flow(tmp_path):
    from dex_frax import segments

    original = dict(fingerprint='seed-a', conversion_contract=dict(refine_flow='A', foot_contact='v2'))
    for name in ('motion.pkl', 'motion.json', 'motion.initial.npz', 'motion.diagnostics.npz', 'motion.npz'):
        (tmp_path/name).write_text(name)
    (tmp_path/'identity.json').write_text(json.dumps(original))
    segments.seal(tmp_path, original)
    alternate = dict(original, fingerprint='seed-b', conversion_contract=dict(refine_flow='B', foot_contact='v2'))
    assert segments.verify(tmp_path, alternate)['fingerprint'] == 'seed-a'

    changed_contact = dict(alternate, conversion_contract=dict(refine_flow='B', foot_contact='v1'))
    with pytest.raises(ValueError, match='Frax checkpoint identity changed'):
        segments.verify(tmp_path, changed_contact)
