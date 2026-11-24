"""Conversion semantics shared by resume and initial reuse; no resource knobs."""
import os

DEFAULT_FOOT_CONTACT = "v2"
UNLIMITED_REPAIR_BUDGET = "unlimited"
REFINE_FLOWS = ('A', 'B', 'C')
DEFAULT_REFINE_FLOW = 'A'
# Checkpoints written before the flow field used B's specialist cascade.
LEGACY_CHECKPOINT_FLOW = 'B'

def uses_original_frax(args):
    return getattr(args, 'profile', None) in ('fast', 'refine') or bool(getattr(args, 'legacy_frax', False))

def conversion_contract(args):
    from .identity import SOLVER_POLICY,QUALITY_POLICY,REFINE_INITIAL_POLICY
    from .dispatch import use_unified
    from .pelvis_priority import INITIAL_POLICY
    initial_policy = 'original_frax_full_motion' if uses_original_frax(args) else INITIAL_POLICY
    if getattr(args, 'profile', None) == 'refine' and getattr(args, 'mode', None) != 'v8':
        initial_policy = REFINE_INITIAL_POLICY
    result = dict(schema=7,ground_running=bool(getattr(args,'ground_running',False)),bvh_units=getattr(args,'bvh_units','cm'),bvh_translations=getattr(args,'bvh_translations','absolute'),allow_local_slowdown=bool(getattr(args,'allow_local_slowdown',False)),retiming_policy='prepared_target_local_root_slowdown_v2',refine_initial_policy=initial_policy, profile=getattr(args,'profile',None), repair_budget=getattr(args,'repair_budget',None), arm_principles='whole_arm_resemblance_v2', repair_similarity=getattr(args,'repair_similarity','soft'),pipeline=getattr(args,'pipeline','unified'),
        unified_repair=getattr(args,'unified_repair',use_unified(args)), fps=50., iterations=args.iterations,
        initial_iterations=args.initial_iterations,max_frames=getattr(args,'max_frames',None),
        legacy_frax=uses_original_frax(args),
        ik_block_frames=int(getattr(args,'ik_block_frames',None) or ('128' if hasattr(args,'manifest') else os.environ.get('DEX_FRAX_IK_BLOCK_FRAMES','128'))),
        solver_policy=SOLVER_POLICY,quality_policy=QUALITY_POLICY)
    foot_contact = getattr(args, 'foot_contact', DEFAULT_FOOT_CONTACT)
    result['foot_contact'] = foot_contact
    if result['profile'] == 'refine':
        result['refine_flow'] = getattr(args, 'refine_flow', DEFAULT_REFINE_FLOW)
    if getattr(args,'source',None)=='smplx':
        from .smplx_cli import preparation_identity
        result['smplx_input']=preparation_identity(args)
    if getattr(args,'scene',None) is not None:
        from .cli import sha
        result['scene_sha256']=sha(args.scene)
    return result
