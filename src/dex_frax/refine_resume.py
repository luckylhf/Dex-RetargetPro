"""Identity-checked continuation from an exported refine checkpoint."""
import json
from pathlib import Path
import numpy as np
from .contract import LEGACY_CHECKPOINT_FLOW


# Exact transitions that keep the old reference and permit a fresh audit.
_RECHECK_COMPATIBLE = {
    'v8/audit.py': (
        ('c8f536ba479076689032a36a3725fbd4e9befc059774443796e3ec7502439967',
         '60c667b6ba0385df09f7477b681b50891668c1265d00e79b42982cc44fb167f4',
         'b2de74cdd320dfb8da973702a9e846f4e5d6f38d570d81b69856f7f087ff0726',
         'a035b9ff99d654147a4aae09268bda195f619c2ad4579254683a652ec67f0156'),
        'c1107b9365a72b3867e003fada997d8d2578f0eb971685426b7dd6bd31e69ee1',
    ),
    'model_cache.py': (
        ('2869ce9217952d3877e22c15877dbc49448c32dbbd86d0ff249236a34101cd7c',),
        'bb8c9ef7e1d478bd2c501b13b31549ea5dda2b8f055b5ac3176566587f0820a9',
    ),
    'foot_collision.py': (
        ('0508aa097e6a1e1ce3ea590cfff9da008ec4ae52b67a37cabe8102b5d457ceca',),
        'cde88ffc13f25f416cad791e8d364701517192c3a0ff5feeb30c3162a1772470',
    ),
    'contact_routed_initial.py': (
        ('05970390c54db70e60fa51065f25d1f896e00f0d7e1900ec32cebf4c86ee0699',),
        '664e68c78d3c4c5d93ee5b6862a78c29c8cb40b0dc786ab93afc4e43cbf3dd22',
    ),
    'v8/self_collision.py': (
        ('57767317a4d7a2cbfb887eba4d8f8b80f914ba62658d8f0b738fb89515bd1921',),
        '22e6de2fd14171f2f2009fd26ef5d785323eb20d423e1b6bb25377dcab275b54',
    ),
    'v8/multicontact.py': (
        ('3ad383feee540d5e205aa00796204f144425dd27105313d57ae147a7f9f8f1c0',),
        '0c2b18161959583104c3e9bd2dd0a47da1e4b42d8f4ba78154fc3f74b971fc8b',
    ),
}

# Retain the exact Refine A migration from the shared-pool deployment.
# The transition permits known scheduling and timer changes, plus the fixes to
# candidate metadata and integer-frame retiming. All artifact checks still run.
_POOL_PREVIOUS = {
    'cascade.py': ('9c733f866fd3dc0ec4f3d7fc5af88f309726e1eda72ca6cc343cab32aa088ae1',),
    'cli.py': ('f61ab70fdd95493c72c08a871bf7562a83a366e7060f1bf051482045d5270551',
               'f1d8e909923cbdd3ba258093d3c1614184460c5273e7d31e20518bfe72151d9b'),
    'contract.py': ('09a4163c61b9ee232430a5f6516157f0838b7d6b90f972b19972a69a389871eb',),
    # Landing-join recovery only affects inputs that previously failed before export.
    'contact_routed_initial.py': ('05970390c54db70e60fa51065f25d1f896e00f0d7e1900ec32cebf4c86ee0699',),
    'managed_batch.py': ('2d8939296717d1379ed129899599371d8dddb7bf741fa3a05240e2e81deaf7eb',),
    'model_cache.py': ('2869ce9217952d3877e22c15877dbc49448c32dbbd86d0ff249236a34101cd7c',),
    'foot_collision.py': ('0508aa097e6a1e1ce3ea590cfff9da008ec4ae52b67a37cabe8102b5d457ceca',),
    'resources.py': ('3762a57e75f4835978173577d1d94e37138e47161c481ac4400a4f6fdc4b9d97',
                     'e2912af478519bbb53fe4936f5b0b1285ab2932d0cc4e29696aba5c280f7edd5'),
    'memory_pool.py': (None, 'a7eb8c33c2cbdeff754be5ea3126ef90ea268545e0689a9df8c1fb672d2bb11d',
                       'bfdd384f44ece8750cccae97c19e65cf1a8ecf207f52e6421c1295a4260a9ffc'),
    'memory_pool_batch.py': (
        None, '878aee12f1f8a25cfe1871424e2530c1f323cf9a138f2477bf0f8d78ef487d2f',
        '42a56eecb7320ed13f441371a5b899dca5302e18c8e9d4f6eb862ce771c4c39d',
        '095b78cf0ffdf5c4297c9a70e21bbce73b9aa318c90c4467dcccd80dc962d7e1',
        'fb72199814dd5a0a47a22160b2f2bc841cab4707bdcb044095baaa581a9c962d',
        'ebf45621e447a5f8ab78b482c26ba11af69f6a32623896f61ec782768826b062',
        'fbb60d765463f75f13f00eedd8ebfe2114dbc2121e4fc8be62b3d9b727b10cc6',
        '698a3b6c3e020cd65b0305a3f135713803591f87e39c55424424f7d54e7fb6d5',
        '70e5c53c8ddf6ee7c18e649fca911008ada847ea2357351594e665305ffb2c95',
        '4b2870da824464da3d979fa914a29cb0ffbec27f06b73218b8c92322a211b8c7',
    ),
    'pipeline_explain.py': ('13a2e626a09a58a06021b45d34641e9b721805785e9d8768f9df5ceeb86d4f97',),
    'retarget_batch.py': ('369feb1af8b201bed6a2f4488a3060d38d189a1dc2f6bb461efba28bdbbd1307',),
    'v8/audit.py': ('c8f536ba479076689032a36a3725fbd4e9befc059774443796e3ec7502439967',
                    '60c667b6ba0385df09f7477b681b50891668c1265d00e79b42982cc44fb167f4',
                    'b2de74cdd320dfb8da973702a9e846f4e5d6f38d570d81b69856f7f087ff0726',
                    'a035b9ff99d654147a4aae09268bda195f619c2ad4579254683a652ec67f0156'),
    'v8/convert.py': ('2ebeb99eeb5c358956f2df36f0b0c40f4b9f8ab0c01df92a904b7f2e737223b1',
                      '3074eabe4decb905c71bb55549376cce96cb99df1cf7d565131ed2d4f8b78207'),
    'refine_resume.py': (
        '0d8c397c765119170b6f645c819e493f65d30013db6b6e025fedeb667c6a5c00',
        '30fdf08907f97bd772a027361a0a346ee78eaf52033174e608a7c7d348cb6d70',
        'afd9c0cfec5d23b92bc267d1cf5dc3c2f0ec74e1ce7fd73cbc351c0642ef02be',
        '4cb29c8c891b353c7ef39117c0314692a2d6da7a22873aa0dd7e22404d88b72e',
        '4bba2c8d2906ec1504f7573bb89e026bbab04e8317daa02ebdaa1237159e97e6',
        '07a3c74f45e0bbe80b628714582fdaefad5befb2c4b9325af922e6c8c6556a3b',
        'e598123f64d1abc6aae62ba3260919c28899769b9a5e241ae2c3970651322023',
        '9bbbe5718014c409051508d9caad8c3af74a337eb5e6161f87cc0158b1d775a5',
        '91e6145aefe48b84cecac9ee07ec46205e84bc10eb2363fd5e72ce602fecd6a5',
        'fe0d4056b2ab5d80362ac8f90e2d98c3780c3b44fb5c06c20c08138b3907f244',
        'fba61aa2e54f51cab52afecd08e0568df8cef5a8f9a34aebe88ebcdb3c622f4f',
        '9df7b27e6e8e26dce9a1f315568c4ce1ea337e8182e74830bafbc83fa4fdf452',
        'cd1072a611d348b6d4d082a3cfbbdc6f5ad310e4a3ea547f934fb7b9a8ccd455',
        '634f75d49fbaa93ed99fcad4338553896c086ee22815e3f7fcb49c5b83127c6f',
        '0e7914dc2db9e958a89488260960575954574636057b1f417a243781df046bef',
    ),
    'refine_pipeline.py': (
        '4c18c7778d1191dc58920a7381194ead411f03493d3ad69f5e04b1076f7213f2',
        '5509c66056f039dfc03e0c441a2aa170f5addcde0938b697a7907c68168e3c6d',
        '6ad937c38d2467e162745535e3dbd016438f74f096ad1196c7b1d1096e3d70f0',
        '57f3c3c85e0a7dc5c013fe26046f5e882f5a964ce58942e5b3bbe471b863c44e',
        '0c9683e77a98f21877a1677e45c08e9414a8098bee99731fee865ddb9874b5b4',
    ),
    'refine_dispatch_budget.py': (None,'3eea4363bd04eb2a7e9f39d81742d81db63a6fa959abf47bda9886678a4bb884'),
    'refine_queue.py': ('d21864417bcaefd791586f649886aa9e92c6841dccf7f7da24a90a370e8d43b2',),
    'refine_budget.json': ('61b5f89498c7f034893d1acca5a8e91c7bdcd9ad8c27ca469635b8aea11ae1ac',),
    'cross_foot_repair.py': (None,'f981b8a0838b7beb75156b2704d5ec29e9966f586e9221c0c77b74a39d46757e'),
    'contact_timing.py': (None,'ee73c0f5fcebb3cfb2ddc7a8c1be358d17643b617b08f65a8d8063bf13182fef'),
    'retiming.py': ('bb4b08b97612ec8734ceee8f5d535e6a752867022a2025802dbeb10ea6806332',),
}
_POOL_FIXED = {
    **{name:_RECHECK_COMPATIBLE[name][1]
       for name in ('v8/audit.py','model_cache.py','foot_collision.py','contact_routed_initial.py')},
    'refine_pipeline.py': '55c8998a08ccf19494f36632799fd9a360338616d8d2a23be42d12c611c68693',
    'retiming.py': 'b83626cf6db78716a47754377e5eedb76e3bcb5ee085dbc9b41c368171452de0',
}


def _pool_compatible(before, after):
    from .cli import sha
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    if not changed or not changed <= _POOL_PREVIOUS.keys() or 'memory_pool_batch.py' not in after:
        return False
    root = Path(__file__).parent
    for name in changed:
        expected = _POOL_FIXED.get(name) or sha(root/name)
        if before.get(name) not in _POOL_PREVIOUS[name] or after.get(name) != expected:
            return False
    return True


def locate(value):
    p=Path(value).resolve()
    if p.is_file():p=p.parent
    for folder in (p,p/'v8',p/'frax'):
        if (folder/'pre_final_retime').is_dir():
            raise ValueError('Completed final timing uses a different time grid; use a fresh conversion')
        if (folder/'refine_state.json').exists():return folder
        if (folder/'identity.json').exists() and (folder/'run.json').exists():return folder
    raise ValueError('continue-from requires a verified fast result or refine checkpoint')


def verify(folder,identity,recheck=False):
    from .cli import sha
    old=json.loads((folder/'identity.json').read_text())
    if recheck:
        initial={'scene_support.py','robot_profile.py','smplx_input.py','smplx_source.py','source_integrity.py','source_support.py','human_motion_constants.py','v8/bvh.py','foot_transition.py','contact_phase.py','phase_height.py','phase_path.py','ik.py','pelvis_priority.py','kinematics.py','shoulder_reference.py','contact_routed_initial.py',
                 'frax_foot_constraints.py','frax_arm_constraints.py','airborne.py',
                 'v8/contacts.py','v8/multicontact.py','retiming.py'}
        safety={'acceptance.py','quality_pose.py','quality_v8.py','audit_cache.py','support_quality.py','temporal_quality.py','ankle_collision.py',
                'v8/audit.py','v8/anatomy.py','v8/robot.py','v8/self_collision.py','foot_collision.py','model_cache.py'}
        names=set(old['implementation'])|set(identity['implementation'])
        for name in names:
            before, after = old['implementation'].get(name), identity['implementation'].get(name)
            previous, expected = _RECHECK_COMPATIBLE.get(name, ((), None))
            if before in previous and after == expected:
                continue
            defines_initial = name in initial or name.startswith(('full_chain', 'frax_')) and name.endswith('.py')
            if defines_initial and old['implementation'].get(name)!=identity['implementation'].get(name):
                raise ValueError('Recheck initial solver changed; recompute initial: '+name)
            if (name.endswith('.json') and name != 'refine_budget.json') or name in safety:
                if old['implementation'].get(name)!=identity['implementation'].get(name):
                    raise ValueError('Recheck safety policy changed: '+name)
        run=json.loads((folder/'run.json').read_text())
        for name in ('motion.json','motion.diagnostics.npz','quality.json'):
            if run.get('files',{}).get(name)!=sha(folder/name):raise ValueError('Recheck artifact hash changed: '+name)
    pool_migration = (old['conversion_contract'].get('refine_flow') == 'A'
                      and (folder/'refine_state.json').is_file()
                      and _pool_compatible(old['implementation'],identity['implementation']))
    for key in ('input_sha256','assets','implementation','frax_implementation','runtime'):
        if (recheck or pool_migration) and key=='implementation':continue
        if old[key]!=identity[key]:raise ValueError('Continuation identity mismatch: '+key)
    fast=old['conversion_contract'].get('profile')=='fast'
    def contract(value):
        result=dict(value['conversion_contract']);result.pop('repair_budget',None)
        if fast:
            for key in ('profile','unified_repair','allow_local_slowdown','refine_initial_policy','refine_flow'):
                result.pop(key,None)
        else:
            result.setdefault('refine_flow',LEGACY_CHECKPOINT_FLOW)
        return result
    if identity['conversion_contract'].get('profile')!='refine':raise ValueError('Continuation target must be refine')
    if contract(old)!=contract(identity):raise ValueError('Continuation conversion contract changed')
    if fast:
        run=json.loads((folder/'run.json').read_text())
        if run.get('status')!='complete':raise ValueError('Fast source is not complete')
        for name in ('motion.pkl','motion.initial.npz','motion.json','motion.diagnostics.npz','quality.json'):
            if run.get('files',{}).get(name)!=sha(folder/name):raise ValueError('Fast artifact hash changed: '+name)
        return {'from_fast':True}
    from .refine_checkpoint import artifact
    state=json.loads((folder/'refine_state.json').read_text())
    if state['baseline_sha256']!=sha(folder/'frax_baseline/motion.pkl'):raise ValueError('Fixed Frax reference hash changed')
    if state.get('schema')!=3:raise ValueError('Continuation requires current dependency queue schema')
    artifact(folder,state,'motion.pkl')
    artifact(folder,state,'avoidance_reference.npy')
    return state


class Initial:
    improved=False
    strict=True
    def __init__(self,folder):
        self.folder=folder
        self.meta=json.loads((folder/'motion.json').read_text())
        self.v8_refine_checkpoint=self.meta.get('pose_solver')=='v8_trf'
        self.report=dict(self.meta.get('frax') or {})
        self.report['continued_without_ik']=True
    def __call__(self,robot,t,cfg,progress):
        from .initial_reuse import target_digest
        if target_digest(t)!=self.meta['prepared_targets_sha256']:raise ValueError('Continuation targets changed')
        baseline=self.folder/'frax_baseline' if (self.folder/'refine_state.json').exists() else self.folder
        with np.load(baseline/'motion.initial.npz',allow_pickle=False) as data:
            if not np.array_equal(data['joint_names'],robot.names):raise ValueError('Continuation joints changed')
            return data['x'].copy(),data['solver_records'].copy()
