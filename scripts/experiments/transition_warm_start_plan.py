"""Fixed follow-up budget and fresh confirmation expressions for the warm start."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.probes.transition_validation_plan import plan as previous_plan, ENTITY
from vision_memory.training.latent_bank_unet import stable_seed

TRAINING_COMMIT = 'd9a1a117cd497ad72d5bcc1630d657a43cebd611'
PARENT_CHECKPOINT = 'b4251975684314171ae35bfbd1db2e4009b14a39eeb5e5127834824fd6d8cdf1'
PARENT_RESULT = '3d747a7c5ba97430587aa11d4787e708df7bc2ebfee8430c4a358ee7ea28f8c8'
BANK_SHA = '962f02846ed1a1933e6c219604bc22ee520e28f2dfe2721e26f111dc36ea122e'


def all_noise_seeds(value):
    return {seed for case in value['single_writes'] for seed in case['noise_seeds']} | {
        step['noise_seed'] for chain in value['rgb_chains'] for step in chain['steps']}


def plan():
    seed = 20260914
    validation = previous_plan(seed)
    writes = {state: (f'Store {state} as the music preference for {ENTITY}, replacing any earlier choice.',
                      f"Set {ENTITY}'s current music preference to {state}.") for state in ('ambient', 'jazz')}
    writes['clear'] = (f'Remove the music preference stored for {ENTITY}; discard the previous selection.',
                       f"Forget {ENTITY}'s saved music choice so that no music option is selected.")
    noops = (f'Record that {ENTITY} passed a routine inspection, keeping the saved music preference as it was.',
             f'Inventory information changed for {ENTITY}; preserve its stored music selection.')
    for case in validation['single_writes']:
        if case['style'].startswith('new_wording_'):
            case['event_text'] = writes[case['state']][int(case['style'][-1]) - 1]
    for chain in validation['rgb_chains']:
        for step in chain['steps']:
            if step['event_style'].startswith('new_wording_'):
                index = int(step['event_style'][-1]) - 1
                step['event_text'] = noops[index] if step['operation'] == 'noop' else writes[step['expected_state']][index]
    forbidden = all_noise_seeds(previous_plan(20260913))
    for training_seed in (20260913, seed):
        forbidden.update(stable_seed(training_seed, 'training-noise', i) for i in range(14000))
        forbidden.update(stable_seed(training_seed, 'heldout-evaluation-noise', i) for i in range(8))
    if all_noise_seeds(validation).intersection(forbidden):
        raise ValueError('Warm-start confirmation reused a prior or training noise')
    return {'training_commit': TRAINING_COMMIT, 'parent_checkpoint_sha256': PARENT_CHECKPOINT,
        'parent_result_sha256': PARENT_RESULT, 'bank_sha256': BANK_SHA,
        'additional_optimizer_steps': 2880, 'training_seed': seed, 'gradient_accumulation_steps': 4,
        'additional_draws': 11520, 'draws_per_condition': 256, 'conditions': 45, 'semantic_questions': 1,
        'optimizer': {'kind': 'fresh AdamW', 'lr': 5e-5, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 1e-4, 'clip_norm': 1.},
        'flow_protocol': 'official', 'trainable_scope': 'full_unet', 'model_variant': 'base',
        'native_steps': 28, 'guidance_scale': 1., 'image_guidance_scale': 1.,
        'development_noise_seeds': [stable_seed(seed, 'heldout-evaluation-noise', i) for i in range(4)],
        'development_matched_rows_per_phase': 900, 'development_raw_rows_per_phase': 1350,
        'baseline': 'actual new-seed evaluation of sealed parent parameters before additional optimization',
        'data_change': 'none: same45-condition bank and three teachers; no validation images or expressions added',
        'selection': 'fixed additional2880-update endpoint, no best-checkpoint or early-score selection',
        'validation': validation,
        'scope': 'one entity and three states; this follow-up cannot establish unseen-entity or multi-fact usability'}


if __name__ == '__main__':
    value = plan()
    previous = previous_plan(20260913)
    old_expressions = {case['event_text'] for case in previous['single_writes'] if 'event_text' in case} | {
        step['event_text'] for chain in previous['rgb_chains'] for step in chain['steps'] if 'event_text' in step}
    bank = json.loads((ROOT / 'reports/official-alignment-results-20260913/transition-wording-bank-manifest.json').read_bytes())
    old_expressions.update(group['event_text'] for group in bank['groups'])
    fresh = {case['event_text'] for case in value['validation']['single_writes'] if 'event_text' in case} | {
        step['event_text'] for chain in value['validation']['rgb_chains'] for step in chain['steps'] if 'event_text' in step}
    if len(fresh) != 8 or fresh.intersection(old_expressions):
        raise ValueError('Require eight expressions absent from the bank and prior confirmation')
    output = ROOT / 'reports/official-transition-warm-start-plan-20260913.json'
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'training_updates': value['additional_optimizer_steps'], 'new_event_expressions': len(fresh),
        'single_images': sum(len(case['noise_seeds']) for case in value['validation']['single_writes']),
        'chains': len(value['validation']['rgb_chains'])}))
