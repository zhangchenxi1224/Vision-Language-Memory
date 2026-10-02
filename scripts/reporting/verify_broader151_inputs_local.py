"""Reconstruct all151 downloaded conditions and the actual preregistered plan."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.reporting.collect_transition_endpoint import sha, read
from scripts.experiments.build_broader_writer_bank import merge
from scripts.experiments.broader_writer_protocol import augment, training_plan

directory = ROOT / 'reports/official-alignment-results-20260913'
expected = {
    'historical-five-query-refinement-complete.json': 'e52d4777e99ea6f2dbf8614ce0b0acd2afc2e305382adb2fd781bd596179688d',
    'historical-five-query-refinement-plan.json': '939d1eacd7e3040f5ed06dfe3c48c770385b8cd538dcc5663ff9a91ed84f172c',
    'historical-five-query-refined-bank-manifest.json': '76ecd98f2bea422b756e7691425973ec9694f44fabca5905731c963a0f38b7dd',
    'broader151-bank-manifest.json': 'c27cd65dab809deabb5f2cb08891517d3590244651d08a8c6763c84fea901592',
    'broader151-bank-complete.json': 'e74951a84c76c7774b0d49dcee7e5ad5a2b8e89bcfdab552f30d517cc1a6167b',
    'broader151-preregistered-plan.json': 'c4a6986edf1330e27af5b91e5105ad6e3806040f01d562f791edbd392bd16834',
    'transition-wording-bank-manifest.json': '962f02846ed1a1933e6c219604bc22ee520e28f2dfe2721e26f111dc36ea122e',
    'broader151-gradient-preflight.json': '0b3dac2a8a89c421e9f56c63264c3e333d0aef8304d4abb2316888c14d56a89d',
    'broader151-initial-parameters.json': '4227d316fbdf698b4a99c8682d22cdcdf23ca9ad36cf47791d72ec2f256879a2',
}
for name, digest in expected.items():
    if sha(directory / name) != digest:
        raise ValueError('Downloaded input differs from its remote seal: ' + name)
transition = read(directory / 'transition-wording-bank-manifest.json')
refined = read(directory / 'historical-five-query-refined-bank-manifest.json')
bank = read(directory / 'broader151-bank-manifest.json')
reconstructed = augment(merge(transition, refined, bank['provenance']))
if reconstructed != bank:
    raise ValueError('Actual151-group bank differs from complete semantic reconstruction')
registered = read(directory / 'broader151-preregistered-plan.json')
if registered != training_plan(expected['broader151-bank-manifest.json'], '84cdfdb58ace96954243de5caf427948717c9abf'):
    raise ValueError('Actual preregistration differs from the fixed budget and fresh cases')
gradient = read(directory / 'broader151-gradient-preflight.json')
initial = read(directory / 'broader151-initial-parameters.json')
if (not gradient['passed'] or not gradient['identical_draws_and_losses']
        or gradient['serial_microbatches'] != gradient['parallel_microbatches']
        or gradient['gradient_relative_l2_error'] > 2e-6 or gradient['gradient_relative_max_error'] > 2e-6
        or initial['parameter_sha256_by_rank'] != ['0025dd0c573218179857beaf7e48a4dc7f9d86af5c07056962bea34fb3f6294d'] * 4
        or not initial['bitwise_rank_agreement']):
    raise ValueError('Actual gradient parity or parent parameter loading failed')
result = {'input_sha256': expected, 'conditions_reconstructed': 151, 'semantic_questions': 17,
    'original_transition_groups_unchanged': bank['groups'][:45] == transition['groups'],
    'raw_target_tensor_identities': len({teacher['latent_sha256'] for teacher in bank['teachers']}),
    'historical_refined_questions': 16, 'training_updates_registered': 4832,
    'transition_validation_chains': len(registered['transition_validation']['rgb_chains']),
    'prefix_validation_cases': len(registered['prefix_validation']['cases']),
    'actual_gradient_relative_l2_error': gradient['gradient_relative_l2_error'],
    'initial_parameters_match_parent_on_all_four_ranks': True,
    'scope': 'Downloaded metadata, complete input identities and event/query semantics reverified locally. Actual target tensor bytes and full refinement raw/trajectory archives remain remote; no local numerical target verification or shared-Writer functional result is claimed.'}
(directory / 'broader151-input-local-verification.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result))
