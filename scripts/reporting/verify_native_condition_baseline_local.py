"""Recount a CPU-verified native-condition baseline archive; tensors stay remote."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.reporting.collect_transition_endpoint import read,jsonl,sha
from scripts.reporting.verify_broader_outputs_local import unpack
from scripts.reporting.collect_broader_endpoint import registered_protocol,phase_summary,NATIVE_CONDITION_COMMIT,BANK_SHA
from scripts.experiments.native_condition_protocol import REFERENCE_RESULT


def verify(archive,digest):
    with tempfile.TemporaryDirectory(prefix='native-baseline-local-',dir=ROOT/'.cache') as temporary:
        root=Path(temporary)
        unpack(archive,digest,root)
        summary=read(root/'verified-summary.json')
        for name,expected in summary['artifact_hashes'].items():
            if sha(root/name)!=expected:
                raise ValueError('Portable baseline artifact differs from CPU collection')
        bank=read(root/'bank/manifest.json')
        commit,plan,plan_sha=registered_protocol(bank,NATIVE_CONDITION_COMMIT)
        if sha(root/'bank/manifest.json')!=BANK_SHA or sha(root/'preregistered-experiment.json')!=plan_sha:
            raise ValueError('Bank or exact native-condition plan changed')
        identity=read(root/'train/identity.json')
        if (identity['git_commit']!=commit or identity['prompt_style']!='native_base'
                or summary['training_commit']!=commit or summary['plan_sha256']!=plan_sha):
            raise ValueError('Native training identity differs')
        phase,_=phase_summary(jsonl(root/'train/baseline/generations.jsonl'),bank,'baseline')
        if json.loads(json.dumps(phase))!=summary['baseline']:
            raise ValueError('Complete local raw recount differs from remote summary')
        seal=read(root/'train/baseline/complete.json')
        expected_samples={hashlib.sha256(group['question_id'].encode()).hexdigest()[:16]+f'-seed-{seed:02d}.pt'
            for group in bank['groups'] for seed in range(2)}
        if set(seal['artifact_hashes'])!=expected_samples|{'generations.jsonl','summary.json'}:
            raise ValueError('Incomplete registered baseline sample hashes')
        for name in ('generations.jsonl','summary.json'):
            if sha(root/'train/baseline'/name)!=seal['artifact_hashes'][name]:
                raise ValueError('Raw baseline differs from its training seal')
        gate=read(root/'train/baseline-reference-check.json')
        if (gate!=summary['reference_gate'] or gate['reference_result_sha256']!=REFERENCE_RESULT
                or set(gate['samples'])!=expected_samples or len(gate['samples'])!=302
                or not all(gate[key] is True for key in ('bitwise_latents_and_images','bitwise_trajectories','identical_raw_generation_records'))):
            raise ValueError('Missing complete paired baseline CPU gate')
        initial=read(root/'train/parallel-initial-parameters.json')
        first=read(root/'train/parallel-parameters-step-000001.json')
        if (initial!=summary['initial_parameters'] or first!=summary['first_step_parameters']
                or set(initial['parameter_sha256_by_rank'])!={'0025dd0c573218179857beaf7e48a4dc7f9d86af5c07056962bea34fb3f6294d'}):
            raise ValueError('Wrong initial or first-update parameters')
        for proof in (initial,first):
            if not proof['bitwise_rank_agreement'] or len(proof['parameter_sha256_by_rank'])!=4 or len(set(proof['parameter_sha256_by_rank']))!=1:
                raise ValueError('Four replicas disagree')
        if initial['parameter_sha256_by_rank']==first['parameter_sha256_by_rank']:
            raise ValueError('No actual initial parameter update')
        gradient=read(root/'train/parallel-gradient-preflight.json')
        metric=read(root/'train/metrics/step-000001.json')
        if (gradient!=summary['gradient_preflight'] or metric!=summary['first_update'] or metric['optimizer_step']!=1
                or metric['microbatches']!=gradient['parallel_microbatches'] or not gradient['identical_draws_and_losses']
                or not gradient['passed'] or not 0<=gradient['gradient_relative_l2_error']<=2e-6
                or not 0<=gradient['gradient_relative_max_error']<=2e-6):
            raise ValueError('First-update or gradient parity evidence differs')
        return {'archive_sha256':digest,'plan_sha256':plan_sha,'baseline':phase,
            'paired_baseline_samples':302,'baseline_raw_rows_recounted':phase['raw_rows'],
            'first_update_verified':True,'four_rank_parameter_agreement':True,
            'scope':'All portable baseline seals and3020 raw rows recounted locally.604 paired PTs were checked by the remote CPU collector; no local tensor-equality claim. Final training and functional validation remain outstanding.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=verify(args.archive,args.sha256)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'raw_rows_recounted':result['baseline_raw_rows_recounted'],
        'baseline_correct_eos':result['baseline']['correct_eos'],'first_update_verified':True}))
