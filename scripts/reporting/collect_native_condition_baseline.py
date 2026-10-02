"""Collect a complete paired native-condition baseline while optimization continues."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','run','reference','output-prefix'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    source=args.source_root
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!='9e27050ea3fe1e7d54fe81714244f93ae07bac81':
        raise ValueError('Require the fixed native-condition verification source')
    if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():
        raise ValueError('Verification source changed')
    sys.path[:0]=[str(source),str(source/'src')]
    from scripts.reporting.collect_transition_endpoint import read,jsonl,sha
    from scripts.reporting.collect_broader_endpoint import registered_protocol,phase_summary,NATIVE_CONDITION_COMMIT,BANK_SHA
    from scripts.experiments.native_condition_protocol import REFERENCE_RESULT
    from scripts.train.train_latent_bank_unet import verify_initialized_baseline_reference
    train=args.run/'train'
    bank=read(args.run/'bank/manifest.json')
    commit,plan,plan_sha=registered_protocol(bank,NATIVE_CONDITION_COMMIT)
    identity=read(train/'identity.json')
    if (sha(args.run/'bank/manifest.json')!=BANK_SHA or sha(args.run/'preregistered-experiment.json')!=plan_sha
            or identity['git_commit']!=commit or identity['prompt_style']!='native_base'
            or identity['steps']!=4832):
        raise ValueError('Registered training identity changed')
    # The verifier writes its report, so run it in temporary storage while
    # reading the actual baseline PTs through a symlink. Do not rewrite a live run.
    with tempfile.TemporaryDirectory(prefix='native-baseline-check-') as temporary:
        current=Path(temporary)
        for name in ('identity.json','runtime.json'):
            shutil.copyfile(train/name,current/name)
        (current/'baseline').symlink_to(train/'baseline',target_is_directory=True)
        gate=verify_initialized_baseline_reference(current,args.reference,REFERENCE_RESULT,native_condition_control=True)
    if gate!=read(train/'baseline-reference-check.json') or len(gate['samples'])!=302:
        raise ValueError('Actual complete baseline gate differs from the training gate')
    phase,_=phase_summary(jsonl(train/'baseline/generations.jsonl'),bank,'baseline')
    initial=read(train/'parallel-initial-parameters.json')
    first=read(train/'parallel-parameters-step-000001.json')
    for proof in (initial,first):
        if not proof['bitwise_rank_agreement'] or len(proof['parameter_sha256_by_rank'])!=4 or len(set(proof['parameter_sha256_by_rank']))!=1:
            raise ValueError('Four-rank parameters differ')
    if initial['parameter_sha256_by_rank']==first['parameter_sha256_by_rank']:
        raise ValueError('The first optimizer update did not change parameters')
    gradient=read(train/'parallel-gradient-preflight.json')
    if (not gradient['passed'] or not gradient['identical_draws_and_losses']
            or not 0<=gradient['gradient_relative_l2_error']<=2e-6
            or not 0<=gradient['gradient_relative_max_error']<=2e-6):
        raise ValueError('Actual gradient parity failed')
    first_metric=read(train/'metrics/step-000001.json')
    if first_metric['optimizer_step']!=1 or first_metric['microbatches']!=gradient['parallel_microbatches']:
        raise ValueError('The actual first update differs from its preregistered draws')
    names=['bank/manifest.json','bank/complete.json','preregistered-experiment.json','commands.json',
        'train/identity.json','train/runtime.json','train/baseline-reference-check.json',
        'train/baseline/complete.json','train/baseline/summary.json','train/baseline/generations.jsonl',
        'train/parallel-initial-parameters.json','train/parallel-gradient-preflight.json',
        'train/parallel-parameters-step-000001.json','train/metrics/step-000001.json']
    hashes={name:sha(args.run/name) for name in names}
    summary={'plan_sha256':plan_sha,'training_commit':commit,'baseline':phase,'reference_gate':gate,
        'initial_parameters':initial,'first_step_parameters':first,'gradient_preflight':gradient,
        'first_update':first_metric,'artifact_hashes':hashes,'current_and_reference_baseline_pts_checked':604,
        'scope':'All302 actual paired baseline PTs and raw records rechecked on CPU. One optimizer update and four-rank agreement verified. Training is ongoing; this is not a final model or functional success.'}
    summary_path=Path(str(args.output_prefix)+'-summary.json')
    archive_path=Path(str(args.output_prefix)+'-evidence.tgz')
    if summary_path.exists() or archive_path.exists():
        raise ValueError('Evidence output already exists')
    summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    with tarfile.open(archive_path,'w:gz') as archive:
        for name in names:
            if sha(args.run/name)!=hashes[name]:
                raise ValueError('Immutable baseline artifact changed while collecting')
            archive.add(args.run/name,arcname=name)
        archive.add(summary_path,arcname='verified-summary.json')
    print(json.dumps({'summary_sha256':sha(summary_path),'archive_sha256':sha(archive_path),
        'baseline_correct_eos':phase['correct_eos'],'paired_pts_checked':604,'first_update_verified':True}))


if __name__=='__main__':
    main()
