"""Recompute all saved branch velocities and source/noise inputs on an independent CPU."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','run','first-step-run','new-training-run','output-prefix'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source_root,text=True).strip()!='105f52140f80f562e8d077422fef480931fdd506'
            or subprocess.check_output(['git','status','--porcelain'],cwd=args.source_root,text=True).strip()):
        raise ValueError('Require the clean fixed diagnostic source')
    sys.path[:0]=[str(args.source_root),str(args.source_root/'src')]
    import torch
    torch.set_num_threads(1)
    from scripts.reporting.collect_transition_endpoint import read,jsonl,sha
    from scripts.probes.native_training_branch_parity import relative_errors,NEW_RUNTIME_SHA,FIRST_STEP_COMPLETE_SHA,TOLERANCE
    from vision_memory.repro import canonical_tensor_sha256
    complete=read(args.run/'complete.json')
    if sha(args.first_step_run/'complete.json')!=FIRST_STEP_COMPLETE_SHA or sha(args.new_training_run/'train/runtime.json')!=NEW_RUNTIME_SHA:
        raise ValueError('Prior evidence or actual new training conditions changed')
    prior_complete=read(args.first_step_run/'complete.json')
    runtime=read(args.new_training_run/'train/runtime.json')
    for name,digest in complete['artifact_hashes'].items():
        if sha(args.run/name)!=digest:raise ValueError('Diagnostic artifact changed: '+name)
    identity=read(args.run/'identity.json')
    if (identity!=complete['identity'] or identity['optimizer_updates']!=0 or identity['relative_tolerance']!=TOLERANCE
            or identity['new_training_runtime_sha256']!=NEW_RUNTIME_SHA or identity['first_step_complete_sha256']!=FIRST_STEP_COMPLETE_SHA):
        raise ValueError('Wrong frozen diagnostic identity')
    prior_rows=jsonl(args.first_step_run/'cells.jsonl')
    if sha(args.first_step_run/'cells.jsonl')!=prior_complete['artifact_hashes']['cells.jsonl']:
        raise ValueError('Prior full matrix changed')
    prior={(row['question_id'],row['noise_seed']):row for row in prior_rows}
    rows=jsonl(args.run/'cells.jsonl')
    if len(rows)!=302 or len(prior)!=302 or complete['cells']!=302:
        raise ValueError('Incomplete registered matrix')
    seen=set()
    peaks={name:{'relative_l2':0.,'relative_max':0.} for name in ('train_vs_native_branch','train_vs_native_cfg','integer_vs_float')}
    passing=0
    for row in rows:
        key=row['question_id'],row['noise_seed']
        if key in seen or key not in prior or row['artifact']!=prior[key]['artifact']:
            raise ValueError('Missing, repeated or replaced diagnostic cell')
        seen.add(key)
        name=row['artifact']
        if sha(args.first_step_run/name)!=prior_complete['artifact_hashes'][name]:raise ValueError('Prior velocity tensor changed')
        old=torch.load(args.first_step_run/name,map_location='cpu',weights_only=True)
        values=torch.load(args.run/name,map_location='cpu',weights_only=True)
        if any(value.dtype!=torch.float32 or not torch.isfinite(value).all() for value in values.values()):
            raise ValueError('Expected actual finite FP32 tensors')
        source_input=values['native_conditional_input']
        noise,source=source_input.chunk(2,dim=3)
        actual_noise=torch.randn(noise.shape,generator=torch.Generator().manual_seed(row['noise_seed']),dtype=torch.float32)
        binding=runtime['additional_protocol_binding']['source_bindings'][key[0]]
        if (not torch.equal(noise,old['noise']) or not torch.equal(noise,actual_noise)
                or canonical_tensor_sha256(source)!=binding['official_source_sha256']
                or row['condition_sha256']!=runtime['condition_sha256'][key[0]]
                or not row['native_encoder_and_source_input_bitwise_equal'] or not row['native_first_step_bitwise_reproduced']
                or not torch.equal(values['native_post_cfg'],old['native_sigma1'])
                or not torch.equal(values['native_next_state'],old['native_next_state'])):
            raise ValueError('Actual source/noise/condition or native reference differs')
        sigmas=runtime['effective_inference_sigmas'][key[0]]
        update=noise+(torch.tensor(sigmas[1])-torch.tensor(sigmas[0]))*values['native_post_cfg']
        if not torch.equal(update,values['native_next_state']):raise ValueError('Actual native Euler update differs')
        pairs={'train_vs_native_branch':('training_integer1000','native_text_branch'),
            'train_vs_native_cfg':('training_integer1000','native_post_cfg'),
            'integer_vs_float':('training_integer1000','training_float1000')}
        errors={name:relative_errors(values[left],values[right]) for name,(left,right) in pairs.items()}
        recorded=row['relative_errors']
        if set(recorded)!=set(errors) or any(set(recorded[name])!=set(error) or any(
                not math.isclose(value,recorded[name][metric],rel_tol=1e-10,abs_tol=1e-14)
                for metric,value in error.items()) for name,error in errors.items()):
            raise ValueError('Independent CPU velocity reduction differs')
        passed=all(value<=TOLERANCE for error in errors.values() for value in error.values())
        if passed!=row['within_registered_tolerance']:raise ValueError('Recorded pass contradicts actual velocity tensors')
        passing+=int(passed)
        for name,error in errors.items():
            for metric,value in error.items():peaks[name][metric]=max(peaks[name][metric],value)
    if seen!=set(prior) or complete['all_velocity_comparisons_within_tolerance']!=(passing==302):
        raise ValueError('Wrong complete outcome or matrix coverage')
    summary={'complete_sha256':sha(args.run/'complete.json'),'identity':identity,'cells':302,'passing_cells':passing,
        'max_errors':peaks,'current_and_prior_pts_checked':604,'all_velocity_comparisons_within_tolerance':passing==302,
        'scope':'Independent CPU recheck of all saved velocities, real source/noise inputs and Euler updates. Encoder/mask equality was observed by the frozen GPU hook. No Reader accuracy, final training or general-memory claim.'}
    summary_path=Path(str(args.output_prefix)+'-summary.json')
    archive_path=Path(str(args.output_prefix)+'-evidence.tgz')
    if summary_path.exists() or archive_path.exists():raise ValueError('Collection output already exists')
    summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    with tarfile.open(archive_path,'w:gz') as archive:
        for name in ('identity.json','complete.json','cells.jsonl'):
            archive.add(args.run/name,arcname=name)
        archive.add(args.first_step_run/'complete.json',arcname='prior-first-step-complete.json')
        archive.add(args.first_step_run/'cells.jsonl',arcname='prior-first-step-cells.jsonl')
        archive.add(args.new_training_run/'train/runtime.json',arcname='native-training-runtime.json')
        archive.add(summary_path,arcname='verified-summary.json')
    print(json.dumps({'summary_sha256':sha(summary_path),'archive_sha256':sha(archive_path),
        'passing_cells':passing,'cells':302,'max_errors':peaks}))


if __name__=='__main__':
    main()
