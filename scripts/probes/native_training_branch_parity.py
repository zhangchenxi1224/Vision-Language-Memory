"""Compare the actual native conditional branch with the new single-row training path."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
NEW_RUNTIME_SHA='b97bf55f679cb94805ef769b9e748f09a55182a8f36fb9dcb48c6fcfb7bdd1a0'
FIRST_STEP_COMPLETE_SHA='a469e8a20b22bc5913ea92f8477d9ac1081390b0cd25ac7db744a083db7c621c'
TOLERANCE=2e-6


def relative_errors(actual,reference):
    import torch
    actual,reference=actual.detach().cpu().double(),reference.detach().cpu().double()
    if actual.shape!=reference.shape or not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError('Velocity shape or finiteness differs')
    difference=actual-reference
    return {'relative_l2':float(difference.norm()/reference.norm().clamp_min(1e-12)),
            'relative_max':float(difference.abs().max()/reference.abs().max().clamp_min(1e-12))}


def capture_native(context,noise,condition):
    import torch
    from scripts.probes.logical_first_step_condition import first_native_step
    pipe=context['inference_sampler'].pipeline
    original_encoder=pipe.encode_prompt
    seen=[]
    captured={}
    expected_input=torch.cat((noise,context['source']),dim=3)
    def encoder(*args,**kwargs):
        embeds,mask=original_encoder(*args,**kwargs)
        if not torch.equal(embeds[2:3],condition.prompt_embeds) or not torch.equal(mask[2:3],condition.attention_mask):
            raise ValueError('New cached condition or mask differs from actual native encoder row2')
        seen.append(True)
        return embeds,mask
    def observe(module,args,kwargs,output):
        if captured or args[0].shape[0]!=3 or not torch.equal(args[0][2:3],expected_input):
            raise ValueError('Native conditional source/noise concatenation differs')
        captured.update(native_text_branch=output[0][2:3,...,:noise.shape[-1]].detach().clone(),
                        native_conditional_input=args[0][2:3].detach().clone())
    pipe.encode_prompt=encoder
    handle=pipe.unet.register_forward_hook(observe,with_kwargs=True)
    try:
        native=first_native_step(context['inference_sampler'],source=context['source'],noise=noise)
    finally:
        handle.remove()
        pipe.encode_prompt=original_encoder
    if len(seen)!=1 or not captured:
        raise ValueError('Missing actual native encoder/UNet observations')
    return {**captured,'native_post_cfg':native['velocity'],'native_next_state':native['next_state']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-run','new-training-run','first-step-run','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--deadline-unix',type=float,required=True)
    parser.add_argument('--worker',action='store_true')
    a=parser.parse_args()
    if not math.isfinite(a.deadline_unix) or a.deadline_unix<=time.time():
        raise ValueError('Require a finite future deadline')
    from scripts.train import train_latent_bank_unet as train
    from scripts.reporting.collect_transition_endpoint import read,jsonl,sha
    from scripts.reporting.collect_broader_endpoint import parent_binding
    from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    command=read(a.parent_run/'commands.json')['commands'][-1]
    indices=[i for i,value in enumerate(command) if Path(value).name=='train_latent_bank_unet.py']
    if len(indices)!=1:raise ValueError('Missing unique bound training command')
    args=train.parser().parse_args(command[indices[0]+1:])
    bank,parent_identity,parent_result=parent_binding(a.parent_run,args.bank_manifest,
        logical_sampling_commit='bb34092ab0d1292c87d16d9632716b218f54054b')
    if not a.worker:
        return subprocess.call([sys.executable,'-u',str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],
            env={**os.environ,**snapshot_environment(bank),**REQUIRED_DETERMINISM_ENV,
                 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},timeout=a.deadline_unix-time.time())
    if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=a.expected_commit
            or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()):
        raise ValueError('Require the clean fixed diagnostic checkout')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPU occupied; do not interfere with existing work')
    import torch
    from vision_memory.repro import configure_strict_cuda_determinism,canonical_tensor_sha256
    from vision_memory.training.checkpoint import load_trainable_weights
    from vision_memory.training.latent_bank_unet import predict_velocity
    from vision_memory.dreamlite.conditioning import encode_native_base_edit_condition
    from scripts.train.official_base_runtime import audit_inference_only_runtime
    configure_strict_cuda_determinism(args.seed)
    if torch.cuda.device_count()!=1 or torch.cuda.mem_get_info(0)[0]<70*1024**3:
        raise ValueError('Require one idle H200 with70GiB free')
    args.dreamlite_device=args.reader_device='cuda:0'
    args.colocate_models=True
    if (sha(a.new_training_run/'train/runtime.json')!=NEW_RUNTIME_SHA
            or sha(a.first_step_run/'complete.json')!=FIRST_STEP_COMPLETE_SHA):
        raise ValueError('Sealed new training conditions or prior actual first steps changed')
    new_runtime=read(a.new_training_run/'train/runtime.json')
    first_seal=read(a.first_step_run/'complete.json')
    cells=jsonl(a.first_step_run/'cells.jsonl')
    if sha(a.first_step_run/'cells.jsonl')!=first_seal['artifact_hashes']['cells.jsonl'] or len(cells)!=302:
        raise ValueError('Incomplete fixed first-step matrix')
    indexed={(row['question_id'],row['noise_seed']):row for row in cells}
    if len(indexed)!=302:raise ValueError('Repeated first-step cell')
    a.output.mkdir(parents=True,exist_ok=False)
    runtime=train.load_runtime(args,bank)
    prior_runtime=read(a.parent_run/'train/runtime.json')
    if (runtime['snapshots']!=prior_runtime['snapshots'] or runtime['protocol_binding']!=prior_runtime['additional_protocol_binding']
            or {key:canonical_tensor_sha256(value['condition'].prompt_embeds) for key,value in runtime['contexts'].items()}!=prior_runtime['condition_sha256']):
        raise ValueError('Frozen parent runtime changed')
    checkpoint=a.parent_run/'train/checkpoint-final.pt'
    if sha(checkpoint)!=parent_result['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    loaded=load_trainable_weights(checkpoint,trainable_module=runtime['pipe'].unet)
    if loaded['optimizer_step']!=4832 or loaded['manifest']!={**parent_identity,**prior_runtime}:
        raise ValueError('Frozen checkpoint identity differs')
    del loaded
    for module in (runtime['pipe'].unet,runtime['pipe'].vae,runtime['pipe'].text_encoder,runtime['reader']):
        module.eval().requires_grad_(False)
    frozen=train.frozen_versions(runtime['pipe'],runtime['reader'])
    identity={'probe_commit':a.expected_commit,'source_hashes':train.source_hashes(),'checkpoint_sha256':sha(checkpoint),
        'new_training_runtime_sha256':NEW_RUNTIME_SHA,'first_step_complete_sha256':FIRST_STEP_COMPLETE_SHA,
        'bank_sha256':sha(args.bank_manifest),'cells_expected':302,'optimizer_updates':0,'relative_tolerance':TOLERANCE,
        'scope':'Input/velocity compatibility at fixed sigma1 on all302 prior development cells. Integer1000 is a boundary diagnostic, not a training draw. No Reader scores, optimization, inference policy or running experiment changes.'}
    train.write_json(a.output/'identity.json',identity)
    output_cells=[]
    with torch.no_grad():
        for group in bank['groups']:
            qid=group['question_id']
            context=runtime['contexts'][qid]
            condition=encode_native_base_edit_condition(runtime['pipe'],context['inference_sampler'].source_image,
                group['event_text'],device=runtime['vae_device'],dtype=torch.float32)
            condition_sha=canonical_tensor_sha256(condition.prompt_embeds)
            if condition_sha!=new_runtime['condition_sha256'][qid]:
                raise ValueError('Actual new condition differs from the running four-GPU training condition')
            for reference in (row for row in cells if row['question_id']==qid):
                if time.time()>=a.deadline_unix:raise TimeoutError('Branch diagnostic deadline reached')
                name=reference['artifact']
                if sha(a.first_step_run/name)!=first_seal['artifact_hashes'][name]:raise ValueError('Prior first-step tensor changed')
                prior=torch.load(a.first_step_run/name,map_location='cpu',weights_only=True)
                noise=torch.randn(context['source'].shape,generator=torch.Generator().manual_seed(reference['noise_seed']),dtype=torch.float32)
                if not torch.equal(noise,prior['noise']):raise ValueError('Registered Gaussian noise changed')
                noise=noise.to(context['source'].device)
                observed=capture_native(context,noise,condition)
                if not torch.equal(observed['native_post_cfg'].cpu(),prior['native_sigma1']) or not torch.equal(observed['native_next_state'].cpu(),prior['native_next_state']):
                    raise ValueError('Actual native first step differs from the prior real diagnostic')
                for label,integer in (('training_integer1000',True),('training_float1000',False)):
                    observed[label]=predict_velocity(runtime['sampler'],noise,context['source'],1.,
                        condition.prompt_embeds,condition.attention_mask,integer_timestep=integer)
                observed={key:value.cpu() for key,value in observed.items()}
                pairs={'train_vs_native_branch':('training_integer1000','native_text_branch'),
                    'train_vs_native_cfg':('training_integer1000','native_post_cfg'),
                    'integer_vs_float':('training_integer1000','training_float1000')}
                errors={key:relative_errors(observed[left],observed[right]) for key,(left,right) in pairs.items()}
                passed=all(value<=TOLERANCE for error in errors.values() for value in error.values())
                train.atomic_tensor(a.output/name,observed)
                row={'question_id':qid,'noise_seed':reference['noise_seed'],'artifact':name,
                    'condition_sha256':condition_sha,'mask_sha256':canonical_tensor_sha256(condition.attention_mask),
                    'native_encoder_and_source_input_bitwise_equal':True,'native_first_step_bitwise_reproduced':True,
                    'relative_errors':errors,'within_registered_tolerance':passed}
                output_cells.append(row)
                train.append_jsonl(a.output/'cells.jsonl',row)
                print(len(output_cells),qid,passed,errors,flush=True)
    if len(output_cells)!=302:raise ValueError('Incomplete full comparison')
    audit_inference_only_runtime(runtime,frozen)
    runtime['verify_additional_bindings']()
    if train.source_hashes()!=identity['source_hashes']:raise ValueError('Diagnostic source changed')
    train.write_json(a.output/'complete.json',{'identity':identity,'cells':302,
        'all_velocity_comparisons_within_tolerance':all(row['within_registered_tolerance'] for row in output_cells),
        'artifact_hashes':{path.name:sha(path) for path in sorted(a.output.iterdir()) if path.is_file()}})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
