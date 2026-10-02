"""Measure full28-step raw-training-condition inference on every fixed151-bank cell."""
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-run','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--deadline-unix',type=float,required=True)
    parser.add_argument('--worker',action='store_true')
    a=parser.parse_args()
    if not math.isfinite(a.deadline_unix) or a.deadline_unix<=time.time():raise ValueError('Require a future finite deadline')
    from scripts.train import train_latent_bank_unet as train
    from scripts.reporting.collect_transition_endpoint import read,jsonl,sha
    from scripts.reporting.collect_broader_endpoint import parent_binding,phase_summary
    from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    from vision_memory.training.latent_bank_unet import load_teacher_bank
    command=read(a.parent_run/'commands.json')['commands'][-1]
    indices=[i for i,value in enumerate(command) if Path(value).name=='train_latent_bank_unet.py']
    if len(indices)!=1:raise ValueError('Missing unique bound training command')
    args=train.parser().parse_args(command[indices[0]+1:])
    bank,identity,parent_result=parent_binding(a.parent_run,args.bank_manifest,
        logical_sampling_commit='bb34092ab0d1292c87d16d9632716b218f54054b')
    _,teachers=load_teacher_bank(args.bank_manifest)
    if not a.worker:
        return subprocess.call([sys.executable,'-u',str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],
            env={**os.environ,**snapshot_environment(bank),**REQUIRED_DETERMINISM_ENV,
                 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},timeout=a.deadline_unix-time.time())
    if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=a.expected_commit
            or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()):
        raise ValueError('Require the registered clean diagnostic source')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPU occupied; retain existing work')
    import torch
    from vision_memory.repro import configure_strict_cuda_determinism,canonical_tensor_sha256
    from vision_memory.training.checkpoint import load_trainable_weights
    from scripts.probes.official_base_guidance import TrainingConditionNativeSampler
    from scripts.train.official_base_runtime import audit_inference_only_runtime
    configure_strict_cuda_determinism(args.seed)
    if torch.cuda.device_count()!=1 or torch.cuda.mem_get_info(0)[0]<70*1024**3:
        raise ValueError('Require one idle H200 with70GiB free')
    args.dreamlite_device=args.reader_device='cuda:0'
    args.colocate_models=True
    a.output.mkdir(parents=True,exist_ok=False)
    runtime=train.load_runtime(args,bank)
    parent_runtime=read(a.parent_run/'train/runtime.json')
    if (runtime['snapshots']!=parent_runtime['snapshots'] or runtime['protocol_binding']!=parent_runtime['additional_protocol_binding']
            or {key:canonical_tensor_sha256(value['condition'].prompt_embeds) for key,value in runtime['contexts'].items()}!=parent_runtime['condition_sha256']):
        raise ValueError('Frozen parent model/source/condition runtime differs')
    checkpoint=a.parent_run/'train/checkpoint-final.pt'
    if sha(checkpoint)!=parent_result['checkpoint_sha256']:raise ValueError('Parent checkpoint changed')
    loaded=load_trainable_weights(checkpoint,trainable_module=runtime['pipe'].unet)
    if loaded['optimizer_step']!=4832 or loaded['manifest']!={**identity,**parent_runtime}:
        raise ValueError('Frozen checkpoint identity differs')
    del loaded
    for context in runtime['contexts'].values():
        if context['inference_sampler'].guidance_scale!=1.:raise ValueError('Parent native guidance differs')
        context['inference_sampler']=TrainingConditionNativeSampler(context['inference_sampler'],context['condition'])
    for module in (runtime['pipe'].unet,runtime['pipe'].vae,runtime['pipe'].text_encoder,runtime['reader']):
        module.eval().requires_grad_(False)
    frozen=train.frozen_versions(runtime['pipe'],runtime['reader'])
    args.output_dir=a.output
    phase='training_raw_guidance1'
    registration={'probe_commit':a.expected_commit,'source_hashes':train.source_hashes(),
        'parent_result_sha256':sha(a.parent_run/'train/result.json'),'checkpoint_sha256':sha(checkpoint),
        'bank_sha256':sha(args.bank_manifest),'parent_runtime_sha256':sha(a.parent_run/'train/runtime.json'),
        'condition_style':'cached upstream raw-event training embedding and mask, repeated into the three native branches',
        'native_steps':28,'guidance_scale':1.,'image_guidance_scale':1.,'optimizer_updates':0,
        'seed':args.seed,'eval_seeds':2,'groups':151,'raw_rows':3020,'matched_rows':1510,
        'scope':'Full fixed observed development matrix, same final bb checkpoint/noise/source/Reader. Only inference condition encoding changes. Does not replace native training/validation or establish unseen tasks and RGB chains.'}
    train.write_json(a.output/'identity.json',registration)
    runtime['should_pause']=lambda:time.time()>=a.deadline_unix-90
    train.evaluate(args,runtime,bank,teachers,phase)
    raw_summary,after=phase_summary(jsonl(a.output/phase/'generations.jsonl'),bank,phase)
    before_dir=a.parent_run/'train/trained'
    before_complete=read(before_dir/'complete.json')
    if sha(before_dir/'generations.jsonl')!=before_complete['artifact_hashes']['generations.jsonl']:
        raise ValueError('Native parent generation records changed')
    native_summary,before=phase_summary(jsonl(before_dir/'generations.jsonl'),bank,'trained')
    changes=Counter()
    for key,previous in before.items():
        current=after[key]
        if key[1]=='matched':changes[f"{int(previous['passed'])}->{int(current['passed'])}"]+=1
        elif current!=previous:raise ValueError('A fixed negative control changed')
    audit_inference_only_runtime(runtime,frozen)
    runtime['verify_additional_bindings']()
    if train.source_hashes()!=registration['source_hashes']:raise ValueError('Probe source changed')
    train.write_json(a.output/'complete.json',{'identity':registration,'phase':phase,
        'phase_complete_sha256':sha(a.output/phase/'complete.json'),'native_summary':native_summary,
        'raw_summary':raw_summary,'matched_changes':changes,'development_all_correct_eos':raw_summary['correct_eos']==1510})
    print({'native_correct':native_summary['correct_eos'],'raw_condition_correct':raw_summary['correct_eos'],
        'changes':dict(changes)},flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
