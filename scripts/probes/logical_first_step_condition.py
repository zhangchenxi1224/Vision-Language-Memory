"""Fixed-checkpoint first-step controls for every logical-sampling development cell.

Reproduce the stored native first Euler update before comparing raw training
conditioning at sigma1 and an actual official FM draw at sigma0.999. Zero updates,
no new Reader scoring, no subset selection, and no replacement inference policy.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


class StepCaptured(Exception):
    pass


def first_native_step(native, *, source, noise, condition=None):
    """Capture actual post-CFG velocity and stop after one unchanged Euler update."""
    from scripts.probes.official_base_guidance import TrainingConditionNativeSampler
    sampler = native if condition is None else TrainingConditionNativeSampler(native, condition)
    scheduler = native.pipeline.scheduler
    original = scheduler.step
    result = {}
    def capture(model_output, timestep, sample, **kwargs):
        if result or float(timestep) != 1000.:
            raise ValueError('Require exactly the native sigma1 initial update')
        update = original(model_output, timestep, sample, **kwargs)
        result.update(velocity=model_output.detach().clone(), state=sample.detach().clone(),
                      next_state=update[0].detach().clone())
        raise StepCaptured()
    scheduler.step = capture
    try:
        try:
            sampler(source_latents=source, noise_latents=noise, num_steps=28)
        except StepCaptured:
            pass
        else:
            raise RuntimeError('Native sampler did not execute the first-step capture')
    finally:
        scheduler.step = original
    if not result:
        raise RuntimeError('Missing first native step')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    if not math.isfinite(a.deadline_unix) or a.deadline_unix <= time.time():
        raise ValueError('Require a finite future deadline')
    from scripts.train import train_latent_bank_unet as train
    from scripts.reporting.collect_transition_endpoint import read, sha
    from scripts.reporting.collect_broader_endpoint import parent_binding
    from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    command = read(a.parent_run / 'commands.json')['commands'][-1]
    entries = [i for i,v in enumerate(command) if Path(v).name == 'train_latent_bank_unet.py']
    if len(entries) != 1:
        raise ValueError('Require the bound training command')
    args = train.parser().parse_args(command[entries[0] + 1:])
    bank, parent_identity, parent_result = parent_binding(a.parent_run, args.bank_manifest,
        logical_sampling_commit='bb34092ab0d1292c87d16d9632716b218f54054b')
    if not a.worker:
        return subprocess.call([sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
            env={**os.environ, **snapshot_environment(bank), **REQUIRED_DETERMINISM_ENV,
                 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1'}, timeout=a.deadline_unix-time.time())
    if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=a.expected_commit
            or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()):
        raise ValueError('Require the registered clean probe source')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPU occupied; preserve the running complete validation')
    import torch
    from vision_memory.repro import configure_strict_cuda_determinism, canonical_tensor_sha256
    from vision_memory.training.checkpoint import load_trainable_weights
    from vision_memory.training.latent_bank_unet import official_flow_bridge, predict_velocity
    from scripts.train.official_base_runtime import audit_inference_only_runtime
    configure_strict_cuda_determinism(args.seed)
    if torch.cuda.device_count()!=1 or torch.cuda.mem_get_info(0)[0] < 70*1024**3:
        raise ValueError('Require one idle H200 with70GiB free')
    args.dreamlite_device=args.reader_device='cuda:0'
    args.colocate_models=True
    a.output.mkdir(parents=True,exist_ok=False)
    runtime=train.load_runtime(args,bank)
    parent_runtime=read(a.parent_run/'train/runtime.json')
    if (runtime['snapshots']!=parent_runtime['snapshots']
            or runtime['protocol_binding']!=parent_runtime['additional_protocol_binding']
            or {k:canonical_tensor_sha256(v['condition'].prompt_embeds) for k,v in runtime['contexts'].items()}
                !=parent_runtime['condition_sha256']):
        raise ValueError('Runtime or cached raw training conditions changed')
    checkpoint=a.parent_run/'train/checkpoint-final.pt'
    if sha(checkpoint)!=parent_result['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    loaded=load_trainable_weights(checkpoint,trainable_module=runtime['pipe'].unet)
    if loaded['optimizer_step']!=4832 or loaded['manifest']!={**parent_identity,**parent_runtime}:
        raise ValueError('Checkpoint cursor or identity changed')
    del loaded
    for module in (runtime['pipe'].unet,runtime['pipe'].vae,runtime['pipe'].text_encoder,runtime['reader']):
        module.eval().requires_grad_(False)
    frozen=train.frozen_versions(runtime['pipe'],runtime['reader'])
    teacher_records={v['teacher_id']:v for v in bank['teachers']}
    parent_phase=a.parent_run/'train/trained'
    seal=read(parent_phase/'complete.json')
    identity={'source_commit':a.expected_commit,'source_hashes':train.source_hashes(),
        'parent_result_sha256':sha(a.parent_run/'train/result.json'),'checkpoint_sha256':sha(checkpoint),
        'bank_sha256':sha(args.bank_manifest),'runtime_sha256':sha(a.parent_run/'train/runtime.json'),
        'optimizer_updates':0,'cells_expected':302,'deadline_unix':a.deadline_unix,
        'arms':['native_sigma1','training_raw_sigma1','training_raw_sigma0.999_integer999'],
        'scope':'All151 conditions and both sealed development noises. First-step diagnostic only; no functional score or inference-policy change.'}
    train.write_json(a.output/'identity.json',identity)
    cells=[]
    with torch.no_grad():
        for group in bank['groups']:
            if len(group['teacher_ids'])!=1:
                raise ValueError('Require the sealed single-target bank')
            record=teacher_records[group['teacher_ids'][0]]
            target_path=Path(record['latent_path'])
            if not target_path.is_absolute(): target_path=args.bank_manifest.parent/target_path
            if sha(target_path)!=record['latent_file_sha256']:
                raise ValueError('Target tensor changed')
            target=torch.load(target_path,map_location='cpu',weights_only=True)
            if record.get('latent_tensor_key'): target=target[record['latent_tensor_key']]
            target=target.to(runtime['vae_device'],dtype=torch.float32)
            context=runtime['contexts'][group['question_id']]
            source=context['source']
            for index in range(2):
                if time.time()>=a.deadline_unix: raise TimeoutError('Probe deadline reached')
                name=hashlib.sha256(group['question_id'].encode()).hexdigest()[:16]+f'-seed-{index:02d}.pt'
                if sha(parent_phase/name)!=seal['artifact_hashes'][name]:
                    raise ValueError('Sealed development trajectory changed')
                prior=torch.load(parent_phase/name,map_location='cpu',weights_only=True)
                noise=torch.randn(source.shape,generator=torch.Generator().manual_seed(prior['noise_seed']),dtype=torch.float32).to(source.device)
                if prior['question_id']!=group['question_id'] or not torch.equal(noise.cpu(),prior['trajectory'][0]):
                    raise ValueError('Condition or registered Gaussian initial state changed')
                native=first_native_step(context['inference_sampler'],source=source,noise=noise)
                if not torch.equal(native['state'],noise) or not torch.equal(native['next_state'].cpu(),prior['trajectory'][1]):
                    raise ValueError('Native first Euler update does not reproduce the sealed trajectory bitwise')
                raw=first_native_step(context['inference_sampler'],source=source,noise=noise,condition=context['condition'])
                if not torch.equal(raw['state'],noise): raise ValueError('Raw control changed initial state')
                state,label=official_flow_bridge(noise,target,.999)
                velocity=predict_velocity(runtime['sampler'],state,source,.999,context['condition'].prompt_embeds,
                    context['condition'].attention_mask,integer_timestep=True)
                velocities={'native_sigma1':native['velocity'],'training_raw_sigma1':raw['velocity'],
                    'training_raw_sigma0.999_integer999':velocity}
                metrics={key:float((value.double()-label.double()).square().mean()) for key,value in velocities.items()}
                train.atomic_tensor(a.output/name,{'noise':noise.cpu(),'native_next_state':native['next_state'].cpu(),
                    'raw_next_state':raw['next_state'].cpu(),**{k:v.cpu() for k,v in velocities.items()}})
                cell={'question_id':group['question_id'],'noise_seed':prior['noise_seed'],'artifact':name,
                    'parent_artifact_sha256':seal['artifact_hashes'][name],'target_sha256':record['latent_sha256'],
                    'native_first_update_bitwise_equal':True,'velocity_mse_against_noise_minus_target':metrics}
                cells.append(cell)
                train.append_jsonl(a.output/'cells.jsonl',cell)
                print(len(cells),group['question_id'],index,metrics,flush=True)
                del prior,native,raw,velocities,velocity
    if len(cells)!=302: raise ValueError('Incomplete development matrix')
    audit_inference_only_runtime(runtime,frozen)
    runtime['verify_additional_bindings']()
    if train.source_hashes()!=identity['source_hashes']:
        raise ValueError('Probe source changed during execution')
    train.write_json(a.output/'complete.json',{'identity':identity,'cells':len(cells),
        'artifact_hashes':{path.name:sha(path) for path in sorted(a.output.iterdir()) if path.is_file()}})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
