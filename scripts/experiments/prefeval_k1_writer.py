"""K1 shared Writer: reuse official FM helpers and unchanged native RGB inference."""
import argparse
import gc
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import torch
from PIL import Image
from scripts.experiments.prefeval_k1_data import load_records, load_training_records, event_text, sha
from scripts.experiments.prefeval_k1_teacher import save_json, atomic_save
from scripts.eval.prefeval_rgb import append
from scripts.train.latent_r11_vae_oracle import _save_image
from vision_memory.dreamlite import DifferentiableDreamLiteMobileSampler
from vision_memory.dreamlite.conditioning import encode_native_base_edit_condition
from vision_memory.dreamlite.native_base import NativeBaseEditSampler
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
from vision_memory.training.latent_bank_unet import official_flow_bridge, predict_velocity, stable_seed, OFFICIAL_REFERENCE_COMMIT
from vision_memory.training.checkpoint import save_training_checkpoint, load_training_checkpoint, load_trainable_weights
from vision_memory.repro import configure_strict_cuda_determinism

def load_pipe(args):
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=args.official_source, text=True).strip()
    assert commit == OFFICIAL_REFERENCE_COMMIT
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'], cwd=args.official_source, text=True).strip()
    sys.path.insert(0, str(args.official_source))
    from dreamlite import DreamLitePipelineLoRA
    pipe = DreamLitePipelineLoRA.from_pretrained(args.base, local_files_only=True, torch_dtype=torch.float32).to(args.device)
    for module in [pipe.unet, pipe.vae, pipe.text_encoder]:
        module.eval().requires_grad_(False)
    pipe.unet.requires_grad_(True)
    load_trainable_weights(args.checkpoint, trainable_module=pipe.unet)
    pipe.set_progress_bar_config(disable=True)
    return pipe

@torch.no_grad()
def encode_source(pipe, image, device):
    return pipe.prepare_image_latents(pipe.image_processor.preprocess(image), dtype=torch.float32, device=device)

@torch.no_grad()
def cache_condition(pipe, path, text, device):
    image = Image.open(path).convert('RGB') if path else Image.new('RGB', (1024,1024), (128,128,128))
    assert image.size == (1024,1024)
    source = encode_source(pipe, image, device)
    c = encode_native_base_edit_condition(pipe, image, text, device=device, dtype=torch.float32)
    return {'source': source.cpu(), 'embeds': c.prompt_embeds.cpu(), 'mask': c.attention_mask.cpu()}

def train(args, pipe, rows):
    assert args.split in {'pilot', 'train'}, 'No optimized targets allowed for dev/eval'
    assert all(0 < step <= args.steps for step in args.snapshot_steps)
    if args.stage == 'retain':
        assert args.sources is not None
    args.output.mkdir(parents=True, exist_ok=True)
    targets, cache = {}, {}
    target_hashes = {}
    for i, row in enumerate(rows):
        pid = row['base_pair_id']
        parent = args.teachers / pid.replace(':','_')
        done = json.loads((parent / 'complete.json').read_text())
        assert done['step'] == 288 and done['binding']['arm'] == args.arm
        assert done['latent_sha256'] == sha(parent / 'latent.pt')
        targets[pid] = torch.load(parent / 'latent.pt', map_location='cpu', weights_only=True)
        target_hashes[pid] = done['latent_sha256']
        cache[pid,0] = cache_condition(pipe, None, event_text(row['history'][:2]), args.device)
        if args.stage == 'retain':
            for position in range(1,11):
                # prefix-00 contains initial write; prefix-(p-1) is input to distractor p.
                source_position = 0 if args.retain_source_mode == 'initial' else position-1
                path = args.sources / pid.replace(':','_') / 'seed-0' / f'prefix-{source_position:02d}.png'
                cache[pid,position] = cache_condition(pipe, path,
                    event_text(row['history'][position*2:position*2+2]), args.device)
        print(json.dumps({'cached': i+1, 'total': len(rows), 'stage': args.stage}), flush=True)
    manifest = {'arm': args.arm, 'stage': args.stage, 'steps': args.steps, 'effective_batch': 4,
        'split': args.split, 'snapshot_steps': sorted(set(args.snapshot_steps)),
        'parent_sha256': sha(args.checkpoint), 'targets': target_hashes, 'source_root': str(args.sources),
        'flow': 'official target/noise; source condition only', 'seed': 20260924,
        'official_commit': OFFICIAL_REFERENCE_COMMIT,
        'implementation_sha256': sha(Path(__file__))}
    if args.retain_source_mode == 'initial':
        manifest['retain_source_mode'] = 'initial_student_png_for_all_distractor_positions'
    if args.retain_target_mode == 'source':
        manifest['retain_target_mode'] = 'official_vae_encode_of_actual_source_png'
        manifest['source_png_hashes'] = {row['base_pair_id']:sha(args.sources/row['base_pair_id'].replace(':','_')/'seed-0'/'prefix-00.png') for row in rows}
    save_json(args.output / 'manifest.json', manifest)
    predictor = DifferentiableDreamLiteMobileSampler.from_pipeline(pipe, checkpoint_unet=False)
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=5e-5, betas=(.9,.999), eps=1e-8, weight_decay=1e-4)
    checkpoint = args.output / 'resume.pt'
    first = 0
    if checkpoint.exists():
        saved = load_training_checkpoint(checkpoint, trainable_module=pipe.unet, optimizer=optimizer, expected_manifest=manifest)
        first = saved['optimizer_step']
    started = time.monotonic()
    n = len(rows)
    for step in range(first, args.steps):
        optimizer.zero_grad(set_to_none=True)
        values = []
        for micro in range(4):
            draw = step*4 + micro
            # Retain stage alternates initial/retain draws; each lane balances preference independently.
            lane_draw = draw//2 if args.stage == 'retain' else draw
            cycle, offset = divmod(lane_draw,n)
            order = torch.randperm(n, generator=torch.Generator().manual_seed(stable_seed(20260924,'order',cycle))).tolist()
            row = rows[order[offset]]
            pid = row['base_pair_id']
            position = 1 + cycle % 10 if args.stage == 'retain' and draw % 2 else 0
            c = cache[pid,position]
            source = c['source'].to(args.device)
            target = (source.detach() if args.retain_target_mode == 'source' and position > 0
                      else targets[pid].to(args.device))
            rng = torch.Generator().manual_seed(stable_seed(20260924,'sigma',draw))
            sigma = float(torch.rand((),generator=rng))
            generator = torch.Generator(device=args.device).manual_seed(stable_seed(20260924,'noise',draw))
            noise = torch.randn(target.shape, generator=generator, device=args.device, dtype=target.dtype)
            state, velocity = official_flow_bridge(noise,target,sigma)
            pred = predict_velocity(predictor,state,source,sigma,c['embeds'].to(args.device),c['mask'].to(args.device),integer_timestep=True)
            loss = (pred.float()-velocity.float()).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite FM loss')
            (loss/4).backward()
            values.append({'pair_id':pid,'position':position,'sigma':sigma,'mse':float(loss.detach())})
        norm = torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        append(args.output/'optimization.jsonl', {'step':step+1,'draws':values,'grad_norm':float(norm),'seconds':time.monotonic()-started})
        if (step+1)%32==0:
            print(json.dumps({'step':step+1,'mse':sum(x['mse'] for x in values)/4,'seconds':time.monotonic()-started}),flush=True)
        if (step+1)%128==0 or step+1==args.steps:
            save_training_checkpoint(checkpoint,trainable_module=pipe.unet,optimizer=optimizer,epoch=0,
                episode_cursor=(step+1)*4,optimizer_step=step+1,manifest=manifest)
        if step+1 in args.snapshot_steps:
            save_inference_checkpoint(args.output/f'checkpoint-step-{step+1:06d}.pt', pipe, step+1, manifest)
    # A compact inference checkpoint excludes optimizer; resumes keep the separate complete checkpoint.
    save_inference_checkpoint(args.output/'checkpoint-final.pt', pipe, args.steps, manifest)
    save_json(args.output/'complete.json',{'steps':args.steps,'checkpoint_sha256':sha(args.output/'checkpoint-final.pt')})


def save_inference_checkpoint(path, pipe, step, manifest):
    atomic_save(path, {'schema_version':1,
        'trainable_state':{k:p.detach().cpu() for k,p in pipe.unet.named_parameters()},
        'optimizer_step':step,'manifest':manifest})

@torch.no_grad()
def rollout(args, pipe, rows):
    pipe.unet.requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)
    binding = {'checkpoint_sha256':sha(args.checkpoint),'split':args.split,
        'steps':28,'cfg':1,'noise_chains':args.noise_chains,'inter_turns':args.inter_turns,
        'state':'only reopened uint8 RGB PNG; fresh Gaussian each write'}
    if args.probe_initial_sources:
        binding['probe_initial_sources'] = str(args.probe_initial_sources)
        binding['scope'] = 'fixed_training_source_one_step_probe_not_new_initial_write'
    if args.split=='official':
        binding['benchmark_history_sha256']=sha(args.history_file)
        binding['history_protocol']=rows[0]['history_protocol']
    save_json(args.output/'manifest.json',binding)
    for row in rows:
        pid = row['base_pair_id']
        for chain in range(args.noise_chains):
            out = args.output / pid.replace(':','_') / f'seed-{chain}'
            out.mkdir(parents=True, exist_ok=True)
            if (out/'complete.json').exists():
                done = json.loads((out/'complete.json').read_text())
                assert done['binding'] == binding
                assert all(sha(out/name)==h for name,h in done['png_hashes'].items())
                continue
            previous = None
            hashes = {}
            for position in range(args.inter_turns+1):
                if position == 0 and args.probe_initial_sources:
                    original=args.probe_initial_sources/pid.replace(':','_')/'seed-0'/'prefix-00.png'
                    done=json.loads((original.parent/'complete.json').read_text())
                    assert sha(original)==done['png_hashes'][original.name]
                    current=out/'prefix-00.png'
                    shutil.copyfile(original,current)
                    hashes[current.name]=sha(current)
                    previous=current
                    continue
                # Only the PNG is carried to the next update, never output.latents.
                source_image = Image.open(previous).convert('RGB') if previous else Image.new('RGB',(1024,1024),(128,128,128))
                source = encode_source(pipe,source_image,args.device)
                text = event_text(row['history'][position*2:position*2+2])
                seed = stable_seed(20260924,f'rollout:{pid}:{chain}',position)
                generator = torch.Generator(device=args.device).manual_seed(seed)
                noise = torch.randn(source.shape,generator=generator,device=args.device,dtype=source.dtype)
                sampler = NativeBaseEditSampler(pipe,source_image=source_image,event_text=text,guidance_scale=1.)
                generated = sampler(source_latents=source,noise_latents=noise,num_steps=28,return_trajectory=False)
                current = out / f'prefix-{position:02d}.png'
                _save_image(current,decode_model_latents_unit_interval(pipe.vae,generated.latents,clamp=True))
                hashes[current.name] = sha(current)
                append(out/'writes.jsonl', {'position':position,'source_png_sha256':sha(previous) if previous else None,
                    'output_png_sha256':hashes[current.name],'noise_seed':seed,'event':text})
                previous = current
                del source, noise, generated, sampler
            save_json(out/'complete.json',{'binding':binding,'png_hashes':hashes})
            print(json.dumps({'completed':pid,'chain':chain,'inter_turns':args.inter_turns}),flush=True)

def main(args):
    configure_strict_cuda_determinism(0)
    if args.retain_source_mode != 'recursive':
        assert args.mode == 'train' and args.stage == 'retain'
    if args.retain_target_mode == 'source':
        assert args.mode == 'train' and args.stage == 'retain' and args.retain_source_mode == 'initial'
    if args.probe_initial_sources:
        assert args.mode == 'rollout' and args.inter_turns == 1 and args.split == 'pilot'
    rows = load_training_records(args.split) if args.mode == 'train' else load_records(args.split,history_file=args.history_file)
    if args.limit:
        rows = rows[:args.limit]
    pipe = load_pipe(args)
    if args.mode == 'train':
        train(args,pipe,rows)
    else:
        rollout(args,pipe,rows)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','rollout'])
    p.add_argument('--arm',choices=['A','B'],required=True)
    p.add_argument('--stage',choices=['write','retain'],default='write')
    p.add_argument('--split',choices=['pilot','train','dev','official'],default='pilot')
    p.add_argument('--history-file',type=Path)
    for name in ['base','official-source','checkpoint','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--teachers',type=Path)
    p.add_argument('--sources',type=Path)
    p.add_argument('--retain-source-mode', choices=['recursive','initial'], default='recursive')
    p.add_argument('--retain-target-mode', choices=['teacher','source'], default='teacher')
    p.add_argument('--probe-initial-sources',type=Path,help='Offline one-step probe from fixed training PNGs; not a fresh rollout.')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--steps',type=int,default=2048)
    p.add_argument('--snapshot-steps',type=int,nargs='+',default=[])
    p.add_argument('--inter-turns',type=int,default=10)
    p.add_argument('--noise-chains',type=int,default=2)
    p.add_argument('--limit',type=int,default=0)
    main(p.parse_args())
