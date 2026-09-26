"""Independent versus student-proximal target construction; Writer FM is untouched."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from PIL import Image
from diffusers import AutoencoderTiny
from diffusers.image_processor import VaeImageProcessor
from scripts.experiments.prefeval_k1_data import load_training_records, official_mcq, option_order, sha
from scripts.experiments.prefeval_k1_teacher import atomic_save, save_json, query_target
from scripts.experiments.prefeval_multitarget_bank import select_rows
from scripts.eval.prefeval_rgb import load_reader, read_png, append
from scripts.train.latent_r11_vae_oracle import VAELatentOracle, _save_image
from vision_memory.reader.qwen3vl import qwen3vl_target_only_ce, R3_QWEN_READER_RESIZE_CONTRACT
from vision_memory.reader.open_eos import assistant_termination_contract
from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.training.latent_bank_unet import stable_seed
from vision_memory.repro import configure_strict_cuda_determinism


@torch.no_grad()
def qualify(path, row, reader, processor, mcq, device):
    pixels = read_png(path)
    checks = []
    # Full factorial acceptance on TRAIN forms only. No O1/O2/dev access.
    for step in range(12):
        query, _, order = query_target(row, 'B', step, mcq)
        correct = 'ABCD'[order.index(0)]
        generated = generate_short_answer(model=reader, processor=processor, image=pixels,
            query=query, device=device, max_new_tokens=32)
        predicted = mcq['extract_choice'](generated['raw'])
        checks.append({'family': f'T{step%3+1}', 'correct_letter': correct,
            'option_order': order, 'predicted_letter': predicted,
            'correct': predicted == correct, 'generated': generated})
    return checks


def main(a):
    configure_strict_cuda_determinism(0)
    torch.set_num_threads(1)
    rows = select_rows(load_training_records('train'), a.ids_file)
    all_tasks = [(arm, row, k) for row in rows for k in range(a.targets) for arm in ['F8', 'S8']]
    tasks = all_tasks[a.shard::a.shards]
    mcq = official_mcq(a.prefeval)
    vae = AutoencoderTiny.from_pretrained(a.base, subfolder='vae', local_files_only=True,
        torch_dtype=torch.float32).to(a.device).eval().requires_grad_(False)
    assert vae.config.scaling_factor == 1 and vae.config.shift_factor == 0
    processor, reader = load_reader(a.reader, a.device)
    termination = assistant_termination_contract(reader, processor)
    gray = VaeImageProcessor(vae_scale_factor=8).preprocess(Image.new('RGB',(1024,1024),(128,128,128)))
    with torch.no_grad():
        gray_latent = vae.encode(gray.to(a.device)).latents.detach()
    binding = {'steps': a.steps, 'lr': .05, 'targets': a.targets,
        'ids_sha256': sha(a.ids_file), 'proximal_lambda': a.proximal_lambda,
        'trust_rms': a.trust_rms, 'independent_init_noise_rms': .1,
        'target_entry': 'direct optimized model latent; same as prior B teacher/FM',
        'reader': str(a.reader), 'base': str(a.base),
        'mcq_source_sha256': sha(a.prefeval/'utils/utils_mcq.py'),
        'implementation_sha256': sha(Path(__file__))}
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for index, (arm, row, k) in enumerate(tasks):
        pid = row['base_pair_id']
        out = a.output/arm/pid.replace(':','_')/f'target-{k}'
        out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():
            done=json.loads((out/'complete.json').read_text())
            assert done['binding']==binding and done['png_sha256']==sha(out/'memory.png')
            assert done['latent_sha256']==sha(out/'latent.pt')
            continue
        if arm == 'S8':
            # Four distinct seeds under each allowed acknowledgment, eight total.
            source=a.starts/f'V{k%2}'/pid.replace(':','_')/f'seed-{k//2}'
            receipt=json.loads((source/'complete.json').read_text())
            assert receipt['png_hashes']['prefix-00.png']==sha(source/'prefix-00.png')
            initial=torch.load(source/'student-latent.pt',map_location=a.device,weights_only=True)
            source_hash=sha(source/'student-latent.pt')
        else:
            gen=torch.Generator(device=a.device).manual_seed(stable_seed(20260927,f'F8-init:{pid}',k))
            initial=gray_latent+.1*torch.randn(gray_latent.shape,device=a.device,generator=gen)
            source_hash=None
        oracle=VAELatentOracle(vae=vae,initial_latent=initial,compute_dtype=torch.float32)
        optimizer=torch.optim.Adam([oracle.latent_fp32],lr=.05,betas=(.9,.999),eps=1e-8)
        first=0
        if (out/'resume.pt').exists():
            ck=torch.load(out/'resume.pt',map_location=a.device,weights_only=False)
            assert ck['binding']==binding and ck['source_hash']==source_hash
            with torch.no_grad(): oracle.latent_fp32.copy_(ck['latent'])
            optimizer.load_state_dict(ck['optimizer'])
            first=ck['step']
            # Discard only this worker's uncommitted log tail after preemption.
            log=out/'optimization.jsonl'
            if log.exists():
                lines=[line for line in log.read_text().splitlines() if json.loads(line)['step']<=first]
                log.write_text(''.join(line+'\n' for line in lines))
        for step in range(first,a.steps):
            optimizer.zero_grad(set_to_none=True)
            pixels=oracle.image()
            quantized=pixels+(pixels.mul(255).round().div(255)-pixels).detach()
            query,target,order=query_target(row,'B',step,mcq)
            ce=qwen3vl_target_only_ce(model=reader,processor=processor,image=quantized[0],
                query=query,target=target+termination['assistant_end_token_text'],device=a.device,
                require_image_grad=True,reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
                deterministic_ce=True)
            distance=(oracle.latent_fp32-initial).square().mean()
            loss=ce.loss+(a.proximal_lambda*distance if arm=='S8' else 0.)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite teacher loss')
            loss.backward()
            grad=oracle.latent_fp32.grad
            if grad is None or not torch.isfinite(grad).all() or not torch.any(grad!=0):
                raise RuntimeError('Missing/nonfinite/zero teacher gradient')
            optimizer.step()
            with torch.no_grad():
                if arm=='S8':
                    delta=oracle.latent_fp32-initial
                    rms=delta.square().mean().sqrt()
                    oracle.latent_fp32.copy_(initial+delta*min(1.,a.trust_rms/max(float(rms),1e-12)))
            record={'arm':arm,'pair_id':pid,'target_index':k,'step':step+1,'ce':float(ce.loss.detach()),
                'distance_rms':float((oracle.latent_fp32-initial).square().mean().sqrt().detach()),
                'family':f'T{step%3+1}','option_order':order,'elapsed_seconds':time.monotonic()-started}
            append(out/'optimization.jsonl',record)
            if (step+1)%24==0 or step+1==a.steps:
                atomic_save(out/'resume.pt',{'latent':oracle.latent_fp32.detach(),
                    'optimizer':optimizer.state_dict(),'step':step+1,'binding':binding,'source_hash':source_hash})
                print(json.dumps(record),flush=True)
        with torch.no_grad(): _save_image(out/'memory.png',oracle.image())
        atomic_save(out/'latent.pt',oracle.latent_fp32.detach().cpu())
        checks=qualify(out/'memory.png',row,reader,processor,mcq,a.device)
        result={'pair_id':pid,'arm':arm,'target_index':k,'binding':binding,'source_hash':source_hash,
            'qualified':all(x['correct'] for x in checks),'checks':checks,
            'png_sha256':sha(out/'memory.png'),'latent_sha256':sha(out/'latent.pt'),
            'distance_rms':float((oracle.latent_fp32-initial).square().mean().sqrt().detach())}
        save_json(out/'complete.json',result)
        print(json.dumps({'completed':index+1,'total':len(tasks),'arm':arm,'pair_id':pid,
            'target_index':k,'passed':sum(x['correct'] for x in checks),'of':12}),flush=True)
        del oracle,optimizer,ce,pixels,quantized,loss,grad,initial
    save_json(a.output/f'finished-{a.shard}.json',{'binding':binding,'tasks':len(tasks)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base','reader','prefeval','starts','ids-file','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--targets',type=int,default=8)
    p.add_argument('--steps',type=int,default=288)
    p.add_argument('--proximal-lambda',type=float,default=.1)
    p.add_argument('--trust-rms',type=float,default=1.)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=4)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    assert args.targets in (1,2,4,8) and args.steps>0 and args.trust_rms>0 and args.proximal_lambda>=0
    main(args)
