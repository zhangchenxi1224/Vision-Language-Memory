"""L1 extension: replace the only active preference in a real final RGB endpoint."""
import argparse
import json
from pathlib import Path
import sys
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.experiments.prefeval_k1_data import load_records,event_text,sha
from scripts.experiments.prefeval_k1_teacher import save_json
from scripts.experiments.prefeval_k1_writer import load_pipe,encode_source
from scripts.train.latent_r11_vae_oracle import _save_image
from vision_memory.dreamlite.native_base import NativeBaseEditSampler
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
from vision_memory.training.latent_bank_unet import stable_seed
from vision_memory.repro import configure_strict_cuda_determinism

@torch.no_grad()
def main(args):
    configure_strict_cuda_determinism(0)
    rows=load_records(args.split)
    pipe=load_pipe(args)
    pipe.unet.requires_grad_(False)
    binding={'checkpoint_sha256':sha(args.checkpoint),'split':args.split,
        'protocol':'single_active_preference_replacement_extension_not_official_benchmark',
        'source_root':str(args.sources),'source_prefix':10,'noise_chains':args.noise_chains,
        'steps':28,'cfg':1,'only_persistent_input':'reopened RGB PNG'}
    args.output.mkdir(parents=True,exist_ok=True)
    save_json(args.output/'manifest.json',binding)
    for row in rows:
        pid=row['base_pair_id']
        peers=[r for r in rows if r['topic']==row['topic'] and r['history'][0]['content']!=row['history'][0]['content']]
        assert peers, 'Need a genuinely different prior preference'
        # Deterministic same-topic pairing, independent of any generated answer.
        prior=min(peers,key=lambda r:stable_seed(20260924,pid+':prior:'+r['base_pair_id'],0))
        exchange=[{'role':'user','content':'My earlier preference no longer applies. My current preference is: '+row['history'][0]['content']},
                  dict(row['history'][1])]
        for chain in range(args.noise_chains):
            out=args.output/pid.replace(':','_')/f'seed-{chain}'
            out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():
                done=json.loads((out/'complete.json').read_text())
                assert done['binding']==binding and sha(out/'prefix-00.png')==done['png_hashes']['prefix-00.png']
                continue
            previous=args.sources/prior['base_pair_id'].replace(':','_')/f'seed-{chain}'/'prefix-10.png'
            source_done=json.loads((previous.parent/'complete.json').read_text())
            assert sha(previous)==source_done['png_hashes'][previous.name]
            image=Image.open(previous).convert('RGB')
            source=encode_source(pipe,image,args.device)
            seed=stable_seed(20260924,f'replace:{pid}:{chain}',0)
            noise=torch.randn(source.shape,device=args.device,dtype=source.dtype,
                              generator=torch.Generator(device=args.device).manual_seed(seed))
            sampler=NativeBaseEditSampler(pipe,source_image=image,event_text=event_text(exchange),guidance_scale=1.)
            result=sampler(source_latents=source,noise_latents=noise,num_steps=28,return_trajectory=False)
            _save_image(out/'prefix-00.png',decode_model_latents_unit_interval(pipe.vae,result.latents,clamp=True))
            save_json(out/'complete.json',{'binding':binding,'png_hashes':{'prefix-00.png':sha(out/'prefix-00.png')},
                'prior_pair_id':prior['base_pair_id'],'new_pair_id':pid,'source_png_sha256':sha(previous),
                'exchange':exchange,'noise_seed':seed})
        print(json.dumps({'replaced':pid,'prior':prior['base_pair_id']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base','official-source','checkpoint','sources','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=['pilot','dev'],default='dev')
    p.add_argument('--noise-chains',type=int,default=2)
    p.add_argument('--device',default='cuda:0')
    main(p.parse_args())
