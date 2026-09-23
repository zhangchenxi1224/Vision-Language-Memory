"""Official-answer / MCQ latent pilot; native model code is kept unchanged."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from vision_memory.prefeval.official_ab import records, sha, training_item, validate_forms

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)

def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')

def atomic_tensor(path, value):
    import torch
    temp = path.with_suffix('.tmp.pt')
    torch.save(value, temp)
    temp.replace(path)

def identity(a):
    if not socket.gethostname().startswith('dl-clear-retain-h200x4-20260914'):
        raise RuntimeError('GPU work is restricted to the designated notebook')
    return dict(host=socket.gethostname(), pid=os.getpid(), shard=a.shard, shards=a.shards,
        phase=a.phase, arm=a.arm, commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        reader=str(a.reader), base=str(a.base), time=time.time(),
        gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True))

def author(a):
    """Only original question is shown; no preference, answer, options or outcomes."""
    import torch
    from scripts.eval.prefeval_rgb import load_reader
    processor, model = load_reader(a.reader, a.device)
    tokenizer = processor.tokenizer
    out = a.output / 'questions'
    out.mkdir(parents=True, exist_ok=True)
    write(out / f'author-{a.shard}.json', identity(a))
    selected = records(a.report)[a.shard::a.shards]
    for r in selected:
        path = out / (r['id'].replace(':','-') + '.json')
        if path.exists():
            continue
        prompt = ('Rewrite the question below in four different English forms. Preserve the exact task, '
            'entities, locations, quantities, negations, scope and answer applicability. Do not answer it, '
            'invent user preferences, add constraints, explanations or new facts. Each version must be a '
            'substantive rewriting of the question, not a wrapper around a verbatim copy. '
            'Return only one JSON object with exactly four string values: '
            'T2 = direct interrogative paraphrase; T3 = imperative request; '
            'O1 = a short situation statement using only facts already in the question followed by a question; '
            'O2 = an indirect or hypothetical request, with no extra condition. '
            'Keep natural phrasing and all named entities unchanged.\nOriginal question: ' + r['question'])
        text = tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
        batch = tokenizer(text, return_tensors='pt').to(a.device)
        with torch.no_grad():
            gen = model.generate(**batch, do_sample=False, max_new_tokens=600)
        raw = tokenizer.decode(gen[0,batch.input_ids.shape[1]:],skip_special_tokens=True)
        append(out/f'raw-{a.shard}.jsonl', dict(id=r['id'], question=r['question'], prompt=prompt, raw=raw))
        value = json.loads(raw[raw.index('{'):raw.rindex('}')+1])
        forms = dict(T1=r['question'], **value)
        issue = None
        try: validate_forms(r['question'],forms)
        except ValueError as error: issue=str(error)
        # Preserve imperfect drafts for semantic correction without regenerating
        # already authored rows or selecting examples using task outcomes.
        write(path, dict(id=r['id'],split=r['split'],forms=forms,review_issue=issue,
            source='frozen Qwen3-VL-4B text-only author; pending semantic review'))
        print(json.dumps(dict(authored=r['id'],shard=a.shard)),flush=True)
    write(out/f'complete-{a.shard}.json',dict(count=len(selected)))

def train(a):
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderTiny
    from scripts.eval.prefeval_rgb import load_reader
    from scripts.train.latent_r11_vae_oracle import VAELatentOracle, encode_model_latent
    from vision_memory.reader.open_eos import assistant_termination_contract
    from vision_memory.reader.qwen3vl import qwen3vl_target_only_ce, R3_QWEN_READER_RESIZE_CONTRACT
    from vision_memory.reader.open_answer import generate_short_answer

    question_bank = json.loads(a.questions.read_text(encoding='utf-8'))
    if question_bank['status'] != 'semantic_reviewed_before_training':
        raise ValueError('Review paraphrase semantics before optimizing any target')
    selected = [r for r in records(a.report) if r['split']=='train'][a.shard::a.shards]
    device = torch.device(a.device)
    processor, reader = load_reader(a.reader, device)
    termination = assistant_termination_contract(reader, processor)
    vae = AutoencoderTiny.from_pretrained(a.base/'vae',local_files_only=True,torch_dtype=torch.float32).to(device)
    vae.eval().requires_grad_(False)
    with torch.no_grad():
        initial = encode_model_latent(vae, torch.full((1,3,1024,1024),128/255,device=device))
    root = a.output / 'teachers' / a.arm
    root.mkdir(parents=True, exist_ok=True)
    write(root/f'identity-{a.shard}.json',identity(a))
    for r in selected:
        forms = validate_forms(r['question'],question_bank['records'][r['id']]['forms'])
        out = root / r['id'].replace(':','-')
        out.mkdir(exist_ok=True)
        binding = dict(id=r['id'],arm=a.arm,questions_sha=sha(a.questions),steps=a.steps,
                       objective='answer_and_single_end_token_mean_ce',lr=.05,quantization='uint8_forward_STE_backward')
        if (out/'complete.json').exists():
            done=json.loads((out/'complete.json').read_text())
            if done['binding']!=binding or sha(out/'latent.pt')!=done['latent_sha256']:
                raise ValueError('Existing endpoint differs')
            continue
        oracle=VAELatentOracle(vae=vae,initial_latent=initial,compute_dtype=torch.float32)
        opt=torch.optim.Adam([oracle.latent_fp32],lr=.05,betas=(.9,.999),eps=1e-8)
        start=0
        if (out/'resume.pt').exists():
            ck=torch.load(out/'resume.pt',map_location=device,weights_only=False)
            if ck['binding']!=binding: raise ValueError('Resume inputs changed')
            with torch.no_grad(): oracle.latent_fp32.copy_(ck['latent'])
            opt.load_state_dict(ck['optimizer']);start=ck['step']
        attempt=str(time.time_ns())
        write(out/f'attempt-{attempt}.json',dict(resume_step=start,binding=binding))
        for step in range(start,a.steps):
            began=time.monotonic();opt.zero_grad(set_to_none=True)
            item=training_item(r,forms,a.arm,step)
            pixels=oracle.image()
            rounded=(pixels*255).round().clamp(0,255)/255
            rgb=pixels+(rounded-pixels).detach()
            ce=qwen3vl_target_only_ce(model=reader,processor=processor,image=rgb[0],query=item['query'],
                target=item['target']+termination['assistant_end_token_text'],device=device,
                require_image_grad=True,reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
            if not torch.isfinite(ce.loss): raise RuntimeError('Nonfinite CE')
            ce.loss.backward()
            grad=oracle.latent_fp32.grad
            if grad is None or not torch.isfinite(grad).all() or not torch.any(grad!=0):
                raise RuntimeError('Latent has no finite nonzero gradient')
            opt.step()
            row=dict(step=step+1,attempt=attempt,family=item['family'],loss=float(ce.loss.detach()),
                     target_tokens=ce.target_ids.numel(),order=item['order'],seconds=time.monotonic()-began)
            append(out/'optimization.jsonl',row)
            if (step+1)%24==0 or step+1==a.steps:
                atomic_tensor(out/'resume.pt',dict(binding=binding,step=step+1,
                    latent=oracle.latent_fp32.detach(),optimizer=opt.state_dict()))
                print(json.dumps(dict(id=r['id'],arm=a.arm,**row)),flush=True)
        atomic_tensor(out/'latent.pt',oracle.latent_fp32.detach().cpu())
        with torch.no_grad():
            rgb=(oracle.image()[0]*255).round().clamp(0,255).byte().permute(1,2,0).cpu().numpy()
        Image.fromarray(rgb).save(out/'memory.png')
        # Sanity output only: endpoint remains fixed and no examples are filtered.
        png=torch.from_numpy(np.array(Image.open(out/'memory.png'),copy=True)).permute(2,0,1).float()/255
        g=generate_short_answer(model=reader,processor=processor,image=png,query=r['question'],
            device=device,max_new_tokens=300)
        write(out/'original-generation.json',dict(id=r['id'],generation=g,score='not_judged'))
        write(out/'complete.json',dict(binding=binding,latent_sha256=sha(out/'latent.pt'),png_sha256=sha(out/'memory.png')))
    write(root/f'complete-{a.shard}.json',dict(states=len(selected),steps_per_state=a.steps))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['author','train'])
    p.add_argument('--report',type=Path,default=ROOT/'reports/prefeval-official-alignment-20260923')
    p.add_argument('--reader',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--questions',type=Path)
    p.add_argument('--device',default='cuda:0');p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--arm',choices=['A','B'],default='A')
    p.add_argument('--steps',type=int,default=288)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    try:
        {'author':author,'train':train}[a.phase](a)
    except BaseException:
        import traceback
        write(a.output/f'failure-{a.phase}-{a.arm}-{a.shard}-{int(time.time())}.json',
              dict(error=traceback.format_exc(),host=socket.gethostname()))
        raise

if __name__=='__main__':main()
