"""At most eight registered first-divergence logit inspections; no answer retries."""
import argparse,json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from scripts.probes.prefeval_rgb_endpoint_diagnostic import (load_json,load_reader,read_png,matched_images,
    file_sha,tensor_sha,configure_strict_cuda_determinism,decode_model_latents_unit_interval,REPORT)
from vision_memory.reader.qwen3vl import _prepare_reader_image,R3_QWEN_READER_RESIZE_CONTRACT

def token_logits(logits,gold,observed):
    x=logits.float();top=x.topk(5)
    alt=x.clone();alt[gold]=-float('inf')
    return dict(argmax=int(x.argmax()),gold_id=gold,observed_id=observed,gold_logit=float(x[gold]),
        observed_logit=float(x[observed]),gold_minus_best_alternative=float(x[gold]-alt.max()),
        top_ids=top.indices.tolist(),top_logits=top.values.tolist())

def main(a):
    configure_strict_cuda_determinism(0)
    summary=load_json(a.diagnostic/'verified.json');selected=summary['tf_argmax_generation_mismatches'][:8]
    if a.output.exists():raise ValueError('First-divergence output already exists')
    from diffusers import AutoencoderTiny
    vae=AutoencoderTiny.from_pretrained(str(a.base),subfolder='vae',local_files_only=True,torch_dtype=torch.float32).to(a.device)
    vae.eval().requires_grad_(False);processor,reader=load_reader(a.reader,a.device)
    versions=[(p,int(p._version)) for module in (vae,reader) for p in module.parameters()]
    registration=load_json(REPORT/'endpoint-diagnostic-v1.json');results=[]
    with torch.no_grad():
        for cell in selected:
            sid=cell['target'];condition=cell['condition'];qid=cell['query_id'];src=a.source/sid
            rows=[json.loads(s) for s in (a.diagnostic/sid/'recovery.jsonl').read_text().splitlines()]
            r=next(r for r in rows if r['condition']==condition and r['query']['query_id']==qid)
            t=registration['targets'][sid];assert file_sha(src/'latent.pt')==t['artifacts']['latent.pt']
            z=torch.load(src/'latent.pt',map_location=a.device,weights_only=True)
            decoded=decode_model_latents_unit_interval(vae,z,clamp=True)[0]
            image=matched_images(decoded,read_png(src/'memory.png'),a.device)[condition]
            resized,_=_prepare_reader_image(image,do_resize=None,reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
            batch=processor(text=[r['generation']['chat_prompt']],images=[resized],return_tensors='pt',do_rescale=False,do_resize=False).to(a.device)
            assert batch['input_ids'][0].tolist()==r['generation']['input_token_ids']
            assert tensor_sha(batch['pixel_values'])==r['prefixes']['pixel_values']['sha']
            gold=r['ce']['gold_token_ids'];observed=r['generation']['generated_token_ids'];i=cell['first_divergent_token']
            assert i is not None and i<len(gold) and i<len(observed) and gold[:i]==observed[:i]
            prefix_len=batch['input_ids'].shape[1]
            def expanded(n):
                b=dict(batch);suffix=torch.tensor([gold[:n]],device=a.device,dtype=torch.long)
                b['input_ids']=torch.cat((batch['input_ids'],suffix),1)
                b['attention_mask']=torch.cat((batch['attention_mask'],torch.ones_like(suffix)),1)
                if 'mm_token_type_ids' in b:b['mm_token_type_ids']=torch.cat((batch['mm_token_type_ids'],torch.zeros_like(suffix)),1)
                return b
            # Prefix-only forward isolates context-length effects from full teacher forcing.
            direct=reader(**expanded(i),use_cache=False,return_dict=True)
            uncached=token_logits(direct.logits[0,-1],gold[i],observed[i]);del direct
            # Replay the already-observed prefix into the model's normal cache preparation.
            # No new answer is sampled/selected and no decoded answer is rescored.
            cache=None;cache_receipt=[]
            for n in range(i+1):
                b=expanded(n);ids=b.pop('input_ids')
                position=torch.arange(prefix_len,device=a.device) if n==0 else torch.tensor([prefix_len+n-1],device=a.device)
                prepared=reader.prepare_inputs_for_generation(ids,past_key_values=cache,cache_position=position,use_cache=True,**b)
                if n and 'mm_token_type_ids' in prepared:prepared['mm_token_type_ids']=torch.zeros_like(prepared['input_ids'])
                output=reader(**prepared,return_dict=True)
                cache=output.past_key_values
                cache_receipt.append(dict(prefix_answer_tokens=n,argmax=int(output.logits[0,-1].argmax())))
                if n==i:cached=token_logits(output.logits[0,-1],gold[i],observed[i])
                del output
            results.append(dict(**cell,gold_id=gold[i],observed_id=observed[i],
                original_teacher_forced_gold_margin=r['ce']['gold_minus_alternative_margin'][i],
                uncached_prefix=uncached,cached_prefix=cached,cache_prefix_trace=cache_receipt,
                cached_reproduces_observed=cached['argmax']==observed[i],endpoint_sha=t['artifacts']['latent.pt']))
            assert tensor_sha(z)==t['latent_tensor_sha']
            for param,version in versions:assert int(param._version)==version and not param.requires_grad and param.grad is None
    report=dict(selection='first eight lexicographic target/condition/query_id cells with all gold tokens TF argmax-correct but differing generation',
        total_mismatches=len(summary['tf_argmax_generation_mismatches']),inspected=len(results),results=results,
        optimizer_updates=0,full_answer_generations=0,
        code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('source','diagnostic','output','base','reader'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--device',default='cuda:0');main(ap.parse_args())
