"""Read frozen PNG endpoints with official MCQ and free-generation tasks."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.experiments.prefeval_k1_data import load_records, official_mcq, option_order, sha
from scripts.experiments.prefeval_k1_teacher import save_json
from scripts.eval.prefeval_rgb import load_reader, read_png, append
from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.repro import configure_strict_cuda_determinism

@torch.no_grad()
def text_generate(reader, processor, messages, device, tokens):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[text], return_tensors='pt').to(device)
    result = reader.generate(**batch, max_new_tokens=tokens, do_sample=False, num_beams=1, use_cache=True)
    generated = result[0, batch['input_ids'].shape[1]:].tolist()
    stops = reader.generation_config.eos_token_id
    if isinstance(stops,int):
        stops=[stops]
    return {'raw':processor.tokenizer.decode(generated, skip_special_tokens=True),
        'generated_token_ids':generated,'truncated':not any(t in stops for t in generated)}

def main(args):
    configure_strict_cuda_determinism(0)
    rows=load_records(args.split,history_file=args.history_file)
    if args.split=='official':
        assert args.kind=='student', 'Do not optimize or evaluate teacher targets for held-out official rows'
    by_topic=defaultdict(list)
    for r in rows:
        by_topic[r['topic']].append(r['base_pair_id'])
    selected=rows[args.shard::args.shards]
    if args.limit:
        selected=selected[:args.limit]
    mcq=official_mcq(ROOT/'third_party/prefeval_reference')
    processor,reader=load_reader(args.reader,args.device)
    args.output.mkdir(parents=True,exist_ok=True)
    dest=args.output/f'readback-{args.shard}.jsonl'
    completed=set()
    if dest.exists():
        for line in dest.read_text().splitlines():
            r=json.loads(line)
            completed.add(tuple(r[k] for k in ['pair_id','chain','prefix','control','family','task']))
    counts=defaultdict(lambda:[0,0])
    prefixes=[0] if args.kind=='teacher' else [int(x) for x in args.prefixes.split(',')]
    chains=range(1 if args.kind=='teacher' else args.noise_chains)
    controls=args.controls.split(',')
    families=args.families.split(',')
    history_protocol='official SFT exchanges; not final benchmark acknowledgment'
    if args.kind=='student':
        manifest=json.loads((args.images/'manifest.json').read_text())
        if args.split=='official':
            assert manifest['benchmark_history_sha256']==sha(args.history_file)
            history_protocol=rows[0]['history_protocol']
        if manifest.get('protocol')=='single_active_preference_replacement_extension_not_official_benchmark':
            history_protocol=manifest['protocol']
            if 'text' in controls:
                raise ValueError('Replacement text baseline needs its actual old history plus overwrite exchange')
    for row in selected:
        pid=row['base_pair_id']
        peers=by_topic[row['topic']]
        donor=peers[(peers.index(pid)+1)%len(peers)]
        assert donor!=pid
        for chain in chains:
            for prefix in prefixes:
                def endpoint(source_id):
                    base=args.images/source_id.replace(':','_')
                    if args.kind=='teacher':
                        done=json.loads((base/'complete.json').read_text())
                        assert done['step']==288, 'Technical smoke is not an evaluated teacher'
                        png=base/'memory.png'
                        assert sha(png)==done['png_sha256']
                        return png
                    base=base/f'seed-{chain}'
                    done=json.loads((base/'complete.json').read_text())
                    png=base/f'prefix-{prefix:02d}.png'
                    assert sha(png)==done['png_hashes'][png.name]
                    return png
                for control in controls:
                    # A blank image is independent of history/chain; text is independent of noise.
                    # Reuse these baseline rows at other endpoints rather than regenerating duplicates.
                    if control == 'blank' and (prefix != 0 or chain != 0):
                        continue
                    if control == 'text' and chain != 0:
                        continue
                    path=None
                    if control in ['memory','mismatch']:
                        path=endpoint(pid if control=='memory' else donor)
                        pixels=read_png(path)
                    else:
                        pixels=torch.full((3,1024,1024),128/255.)
                    png_hash=sha(path) if path else None
                    for family in families:
                        question=row['forms'][family]  # Fail if requested OOD has not been authored.
                        for task in ['free','mcq']:
                            key=(pid,chain,prefix,control,family,task)
                            if key in completed:
                                continue
                            step=int.from_bytes(hashlib.sha256(f'eval:{pid}:{family}'.encode()).digest()[:4],'big')
                            order,correct=option_order(pid,step)
                            query=question+' (Please respond within 300 words.)' if task=='free' else question+mcq['get_mcq_question_format']([row['options'][i] for i in order])
                            tokens=300 if task=='free' else 32
                            if control=='text':
                                history=row['history'][:2*(prefix+1)]
                                generated=text_generate(reader,processor,history+[{'role':'user','content':query}],args.device,tokens)
                            else:
                                generated=generate_short_answer(model=reader,processor=processor,image=pixels,
                                    query=query,device=args.device,max_new_tokens=tokens)
                            record=dict(zip(['pair_id','chain','prefix','control','family','task'],key))
                            record.update({'png_sha256':png_hash,'png_path':str(path) if path else None,
                                'donor_pair_id':donor if control=='mismatch' else None,
                                'question':question,'reader_query':query,'generated':generated,
                                'max_new_tokens':tokens,'history_protocol':history_protocol,
                                'preference':row['history'][0]['content'],'split':args.split,'endpoint_kind':args.kind})
                            if task=='mcq':
                                predicted=mcq['extract_choice'](generated['raw'])
                                record.update({'option_order':order,'correct_letter':'ABCD'[correct],
                                    'predicted_letter':predicted,'correct':predicted=='ABCD'[correct],
                                    'parse_failure':predicted is None})
                                counts[control,family][0]+=int(record['correct'])
                                counts[control,family][1]+=1
                            else:
                                record['official_judge_status']='pending'
                            append(dest,record)
                            completed.add(key)
                print(json.dumps({'evaluated':pid,'prefix':prefix,'chain':chain,'shard':args.shard}),flush=True)
    save_json(args.output/f'finished-{args.shard}.json',{'status':'generation_completed_judge_pending','items':len(completed),
        'new_mcq_counts':{'/'.join(k):v for k,v in counts.items()}})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reader',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=['teacher','student'],required=True)
    p.add_argument('--split',choices=['pilot','dev','official'],default='pilot')
    p.add_argument('--history-file',type=Path)
    p.add_argument('--controls',default='memory,blank,mismatch,text')
    p.add_argument('--families',default='T1,T2,T3,O1,O2')
    p.add_argument('--prefixes',default='0,5,10')
    p.add_argument('--noise-chains',type=int,default=2)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    main(p.parse_args())
