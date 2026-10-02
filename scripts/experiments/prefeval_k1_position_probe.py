"""Training-side T1 diagnostic: read one frozen image with all four correct positions."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.experiments.prefeval_k1_data import load_records,official_mcq,option_order,sha
from scripts.eval.prefeval_rgb import load_reader,read_png,append
from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.repro import configure_strict_cuda_determinism

def main(args):
    configure_strict_cuda_determinism(0)
    rows=load_records(args.split)
    if args.limit:
        rows=rows[:args.limit]
    mcq=official_mcq(ROOT/'third_party/prefeval_reference')
    processor,reader=load_reader(args.reader,'cuda:0')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    done={}
    if args.output.exists():
        for r in map(json.loads,args.output.read_text(encoding='utf-8').splitlines()):
            assert r.get('endpoint_kind','teacher')==args.kind
            assert r.get('split','pilot')==args.split
            assert r.get('prefix',0)==args.prefix
            assert r.get('order_mode','legacy-shuffle')==args.order_mode
            done[r['pair_id'],r.get('chain',0),r['position']]=r['png_sha256']
    for row in rows:
        for chain in range(1 if args.kind=='teacher' else args.noise_chains):
            base=args.images/row['base_pair_id'].replace(':','_')
            path=base/'memory.png' if args.kind=='teacher' else base/f'seed-{chain}'/f'prefix-{args.prefix:02d}.png'
            endpoint=json.loads((path.parent/'complete.json').read_text())
            digest=sha(path)
            if args.kind=='teacher':
                assert endpoint['step']==288 and digest==endpoint['png_sha256']
            else:
                assert digest==endpoint['png_hashes'][path.name]
            image=read_png(path)
            positions=['official',0,1,2,3] if args.order_mode=='official-cyclic' else range(4)
            for position in positions:
                key=(row['base_pair_id'],chain,position)
                if key in done:
                    assert done[key]==digest,'Resume image differs'
                    continue
                if args.order_mode=='official-cyclic':
                    shift=0 if position=='official' else position
                    order=list(range(4))[-shift:]+list(range(4))[:-shift] if shift else list(range(4))
                    correct=order.index(0)
                else:
                    order,correct=option_order(row['base_pair_id'],position*3)
                assert correct==(0 if position=='official' else position)
                query=row['forms']['T1']+mcq['get_mcq_question_format']([row['options'][i] for i in order])
                generated=generate_short_answer(model=reader,processor=processor,image=image,query=query,
                    device='cuda:0',max_new_tokens=32)
                pred=mcq['extract_choice'](generated['raw'])
                record={'pair_id':row['base_pair_id'],'family':'T1','position':position,'order':order,
                    'split':args.split,'order_mode':args.order_mode,
                    'endpoint_kind':args.kind,'chain':chain,'prefix':args.prefix,
                    'png_sha256':digest,'generated':generated,'predicted_letter':pred,
                    'correct_letter':'ABCD'[correct],'correct':pred=='ABCD'[correct]}
                append(args.output,record)
                print(json.dumps({k:v for k,v in record.items() if k!='generated'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reader',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=['teacher','student'],default='teacher')
    p.add_argument('--split',choices=['pilot','dev'],default='pilot')
    p.add_argument('--order-mode',choices=['legacy-shuffle','official-cyclic'],default='legacy-shuffle')
    p.add_argument('--noise-chains',type=int,choices=[1,2],default=2)
    p.add_argument('--prefix',type=int,choices=[0,5,10],default=0)
    p.add_argument('--limit',type=int,default=0)
    main(p.parse_args())
