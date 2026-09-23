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
    rows=load_records()
    if args.limit:
        rows=rows[:args.limit]
    mcq=official_mcq(ROOT/'third_party/prefeval_reference')
    processor,reader=load_reader(args.reader,'cuda:0')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    done=set()
    if args.output.exists():
        done={(r['pair_id'],r['position']) for r in map(json.loads,args.output.read_text().splitlines())}
    for row in rows:
        path=args.images/row['base_pair_id'].replace(':','_')/'memory.png'
        endpoint=json.loads((path.parent/'complete.json').read_text())
        assert endpoint['step']==288 and sha(path)==endpoint['png_sha256']
        image=read_png(path)
        for position in range(4):
            if (row['base_pair_id'],position) in done:
                continue
            order,correct=option_order(row['base_pair_id'],position*3)
            assert correct==position
            query=row['forms']['T1']+mcq['get_mcq_question_format']([row['options'][i] for i in order])
            generated=generate_short_answer(model=reader,processor=processor,image=image,query=query,
                device='cuda:0',max_new_tokens=32)
            pred=mcq['extract_choice'](generated['raw'])
            record={'pair_id':row['base_pair_id'],'family':'T1','position':position,'order':order,
                'png_sha256':sha(path),'generated':generated,'predicted_letter':pred,
                'correct_letter':'ABCD'[correct],'correct':pred=='ABCD'[correct]}
            append(args.output,record)
            print(json.dumps({k:v for k,v in record.items() if k!='generated'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reader',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    main(p.parse_args())
