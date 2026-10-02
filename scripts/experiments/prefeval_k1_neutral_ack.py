"""Generate short neutral Reader acknowledgments for train/dev input augmentation."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.experiments.prefeval_k1_data import load_records,sha
from scripts.experiments.prefeval_k1_evaluate import text_generate
from scripts.experiments.prefeval_k1_teacher import save_json
from scripts.eval.prefeval_rgb import load_reader
from vision_memory.repro import configure_strict_cuda_determinism

SYSTEM=('You are a helpful assistant. The user is sharing a preference. '
        'Respond with one brief neutral acknowledgment only. Do not restate or interpret the preference. '
        'Do not add advice, recommendations, questions, new conditions, or other information.')


def main(args):
    configure_strict_cuda_determinism(0)
    args.output.mkdir(parents=True,exist_ok=True)
    rows=load_records('train')+load_records('dev')
    selected=rows[args.shard::args.shards]
    binding={'reader':str(args.reader),'system_prompt':SYSTEM,'max_new_tokens':48,'greedy':True,
             'input':'current preference only; training augmentation, not official benchmark acknowledgment',
             'ids':[r['base_pair_id'] for r in selected]}
    identity=args.output/f'identity-{args.shard}.json'
    if identity.exists():
        assert json.loads(identity.read_text())==binding
    else:
        save_json(identity,binding)
    processor,reader=load_reader(args.reader,args.device)
    for index,row in enumerate(selected):
        pid=row['base_pair_id']
        path=args.output/(pid.replace(':','_')+'.json')
        if path.exists():
            assert json.loads(path.read_text())['preference']==row['history'][0]['content']
            continue
        preference=row['history'][0]['content']
        generated=text_generate(reader,processor,[{'role':'system','content':SYSTEM},
            {'role':'user','content':preference}],args.device,48)
        save_json(path,{'pair_id':pid,'preference':preference,'split':row['split'],
                       'generated':generated,'review_status':'pending'})
        print(json.dumps({'acknowledged':pid,'item':index+1,'total':len(selected),'raw':generated['raw'],
                          'truncated':generated['truncated']}),flush=True)
    save_json(args.output/f'finished-{args.shard}.json',{'count':len(selected),'identity_sha256':sha(identity)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reader',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=2)
    p.add_argument('--device',default='cuda:0')
    main(p.parse_args())
