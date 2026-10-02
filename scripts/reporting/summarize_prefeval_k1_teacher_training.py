"""Describe the complete paired target bank; optimization losses are not accuracy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.prefeval_k1_data import load_records,sha


def main(args):
    records=[]
    for arm in ['A','B']:
        for row in load_records('pilot'):
            pid=row['base_pair_id']
            folder=args.bank/arm/pid.replace(':','_')
            done=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
            trace=folder/'optimization.jsonl'
            steps=[json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
            assert done['step']==288 and done['binding']['arm']==arm
            assert [s['step'] for s in steps]==list(range(1,289)), (arm,pid,'missing or duplicated accepted updates')
            assert Counter(s['family'] for s in steps)=={'T1':96,'T2':96,'T3':96}
            if arm=='B':
                balance=Counter((s['family'],s['option_order'].index(0)) for s in steps)
                assert len(balance)==12 and set(balance.values())=={24}
            assert sha(folder/'memory.png')==done['png_sha256']
            assert sha(folder/'latent.pt')==done['latent_sha256']
            records.append({'arm':arm,'pair_id':pid,'accepted_updates':len(steps),
                'first_12_mean_ce':statistics.mean(s['ce'] for s in steps[:12]),
                'last_12_mean_ce':statistics.mean(s['ce'] for s in steps[-12:]),
                'last_recorded_ce':steps[-1]['ce'],'trace_sha256':sha(trace),
                'png_sha256':done['png_sha256'],'latent_sha256':done['latent_sha256'],
                'execution_commit':done['binding']['commit']})
    summary={}
    for arm in ['A','B']:
        selected=[r for r in records if r['arm']==arm]
        values=[r['last_12_mean_ce'] for r in selected]
        summary[arm]={'states':len(selected),'accepted_updates':sum(r['accepted_updates'] for r in selected),
            'last_12_ce_across_states':{'min':min(values),'median':statistics.median(values),'max':max(values)}}
    result={'status':'complete_target_bank_not_readback_accuracy','summary':summary,'records':records,
        'interpretation':'CE is measured before each optimizer update. The last 12 cover three forms and four correct positions. Arms use different targets and token lengths; CE magnitudes are not comparable memory-accuracy scores.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
