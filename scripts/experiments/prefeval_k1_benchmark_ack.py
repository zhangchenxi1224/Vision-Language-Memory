"""Prepare official explicit benchmark histories using only the tested Reader's replies."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.experiments.prefeval_k1_data import ALIGN, official_eval_disclosures, sha
from scripts.experiments.prefeval_k1_evaluate import text_generate
from scripts.experiments.prefeval_k1_teacher import save_json
from scripts.eval.prefeval_rgb import load_reader
from vision_memory.repro import configure_strict_cuda_determinism


def main(args):
    configure_strict_cuda_determinism(0)
    binding={'reader':str(args.reader),'benchmark_sha256':sha(ALIGN/'data/benchmark-disclosures.jsonl.gz'),
        'system_prompt':'You are a helpful assistant.','max_new_tokens':300,'do_sample':False,
        'input':'official explicit preference only; no future query, options, or SFT acknowledgment'}
    result={'binding':binding,'acknowledgments':{}}
    if args.output.exists():
        result=json.loads(args.output.read_text(encoding='utf-8'))
        assert result['binding']==binding
    processor,reader=load_reader(args.reader,args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for row in official_eval_disclosures():
        pid=row['base_pair_id']
        if pid in result['acknowledgments']:
            continue
        preference=row['input']['disclosure'][0]['content']
        generated=text_generate(reader,processor,[{'role':'system','content':binding['system_prompt']},
            {'role':'user','content':preference}],args.device,300)
        assert generated['raw'].strip()
        result['acknowledgments'][pid]={'preference':preference,'generated':generated}
        save_json(args.output,result)
        print(json.dumps({'acknowledged':pid,'truncated':generated['truncated']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reader',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    main(p.parse_args())
