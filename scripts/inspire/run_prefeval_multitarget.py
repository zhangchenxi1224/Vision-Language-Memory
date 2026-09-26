"""Isolated F1/F8/S8 pilot; bounded stages and real-PNG evaluation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.experiments.prefeval_k1_data import load_training_records, sha
from scripts.experiments.prefeval_multitarget_bank import freeze_bank

PROJECT=Path('/inspire/ssd/project/exploration-topic/czxs26210936')
PYTHON=PROJECT/'envs/vlm-r3-ngc2502/bin/python'
MODELS=Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')
BASE=MODELS/'DreamLite-base-a9a0f15-20260907'
READER=MODELS/'Qwen3-VL-4B-Instruct'
OFFICIAL=PROJECT/'Vision-Language-Memory/third_party/DreamLite'
PREFEVAL=PROJECT/'repos/prefeval-k1-l0-l2-20260924/third_party/prefeval_reference'
PARENT=PROJECT/'runs/prefeval-b-mcq-20260925/robust730/train/checkpoint-final.pt'
VARIANTS=PROJECT/'runs/prefeval-b-mcq-20260925/variants-train.json'
WRITER=ROOT/'scripts/experiments/prefeval_k1_writer.py'
EVAL=ROOT/'scripts/experiments/prefeval_k1_evaluate.py'
TEACHER=ROOT/'scripts/experiments/prefeval_multitarget_teacher.py'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temp.replace(path)


def immutable(path,value):
    if path.exists():
        assert json.loads(path.read_text())==value, f'Changed experiment binding: {path}'
    else: save(path,value)


def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    state={'host':os.uname().nodename,'pid':os.getpid(),'commit':commit,'code':str(ROOT)}
    def status(stage,**extra):
        value=dict(state,stage=stage,time_utc=datetime.now(timezone.utc).isoformat(),**extra)
        save(a.output/'controller.json',value)
        print(json.dumps(value),flush=True)
    def jobs(label,commands):
        status(label)
        children=[]
        for gpu,command in commands:
            busy=subprocess.check_output(['nvidia-smi','-i',str(gpu),
                '--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if busy: raise RuntimeError(f'GPU {gpu} occupied; refusing overlap: {busy}')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
            stream=(a.output/f'{label}-gpu{gpu}.log').open('a')
            command=list(map(str,command))
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
            children.append((p,stream))
            save(a.output/f'{label}-gpu{gpu}-process.json',dict(state,gpu=gpu,pid=p.pid,command=command))
        results=[]
        for child,stream in children:
            results.append(child.wait())
            stream.close()
        if any(results):
            status('failed',failed_stage=label,exit_codes=results)
            raise RuntimeError(f'{label}: {results}')
    rows=load_training_records('pilot')
    if a.smoke: rows=rows[:2]
    ids=[r['base_pair_id'] for r in rows]
    ids_file=a.output/'ids.json'
    immutable(ids_file,{'ids':ids,'scope':'train730 fixed pilot64; not unseen-preference generalization'})
    parent_receipt=json.loads((PARENT.parent/'complete.json').read_text())
    assert parent_receipt['steps']==23360 and parent_receipt['checkpoint_sha256']==sha(PARENT)
    immutable(a.output/'protocol.json',{'commit':commit,'parent':str(PARENT),'parent_sha256':sha(PARENT),
        'ids':ids,'targets':a.targets,'teacher_steps':a.teacher_steps,'fm_steps':a.fm_steps,
        'cfg':1,'generation_steps':28,'target_entry':'direct optimized model-space latent, matching prior B',
        'teacher_forms':['T1','T2','T3'],'selection_forms':['T1','T2','T3'],
        'excluded_from_selection':['dev90','O1','O2','official180','V2'],
        'arms':['F1','F8','S8'],'proximal_lambda':.1,'trust_rms':1.,
        'qualification':'every T1/T2/T3 x four correct-option positions must parse and be correct',
        'single_sample_primary':True,'no_test_time_target_selection':True,'smoke':a.smoke})
    common=['--arm','B','--base',BASE,'--official-source',OFFICIAL,'--split','train',
        '--ids-file',ids_file,'--initial-variants',VARIANTS]
    starts=a.output/'starts'
    # Same physical allocation, no changes to B730/C/R hosts or their directories.
    for variant in range(2):
        commands=[]
        for gpu in range(4):
            commands.append((gpu,[PYTHON,WRITER,'rollout',*common,'--checkpoint',PARENT,
                '--initial-variant',variant,'--inter-turns',0,'--noise-chains',max(1,(a.targets+1)//2),
                '--noise-domain','mt8-teacher','--shard-index',gpu,'--shard-count',4,
                '--output',starts/f'V{variant}']))
        jobs(f'starts-V{variant}',commands)
    jobs('teachers',[(gpu,[PYTHON,TEACHER,'--base',BASE,'--reader',READER,'--prefeval',PREFEVAL,
        '--starts',starts,'--ids-file',ids_file,'--output',a.output/'teachers',
        '--steps',a.teacher_steps,'--targets',a.targets,'--shard',gpu,'--shards',4]) for gpu in range(4)])
    banks={arm:freeze_bank(a.output,ids,arm) for arm in ['F1','F8','S8']}
    for arm,bank in banks.items(): save(a.output/f'{arm}-bank.json',bank)
    missing={arm:bank['missing_preferences'] for arm,bank in banks.items() if not bank['ready']}
    if missing:
        status('teacher_repair_required',missing_preferences=missing)
        return
    if a.smoke:
        status('smoke_teacher_complete',qualified_counts={arm:[v['qualified'] for v in b['coverage'].values()] for arm,b in banks.items()})
        return
    jobs('fm',[(gpu,[PYTHON,WRITER,'train',*common,'--checkpoint',PARENT,
        '--target-bank',a.output/f'{arm}-bank.json','--stage','write','--steps',a.fm_steps,
        '--output',a.output/arm/'train']) for gpu,arm in enumerate(['F1','F8','S8'])])
    for variant in range(2):
        jobs(f'eval-generate-V{variant}',[(gpu,[PYTHON,WRITER,'rollout',*common,
            '--checkpoint',PARENT if arm=='B0' else a.output/arm/'train/checkpoint-final.pt','--initial-variant',variant,
            '--inter-turns',0,'--noise-chains',8,'--noise-domain','mt8-eval',
            '--output',a.output/arm/f'eval-V{variant}']) for gpu,arm in enumerate(['F1','F8','S8','B0'])])
        jobs(f'eval-read-V{variant}',[(gpu,[PYTHON,EVAL,'--kind','student','--split','train',
            '--ids-file',ids_file,'--reader',READER,'--initial-variants',VARIANTS,'--initial-variant',variant,
            '--images',a.output/arm/f'eval-V{variant}','--output',a.output/arm/f'read-V{variant}',
            '--families','T1,T2,T3','--tasks','mcq','--prefixes','0','--noise-chains',8,
            '--controls','memory,mismatch,blank,text']) for gpu,arm in enumerate(['F1','F8','S8','B0'])])
    results={}
    for arm in ['F1','F8','S8','B0']:
        counts=defaultdict(lambda:[0,0])
        per_image=defaultdict(list)
        per_preference=defaultdict(list)
        for variant in range(2):
            records=[json.loads(x) for x in (a.output/arm/f'read-V{variant}/readback-0.jsonl').read_text().splitlines()]
            for r in records:
                key=f"V{variant}/{r['control']}/{r['family']}"
                counts[key][0]+=int(r['correct']); counts[key][1]+=1
                if r['control']=='memory':
                    per_image[r['pair_id'],variant,r['chain']].append(r['correct'])
                    per_preference[r['pair_id']].append(int(r['correct']))
        assert len(per_image)==len(ids)*2*8 and all(len(x)==3 for x in per_image.values())
        results[arm]={'correct_total':dict(counts),'all_three_forms_correct':sum(all(x) for x in per_image.values()),
            'images':len(per_image),'per_preference':{k:sum(v)/len(v) for k,v in per_preference.items()},
            'qualified_target_counts':{k:v['qualified'] for k,v in banks[arm]['coverage'].items()} if arm!='B0' else {}}
    save(a.output/'results.json',results)
    status('pilot_complete_iteration_review_required',results=str(a.output/'results.json'))
    save(a.output/'complete.json',{'status':'pilot_complete','results_sha256':sha(a.output/'results.json'),
        'next':'Interpret training-only paired results; then student refresh, topic structure, or scale730. No held-out selection.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--targets',type=int,default=8)
    p.add_argument('--teacher-steps',type=int,default=288)
    p.add_argument('--fm-steps',type=int,default=2048)
    a=p.parse_args()
    main(a)
