"""Launch independent workers once on the explicitly designated GPU notebook."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['author','train','pilot','write-rollout','write-students','train-input'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--questions',type=Path)
    p.add_argument('--rollout-shard',type=int,choices=[0,1],default=0)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not socket.gethostname().startswith('dl-clear-retain-h200x4-20260914'):
        raise RuntimeError('Wrong notebook')
    root=Path(__file__).resolve().parents[2]
    models=Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')
    a.output.mkdir(parents=True,exist_ok=True)
    suffix='-1' if a.phase=='write-rollout' and a.rollout_shard==1 else ''
    receipt=a.output/f'dispatch-{a.phase}{suffix}.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if not a.resume: raise RuntimeError('Already dispatched; read existing receipt')
        if old['host']==socket.gethostname() and any(Path(f'/proc/{w["pid"]}').exists() for w in old['workers']):
            raise RuntimeError('A previous worker is still alive')
        receipt.rename(a.output/f'dispatch-{a.phase}-{old["session_id"]}.json')
    value=dict(session_id=uuid.uuid4().hex,phase=a.phase,host=socket.gethostname(),started_at=time.time(),workers=[],
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        gpu_binding=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True))
    assignments=[(g,'A',g,4) for g in range(4)] if a.phase=='author' else [(0,'A',0,2),(1,'A',1,2),(2,'B',0,2),(3,'B',1,2)]
    if a.phase=='pilot': assignments=[('0,1','A',0,1),('2,3','B',0,1)]
    if a.phase=='train-input':
        for arm in ('A','B'):
            if not (a.output/'pipeline'/arm/'complete.json').exists():
                raise RuntimeError('Finish the fixed benchmark before the exact training-input diagnostic')
    if a.phase=='write-rollout':
        assignments=[(0,'A',a.rollout_shard,2),(2,'B',a.rollout_shard,2)]
        for _,arm,_,_ in assignments:
            if not (a.output/'writers'/arm/'write/complete.json').exists():
                raise RuntimeError('Finish the fixed FM endpoint before RGB inference')
            if (a.output/'pipeline'/arm/f'job-rollout-{a.rollout_shard}.json').exists():
                raise RuntimeError('The existing driver has already scheduled this rollout')
    if a.phase=='write-students':
        assignments=[(0,'A',0,2),(2,'B',0,2)]
        sys.path.insert(0,str(root/'src'))
        from vision_memory.prefeval.official_ab import records
        selected=records(root/'reports/prefeval-official-alignment-20260923')
        for _,arm,_,_ in assignments:
            if (a.output/'pipeline'/arm/'job-student-eval-0.json').exists():
                raise RuntimeError('The existing driver has already scheduled student readings')
            for r in selected:
                for seed in range(2):
                    if not (a.output/'rollouts'/arm/'write'/r['id'].replace(':','-')/f'seed-{seed}'/'complete.json').exists():
                        raise RuntimeError('Finish all PNGs, including mismatch donors, before student readings')
    for gpu,arm,shard,shards in assignments:
        log=a.output/f'{a.phase}-{arm}-{shard}-{value["session_id"]}.log'
        cmd=[sys.executable,'-u',str(root/'scripts/experiments/prefeval_official_ab.py'),a.phase,
            '--reader',str(models/'Qwen3-VL-4B-Instruct'),'--base',str(models/'DreamLite-base-a9a0f15-20260907'),
            '--output',str(a.output),'--arm',arm,'--shard',str(shard),'--shards',str(shards)]
        if a.questions: cmd+=['--questions',str(a.questions)]
        if a.phase=='pilot':
            cmd=[sys.executable,'-u',str(root/'scripts/inspire/run_prefeval_official_pilot.py'),
                 '--output',str(a.output),'--arm',arm,'--gpus',str(gpu)]
            if a.resume:cmd+=['--resume']
        if a.phase=='train-input':
            cmd=[sys.executable,'-u',str(root/'scripts/inspire/run_prefeval_train_input_diagnostic.py'),
                 '--output',str(a.output),'--arm',arm,'--shard',str(shard),'--shards',str(shards)]
        if a.phase in ('write-rollout','write-students'):
            cmd=[sys.executable,'-u',str(root/'scripts/eval/prefeval_official_rgb.py'),
                 'rollout' if a.phase=='write-rollout' else 'students',
                 '--output',str(a.output),'--arm',arm,'--stage','write','--shard',str(shard),'--shards',str(shards),
                 '--questions',str(root/'reports/prefeval-official-alignment-20260923/pilot-questions-3plus2.json')]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        with log.open('ab',buffering=0) as stream:
            worker=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value['workers'].append(dict(pid=worker.pid,gpu=gpu,arm=arm,shard=shard,log=str(log),command=cmd))
        receipt.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)

if __name__=='__main__': main()
