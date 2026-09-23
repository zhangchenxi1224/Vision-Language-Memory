"""Fixed-budget pilot continuation, with explicit durable process receipts."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.experiments.prefeval_official_ab import write

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['A','B'],required=True);p.add_argument('--gpus',required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not socket.gethostname().startswith('dl-clear-retain-h200x4-20260914'):raise RuntimeError('Wrong notebook')
    gpus=a.gpus.split(',')
    if len(gpus)!=2:raise ValueError('Exactly two allocated GPUs per arm')
    work=a.output/'pipeline'/a.arm;work.mkdir(parents=True,exist_ok=True)
    if (work/'complete.json').exists():return
    if (work/'running.json').exists():
        old=json.loads((work/'running.json').read_text())
        if old['host']==socket.gethostname() and alive(old['pid']):raise RuntimeError('Existing driver alive')
        if not a.resume:raise RuntimeError('Previous run exists; inspect then resume explicitly')
        if old['host']==socket.gethostname():
            for path in work.glob('job-*.json'):
                child=json.loads(path.read_text())
                if alive(child['pid']):raise RuntimeError('Previous child still alive; do not duplicate it')
        (work/'running.json').rename(work/f'previous-{old["session_id"]}.json')
    session=uuid.uuid4().hex
    write(work/'running.json',dict(session_id=session,pid=os.getpid(),host=socket.gethostname(),gpus=gpus,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    jobs=[]
    def spawn(label,gpu,script,args):
        cmd=[sys.executable,'-u',str(ROOT/script),*args]
        path=work/f'{label}-{session}.log'
        with path.open('ab',buffering=0) as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',
                TOKENIZERS_PARALLELISM='false'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        jobs.append(child)
        write(work/f'job-{label}.json',dict(pid=child.pid,gpu=gpu,command=cmd,log=str(path),session_id=session))
        return child
    def wait(child):
        code=child.wait()
        if code:raise RuntimeError(f'Worker PID {child.pid} exited {code}; retain logs and checkpoints')
    common=['--output',str(a.output),'--arm',a.arm]
    q=ROOT/'reports/prefeval-official-alignment-20260923/pilot-questions-3plus2.json'
    evaluation=['--questions',str(q),*common]
    try:
        # No extra optimizer steps or success-based filtering: wait for the two
        # previously launched workers' full 64-state endpoint bank.
        dispatch=json.loads((a.output/'dispatch-train.json').read_text())
        workers=[x for x in dispatch['workers'] if x['arm']==a.arm]
        while not all((a.output/'teachers'/a.arm/f'complete-{i}.json').exists() for i in (0,1)):
            if dispatch['host']!=socket.gethostname():raise RuntimeError('Teacher dispatch belongs to an old host')
            if any(not (a.output/'teachers'/a.arm/f'complete-{w["shard"]}.json').exists() and not alive(w['pid']) for w in workers):
                raise RuntimeError('A teacher worker stopped before completing its shard')
            time.sleep(15)
        fm=spawn('fm-write',gpus[0],'scripts/train/train_prefeval_official_fm.py',common+['--stage','write'])
        teacher=spawn('teacher-eval',gpus[1],'scripts/eval/prefeval_official_rgb.py',['teachers',*evaluation])
        wait(teacher)
        references=spawn('references',gpus[1],'scripts/eval/prefeval_official_rgb.py',
            ['references',*evaluation,'--shard','0' if a.arm=='A' else '1','--shards','2'])
        wait(references);wait(fm)
        rollouts=[spawn(f'rollout-{i}',gpu,'scripts/eval/prefeval_official_rgb.py',
            ['rollout',*evaluation,'--stage','write','--shard',str(i),'--shards','2']) for i,gpu in enumerate(gpus)]
        for child in rollouts:wait(child)
        reads=[spawn(f'student-eval-{i}',gpu,'scripts/eval/prefeval_official_rgb.py',
            ['students',*evaluation,'--stage','write','--shard',str(i),'--shards','2']) for i,gpu in enumerate(gpus)]
        for child in reads:wait(child)
        write(work/'complete.json',dict(session_id=session,status='fixed_write_pilot_complete',
            next='Summarize teachers/students/controls; official natural-answer judge pending; then decide retain stage from evidence'))
    except BaseException:
        import traceback
        # Do not kill a healthy independent job when a sibling fails. Record it
        # so a resume cannot silently create a second GPU worker.
        write(work/f'failure-{session}.json',dict(error=traceback.format_exc(),
            children=[dict(pid=c.pid,returncode=c.poll()) for c in jobs]))
        raise

if __name__=='__main__':main()
