"""Run two independent direct-z EOS lanes on the user's allocated notebook."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.experiments.run_direct_latent_geometry import load, summarize
from scripts.experiments.run_r11_open_answer_replay import write_json
P=Path('/inspire/ssd/project/exploration-topic/czxs26210936')
M=Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args()
    if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=args.expected_commit
            or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()):
        raise RuntimeError('Exact clean source commit required')
    args.output_root.mkdir(parents=True,exist_ok=True)
    with (args.output_root/'.campaign.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv,noheader,nounits'],text=True)
        records=[line.split(',') for line in gpus.splitlines() if line.strip()]
        if len(records)!=4 or any('H200' not in r[1] or int(r[2])<140000 for r in records):
            raise RuntimeError('Need four full-memory H200 GPUs')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPUs have existing compute processes')
        env=os.environ.copy()
        env.update({'PYTHONHASHSEED':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1',
            'MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256':'1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159',
            'VLM_READER_SNAPSHOT_MANIFEST_SHA256':'159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c'})
        launched={'instance':'dl-base-h200x4-20260907','commit':args.expected_commit,'started_epoch':time.time(),
                  'deadline_epoch':args.deadline,'planned_runs':96,'pid':os.getpid(),'gpus':gpus,'lanes':[]}
        children=[]; logs=[]; interrupted=[]
        signal.signal(signal.SIGTERM,lambda s,f:interrupted.append(s))
        signal.signal(signal.SIGINT,lambda s,f:interrupted.append(s))
        try:
            for lane in range(2):
                command=[sys.executable,'-u',str(ROOT/'scripts/experiments/run_direct_latent_geometry.py'),
                    '--lane',str(lane),'--output-dir',str(args.output_root/f'lane-{lane}'),
                    '--expected-commit',args.expected_commit,'--deadline',str(args.deadline),
                    '--dreamlite',str(M/'DreamLite-mobile'),'--reader',str(M/'Qwen3-VL-4B-Instruct'),
                    '--train',str(P/'data/vision-language-memory-r3/formal-v1/train.jsonl'),
                    '--dev',str(P/'data/vision-language-memory-r3/formal-v1/dev.jsonl'),
                    '--old-multistart-root',str(P/'runs/vision-language-memory-r11-open/multistart-5d06b76-20260907-round02')]
                lane_env={**env,'CUDA_VISIBLE_DEVICES':f'{lane*2},{lane*2+1}'}
                log=(args.output_root/f'lane-{lane}.log').open('a'); logs.append(log)
                child=subprocess.Popen(command,cwd=ROOT,env=lane_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                children.append(child)
                launched['lanes'].append({'lane':lane,'gpu_pair':lane_env['CUDA_VISIBLE_DEVICES'],'pid':child.pid,'command':command})
            write_json(args.output_root/'launch.json',launched)
            while True:
                codes=[p.poll() for p in children]
                write_json(args.output_root/'status.json',{'state':'running','lane_returncodes':codes,'checked_epoch':time.time()})
                if interrupted or any(c not in (None,0) for c in codes):
                    raise RuntimeError(f'Worker failure/interruption: {codes}, {interrupted}')
                if all(c==0 for c in codes): break
                if time.time()>args.deadline+60:
                    raise RuntimeError('Deadline exceeded; preserving incomplete artifacts for audit')
                time.sleep(10)
            summary=summarize(args.output_root,load(ROOT/'configs/experiments/direct_latent_geometry.json'))
            write_json(args.output_root/'terminal.json',{'status':'completed' if summary['complete'] else 'paused_at_run_boundary',
                'completed':summary['completed_count'],'planned':96,'success':summary['success_count'],'finished_epoch':time.time()})
        except BaseException as error:
            for child in children:
                if child.poll() is None: child.terminate()
            for child in children:
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            write_json(args.output_root/'terminal.json',{'status':'failed','error':str(error),'finished_epoch':time.time()})
            raise
        finally:
            for log in logs: log.close()


if __name__=='__main__': main()
