"""Idempotent bootstrap on the dedicated idle two-H200 allocation only."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--instance',default='vlm-unet-fit-h200x2-20260909')
    p.add_argument('--warmup-only',action='store_true')
    p.add_argument('--wait-for-warmup',action='store_true')
    p.add_argument('--lease-minutes',type=int,default=165)
    args=p.parse_args()
    assert args.instance in ['vlm-unet-fit-h200x2-20260909','dl-base-h200x4-20260907']
    assert 5<=args.lease_minutes<=165
    assert socket.gethostname().startswith(args.instance+'--'),'Wrong allocation'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==args.commit
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.dispatch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.wait_for_warmup and not (args.output/'warmup-complete.json').exists():
            if (args.output/'failure.json').exists():
                raise RuntimeError('Warmup failed; inspect evidence before continuation')
            terminal=args.output/'terminal.json'
            if not (terminal.exists() and json.loads(terminal.read_text()).get('status')=='paused'):
                print(json.dumps({'status':'waiting_for_warmup_boundary'}))
                return 11
        dispatch=args.output/('warmup-dispatch.json' if args.warmup_only else 'dispatch.json')
        if dispatch.exists():
            previous=json.loads(dispatch.read_text())
            process=Path('/proc')/str(previous['pid'])/'cmdline'
            if process.exists() and str(args.output).encode() in process.read_bytes():
                print(json.dumps({'status':'already_running','pid':previous['pid']}))
                return
            if (args.output/'terminal.json').exists():
                print((args.output/'terminal.json').read_text())
                return
            raise RuntimeError('Previous dispatch ended unexpectedly; inspect evidence before retrying')
        devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
        selected={line.split(',')[1].strip() for line in devices.splitlines() if int(line.split(',')[0]) in [0,1]}
        assert len(selected)==2
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        assert not(selected & set(active.splitlines())),'Selected GPUs occupied; will not interrupt them'
        python=Path('/inspire/ssd/project/exploration-topic/czxs26210936/envs/vlm-r3-ngc2502/bin/python')
        assert python.is_file()
        deadline=time.time()+args.lease_minutes*60
        command=[str(python),'-u',str(ROOT/'scripts/train/run_unet_learnability.py'),
            '--config',str(ROOT/'configs/experiments/unet_learnability.json'),
            '--output',str(args.output),'--expected-commit',args.commit,
            '--expected-hostname',socket.gethostname(),'--deadline',str(deadline)]
        if args.warmup_only:
            command.append('--warmup-only')
        env=os.environ.copy()
        env['CUDA_VISIBLE_DEVICES']='0,1'
        env.update(PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        with (args.output/'campaign.log').open('ab',buffering=0) as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,
                                  stderr=subprocess.STDOUT,start_new_session=True)
        value={'pid':proc.pid,'instance':args.instance,'hostname':socket.gethostname(),'command':command,
               'epoch':time.time(),'deadline_epoch':deadline,'source_commit':args.commit,
               'lease_minutes':args.lease_minutes,'warmup_only':args.warmup_only,
               'policy':'only verified idle GPUs; never stop other training'}
        dispatch.write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(value))


if __name__=='__main__':
    raise SystemExit(main())
