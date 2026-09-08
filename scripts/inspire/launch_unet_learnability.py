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
    args=p.parse_args()
    assert socket.gethostname().startswith(args.instance+'--'),'Wrong allocation'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==args.commit
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.dispatch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dispatch=args.output/'dispatch.json'
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
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        assert not active.strip(),'Allocation GPUs occupied; will not interrupt them'
        python=Path('/inspire/ssd/project/exploration-topic/czxs26210936/envs/vlm-r3-ngc2502/bin/python')
        assert python.is_file()
        deadline=time.time()+165*60
        command=[str(python),'-u',str(ROOT/'scripts/train/run_unet_learnability.py'),
            '--config',str(ROOT/'configs/experiments/unet_learnability.json'),
            '--output',str(args.output),'--expected-commit',args.commit,
            '--expected-hostname',socket.gethostname(),'--deadline',str(deadline)]
        env=os.environ.copy()
        env.pop('CUDA_VISIBLE_DEVICES',None)
        env.update(PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        with (args.output/'campaign.log').open('ab',buffering=0) as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,
                                  stderr=subprocess.STDOUT,start_new_session=True)
        value={'pid':proc.pid,'instance':args.instance,'hostname':socket.gethostname(),'command':command,
               'epoch':time.time(),'deadline_epoch':deadline,'source_commit':args.commit,
               'platform_auto_stop_minutes':180,'policy':'separate allocation; never stop other training'}
        dispatch.write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(value))


if __name__=='__main__':
    main()
