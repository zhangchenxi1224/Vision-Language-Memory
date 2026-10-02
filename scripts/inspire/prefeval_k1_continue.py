"""Wait for the fixed teacher banks, then dispatch paired FM and frozen readback."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-k1-l0-l2-20260924')

def main(args):
    control=RUN/'continuation'
    control.mkdir(exist_ok=True)
    lock=(control/'active.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (control/'supervisor.pid').write_text(str(os.getpid()))
    started=time.monotonic()
    while True:
        failed=list((RUN/'pilot').glob('*/failed-*.txt'))
        if failed:
            raise RuntimeError('Teacher worker failed: '+str(failed))
        ready=all((RUN/'pilot'/arm/f'finished-{shard}.json').exists() for arm in ['A','B'] for shard in range(2))
        if ready:
            break
        if time.monotonic()-started > 7200:
            raise TimeoutError('Teacher wait exceeded two hours; inspect before continuing')
        time.sleep(30)
    children=[]
    for arm,kind,gpu,script in [
        ('A','readback',0,'run_prefeval_k1_teacher_readback.sh'),
        ('A','writer',1,'run_prefeval_k1_writer.sh'),
        ('B','readback',2,'run_prefeval_k1_teacher_readback.sh'),
        ('B','writer',3,'run_prefeval_k1_writer.sh'),
    ]:
        if args.gpu_count==2:
            gpu=0 if arm=='A' else 1
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'K1_CODE_ROOT':str(args.code_root)}
        log=(control/f'{arm}-{kind}.log').open('a')
        process=subprocess.Popen(['bash',str(args.code_root/'scripts/inspire'/script),arm],cwd=args.code_root,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append((arm,kind,process,log))
    (control/'children.json').write_text(json.dumps([{'arm':a,'kind':k,'pid':p.pid} for a,k,p,_ in children],indent=2))
    codes=[]
    for arm,kind,p,log in children:
        code=p.wait()
        codes.append(code)
        (control/f'{arm}-{kind}-exit.txt').write_text(str(code))
        log.close()
    (control/'exit-status.txt').write_text(str(int(any(codes))))
    if any(codes):
        raise RuntimeError('One or more continuation workers failed; inspect preserved logs')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-count',type=int,choices=[2,4],default=4)
    p.add_argument('--code-root',type=Path,default=ROOT)
    main(p.parse_args())
