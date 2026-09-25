"""One Reader GPU: full B teachers, then fixed C/I endpoints as they become ready."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
PROJECT=Path('/inspire/ssd/project/exploration-topic/czxs26210936')
RUN=PROJECT/'runs/prefeval-b-mcq-20260925'
OUTPUT=RUN/'readback-queue'
PYTHON=PROJECT/'envs/vlm-r3-ngc2502/bin/python'
READER='/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory/Qwen3-VL-4B-Instruct'


def execute(name,script,args):
    receipt=OUTPUT/(name+'-complete.json')
    if receipt.exists():
        return
    command=[str(PYTHON),str(ROOT/'scripts/experiments'/script),*map(str,args)]
    (OUTPUT/'active.json').write_text(json.dumps({'name':name,'command':command,'started_unix':time.time()},indent=2))
    with (OUTPUT/(name+'.log')).open('a') as log:
        subprocess.run(command,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)
    receipt.write_text(json.dumps({'command':command,'completed_unix':time.time()},indent=2))


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=(OUTPUT/'launcher.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',PYTHONHASHSEED='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    bank=PROJECT/'runs/prefeval-k1-scale730-20260924/teachers/B'
    for split,families in [('train','T1'),('pilot','T1,T2,T3,O1,O2')]:
        name='teacher-'+split
        execute(name,'prefeval_k1_evaluate.py',['--kind','teacher','--split',split,'--reader',READER,
            '--images',bank,'--output',OUTPUT/name,'--families',families,'--controls','memory,mismatch,blank,text','--tasks','mcq'])
    pending=[(lane,split,n) for lane in ['canonical64','identity64'] for split,n in [('pilot',64),('dev',90)]]
    deadline=time.monotonic()+7*24*3600
    while pending:
        advanced=False
        for lane,split,count in list(pending):
            images=(PROJECT/'runs/prefeval-k1-initial-source-retain-20260924/B'/split if lane=='canonical64'
                    else RUN/lane/split)
            if len(list(images.glob('*/seed-*/complete.json')))!=count*2:
                continue
            name=lane+'-'+split
            execute(name,'prefeval_k1_evaluate.py',['--kind','student','--split',split,'--reader',READER,
                '--images',images,'--output',OUTPUT/name,'--families','T2,T3,O1,O2','--prefixes','0,1,5,10',
                '--noise-chains','2','--controls','memory,mismatch,blank,text','--tasks','mcq'])
            for prefix in [0,10]:
                execute(name+f'-positions-{prefix}','prefeval_k1_position_probe.py',[
                    '--kind','student','--split',split,'--reader',READER,'--images',images,'--prefix',prefix,
                    '--order-mode','official-cyclic','--output',OUTPUT/(name+f'-positions-{prefix}.jsonl')])
            pending.remove((lane,split,count))
            advanced=True
        if not advanced:
            (OUTPUT/'active.json').write_text(json.dumps({'waiting_for':pending,'time_unix':time.time()}))
            if time.monotonic()>deadline:
                raise TimeoutError('Fixed endpoint queue exceeded seven days; inspect producers')
            time.sleep(30)


if __name__=='__main__':
    try:
        main()
    except BaseException:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        (OUTPUT/'exit-status.txt').write_text('1\n')
        raise
    (OUTPUT/'exit-status.txt').write_text('0\n')
