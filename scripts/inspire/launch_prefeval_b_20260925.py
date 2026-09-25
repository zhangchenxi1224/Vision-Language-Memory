"""Launch one explicitly named independent lane on the current authorized host."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b-mcq-20260925')


def main(args):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    receipt=OUTPUT/f'launch-{args.lane}.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        try:
            os.kill(old['pid'],0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('A recorded process is still alive; inspect it before relaunch')
    if args.lane=='identity64':
        command=['bash','scripts/inspire/run_prefeval_k1_identity64.sh']
    elif args.lane=='canonical64':
        command=['bash','scripts/inspire/run_prefeval_k1_canonical64_readback.sh']
    elif args.lane.startswith('ack-'):
        command=['bash','scripts/inspire/run_prefeval_k1_neutral_ack.sh',args.lane[-1]]
    elif args.lane=='readback':
        command=['/inspire/ssd/project/exploration-topic/czxs26210936/envs/vlm-r3-ngc2502/bin/python',
                 'scripts/inspire/run_prefeval_k1_b_readback_queue.py']
    else:
        command=['bash','scripts/inspire/run_prefeval_k1_robust730.sh']
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),K1_CODE_ROOT=str(ROOT))
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log=OUTPUT/f'launch-{args.lane}-{stamp}.log'
    with log.open('w') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    record={'lane':args.lane,'pid':process.pid,'gpu':args.gpu,'command':command,'code_root':str(ROOT),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'time_utc':stamp,'log':str(log),'host':os.uname().nodename}
    receipt.write_text(json.dumps(record,indent=2))
    (OUTPUT/f'launch-{args.lane}-{stamp}.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k!='host'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lane',choices=['identity64','canonical64','ack-0','ack-1','robust730','readback'],required=True)
    p.add_argument('--gpu',type=int,required=True)
    main(p.parse_args())
