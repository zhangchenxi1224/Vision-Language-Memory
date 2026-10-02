"""Generate/read the exact SFT initial exchange, after the fixed benchmark pilot."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.prefeval_official_ab import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['A','B'],required=True);p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,default=2);a=p.parse_args()
    for arm in ('A','B'):
        if not (a.output/'pipeline'/arm/'complete.json').exists():
            raise RuntimeError('Finish and analyze the original fixed benchmark before this diagnostic')
    destination=a.output/'pipeline-train-input'/a.arm
    marker=destination/f'complete-{a.shard}.json'
    if marker.exists():return
    common=['--output',str(a.output),'--arm',a.arm,'--stage','write','--split','training-initial',
        '--shard',str(a.shard),'--shards',str(a.shards),'--questions',
        str(ROOT/'reports/prefeval-official-alignment-20260923/pilot-questions-3plus2.json')]
    for phase in ('rollout','students'):
        cmd=[sys.executable,'-u',str(ROOT/'scripts/eval/prefeval_official_rgb.py'),phase,*common]
        child=subprocess.Popen(cmd,cwd=ROOT)
        write(destination/f'{phase}-{a.shard}.json',dict(pid=child.pid,command=cmd))
        if child.wait():raise RuntimeError(f'Train-input {phase} failed; preserve logs and endpoints')
    write(marker,dict(status='exact_train_input_diagnostic_complete',arm=a.arm,shard=a.shard,shards=a.shards))


if __name__=='__main__':main()
