"""One-GPU-hour, two-worker read-only consumer; no training or PNG generation."""
import argparse
import concurrent.futures
from pathlib import Path
import os
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.experiments.prefeval_route_functional import (
    load_reference, verify_assets, read, save_once, sha, report, cohort_rows,
)
from scripts.inspire.run_writer_readout import execute, accrued_seconds, other_seconds
from scripts.inspire.run_prompt_matching_parallel import write_json


def remaining_seconds(output):
    previous=other_seconds(output.parent)+accrued_seconds(output.parent/'writer-readout-v1')
    current=accrued_seconds(output)
    return min(3600-current,16*3600-previous-current),previous,current


def main(args):
    if args.output.resolve() != args.reference.resolve().parent/'route-functional-v1':
        raise ValueError('Output must be the isolated route-functional-v1 sibling')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Require clean frozen checkout')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    protocol=ROOT/'configs/experiments/context_readout_audit.json'
    args.output.mkdir(parents=True,exist_ok=True)
    claim=args.output/'active-owner'
    claim.mkdir()
    try:
        write_json(claim/'owner.json',dict(pid=os.getpid(),host=socket.gethostname(),started=time.time(),commit=commit))
        for folder in ('attempts','receipts','logs'):
            (args.output/folder).mkdir(exist_ok=True)
        plan=dict(commit=commit,protocol_sha256=sha(protocol),
            plan_sha256=sha(ROOT/'reports/context-coverage-20261006/ROUTE_FUNCTIONAL_PLAN.md'),
            reference=str(args.reference),reader=str(args.reader),gpu_hours_cap=1,campaign_cap=16,
            expected_new_rows=8640,expected_combined_rows=23760)
        save_once(args.output/'plan.json',plan)
        left,previous,current=remaining_seconds(args.output)
        if left<=60:
            raise ValueError('Budget exhausted; no automatic renewal')
        _,reference=load_reference(args.reference,read(protocol),protocol)
        assets=verify_assets()
        save_once(args.output/'inputs.json',dict(reference=reference,upstream=assets))
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise ValueError('GPUs occupied')
        if len(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines())!=2:
            raise ValueError('Require the verified two-GPU allocation')
        foreign=[p for p in args.output.parent.glob('*/active-owner/owner.json') if p != claim/'owner.json']
        if foreign:
            raise ValueError('Another campaign owner exists')
        write_json(args.output/'status.json',dict(status='running',time=time.time(),commit=commit,
            previous_gpu_hours=previous/3600,remaining_gpu_seconds=left))
        # 30 GPU-seconds per worker reserved for termination grace; charge each process once.
        deadline=time.monotonic()+min(6*3600,(left-60)/2)
        jobs=[dict(name=f'evaluate-{i}',gpu=i,command=[sys.executable,
            'scripts/experiments/prefeval_route_functional.py','--output',str(args.output),
            '--reader',str(args.reader),'--protocol',str(protocol),'--shard',str(i)]) for i in range(2)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for future in [pool.submit(execute,j,args.output,deadline) for j in jobs]:
                future.result()
        result=report(args.output,args.reference,protocol)
        write_json(args.output/'status.json',dict(status='completed',time=time.time(),
            new_rows=result['new_rows'],combined_rows=result['combined_rows'],
            gpu_hours=accrued_seconds(args.output)/3600,campaign_gpu_hours=(previous+accrued_seconds(args.output))/3600))
    except BaseException as exc:
        write_json(args.output/'status.json',dict(status='failed',error=str(exc),time=time.time()))
        raise
    finally:
        (claim/'owner.json').unlink(missing_ok=True)
        claim.rmdir()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','reference','reader'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
