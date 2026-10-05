"""Bounded read-only follow-up; shares the original campaign's 16 GPUh ceiling."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.inspire.run_prompt_matching_parallel import run_job,write_json


def main(args):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Frozen clean checkout required')
    args.output.mkdir(parents=True,exist_ok=True)
    claim=args.output/'active-owner';claim.mkdir()
    try:
        write_json(claim/'owner.json',{'pid':os.getpid(),'hostname':socket.gethostname(),'started':time.time(),'commit':commit})
        for name in ('receipts','attempts','logs'):(args.output/name).mkdir(exist_ok=True)
        jobs=[]
        for shard in range(2):
            cmd=[sys.executable,'scripts/experiments/prefeval_context_readout.py','--shard',str(shard)]
            for name in ('pilot','reference','reader','ids_file','protocol','output'):
                cmd+=['--'+name.replace('_','-'),str(getattr(args,name))]
            jobs.append({'name':f'readout-{shard}','gpu':shard,'command':cmd})
        plan={'commit':commit,'jobs':jobs,'read_only':True,'gpu_hours_cap':1,'shared_campaign_cap':16}
        if (args.output/'plan.json').exists() and json.loads((args.output/'plan.json').read_text())!=plan:raise ValueError('Plan changed')
        write_json(args.output/'plan.json',plan)
        for f in (args.output/'receipts').glob('*.json'):
            v=json.loads(f.read_text())
            if v['exit_code']:
                archive=args.output/'attempts'/(f.stem+'-'+str(v['started'])+'.json')
                if not archive.exists():write_json(archive,v)
                f.unlink()
        def seconds(paths):
            return sum(v['finished']-v['started'] for p in paths for v in [json.loads(p.read_text())])
        prior=seconds(list((args.output/'receipts').glob('*.json'))+list((args.output/'attempts').glob('*.json')))
        training=seconds(list((args.pilot/'receipts').glob('*.json'))+list((args.pilot/'attempts').glob('*.json')))
        remaining=min(3600-prior,16*3600-training-prior)
        if remaining<=0:raise ValueError('Readout/campaign GPU budget exhausted')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('GPU processes already present')
        deadline=time.monotonic()+remaining/2
        write_json(args.output/'status.json',{'status':'running','commit':commit,'prior_gpu_seconds':prior,'pilot_gpu_seconds':training})
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for f in [pool.submit(run_job,j,args.output,deadline) for j in jobs]:print(json.dumps(f.result()),flush=True)
        subprocess.run([sys.executable,'scripts/reporting/context_readout_report.py','--run',str(args.output),
            '--ids-file',str(args.ids_file),'--protocol',str(args.protocol),'--output',str(args.output/'comparison.json')],cwd=ROOT,check=True)
        write_json(args.output/'status.json',{'status':'completed','commit':commit,'time':time.time()})
    except BaseException as error:
        write_json(args.output/'status.json',{'status':'failed','error':str(error),'time':time.time()});raise
    finally:
        (claim/'owner.json').unlink(missing_ok=True);claim.rmdir()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot','reference','reader','ids-file','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
