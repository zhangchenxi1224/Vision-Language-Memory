"""Bounded two-GPU consumer of completed upstream Writer artifacts."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.prefeval_writer_readout import (
    ENDPOINTS, verify_completion, verify_assets, read, save_once, cohort_rows,
)
from scripts.experiments.prefeval_k1_data import sha
from scripts.inspire.run_prompt_matching_parallel import write_json


def accrued_seconds(output):
    total = 0
    for path in (output / 'attempts').glob('*.json'):
        value = read(path)
        if 'finished' not in value:
            raise ValueError('Unsettled previous worker; prove stopped and settle cost first: ' + str(path))
        total += value['finished'] - value['started']
    return total


def other_seconds(campaign):
    total = 0
    for stage in ('smoke', 'pilot', 'readout-v1'):
        for kind in ('receipts', 'attempts'):
            for p in (campaign / stage / kind).glob('*.json'):
                v = read(p)
                total += v['finished'] - v['started']
    return total


def execute(job, output, deadline):
    receipt = output / 'receipts' / (job['name'] + '.json')
    digest = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    if receipt.exists():
        old = read(receipt)
        if old['command_sha256'] != digest:
            raise ValueError('Frozen command changed')
        return old
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['gpu']), PYTHONUNBUFFERED='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONHASHSEED='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    attempt = output / 'attempts' / (job['name'] + '-' + uuid.uuid4().hex + '.json')
    value = dict(job=job, command_sha256=digest, hostname=socket.gethostname(), started=time.time())
    write_json(attempt, value)
    with (output / 'logs' / (job['name'] + '.log')).open('a') as log:
        process = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        value['pid'] = process.pid
        write_json(attempt, value)
        try:
            code = process.wait(timeout=max(0.1, deadline-time.monotonic()))
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            code = 124
        value.update(finished=time.time(), exit_code=code)
        write_json(attempt, value)
    if code:
        raise RuntimeError(job['name'] + ' failed; attempt preserved')
    write_json(receipt, value)
    return value


def main(args):
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Require frozen clean checkout')
    args.output.mkdir(parents=True, exist_ok=True)
    claim = args.output / 'active-owner'
    claim.mkdir()
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write_json(claim / 'owner.json', {'pid':os.getpid(), 'host':socket.gethostname(), 'started':time.time(), 'commit':commit})
        for name in ('attempts', 'receipts', 'logs', 'endpoints'):
            (args.output / name).mkdir(exist_ok=True)
        protocol = ROOT / 'configs/experiments/context_readout_audit.json'
        plan = {'commit':commit, 'queries_sha256':sha(protocol), 'plan_sha256':sha(ROOT / 'reports/context-coverage-20261006/WRITER_READOUT_PLAN.md'),
                'endpoints':list(ENDPOINTS), 'cohorts':{s:[r['base_pair_id'] for r in rs] for s,rs in cohort_rows().items()},
                'gpu_hours_cap':4, 'campaign_cap':16, 'base':str(args.base), 'reader':str(args.reader), 'official_source':str(args.official_source)}
        save_once(args.output / 'plan.json', plan)
        remaining = min(4*3600-accrued_seconds(args.output), 16*3600-other_seconds(args.output.parent)-accrued_seconds(args.output))
        if remaining <= 60:
            raise ValueError('Budget exhausted; no automatic renewal')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise ValueError('GPUs occupied')
        # Reserve stop/kill grace within the budget, conservatively charge two GPUs even in a one-job group.
        deadline = time.monotonic() + min(6*3600, (remaining-60)/2)
        write_json(args.output / 'status.json', {'status':'running', 'commit':commit, 'remaining_gpu_seconds':remaining})
        def bind_ready():
            ready = []
            for endpoint in ENDPOINTS:
                value = verify_completion(endpoint)
                if value is not None:
                    save_once(args.output / 'endpoints' / (endpoint + '.json'), value)
                    ready.append(endpoint)
            return ready
        ready = bind_ready()
        if 'b730' not in ready:
            raise ValueError('Completed parent missing')
        verify_assets(args.output, ['b730'])
        def run_rollouts(endpoints):
            jobs = []
            for e in endpoints:
                if e == 'b730':
                    continue
                jobs.append({'name':'rollout-'+e, 'gpu':ENDPOINTS.index(e)-1, 'command':[sys.executable,
                    'scripts/experiments/prefeval_writer_readout.py','rollout','--endpoint',e,
                    '--output',str(args.output),'--base',str(args.base),'--official-source',str(args.official_source)]})
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                for f in [pool.submit(execute,j,args.output,deadline) for j in jobs]:
                    f.result()
        run_rollouts(ready)
        refreshed = bind_ready()
        run_rollouts([e for e in refreshed if e not in ready])
        if set(refreshed) != set(ENDPOINTS):
            write_json(args.output / 'status.json', {'status':'waiting_upstream', 'ready':refreshed,
                'gpu_hours':accrued_seconds(args.output)/3600, 'time':time.time()})
            return
        verify_assets(args.output)
        jobs = [{'name':f'evaluate-{shard}', 'gpu':shard, 'command':[sys.executable,
            'scripts/experiments/prefeval_writer_readout.py','evaluate','--shard',str(shard),
            '--output',str(args.output),'--reader',str(args.reader),'--protocol',str(protocol)]} for shard in range(2)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for f in [pool.submit(execute,j,args.output,deadline) for j in jobs]:
                f.result()
        from scripts.reporting.writer_readout_report import report
        result = report(args.output, protocol)
        write_json(args.output / 'status.json', {'status':'completed', 'rows':result['rows'],
            'gpu_hours':accrued_seconds(args.output)/3600, 'time':time.time()})
    except BaseException as exc:
        write_json(args.output / 'status.json', {'status':'failed', 'error':str(exc), 'time':time.time()})
        raise
    finally:
        (claim / 'owner.json').unlink(missing_ok=True)
        claim.rmdir()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output','base','reader','official-source'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
