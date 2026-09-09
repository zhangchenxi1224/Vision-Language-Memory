"""Idempotent launch of the isolated single-target causal audit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
P = Path('/inspire/ssd/project/exploration-topic/czxs26210936')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lease-minutes', type=int, required=True)
    args = parser.parse_args()
    assert 15 <= args.lease_minutes <= 105
    assert socket.gethostname().startswith('vlm-unet-diag-h200x2-20260909--')
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == args.commit
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dispatch = Path(str(args.output)+'.dispatch.json')
    with Path(str(args.output)+'.dispatch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if dispatch.exists():
            prior = json.loads(dispatch.read_text())
            proc = Path('/proc')/str(prior['pid'])/'cmdline'
            print(json.dumps({'status': 'already_dispatched', 'pid': prior['pid'],
                'alive': proc.exists() and str(args.output).encode() in proc.read_bytes()}))
            return
        assert not args.output.exists(), 'Prior output requires inspection, never overwrite'
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        assert not active.strip(), 'Allocation occupied; do not interrupt any training'
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='0,1', PYTHONHASHSEED='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        deadline = time.time()+args.lease_minutes*60
        command = [str(P/'envs/vlm-r3-ngc2502/bin/python'), '-u',
            str(ROOT/'scripts/analysis/audit_unet_single_target.py'),
            '--config', str(ROOT/'configs/experiments/unet_learnability.json'),
            '--original', str(P/'runs/unet-learnability/fd07eb1-20260909-r01'),
            '--output', str(args.output), '--expected-commit', args.commit,
            '--deadline', str(deadline), '--steps', '64']
        with Path(str(args.output)+'.log').open('ab', buffering=0) as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {'pid': proc.pid, 'hostname': socket.gethostname(), 'commit': args.commit,
            'output': str(args.output), 'epoch': time.time(), 'deadline_epoch': deadline, 'command': command}
        dispatch.write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
