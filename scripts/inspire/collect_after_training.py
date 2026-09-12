"""Wait for one live training process, then collect its sealed endpoint on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'bank', 'collector', 'source', 'output-prefix', 'status'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--parent-pid', type=int, required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    a = p.parse_args()
    collector_sha = digest(a.collector)
    command = [sys.executable, '-u', str(a.collector), '--run', str(a.run), '--bank', str(a.bank), '--output-prefix', str(a.output_prefix)]
    proc = Path(f'/proc/{a.parent_pid}/cmdline')
    if not proc.exists() or b'train_latent_bank_unet.py' not in proc.read_bytes():
        raise ValueError('The specified training worker is not live')
    with a.status.open('x') as stream:
        json.dump({'state': 'waiting_for_training', 'parent_pid': a.parent_pid,
                   'collector_sha256': collector_sha, 'command': command, 'deadline_unix': a.deadline_unix}, stream)
    def status(state, **extra):
        temporary = a.status.with_suffix('.tmp')
        temporary.write_text(json.dumps({'state': state, 'parent_pid': a.parent_pid,
            'collector_sha256': collector_sha, 'command': command, **extra}, indent=2) + '\n')
        temporary.replace(a.status)
    try:
        missing = 0
        while time.time() < a.deadline_unix:
            terminal = a.run / 'terminal.json'
            if terminal.exists():
                current = json.loads(terminal.read_text())
                if current.get('state') == 'completed':
                    break
                if current.get('state') in ('failed', 'paused', 'error'):
                    raise RuntimeError('Parent ended without a completed endpoint: ' + str(current.get('state')))
            live = proc.exists() and b'train_latent_bank_unet.py' in proc.read_bytes()
            missing = 0 if live else missing + 1
            if missing >= 3:
                raise RuntimeError('Worker disappeared without a completed terminal after grace period')
            time.sleep(20)
        else:
            raise TimeoutError('Collection wait deadline reached')
        if digest(a.collector) != collector_sha:
            raise RuntimeError('Collector changed while waiting')
        status('collecting_completed_endpoint')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'PYTHONPATH': str(a.source / 'src')}
        subprocess.run(command, env=env, check=True, timeout=max(1, a.deadline_unix - time.time()))
        if digest(a.collector) != collector_sha:
            raise RuntimeError('Collector changed during collection')
        status('completed', summary_sha256=digest(str(a.output_prefix) + '-summary.json'))
        return 0
    except Exception as error:
        status('failed', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
