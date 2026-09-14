"""Wait for both complete validation suites, then read their full PNG matrices on four idle GPUs."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--reader-model', type=Path, required=True)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    p.add_argument('--continuation-validation-commit')
    a = p.parse_args()
    from scripts.experiments.png_readback_protocol import source_spec, LANES, plan
    sources, _ = source_spec(a.continuation_validation_commit)
    protocol_arguments = (['--continuation-validation-commit', a.continuation_validation_commit]
        if a.continuation_validation_commit else [])
    if (len(a.expected_commit) != 40 or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != a.expected_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()):
        raise ValueError('Require the exact clean source commit')
    if not math.isfinite(a.deadline_unix) or time.time() >= a.deadline_unix:
        raise ValueError('Require finite future deadline')
    prefix = a.expected_commit[:7] + '-png-readback'
    status = a.runs / (prefix + '-status.json')
    import fcntl
    lock = (a.runs / (prefix + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if status.exists():
        raise ValueError('Existing readback evidence; inspect instead of overwriting')
    def record(stage, state, **extra):
        tmp = status.with_suffix('.tmp')
        tmp.write_text(json.dumps({'stage': stage, 'state': state, 'time_unix': time.time(),
            'readback_commit': a.expected_commit, 'deadline_unix': a.deadline_unix, **extra}, indent=2) + '\n')
        tmp.replace(status)
    def deadline():
        if time.time() >= a.deadline_unix:
            raise TimeoutError('PNG suite reached its fixed deadline')
    def stop(child):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    try:
        record('waiting_for_both_complete_suites', 'waiting', plan=plan(a.continuation_validation_commit))
        while True:
            deadline()
            done = []
            for _, (commit, source_prefix) in sources.items():
                path = a.runs / (source_prefix + '-completion-suite-status.json')
                if not path.exists():
                    done.append(False)
                    continue
                value = json.loads(path.read_bytes())
                parent_name = ('4fbc857-clear-retention-full4832' if a.continuation_validation_commit else 'b9f90e9-historical-wording-full4832')
                if value['validation_commit'] != commit or value['parent'] != str(a.runs / parent_name):
                    raise ValueError('Preceding suite identity differs')
                if value['state'] in ('failed', 'needs_attention'):
                    raise RuntimeError('Preceding suite failed operationally; preserve and inspect it')
                done.append(value['state'] == 'completed' and value['stage'] == 'all_registered_workloads_finished')
            if all(done):
                break
            time.sleep(15)
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            deadline()
            time.sleep(15)
        summaries = {}
        for validation_set, (_, source_prefix) in sources.items():
            children, logs, outputs = [], [], {}
            try:
                for device, lane in enumerate(LANES):
                    output = a.runs / (prefix + '-' + validation_set + '-' + lane)
                    if output.exists():
                        raise ValueError('Refuse to overwrite a previous readback lane')
                    outputs[lane] = output
                    log = Path(str(output) + '.log').open('w')
                    logs.append(log)
                    command = [sys.executable, '-u', str(ROOT / 'scripts/probes/complete_png_readback.py'),
                        '--runs', str(a.runs), '--reader-model', str(a.reader_model), '--output', str(output),
                        '--validation-set', validation_set, '--lane', lane, '--device', str(device),
                        '--expected-commit', a.expected_commit, '--deadline-unix', str(a.deadline_unix), *protocol_arguments]
                    children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
                record('reading_' + validation_set, 'running', worker_pids=[child.pid for child in children])
                while any(child.poll() is None for child in children):
                    deadline()
                    if any(child.poll() not in (None, 0) for child in children):
                        raise RuntimeError('PNG lane failed; retain its partial records')
                    time.sleep(10)
                if any(child.returncode for child in children):
                    raise RuntimeError('PNG readback lane failed')
            finally:
                for child in children:
                    stop(child)
                for log in logs:
                    log.close()
            for lane, output in outputs.items():
                deadline()
                command = [sys.executable, '-u', str(ROOT / 'scripts/reporting/collect_png_readback.py'),
                    '--run', str(output), '--source', str(a.runs / (source_prefix + '-' + lane)),
                    '--expected-commit', a.expected_commit, '--output-prefix', str(output), '--archive', *protocol_arguments]
                with Path(str(output) + '-collection.log').open('w') as log:
                    child = subprocess.Popen(command, cwd=ROOT, env={**env, 'CUDA_VISIBLE_DEVICES': ''},
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        if child.wait(timeout=max(1, a.deadline_unix - time.time())):
                            raise RuntimeError('Complete PNG evidence collection failed')
                    finally:
                        stop(child)
                summaries[validation_set + '/' + lane] = json.loads(Path(str(output) + '-summary.json').read_bytes())
        record('all_png_readback_finished', 'completed',
            all_matched_correct_eos=all(value['png']['all_generated_correct_eos'] for value in summaries.values()),
            all_chain_parity_passed=all(value['chain_parity_passed'] for value in summaries.values()),
            matched_correct_eos=sum(value['png']['matched_correct_eos'] for value in summaries.values()),
            raw_rows=sum(value['png']['raw_rows'] for value in summaries.values()))
        return 0
    except BaseException as exc:
        record('exception', 'failed', error=repr(exc))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
