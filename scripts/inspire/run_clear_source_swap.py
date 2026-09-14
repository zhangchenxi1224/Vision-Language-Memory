"""Replay fixed clear/no-op blocks through the unchanged CLI with crossed weights and RGB sources."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def collect(packet, output, plan_sha):
    plan = read(packet / 'plan.json')
    if sha(packet / 'plan.json') != plan_sha or len(plan['jobs']) != 32:
        raise ValueError('Changed fixed 32-job diagnostic plan')
    groups = {}
    mismatches = []
    for job in plan['jobs']:
        directory = output / job['name']
        commands_path = packet / job['commands']
        source_path = packet / job['initial_image']
        if sha(commands_path) != job['commands_sha256'] or sha(source_path) != job['initial_image_sha256']:
            raise ValueError('Changed command or actual source PNG')
        complete = read(directory / 'complete.json')
        if (complete['commands'], complete['writes'], complete['reads']) != (12, 2, 10):
            raise ValueError('Require all two writes and ten reads')
        if (complete['package_manifest_sha256'] != plan['models'][job['model']]['manifest_sha256']
                or complete['command_file_sha256'] != job['commands_sha256']):
            raise ValueError('Wrong actual CLI package or commands')
        rows = [json.loads(line) for line in (directory / 'results.jsonl').read_bytes().splitlines()]
        commands = [json.loads(line) for line in commands_path.read_bytes().splitlines()]
        if len(rows) != 12 or any(row['index'] != i or row['op'] != commands[i]['op'] for i, row in enumerate(rows)):
            raise ValueError('Changed or missing CLI result')
        writes = [row for row in rows if row['op'] == 'write']
        reads = [row for row in rows if row['op'] == 'read']
        group = groups.setdefault(job['model'] + '/' + job['source'], {'correct_eos': 0, 'reads': 0, 'jobs': []})
        correct = 0
        for index, row in enumerate(writes, 1):
            if row['memory_image'] != f'memory-{index:04d}.png' or sha(directory / row['memory_image']) != row['image_file_sha256']:
                raise ValueError('Changed actual output PNG')
            if job['model'] == job['source'] and row['image_file_sha256'] != job['reference']['png_sha256'][index-1]:
                mismatches.append({'job': job['name'], 'write': index, 'field': 'PNG bytes'})
        if sha(directory / 'memory-final.png') != complete['final_image_sha256'] or complete['final_image_sha256'] != writes[-1]['image_file_sha256']:
            raise ValueError('Final persisted PNG differs')
        for index, (row, expected) in enumerate(zip(reads, job['reference']['reads'], strict=True)):
            if row['query'] != expected['query']:
                raise ValueError('Reader query changed')
            correct += int(row['generated_token_ids'] == [2152, 4541, 21933, 151645])
            if job['model'] == job['source']:
                for field in ('query', 'raw', 'raw_with_special_tokens', 'input_token_ids', 'generated_token_ids', 'eos_token_ids', 'eos_reached', 'truncated', 'finish_reason'):
                    if row[field] != expected[field]:
                        mismatches.append({'job': job['name'], 'read': index, 'field': field})
        group['correct_eos'] += correct
        group['reads'] += 10
        group['jobs'].append({'name': job['name'], 'correct_eos': correct,
            'raw': [row['raw'] for row in reads], 'result_sha256': sha(directory / 'results.jsonl')})
    if set(groups) != {'old/old', 'old/new', 'new/old', 'new/new'} or any(group['reads'] != 80 for group in groups.values()):
        raise ValueError('Incomplete factorial diagnostic')
    return {'plan_sha256': plan_sha, 'groups': groups, 'diagonal_parity_pass': not mismatches,
        'diagonal_mismatches': mismatches, 'jobs': 32, 'writes': 64, 'reads': 320,
        'scope': 'Post-hoc causal diagnostic of sequence0/1 clear and following no-op, all four repetitions; not a new holdout or full acceptance score.'}


def run(a):
    packet, output = a.packet.resolve(), a.output.resolve()
    if sha(packet / 'plan.json') != a.plan_sha256 or sha(__file__) != a.script_sha256:
        raise ValueError('Packet or orchestration source differs')
    plan = read(packet / 'plan.json')
    runtime = Path(plan['runtime_root'])
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime, text=True).strip() != plan['runtime_commit']
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=runtime, text=True).strip()
            or sha(runtime / 'scripts/inference/rgb_memory.py') != plan['cli_sha256']):
        raise ValueError('Require the unchanged clean CLI runtime')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('Require actually idle GPUs; do not evict work')
    if len(subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True).splitlines()) != 4:
        raise ValueError('Require the allocated four-GPU instance')
    if output.exists() or time.time() >= a.deadline_unix or len(plan['jobs']) != 32:
        raise ValueError('Existing output, expired deadline, or incomplete plan')
    output.mkdir(parents=True)
    status = output / 'status.json'
    executed = []
    try:
        for offset in range(0, 32, 4):
            children, streams = [], []
            try:
                for device, job in enumerate(plan['jobs'][offset:offset+4]):
                    if sha(packet / job['commands']) != job['commands_sha256'] or sha(packet / job['initial_image']) != job['initial_image_sha256']:
                        raise ValueError('Changed initial PNG or commands')
                    package = Path(plan['models'][job['model']]['package'])
                    if sha(package / 'manifest.json') != plan['models'][job['model']]['manifest_sha256']:
                        raise ValueError('Changed package identity')
                    command = [sys.executable, '-u', str(runtime / 'scripts/inference/rgb_memory.py'),
                        '--package', str(package), '--base-model', plan['base_model'], '--official-source', plan['official_source'],
                        '--reader-model', plan['reader_model'], '--commands', str(packet / job['commands']),
                        '--initial-image', str(packet / job['initial_image']), '--device', f'cuda:{device}', '--output', str(output / job['name'])]
                    stream = (output / (job['name'] + '.log')).open('w')
                    streams.append(stream)
                    children.append(subprocess.Popen(command, cwd=runtime, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
                    executed.append({'name': job['name'], 'command': command, 'pid': children[-1].pid,
                        'initial_image_sha256': sha(packet / job['initial_image'])})
                write(output / 'executed.json', executed)
                write(status, {'state': 'running', 'jobs_started': len(executed), 'time_unix': time.time(), 'plan_sha256': a.plan_sha256})
                while any(child.poll() is None for child in children):
                    if time.time() >= a.deadline_unix or any(child.poll() not in (None, 0) for child in children):
                        raise RuntimeError('CLI failure or diagnostic deadline; retain outputs')
                    time.sleep(3)
                if any(child.returncode for child in children):
                    raise RuntimeError('CLI job failed')
            finally:
                for child in children:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                for stream in streams:
                    stream.close()
        result = collect(packet, output, a.plan_sha256)
        write(output / 'result.json', result)
        write(status, {'state': 'completed', 'time_unix': time.time(), 'plan_sha256': a.plan_sha256,
            'diagonal_parity_pass': result['diagonal_parity_pass'], 'result_sha256': sha(output / 'result.json')})
    except BaseException as error:
        write(status, {'state': 'failed', 'time_unix': time.time(), 'error': repr(error), 'plan_sha256': a.plan_sha256})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--script-sha256', required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    run(p.parse_args())
