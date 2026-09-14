"""Finish the fixed4832 endpoint and run four disjoint validation lanes plus CLI replay."""
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
PROJECT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
RUNS = PROJECT / 'runs/dreamlite-official-alignment'
MODELS = Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--logical-sampling-commit')
    parser.add_argument('--parent-run', type=Path)
    parser.add_argument('--inference-condition', choices=('native', 'training_raw'), default='native')
    parser.add_argument('--raw-control-run', type=Path)
    parser.add_argument('--prior-suite-status', type=Path)
    parser.add_argument('--validation-set', choices=('registered', 'fresh_wording_v1'), default='registered')
    parser.add_argument('--prior-validation-status', type=Path)
    a = parser.parse_args()
    if not math.isfinite(a.deadline_unix) or a.deadline_unix <= time.time():
        raise ValueError('A finite future suite deadline is required')
    if (len(a.expected_commit) != 40 or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != a.expected_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()):
        raise ValueError('Require the explicitly locked clean validation source')
    if bool(a.logical_sampling_commit) != bool(a.parent_run):
        raise ValueError('Paired sampling validation requires explicit parent path and exact training commit')
    parent = a.parent_run or RUNS / '84cdfdb-broader151-full4832'
    raw_control = a.inference_condition == 'training_raw'
    fresh_validation = a.validation_set == 'fresh_wording_v1'
    generated_source = a.logical_sampling_commit == 'ef163b26e33f62c496ed0da8744ebb7bf1163873'
    continuation = generated_source or a.logical_sampling_commit == '4fbc85725d78427235757ace2661d086b896a97f'
    native_baseline = a.logical_sampling_commit == '03f8467e5a1201c2dbd9d12484bf2338d7837727'
    prior_validation_commit = '7b82309eb49e913d8f028230ac5f79c743d28a46'
    if fresh_validation:
        if native_baseline:
            prior_validation_commit = '9e27050ea3fe1e7d54fe81714244f93ae07bac81'
            if (raw_control or parent != RUNS / '03f8467-native-condition-full4832'
                    or a.prior_validation_status != RUNS / '9e27050-logical-completion-suite-status.json'):
                raise ValueError('Native baseline requires the fixed 03 endpoint and its complete registered suite')
        elif continuation:
            prior_validation_commit = a.expected_commit
            expected_parent = 'ef163b2-generated-source-full4832' if generated_source else '4fbc857-clear-retention-full4832'
            if (raw_control or parent != RUNS / expected_parent
                    or a.prior_validation_status != RUNS / (a.expected_commit[:7] + '-logical-completion-suite-status.json')):
                raise ValueError('Continuation regression requires its own complete registered suite')
        elif (raw_control or a.logical_sampling_commit != 'b9f90e956eea7bda15f638c8877919941ce4fec5'
                or parent != RUNS / 'b9f90e9-historical-wording-full4832'
                or a.prior_validation_status != RUNS / '7b82309-logical-completion-suite-status.json'):
            raise ValueError('Fresh acceptance requires the fixed b9 endpoint and its full registered suite')
    elif a.prior_validation_status is not None:
        raise ValueError('Prior validation dependency is specific to fresh acceptance')
    if raw_control:
        if (a.logical_sampling_commit != 'bb34092ab0d1292c87d16d9632716b218f54054b'
                or a.raw_control_run != RUNS/'1f86d56-broader-raw-condition'
                or a.prior_suite_status != RUNS/'9e27050-logical-completion-suite-status.json'):
            raise ValueError('Require fixed raw control and prior native suite before reusing four GPUs')
    elif a.raw_control_run is not None or a.prior_suite_status is not None:
        raise ValueError('Raw-control dependencies require explicit raw inference policy')
    prefix = a.expected_commit[:7] + ('-raw-condition' if raw_control else ('-logical' if a.logical_sampling_commit else '-broader'))
    if fresh_validation:
        prefix = a.expected_commit[:7] + '-fresh-wording'
    status_name = prefix + '-completion-suite' if a.logical_sampling_commit else 'broader151-completion-suite'
    status = RUNS / (status_name + '-status.json')
    import fcntl
    lock = (RUNS / (status_name + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if status.exists():
        raise ValueError('Suite already has evidence; inspect before resuming any stage')
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    cpu_env = {**env, 'CUDA_VISIBLE_DEVICES': ''}
    protocol_arguments = ['--logical-sampling-commit', a.logical_sampling_commit] if a.logical_sampling_commit else []
    condition_arguments = ['--inference-condition', 'training_raw'] if raw_control else []
    validation_arguments = ['--validation-set', a.validation_set] if fresh_validation else []
    def record(stage, state, **extra):
        temporary = status.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'stage': stage, 'state': state, 'time_unix': time.time(),
            'deadline_unix': a.deadline_unix, 'validation_commit': a.expected_commit, 'parent': str(parent), **extra}, indent=2) + '\n')
        temporary.replace(status)
    def stop(child):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    def execute(script, arguments, label, *, cpu=True):
        remaining = a.deadline_unix - time.time()
        if remaining <= 0:
            raise TimeoutError('Suite deadline reached')
        command = [sys.executable, '-u', str(ROOT / script), *map(str, arguments)]
        record(label, 'running', command=command)
        with (RUNS / (prefix + '-' + label + '.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=cpu_env if cpu else env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = child.wait(timeout=remaining)
            except BaseException:
                stop(child)
                raise
        if returncode:
            stop(child)
            raise RuntimeError(label + ' failed with code ' + str(returncode))
    lanes = [('single_writes', 'confirmation', 0, None), ('rgb_chains', 'chains', 1, None),
             ('historical_prefixes', 'prefix0', 2, 0), ('historical_prefixes', 'prefix1', 3, 1)]
    outputs = {label: RUNS / (prefix + '-' + label) for _, label, _, _ in lanes}
    package, prepared, inference = [RUNS / (prefix + '-' + label) for label in ('package', 'parity', 'inference')]
    try:
        if fresh_validation:
            record('waiting_for_registered_suite', 'waiting')
            while True:
                if time.time() >= a.deadline_unix:
                    raise TimeoutError('Registered suite did not finish before fresh acceptance deadline')
                if a.prior_validation_status.exists():
                    prior = json.loads(a.prior_validation_status.read_bytes())
                    if prior['validation_commit'] != prior_validation_commit or prior['parent'] != str(parent):
                        raise ValueError('The preceding validation identity changed')
                    if prior['state'] in ('failed', 'needs_attention'):
                        raise RuntimeError('The registered suite failed operationally; inspect its evidence')
                    if prior['state'] == 'completed' and prior['stage'] == 'all_registered_workloads_finished':
                        break
                time.sleep(15)
        if raw_control:
            record('waiting_for_native_suite_and_full_raw_evidence', 'waiting')
            while True:
                paths = [a.prior_suite_status, RUNS/'1f86d56-raw-evidence-driver-status.json']
                states = [json.loads(path.read_bytes()) for path in paths]
                if any(state['state'] in ('failed', 'needs_attention') for state in states):
                    raise RuntimeError('A prerequisite failed; preserve it and inspect before more GPU work')
                if all(state['state'] == 'completed' for state in states):
                    break
                if time.time() >= a.deadline_unix:
                    raise TimeoutError('Native suite or full raw evidence did not finish before the deadline')
                time.sleep(20)
            from scripts.reporting.collect_transition_endpoint import sha
            evidence = RUNS/'1f86d56-raw-condition-verified-evidence.tgz'
            if sha(evidence) != states[1]['archive_sha256']:
                raise ValueError('Independent full raw evidence archive differs from CPU completion')
        record('waiting_for_fixed_endpoint', 'waiting')
        while not (parent / 'terminal.json').exists():
            if time.time() >= a.deadline_unix:
                raise TimeoutError('Parent did not complete before the suite deadline')
            time.sleep(15)
        terminal = json.loads((parent / 'terminal.json').read_bytes())
        if terminal.get('state') != 'completed':
            record('parent_not_completed', 'needs_attention', parent_terminal=terminal)
            return 75
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            if time.time() >= a.deadline_unix:
                raise TimeoutError('GPUs remain occupied; no work was evicted')
            time.sleep(10)
        if a.deadline_unix - time.time() < 60 * 60:
            raise TimeoutError('Require one hour for the complete four-lane validation and package replay')
        if any(path.exists() for path in (*outputs.values(), package, prepared, inference)):
            raise ValueError('Output exists; refuse duplicate execution')
        bank = parent / 'bank/manifest.json'
        if not raw_control and not fresh_validation:
            execute('scripts/reporting/collect_broader_endpoint.py', ['--run', parent, '--bank', bank,
                '--output-prefix', RUNS / (prefix + '-endpoint'), *protocol_arguments], 'endpoint-collection')
        if a.logical_sampling_commit in ('b9f90e956eea7bda15f638c8877919941ce4fec5', '4fbc85725d78427235757ace2661d086b896a97f', 'ef163b26e33f62c496ed0da8744ebb7bf1163873') and not fresh_validation:
            execute('scripts/reporting/collect_native_endpoint_tensors.py', ['--run', parent,
                '--output-prefix', RUNS / (prefix + '-final-tensors'),
                '--expected-source-commit', a.expected_commit, *protocol_arguments], 'final-tensor-collection')
        children, logs = [], []
        try:
            for mode, label, device, lane in lanes:
                command = [sys.executable, '-u', str(ROOT / 'scripts/probes/official_broader_confirmation.py'),
                    '--parent-run', str(parent), '--output', str(outputs[label]), '--mode', mode,
                    '--device', str(device), '--deadline-unix', str(a.deadline_unix), '--diagnostic', *protocol_arguments,
                    *condition_arguments, *validation_arguments]
                if raw_control:
                    command += ['--raw-control-run', str(a.raw_control_run)]
                if lane is not None:
                    command += ['--prefix-lane', str(lane)]
                log = (RUNS / (prefix + '-' + label + '.log')).open('w')
                logs.append(log)
                children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            record('four_independent_validation_lanes', 'running', worker_pids=[child.pid for child in children],
                outputs={key: str(value) for key, value in outputs.items()})
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError('An independent lane failed; retain all partial evidence')
                if time.time() >= a.deadline_unix:
                    raise TimeoutError('Independent validation reached its deadline')
                time.sleep(10)
            if any(child.returncode for child in children):
                raise RuntimeError('Independent validation failed')
        finally:
            for child in children:
                if child.poll() is None or child.returncode:
                    stop(child)
            for log in logs:
                log.close()
        summaries = {}
        for label, output in outputs.items():
            execute('scripts/reporting/collect_broader_validation.py', ['--run', output, '--parent', parent,
                '--bank', bank, '--output-prefix', output, '--expected-probe-commit', a.expected_commit,
                *protocol_arguments, *condition_arguments, *validation_arguments], label + '-collection')
            summaries[label] = json.loads(Path(str(output) + '-summary.json').read_bytes())
        execute('scripts/inference/export_rgb_writer.py', ['--parent-run', parent, '--output', package,
            *condition_arguments], 'package-export')
        execute('scripts/probes/rgb_package_parity.py', ['prepare', '--reference', outputs['chains'], '--package', package,
            '--output', prepared, '--broader', *protocol_arguments], 'package-prepare')
        execute('scripts/inference/rgb_memory.py', ['--package', package,
            '--base-model', MODELS / 'DreamLite-base-a9a0f15-20260907',
            '--official-source', PROJECT / 'Vision-Language-Memory/third_party/DreamLite',
            '--reader-model', MODELS / 'Qwen3-VL-4B-Instruct', '--commands', prepared / 'commands.jsonl', '--output', inference],
            'independent-package-inference', cpu=False)
        execute('scripts/probes/rgb_package_parity.py', ['verify', '--prepared', prepared, '--inference', inference], 'package-parity')
        record('all_registered_workloads_finished', 'completed',
            functional_all_registered_correct=all(value['all_generated_correct_eos'] for value in summaries.values()),
            matched_results={label: [value['matched_correct_eos'], value['matched_rows']] for label, value in summaries.items()},
            scope=('Previously observed complete wording/noise matrices measured on the fixed 03 parent endpoint, all four lanes and actual CLI replay; not a new holdout.' if native_baseline and fresh_validation else
                'Previously observed complete wording/noise matrices used as continuation regression, with all four lanes and real CLI replay.' if continuation else
                'New event wording and noise on seen semantic questions, complete four lanes and actual CLI replay.' if fresh_validation else
                'Observed c2ec407 cases reused as a paired sampling diagnostic; no fresh holdout claim.' if a.logical_sampling_commit else
                'Seen questions; fresh transition expressions and noise, historical original/reworded full prefixes. Not unseen entities or simultaneous multi-fact retention.'))
        return 0
    except BaseException as error:
        record('suite_error', 'failed', error=str(error), functional_success=False)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
