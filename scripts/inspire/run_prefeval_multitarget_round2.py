"""Training-only S1/C8 follow-up to the completed, immutable F1/F8/S8 pilot."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.inspire.run_prefeval_multitarget import (
    PROJECT, PYTHON, BASE, READER, OFFICIAL, PREFEVAL, PARENT, VARIANTS,
    WRITER, EVAL, TEACHER, save, immutable, sha)
from scripts.experiments.prefeval_multitarget_bank import freeze_bank, fixed_single_bank

PRIOR = PROJECT/'runs/prefeval-multitarget-20260927/pilot64'


def main(a):
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    state = {'host': os.uname().nodename, 'pid': os.getpid(), 'commit': commit, 'code': str(ROOT)}

    def status(stage, **extra):
        value = dict(state, stage=stage, time_utc=datetime.now(timezone.utc).isoformat(), **extra)
        save(a.output/'controller.json', value)
        print(json.dumps(value), flush=True)

    def jobs(label, commands):
        status(label)
        # Inspect the whole requested allocation before starting any worker.
        for gpu, _ in commands:
            busy = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            if busy:
                raise RuntimeError(f'GPU {gpu} occupied; refusing overlap: {busy}')
        children = []
        for gpu, command in commands:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            stream = (a.output/f'{label}-gpu{gpu}.log').open('a')
            command = list(map(str, command))
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
            children.append((child, stream))
            save(a.output/f'{label}-gpu{gpu}-process.json', dict(state, gpu=gpu, pid=child.pid, command=command))
        codes = []
        for child, stream in children:
            codes.append(child.wait())
            stream.close()
        if any(codes):
            status('failed', failed_stage=label, exit_codes=codes)
            raise RuntimeError(f'{label}: {codes}')

    previous_done = json.loads((PRIOR/'complete.json').read_text())
    assert previous_done['status'] == 'pilot_complete'
    assert previous_done['results_sha256'] == sha(PRIOR/'results.json')
    prior_protocol = json.loads((PRIOR/'protocol.json').read_text())
    assert sha(PARENT) == prior_protocol['parent_sha256']
    registered_ids = json.loads((PRIOR/'ids.json').read_text())['ids']
    assert len(registered_ids) == 64 and registered_ids == prior_protocol['ids']
    ids = registered_ids[:2] if a.smoke else registered_ids
    ids_file = a.output/'ids.json'
    immutable(ids_file, {'ids': ids, 'scope': 'same previously trained pilot64; no new-preference generalization'})
    targets = 2 if a.smoke else 8
    immutable(a.output/'protocol.json', {
        'commit': commit, 'prior_results_sha256': sha(PRIOR/'results.json'),
        'parent': str(PARENT), 'parent_sha256': prior_protocol['parent_sha256'],
        'F1_anchor_bank_sha256': sha(PRIOR/'F1-bank.json'),
        'S8_source_bank_sha256': sha(PRIOR/'S8-bank.json'),
        'ids': ids, 'arms': ['S1', 'C8'], 'smoke': a.smoke, 'targets': targets,
        'S1': 'fixed hash-selected one target per preference from the existing qualified S8 pool',
        'C8': 'one exact F1 target replay plus seven independent local perturbations and corrections',
        'teacher_steps': 288, 'anchor_optimizer_steps': 0, 'teacher_lr': .05,
        'anchor_initial_perturbation_rms': .03, 'anchor_projection_rms': .1,
        'proximal_lambda': .1, 'proximal_center': 'each perturbed initialization',
        'projection_center': 'the same per-preference F1 anchor',
        'fm_steps': 2048, 'effective_batch': 4, 'fm_lr': 5e-5,
        'cfg': 1, 'generation_steps': 28, 'target_entry': 'direct optimized model latent',
        'qualification': 'saved and reopened PNG; all T1/T2/T3 x four correct-option positions',
        'evaluation': 'V0/V1 x eight paired mt8-eval seeds; memory/mismatch/blank/text',
        'excluded_from_selection': ['dev90', 'O1', 'O2', 'official180', 'V2'],
        'no_test_time_target_selection': True})
    if (a.output/'complete.json').exists():
        done = json.loads((a.output/'complete.json').read_text())
        assert done['results_sha256'] == sha(a.output/'results.json')
        print('Round already complete; review results rather than restarting.', flush=True)
        return

    jobs('teachers-C8', [(gpu, [PYTHON, TEACHER, '--base', BASE, '--reader', READER,
        '--prefeval', PREFEVAL, '--starts', PRIOR/'starts', '--ids-file', ids_file,
        '--output', a.output/'teachers', '--arms', 'C8', '--anchor-bank', PRIOR/'F1-bank.json',
        '--anchor-init-rms', .03, '--anchor-trust-rms', .1, '--proximal-lambda', .1,
        '--steps', 288, '--targets', targets, '--shard', gpu, '--shards', 4]) for gpu in range(4)])
    c8 = freeze_bank(a.output, ids, 'C8')
    assert all(x['attempted'] == targets for x in c8['coverage'].values())
    s8 = json.loads((PRIOR/'S8-bank.json').read_text())
    assert set(s8['targets']) == set(registered_ids)
    s1 = fixed_single_bank(s8, 'S1')
    s1['targets'] = {pid: s1['targets'][pid] for pid in ids}
    s1['coverage'] = {pid: s1['coverage'][pid] for pid in ids}
    banks = {'S1': s1, 'C8': c8}
    for arm, bank in banks.items():
        immutable(a.output/f'{arm}-bank.json', bank)
    if not c8['ready']:
        status('teacher_repair_required', missing_preferences=c8['missing_preferences'])
        raise RuntimeError('C8 has zero-qualified preferences; retain them and repair in a separate version.')
    if a.smoke:
        status('smoke_teacher_complete', qualified_counts=c8['coverage'])
        return

    common = ['--arm', 'B', '--base', BASE, '--official-source', OFFICIAL, '--split', 'train',
        '--ids-file', ids_file, '--initial-variants', VARIANTS]
    jobs('fm', [(gpu, [PYTHON, WRITER, 'train', *common, '--checkpoint', PARENT,
        '--target-bank', a.output/f'{arm}-bank.json', '--stage', 'write', '--steps', 2048,
        '--output', a.output/arm/'train']) for gpu, arm in enumerate(banks)])
    for arm in banks:
        for variant in range(2):
            images = a.output/arm/f'eval-V{variant}'
            jobs(f'eval-generate-{arm}-V{variant}', [(gpu, [PYTHON, WRITER, 'rollout', *common,
                '--checkpoint', a.output/arm/'train/checkpoint-final.pt', '--initial-variant', variant,
                '--inter-turns', 0, '--noise-chains', 8, '--noise-domain', 'mt8-eval',
                '--shard-index', gpu, '--shard-count', 4, '--output', images]) for gpu in range(4)])
            manifests = [json.loads((images/f'manifest-shard-{gpu}.json').read_text()) for gpu in range(4)]
            assert all(x == manifests[0] for x in manifests)
            assert all((images/pid.replace(':', '_')/f'seed-{k}/complete.json').exists()
                       for pid in ids for k in range(8))
            immutable(images/'manifest.json', manifests[0])
            jobs(f'eval-read-{arm}-V{variant}', [(gpu, [PYTHON, EVAL, '--kind', 'student',
                '--split', 'train', '--ids-file', ids_file, '--reader', READER,
                '--initial-variants', VARIANTS, '--initial-variant', variant, '--images', images,
                '--output', a.output/arm/f'read-V{variant}', '--families', 'T1,T2,T3', '--tasks', 'mcq',
                '--prefixes', 0, '--noise-chains', 8, '--controls', 'memory,mismatch,blank,text',
                '--shard', gpu, '--shards', 4]) for gpu in range(4)])
    results = {}
    for arm, bank in banks.items():
        counts = defaultdict(lambda: [0, 0])
        per_image, per_preference = defaultdict(list), defaultdict(list)
        readback_hashes = {}
        parse_failures = 0
        for variant in range(2):
            seen = set()
            for shard in range(4):
                path = a.output/arm/f'read-V{variant}/readback-{shard}.jsonl'
                readback_hashes[str(path)] = sha(path)
                for line in path.read_text().splitlines():
                    r = json.loads(line)
                    key = tuple(r[k] for k in ['pair_id', 'chain', 'prefix', 'control', 'family', 'task'])
                    assert key not in seen and r['pair_id'] in ids
                    seen.add(key)
                    category = f"V{variant}/{r['control']}/{r['family']}"
                    counts[category][0] += int(r['correct'])
                    counts[category][1] += 1
                    parse_failures += int(r['parse_failure'])
                    if r['control'] == 'memory':
                        per_image[r['pair_id'], variant, r['chain']].append(r['correct'])
                        per_preference[r['pair_id']].append(int(r['correct']))
            assert len(seen) == len(ids)*54
        assert len(per_image) == len(ids)*16 and all(len(x) == 3 for x in per_image.values())
        for variant in range(2):
            for control in ['memory', 'mismatch', 'blank', 'text']:
                for family in ['T1', 'T2', 'T3']:
                    assert counts[f'V{variant}/{control}/{family}'][1] == len(ids)*(8 if control in ['memory', 'mismatch'] else 1)
        results[arm] = {'correct_total': dict(counts), 'images': len(per_image),
            'all_three_forms_correct': sum(all(x) for x in per_image.values()),
            'per_preference': {k: sum(v)/len(v) for k, v in per_preference.items()},
            'qualified_target_counts': {k: len(v) for k, v in bank['targets'].items()},
            'candidate_coverage': bank['coverage'], 'parse_failures': parse_failures,
            'readback_sha256': readback_hashes,
            'checkpoint_sha256': sha(a.output/arm/'train/checkpoint-final.pt')}
    save(a.output/'results.json', results)
    status('round2_complete_iteration_review_required', results=str(a.output/'results.json'))
    save(a.output/'complete.json', {'status': 'round2_complete', 'results_sha256': sha(a.output/'results.json'),
        'next': 'Compare S1/S8 and C8/F1/F8 on paired training preferences; inspect target diversity before scaling or PNG recurrence.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args())
