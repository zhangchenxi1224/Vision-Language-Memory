"""New C8 student snapshot correction plus replay, versus equal-budget old replay."""
import argparse
import copy
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
    save, immutable, sha)
from scripts.experiments.prefeval_multitarget_bank import freeze_bank

PRIOR = PROJECT/'runs/prefeval-multitarget-20260927/pilot64'
ROUND2 = PROJECT/'runs/prefeval-multitarget-20260927/round2/pilot64'
EVALUATED = PROJECT/'runs/prefeval-multitarget-20260927/round2/recovery-20260928-0045'
FROZEN = PROJECT/'repos/prefeval-multitarget-round2'
WRITER = FROZEN/'scripts/experiments/prefeval_k1_writer.py'
EVAL = FROZEN/'scripts/experiments/prefeval_k1_evaluate.py'
TEACHER = FROZEN/'scripts/experiments/prefeval_multitarget_teacher.py'
PARENT = ROUND2/'C8/train/checkpoint-final.pt'
PARENT_SHA = 'efd7b4a22a40508e9ae5a9d7ca72305f5a1c8b3130b3e22c5e2b72aa7bb36978'


def main(a):
    os.fstat(9)  # The launcher owns the root pipeline lock; children inherit it too.
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
        for gpu in range(2):
            busy = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            if busy:
                raise RuntimeError(f'GPU {gpu} occupied; refusing overlap: {busy}')
        for offset in range(0, len(commands), 2):
            children = []
            for shard, command in commands[offset:offset+2]:
                gpu = shard % 2
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                    PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0',
                    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
                stream = (a.output/f'{label}-shard{shard}.log').open('a')
                command = list(map(str, command))
                child = subprocess.Popen(command, cwd=FROZEN, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, pass_fds=(9, lock.fileno()))
                children.append((child, stream))
                save(a.output/f'{label}-shard{shard}-process.json', dict(state, gpu=gpu, shard=shard, pid=child.pid, command=command))
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
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=FROZEN,text=True).strip() == '224cc77d790cf3967b5a56ce2e77c364959435a2'
    assert json.loads((EVALUATED/'complete.json').read_text())['results_sha256'] == sha(EVALUATED/'results.json')
    assert json.loads((EVALUATED/'results.json').read_text())['C8']['checkpoint_sha256'] == PARENT_SHA
    assert sha(PARENT) == PARENT_SHA
    registered_ids = json.loads((PRIOR/'ids.json').read_text())['ids']
    assert len(registered_ids) == 64 and registered_ids == prior_protocol['ids']
    ids = registered_ids[:2] if a.smoke else registered_ids
    ids_file = a.output/'ids.json'
    immutable(ids_file, {'ids': ids, 'scope': 'same previously trained pilot64; no new-preference generalization'})
    targets = 2 if a.smoke else 8
    immutable(a.output/'protocol.json', {
        'commit': commit, 'prior_results_sha256': sha(PRIOR/'results.json'),
        'parent': str(PARENT), 'parent_sha256': PARENT_SHA,
        'evaluated_round2_sha256': sha(EVALUATED/'results.json'),
        'replay_bank_sha256': sha(ROUND2/'C8-bank.json'),
        'ids': ids, 'arms': ['C8R', 'U16'], 'smoke': a.smoke, 'new_targets': targets,
        'C8R': 'all old qualified C8 targets replayed; equal additional FM budget',
        'U16': 'uniform union within preference: all old qualified C8 targets plus up to eight new corrected student targets',
        'teacher_steps': 288, 'replay_optimizer_steps': 0, 'teacher_lr': .05,
        'student_snapshot': 'completed C8; V0/V1 x four mt8-teacher seeds; paired to first-round teacher seeds',
        'proximal_lambda': .1, 'proximal_center': 'each actual C8 student latent',
        'projection_center': 'each actual C8 student latent', 'projection_rms': .1,
        'new_target_cost': '512 new optimized candidates; all 511 old qualified targets reused without optimization',
        'combined_target_count': 'actual old K_i plus actual new K_i; at most 16, not 16 proven semantic modes',
        'frozen_worker_commit': '224cc77d790cf3967b5a56ce2e77c364959435a2',
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

    common = ['--arm', 'B', '--base', BASE, '--official-source', OFFICIAL, '--split', 'train',
        '--ids-file', ids_file, '--initial-variants', VARIANTS]
    for variant in range(2):
        jobs(f'starts-V{variant}', [(gpu, [PYTHON, WRITER, 'rollout', *common,
            '--checkpoint', PARENT, '--initial-variant', variant, '--inter-turns', 0,
            '--noise-chains', targets//2, '--noise-domain', 'mt8-teacher',
            '--shard-index', gpu, '--shard-count', 2, '--output', a.output/'starts'/f'V{variant}']) for gpu in range(2)])
    jobs('teachers-new', [(gpu, [PYTHON, TEACHER, '--base', BASE, '--reader', READER,
        '--prefeval', PREFEVAL, '--starts', a.output/'starts', '--ids-file', ids_file,
        '--output', a.output/'teachers', '--arms', 'S8', '--trust-rms', .1, '--proximal-lambda', .1,
        '--steps', 288, '--targets', targets, '--shard', gpu, '--shards', 2]) for gpu in range(2)])
    fresh = freeze_bank(a.output, ids, 'S8')
    immutable(a.output/'fresh-bank.json', fresh)
    assert all(x['attempted'] == targets for x in fresh['coverage'].values())
    if not fresh['ready']:
        status('teacher_repair_required', missing_preferences=fresh['missing_preferences'])
        raise RuntimeError('Zero qualified NEW targets; retain preferences and repair in a separate version.')
    replay = json.loads((ROUND2/'C8-bank.json').read_text())
    assert replay['ready'] and set(replay['targets']) == set(registered_ids)
    replay['arm'] = 'C8R'
    replay['targets'] = {pid: replay['targets'][pid] for pid in ids}
    replay['coverage'] = {pid: replay['coverage'][pid] for pid in ids}
    union = copy.deepcopy(replay)
    union['arm'] = 'U16'
    union['old_coverage'], union['new_coverage'] = replay['coverage'], fresh['coverage']
    for pid in ids:
        union['targets'][pid] = [dict(x, origin='old_C8') for x in replay['targets'][pid]] + [dict(x, origin='new_C8_student') for x in fresh['targets'][pid]]
        union['coverage'][pid] = {k: replay['coverage'][pid][k] + fresh['coverage'][pid][k] for k in ['attempted','qualified']}
    banks = {'C8R': replay, 'U16': union}
    for arm, bank in banks.items():
        immutable(a.output/f'{arm}-bank.json', bank)
    if a.smoke:
        status('smoke_teacher_complete', qualified_counts=fresh['coverage'])
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
    status('round3_complete_iteration_review_required', results=str(a.output/'results.json'))
    save(a.output/'complete.json', {'status': 'round3_complete', 'results_sha256': sha(a.output/'results.json'),
        'next': 'Compare U16 with C8R and original C8 on paired train64; then scale reliable local targets to 730 and real PNG K1 retention.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args())
