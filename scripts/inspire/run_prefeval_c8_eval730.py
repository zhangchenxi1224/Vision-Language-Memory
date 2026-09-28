"""Fixed C8 and B0 evaluation on train730; no teacher optimization or FM training."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
TASK = PROJECT/'runs/prefeval-multitarget-20260927'
FROZEN = PROJECT/'repos/prefeval-multitarget-round2'
FROZEN_COMMIT = '224cc77d790cf3967b5a56ce2e77c364959435a2'
PYTHON = PROJECT/'envs/vlm-r3-ngc2502/bin/python'
MODELS = Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')
VARIANTS = PROJECT/'runs/prefeval-b-mcq-20260925/variants-train.json'
CHECKPOINTS = {
    'C8': (TASK/'round2/pilot64/C8/train/checkpoint-final.pt',
           'efd7b4a22a40508e9ae5a9d7ca72305f5a1c8b3130b3e22c5e2b72aa7bb36978'),
    'B0': (PROJECT/'runs/prefeval-b-mcq-20260925/robust730/train/checkpoint-final.pt',
           '6c8eb92f629ccdf6932407124190d651e8231058ef6f232a0a59380370d44c8d'),
}
FAMILIES = ['T1', 'T2', 'T3']
CONTROLS = ['memory', 'mismatch', 'blank', 'text']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value, immutable=False):
    if immutable and path.exists():
        assert json.loads(path.read_text()) == value, f'Changed binding: {path}'
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def git_head(path):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()


def prepare(output):
    assert git_head(FROZEN) == FROZEN_COMMIT
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=FROZEN, text=True).strip()
    sys.path.insert(0, str(FROZEN))
    from scripts.experiments.prefeval_k1_data import load_records, ALIGN, REPORT
    from scripts.experiments.prefeval_k1_variants import load_variants, apply_variant
    rows = load_records('train')
    ids = [r['base_pair_id'] for r in rows]
    pilot = json.loads((TASK/'pilot64/ids.json').read_text())['ids']
    assert len(ids) == len(set(ids)) == 730 and len(set(pilot)) == 64
    assert set(pilot) < set(ids)
    assert all(all(k in r['forms'] for k in FAMILIES) for r in rows)
    variants = load_variants(VARIANTS, rows)
    for variant in range(2):
        assert all(apply_variant(r, variants, variant)['history'][0] == r['history'][0] for r in rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row['topic']].append(row['base_pair_id'])
    assert all(len(v) > 1 for v in groups.values())
    donors = {pid: peers[(i+1) % len(peers)] for peers in groups.values() for i, pid in enumerate(peers)}
    for checkpoint, expected in CHECKPOINTS.values():
        assert sha(checkpoint) == expected, f'Checkpoint changed: {checkpoint}'
    upstream_done = TASK/'round2/recovery-20260928-0045/complete.json'
    upstream_results = upstream_done.with_name('results.json')
    assert json.loads(upstream_done.read_text())['results_sha256'] == sha(upstream_results)
    assert json.loads(upstream_results.read_text())['C8']['checkpoint_sha256'] == CHECKPOINTS['C8'][1]
    data = [ALIGN/'WRITER_IMPLEMENTATION_SPLIT.json', ALIGN/'data/sft-train-10interturn.jsonl.gz',
            ALIGN/'data/benchmark-disclosures.jsonl.gz', REPORT/'train-question-forms.json', VARIANTS]
    protocol = {
        'controller_commit': git_head(ROOT), 'worker_commit': FROZEN_COMMIT,
        'scope': 'Fixed pilot64-adapted C8 versus its B0 parent on train730; all 730 were in B0 training.',
        'ids': ids, 'pilot64_ids': pilot, 'other666_ids': [p for p in ids if p not in set(pilot)],
        'topics': dict(Counter(r['topic'] for r in rows)), 'donors': donors,
        'checkpoints': {arm: {'path': str(p), 'sha256': h} for arm, (p, h) in CHECKPOINTS.items()},
        'data_sha256': {str(p): sha(p) for p in data},
        'initial_variants': [0, 1], 'noise_chains': 8, 'noise_domain': 'mt8-eval',
        'generation_steps': 28, 'cfg': 1, 'inter_turns': 0,
        'families': FAMILIES, 'controls': CONTROLS, 'logical_shards': 4, 'physical_gpus': 2,
        'images_per_arm': 11680, 'unique_readbacks_per_arm': 78840,
        'teacher_updates': 0, 'fm_updates': 0, 'best_of_n': False,
        'report_strata': ['all730', 'adapted64', 'other666'],
        'primary': ['T1 single-generation accuracy', 'same-image T1/T2/T3 all correct', 'T1 memory minus mismatch'],
        'not_measured': ['C8 trained on all730', 'unseen-preference generalization', 'recursive K1 retention'],
        'excluded': ['dev90', 'official holdout', 'V2', 'O1', 'O2'],
    }
    save(output/'ids.json', {'ids': ids, 'scope': protocol['scope']}, immutable=True)
    save(output/'protocol.json', protocol, immutable=True)
    return protocol


def summarize(output, protocol):
    ids, pilot = protocol['ids'], set(protocol['pilot64_ids'])
    results = {}
    for arm in CHECKPOINTS:
        per_pref = {pid: {'correct_total': defaultdict(lambda: [0, 0]), 'all_three': [0, 0]} for pid in ids}
        hashes, parse_failures = {}, 0
        for variant in range(2):
            seen, images = set(), defaultdict(dict)
            for shard in range(4):
                path = output/arm/f'read-V{variant}'/f'readback-{shard}.jsonl'
                hashes[str(path)] = sha(path)
                for line in path.read_text().splitlines():
                    r = json.loads(line)
                    pid, chain, control, family = (r[k] for k in ['pair_id', 'chain', 'control', 'family'])
                    key = (pid, chain, r['prefix'], control, family, r['task'])
                    assert key not in seen and pid in ids[shard::4]
                    assert r['prefix'] == 0 and r['task'] == 'mcq' and family in FAMILIES and control in CONTROLS
                    assert chain in (range(8) if control in ['memory', 'mismatch'] else [0])
                    if control == 'mismatch':
                        assert r['donor_pair_id'] == protocol['donors'][pid]
                    seen.add(key)
                    category = f'V{variant}/{control}/{family}'
                    counts = per_pref[pid]['correct_total'][category]
                    counts[0] += int(r['correct'])
                    counts[1] += 1
                    parse_failures += int(r['parse_failure'])
                    if control == 'memory':
                        images[pid, chain][family] = bool(r['correct'])
            assert len(seen) == len(ids)*54
            assert len(images) == len(ids)*8
            for (pid, chain), values in images.items():
                assert set(values) == set(FAMILIES)
                per_pref[pid]['all_three'][0] += int(all(values.values()))
                per_pref[pid]['all_three'][1] += 1
        for item in per_pref.values():
            assert item['all_three'][1] == 16
            assert len(item['correct_total']) == 24
            for category, counts in item['correct_total'].items():
                assert counts[1] == (8 if category.split('/')[1] in ['memory', 'mismatch'] else 1)
        strata = {}
        scopes = [('all730', ids), ('adapted64', [p for p in ids if p in pilot]), ('other666', [p for p in ids if p not in pilot])]
        scopes += [(f'topic/{topic}', [p for p in ids if p.split(':')[0] == topic]) for topic in sorted({p.split(':')[0] for p in ids})]
        for label, subset in scopes:
            totals = defaultdict(lambda: [0, 0])
            all_three = [0, 0]
            for pid in subset:
                item = per_pref[pid]
                for key, values in item['correct_total'].items():
                    for i in range(2):
                        totals[key][i] += values[i]
                for i in range(2):
                    all_three[i] += item['all_three'][i]
            t1_per_pref = [sum(per_pref[p]['correct_total'][f'V{v}/memory/T1'][0] for v in range(2)) for p in subset]
            strata[label] = {'preferences': len(subset), 'all_three': all_three, 'correct_total': dict(totals),
                'all16_T1_correct_preferences': sum(v == 16 for v in t1_per_pref),
                'all16_three_forms_correct_preferences': sum(per_pref[p]['all_three'][0] == 16 for p in subset),
                'all16_T1_incorrect_preferences': sum(v == 0 for v in t1_per_pref)}
        results[arm] = {'strata': strata, 'per_preference': per_pref, 'parse_failures': parse_failures,
                        'readback_sha256': hashes, 'checkpoint_sha256': CHECKPOINTS[arm][1]}
    # Paired intervals resample preferences, keeping all variants and seeds together.
    import numpy as np
    comparisons = {}
    for label, subset in [('all730', ids), ('adapted64', [p for p in ids if p in pilot]), ('other666', [p for p in ids if p not in pilot])]:
        deltas = []
        for pid in subset:
            metrics = {}
            for arm in CHECKPOINTS:
                r = results[arm]['per_preference'][pid]
                memory = sum(r['correct_total'][f'V{v}/memory/T1'][0] for v in range(2))/16
                mismatch = sum(r['correct_total'][f'V{v}/mismatch/T1'][0] for v in range(2))/16
                metrics[arm] = np.array([memory, r['all_three'][0]/16, memory-mismatch])
            deltas.append(metrics['C8']-metrics['B0'])
        values = np.asarray(deltas)
        rng = np.random.default_rng(20260928)
        bootstrap = np.array([values[rng.integers(len(values), size=len(values))].mean(axis=0) for _ in range(10000)])
        comparisons[label] = {name: {'difference_pp': float(values.mean(axis=0)[i]*100),
            'paired_preference_bootstrap_95ci_pp': (np.percentile(bootstrap[:, i], [2.5, 97.5])*100).tolist()}
            for i, name in enumerate(['T1', 'all_three', 'T1_matching_gain'])}
    return {'arms': results, 'C8_minus_B0': comparisons, 'bootstrap_seed': 20260928,
            'bootstrap_replicates': 10000, 'scope': protocol['scope']}


def main(args):
    output = args.output.resolve()
    assert output.parent == (TASK/'eval730').resolve(), 'Use an independent directory under this task/eval730.'
    if args.prepare_only:
        p = prepare(output)
        print(json.dumps({'status': 'prepared_not_launched', 'preferences': len(p['ids']), 'images_per_arm': p['images_per_arm']}))
        return
    os.fstat(9)  # Root lock belongs to the launcher and is inherited by every worker.
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'host': os.uname().nodename, 'pid': os.getpid(), 'commit': git_head(ROOT), 'code': str(ROOT)}

    def status(stage, **extra):
        save(output/'controller.json', dict(state, stage=stage, time_utc=datetime.now(timezone.utc).isoformat(), **extra))

    def jobs(label, commands):
        status(label)
        for offset in range(0, len(commands), 2):
            for gpu in range(2):
                busy = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
                assert not busy, f'GPU {gpu} occupied; refusing overlap: {busy}'
            children = []
            for shard, command in commands[offset:offset+2]:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard % 2), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                    PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0',
                    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
                stream = (output/f'{label}-shard{shard}.log').open('a')
                command = list(map(str, command))
                child = subprocess.Popen(command, cwd=FROZEN, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, pass_fds=(9, lock.fileno()))
                children.append((child, stream))
                save(output/f'{label}-shard{shard}-process.json', dict(state, gpu=shard % 2, shard=shard, pid=child.pid, command=command))
            codes = []
            for child, stream in children:
                codes.append(child.wait())
                stream.close()
            assert not any(codes), f'{label} exit codes: {codes}'

    try:
        protocol = prepare(output)
        done = output/'complete.json'
        if done.exists():
            assert json.loads(done.read_text())['results_sha256'] == sha(output/'results.json')
            return
        writer = FROZEN/'scripts/experiments/prefeval_k1_writer.py'
        evaluator = FROZEN/'scripts/experiments/prefeval_k1_evaluate.py'
        common = ['--arm', 'B', '--base', MODELS/'DreamLite-base-a9a0f15-20260907',
            '--official-source', PROJECT/'Vision-Language-Memory/third_party/DreamLite',
            '--split', 'train', '--ids-file', output/'ids.json', '--initial-variants', VARIANTS]
        for arm, (checkpoint, expected) in CHECKPOINTS.items():
            for variant in range(2):
                images = output/arm/f'eval-V{variant}'
                jobs(f'generate-{arm}-V{variant}', [(s, [PYTHON, writer, 'rollout', *common,
                    '--checkpoint', checkpoint, '--initial-variant', variant, '--inter-turns', 0,
                    '--noise-chains', 8, '--noise-domain', 'mt8-eval', '--shard-index', s,
                    '--shard-count', 4, '--output', images]) for s in range(4)])
                manifests = [json.loads((images/f'manifest-shard-{s}.json').read_text()) for s in range(4)]
                assert all(m == manifests[0] for m in manifests)
                assert manifests[0]['checkpoint_sha256'] == expected
                for pid in protocol['ids']:
                    for chain in range(8):
                        folder = images/pid.replace(':', '_')/f'seed-{chain}'
                        receipt = json.loads((folder/'complete.json').read_text())
                        assert receipt['binding'] == manifests[0]
                        assert receipt['png_hashes'] == {'prefix-00.png': sha(folder/'prefix-00.png')}
                save(images/'manifest.json', manifests[0], immutable=True)
                jobs(f'read-{arm}-V{variant}', [(s, [PYTHON, evaluator, '--kind', 'student',
                    '--split', 'train', '--ids-file', output/'ids.json', '--reader', MODELS/'Qwen3-VL-4B-Instruct',
                    '--initial-variants', VARIANTS, '--initial-variant', variant, '--images', images,
                    '--output', output/arm/f'read-V{variant}', '--families', 'T1,T2,T3', '--tasks', 'mcq',
                    '--prefixes', 0, '--noise-chains', 8, '--controls', 'memory,mismatch,blank,text',
                    '--shard', s, '--shards', 4]) for s in range(4)])
        save(output/'results.json', summarize(output, protocol))
        status('complete_analysis_required')
        save(output/'complete.json', {'status': 'fixed_C8_B0_train730_evaluation_complete',
            'results_sha256': sha(output/'results.json'), 'protocol_sha256': sha(output/'protocol.json')})
    except Exception as error:
        save(output/'failure.json', dict(state, time_utc=datetime.now(timezone.utc).isoformat(), error=repr(error)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    main(p.parse_args())
