"""Fixed C8/B0 real-PNG recursion to ten distractor exchanges on train730."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('eval730_helpers', ROOT/'scripts/inspire/run_prefeval_c8_eval730.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
PROJECT, TASK, FROZEN, PYTHON = base.PROJECT, base.TASK, base.FROZEN, base.PYTHON
sha, save = base.sha, base.save
PREFIXES = [0, 5, 10]


def prepare(output, smoke):
    source = TASK/'eval730/fixed-c8-b0-20260928/protocol.json'
    assert sha(source) == 'c670e3db2d7d54bf449999cf693d7544fb54bbc7040420b641c31484e6f641dc'
    prior = json.loads(source.read_text())
    assert base.git_head(FROZEN) == base.FROZEN_COMMIT
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=FROZEN, text=True).strip()
    for path, expected in prior['data_sha256'].items():
        assert sha(path) == expected
    for path, expected in base.CHECKPOINTS.values():
        assert sha(path) == expected
    sys.path.insert(0, str(FROZEN))
    from scripts.experiments.prefeval_k1_data import load_records, event_text
    from scripts.experiments.prefeval_k1_variants import load_variants, apply_variant
    rows = load_records('train')
    assert [r['base_pair_id'] for r in rows] == prior['ids']
    assert all(len(r['history']) == 22 and [m['role'] for m in r['history']] == ['user', 'assistant']*11 for r in rows)
    ids = prior['pilot64_ids'][:2] if smoke else prior['ids']
    by_id = {r['base_pair_id']: r for r in rows}
    selected = [by_id[pid] for pid in ids]
    groups = defaultdict(list)
    for row in selected:
        groups[row['topic']].append(row['base_pair_id'])
    assert all(len(peers) >= 2 for peers in groups.values())
    donors = {pid: peers[(i+1) % len(peers)] for peers in groups.values() for i, pid in enumerate(peers)}
    variants = [0] if smoke else [0, 1]
    artifact = load_variants(base.VARIANTS, selected)
    # This is audit metadata only. The frozen Writer reads one current exchange itself.
    event_hashes = {}
    import hashlib
    for v in variants:
        event_hashes[f'V{v}'] = {}
        for row in selected:
            changed = apply_variant(row, artifact, v)
            event_hashes[f'V{v}'][row['base_pair_id']] = [hashlib.sha256(event_text(changed['history'][i*2:i*2+2]).encode()).hexdigest() for i in range(11)]
    chains = 1 if smoke else 2
    protocol = dict(prior, controller_commit=base.git_head(ROOT),
        scope='Fixed pilot64-adapted C8 versus B0; train730 real PNG recursion, not unseen-preference generalization.',
        source_protocol_sha256=sha(source), smoke=smoke, ids=ids, donors=donors,
        arms=['C8'] if smoke else ['C8', 'B0'], initial_variants=variants, noise_chains=chains,
        inter_turns=10, measured_prefixes=PREFIXES, logical_shards=2 if smoke else 4, physical_gpus=4,
        images_per_arm=len(ids)*len(variants)*chains*11,
        unique_readbacks_per_arm=len(ids)*len(variants)*(18*chains+12),
        current_exchange_sha256=event_hashes,
        state='Previous saved/reopened uint8 RGB PNG plus current user/assistant exchange only; fresh noise every write.',
        noise_budget='Two fixed chains (indices 0,1) per V0/V1; predeclared subset of the single-write eight-seed domain.',
        lock_policy='Independent long10 pipeline/controller and physical allocation locks; root serial pipeline is not acquired.',
        not_measured=['C8 trained on all730', 'unseen-preference generalization', 'retention-trained C8'],
        primary=['T1/all-three/matching gain at 0,5,10', 'paired 5/10-minus-0 change', 'retained correctness conditional on initially correct'])
    if smoke:
        protocol['scope'] = 'Two-preference real-PNG chain smoke; integrity gate only, never a method score gate.'
    save(output/'ids.json', {'ids': ids, 'scope': protocol['scope']}, immutable=True)
    save(output/'protocol.json', protocol, immutable=True)
    return protocol


def verify_png_chains(images, protocol, arm, variant):
    binding = json.loads((images/'manifest-shard-0.json').read_text())
    assert binding['checkpoint_sha256'] == base.CHECKPOINTS[arm][1]
    assert binding['noise_chains'] == protocol['noise_chains'] and binding['inter_turns'] == 10
    assert binding['initial_variant'] == variant and binding['noise_domain'] == 'mt8-eval'
    for shard in range(protocol['logical_shards']):
        assert json.loads((images/f'manifest-shard-{shard}.json').read_text()) == binding
    import hashlib
    for pid in protocol['ids']:
        for chain in range(protocol['noise_chains']):
            directory = images/pid.replace(':', '_')/f'seed-{chain}'
            complete = json.loads((directory/'complete.json').read_text())
            assert complete['binding'] == binding
            actual = {f'prefix-{i:02d}.png': sha(directory/f'prefix-{i:02d}.png') for i in range(11)}
            assert complete['png_hashes'] == actual
            writes = [json.loads(line) for line in (directory/'writes.jsonl').read_text().splitlines()]
            assert len(writes) == 11 and [r['position'] for r in writes] == list(range(11))
            for i, record in enumerate(writes):
                assert record['source_png_sha256'] == (actual[f'prefix-{i-1:02d}.png'] if i else None)
                assert record['output_png_sha256'] == actual[f'prefix-{i:02d}.png']
                assert hashlib.sha256(record['event'].encode()).hexdigest() == protocol['current_exchange_sha256'][f'V{variant}'][pid][i]
    save(images/'manifest.json', binding, immutable=True)
    return {'chains': len(protocol['ids'])*protocol['noise_chains'], 'pngs': len(protocol['ids'])*protocol['noise_chains']*11,
            'all_png_hashes_verified': True, 'all_10_previous_png_links_verified': True, 'current_exchange_only_verified': True}


def summarize(output, protocol):
    ids = protocol['ids']
    chains, variants = protocol['noise_chains'], protocol['initial_variants']
    samples = chains*len(variants)
    results, matrix = {}, {}
    for arm in protocol['arms']:
        counts = {pid: defaultdict(lambda: [0, 0]) for pid in ids}
        memory, files, parse_failures = {}, {}, 0
        for variant in variants:
            seen = set()
            for shard in range(protocol['logical_shards']):
                path = output/arm/f'read-V{variant}'/f'readback-{shard}.jsonl'
                files[str(path)] = sha(path)
                for line in path.read_text().splitlines():
                    r = json.loads(line)
                    pid, chain, prefix, control, family = [r[k] for k in ['pair_id', 'chain', 'prefix', 'control', 'family']]
                    key = (pid, chain, prefix, control, family, r['task'])
                    assert key not in seen and pid in ids[shard::protocol['logical_shards']]
                    assert prefix in PREFIXES and family in base.FAMILIES and control in base.CONTROLS and r['task'] == 'mcq'
                    assert chain in (range(chains) if control in ['memory', 'mismatch'] else [0])
                    assert control != 'blank' or prefix == 0
                    if control == 'mismatch':
                        assert r['donor_pair_id'] == protocol['donors'][pid]
                    seen.add(key)
                    metric = f'V{variant}/{prefix}/{control}/{family}'
                    counts[pid][metric][0] += int(r['correct'])
                    counts[pid][metric][1] += 1
                    parse_failures += int(r['parse_failure'])
                    if control == 'memory':
                        memory[pid, variant, chain, prefix, family] = bool(r['correct'])
            assert len(seen) == len(ids)*(18*chains+12)
        per_pref = {}
        for pid in ids:
            item = {'correct_total': dict(counts[pid]), 'prefixes': {}}
            for prefix in PREFIXES:
                for variant in variants:
                    for control in base.CONTROLS:
                        if control == 'blank' and prefix != 0:
                            continue
                        for family in base.FAMILIES:
                            assert counts[pid][f'V{variant}/{prefix}/{control}/{family}'][1] == (chains if control in ['memory', 'mismatch'] else 1)
                t1 = sum(memory[pid, v, c, prefix, 'T1'] for v in variants for c in range(chains))
                mismatch = sum(counts[pid][f'V{v}/{prefix}/mismatch/T1'][0] for v in variants)
                all_three = sum(all(memory[pid, v, c, prefix, f] for f in base.FAMILIES) for v in variants for c in range(chains))
                initially_correct = sum(memory[pid, v, c, 0, 'T1'] for v in variants for c in range(chains))
                retained = sum(memory[pid, v, c, 0, 'T1'] and memory[pid, v, c, prefix, 'T1'] for v in variants for c in range(chains))
                item['prefixes'][str(prefix)] = {'T1': [t1, samples], 'all_three': [all_three, samples],
                    'mismatch_T1': [mismatch, samples], 'matching_gain_pp': 100*(t1-mismatch)/samples,
                    'retained_T1_given_initially_correct': [retained, initially_correct]}
            per_pref[pid] = item
        scopes = [('all730', ids), ('adapted64', [p for p in ids if p in protocol['pilot64_ids']]), ('other666', [p for p in ids if p not in protocol['pilot64_ids']])]
        scopes += [(f'topic/{t}', [p for p in ids if p.split(':')[0] == t]) for t in sorted({p.split(':')[0] for p in ids})]
        strata = {}
        for label, subset in scopes:
            if not subset:
                continue
            strata[label] = {}
            for prefix in PREFIXES:
                values = [per_pref[p]['prefixes'][str(prefix)] for p in subset]
                total = {k: [sum(v[k][i] for v in values) for i in range(2)] for k in ['T1', 'all_three', 'mismatch_T1', 'retained_T1_given_initially_correct']}
                total['matching_gain_pp'] = 100*(total['T1'][0]-total['mismatch_T1'][0])/total['T1'][1]
                total['all_samples_T1_correct_preferences'] = sum(v['T1'][0] == samples for v in values)
                total['all_samples_three_forms_correct_preferences'] = sum(v['all_three'][0] == samples for v in values)
                strata[label][str(prefix)] = total
        results[arm] = {'strata': strata, 'per_preference': per_pref, 'readback_sha256': files, 'parse_failures': parse_failures,
                        'checkpoint_sha256': base.CHECKPOINTS[arm][1]}
        matrix[arm] = per_pref
    import numpy as np
    comparisons = {}
    for label, subset in [('all730', ids), ('adapted64', [p for p in ids if p in protocol['pilot64_ids']]), ('other666', [p for p in ids if p not in protocol['pilot64_ids']])]:
        if not subset:
            continue
        contrasts = {}
        def metrics(arm, pid, prefix):
            v = matrix[arm][pid]['prefixes'][str(prefix)]
            return np.array([v['T1'][0]/samples, v['all_three'][0]/samples, v['matching_gain_pp']/100])
        for arm in protocol['arms']:
            for depth in [5, 10]:
                contrasts[f'{arm}/{depth}-minus-0'] = np.array([metrics(arm, p, depth)-metrics(arm, p, 0) for p in subset])
        if 'B0' in matrix:
            for prefix in PREFIXES:
                contrasts[f'C8-minus-B0/{prefix}'] = np.array([metrics('C8', p, prefix)-metrics('B0', p, prefix) for p in subset])
        comparisons[label] = {}
        for name, values in contrasts.items():
            rng = np.random.default_rng(20260928)
            bootstrap = np.array([values[rng.integers(len(values), size=len(values))].mean(axis=0) for _ in range(10000)])
            comparisons[label][name] = {metric: {'difference_pp': float(values.mean(axis=0)[i]*100),
                'paired_preference_bootstrap_95ci_pp': (np.percentile(bootstrap[:, i], [2.5, 97.5])*100).tolist()}
                for i, metric in enumerate(['T1', 'all_three', 'matching_gain'])}
    return {'scope': protocol['scope'], 'arms': results, 'comparisons': comparisons, 'blank_control': 'Only read at prefix0; identical fixed gray image for 5/10.', 'smoke': protocol['smoke']}


def main(args):
    output = args.output.resolve()
    assert output.parent == (TASK/'long10').resolve()
    if args.prepare_only:
        p = prepare(output, args.smoke)
        print(json.dumps({'status': 'prepared_not_launched', 'smoke': args.smoke, 'images_per_arm': p['images_per_arm'], 'readbacks_per_arm': p['unique_readbacks_per_arm']}))
        return
    os.fstat(9)  # Independent long10 pipeline lock, not the active serial root lock.
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    allocation_dir = TASK/'allocation-locks'
    allocation_dir.mkdir(exist_ok=True)
    allocation = (allocation_dir/f'{os.uname().nodename}.lock').open('a+')
    fcntl.flock(allocation, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'host': os.uname().nodename, 'pid': os.getpid(), 'commit': base.git_head(ROOT), 'code': str(ROOT)}
    def status(stage, **extra):
        save(output/'controller.json', dict(state, stage=stage, time_utc=datetime.now(timezone.utc).isoformat(), **extra))
    def jobs(label, commands):
        for gpu in range(4):
            busy = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            assert not busy, f'GPU {gpu} occupied: {busy}'
        status(label)
        children = []
        for shard, command in commands:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            stream = (output/f'{label}-shard{shard}.log').open('a')
            command = list(map(str, command))
            child = subprocess.Popen(command, cwd=FROZEN, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, pass_fds=(9, lock.fileno(), allocation.fileno()))
            children.append((child, stream))
            save(output/f'{label}-shard{shard}-process.json', dict(state, gpu=shard, pid=child.pid, command=command))
        codes = []
        for child, stream in children:
            codes.append(child.wait()); stream.close()
        assert not any(codes), f'{label}: {codes}'
    try:
        status('preflight')
        protocol = prepare(output, args.smoke)
        if (output/'complete.json').exists():
            assert json.loads((output/'complete.json').read_text())['results_sha256'] == sha(output/'results.json')
            return
        if not args.smoke:
            smoke = TASK/'long10/smoke'
            done = json.loads((smoke/'complete.json').read_text())
            assert done['results_sha256'] == sha(smoke/'results.json')
            assert done['status'] == 'long10_integrity_smoke_complete'
        writer = FROZEN/'scripts/experiments/prefeval_k1_writer.py'
        evaluator = FROZEN/'scripts/experiments/prefeval_k1_evaluate.py'
        common = ['--arm', 'B', '--base', base.MODELS/'DreamLite-base-a9a0f15-20260907',
            '--official-source', PROJECT/'Vision-Language-Memory/third_party/DreamLite', '--split', 'train',
            '--ids-file', output/'ids.json', '--initial-variants', base.VARIANTS]
        audit = {}
        for arm in protocol['arms']:
            for variant in protocol['initial_variants']:
                images = output/arm/f'eval-V{variant}'
                jobs(f'generate-{arm}-V{variant}', [(s, [PYTHON, writer, 'rollout', *common,
                    '--checkpoint', base.CHECKPOINTS[arm][0], '--initial-variant', variant, '--inter-turns', 10,
                    '--noise-chains', protocol['noise_chains'], '--noise-domain', 'mt8-eval',
                    '--shard-index', s, '--shard-count', protocol['logical_shards'], '--output', images]) for s in range(protocol['logical_shards'])])
                audit[f'{arm}/V{variant}'] = verify_png_chains(images, protocol, arm, variant)
                save(output/'png-chain-audit.json', audit)
                jobs(f'read-{arm}-V{variant}', [(s, [PYTHON, evaluator, '--kind', 'student', '--split', 'train',
                    '--ids-file', output/'ids.json', '--reader', base.MODELS/'Qwen3-VL-4B-Instruct',
                    '--initial-variants', base.VARIANTS, '--initial-variant', variant, '--images', images,
                    '--output', output/arm/f'read-V{variant}', '--families', 'T1,T2,T3', '--tasks', 'mcq',
                    '--prefixes', '0,5,10', '--noise-chains', protocol['noise_chains'],
                    '--controls', 'memory,mismatch,blank,text', '--shard', s, '--shards', protocol['logical_shards']]) for s in range(protocol['logical_shards'])])
        save(output/'results.json', summarize(output, protocol))
        status('smoke_integrity_complete' if args.smoke else 'complete_analysis_required')
        save(output/'complete.json', {'status': 'long10_integrity_smoke_complete' if args.smoke else 'fixed_C8_B0_long10_complete',
            'results_sha256': sha(output/'results.json'), 'protocol_sha256': sha(output/'protocol.json')})
    except Exception as error:
        save(output/'failure.json', dict(state, time_utc=datetime.now(timezone.utc).isoformat(), error=repr(error)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--smoke', action='store_true')
    main(p.parse_args())
