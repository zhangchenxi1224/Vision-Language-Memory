"""Audit the complete prespecified C8/V0 long10 slice without changing the run."""
import copy
import hashlib
import importlib.util
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
TASK = PROJECT/'runs/prefeval-multitarget-20260927'
SOURCE = TASK/'long10/recovery-20260929-0005-2gpu'
OUTPUT = TASK/'long10/analysis-c8-v0-20260929-1330'
FROZEN = PROJECT/'repos/prefeval-multitarget-round2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not OUTPUT.exists()
    protocol = json.loads((SOURCE/'protocol.json').read_text())
    assert sha(SOURCE/'protocol.json') == 'b43fd72a86e8a50547b359ac1ad7b0bf97bcdcbc90cb542f884206980caaae33'
    code = PROJECT/'repos/prefeval-c8-long10-20260928/scripts/inspire/run_prefeval_c8_long10.py'
    assert sha(code) == 'c5a9051c1d8426f45e017b69c960188c0eacf2c2dc507a782fc264a63783e4ce'
    spec = importlib.util.spec_from_file_location('frozen_long10', code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.base.git_head(FROZEN) == '224cc77d790cf3967b5a56ce2e77c364959435a2'
    checkpoint = module.base.CHECKPOINTS['C8']
    assert sha(checkpoint[0]) == checkpoint[1]
    for path, expected_hash in protocol['data_sha256'].items():
        assert sha(path) == expected_hash
    sys.path.insert(0, str(TASK/'long10/audit-deps-20260929'))
    sys.path[:0] = [str(FROZEN), str(FROZEN/'src')]
    from scripts.experiments.prefeval_k1_data import load_records, event_text, official_mcq, option_order
    from scripts.experiments.prefeval_k1_variants import load_variants, apply_variant
    from scripts.experiments.prefeval_k1_source_bank import noise_namespace
    from vision_memory.training.latent_bank_unet import stable_seed
    rows = load_records('train')
    variants = load_variants(module.base.VARIANTS, rows)
    rowmap = {r['base_pair_id']: apply_variant(r, variants, 0) for r in rows}
    ids = protocol['ids']
    assert len(ids) == 730 and protocol['noise_chains'] == 2
    images = SOURCE/'C8/eval-V0'
    binding = json.loads((images/'manifest.json').read_text())
    assert binding['checkpoint_sha256'] == checkpoint[1] and binding['noise_chains'] == 2
    assert binding['inter_turns'] == 10 and binding['initial_variant'] == 0
    assert binding['noise_domain'] == 'mt8-eval' and binding['initial_variants_sha256'] == sha(module.base.VARIANTS)
    pngs, evidence = {}, [SOURCE/'protocol.json', SOURCE/'ids.json', images/'manifest.json']
    for s in range(4):
        f = images/f'manifest-shard-{s}.json'
        assert json.loads(f.read_text()) == binding
        evidence.append(f)
    for pid in ids:
        for c in range(2):
            directory = images/pid.replace(':', '_')/f'seed-{c}'
            done = json.loads((directory/'complete.json').read_text())
            assert done['binding'] == binding
            actual = {f'prefix-{i:02d}.png': sha(directory/f'prefix-{i:02d}.png') for i in range(11)}
            assert done['png_hashes'] == actual
            writes = [json.loads(line) for line in (directory/'writes.jsonl').read_text().splitlines()]
            assert len(writes) == 11
            for i, w in enumerate(writes):
                assert w['position'] == i and w['output_png_sha256'] == actual[f'prefix-{i:02d}.png']
                assert w['source_png_sha256'] == (actual[f'prefix-{i-1:02d}.png'] if i else None)
                assert w['event'] == event_text(rowmap[pid]['history'][2*i:2*i+2])
                assert hashlib.sha256(w['event'].encode()).hexdigest() == protocol['current_exchange_sha256']['V0'][pid][i]
                assert w['noise_seed'] == stable_seed(20260924, noise_namespace(pid, c, 'mt8-eval'), i)
                pngs[str(directory/f'prefix-{i:02d}.png')] = actual[f'prefix-{i:02d}.png']
            evidence += [directory/'complete.json', directory/'writes.jsonl']
    assert len(pngs) == 16060
    print(json.dumps({'verified_pngs': len(pngs), 'chains': 1460}), flush=True)
    mcq = official_mcq(FROZEN/'third_party/prefeval_reference')
    seen, readhashes, receipts = set(), {}, {}
    for s in range(4):
        read = SOURCE/'C8/read-V0'/f'readback-{s}.jsonl'
        done = read.with_name(f'finished-{s}.json')
        readhashes[str(read)] = sha(read)
        receipts[str(s)] = json.loads(done.read_text())
        lines = read.read_text().splitlines()
        assert len(lines) == len(ids[s::4])*48 == receipts[str(s)]['items']
        for line in lines:
            r = json.loads(line)
            pid, c, depth, control, family, task = [r[k] for k in ['pair_id', 'chain', 'prefix', 'control', 'family', 'task']]
            key = (pid, c, depth, control, family, task)
            assert key not in seen and pid in ids[s::4]
            seen.add(key)
            step = int.from_bytes(hashlib.sha256(f'eval:{pid}:{family}'.encode()).digest()[:4], 'big')
            order, correct = option_order(pid, step)
            predicted = mcq['extract_choice'](r['generated']['raw'])
            assert r['option_order'] == order and r['correct_letter'] == 'ABCD'[correct]
            assert r['predicted_letter'] == predicted and r['correct'] == (predicted == 'ABCD'[correct])
            assert r['parse_failure'] == (predicted is None)
            assert r['question'] == rowmap[pid]['forms'][family]
            if control in ['memory', 'mismatch']:
                donor = pid if control == 'memory' else protocol['donors'][pid]
                expected = images/donor.replace(':', '_')/f'seed-{c}'/f'prefix-{depth:02d}.png'
                assert r['png_path'] == str(expected) and r['png_sha256'] == pngs[str(expected)]
                if control == 'mismatch':
                    assert r['donor_pair_id'] == donor
            else:
                assert r['png_path'] is None and r['png_sha256'] is None
        assert sha(read) == readhashes[str(read)]
        evidence += [read, done]
    expected = {(pid, c, d, control, family, 'mcq') for pid in ids for d in [0, 5, 10]
                for control in ['memory', 'mismatch', 'blank', 'text'] if control != 'blank' or d == 0
                for c in (range(2) if control in ['memory', 'mismatch'] else [0]) for family in ['T1', 'T2', 'T3']}
    assert seen == expected and len(seen) == 35040
    # The unchanged frozen summarizer accepts an explicitly scoped in-memory slice.
    # The persisted formal protocol and live run are never modified.
    sliced = copy.deepcopy(protocol)
    sliced['arms'], sliced['initial_variants'] = ['C8'], [0]
    summary = module.summarize(SOURCE, sliced)
    report = {'status': 'completed_preplanned_C8_V0_long10_slice_only_not_final_comparison',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source': str(SOURCE),
        'scope': 'C8 only, V0 only, all730 and two prespecified seeds; no B0 or V1. No model or budget selection.',
        'limitations': ['All730 were in B0 training, only64 in C8 adaptation.', 'C8 is single-write trained, not retention trained.',
            'Conditional retention is separate from unconditional accuracy.', 'No final C8 versus B0 conclusion.'],
        'protocol_sha256': sha(SOURCE/'protocol.json'), 'verified_png_sha256': pngs,
        'verified_chains': 1460, 'unique_readbacks': 35040, 'readback_sha256': readhashes,
        'finished_receipts': receipts, 'results': summary, 'script_sha256': sha(Path(__file__))}
    OUTPUT.mkdir()
    result = OUTPUT/'C8_LONG10_V0_STAGE_RESULTS.json'
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    evidence += [result, Path(__file__)]
    manifest = OUTPUT/'evidence-manifest.json'
    manifest.write_text(json.dumps({str(p.relative_to(TASK)): sha(p) for p in evidence}, indent=2)+'\n')
    archive = TASK/'c8-long10-v0-stage-evidence-20260929.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tar:
        for path in evidence + [manifest]:
            tar.add(path, arcname=str(path.relative_to(TASK)), recursive=False)
    receipt = {'path': str(archive), 'sha256': sha(archive), 'files': len(evidence)+1,
        'png_policy': 'All16060 PNG bytes rehashed on shared disk; archive holds all complete/writes and hashes.'}
    (OUTPUT/'archive.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'strata': {k: summary['arms']['C8']['strata'][k] for k in ['all730', 'adapted64', 'other666']},
        'comparisons': summary['comparisons'], 'parse_failures': summary['arms']['C8']['parse_failures'], 'archive': receipt}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
