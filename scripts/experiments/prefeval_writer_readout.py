"""Read-only functional diagnostic of complete shared Writer endpoints."""
import argparse
from collections import defaultdict
import json
import hashlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.experiments.prefeval_k1_data import load_records, event_text, sha
from scripts.experiments.prefeval_multitarget_bank import select_rows

ENDPOINTS = ('b730', 'direct-20261005', 'direct-20261006')
PARENT_SHA = '088c003d24cb2982571d763dd218c57f72ce80853615a09a97d8b760ae880c5b'
OLD = Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b730-exposure512-20260927')
UPSTREAM = Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/runs/dreamlite-aris-20261005/mainline')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_once(path, value):
    from scripts.inspire.run_prompt_matching_parallel import write_json
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and read(path) != value:
        raise ValueError('Frozen identity changed: ' + str(path))
    if not path.exists():
        write_json(path, value)


def cohort_rows():
    pilot = select_rows(load_records('pilot'), ROOT / 'configs/experiments/context_coverage_ids.json')
    dev = load_records('dev')
    train_ids = {r['base_pair_id'] for r in load_records('train')}
    if len(pilot) != 16 or len(dev) != 90 or train_ids & {r['base_pair_id'] for r in dev}:
        raise ValueError('Frozen cohort or train/dev separation changed')
    return {'pilot': pilot, 'dev': dev}


def verify_completion(endpoint):
    folder = OLD / 'train' if endpoint == 'b730' else UPSTREAM / ('seed-' + endpoint.split('-')[1]) / 'warmup'
    receipt = folder / 'complete.json'
    if not receipt.exists():
        return None
    done = read(receipt)
    checkpoint = folder / 'checkpoint-final.pt'
    if sha(checkpoint) != done['checkpoint_sha256']:
        raise ValueError('Checkpoint receipt mismatch')
    if endpoint == 'b730':
        if done['checkpoint_sha256'] != PARENT_SHA or done['steps'] != 93440:
            raise ValueError('Wrong B730 endpoint')
    else:
        manifest = read(folder / 'manifest.json')
        if (done['status'] != 'completed' or done['steps'] != 128 or done['manifest'] != manifest
                or hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest() != done['manifest_sha256']
                or sha(folder / 'optimization.jsonl') != done['optimization_sha256']):
            raise ValueError('Incomplete upstream endpoint')
        expected = {'seed': int(endpoint.split('-')[1]), 'stage': 'warmup', 'mode': 'use',
                    'steps': 128, 'parent_sha256': PARENT_SHA, 'split': 'train', 'preferences': 730,
                    'sample_steps': 28, 'cfg': 1.0, 'pixels': 1024,
                    'supervision': 'full_vocabulary_prompt_matching'}
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError('Unexpected upstream scientific manifest')
        updates = [json.loads(s)['step'] for s in (folder / 'optimization.jsonl').read_text().splitlines()]
        if updates != list(range(1, 129)):
            raise ValueError('Incomplete optimization denominator')
    return {'checkpoint': str(checkpoint), 'checkpoint_sha256': done['checkpoint_sha256'],
            'receipt': str(receipt), 'receipt_sha256': sha(receipt)}


def images_root(output, endpoint, split):
    return OLD / 'images' / f'step-093440-{split}-V0' if endpoint == 'b730' else output / 'images' / endpoint / split


def verify_assets(output, endpoints=ENDPOINTS):
    from vision_memory.training.latent_bank_unet import stable_seed
    assets = {}
    for endpoint in endpoints:
        binding = read(output / 'endpoints' / (endpoint + '.json'))
        for split, rows in cohort_rows().items():
            root = images_root(output, endpoint, split)
            manifest = read(root / 'manifest.json')
            required = {'checkpoint_sha256': binding['checkpoint_sha256'], 'split': split,
                        'steps': 28, 'cfg': 1, 'noise_chains': 2, 'inter_turns': 0}
            if any(manifest.get(k) != v for k, v in required.items()) or manifest.get('noise_domain', 'eval') != 'eval':
                raise ValueError('Rollout protocol mismatch')
            if endpoint == 'b730' and manifest.get('initial_variant') != 0:
                raise ValueError('Only the unchanged original V0 may be reused')
            for row in rows:
                pid = row['base_pair_id']
                for chain in range(2):
                    folder = root / pid.replace(':', '_') / f'seed-{chain}'
                    done = read(folder / 'complete.json')
                    png = folder / 'prefix-00.png'
                    writes = [json.loads(s) for s in (folder / 'writes.jsonl').read_text().splitlines()]
                    expected_seed = stable_seed(20260924, f'rollout:{pid}:{chain}', 0)
                    if done['binding'] != manifest or sha(png) != done['png_hashes']['prefix-00.png']:
                        raise ValueError('PNG identity mismatch')
                    # Failed initial writes may leave an identical diagnostic log tail; the published
                    # completion is accepted only if every such initial entry binds the same event/noise.
                    if not writes or any(w['position'] != 0 or w['event'] != event_text(row['history'][:2])
                            or w['noise_seed'] != expected_seed or w['output_png_sha256'] != sha(png) for w in writes):
                        raise ValueError('Source event/noise does not match paired protocol')
                    assets[f'{endpoint}|{pid}|{chain}'] = {'path': str(png), 'sha256': sha(png),
                                                         'complete_sha256': sha(folder / 'complete.json')}
    return assets


def expected_keys(ids, queries):
    return {(pid, q['id'], e, c, chain) for pid in ids for q in queries for e in ENDPOINTS
            for c in ('memory', 'mismatch') for chain in range(2)} | {
                (pid, q['id'], c, c, 0) for pid in ids for q in queries for c in ('blank', 'text')}


def rollout(args):
    from scripts.experiments.prefeval_k1_writer import load_pipe, rollout as native_rollout
    from vision_memory.repro import configure_strict_cuda_determinism
    binding = read(args.output / 'endpoints' / (args.endpoint + '.json'))
    if verify_completion(args.endpoint) != binding:
        raise ValueError('Upstream changed after binding')
    if args.endpoint == 'b730':
        verify_assets(args.output, ['b730'])
        return
    configure_strict_cuda_determinism(0)
    common = dict(base=args.base, official_source=args.official_source, device=args.device,
                  checkpoint=Path(binding['checkpoint']), noise_chains=2, inter_turns=0,
                  probe_initial_sources=None, noise_domain='eval', initial_variants=None,
                  shard_count=1)
    pipe = load_pipe(SimpleNamespace(**common))
    for split, rows in cohort_rows().items():
        native_rollout(SimpleNamespace(**common, split=split, output=images_root(args.output, args.endpoint, split)), pipe, rows)
    assets = verify_assets(args.output, [args.endpoint])
    save_once(args.output / 'images' / args.endpoint / 'verified.json', {'count': len(assets), 'assets': assets})


def evaluate(args):
    import torch
    from scripts.experiments.prefeval_context_readout import validate_protocol
    from scripts.experiments.prefeval_prompt_matching import get_history_target, supervision_binding, history_teacher_query
    from scripts.eval.prefeval_rgb import load_reader, read_png, append
    from scripts.experiments.prefeval_k1_teacher import save_json
    from vision_memory.reader.open_eos import assistant_termination_contract
    from vision_memory.reader.prompt_matching import qwen3vl_continuation_logits, soft_target_kl_divergence
    from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
    from vision_memory.repro import configure_strict_cuda_determinism
    configure_strict_cuda_determinism(0)
    torch.set_num_threads(1)
    spec = read(args.protocol)
    validate_protocol(spec)
    cohorts = cohort_rows()
    all_rows = [r for rows in cohorts.values() for r in rows]
    rows = all_rows[args.shard::2]
    split_of = {r['base_pair_id']: split for split, rs in cohorts.items() for r in rs}
    peers = defaultdict(list)
    for r in all_rows:
        peers[split_of[r['base_pair_id']], r['topic']].append(r['base_pair_id'])
    assets = verify_assets(args.output)
    binding = supervision_binding(SimpleNamespace(supervision='prompt_matching', temperature=1., reader=args.reader,
        teacher_max_new_tokens=512, teacher_cache=None, boundary_recovery_from=None), all_rows)
    for e in ENDPOINTS[1:]:
        upstream = read(read(args.output / 'endpoints' / (e + '.json'))['receipt'])['manifest']['teacher_binding']
        for key in ('reader_weights_sha256', 'reader_config_sha256', 'target_pipeline_sha256', 'reader_objective_sha256'):
            if upstream[key] != binding[key]:
                raise ValueError('Reader identity differs from training: ' + key)
    identity = {'protocol_sha256': sha(args.protocol), 'source_sha256': sha(Path(__file__)),
                'assignment': [r['base_pair_id'] for r in rows], 'assets': assets, 'reader': binding,
                'data_sha256': sha(ROOT / 'reports/prefeval-official-alignment-20260923/data/sft-train-10interturn.jsonl.gz'),
                'split_sha256': sha(ROOT / 'reports/prefeval-official-alignment-20260923/WRITER_IMPLEMENTATION_SPLIT.json')}
    ident = args.output / f'identity-{args.shard}.json'
    save_once(ident, identity)
    dest = args.output / f'readout-{args.shard}.jsonl'
    completed = {}
    for line in dest.read_text().splitlines() if dest.exists() else []:
        value = json.loads(line)
        key = tuple(value[k] for k in ('pair_id', 'query_id', 'endpoint', 'control', 'chain'))
        if key in completed:
            raise ValueError('Duplicate readout row')
        completed[key] = value
    expected = expected_keys(identity['assignment'], spec['queries'])
    if not set(completed) <= expected:
        raise ValueError('Unexpected readout rows')
    processor, reader = load_reader(args.reader, args.device)
    termination = assistant_termination_contract(reader, processor)
    gray = torch.full((3, 1024, 1024), 128 / 255, dtype=torch.float32, device=args.device)
    with torch.no_grad():
        for row in rows:
            pid = row['base_pair_id']
            same = peers[split_of[pid], row['topic']]
            donor = same[(same.index(pid) + 1) % len(same)]
            if donor == pid:
                raise ValueError('Invalid mismatch control')
            for q in spec['queries']:
                target, path = get_history_target(cache_root=args.output / 'targets', binding=binding, row=row,
                    query=q['query'], model=reader, processor=processor, reference=gray, device=args.device,
                    assistant_end_token_id=termination['assistant_end_token_id'])
                teacher = target['logits'].to(args.device)
                cases = [('blank', 'blank', 0), ('text', 'text', 0)] + [
                    (e, c, chain) for e in ENDPOINTS for c in ('memory', 'mismatch') for chain in range(2)]
                for endpoint, control, chain in cases:
                    key = (pid, q['id'], endpoint, control, chain)
                    if key in completed:
                        continue
                    source_pid = donor if control == 'mismatch' else pid
                    asset = assets[f'{endpoint}|{source_pid}|{chain}'] if control in ('memory', 'mismatch') else None
                    pixels = gray if asset is None else read_png(asset['path']).to(args.device)
                    query = history_teacher_query(row, q['query']) if control == 'text' else q['query']
                    out = qwen3vl_continuation_logits(model=reader, processor=processor, image=pixels,
                        query=query, target_ids=torch.tensor([target['target_ids']]), device=args.device,
                        require_image_grad=False, reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
                    kl = float(soft_target_kl_divergence(out.target_logits, teacher))
                    if not math.isfinite(kl):
                        raise ValueError('Nonfinite KL')
                    record = dict(zip(('pair_id', 'query_id', 'endpoint', 'control', 'chain'), key))
                    record.update(split=split_of[pid], family=q['family'], kl=kl, png=asset,
                        donor_pair_id=donor if control == 'mismatch' else None, teacher_target=str(path),
                        teacher_logits_sha256=target['logits_sha256'], target_ids=target['target_ids'])
                    append(dest, record)
                    completed[key] = record
                del teacher, target, out
            print(json.dumps({'completed': pid, 'shard': args.shard, 'rows': len(completed)}), flush=True)
    if set(completed) != expected:
        raise ValueError('Incomplete denominator')
    save_json(args.output / f'finished-{args.shard}.json', {'rows': len(completed),
        'identity_sha256': sha(ident), 'readout_sha256': sha(dest)})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['rollout', 'evaluate'])
    p.add_argument('--endpoint', choices=ENDPOINTS)
    p.add_argument('--shard', type=int, choices=[0, 1])
    for name in ('output', 'base', 'official-source', 'reader', 'protocol'):
        p.add_argument('--' + name, type=Path, required=name == 'output')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    (rollout if args.phase == 'rollout' else evaluate)(args)
