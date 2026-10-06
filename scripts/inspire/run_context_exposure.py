"""Bounded 128-to-256 update extension with exact optimizer/RNG continuity."""
import argparse
from collections import Counter
import concurrent.futures
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.inspire import run_context_population_readout as p
from scripts.experiments.prefeval_route_functional import read, read_lines, sha, save_once, new_keys, index_rows, load_target, validate_new_row, KEYS
from scripts.inspire.run_writer_readout import execute, accrued_seconds, other_seconds
from scripts.inspire.run_prompt_matching_parallel import write_json
from scripts.reporting.context_coverage_report import paired_interval
from scripts.inspire.run_context_population import PARENT_SHA, OLD

RUN = p.RUN
PREVIOUS = RUN/'context-population-readout-v1'
TRAIN_SOURCE = p.SOURCE/'writer32'
RESUME_SHA = '0f2fb4b15ffea8e1843610f8d62ce191e6b695c08aaa7d89df12b64b526af057'
ENDPOINT = 'writer32-256'


def equivalent(a, b):
    import torch
    import numpy as np
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if type(a) != type(b): return False
    if isinstance(a, dict): return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    return a == b


def extended_payload(source):
    if (source['schema_version'] != 1 or source['optimizer_step'] != 128 or source['episode_cursor'] != 512
            or source['manifest']['steps'] != 128 or source['epoch'] != 0):
        raise ValueError('Not the completed 128-step cursor')
    if not {'python', 'numpy', 'torch_cpu'} <= source['rng_state'].keys() or not source['optimizer']['state']:
        raise ValueError('Incomplete optimizer/RNG state')
    if {int(s['step']) for s in source['optimizer']['state'].values()} != {128}:
        raise ValueError('Optimizer cursor mismatch')
    return {**source, 'manifest': {**source['manifest'], 'steps': 256}}


def prepare_resume(output):
    import torch
    from scripts.experiments.prefeval_k1_teacher import atomic_save
    source = TRAIN_SOURCE/'resume.pt'; final = TRAIN_SOURCE/'checkpoint-final.pt'
    if sha(source) != RESUME_SHA or sha(final) != p.WRITER32_SHA:
        raise ValueError('Frozen source checkpoint changed')
    old = torch.load(source, map_location='cpu', weights_only=False)
    frozen = torch.load(final, map_location='cpu', weights_only=True)
    if old['manifest'] != read(TRAIN_SOURCE/'manifest.json') or old['manifest'] != frozen['manifest']:
        raise ValueError('Source manifest mismatch')
    if not equivalent(old['trainable_state'], frozen['trainable_state']):
        raise ValueError('Resume and inference weights disagree')
    if old['manifest']['implementation_sha256'] != sha(ROOT/'scripts/experiments/prefeval_k1_writer.py'):
        raise ValueError('Native trainer changed')
    extended = extended_payload(old)
    folder = output/'train'; folder.mkdir(exist_ok=True)
    seed = folder/'resume-initial128.pt'
    if seed.exists():
        if not equivalent(torch.load(seed, map_location='cpu', weights_only=False), extended):
            raise ValueError('Initial resume changed')
    else:
        if (folder/'resume.pt').exists() or (folder/'optimization.jsonl').exists():
            raise ValueError('Unclaimed partial training')
        atomic_save(seed, extended)
        if not equivalent(torch.load(seed, map_location='cpu', weights_only=False), extended):
            raise ValueError('Resume serialization changed state')
        shutil.copyfile(seed, folder/'resume.pt')
        shutil.copyfile(TRAIN_SOURCE/'optimization.jsonl', folder/'optimization.jsonl')
    receipt = dict(source_resume=str(source), source_resume_sha256=RESUME_SHA, source_final_sha256=p.WRITER32_SHA,
        initial_resume_sha256=sha(seed), source_log_sha256=sha(TRAIN_SOURCE/'optimization.jsonl'),
        original_manifest=old['manifest'], extended_manifest=extended['manifest'],
        allowed_change='manifest.steps:128->256 only; all tensor, Adam, RNG and cursor state identical',
        optimizer_step=128, episode_cursor=512, prefix_rows=128)
    save_once(output/'resume-extension.json', receipt)
    return receipt


def budget(output):
    previous = other_seconds(RUN) + sum(accrued_seconds(output.parent/name) for name in
        ('writer-readout-v1', 'route-functional-v1', 'context-fit-v1', 'context-dev-v1',
         'context-population32-v1', 'context-population-readout-v1'))
    current = accrued_seconds(output)
    return min(1.5*3600-current, 16*3600-previous-current), previous, current


def previous_inputs():
    p.report(PREVIOUS)
    old, ref = p.load_reused(read(p.PROTOCOL))
    files = dict(ref['files']); targets = dict(ref['targets'])
    for shard in range(2):
        old += read_lines(PREVIOUS/f'readout-{shard}.jsonl')
        targets.update(read(PREVIOUS/f'targets-{shard}.json'))
        for name in (f'readout-{shard}.jsonl', f'identity-{shard}.json', f'finished-{shard}.json', f'targets-{shard}.json'):
            path = PREVIOUS/name; files[str(path)] = sha(path)
    for path in (PREVIOUS/'comparison.json', PREVIOUS/'inputs.json', PREVIOUS/'source.json'):
        files[str(path)] = sha(path)
    ids = [r['base_pair_id'] for r in p.population()[0]]
    index_rows(old, p.expected_keys(ids, read(p.PROTOCOL)))
    return old, dict(files=files, targets=targets, reader=ref['reader'], images=read(PREVIOUS/'inputs.json')['images'])


def verify_training(output):
    import torch
    from vision_memory.training.latent_bank_unet import stable_seed
    folder = output/'train'; binding = read(output/'resume-extension.json'); m = read(folder/'manifest.json')
    if m != binding['extended_manifest'] or m != {**read(TRAIN_SOURCE/'manifest.json'), 'steps': 256}:
        raise ValueError('Scientific factor changed beyond update budget')
    original_log = (TRAIN_SOURCE/'optimization.jsonl').read_bytes(); log_bytes = (folder/'optimization.jsonl').read_bytes()
    if sha(TRAIN_SOURCE/'optimization.jsonl') != binding['source_log_sha256'] or not log_bytes.startswith(original_log):
        raise ValueError('Old 128 update prefix changed')
    logs = read_lines(folder/'optimization.jsonl'); rows = p.population()[0]; counts = Counter()
    if [r['step'] for r in logs] != list(range(1, 257)): raise ValueError('Incomplete optimization denominator')
    for r in logs:
        if not math.isfinite(r['grad_norm']) or r['grad_norm'] <= 0 or len(r['draws']) != 4: raise ValueError('Bad gradients')
        for micro, draw in enumerate(r['draws']):
            i = (r['step']-1)*4+micro; cycle, offset = divmod(i, 32)
            order = torch.randperm(32, generator=torch.Generator().manual_seed(stable_seed(20260924, 'order', cycle))).tolist()
            sigma = float(torch.rand((), generator=torch.Generator().manual_seed(stable_seed(20260924, 'sigma', i))))
            if (draw['pair_id'] != rows[order[offset]]['base_pair_id'] or draw['sigma'] != sigma
                    or draw['position'] != 0 or not math.isfinite(draw['mse'])): raise ValueError('Draw continuity mismatch')
            counts[draw['pair_id']] += 1
    if len(counts) != 32 or set(counts.values()) != {32}: raise ValueError('Wrong per-history exposure')
    final = torch.load(folder/'checkpoint-final.pt', map_location='cpu', weights_only=True)
    resume = torch.load(folder/'resume.pt', map_location='cpu', weights_only=False)
    if (final['manifest'] != m or resume['manifest'] != m or final['optimizer_step'] != 256 or resume['optimizer_step'] != 256
            or resume['episode_cursor'] != 1024 or {int(v['step']) for v in resume['optimizer']['state'].values()} != {256}
            or not equivalent(final['trainable_state'], resume['trainable_state'])
            or any(not torch.isfinite(t).all() for t in final['trainable_state'].values())):
        raise ValueError('Invalid final checkpoint/cursor')
    checksum = sha(folder/'checkpoint-final.pt')
    if read(folder/'complete.json') != dict(steps=256, checkpoint_sha256=checksum): raise ValueError('Completion changed')
    return dict(total_updates=256, new_updates=128, total_draws=1024, draws_per_history=32,
        prefix_exact=True, finite_nonzero_gradients=True, checkpoint_sha256=checksum, resume_sha256=sha(folder/'resume.pt'))


def verify_images(output):
    from PIL import Image
    from vision_memory.training.latent_bank_unet import stable_seed
    checkpoint = output/'train/checkpoint-final.pt'; images = output/'images'; m = read(images/'manifest.json')
    expected = dict(checkpoint_sha256=sha(checkpoint), split='pilot', steps=28, cfg=1, noise_chains=2,
        inter_turns=0, state='only reopened uint8 RGB PNG; fresh Gaussian each write')
    if m != expected or len(list(images.glob('*/seed-*/complete.json'))) != 64: raise ValueError('Image denominator/protocol')
    assets = {}
    for row in p.population()[0]:
        pid = row['base_pair_id']
        for chain in range(2):
            folder = images/pid.replace(':', '_')/f'seed-{chain}'; png = folder/'prefix-00.png'; h = sha(png)
            if read(folder/'complete.json') != dict(binding=m, png_hashes={png.name: h}): raise ValueError('PNG hash binding')
            if read_lines(folder/'writes.jsonl') != [dict(position=0, source_png_sha256=None, output_png_sha256=h,
                    noise_seed=stable_seed(20260924, f'rollout:{pid}:{chain}', 0), event=p.event_text(row['history'][:2]))]:
                raise ValueError('Image history/noise mismatch')
            with Image.open(png) as image:
                if image.mode != 'RGB' or image.size != (1024, 1024): raise ValueError('Invalid PNG')
                image.load()
            assets[f'{ENDPOINT}|{pid}|{chain}'] = dict(path=str(png), sha256=h, complete_sha256=sha(folder/'complete.json'))
    files = {str(path): sha(path) for path in (checkpoint, images/'manifest.json', output/'train/complete.json')}
    return dict(assets=assets, files=files)


def summarize(old, new, spec):
    groups = p.strata(); ids = [r['base_pair_id'] for rs in groups.values() for r in rs]
    table = index_rows(old, p.expected_keys(ids, spec)); table.update(index_rows(new, new_keys(ids, spec, (ENDPOINT,))))
    for pid in ids:
        for q in spec['queries']:
            ref = table[pid, q['id'], 'text', 'text', 0]
            for key in new_keys([pid], {'queries': [q]}, (ENDPOINT,)):
                if any(table[key][k] != ref[k] for k in ('teacher_target', 'target_ids', 'teacher_logits_sha256')):
                    raise ValueError('Unpaired teacher distribution')
    result = dict(combined_rows=len(table), reused_rows=len(old), new_rows=len(new),
        metric='Training-side teacher-prefix KL; fixed32 population,128 vs256 updates; no promotion', strata={})
    for group, rows in groups.items():
        ids = [r['base_pair_id'] for r in rows]; out = dict(independent_n=16, families={})
        for family in ('recall', 'application', 'neutral'):
            qs = [q['id'] for q in spec['queries'] if q['family'] == family]
            def vals(e, c):
                ns = [0] if c in ('blank', 'text') else [0, 1]
                return [sum(table[pid, q, e, c, n]['kl'] for q in qs for n in ns)/(len(qs)*len(ns)) for pid in ids]
            mean = lambda v: sum(v)/len(v)
            blank = vals('blank', 'blank'); mem = vals(ENDPOINT, 'memory')
            f = dict(blank_kl=mean(blank), text_self_consistency_kl=mean(vals('text', 'text')), endpoints={})
            for e in (*p.ENDPOINTS, ENDPOINT):
                a, b = vals(e, 'memory'), vals(e, 'mismatch')
                f['endpoints'][e] = dict(memory_kl=mean(a), mismatch_kl=mean(b),
                    mismatch_minus_memory=paired_interval([x-y for x, y in zip(b, a)]),
                    blank_minus_memory=paired_interval([x-y for x, y in zip(blank, a)]))
            f['updates128_minus256'] = paired_interval([x-y for x, y in zip(vals('writer32', 'memory'), mem)])
            f['writer16_minus256'] = paired_interval([x-y for x, y in zip(vals('writer16', 'memory'), mem)])
            out['families'][family] = f
        result['strata'][group] = out
    return result


def evaluate(args):
    import torch
    from scripts.eval.prefeval_rgb import load_reader, read_png, append
    from vision_memory.reader.open_eos import assistant_termination_contract
    from vision_memory.reader.prompt_matching import qwen3vl_continuation_logits, soft_target_kl_divergence
    from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
    from vision_memory.repro import configure_strict_cuda_determinism
    configure_strict_cuda_determinism(0); torch.set_num_threads(1)
    spec = read(p.PROTOCOL); p.validate_protocol(spec); frozen = read(args.output/'inputs.json'); ref = frozen['reference']
    for path, h in {**ref['files'], **frozen['images']['files']}.items():
        if sha(Path(path)) != h: raise ValueError('Frozen source changed')
    assets = frozen['images']['assets']; donors, _ = p.donors_and_groups()
    for asset in assets.values():
        if sha(Path(asset['path'])) != asset['sha256']: raise ValueError('PNG changed')
    binding = p.reader_binding(args.reader)
    if binding != ref['reader']: raise ValueError('Reader changed')
    rows = p.population()[0][args.shard::2]; queries = {q['id']: q for q in spec['queries']}
    identity = dict(inputs_sha256=sha(args.output/'inputs.json'), source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(p.PROTOCOL), assignment=[r['base_pair_id'] for r in rows])
    ident = args.output/f'identity-{args.shard}.json'; save_once(ident, identity)
    dest = args.output/f'readout-{args.shard}.jsonl'; records = read_lines(dest) if dest.exists() else []
    expected = new_keys(identity['assignment'], spec, (ENDPOINT,)); keys = [tuple(r[k] for k in KEYS) for r in records]
    if len(set(keys)) != len(keys) or not set(keys) <= expected: raise ValueError('Foreign/duplicate partial rows')
    for r in records:
        validate_new_row(r, ref['targets'], assets, donors, queries, 'pilot')
        if not math.isfinite(r['kl']): raise ValueError('Nonfinite partial row')
    keys = set(keys); processor, reader = load_reader(args.reader, 'cuda:0')
    eos = assistant_termination_contract(reader, processor)['assistant_end_token_id']
    save_once(args.output/f'termination-{args.shard}.json', dict(eos=eos, pad=processor.tokenizer.pad_token_id))
    with torch.no_grad():
        for row in rows:
            pid = row['base_pair_id']
            for q in spec['queries']:
                meta = ref['targets'][f'{pid}|{q["id"]}']
                target = load_target(meta, binding, row, q['query'], eos, processor.tokenizer.pad_token_id)
                teacher = target['logits'].to('cuda:0'); out = None
                for key in sorted(new_keys([pid], {'queries': [q]}, (ENDPOINT,))):
                    if key in keys: continue
                    _, _, endpoint, control, chain = key; donor = donors[pid] if control == 'mismatch' else None
                    asset = assets[f'{endpoint}|{donor or pid}|{chain}']
                    out = qwen3vl_continuation_logits(model=reader, processor=processor, image=read_png(asset['path']).to('cuda:0'),
                        query=q['query'], target_ids=torch.tensor([target['target_ids']]), device='cuda:0',
                        require_image_grad=False, reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
                    kl = float(soft_target_kl_divergence(out.target_logits, teacher))
                    if not math.isfinite(kl): raise ValueError('Nonfinite KL')
                    value = dict(zip(KEYS, key)); value.update(split='pilot', family=q['family'], kl=kl, png=asset,
                        donor_pair_id=donor, teacher_target=meta['path'], teacher_logits_sha256=target['logits_sha256'], target_ids=target['target_ids'])
                    append(dest, value); keys.add(key)
                del teacher, target, out
            print(json.dumps(dict(completed=pid, shard=args.shard, rows=len(keys))), flush=True)
    if keys != expected: raise ValueError('Incomplete denominator')
    save_once(args.output/f'finished-{args.shard}.json', dict(rows=len(keys), identity_sha256=sha(ident), readout_sha256=sha(dest)))


def report(output):
    old, ref = previous_inputs(); frozen = read(output/'inputs.json'); spec = read(p.PROTOCOL)
    if ref != frozen['reference'] or verify_images(output) != frozen['images']: raise ValueError('Frozen inputs changed')
    if verify_training(output) != read(output/'training-audit.json'): raise ValueError('Training audit changed')
    rows = p.population()[0]; new = []; donors, _ = p.donors_and_groups(); queries = {q['id']: q for q in spec['queries']}
    for shard in range(2):
        path, ident = output/f'readout-{shard}.jsonl', output/f'identity-{shard}.json'
        identity = read(ident); done = read(output/f'finished-{shard}.json'); records = read_lines(path)
        if (done != dict(rows=len(records), identity_sha256=sha(ident), readout_sha256=sha(path))
                or identity != dict(inputs_sha256=sha(output/'inputs.json'), source_sha256=sha(Path(__file__)),
                    protocol_sha256=sha(p.PROTOCOL), assignment=[r['base_pair_id'] for r in rows[shard::2]])):
            raise ValueError('Readout receipt changed')
        index_rows(records, new_keys(identity['assignment'], spec, (ENDPOINT,)))
        for row in records: validate_new_row(row, ref['targets'], frozen['images']['assets'], donors, queries, 'pilot')
        new.extend(records)
    if read(output/'termination-0.json') != read(PREVIOUS/'termination-0.json') or read(output/'termination-1.json') != read(PREVIOUS/'termination-1.json'):
        raise ValueError('Reader termination changed')
    result = summarize(old, new, spec); save_once(output/'comparison.json', result)
    return result


def jobs(args, phase):
    common = [sys.executable, 'scripts/experiments/prefeval_k1_writer.py', phase, '--arm', 'B', '--split', 'pilot',
        '--ids-file', str(p.ALL_IDS), '--base', str(args.base), '--official-source', str(args.official_source)]
    if phase == 'train':
        common += ['--checkpoint', str(OLD/'train/checkpoint-final.pt'), '--output', str(args.output/'train'),
            '--teachers', str(p.SOURCE/'bank'), '--teacher-supervision', 'prompt_matching', '--teacher-steps', '288', '--steps', '256']
    elif phase == 'rollout':
        common += ['--checkpoint', str(args.output/'train/checkpoint-final.pt'), '--output', str(args.output/'images'), '--inter-turns', '0', '--noise-chains', '2']
    else: raise ValueError('Invalid phase')
    return [dict(name=phase, gpu=0, command=common)]


def main(args):
    if args.output.resolve() != RUN/'context-exposure-v1': raise ValueError('Wrong output')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(): raise ValueError('Dirty source')
    args.output.mkdir(parents=True, exist_ok=True); claim = args.output/'active-owner'; claim.mkdir()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        write_json(claim/'owner.json', dict(pid=os.getpid(), host=socket.gethostname(), started=time.time(), commit=commit))
        for name in ('attempts', 'receipts', 'logs'): (args.output/name).mkdir(exist_ok=True)
        save_once(args.output/'plan.json', dict(commit=commit, plan_sha256=sha(ROOT/'reports/context-coverage-20261006/CONTEXT_EXPOSURE_PLAN.md'),
            source=str(TRAIN_SOURCE), source_resume_sha256=RESUME_SHA, protocol_sha256=sha(p.PROTOCOL),
            base=str(args.base), reader=str(args.reader), official_source=str(args.official_source),
            phase_gpu_hours_cap=1.5, campaign_cap=16, steps=256, new_steps=128, new_rows=1536, combined_rows=5376))
        _, ref = previous_inputs()
        prepare_resume(args.output)
        log = args.output/'train/optimization.jsonl'
        if len(read_lines(log)) > 128 and not (args.output/'train/complete.json').exists():
            raise ValueError('Partial continuation requires checkpoint/tail recovery audit')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(): raise ValueError('GPU occupied')
        if len(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()) != 2: raise ValueError('Require two GPUs')
        if [x for x in RUN.glob('*/active-owner/owner.json') if x != claim/'owner.json']: raise ValueError('Another owner')
        left, previous, _ = budget(args.output)
        if left <= 60: raise ValueError('Budget exhausted')
        deadline = time.monotonic() + min(6*3600, (left-60)/2)
        def group(items):
            if time.monotonic()+30 >= deadline: raise ValueError('No time for next stage')
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                for f in [pool.submit(execute, j, args.output, deadline) for j in items]: f.result()
        write_json(args.output/'status.json', dict(status='training', time=time.time(), commit=commit))
        group(jobs(args, 'train')); save_once(args.output/'training-audit.json', verify_training(args.output))
        write_json(args.output/'status.json', dict(status='rollout', time=time.time(), commit=commit))
        group(jobs(args, 'rollout')); save_once(args.output/'inputs.json', dict(reference=ref, images=verify_images(args.output)))
        write_json(args.output/'status.json', dict(status='readout', time=time.time(), commit=commit))
        group([dict(name=f'evaluate-{i}', gpu=i, command=[sys.executable, str(Path(__file__)), 'evaluate',
            '--output', str(args.output), '--reader', str(args.reader), '--shard', str(i)]) for i in range(2)])
        result = report(args.output); current = accrued_seconds(args.output)
        write_json(args.output/'status.json', dict(status='completed', time=time.time(), rows=result['combined_rows'],
            gpu_hours=current/3600, campaign_gpu_hours=(previous+current)/3600))
    except BaseException as exc:
        write_json(args.output/'status.json', dict(status='failed', time=time.time(), error=str(exc))); raise
    finally:
        (claim/'owner.json').unlink(missing_ok=True); claim.rmdir()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=('run', 'evaluate'))
    for name in ('output', 'base', 'reader', 'official-source'): parser.add_argument('--'+name, type=Path, required=name in ('output', 'reader'))
    parser.add_argument('--shard', type=int, choices=(0, 1)); args = parser.parse_args()
    (main if args.phase == 'run' else evaluate)(args)
