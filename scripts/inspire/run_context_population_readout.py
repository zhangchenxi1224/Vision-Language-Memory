"""Frozen, stratified training-side readout of Writer16 versus Writer32."""
import argparse
import concurrent.futures
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.inspire.run_context_population import (
    population, check_parent, verify_bank, verify_writer, ALL_IDS, ADDED_IDS, FIT_SHA, RUN,
)
from scripts.inspire.run_context_fit import verify_images as fit_images
from scripts.inspire.run_context_dev import verify_source as fit_source
from scripts.inspire.run_writer_readout import execute, accrued_seconds, other_seconds
from scripts.inspire.run_prompt_matching_parallel import write_json
from scripts.experiments.prefeval_route_functional import (
    read, read_lines, sha, save_once, load_reference, load_target, index_rows, new_keys,
    donor_map, validate_new_row, cohort_rows, summarize as fit_summary, event_text, KEYS,
)
from scripts.experiments.prefeval_context_readout import validate_protocol
from scripts.reporting.context_coverage_report import paired_interval

ENDPOINTS = ('writer16', 'writer32')
WRITER32_SHA = 'bffc31bc526c8955675ba4d76c0a85ede2b47522db8d410bc15ee8f7c70ae051'
FIT_COMPARISON_SHA = 'd76aa691203d5db89fac94348bff93a8633134e2b6f35f5b500c7ab262459c02'
FIT_CONSUMER_SHA = '192541655aac0333a9b447e49b54544e72864683d5b77ae4259b563e8cfcebf0'
SOURCE = RUN/'context-population32-v1'
FIT = RUN/'context-fit-v1'
REFERENCE = RUN/'writer-readout-v1'
PROTOCOL = ROOT/'configs/experiments/context_readout_audit.json'


@lru_cache(maxsize=1)
def strata():
    rows, added = population()
    added_ids = {r['base_pair_id'] for r in added}
    return {'original16': [r for r in rows if r['base_pair_id'] not in added_ids], 'added16': added}


@lru_cache(maxsize=1)
def donors_and_groups():
    groups = strata()
    donors = {p: d for rows in groups.values() for p, d in donor_map(rows).items()}
    return donors, {r['base_pair_id']: s for s, rows in groups.items() for r in rows}


def expected_keys(ids, spec):
    return new_keys(ids, spec, ENDPOINTS) | {
        (p, q['id'], c, c, 0) for p in ids for q in spec['queries'] for c in ('blank', 'text')}


def reused_keys(spec):
    ids = [r['base_pair_id'] for r in strata()['original16']]
    return new_keys(ids, spec, ('writer16',)) | {
        (p, q['id'], c, c, 0) for p in ids for q in spec['queries'] for c in ('blank', 'text')}


def current_keys(ids, spec):
    return expected_keys(ids, spec) - reused_keys(spec)


def remaining_seconds(output):
    previous = other_seconds(output.parent) + sum(accrued_seconds(output.parent/p) for p in
        ('writer-readout-v1', 'route-functional-v1', 'context-fit-v1', 'context-dev-v1'))
    first = accrued_seconds(output.parent/'context-population32-v1')
    current = accrued_seconds(output)
    return min(3*3600-first-current, 16*3600-previous-first-current), previous, first, current


def verify_sources(prefeval):
    check_parent()
    if read(SOURCE/'status.json')['status'] != 'ready_for_frozen_readout':
        raise ValueError('Population training not complete')
    if any(not (SOURCE/'bank'/r['base_pair_id'].replace(':', '_')).is_symlink() for r in population()[0]):
        raise ValueError('Read-only source bank link missing')
    bank = verify_bank(SOURCE, prefeval)
    if bank != read(SOURCE/'bank.json') or verify_writer(SOURCE, bank) != read(SOURCE/'writer-audit.json'):
        raise ValueError('Population source changed')
    if sha(SOURCE/'writer32/checkpoint-final.pt') != WRITER32_SHA:
        raise ValueError('Writer32 final changed')
    files = {**bank['files'], **fit_source(FIT)}
    for folder, names in ((SOURCE, ('bank.json', 'writer-audit.json', 'plan.json')),
                          (SOURCE/'writer32', ('checkpoint-final.pt', 'manifest.json', 'optimization.jsonl', 'complete.json'))):
        for name in names:
            p = folder/name
            files[str(p)] = sha(p)
    return files


def load_reused(spec):
    """Recompute the entire old fit report before mapping its fixed wide endpoint."""
    old, ref = load_reference(REFERENCE, spec, PROTOCOL, 'pilot')
    frozen = read(FIT/'inputs.json')
    if frozen['reference'] != ref or fit_images(FIT) != frozen['upstream']:
        raise ValueError('Fit source assets changed')
    rows = cohort_rows()['pilot']; donors = donor_map(rows)
    queries = {q['id']: q for q in spec['queries']}; values = []
    files = dict(ref['files'])
    for shard in range(2):
        path, ident = FIT/f'readout-{shard}.jsonl', FIT/f'identity-{shard}.json'
        done = read(FIT/f'finished-{shard}.json'); identity = read(ident); records = read_lines(path)
        if (done != dict(rows=len(records), identity_sha256=sha(ident), readout_sha256=sha(path))
                or identity['inputs_sha256'] != sha(FIT/'inputs.json')
                or identity['protocol_sha256'] != sha(PROTOCOL) or identity['source_sha256'] != FIT_CONSUMER_SHA
                or identity['assignment'] != [r['base_pair_id'] for r in rows[shard::2]]):
            raise ValueError('Old fit receipt/assignment changed')
        index_rows(records, new_keys(identity['assignment'], spec, ('context-narrow', 'context-diverse')))
        for r in records:
            validate_new_row(r, ref['targets'], frozen['upstream']['assets'], donors, queries, 'pilot')
        values.extend(records)
        for p in (path, ident, FIT/f'finished-{shard}.json'):
            files[str(p)] = sha(p)
    if (sha(FIT/'comparison.json') != FIT_COMPARISON_SHA or
            fit_summary(old, values, [r['base_pair_id'] for r in rows], spec,
                        ('context-narrow', 'context-diverse'),
                        {'context-narrow': 'b730', 'context-diverse': 'context-narrow'}) != read(FIT/'comparison.json')):
        raise ValueError('Old fit report changed')
    for p in (FIT/'comparison.json', FIT/'inputs.json'):
        files[str(p)] = sha(p)
    reused = [dict(r, endpoint='writer16', provenance_endpoint='context-diverse')
              for r in values if r['endpoint'] == 'context-diverse']
    reused += [r for r in old if r['control'] in ('blank', 'text')]
    index_rows(reused, reused_keys(spec))
    return reused, dict(files=files, reader=ref['reader'], targets=ref['targets'])


def checkpoints():
    return {'writer16': (FIT/'context-diverse/train/checkpoint-final.pt', FIT_SHA),
            'writer32': (SOURCE/'writer32/checkpoint-final.pt', WRITER32_SHA)}


def verify_images(output):
    from PIL import Image
    from vision_memory.training.latent_bank_unet import stable_seed
    groups = strata(); assets, files = {}, {}
    for endpoint, (checkpoint, checksum) in checkpoints().items():
        if sha(checkpoint) != checksum:
            raise ValueError('Fixed Writer changed')
        files[str(checkpoint)] = checksum
        folders = [(output/endpoint/'images', population()[0])] if endpoint == 'writer32' else [
            (FIT/'context-diverse/images', groups['original16']), (output/endpoint/'images', groups['added16'])]
        for images, rows in folders:
            m = read(images/'manifest.json')
            expected = dict(checkpoint_sha256=checksum, split='pilot', steps=28, cfg=1,
                noise_chains=2, inter_turns=0, state='only reopened uint8 RGB PNG; fresh Gaussian each write')
            if m != expected or len(list(images.glob('*/seed-*/complete.json'))) != len(rows)*2:
                raise ValueError('PNG manifest/denominator mismatch')
            files[str(images/'manifest.json')] = sha(images/'manifest.json')
            for row in rows:
                pid = row['base_pair_id']
                for chain in range(2):
                    folder = images/pid.replace(':', '_')/f'seed-{chain}'; png = folder/'prefix-00.png'; h = sha(png)
                    if read(folder/'complete.json') != dict(binding=m, png_hashes={png.name: h}):
                        raise ValueError('PNG binding changed')
                    expected_write = dict(position=0, source_png_sha256=None, output_png_sha256=h,
                        noise_seed=stable_seed(20260924, f'rollout:{pid}:{chain}', 0), event=event_text(row['history'][:2]))
                    if read_lines(folder/'writes.jsonl') != [expected_write]:
                        raise ValueError('PNG history/noise changed')
                    with Image.open(png) as image:
                        if image.mode != 'RGB' or image.size != (1024, 1024):
                            raise ValueError('Bad PNG')
                        image.load()
                    assets[f'{endpoint}|{pid}|{chain}'] = dict(path=str(png), sha256=h, complete_sha256=sha(folder/'complete.json'))
    return dict(assets=assets, files=files)


def reader_binding(reader):
    from scripts.experiments.prefeval_prompt_matching import supervision_binding
    # Preserve the original Reader identity, including its historical cohort digest.
    rows = [r for rs in cohort_rows().values() for r in rs]
    return supervision_binding(SimpleNamespace(supervision='prompt_matching', temperature=1., reader=reader,
        teacher_max_new_tokens=512, teacher_cache=None, boundary_recovery_from=None), rows)


def validate_row(row, targets, assets, spec):
    donors, groups = donors_and_groups(); pid = row['pair_id']; qid = row['query_id']
    queries = {q['id']: q for q in spec['queries']}
    if row['control'] in ('memory', 'mismatch'):
        validate_new_row(row, targets, assets, donors, queries, 'pilot')
    else:
        t = targets[f'{pid}|{qid}']; c = row['control']
        if (row['split'] != 'pilot' or row['family'] != queries[qid]['family']
                or row['endpoint'] != c or c not in ('blank', 'text') or row['chain'] != 0
                or row['png'] is not None or row['donor_pair_id'] is not None
                or row['teacher_target'] != t['path'] or row['target_ids'] != t['target_ids']
                or row['teacher_logits_sha256'] != t['logits_sha256']):
            raise ValueError('Control target binding mismatch')
    if not math.isfinite(row['kl']) or pid not in groups:
        raise ValueError('Invalid readout')


def summarize(records, spec):
    groups = strata(); ids = [r['base_pair_id'] for rs in groups.values() for r in rs]
    table = index_rows(records, expected_keys(ids, spec))
    for pid in ids:
        for q in spec['queries']:
            ref = table[pid, q['id'], 'text', 'text', 0]
            for key in expected_keys([pid], {'queries': [q]}):
                if any(table[key][k] != ref[k] for k in ('teacher_target', 'target_ids', 'teacher_logits_sha256')):
                    raise ValueError('Unpaired teacher distribution')
    result = dict(combined_rows=len(table), reused_rows=1152, new_rows=2688,
        metric='Conditional teacher-prefix KL; trained-history strata; no dev/generalization or promotion claim', strata={})
    for group, rows in groups.items():
        ids = [r['base_pair_id'] for r in rows]; out = dict(independent_n=len(ids), rows=len(ids)*120, families={})
        for family in ('recall', 'application', 'neutral'):
            qs = [q['id'] for q in spec['queries'] if q['family'] == family]
            def vals(e, c):
                ns = [0] if c in ('blank', 'text') else [0, 1]
                return [sum(table[p, q, e, c, n]['kl'] for q in qs for n in ns)/(len(qs)*len(ns)) for p in ids]
            mean = lambda v: sum(v)/len(v)
            blank, text = vals('blank', 'blank'), vals('text', 'text')
            f = dict(blank_kl=mean(blank), text_self_consistency_kl=mean(text), endpoints={})
            for e in ENDPOINTS:
                mem, wrong = vals(e, 'memory'), vals(e, 'mismatch')
                f['endpoints'][e] = dict(memory_kl=mean(mem), mismatch_kl=mean(wrong),
                    mismatch_minus_memory=paired_interval([a-b for a, b in zip(wrong, mem)]),
                    blank_minus_memory=paired_interval([a-b for a, b in zip(blank, mem)]))
            f['writer16_minus_writer32'] = paired_interval([a-b for a, b in zip(vals('writer16', 'memory'), vals('writer32', 'memory'))])
            out['families'][family] = f
        result['strata'][group] = out
    return result


def evaluate(args):
    import torch
    from scripts.experiments.prefeval_prompt_matching import get_history_target, history_teacher_query
    from scripts.eval.prefeval_rgb import load_reader, read_png, append
    from vision_memory.reader.open_eos import assistant_termination_contract
    from vision_memory.reader.prompt_matching import qwen3vl_continuation_logits, soft_target_kl_divergence
    from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
    from vision_memory.repro import configure_strict_cuda_determinism
    configure_strict_cuda_determinism(0); torch.set_num_threads(1)
    spec = read(PROTOCOL); validate_protocol(spec); frozen = read(args.output/'inputs.json')
    for p, h in {**frozen['reference']['files'], **frozen['images']['files']}.items():
        if sha(Path(p)) != h: raise ValueError('Frozen source changed')
    assets = frozen['images']['assets']
    for a in assets.values():
        if sha(Path(a['path'])) != a['sha256']: raise ValueError('PNG changed')
    binding = reader_binding(args.reader)
    if binding != frozen['reference']['reader']: raise ValueError('Reader identity changed')
    rows = population()[0][args.shard::2]; donors, groups = donors_and_groups()
    identity = dict(inputs_sha256=sha(args.output/'inputs.json'), source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(PROTOCOL), assignment=[r['base_pair_id'] for r in rows])
    ident = args.output/f'identity-{args.shard}.json'; save_once(ident, identity)
    dest = args.output/f'readout-{args.shard}.jsonl'; receipts_path = args.output/f'targets-{args.shard}.json'
    target_meta = read(receipts_path) if receipts_path.exists() else {}
    targets = {**frozen['reference']['targets'], **target_meta}
    records = read_lines(dest) if dest.exists() else []
    expected = current_keys(identity['assignment'], spec)
    keys = [tuple(r[k] for k in KEYS) for r in records]
    if len(set(keys)) != len(keys) or not set(keys) <= expected: raise ValueError('Duplicate/foreign partial rows')
    for r in records: validate_row(r, targets, assets, spec)
    keys = set(keys)
    processor, reader = load_reader(args.reader, args.device)
    eos = assistant_termination_contract(reader, processor)['assistant_end_token_id']
    save_once(args.output/f'termination-{args.shard}.json', dict(eos=eos, pad=processor.tokenizer.pad_token_id))
    gray = torch.full((3, 1024, 1024), 128/255, dtype=torch.float32, device=args.device)
    with torch.no_grad():
        for row in rows:
            pid = row['base_pair_id']
            for q in spec['queries']:
                tk = f'{pid}|{q["id"]}'
                if groups[pid] == 'original16':
                    meta = frozen['reference']['targets'][tk]
                    target = load_target(meta, binding, row, q['query'], eos, processor.tokenizer.pad_token_id)
                else:
                    target, path = get_history_target(cache_root=args.output/'targets', binding=binding, row=row,
                        query=q['query'], model=reader, processor=processor, reference=gray, device=args.device,
                        assistant_end_token_id=eos)
                    meta = dict(path=str(path), file_sha256=sha(path), target_ids=target['target_ids'], logits_sha256=target['logits_sha256'])
                    if tk in target_meta and target_meta[tk] != meta: raise ValueError('New target receipt changed')
                    target_meta[tk] = meta
                    write_json(receipts_path, target_meta)
                teacher = target['logits'].to(args.device); out = None
                for key in sorted(current_keys([pid], {'queries': [q]})):
                    if key in keys: continue
                    _, _, endpoint, control, chain = key
                    donor = donors[pid] if control == 'mismatch' else None
                    asset = assets[f'{endpoint}|{donor or pid}|{chain}'] if control in ('memory', 'mismatch') else None
                    pixels = gray if asset is None else read_png(asset['path']).to(args.device)
                    query = history_teacher_query(row, q['query']) if control == 'text' else q['query']
                    out = qwen3vl_continuation_logits(model=reader, processor=processor, image=pixels, query=query,
                        target_ids=torch.tensor([target['target_ids']]), device=args.device, require_image_grad=False,
                        reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
                    kl = float(soft_target_kl_divergence(out.target_logits, teacher))
                    if not math.isfinite(kl): raise ValueError('Nonfinite KL')
                    value = dict(zip(KEYS, key)); value.update(split='pilot', family=q['family'], kl=kl, png=asset,
                        donor_pair_id=donor, teacher_target=meta['path'], teacher_logits_sha256=target['logits_sha256'], target_ids=target['target_ids'])
                    append(dest, value); keys.add(key)
                del teacher, target, out
            print(json.dumps(dict(completed=pid, shard=args.shard, rows=len(keys))), flush=True)
    if keys != expected: raise ValueError('Incomplete denominator')
    save_once(args.output/f'finished-{args.shard}.json', dict(rows=len(keys), identity_sha256=sha(ident),
        readout_sha256=sha(dest), targets_sha256=sha(receipts_path)))


def report(output, *, verify_tensors=True):
    spec = read(PROTOCOL); reused, ref = load_reused(spec); frozen = read(output/'inputs.json')
    if frozen['reference'] != ref or verify_images(output) != frozen['images']: raise ValueError('Frozen inputs changed')
    for p, h in read(output/'source.json').items():
        if sha(Path(p)) != h: raise ValueError('Training source changed')
    targets = dict(ref['targets']); new = []; rows = population()[0]; added = {r['base_pair_id'] for r in strata()['added16']}
    for shard in range(2):
        path, ident = output/f'readout-{shard}.jsonl', output/f'identity-{shard}.json'
        target_path = output/f'targets-{shard}.json'; meta = read(target_path)
        done = read(output/f'finished-{shard}.json'); identity = read(ident); records = read_lines(path)
        if (done != dict(rows=len(records), identity_sha256=sha(ident), readout_sha256=sha(path), targets_sha256=sha(target_path))
                or identity != dict(inputs_sha256=sha(output/'inputs.json'), source_sha256=sha(Path(__file__)),
                    protocol_sha256=sha(PROTOCOL), assignment=[r['base_pair_id'] for r in rows[shard::2]])):
            raise ValueError('Readout receipt changed')
        assigned_added = {r['base_pair_id'] for r in rows[shard::2]} & added
        if set(meta) != {f'{p}|{q["id"]}' for p in assigned_added for q in spec['queries']} or set(meta) & set(targets):
            raise ValueError('Wrong target denominator or cache overlap')
        for m in meta.values():
            if Path(m['path']).parent.resolve() != (output/'targets').resolve(): raise ValueError('Target outside own output')
        targets.update(meta)
        index_rows(records, current_keys(identity['assignment'], spec)); new.extend(records)
    if len(list((output/'targets').glob('*.pt'))) != 192: raise ValueError('New target cache denominator changed')
    termination = read(output/'termination-0.json')
    if termination != read(output/'termination-1.json'): raise ValueError('Reader termination changed')
    for r in [*reused, *new]: validate_row(r, targets, frozen['images']['assets'], spec)
    if verify_tensors:
        for row in rows:
            for q in spec['queries']:
                load_target(targets[f'{row["base_pair_id"]}|{q["id"]}'], ref['reader'], row, q['query'], termination['eos'], termination['pad'])
    result = summarize([*reused, *new], spec)
    save_once(output/'comparison.json', result)
    return result


def build_jobs(args):
    return [dict(name='rollout-'+e, gpu=i, command=[sys.executable, 'scripts/experiments/prefeval_k1_writer.py',
        'rollout', '--arm', 'B', '--split', 'pilot', '--ids-file', str(ADDED_IDS if e == 'writer16' else ALL_IDS),
        '--base', str(args.base), '--official-source', str(args.official_source), '--checkpoint', str(checkpoints()[e][0]),
        '--output', str(args.output/e/'images'), '--inter-turns', '0', '--noise-chains', '2']) for i, e in enumerate(ENDPOINTS)]


def main(args):
    if args.output.resolve() != RUN/'context-population-readout-v1': raise ValueError('Require isolated output')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(): raise ValueError('Dirty source')
    args.output.mkdir(parents=True, exist_ok=True); claim = args.output/'active-owner'; claim.mkdir()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        write_json(claim/'owner.json', dict(pid=os.getpid(), host=socket.gethostname(), started=time.time(), commit=commit))
        for name in ('attempts', 'receipts', 'logs'): (args.output/name).mkdir(exist_ok=True)
        save_once(args.output/'plan.json', dict(commit=commit, protocol_sha256=sha(PROTOCOL),
            plan_sha256=sha(ROOT/'reports/context-coverage-20261006/CONTEXT_POPULATION_PLAN.md'),
            source=str(SOURCE), base=str(args.base), reader=str(args.reader), official_source=str(args.official_source),
            strata={s: [r['base_pair_id'] for r in rs] for s, rs in strata().items()},
            iteration_cap_gpu_hours=3, campaign_cap_gpu_hours=16, first_phase_cost_must_be_subtracted=True,
            new_rows=2688, reused_rows=1152, combined_rows=3840))
        save_once(args.output/'source.json', verify_sources(args.prefeval))
        _, ref = load_reused(read(PROTOCOL))
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(): raise ValueError('GPU occupied')
        if len(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()) != 2: raise ValueError('Require two GPUs')
        if [p for p in RUN.glob('*/active-owner/owner.json') if p != claim/'owner.json']: raise ValueError('Another owner')
        left, previous, first, _ = remaining_seconds(args.output)
        if left <= 60: raise ValueError('Budget exhausted')
        deadline = time.monotonic() + min(6*3600, (left-60)/2)
        def group(jobs):
            if time.monotonic()+30 >= deadline: raise ValueError('No time for next phase')
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                for f in [pool.submit(execute, j, args.output, deadline) for j in jobs]: f.result()
        write_json(args.output/'status.json', dict(status='rollout', time=time.time(), commit=commit))
        group(build_jobs(args))
        save_once(args.output/'inputs.json', dict(reference=ref, images=verify_images(args.output)))
        write_json(args.output/'status.json', dict(status='readout', time=time.time(), commit=commit))
        group([dict(name=f'evaluate-{i}', gpu=i, command=[sys.executable, str(Path(__file__)), 'evaluate',
            '--output', str(args.output), '--reader', str(args.reader), '--shard', str(i)]) for i in range(2)])
        result = report(args.output)
        current = accrued_seconds(args.output)
        write_json(args.output/'status.json', dict(status='completed', time=time.time(), rows=result['combined_rows'],
            gpu_hours=current/3600, iteration_gpu_hours=(first+current)/3600, campaign_gpu_hours=(previous+first+current)/3600))
    except BaseException as exc:
        write_json(args.output/'status.json', dict(status='failed', time=time.time(), error=str(exc))); raise
    finally:
        (claim/'owner.json').unlink(missing_ok=True); claim.rmdir()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('run', 'evaluate'))
    for name in ('output', 'base', 'reader', 'official-source', 'prefeval'): p.add_argument('--'+name, type=Path, required=name in ('output', 'reader'))
    p.add_argument('--shard', type=int, choices=(0, 1)); p.add_argument('--device', default='cuda:0')
    args = p.parse_args(); (main if args.phase == 'run' else evaluate)(args)
