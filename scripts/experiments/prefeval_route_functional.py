"""Read-only PM-to-FM route diagnostic using completed PNGs and teacher cache."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.experiments.prefeval_writer_readout import (
    ENDPOINTS as OLD_ENDPOINTS, cohort_rows, read, save_once, expected_keys as old_keys,
)
from scripts.experiments.prefeval_k1_data import sha, event_text
from scripts.experiments.prefeval_context_readout import validate_protocol
from scripts.reporting.writer_readout_report import summarize as old_summary
from scripts.reporting.context_coverage_report import paired_interval

KEYS = ('pair_id', 'query_id', 'endpoint', 'control', 'chain')
SEEDS = ('20261005', '20261006')
NEW_ENDPOINTS = tuple('pm-fm-' + s for s in SEEDS)
CHECKPOINTS = dict(zip(SEEDS, (
    'd1956bd2009ebe9e39601e992a5ee53765a194d1fb20a0d8e757749a1efd41e8',
    '647243c2865b440fabdc8c9e69fb4b01fe89a05cb061ff33682c9c078eda2091')))
UPSTREAM = Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/runs/dreamlite-aris-20261005')
VARIANT_SHA = '12ac93a5db22d528b10f2165d701e9c7fa2aec33920699cc2f64218d0a300dd7'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def new_keys(ids, spec, endpoints=NEW_ENDPOINTS):
    return {(p, q['id'], e, c, n) for p in ids for q in spec['queries']
            for e in endpoints for c in ('memory', 'mismatch') for n in range(2)}


def index_rows(rows, expected):
    table = {}
    for row in rows:
        key = tuple(row[k] for k in KEYS)
        if key in table or not math.isfinite(row['kl']):
            raise ValueError('Duplicate or nonfinite readout')
        table[key] = row
    if set(table) != expected:
        raise ValueError('Incomplete or foreign denominator')
    return table


def donor_map(rows):
    peers = defaultdict(list)
    for r in rows:
        peers[r['topic']].append(r['base_pair_id'])
    if any(len(p) < 2 for p in peers.values()):
        raise ValueError('No same-topic donor')
    return {p: group[(i+1) % len(group)] for group in peers.values() for i,p in enumerate(group)}


def read_lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]


def load_reference(reference, spec, protocol, cohort='dev'):
    """Authenticate the old full run before filtering its dev rows; never write it."""
    all_rows, files, identities = [], {}, []
    for shard in range(2):
        done = read(reference / f'finished-{shard}.json')
        source = reference / f'readout-{shard}.jsonl'
        ident = reference / f'identity-{shard}.json'
        if (done['readout_sha256'] != sha(source) or done['identity_sha256'] != sha(ident)
                or read(ident)['protocol_sha256'] != sha(protocol)):
            raise ValueError('Reference receipt/protocol changed')
        records = read_lines(source)
        if len(records) != done['rows']:
            raise ValueError('Reference shard denominator changed')
        all_rows.extend(records)
        identities.append(read(ident))
        for p in (source, ident, reference / f'finished-{shard}.json'):
            files[str(p)] = sha(p)
    groups = {s:[r['base_pair_id'] for r in rs] for s,rs in cohort_rows().items()}
    if old_summary(all_rows, groups, spec) != read(reference / 'comparison.json'):
        raise ValueError('Reference report recomputation mismatch')
    if identities[0]['reader'] != identities[1]['reader']:
        raise ValueError('Reference Reader mismatch')
    files[str(reference/'comparison.json')] = sha(reference/'comparison.json')
    dev_ids = set(groups[cohort])
    dev = [r for r in all_rows if r['pair_id'] in dev_ids]
    table = index_rows(dev, old_keys(dev_ids, spec['queries']))
    targets = {}
    for pid in groups[cohort]:
        for q in spec['queries']:
            row = table[pid,q['id'],'text','text',0]
            path = Path(row['teacher_target'])
            if path.parent.resolve() != (reference/'targets').resolve() or not path.is_file():
                raise ValueError('Reference teacher cache missing or outside frozen directory')
            targets[f'{pid}|{q["id"]}'] = dict(path=str(path), file_sha256=sha(path),
                target_ids=row['target_ids'], logits_sha256=row['teacher_logits_sha256'])
    # Validate old assets once, not once per repeated query/control row.
    old_assets = {r['png']['path']:r['png']['sha256'] for r in dev if r.get('png')}
    for path, checksum in old_assets.items():
        if sha(Path(path)) != checksum:
            raise ValueError('Reference PNG changed')
    return dev, dict(files=files, reader=identities[0]['reader'], targets=targets, old_assets=old_assets)


def verify_assets(upstream=UPSTREAM):
    from PIL import Image
    from vision_memory.training.latent_bank_unet import stable_seed
    assets, files = {}, {}
    for seed, endpoint in zip(SEEDS, NEW_ENDPOINTS):
        train = upstream/'route-comparison-v1'/('seed-'+seed)
        checkpoint, receipt = train/'checkpoint-final.pt', train/'complete.json'
        done = read(receipt)
        if (done['status'] != 'completed' or done['steps'] != 128 or done['targets'] != 512
                or done['checkpoint_sha256'] != CHECKPOINTS[seed] or sha(checkpoint) != CHECKPOINTS[seed]
                or done['optimization_sha256'] != sha(train/'optimization.jsonl')
                or [r['step'] for r in read_lines(train/'optimization.jsonl')] != list(range(1,129))):
            raise ValueError('Upstream endpoint is not the frozen completed endpoint')
        out = upstream/'route-readback-v1/pm-fm'/('seed-'+seed)/'V0'
        images = out/'images'
        manifest = read(images/'manifest.json')
        expected = dict(checkpoint_sha256=CHECKPOINTS[seed], split='dev', steps=28, cfg=1,
            noise_chains=2, inter_turns=0, state='only reopened uint8 RGB PNG; fresh Gaussian each write',
            initial_variants_sha256=VARIANT_SHA, initial_variant=0)
        validated = read(out/'images-validated.json')
        if manifest != expected or validated != dict(pngs=180, manifest_sha256=sha(images/'manifest.json')):
            raise ValueError('Upstream PNGs not validated under the frozen V0 protocol')
        for p in (checkpoint, receipt, images/'manifest.json', out/'images-validated.json'):
            files[str(p)] = sha(p)
        if len(list(images.glob('*/seed-*/complete.json'))) != 180:
            raise ValueError('Incomplete or foreign PNG denominator')
        for row in cohort_rows()['dev']:
            pid = row['base_pair_id']
            for chain in range(2):
                folder = images/pid.replace(':','_')/f'seed-{chain}'
                png = folder/'prefix-00.png'
                checksum = sha(png)
                if read(folder/'complete.json') != dict(binding=manifest, png_hashes={png.name:checksum}):
                    raise ValueError('PNG binding or SHA mismatch')
                if read_lines(folder/'writes.jsonl') != [dict(position=0, source_png_sha256=None,
                        output_png_sha256=checksum, noise_seed=stable_seed(20260924,f'rollout:{pid}:{chain}',0),
                        event=event_text(row['history'][:2]))]:
                    raise ValueError('Unpaired event or noise')
                with Image.open(png) as image:
                    if image.mode != 'RGB' or image.size != (1024,1024):
                        raise ValueError('Invalid PNG format')
                    image.load()
                assets[f'{endpoint}|{pid}|{chain}'] = dict(path=str(png),sha256=checksum,
                    complete_sha256=sha(folder/'complete.json'))
    return dict(assets=assets, files=files)


def validate_new_row(row, targets, assets, donors, queries, split='dev'):
    pid, qid, endpoint, control, chain = (row[k] for k in KEYS)
    t = targets[f'{pid}|{qid}']
    source = donors[pid] if control == 'mismatch' else pid
    expected_donor = donors[pid] if control == 'mismatch' else None
    if (row['split'] != split or row['family'] != queries[qid]['family']
            or row['donor_pair_id'] != expected_donor or row['png'] != assets[f'{endpoint}|{source}|{chain}']
            or row['teacher_target'] != t['path'] or row['target_ids'] != t['target_ids']
            or row['teacher_logits_sha256'] != t['logits_sha256']):
        raise ValueError('Readout target, donor, or asset binding mismatch')


def load_target(meta, binding, row, query, eos, pad):
    import torch
    from scripts.experiments.prefeval_prompt_matching import target_cache_binding, _validate_cached_target
    path = Path(meta['path'])
    if not path.is_file() or sha(path) != meta['file_sha256']:
        raise ValueError('Read-only teacher target missing or changed')
    target = torch.load(path,map_location='cpu',weights_only=True)
    _validate_cached_target(target,target_cache_binding(binding,row,query),assistant_end_token_id=eos,pad_token_id=pad)
    if target['target_ids'] != meta['target_ids'] or target['logits_sha256'] != meta['logits_sha256']:
        raise ValueError('Teacher differs from original readout')
    return target


def summarize(old, new, ids, spec, endpoints=NEW_ENDPOINTS, references=None):
    table = index_rows(old, old_keys(ids,spec['queries']))
    table.update(index_rows(new,new_keys(ids,spec,endpoints)))
    for pid in ids:
        for q in spec['queries']:
            ref = table[pid,q['id'],'text','text',0]
            for endpoint in endpoints:
                for control in ('memory','mismatch'):
                    for chain in range(2):
                        r=table[pid,q['id'],endpoint,control,chain]
                        if any(r[k] != ref[k] for k in ('teacher_logits_sha256','target_ids','teacher_target')):
                            raise ValueError('Unpaired teacher distribution')
    result=dict(new_rows=len(new),reused_rows=len(old),combined_rows=len(table),independent_n=len(ids),
        metric=('Conditional teacher-prefix KL; trained-history fit only; no default promotion' if references else
                'Conditional teacher-prefix KL; exploratory internal dev; no default promotion'),families={})
    for family in ('recall','application','neutral'):
        qs=[q['id'] for q in spec['queries'] if q['family']==family]
        def values(e,c):
            ns=[0] if c in ('blank','text') else [0,1]
            return [sum(table[p,q,e,c,n]['kl'] for q in qs for n in ns)/(len(qs)*len(ns)) for p in ids]
        mean=lambda x:sum(x)/len(x)
        blank,text=values('blank','blank'),values('text','text')
        baseline=values('b730','memory')
        f=dict(blank_kl=mean(blank),text_self_consistency_kl=mean(text),endpoints={})
        for e in (*OLD_ENDPOINTS,*endpoints):
            mem,wrong=values(e,'memory'),values(e,'mismatch')
            v=dict(memory_kl=mean(mem),mismatch_kl=mean(wrong),
                baseline_minus_memory=paired_interval([a-b for a,b in zip(baseline,mem)]),
                mismatch_minus_memory=paired_interval([a-b for a,b in zip(wrong,mem)]),
                blank_minus_memory=paired_interval([a-b for a,b in zip(blank,mem)]))
            if e in endpoints:
                reference=references[e] if references else 'direct-'+e.rsplit('-',1)[1]
                direct=values(reference,'memory')
                name='paired_reference_minus_memory' if references else 'direct_minus_memory'
                v[name]=paired_interval([a-b for a,b in zip(direct,mem)])
                if references:
                    v['paired_reference']=reference
            f['endpoints'][e]=v
        result['families'][family]=f
    return result


def evaluate(args):
    import torch
    from scripts.experiments.prefeval_prompt_matching import supervision_binding
    from scripts.eval.prefeval_rgb import load_reader, read_png, append
    from vision_memory.reader.open_eos import assistant_termination_contract
    from vision_memory.reader.prompt_matching import qwen3vl_continuation_logits, soft_target_kl_divergence
    from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
    from vision_memory.repro import configure_strict_cuda_determinism
    configure_strict_cuda_determinism(0)
    torch.set_num_threads(1)
    spec=read(args.protocol)
    validate_protocol(spec)
    frozen=read(args.output/'inputs.json')
    cohort=frozen.get('cohort','dev')
    endpoints=tuple(frozen.get('endpoints',NEW_ENDPOINTS))
    if (cohort,endpoints) not in (('dev',NEW_ENDPOINTS),('pilot',('context-narrow','context-diverse'))):
        raise ValueError('Unregistered consumer scope')
    for p,h in {**frozen['reference']['files'],**frozen['upstream']['files']}.items():
        if sha(Path(p)) != h:
            raise ValueError('Frozen source changed')
    all_rows=[r for rs in cohort_rows().values() for r in rs]
    binding=supervision_binding(SimpleNamespace(supervision='prompt_matching',temperature=1.,reader=args.reader,
        teacher_max_new_tokens=512,teacher_cache=None,boundary_recovery_from=None),all_rows)
    if binding != frozen['reference']['reader']:
        raise ValueError('Reader identity changed')
    rows=cohort_rows()[cohort]
    donors=donor_map(rows)
    rows=rows[args.shard::2]
    assets=frozen['upstream']['assets']
    for asset in assets.values():
        if sha(Path(asset['path'])) != asset['sha256']:
            raise ValueError('PNG changed after preflight')
    identity=dict(inputs_sha256=sha(args.output/'inputs.json'),source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(args.protocol),assignment=[r['base_pair_id'] for r in rows])
    ident=args.output/f'identity-{args.shard}.json'
    save_once(ident,identity)
    dest=args.output/f'readout-{args.shard}.jsonl'
    completed=read_lines(dest) if dest.exists() else []
    expected=new_keys(identity['assignment'],spec,endpoints)
    keys=[tuple(r[k] for k in KEYS) for r in completed]
    if len(set(keys)) != len(keys) or not set(keys) <= expected:
        raise ValueError('Duplicate or foreign partial rows')
    queries={q['id']:q for q in spec['queries']}
    for r in completed:
        validate_new_row(r,frozen['reference']['targets'],assets,donors,queries,cohort)
        if not math.isfinite(r['kl']):
            raise ValueError('Nonfinite partial row')
    keys=set(keys)
    processor,reader=load_reader(args.reader,args.device)
    eos=assistant_termination_contract(reader,processor)['assistant_end_token_id']
    with torch.no_grad():
        for row in rows:
            pid=row['base_pair_id']
            for q in spec['queries']:
                meta=frozen['reference']['targets'][f'{pid}|{q["id"]}']
                target=load_target(meta,binding,row,q['query'],eos,processor.tokenizer.pad_token_id)
                teacher=target['logits'].to(args.device)
                for endpoint in endpoints:
                    for control in ('memory','mismatch'):
                        for chain in range(2):
                            key=(pid,q['id'],endpoint,control,chain)
                            if key in keys:
                                continue
                            donor=donors[pid] if control=='mismatch' else None
                            asset=assets[f'{endpoint}|{donor or pid}|{chain}']
                            out=qwen3vl_continuation_logits(model=reader,processor=processor,
                                image=read_png(asset['path']).to(args.device),query=q['query'],
                                target_ids=torch.tensor([target['target_ids']]),device=args.device,
                                require_image_grad=False,reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
                            kl=float(soft_target_kl_divergence(out.target_logits,teacher))
                            if not math.isfinite(kl):
                                raise ValueError('Nonfinite KL')
                            value=dict(zip(KEYS,key))
                            value.update(split=cohort,family=q['family'],kl=kl,png=asset,donor_pair_id=donor,
                                teacher_target=meta['path'],teacher_logits_sha256=target['logits_sha256'],target_ids=target['target_ids'])
                            append(dest,value)
                            keys.add(key)
                del teacher,target
            print(json.dumps(dict(completed=pid,shard=args.shard,rows=len(keys))),flush=True)
    if keys != expected:
        raise ValueError('Incomplete denominator')
    save_once(args.output/f'finished-{args.shard}.json',dict(rows=len(keys),identity_sha256=sha(ident),readout_sha256=sha(dest)))


def report(output, reference, protocol):
    spec=read(protocol)
    old,ref=load_reference(reference,spec,protocol)
    frozen=read(output/'inputs.json')
    if ref != frozen['reference'] or verify_assets() != frozen['upstream']:
        raise ValueError('Frozen input changed')
    new=[]
    rows=cohort_rows()['dev']
    donors=donor_map(rows)
    for shard in range(2):
        done=read(output/f'finished-{shard}.json')
        path,ident=output/f'readout-{shard}.jsonl',output/f'identity-{shard}.json'
        identity=read(ident)
        records=read_lines(path)
        if (done['readout_sha256'] != sha(path) or done['identity_sha256'] != sha(ident)
                or done['rows'] != len(records) or identity['inputs_sha256'] != sha(output/'inputs.json')
                or identity['protocol_sha256'] != sha(protocol) or identity['source_sha256'] != sha(Path(__file__))
                or identity['assignment'] != [r['base_pair_id'] for r in rows[shard::2]]):
            raise ValueError('Readout receipt/assignment mismatch')
        index_rows(records,new_keys(identity['assignment'],spec))
        for row in records:
            validate_new_row(row,ref['targets'],frozen['upstream']['assets'],donors,{q['id']:q for q in spec['queries']})
        new.extend(records)
    value=summarize(old,new,[r['base_pair_id'] for r in rows],spec)
    save_once(output/'comparison.json',value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reader',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--shard',type=int,choices=[0,1],required=True)
    p.add_argument('--device',default='cuda:0')
    evaluate(p.parse_args())
