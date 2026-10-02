import collections
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

base = Path(__file__).resolve().parent
receipt = json.loads((base/'archive-receipt.json').read_text())
assert hashlib.sha256((base/'raw.tar.gz').read_bytes()).hexdigest() == receipt['sha256']
with tarfile.open(base/'raw.tar.gz') as tar:
    def data(name):
        return tar.extractfile(name).read()
    summaries = json.loads(data('summaries.json'))
    hashes = {r['path']:r for r in json.loads(data('actual-png-hashes.json'))}
    assert len(hashes)==5840 and len(summaries)==4
    families = ['T1']
    for variant in [0,1]:
        prior = None
        for step in [23360,46720]:
            label = f'step-{step:06d}-train-V{variant}'
            body = data(f'{label}/readback-0.jsonl')
            summary = summaries[label]
            assert hashlib.sha256(body).hexdigest()==summary['files']['readback-0.jsonl']
            rows = [json.loads(line) for line in body.splitlines()]
            keyed = {(r['pair_id'],r['chain'],r['control'],r['family']):r for r in rows}
            ids = {r['pair_id'] for r in rows}
            assert len(rows)==len(keyed)==4380 and len(ids)==730
            assert set(keyed)=={(p,c,k,f) for p in ids for f in families for k in ['memory','mismatch','blank','text'] for c in (range(2) if k in ['memory','mismatch'] else range(1))}
            for row in rows:
                assert row['correct']==(row['predicted_letter']==row['correct_letter'])
                if row['control'] in ['memory','mismatch']:
                    assert row['png_sha256']==hashes[row['png_path']]['sha256']
            for family in families:
                result = summary['families'][family]
                for control in ['memory','mismatch','blank','text']:
                    subset = [r for r in rows if r['control']==control and r['family']==family]
                    assert result['controls'][control]==dict(correct=sum(r['correct'] for r in subset),total=len(subset),parse_failure=sum(r['parse_failure'] for r in subset),truncated=sum(r['generated']['truncated'] for r in subset))
                pairs = [(keyed[p,c,'memory',family]['correct'],keyed[p,c,'mismatch',family]['correct']) for p in ids for c in range(2)]
                assert result['match_minus_mismatch']==sum(a-b for a,b in pairs)
                assert result['repaired']==sum(a and not b for a,b in pairs)
                assert result['regressed']==sum(b and not a for a,b in pairs)
                assert result['both_noise_correct']==sum(all(keyed[p,c,'memory',family]['correct'] for c in range(2)) for p in ids)
                if prior:
                    assert result['versus_step23360']==dict(fixed=sum(keyed[p,c,'memory',family]['correct'] and not prior[p,c,'memory',family]['correct'] for p in ids for c in range(2)),lost=sum(prior[p,c,'memory',family]['correct'] and not keyed[p,c,'memory',family]['correct'] for p in ids for c in range(2)))
            prior = keyed
    for rec in hashes.values():
        p = PurePosixPath(rec['path'])
        complete = json.loads(data(f"png-metadata/{rec['label']}/{p.parent.parent.name}/{p.parent.name}/complete.json"))
        assert complete['png_hashes'][p.name]==rec['sha256']
    checkpoint = json.loads(data('checkpoint-evidence.json'))
    logs = [json.loads(line) for line in data('optimization-committed.jsonl').splitlines()]
    assert [r['step'] for r in logs]==list(range(1,checkpoint['committed_step']+1))
    counts = collections.Counter((d['pair_id'],d['initial_variant']) for r in logs if r['step']<=46720 for d in r['draws'])
    assert len(counts)==1460 and set(counts.values())=={128}
out = dict(archive_sha256=receipt['sha256'],readback_records=17520,png_hashes=5840,all_summaries_recomputed=True,all_paired_changes_vs23360_recomputed=True,complete_unique_denominators=True,readback_png_hashes_match_remote_actual_hashes=True,complete_png_metadata_matches=True,endpoint_step=46720,endpoint_all_730_each_variant_exposures=128,committed_step=checkpoint['committed_step'],training_log_contiguous_unique=True)
(base/'local-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
