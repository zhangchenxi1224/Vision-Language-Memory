import collections
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

base = Path(__file__).resolve().parent
receipt = json.loads((base/'archive-receipt.json').read_text())
assert hashlib.sha256((base/'raw.tar.gz').read_bytes()).hexdigest() == receipt['sha256']
with tarfile.open(base/'raw.tar.gz') as tar:
    content = {m.name:tar.extractfile(m).read() for m in tar if m.isfile()}
    def data(name):
        return content[name]
    summaries = json.loads(data('summaries.json'))
    hashes = {r['path']:r for r in json.loads(data('actual-png-hashes.json'))}
    assert len(hashes)==5840 and len(summaries)==4
    families = ['T1']
    for variant in [0,1]:
        prior = None
        for step in [23360,93440]:
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
    assert checkpoint['endpoint_step'] == checkpoint['committed_step'] == 93440
    assert checkpoint['final_inference_equals_full_resume']
    assert checkpoint['endpoint_exposure_total'] == 512
    logs = [json.loads(line) for line in data('optimization-committed.jsonl').splitlines()]
    assert [r['step'] for r in logs]==list(range(1,checkpoint['committed_step']+1))
    counts = collections.Counter((d['pair_id'],d['initial_variant']) for r in logs if r['step']<=93440 for d in r['draws'])
    assert len(counts)==1460 and set(counts.values())=={256}
    final = json.loads(data('results.json'))
    done = json.loads(data('complete.json'))
    audit = json.loads(data('evidence/final-completion-audit-primary-only.json'))
    expected = {'readback/step-023360-train-V0/summary.json'} | {f'readback/step-{s:06d}-{split}-V{v}/summary.json' for s in [46720,70080,93440] for split in ['train','pilot','dev'] for v in [0,1]}
    assert set(final['evaluations']) == set(audit['summaries']) == expected and len(expected)==19
    assert done['status']=='training_and_registered_evaluation_complete'
    assert final['final_step']==93440 and final['exposures_per_preference']==512 and final['exposures_each_variant']==256
    assert hashlib.sha256(data('results.json')).hexdigest()==audit['root_results_sha256']
    assert hashlib.sha256(data('complete.json')).hexdigest()==audit['root_complete_sha256']
    for name in sorted(expected):
        body = data(name.replace('readback/','registered-summaries/',1))
        assert json.loads(body)==final['evaluations'][name]
        assert hashlib.sha256(body).hexdigest()==audit['summaries'][name]['sha256']
    assert audit['no_live_task_processes'] and audit['all_local_task_locks_free'] and audit['gpu_compute_processes_empty']
    assert audit['training_os_exit_code'] is None and audit['launcher_exit_code']==0
    assert audit['authorized_instance_to_stop']=='prefeval-b-cr-h200x4-20260926' and not audit['reader_host_inspected']
out = dict(archive_sha256=receipt['sha256'],readback_records=17520,png_hashes=5840,all_summaries_recomputed=True,all_paired_changes_vs23360_recomputed=True,complete_unique_denominators=True,readback_png_hashes_match_remote_actual_hashes=True,complete_png_metadata_matches=True,endpoint_step=93440,endpoint_all_730_each_variant_exposures=256,committed_step=checkpoint['committed_step'],training_log_contiguous_unique=True)
out.update(registered_summaries=19,all_registered_summary_hashes_verified=True,root_results_and_complete_hashes_verified=True,primary_only_release_audit_verified=True)
(base/'local-verification.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
for name in ['summaries.json','actual-png-hashes.json','checkpoint-evidence.json','archive-script.py','results.json','complete.json','reader-complete.json','launcher-exit-status.txt','reader-launcher-exit-status.txt','evidence/final-completion-audit-primary-only.json'] + sorted(n for n in content if n.startswith('registered-summaries/')):
    target=base/name
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(content[name])
print(json.dumps(out,indent=2))
