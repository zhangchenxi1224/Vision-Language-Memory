"""Check portable first-step evidence against all independent sealed geometry cells."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.reporting.collect_first_step_condition import ARMS,COMMIT,RESULT,BANK,CHECKPOINT,summarize


def verify(archive,digest,geometry_path):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:
        raise ValueError('Portable archive differs from observed remote SHA')
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        if len(members)!=4 or {m.name for m in members}!={'identity.json','complete.json','cells.jsonl','verified-summary.json'} or not all(m.isfile() for m in members):
            raise ValueError('Unexpected portable archive member')
        data={m.name:tar.extractfile(m).read() for m in members}
    identity,complete,summary=(json.loads(data[name]) for name in ('identity.json','complete.json','verified-summary.json'))
    if (summary['complete_sha256']!=hashlib.sha256(data['complete.json']).hexdigest()
            or identity!=complete['identity'] or identity!=summary['identity']
            or identity['source_commit']!=COMMIT or identity['parent_result_sha256']!=RESULT
            or identity['checkpoint_sha256']!=CHECKPOINT or identity['bank_sha256']!=BANK
            or identity['arms']!=list(ARMS) or identity['optimizer_updates']!=0
            or not summary['all302_velocity_tensors_recomputed_on_cpu']
            or not summary['parent_302_trajectories_and_checkpoint_verified']):
        raise ValueError('Identity, source or fixed parent differs')
    for name in ('identity.json','cells.jsonl'):
        if hashlib.sha256(data[name]).hexdigest()!=complete['artifact_hashes'][name]:
            raise ValueError('Portable file changed')
    if hashlib.sha256(geometry_path.read_bytes()).hexdigest()!='2cdb98ca19fcb7f45f595cfff1ce278c1d9b5bddf832e0909f3074d04029828e':
        raise ValueError('Independent complete geometry evidence changed')
    geometry=json.loads(geometry_path.read_bytes())
    prior={(v['question_id'],v['noise_seed']):v for v in geometry['phases']['trained']['cells']}
    logged=[json.loads(line) for line in data['cells.jsonl'].splitlines()]
    captured={(v['question_id'],v['noise_seed']):v for v in logged}
    cells=summary['cells']
    if len(cells)!=302 or len(logged)!=302 or len(captured)!=302 or set(captured)!=set(prior) or complete['cells']!=302:
        raise ValueError('Missing or duplicated condition/noise cell')
    seen=set()
    for cell in cells:
        key=(cell['question_id'],cell['noise_seed'])
        if key not in prior or key in seen:raise ValueError('Unexpected verified cell')
        seen.add(key)
        reference=prior[key]
        if (cell['parent_artifact_sha256']!=reference['artifact_file_sha256']
                or cell['target_sha256']!=reference['intended_target_sha256']
                or cell['development_correct_eos']!=reference['correct_eos_out_of_five']
                or not cell['native_first_update_bitwise_equal'] or not cell['native_and_raw_cpu_euler_bitwise_equal']):
            raise ValueError('Cell differs from independent development evidence')
        for field,value in captured[key].items():
            if field!='velocity_mse_against_noise_minus_target' and cell[field]!=value:
                raise ValueError('Portable verified row differs from actual probe row')
        for arm in ARMS:
            value=cell['velocity_mse_against_noise_minus_target'][arm]
            if not math.isfinite(value) or value<0 or not math.isclose(value,captured[key]['velocity_mse_against_noise_minus_target'][arm],rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError('Recorded and CPU-recomputed losses differ')
    if summarize(cells)!=summary['partitions']:
        raise ValueError('All-cell aggregation differs')
    return {'archive_sha256':digest,'cells_verified':302,'partitions':summary['partitions'],
        'scope':'All portable cells joined to independent sealed development geometry and loss partitions recounted. Full velocity/PT/Euler recomputation happened remotely on CPU, not locally.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    value=verify(a.archive,a.sha256,a.geometry)
    a.output.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(value['partitions']))
