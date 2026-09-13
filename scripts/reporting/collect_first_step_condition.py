"""Recompute all302 first-step velocity losses from sealed CPU-loaded tensors."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile
import torch

ARMS=('native_sigma1','training_raw_sigma1','training_raw_sigma0.999_integer999')
COMMIT='17f35be7baf9e63376103b1da5d5f3bc84da2af1'
RESULT='717522c160bfeffc13fd1789ef6b6e6694ed3d8778dbfd047eb6f10e5bd2f4ad'
BANK='c27cd65dab809deabb5f2cb08891517d3590244651d08a8c6763c84fea901592'
CHECKPOINT='737735c4d7d3483b38be2f88d8c48fbe3050b40c8f90c0d49b21e30336455616'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def summarize(cells):
    partitions=defaultdict(list)
    for cell in cells:
        partitions['all'].append(cell)
        partitions['development_all_five_correct' if cell['development_correct_eos']==5 else 'development_not_all_five_correct'].append(cell)
    return {key:{'cells':len(values),'arms':{arm:{'mean_mse':statistics.mean(v['velocity_mse_against_noise_minus_target'][arm] for v in values),
            'median_mse':statistics.median(v['velocity_mse_against_noise_minus_target'][arm] for v in values),
            'maximum_mse':max(v['velocity_mse_against_noise_minus_target'][arm] for v in values)} for arm in ARMS},
        'raw_mse_lower_than_native':sum(v['velocity_mse_against_noise_minus_target'][ARMS[1]]<v['velocity_mse_against_noise_minus_target'][ARMS[0]] for v in values)}
        for key,values in partitions.items()}


def collect(run,parent,bank_path):
    identity,complete,bank=read(run/'identity.json'),read(run/'complete.json'),read(bank_path)
    if (identity!=complete['identity'] or identity['source_commit']!=COMMIT or identity['cells_expected']!=302
            or complete['cells']!=302 or identity['arms']!=list(ARMS) or identity['optimizer_updates']!=0
            or identity['parent_result_sha256']!=RESULT or identity['checkpoint_sha256']!=CHECKPOINT
            or sha(parent/'train/result.json')!=RESULT or identity['bank_sha256']!=BANK or sha(bank_path)!=BANK
            or identity['runtime_sha256']!=sha(parent/'train/runtime.json')):
        raise ValueError('Registered parent, source, arms or full matrix changed')
    if sha(parent/'train/checkpoint-final.pt')!=CHECKPOINT:
        raise ValueError('Parent checkpoint changed')
    runtime=read(parent/'train/runtime.json')
    phase=parent/'train/trained'
    parent_seal=read(phase/'complete.json')
    if sha(phase/'generations.jsonl')!=parent_seal['artifact_hashes']['generations.jsonl']:
        raise ValueError('Parent Reader evidence changed')
    parent_rows=[json.loads(line) for line in (phase/'generations.jsonl').read_text().splitlines()]
    rows=[json.loads(line) for line in (run/'cells.jsonl').read_text().splitlines()]
    artifacts={'identity.json','cells.jsonl'}
    expected={hashlib.sha256(group['question_id'].encode()).hexdigest()[:16]+f'-seed-{index:02d}.pt':group
              for group in bank['groups'] for index in range(2)}
    artifacts.update(expected)
    if (len(expected)!=302 or len(rows)!=302 or set(complete['artifact_hashes'])!=artifacts
            or len({v['artifact'] for v in rows})!=302 or {v['artifact'] for v in rows}!=set(expected)):
        raise ValueError('Missing, duplicate or extra diagnostic artifact')
    for name,digest in complete['artifact_hashes'].items():
        if sha(run/name)!=digest:raise ValueError('Diagnostic artifact changed: '+name)
    teachers={v['teacher_id']:v for v in bank['teachers']}
    target_cache={}
    cells=[]
    for row in rows:
        name=row['artifact']
        group=expected[name]
        if len(group['teacher_ids'])!=1 or row['question_id']!=group['question_id'] or not row['native_first_update_bitwise_equal']:
            raise ValueError('Unexpected target condition or failed native reproduction')
        teacher=teachers[group['teacher_ids'][0]]
        target_path=Path(teacher['latent_path'])
        if not target_path.is_absolute():target_path=bank_path.parent/target_path
        if teacher['latent_sha256'] not in target_cache:
            if sha(target_path)!=teacher['latent_file_sha256']:raise ValueError('Target tensor changed')
            target=torch.load(target_path,map_location='cpu',weights_only=True)
            if teacher.get('latent_tensor_key'):target=target[teacher['latent_tensor_key']]
            target_cache[teacher['latent_sha256']]=target
        target=target_cache[teacher['latent_sha256']]
        if row['target_sha256']!=teacher['latent_sha256']:raise ValueError('Wrong target binding')
        if row['parent_artifact_sha256']!=parent_seal['artifact_hashes'][name] or sha(phase/name)!=row['parent_artifact_sha256']:
            raise ValueError('Parent trajectory changed')
        prior=torch.load(phase/name,map_location='cpu',weights_only=True)
        payload=torch.load(run/name,map_location='cpu',weights_only=True)
        if set(payload)!={'noise','native_next_state','raw_next_state',*ARMS}:
            raise ValueError('Unexpected velocity tensor payload')
        if any(t.shape!=target.shape or t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in payload.values()):
            raise ValueError('Invalid velocity tensor')
        noise=torch.randn(target.shape,generator=torch.Generator().manual_seed(row['noise_seed']),dtype=torch.float32)
        if (prior['noise_seed']!=row['noise_seed'] or prior['question_id']!=row['question_id']
                or not torch.equal(noise,payload['noise']) or not torch.equal(noise,prior['trajectory'][0])
                or not torch.equal(payload['native_next_state'],prior['trajectory'][1])):
            raise ValueError('Noise, condition or exact native first state differs')
        # Training constructs this label in FP32, then measures error in FP64.
        label=(noise-target).double()
        recomputed={arm:float((payload[arm].double()-label).square().mean()) for arm in ARMS}
        if set(row['velocity_mse_against_noise_minus_target'])!=set(ARMS) or any(
                not math.isclose(recomputed[arm],row['velocity_mse_against_noise_minus_target'][arm],rel_tol=1e-10,abs_tol=1e-12) for arm in ARMS):
            raise ValueError('Complete CPU velocity loss recount differs')
        sigmas=runtime['effective_inference_sigmas'][row['question_id']]
        if len(sigmas)!=28 or sigmas[0]!=1.:raise ValueError('Native schedule differs')
        for arm,state_name in ((ARMS[0],'native_next_state'),(ARMS[1],'raw_next_state')):
            predicted=noise+(sigmas[1]-sigmas[0])*payload[arm]
            if not torch.equal(predicted,payload[state_name]):
                raise ValueError('Stored first update differs from CPU FP32 Euler reconstruction')
        matched=[v for v in parent_rows if v['condition']=='matched' and v['question_id']==row['question_id'] and v['noise_seed']==row['noise_seed']]
        if len(matched)!=5 or len({v['prompt_id'] for v in matched})!=5:
            raise ValueError('Incomplete parent development queries')
        passed=sum(v['generated_token_ids']==v['scorer']['gold_token_ids']+[151645] for v in matched)
        cells.append({**row,'development_correct_eos':passed,
            'velocity_mse_against_noise_minus_target':recomputed,'native_and_raw_cpu_euler_bitwise_equal':True})
    return {'complete_sha256':sha(run/'complete.json'),'identity':identity,'cells':cells,'partitions':summarize(cells),
        'all302_velocity_tensors_recomputed_on_cpu':True,'parent_302_trajectories_and_checkpoint_verified':True,
        'scope':'All three302-cell velocity arrays and native/raw Euler updates recomputed on CPU. These are fixed-condition first-step errors, not new Reader accuracy or a usable-model claim.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','parent','bank','output_prefix'):p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    result=collect(a.run,a.parent,a.bank)
    summary=Path(str(a.output_prefix)+'-summary.json')
    with summary.open('x') as f:json.dump(result,f,indent=2,sort_keys=True)
    archive=Path(str(a.output_prefix)+'-evidence.tgz')
    with tarfile.open(archive,'x:gz') as tar:
        for name in ('identity.json','complete.json','cells.jsonl'):tar.add(a.run/name,arcname=name)
        tar.add(summary,arcname='verified-summary.json')
    print(json.dumps({'partitions':result['partitions'],'summary_sha256':sha(summary),'archive_sha256':sha(archive)}))
