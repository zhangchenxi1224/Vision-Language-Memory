"""Build three same-question state targets with the proven FP32 oracle recipe."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.probes.official_writer_confirmation import EVENT
TRAIN_PROMPTS=('original_open','paraphrase_1','paraphrase_2')
BANK_SHA='20ef4a9fc53b254fd99b12cbc01cf1a6d41dee8d04dd3120c70ecaa141f30722'
STATES=(('ambient',EVENT,'ambient'),('jazz',EVENT.replace('ambient.','jazz.'),'jazz'),
        ('clear','R3 Train Standard Templates 03: Clear the saved music preference for the indigo desk train 001123.','no active preference'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-commit',required=True)
    p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    from scripts.train import train_latent_bank_unet as train
    from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    from vision_memory.training.latent_bank_unet import load_teacher_bank,file_sha256,member_split
    command=json.loads((a.parent_run/'commands.json').read_text())['commands'][-1]
    args=train.parser().parse_args(command[3:])
    if file_sha256(args.bank_manifest)!=BANK_SHA:
        raise ValueError('The fixed original teacher bank changed')
    bank,tensors=load_teacher_bank(args.bank_manifest)
    if not a.worker:
        env={**os.environ,**snapshot_environment(bank),**REQUIRED_DETERMINISM_ENV,
             'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
        return subprocess.call([sys.executable,'-u',str(Path(__file__).resolve()),'--parent-run',str(a.parent_run),
            '--output',str(a.output),'--expected-commit',a.expected_commit,'--deadline-unix',str(a.deadline_unix),'--worker'],env=env)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if commit!=a.expected_commit or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Oracle construction requires its exact clean source commit')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    rows=[r.split(',') for r in gpu.strip().splitlines()]
    if len(rows)!=2 or any('H200' not in r[0] or int(r[1])<140000 or int(r[2])>100 for r in rows):
        raise RuntimeError('Two idle full H200s are required')
    import torch
    from scripts.train.official_base_runtime import load_base_runtime,audit_inference_only_runtime
    from scripts.experiments import direct_geometry_eos_training as oracle
    from vision_memory.repro import configure_strict_cuda_determinism,canonical_tensor_sha256
    configure_strict_cuda_determinism(0)
    group=bank['groups'][0]
    if len(bank['groups'])!=1 or group['event_text']!=EVENT:
        raise ValueError('Unexpected source question/event')
    tid=member_split(group['teacher_ids'])[0][0]
    member=next(t for t in bank['teachers'] if t['teacher_id']==tid)
    if not tid.endswith('e910e9b89ed3861e'):
        raise ValueError('Hash-selected ambient teacher changed')
    provenance=member['source_runs'][0]
    old=Path(provenance['source_run'])
    bindings=provenance['artifact_bindings']
    for filename,key in [('manifest.json','manifest_sha256'),('latent_index.jsonl','latent_index_sha256')]:
        if file_sha256(old/filename)!=bindings[key]:
            raise RuntimeError('The proven oracle initialization provenance changed')
    old_manifest=json.loads((old/'manifest.json').read_text())
    if (old_manifest['training_prompts']!=list(TRAIN_PROMPTS)
            or old_manifest['prompt_schedule']!='round_robin_zero_based'):
        raise ValueError('Original teacher was not the proven three-prompt recipe')
    index=[json.loads(x) for x in (old/'latent_index.jsonl').read_text().splitlines()]
    entry=next(x for x in index if x['optimizer_step']==0)
    initial_path=old/entry['path']
    if file_sha256(initial_path)!=entry['file_sha256']:
        raise RuntimeError('Original initialization file changed')
    initial=torch.load(initial_path,map_location='cpu',weights_only=True)['latent_fp32']
    if (canonical_tensor_sha256(initial)!=entry['latent_sha256']
            or canonical_tensor_sha256(initial)!=old_manifest['initial_latent_sha256']):
        raise RuntimeError('Original initialization tensor changed')
    a.output.mkdir(parents=True,exist_ok=False)
    args.colocate_models=False
    args.dreamlite_device='cuda:0'
    args.reader_device='cuda:1'
    runtime=load_base_runtime(args,bank,inference_only=True)
    frozen=train.frozen_versions(runtime['pipe'],runtime['reader'])
    context=runtime['contexts'][group['question_id']]
    reference=train.resolve_payload(group,'source_latent',args.bank_manifest)
    train.atomic_tensor(a.output/'initial.pt',initial)
    identity={'commit':commit,'source_hashes':train.source_hashes(),'parent_bank_sha256':BANK_SHA,
        'states':[{'state':s,'event_text':e,'gold':g} for s,e,g in STATES],
        'semantic_question_count':1,'conditional_state_count':3,'writer_checkpoint_used':False,
        'reused_teacher':tid,'initial_source':str(initial_path),'initial_file_sha256':entry['file_sha256'],
        'initial_tensor_sha256':entry['latent_sha256'],'proven_recipe_commit':'46cd36b',
        'optimizer':'Adam','lr':.05,'updates_per_new_state':256,'training_prompts':list(TRAIN_PROMPTS),
        'heldout_prompts':['paraphrase_3','paraphrase_4'],'prompt_schedule':'round_robin_zero_based',
        'snapshots':runtime['snapshots'],'base_protocol_binding':runtime['protocol_binding'],
        'gpu':gpu,'deadline_unix':a.deadline_unix}
    train.write_json(a.output/'identity.json',identity)
    new_teachers=[]
    new_groups=[]
    state_results={}
    for state,event,gold in STATES:
        if time.time()+900>=a.deadline_unix:
            raise TimeoutError('Insufficient remaining lease for another fixed oracle run')
        target={'inputs':group['question_variants'],'scorer_metadata':{'gold':gold}}
        rt={'vae':runtime['pipe'].vae,'reader':runtime['reader'],'processor':runtime['processor'],
            'vae_device':runtime['vae_device'],'reader_device':runtime['reader_device'],
            'termination':runtime['termination'],'training_prompts':TRAIN_PROMPTS,'target':target,
            'controls':{'blank':context['blank'],'fixed_donor':context['donor']}}
        spec={'run_id':'state-'+state,'state':state,'gold':gold,'event_text':event}
        directory=a.output/'runs'/spec['run_id']
        if state=='ambient':
            directory.mkdir(parents=True,exist_ok=False)
            latent=tensors[tid]
            with torch.no_grad():
                pixels=train.decode_model_latents_unit_interval(rt['vae'],latent.to(rt['vae_device']),clamp=True).cpu()
            train.write_json(directory/'manifest.json',{'reused_teacher':member,'fresh_optimization':False})
            records=oracle.evaluate(rt,pixels,spec=spec,step=256,directory=directory,all_prompts=True)
        else:
            oracle.run_one(spec=spec,initial=initial,reference=reference,runtime=rt,output_dir=a.output)
            records=[json.loads(x) for x in (directory/'generations.jsonl').read_text().splitlines()]
            latent=torch.load(directory/'endpoint_raw.pt',map_location='cpu',weights_only=True)['latent_fp32']
        matched=[r for r in records if r['condition']=='matched']
        passed=(len(matched)==5 and len({r['prompt_id'] for r in matched})==5
                and all(r['scorer']['strict_correct'] and r['scorer']['answer_followed_immediately_by_eos'] for r in matched))
        state_results[state]={'all_five_answer_eos':passed,'raw':{r['prompt_id']:r['raw'] for r in matched},
                              'generations_sha256':file_sha256(directory/'generations.jsonl')}
        qid=group['question_id']+'-state-'+state
        tensor_path=a.output/'bank'/'latents'/(state+'.pt')
        train.atomic_tensor(tensor_path,latent.detach().float().cpu())
        new_tid=qid+'-'+canonical_tensor_sha256(latent)[:16]
        teacher={'teacher_id':new_tid,'question_id':qid,'answer':gold,'endpoint_step':256,
            'latent_path':str(tensor_path),'latent_file_sha256':file_sha256(tensor_path),
            'latent_sha256':canonical_tensor_sha256(latent),'target':{'strict_correct':passed,'answer':gold},
            'gold_eos_appended':True,'evaluation_generation':{'do_sample':False,'max_new_tokens':32},
            'source_run':str(directory),'generation_file_sha256':state_results[state]['generations_sha256'],
            'reused_original_teacher':tid if state=='ambient' else None}
        new_teachers.append(teacher)
        new_groups.append({**copy.deepcopy(group),'question_id':qid,'semantic_question_id':group['question_id'],
            'state':state,'event_text':event,'answer':gold,'teacher_ids':[new_tid],
            'planned_count':1,'successful_run_count':int(passed)})
        audit_inference_only_runtime(runtime,frozen)
        train.write_json(a.output/'progress.json',state_results)
    runtime['verify_additional_bindings']()
    if train.source_hashes()!=identity['source_hashes'] or file_sha256(args.bank_manifest)!=BANK_SHA:
        raise RuntimeError('Source or original bank changed during oracle construction')
    passed=all(v['all_five_answer_eos'] for v in state_results.values())
    bank_path=a.output/'bank'/'manifest.json'
    if passed:
        new_bank={'schema':'latent-teacher-bank/v1','bank_status':'sealed','route':'direct',
            'models':bank['models'],'snapshots':bank['snapshots'],'teachers':new_teachers,'groups':new_groups,
            'semantic_question_count':1,'conditional_state_count':3,'provenance':identity}
        train.write_json(bank_path,new_bank)
        load_teacher_bank(bank_path)
    train.write_json(a.output/'complete.json',{'state_results':state_results,'bank_sealed':passed,
        'bank_manifest_sha256':file_sha256(bank_path) if passed else None,
        'identity_sha256':file_sha256(a.output/'identity.json'),'snapshots_verified':True})
    print(json.dumps(state_results),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
