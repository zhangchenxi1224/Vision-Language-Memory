"""Migrate the unchanged B730 experiment to the user-selected normal-priority two-GPU notebook."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import torch

root=Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b730-exposure512-20260927').resolve()
repo=root/'code'
operations=root/'operations/two-gpu-20260928/code'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=operations,text=True).strip()=='2bd72308158ca733a38423e125f820b43d3c2a8b'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=operations,text=True).strip()
recovery=root/'recovery-20260928-2015'
assert not recovery.exists(), 'Recovery receipt already exists; inspect before any retry'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()=='656fdf029c4a7c05e53bccb75483a03cf62d5f54'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
locks=[]
for role in ['primary','auxiliary']:
    assert json.loads((root/'priority-switch-20260928'/role/'stop.json').read_text())['compute_apps_empty']
for name in ['launcher.lock','controller.lock','auxiliary-launcher.lock','auxiliary-controller.lock']:
    f=open(root/name,'a')
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    locks.append(f)
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid(): continue
    try: cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except (FileNotFoundError,ProcessLookupError): continue
    if str(root) in cmd and any(name in cmd for name in [
        'prefeval_k1_write_extension.py','prefeval_k1_evaluate.py',
        'run_prefeval_b730_exposure512.py','launch_prefeval_b730_exposure512.sh',
        'run_prefeval_b730_exposure512_two_gpu.py','launch_prefeval_b730_exposure512_two_gpu.sh',
        'run_prefeval_b730_exposure512_distributed.py','launch_prefeval_b730_exposure512_distributed.sh']):
        raise RuntimeError(f'Live task process found: {p.name}')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
assert len(gpu_info.strip().splitlines())==4 and all('H200' in x for x in gpu_info.strip().splitlines())
recovery.mkdir()
for name in ['controller.json','latest-runtime.json','preflight.json','launch.log','launcher.pid',
             'train/manifest.json','train/continuation.json','train/optimization.jsonl','logs/train.log']:
    p=root/name
    if p.exists():
        dest=recovery/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
shutil.copytree(root/'processes',recovery/'processes')
os.link(root/'train/resume.pt',recovery/'resume-before.pt')
with (recovery/'resume-before.pt').open('rb') as f:
    saved=torch.load(f,map_location='cpu',weights_only=False)
    f.seek(0)
    digest=hashlib.file_digest(f,'sha256').hexdigest()
step=saved['optimizer_step']
assert digest=='37c3d87457242d75a4f4989ab175c4cec8c8fa0cbccd13c73e194ed541eb2ea0'
assert step==89728 and saved['episode_cursor']==step*4
assert len(saved['optimizer']['state'])==1075
assert {int(v['step']) for v in saved['optimizer']['state'].values()}=={step}
assert saved['manifest']==json.loads((root/'train/manifest.json').read_text())
assert saved['manifest']['steps']==93440 and saved['manifest']['snapshot_steps']==[46720,70080]
assert len(saved['rng_state']['torch_cuda'])==1
rows=[json.loads(x) for x in (root/'train/optimization.jsonl').read_text().splitlines()]
assert [r['step'] for r in rows]==list(range(1,rows[-1]['step']+1))
assert rows[-1]['step']==89797

def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

summary_hashes={str(p.relative_to(root)):sha(p) for p in sorted((root/'readback').glob('*/summary.json'))}
assert len(summary_hashes)==13
evaluation_evidence={}
partial=[]
for images in sorted((root/'images').glob('step-070080-*')):
    label=images.name
    complete={str(p.relative_to(root)):sha(p) for p in sorted(images.glob('*/seed-*/complete.json'))}
    readback=root/'readback'/label
    if readback.exists(): shutil.copytree(readback,recovery/'readback-before'/label)
    preserved_rows={}
    for p in sorted(readback.glob('readback-*.jsonl')):
        original=p.read_bytes()
        lines=original.splitlines(keepends=True)
        valid=[]
        for i,line in enumerate(lines):
            try: json.loads(line)
            except ValueError:
                assert i==len(lines)-1, 'Corrupt non-tail readback row requires investigation'
                p.write_bytes(b''.join(valid))
                partial.append(str(p.relative_to(root))+': incomplete trailing JSON archived')
                break
            valid.append(line)
        body=b''.join(valid)
        preserved_rows[str(p.relative_to(root))]=dict(bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),rows=len(valid))
    for directory in sorted(images.glob('*/seed-*')):
        if not directory.is_dir() or (directory/'complete.json').exists() or not any(directory.iterdir()): continue
        assert directory.resolve().is_relative_to((root/'images').resolve())
        dest=recovery/'partial-images'/label/directory.relative_to(images)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(directory),str(dest))
        partial.append(str(directory.relative_to(root)))
    evaluation_evidence[label]=dict(completed_png_metadata=complete,readback_prefix=preserved_rows)
for name in ['failure.json','launcher-exit-status.txt']:
    if (root/name).exists(): shutil.move(str(root/name),str(recovery/name))
report=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),host=socket.gethostname(),
    platform_event='User explicitly selected normal-priority four-H200 primary with two-H200 reader; superseded initial migration stopped before optimizer updates',
    new_node='qb-prod-gpu997',new_notebook='prefeval-b-cr-h200x4-20260926',gpu_before=gpu_info,
    scheduler_commit='2bd72308158ca733a38423e125f820b43d3c2a8b',scheduler_directory=str(operations),all_task_locks_free=True,no_live_task_processes=True,gpu_compute_processes_empty=True,
    restored_step=step,next_step=step+1,last_uncommitted_log_step=rows[-1]['step'],
    uncommitted_updates=rows[-1]['step']-step,optimizer_states=1075,all_optimizer_steps_match=True,
    rng_keys=list(saved['rng_state']),resume_sha256=digest,preserved_resume=str(recovery/'resume-before.pt'),
    completed_summaries=list(summary_hashes),completed_summary_hashes=summary_hashes,
    evaluation_evidence=evaluation_evidence,partial_artifacts_archived=partial,fixed_final_step=93440)
(recovery/'recovery.json').write_text(json.dumps(report,indent=2)+'\n')
for name in ['run_prefeval_b730_exposure512_two_gpu.py','launch_prefeval_b730_exposure512_two_gpu.sh',
        'run_prefeval_b730_exposure512_distributed.py','launch_prefeval_b730_exposure512_distributed.sh']:
    shutil.copy2(operations/'scripts/inspire'/name,recovery/name)
for f in locks: f.close()
result=subprocess.check_output(['bash',str(operations/'scripts/inspire/launch_prefeval_b730_exposure512_distributed.sh'),'primary'],text=True).strip()
(recovery/'launch.json').write_text(json.dumps(dict(launcher_pid=int(result),time=time.time()),indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='evaluation_evidence'},indent=2))
print('evaluation_preserved',json.dumps({k:dict(completed_pngs=len(v['completed_png_metadata']),readback_rows=sum(x['rows'] for x in v['readback_prefix'].values())) for k,v in evaluation_evidence.items()}))
print('new_launcher_pid',result)
