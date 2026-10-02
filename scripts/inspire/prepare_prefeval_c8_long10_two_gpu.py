import importlib.util,json,shutil,hashlib
from pathlib import Path
t=Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-multitarget-20260927')
old=t/'long10/recovery-20260928-2310';new=t/'long10/recovery-20260929-0005-2gpu'
script=t/'long10/resume-two-gpu-20260929.py'
assert not new.exists()
assert not any((old/n).exists() for n in ['launch.json','controller.json','failure.json','complete.json'])
assert not list(old.rglob('readback-*.jsonl'))
spec=importlib.util.spec_from_file_location('resume2',script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=m.frozen.prepare(new,False)
assert m.sha(new/'protocol.json')==m.PROTOCOL_SHA
r=json.loads((old/'recovery-preparation.json').read_text())
assert r['complete_chains']=={'C8/V0':800} and len(r['verified_files'])==10400
for rel,h in r['verified_files'].items():
 source=old/rel;dest=new/rel
 assert m.sha(source)==h
 dest.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(source,dest)
 assert m.sha(dest)==h
for source in old.glob('*/eval-V*/manifest*.json'):
 dest=new/source.relative_to(old);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
 assert m.sha(source)==m.sha(dest)
shell='\n'.join(['#!/usr/bin/env bash','set -euo pipefail',f'exec 9>"{t}/long10/pipeline.lock"','flock -n 9',f'cd "{m.base.FROZEN}"','export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1',f'exec "{m.base.PYTHON}" "{script}" --output "{new}"',''])
(new/'launch-two-gpu.sh').write_text(shell)
report={'source':str(old),'original_source':r['source'],'output':str(new),'copied_complete_chains':800,'copied_pngs':8800,'copied_files':10400,'verified_files':r['verified_files'],'checkpoints':p['checkpoints'],'protocol_sha256':m.sha(new/'protocol.json'),'controller_sha256':m.sha(script),'controller_source_commit':'befc4477361d266790bc99f5e2789b20e0d13158','launcher_sha256':m.sha(new/'launch-two-gpu.sh'),'physical_gpus':2,'logical_shards':4,'mapping':[0,1,0,1],'frozen_protocol_physical_gpus_field_is_historical':True,'gpu_launched':False}
(new/'migration-preparation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='verified_files'}),flush=True)
