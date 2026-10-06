from pathlib import Path
import os,json,time,socket,subprocess,traceback,hashlib,sys,shutil
root=Path('/inspire/ssd/project/exploration-topic/czxs26210936');repo=root/'repos/context-layout-aug-20261006-v2';run=root/'runs/context-coverage-20261006';out=run/'context-layout-aug-readout-v1'
target_instance=sys.argv[1]
def cmd(*args):return subprocess.check_output(args,text=True)
def save(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
try:
 assert not out.exists(), 'Only a new independent output may be used'
 out.mkdir()
 print('preflight: started',flush=True)
 assert json.loads((run/'context-layout-aug-v1/status.json').read_text())['status']=='ready_for_frozen_readout'
 assert json.loads((run/'context-layout-aug-v1/native-parity.json').read_text())['arms']['augmented']['exact_parameters_optimizer_rng']
 host=socket.gethostname();meta=json.loads((run/'resource-owners-readout-0650.json').read_text());inventory=meta['items'];skip=meta['pruned']
 assert time.time()-meta['checked_at']<600,'Owner inventory expired'
 for i in inventory:
  assert hashlib.sha256(Path(i['path']).read_bytes()).hexdigest()==i['sha256'],'Owner changed after inventory'
  assert host not in (i['host'],i['instance']) and i['instance']!=target_instance,str(i)
 print('owner scan complete',len(inventory),flush=True)
 assert not list(run.glob('*/active-owner/owner.json'))
 pids=[55309,json.loads((run/'context-layout-aug-v1/relaunch-storage-0650.json').read_text())['controller_pid']]+[json.loads(p.read_text())['pid'] for p in (run/'context-layout-aug-v1/attempts').glob('*.json')]
 assert json.loads((run/'context-layout-aug-v1/status.json').read_text())['status']=='ready_for_frozen_readout'
 for pid in pids:
  path=Path('/proc')/str(pid)/'cmdline'
  assert not path.exists() or 'context_layout_aug' not in path.read_bytes().replace(b'\x00',b' ').decode(), 'Old process still active'
 assert not (run/'context-layout-aug-v1/active-owner').exists()
 print('process check passed',flush=True)
 compute=cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader');assert not compute.strip()
 gpu=cmd('nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader')
 free=[int(x.strip().split()[0]) for x in cmd('nvidia-smi','--query-gpu=memory.free','--format=csv,noheader').splitlines()];assert len(free)==2 and min(free)>120000
 mem=cmd('free','-m');processes=cmd('ps','-eo','pid,etimes,args')
 assert not any('prefeval_' in line and 'python -c' not in line for line in processes.splitlines())
 print('resource check passed',flush=True)
 assert cmd('git','-C',str(repo),'rev-parse','HEAD').strip()=='844dda1819168b551882577a2946548dc8602f2f'
 assert not cmd('git','-C',str(repo),'status','--porcelain').strip()
 model=Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')
 command=[str(root/'envs/vlm-r3-ngc2502/bin/python'),'scripts/inspire/run_context_layout_aug_readout.py','run','--output',str(out),'--base',str(model/'DreamLite-base-a9a0f15-20260907'),'--reader',str(model/'Qwen3-VL-4B-Instruct'),'--official-source',str(root/'Vision-Language-Memory/third_party/DreamLite')]
 evidence=dict(time=time.time(),host=host,commit=cmd('git','-C',str(repo),'rev-parse','HEAD').strip(),tree=cmd('git','-C',str(repo),'rev-parse','HEAD^{tree}').strip(),command=command,compute_before=compute,gpus=gpu,memory=mem,processes_before='\n'.join(line[:180] for line in processes.splitlines()),other_owner_inventory=inventory,owner_scan_pruned_artifact_directories=sorted(skip),old_completed_worker_pids=pids,notebook=target_instance,owner_inventory_checked_at=meta['checked_at'],tests_linux=38,tests_windows=38)
 evidence['linux_preflight']=json.loads((run/'layout-aug-v2-linux-preflight-0650.json').read_text())
 assert evidence['linux_preflight']['exit_code']==0
 image_store=Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/runs/context-coverage-20261006/checkpoint-store/layout-aug-readout-v1-images')
 assert not image_store.exists()
 assert shutil.disk_usage(image_store.parent).free>10*1024**3
 image_store.mkdir()
 (out/'images').symlink_to(image_store,target_is_directory=True)
 evidence['image_store']=str(image_store)
 save('preflight-0650.json',evidence)
 env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 with (out/'controller.log').open('a') as log:p=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
 evidence.update(controller_pid=p.pid,launched_unix=time.time());save('launch-0650.json',evidence)
 print(json.dumps(dict(controller_pid=p.pid,launched=evidence['launched_unix'],owners=len(inventory),compute_before=compute)))
except BaseException:
 save('prelaunch-error-0650.json',dict(time=time.time(),error=traceback.format_exc()));raise


