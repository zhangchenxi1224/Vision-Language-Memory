"""Read-only final completion/release audit, except for its evidence JSON."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
RUN = ROOT/'runs/prefeval-b730-exposure512-20260927'
CODE = RUN/'code'
role='primary' if socket.gethostname().startswith('prefeval-b-cr-h200x4-20260926--') else 'reader'
assert role=='primary' or socket.gethostname().startswith('prefeval-b-read-h200x2-20260925--')
OUT = RUN/f'evidence/final-completion-audit-{role}.json'
assert not (RUN/'reader-failure.json').exists()
assert (RUN/'reader-complete.json').exists()
assert (RUN/'reader-launcher-exit-status.txt').read_text().strip()=='0'
assert not OUT.exists(), 'Inspect prior final audit before retrying'
assert not (RUN/'failure.json').exists()
done = json.loads((RUN/'complete.json').read_text())
training = json.loads((RUN/'train/complete.json').read_text())
results = json.loads((RUN/'results.json').read_text())
assert done['status'] == 'training_and_registered_evaluation_complete'
assert training['steps'] == results['final_step'] == 93440
assert results['exposures_per_preference'] == 512 and results['exposures_each_variant'] == 256
assert results['dev_policy'] == 'report only; no selection or budget extension'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip() == '656fdf029c4a7c05e53bccb75483a03cf62d5f54'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=CODE,text=True).strip()
locks = []
for name in ['launcher.lock','controller.lock','reader-launcher.lock','reader-controller.lock']+[str(p.relative_to(RUN)) for p in (RUN/'evaluation-locks').glob('*.lock')]:
    lock = (RUN/name).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    locks.append(lock)
tokens = ['prefeval_k1_write_extension.py','prefeval_k1_evaluate.py','run_prefeval_b730_exposure512.py','launch_prefeval_b730_exposure512.sh','run_prefeval_b730_exposure512_distributed.py','launch_prefeval_b730_exposure512_distributed.sh']
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        command = (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except (FileNotFoundError,ProcessLookupError):
        continue
    assert not (str(RUN) in command and any(token in command for token in tokens)), f'Live task process: {proc.name}'
compute = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
assert not compute, 'A GPU compute process is still active; do not stop this instance'
expected = {'readback/step-023360-train-V0/summary.json'} | {f'readback/step-{step:06d}-{split}-V{variant}/summary.json' for step in [46720,70080,93440] for split in ['train','pilot','dev'] for variant in [0,1]}
actual = {str(path.relative_to(RUN)) for path in (RUN/'readback').glob('*/summary.json')}
assert actual == set(results['evaluations']) == expected and len(actual) == 19
def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()
summaries = {}
for name in sorted(expected):
    path = RUN/name
    summary = json.loads(path.read_text())
    assert summary == results['evaluations'][name]
    assert summary['record_count'] == {'train':4380,'pilot':1920,'dev':2700}[summary['split']]
    for filename,digest in summary['files'].items():
        assert Path(filename).name == filename
        assert sha(path.parent/filename) == digest
    summaries[name] = dict(sha256=sha(path),record_count=summary['record_count'])
processes = []
for path in sorted((RUN/'processes').glob('*.json')):
    receipt = json.loads(path.read_text())
    assert receipt['status']=='complete'
    if receipt['label']=='train' and receipt['exit_code'] is None:
        adoption=json.loads((RUN/'train-adoption.json').read_text())
        assert receipt['pid']==adoption['pid'] and receipt['process_start_ticks']==adoption['process_start_ticks']
        assert receipt['completion_evidence']=='adopted process exited and frozen writer published train/complete.json steps93440'
    else:
        assert receipt['exit_code']==0
    processes.append(dict(label=receipt['label'],host=receipt['host'],gpu=receipt['gpu'],pid=receipt['pid'],exit_code=receipt['exit_code']))
assert (RUN/'launcher-exit-status.txt').read_text().strip() == '0'
report = dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),host=socket.gethostname(),role=role,frozen_commit='656fdf029c4a7c05e53bccb75483a03cf62d5f54',fixed_final_step=93440,registered_summaries=19,all_expected_summary_names_match=True,all_readback_file_hashes_match=True,no_live_task_processes=True,all_local_task_locks_free=True,training_os_exit_code=next(p["exit_code"] for p in processes if p["label"]=="train"),gpu_compute_processes_empty=True,launcher_exit_code=0,complete=done,training_complete=training,root_results_sha256=sha(RUN/'results.json'),root_complete_sha256=sha(RUN/'complete.json'),summaries=summaries,process_receipts=processes,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'],text=True))
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(report,indent=2)+'\n')
for lock in locks:
    lock.close()
print(json.dumps({key:value for key,value in report.items() if key not in ['summaries','process_receipts']},indent=2))
