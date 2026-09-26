"""One unchanged sequential FM lane plus three fixed-endpoint evaluation lanes."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO/'src')]
from scripts.experiments.prefeval_k1_data import load_records, official_mcq, sha

ROOT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
OLD = ROOT/'runs/prefeval-b-mcq-20260925'
RUN = ROOT/'runs/prefeval-b730-exposure512-20260927'
MODELS = Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')
PYTHON = ROOT/'envs/vlm-r3-ngc2502/bin/python'
WRITER = REPO/'scripts/experiments/prefeval_k1_write_extension.py'
EVALUATE = REPO/'scripts/experiments/prefeval_k1_evaluate.py'
ENDPOINTS = [46720, 70080, 93440]
COMMON = ['--arm','B','--base',str(MODELS/'DreamLite-base-a9a0f15-20260907'),
          '--official-source',str(ROOT/'Vision-Language-Memory/third_party/DreamLite')]
STOP = threading.Event()
CHILDREN = {}
MUTEX = threading.Lock()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def run_command(label, gpu, command):
    if STOP.is_set():
        raise RuntimeError('Controller stopped')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    with open(RUN/'logs'/f'{label}.log','a') as log:
        proc = subprocess.Popen([str(x) for x in command], cwd=REPO, env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state = dict(label=label, gpu=gpu, pid=proc.pid, command=[str(x) for x in command],
                     host=socket.gethostname(), started=time.time(), status='running')
        save(RUN/'processes'/f'{label}.json', state)
        with MUTEX:
            CHILDREN[label] = proc
        code = proc.wait()
        with MUTEX:
            CHILDREN.pop(label, None)
        state.update(exit_code=code, finished=time.time(), status='complete' if code==0 else 'failed')
        save(RUN/'processes'/f'{label}.json', state)
    if code:
        raise RuntimeError(f'{label} exited {code}; see logs/{label}.log')


def train():
    if (RUN/'train/complete.json').exists():
        assert json.loads((RUN/'train/complete.json').read_text())['steps']==93440
        return
    run_command('train',0,[PYTHON,WRITER,'train',*COMMON,'--split','train','--stage','write',
        '--steps','93440','--snapshot-steps','46720','70080',
        '--initial-variants',OLD/'variants-train.json',
        '--teachers',ROOT/'runs/prefeval-k1-scale730-20260924/teachers/B',
        '--checkpoint',ROOT/'runs/dreamlite-official-alignment/4fbc857-clear-retention-full4832/train/checkpoint-final.pt',
        '--continue-from',OLD/'robust730/train/resume.pt','--output',RUN/'train'])


def checkpoint(step):
    if step==23360:
        return OLD/'robust730/train/checkpoint-final.pt'
    return RUN/'train'/('checkpoint-final.pt' if step==93440 else f'checkpoint-step-{step:06d}.pt')


def read_records(directory):
    return [json.loads(line) for path in sorted(directory.glob('readback-*.jsonl'))
            for line in path.read_text().splitlines()]


def summarize(directory, split, families, baseline):
    records = read_records(directory)
    ids = {r['base_pair_id'] for r in load_records(split)}
    expected = {(pid,chain,control,family) for pid in ids for family in families.split(',')
                for control in ['memory','mismatch','blank','text']
                for chain in (range(2) if control in ['memory','mismatch'] else range(1))}
    keyed = {(r['pair_id'],r['chain'],r['control'],r['family']):r for r in records}
    assert len(records)==len(keyed) and set(keyed)==expected, 'Incomplete or duplicate evaluation'
    parser = official_mcq(REPO/'third_party/prefeval_reference')['extract_choice']
    for row in records:
        assert row['task']=='mcq' and row['prefix']==0
        assert row['correct']==(parser(row['generated']['raw'])==row['correct_letter'])
    old = {(r['pair_id'],r['chain'],r['control'],r['family']):r for r in read_records(baseline)} if baseline else {}
    results = {}
    for family in families.split(','):
        controls = {}
        for control in ['memory','mismatch','blank','text']:
            rows = [r for r in records if r['family']==family and r['control']==control]
            controls[control] = dict(correct=sum(r['correct'] for r in rows), total=len(rows),
                parse_failure=sum(r['parse_failure'] for r in rows),
                truncated=sum(r['generated']['truncated'] for r in rows))
        pairs = [(keyed[pid,c,'memory',family]['correct'],keyed[pid,c,'mismatch',family]['correct'])
                 for pid in ids for c in range(2)]
        result = dict(controls=controls, match_minus_mismatch=sum(a-b for a,b in pairs),
            repaired=sum(a and not b for a,b in pairs), regressed=sum(b and not a for a,b in pairs),
            both_noise_correct=sum(all(keyed[pid,c,'memory',family]['correct'] for c in range(2)) for pid in ids),
            independent_preferences=len(ids))
        relevant = [(k,v) for k,v in keyed.items() if k[2]=='memory' and k[3]==family]
        if old and all(k in old for k,v in relevant):
            result['versus_step23360'] = dict(
                fixed=sum(v['correct'] and not old[k]['correct'] for k,v in relevant),
                lost=sum(old[k]['correct'] and not v['correct'] for k,v in relevant))
        results[family] = result
    save(directory/'summary.json',dict(split=split, record_count=len(records), families=results,
        files={p.name:sha(p) for p in directory.glob('readback-*.jsonl')}))


def evaluate(step, split, variant, gpu):
    label = f'step-{step:06d}-{split}-V{variant}'
    images = RUN/'images'/label
    output = RUN/'readback'/label
    variants = OLD/f'variants-{"dev" if split=="dev" else "train"}.json'
    families = 'T1' if split=='train' else 'T1,T2,T3,O1,O2'
    baseline = (RUN/'readback'/f'step-023360-train-V0' if split=='train' and variant==0
                else OLD/'robust730'/f'readback-final-{split}-V{variant}')
    if (output/'summary.json').exists():
        return
    while not checkpoint(step).exists():
        if STOP.wait(30):
            raise RuntimeError('Training failed before requested endpoint')
    run_command(label+'-rollout',gpu,[PYTHON,WRITER,'rollout',*COMMON,
        '--checkpoint',checkpoint(step),'--split',split,'--initial-variants',variants,
        '--initial-variant',str(variant),'--inter-turns','0','--noise-chains','2','--output',images])
    run_command(label+'-readback',gpu,[PYTHON,EVALUATE,'--kind','student','--split',split,
        '--reader',MODELS/'Qwen3-VL-4B-Instruct','--images',images,'--initial-variants',variants,
        '--initial-variant',str(variant),'--output',output,'--families',families,
        '--prefixes','0','--noise-chains','2','--controls','memory,mismatch,blank,text','--tasks','mcq'])
    summarize(output,split,families,baseline if step!=23360 else None)


def lane(gpu):
    if gpu==2:
        evaluate(23360,'train',0,gpu)  # Missing full-pool V0 baseline, fixed before continuation results.
    for step in ENDPOINTS:
        if gpu in [1,2]:
            evaluate(step,'train',2-gpu,gpu)
        else:
            for split in ['pilot','dev']:
                for variant in [1,0]:
                    evaluate(step,split,variant,gpu)


def terminate_children():
    STOP.set()
    with MUTEX:
        processes = list(CHILDREN.values())
    for proc in processes:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    for name in ['logs','processes']:
        (RUN/name).mkdir(exist_ok=True)
    lock = open(RUN/'controller.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(RUN/'controller.json',dict(pid=os.getpid(),host=socket.gethostname(),repo=str(REPO),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),started=time.time()))
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    futures = [pool.submit(train),*(pool.submit(lane,gpu) for gpu in [1,2,3])]
    try:
        for future in concurrent.futures.as_completed(futures):
            future.result()
        summaries = {str(p.relative_to(RUN)):json.loads(p.read_text()) for p in (RUN/'readback').glob('*/summary.json')}
        assert len(summaries)==19
        save(RUN/'results.json',dict(final_step=93440,exposures_per_preference=512,
            exposures_each_variant=256,dev_policy='report only; no selection or budget extension',evaluations=summaries))
        save(RUN/'complete.json',dict(status='training_and_registered_evaluation_complete',finished=time.time()))
    except BaseException as exc:
        terminate_children()
        save(RUN/'failure.json',dict(error=repr(exc),time=time.time()))
        raise
    finally:
        pool.shutdown(wait=True)


if __name__=='__main__':
    main()
