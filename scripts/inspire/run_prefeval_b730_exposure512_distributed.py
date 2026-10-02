"""Four-GPU primary and two-GPU reader; unchanged registered scientific core."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from run_prefeval_b730_exposure512_two_gpu import (
    ENDPOINTS, FROZEN, FROZEN_COMMIT, RUN, load_core, registered_evaluations)


def adopt_existing_train(core, pid):
    """Preserve the already-running scientific process when replacing its coordinator."""
    receipt_path = RUN/'processes/train.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['pid'] == pid and receipt['host'] == socket.gethostname()
    assert receipt['status'] == 'running' and receipt['gpu'] == 0
    expected = ' '.join(receipt['command'])
    assert str(FROZEN/'scripts/experiments/prefeval_k1_write_extension.py') in expected
    assert '--steps 93440 --snapshot-steps 46720 70080' in expected
    proc = Path('/proc')/str(pid)
    assert (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip() == expected
    assert (proc/'cwd').resolve() == FROZEN
    started_ticks = (proc/'stat').read_text().rsplit(')',1)[1].split()[19]
    adoption = dict(pid=pid,host=socket.gethostname(),command=receipt['command'],
                    process_start_ticks=started_ticks,adopted_at=time.time())
    core.save(RUN/'train-adoption.json',adoption)
    def wait_for_existing_train():
        while proc.exists():
            try:
                stat = (proc/'stat').read_text().rsplit(')',1)[1].split()
                if stat[0] == 'Z':
                    break
                assert stat[19] == started_ticks, 'Adopted PID was reused'
                assert (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip() == expected
            except FileNotFoundError:
                break
            if core.STOP.wait(5):
                # The controller does not own this subprocess. Preserve it for
                # explicit recovery instead of killing healthy training.
                raise RuntimeError('Coordinator stopped; adopted training must be inspected separately')
        done = json.loads((RUN/'train/complete.json').read_text())
        assert done['steps'] == 93440 and (RUN/'train/checkpoint-final.pt').exists()
        receipt.update(status='complete',finished=time.time(),exit_code=None,
            completion_evidence='adopted process exited and frozen writer published train/complete.json steps93440',
            adopted_by=os.getpid(),process_start_ticks=started_ticks)
        core.save(receipt_path,receipt)
    core.train = wait_for_existing_train


def wait_for_reader(core):
    """Read completion only: never execute the other host's assigned matrices."""
    expected = [RUN/f'readback/step-{step:06d}-{split}-V1/summary.json'
                for step in ENDPOINTS for split in ('pilot','dev')]
    while not all(path.exists() for path in expected) or not (RUN/'reader-complete.json').exists():
        if (RUN/'reader-failure.json').exists():
            raise RuntimeError('Reader failed; recover its assigned work on the reader host')
        if core.STOP.wait(30):
            raise RuntimeError('Stopped while waiting for reader summaries')


def lane(core, role, gpu):
    if role == 'primary':
        if gpu == 0:
            core.train()
            split, variant = 'pilot', 0
        elif gpu == 1:
            core.evaluate(23360, 'train', 0, gpu)
            split, variant = 'train', 0
        elif gpu == 2:
            split, variant = 'train', 1
        elif gpu == 3:
            split, variant = 'dev', 0
        else:
            raise ValueError(gpu)
    elif role == 'reader' and gpu in (0, 1):
        split, variant = ('pilot' if gpu == 0 else 'dev'), 1
    else:
        raise ValueError((role, gpu))
    for step in ENDPOINTS:
        core.evaluate(step, split, variant, gpu)


def main(role):
    assert role in ('primary', 'reader')
    expected_host = 'prefeval-b-cr-h200x4-20260926--' if role == 'primary' else 'prefeval-b-read-h200x2-20260925--'
    assert socket.gethostname().startswith(expected_host)
    prefix = '' if role == 'primary' else 'reader-'
    count = 4 if role == 'primary' else 2
    physical = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).splitlines()
    assert len(physical) == count and all('H200' in name for name in physical)
    core = load_core()
    with (RUN / (prefix + 'controller.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        adopted_pid = os.environ.get('B730_ADOPT_TRAIN_PID')
        if adopted_pid:
            assert role == 'primary'
            adopt_existing_train(core,int(adopted_pid))
        scheduler = Path(__file__).resolve()
        core.save(RUN / (prefix + 'controller.json'), dict(
            pid=os.getpid(), host=socket.gethostname(), repo=str(FROZEN), commit=FROZEN_COMMIT,
            started=time.time(), role=role, physical_gpus=list(range(count)), scheduler_path=str(scheduler),
            scheduler_sha256=hashlib.sha256(scheduler.read_bytes()).hexdigest(),
            scheduler_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=scheduler.parent,text=True).strip()))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=count)
        try:
            futures = [pool.submit(lane, core, role, gpu) for gpu in range(count)]
            for future in concurrent.futures.as_completed(futures):
                future.result()
            if role == 'reader':
                core.save(RUN / 'reader-complete.json', dict(status='registered_V1_pilot_dev_complete', finished=time.time()))
            else:
                wait_for_reader(core)
                summaries = {str(p.relative_to(RUN)):json.loads(p.read_text()) for p in (RUN/'readback').glob('*/summary.json')}
                expected = {f'readback/step-{s:06d}-{split}-V{v}/summary.json' for s,split,v in registered_evaluations()}
                assert set(summaries) == expected
                assert json.loads((RUN/'train/complete.json').read_text())['steps'] == 93440
                core.save(RUN/'results.json',dict(final_step=93440,exposures_per_preference=512,
                    exposures_each_variant=256,dev_policy='report only; no selection or budget extension',evaluations=summaries))
                core.save(RUN/'complete.json',dict(status='training_and_registered_evaluation_complete',finished=time.time()))
        except BaseException as exc:
            core.terminate_children()
            core.save(RUN / (prefix+'failure.json'), dict(error=repr(exc),time=time.time()))
            raise
        finally:
            pool.shutdown(wait=True)


if __name__ == '__main__':
    main(sys.argv[1])
