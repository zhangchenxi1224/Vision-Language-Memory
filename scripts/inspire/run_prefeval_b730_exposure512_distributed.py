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
                # Normally these are already complete on the reader. Shared locks
                # also allow safe completion here if that host has failed.
                for step in ENDPOINTS:
                    for split in ('pilot', 'dev'):
                        core.evaluate(step, split, 1, 0)
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
