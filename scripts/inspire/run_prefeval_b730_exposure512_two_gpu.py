"""Two-GPU orchestration around the unmodified, frozen exposure512 core."""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import time

RUN = Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b730-exposure512-20260927')
FROZEN = RUN / 'code'
FROZEN_COMMIT = '656fdf029c4a7c05e53bccb75483a03cf62d5f54'
ENDPOINTS = (46720, 70080, 93440)


def registered_evaluations():
    return [(23360, 'train', 0)] + [
        (step, split, variant) for step in ENDPOINTS
        for split in ('train', 'pilot', 'dev') for variant in (0, 1)]


def lane(core, gpu):
    if gpu == 0:
        core.train()
    elif gpu == 1:
        core.evaluate(23360, 'train', 0, gpu)
    else:
        raise ValueError('Only physical GPUs 0 and 1 are registered')
    for step in ENDPOINTS:
        for split in ('train', 'pilot', 'dev'):
            core.evaluate(step, split, gpu, gpu)


def execute(core):
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        futures = [pool.submit(lane, core, gpu) for gpu in (0, 1)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    except BaseException:
        core.terminate_children()
        raise
    finally:
        pool.shutdown(wait=True)


def main():
    import fcntl
    core_path = FROZEN / 'scripts/inspire/run_prefeval_b730_exposure512.py'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=FROZEN, text=True).strip() == FROZEN_COMMIT
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=FROZEN, text=True).strip()
    spec = importlib.util.spec_from_file_location('b730_frozen_core', core_path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    assert core.REPO == FROZEN and tuple(core.ENDPOINTS) == ENDPOINTS
    for name in ('logs', 'processes'):
        (RUN / name).mkdir(parents=True, exist_ok=True)
    with (RUN / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scheduler = Path(__file__).resolve()
        core.save(RUN / 'controller.json', dict(
            pid=os.getpid(), host=socket.gethostname(), repo=str(FROZEN),
            commit=FROZEN_COMMIT, started=time.time(), physical_gpus=[0, 1],
            scheduler_path=str(scheduler),
            scheduler_sha256=hashlib.sha256(scheduler.read_bytes()).hexdigest(),
            scheduler_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=scheduler.parent, text=True).strip()))
        try:
            execute(core)
            summaries = {str(p.relative_to(RUN)): json.loads(p.read_text())
                         for p in (RUN / 'readback').glob('*/summary.json')}
            expected = {f'readback/step-{step:06d}-{split}-V{variant}/summary.json'
                        for step, split, variant in registered_evaluations()}
            assert set(summaries) == expected
            assert json.loads((RUN / 'train/complete.json').read_text())['steps'] == 93440
            core.save(RUN / 'results.json', dict(final_step=93440, exposures_per_preference=512,
                exposures_each_variant=256, dev_policy='report only; no selection or budget extension', evaluations=summaries))
            core.save(RUN / 'complete.json', dict(status='training_and_registered_evaluation_complete', finished=time.time()))
        except BaseException as exc:
            core.save(RUN / 'failure.json', dict(error=repr(exc), time=time.time()))
            raise


if __name__ == '__main__':
    main()
