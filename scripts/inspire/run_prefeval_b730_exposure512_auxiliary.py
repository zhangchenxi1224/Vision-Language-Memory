"""Four idle GPUs evaluate the existing final pilot/dev matrices only."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from run_prefeval_b730_exposure512_two_gpu import FROZEN, FROZEN_COMMIT, RUN, load_core

ASSIGNMENTS = [(93440, 'pilot', 0, 0), (93440, 'dev', 0, 1),
               (93440, 'pilot', 1, 2), (93440, 'dev', 1, 3)]


def main():
    with (RUN / 'auxiliary-controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip(), 'GPU already occupied'
        import torch
        assert torch.__version__ == '2.7.0a0+ecf3bae40a.nv25.02' and torch.version.cuda == '12.8'
        assert torch.cuda.device_count() == 4
        for gpu in range(4):
            assert 'H200' in torch.cuda.get_device_name(gpu)
            assert (torch.ones(1, device=f'cuda:{gpu}') + 1).item() == 2
        core = load_core()
        core.save(RUN / 'auxiliary-controller.json', dict(
            pid=os.getpid(), host=socket.gethostname(), repo=str(FROZEN), commit=FROZEN_COMMIT,
            scheduler_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True).strip(),
            started=time.time(), assignments=ASSIGNMENTS))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        try:
            futures = [pool.submit(core.evaluate, *assignment) for assignment in ASSIGNMENTS]
            for future in concurrent.futures.as_completed(futures):
                future.result()
            core.save(RUN / 'auxiliary-complete.json', dict(status='registered_pilot_dev_complete', finished=time.time()))
        except BaseException as exc:
            core.terminate_children()
            core.save(RUN / 'auxiliary-failure.json', dict(error=repr(exc), time=time.time()))
            raise
        finally:
            pool.shutdown(wait=True)


if __name__ == '__main__':
    main()
