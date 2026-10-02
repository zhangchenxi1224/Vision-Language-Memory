"""Wait on CPU for the full frozen control, then independently verify all artifacts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
    runs = root/'runs/dreamlite-official-alignment'
    status = runs/'1f86d56-raw-evidence-driver-status.json'
    output = runs/'1f86d56-raw-condition-verified'
    if status.exists() or Path(str(output)+'-summary.json').exists():
        raise ValueError('Collection already registered; inspect existing work')
    def record(state, **extra):
        status.write_text(json.dumps({'state': state, 'time_unix': time.time(), **extra}, indent=2)+'\n')
    try:
        record('waiting_for_full_raw_control')
        while True:
            probe = json.loads((runs/'1f86d56-broader-raw-driver-status.json').read_bytes())
            if probe['state'] == 'failed':
                raise RuntimeError('Full raw control failed; preserve partial data without scoring')
            if probe['state'] == 'completed':
                break
            if time.time() >= 1789336200:  # 2026-09-14 05:50 Asia/Shanghai
                raise TimeoutError('No full raw control by its registered execution window')
            time.sleep(20)
        run = runs/'1f86d56-broader-raw-condition'
        if hashlib.sha256((run/'complete.json').read_bytes()).hexdigest() != probe['complete_sha256']:
            raise ValueError('Probe completion differs from terminal driver seal')
        parent = runs/'bb34092-logical31-full4832'
        command = json.loads((parent/'commands.json').read_bytes())['commands'][-1]
        bank = command[command.index('--bank-manifest')+1]
        invocation = [sys.executable, '-u', str(runs/'collect-broader-raw-condition-20260914.py'),
            '--source-root', str(root/'repos/dreamlite-broader-raw-control-20260914'),
            '--run', str(run), '--parent', str(parent), '--bank', bank, '--output-prefix', str(output)]
        record('verifying_all_raw_rows_and_tensors', command=invocation)
        subprocess.run(invocation, check=True, timeout=3600,
            env={**os.environ, 'PYTHONUTF8': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        summary = json.loads(Path(str(output)+'-summary.json').read_bytes())
        record('completed', raw_correct_eos=summary['raw_summary']['correct_eos'],
            archive_sha256=hashlib.sha256(Path(str(output)+'-evidence.tgz').read_bytes()).hexdigest(),
            local_download_and_full_recount_still_required=True)
    except BaseException as error:
        record('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
