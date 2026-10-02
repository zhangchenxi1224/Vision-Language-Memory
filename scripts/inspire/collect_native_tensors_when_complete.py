"""Wait on CPU for final native evaluation and independently inspect every final PT."""
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
    parent = runs/'03f8467-native-condition-full4832'
    status = runs/'03f8467-native-tensors-driver-status.json'
    output = runs/'03f8467-native-final-tensors'
    if status.exists() or Path(str(output)+'-summary.json').exists():
        raise ValueError('Final tensor audit already registered; inspect existing work')
    def record(state, **extra):
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps({'state': state, 'time_unix': time.time(), **extra}, indent=2)+'\n')
        temporary.replace(status)
    try:
        record('waiting_for_complete_native_endpoint')
        while True:
            suite = json.loads((runs/'9e27050-logical-completion-suite-status.json').read_bytes())
            if suite['state'] in ('failed', 'needs_attention'):
                raise RuntimeError('Native suite needs inspection before additional collection')
            if suite['stage'] not in ('waiting_for_fixed_endpoint', 'endpoint-collection') and (parent/'terminal.json').exists():
                break
            if time.time() >= 1789344000:  # 08:00 Asia/Shanghai
                raise TimeoutError('Native endpoint did not complete by its execution deadline')
            time.sleep(20)
        command = [sys.executable, '-u', str(runs/'collect-native-endpoint-tensors-20260914.py'),
            '--source-root', str(root/'repos/dreamlite-native-validation-20260914'),
            '--run', str(parent), '--output-prefix', str(output)]
        record('checking_all_final_native_tensors', command=command)
        subprocess.run(command, check=True, timeout=3600,
            env={**os.environ, 'PYTHONUTF8': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        summary = json.loads(Path(str(output)+'-summary.json').read_bytes())
        record('completed', correct_eos=summary['development']['correct_eos'],
            archive_sha256=hashlib.sha256(Path(str(output)+'-evidence.tgz').read_bytes()).hexdigest(),
            local_download_and_recount_still_required=True)
    except BaseException as error:
        record('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
