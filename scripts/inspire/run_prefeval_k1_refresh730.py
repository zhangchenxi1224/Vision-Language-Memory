"""Run C730 and R730 within the previously allocated two-arm retention budget.

C keeps the registered fixed initial-source bank. R alternates current-model
PNG rollouts and official FM updates, preserving the optimizer across rounds.
Each lane holds a shared-filesystem lock inherited by all GPU children.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.experiments.prefeval_k1_data import load_training_records, sha
from scripts.experiments.prefeval_k1_refresh_bank import (
    ROUNDS, TOTAL_STEPS, freeze_refresh_bank, round_bounds, source_checkpoint,
)

PROJECT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
RUN = PROJECT / 'runs/prefeval-b-mcq-20260925'
PYTHON = PROJECT / 'envs/vlm-r3-ngc2502/bin/python'
MODELS = Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')
CONTROL_CODE = PROJECT / 'repos/prefeval-b-deploy-52d98d8'
PARENT = RUN / 'robust730/train/checkpoint-final.pt'
WRITER = ROOT / 'scripts/experiments/prefeval_k1_writer.py'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def environment(gpu, code=ROOT):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), K1_CODE_ROOT=str(code),
        CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', PYTHONHASHSEED='0', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')


def common():
    return ['--arm', 'B', '--base', str(MODELS/'DreamLite-base-a9a0f15-20260907'),
            '--official-source', str(PROJECT/'Vision-Language-Memory/third_party/DreamLite')]


class Runner:
    def __init__(self, arm, gpus):
        self.arm, self.gpus = arm, gpus
        self.output = RUN / f'retain730-{arm}'
        self.output.mkdir(parents=True, exist_ok=True)
        self.record = {'arm': arm, 'host': os.uname().nodename, 'pid': os.getpid(), 'gpus': gpus,
            'code_root': str(ROOT), 'commit': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}

    def status(self, stage, **extra):
        value = dict(self.record, stage=stage, time_utc=datetime.now(timezone.utc).isoformat(), **extra)
        save(self.output/'driver-state.json', value)
        print(json.dumps(value), flush=True)

    def wait_parent(self):
        self.status('waiting_for_B730_fixed_23360_endpoint')
        while not (PARENT.parent/'complete.json').exists():
            time.sleep(30)
        assert json.loads((PARENT.parent/'complete.json').read_text()) == {
            'steps': TOTAL_STEPS, 'checkpoint_sha256': sha(PARENT)}

    def idle(self, gpu):
        while True:
            output = subprocess.check_output(['nvidia-smi','-i',str(gpu),
                '--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
            if not output:
                return
            self.status('waiting_for_idle_gpu', gpu=gpu, active_pids=output)
            time.sleep(30)

    def run_jobs(self, label, jobs, code=ROOT):
        children, streams, receipts = [], [], []
        try:
            for index, (gpu, command) in enumerate(jobs):
                self.idle(gpu)
                log = self.output/f'{label}-{index}.log'
                stream = log.open('a')
                streams.append(stream)
                process = subprocess.Popen(list(map(str, command)), cwd=code, env=environment(gpu,code),
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    pass_fds=(self.lock.fileno(),))
                children.append(process)
                receipts.append({'pid': process.pid, 'gpu': gpu, 'command': list(map(str,command)),
                                 'log': str(log), 'code_root': str(code)})
                self.status(label, children=receipts)
            codes = [process.wait() for process in children]
            save(self.output/f'{label}-exit.json', {'children': receipts, 'exit_codes': codes})
            assert all(code == 0 for code in codes), f'{label}: child failure {codes}'
        finally:
            for stream in streams:
                stream.close()

    def control(self):
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=CONTROL_CODE,text=True).strip().startswith('52d98d8')
        if not (RUN/'sourcebank730/bank.json').exists():
            for variant in range(2):
                self.run_jobs(f'control-source-V{variant}', [(self.gpus[0], ['bash',
                    'scripts/inspire/run_prefeval_k1_sourcebank730.sh', str(variant)])], CONTROL_CODE)
            self.status('freezing_control_bank')
            subprocess.run([str(PYTHON), 'scripts/experiments/prefeval_k1_source_bank.py',
                '--root', str(RUN/'sourcebank730'), '--checkpoint', str(PARENT),
                '--variants', str(RUN/'variants-train.json')], cwd=CONTROL_CODE, check=True)
        self.run_jobs('control-train-and-evaluate', [(self.gpus[0],
            ['bash','scripts/inspire/run_prefeval_k1_retain730.sh','C'])], CONTROL_CODE)

    def refreshed(self):
        train = self.output/'train'
        sources = self.output/'sources'
        rows = load_training_records('train')
        assert len(rows) == 730
        for index in range(ROUNDS):
            start, stop = round_bounds(index)
            complete = train/f'round-{index}-complete.json'
            if complete.exists() and (index < ROUNDS-1 or (train/'complete.json').exists()):
                done = json.loads(complete.read_text())
                assert done['step'] == stop
                assert sha(train/f'checkpoint-step-{stop:06d}.pt') == done['checkpoint_sha256']
                continue
            checkpoint = source_checkpoint(train, PARENT, index)
            if index:
                previous = json.loads((train/f'round-{index-1}-complete.json').read_text())
                assert previous['step'] == start and sha(checkpoint) == previous['checkpoint_sha256']
            bank = sources/f'round-{index}'
            if not (bank/'bank.json').exists():
                jobs = []
                for shard, gpu in enumerate(self.gpus):
                    command = [PYTHON, WRITER, 'rollout', *common(), '--split','train',
                        '--checkpoint',checkpoint,'--initial-variants',RUN/'variants-train.json',
                        '--initial-variant',str(index % 2),'--inter-turns','9','--noise-chains','1',
                        '--noise-domain',f'refresh-{index}','--shard-count','2','--shard-index',str(shard),
                        '--output',bank/f'shard-{shard}']
                    jobs.append((gpu,command))
                self.run_jobs(f'round-{index}-current-student-rollout', jobs)
                self.status(f'round-{index}-freezing-training-sources')
                freeze_refresh_bank(bank,rows,checkpoint,RUN/'variants-train.json',index)
            command = [PYTHON,WRITER,'train',*common(),'--split','train','--stage','retain',
                '--steps',str(TOTAL_STEPS),'--retain-source-mode','refresh-bank','--refresh-round',str(index),
                '--initial-variants',RUN/'variants-train.json','--sources',sources,
                '--teachers',PROJECT/'runs/prefeval-k1-scale730-20260924/teachers/B',
                '--checkpoint',PARENT,'--output',train]
            self.run_jobs(f'round-{index}-fm-{start}-{stop}',[(self.gpus[0],command)])
        self.evaluate()

    def evaluate(self):
        # No training PNGs enter this evaluation: fresh two-seed, full ten-update rollouts.
        checkpoint = self.output/'train/checkpoint-final.pt'
        for split in ['dev','pilot','train']:
            variants = RUN / ('variants-dev.json' if split == 'dev' else 'variants-train.json')
            label = f'final-{split}-V1'
            images = self.output/label
            commands = [[PYTHON,WRITER,'rollout',*common(),'--split',split,'--checkpoint',checkpoint,
                '--initial-variants',variants,'--initial-variant','1','--inter-turns','10',
                '--noise-chains','2','--output',images],
                [PYTHON,ROOT/'scripts/experiments/prefeval_k1_evaluate.py','--kind','student','--split',split,
                 '--reader',MODELS/'Qwen3-VL-4B-Instruct','--images',images,
                 '--initial-variants',variants,'--initial-variant','1',
                 '--output',self.output/f'readback-{label}','--families',
                 'T1' if split == 'train' else 'T1,T2,T3,O1,O2','--prefixes','0,1,5,10',
                 '--noise-chains','2','--controls','memory,mismatch,blank,text','--tasks','mcq']]
            if split != 'train':
                for prefix in [0,10]:
                    commands.append([PYTHON,ROOT/'scripts/experiments/prefeval_k1_position_probe.py',
                        '--kind','student','--split',split,'--reader',MODELS/'Qwen3-VL-4B-Instruct',
                        '--images',images,'--prefix',str(prefix),'--order-mode','official-cyclic',
                        '--output',self.output/f'positions-{label}-{prefix}.jsonl'])
            for index, command in enumerate(commands):
                self.run_jobs(f'{label}-evaluation-{index}',[(self.gpus[0],command)])

    def run(self):
        import fcntl
        # Inherited by children: restarting a dead coordinator cannot duplicate an orphaned GPU worker.
        with (self.output/'refresh-driver.lock').open('a') as self.lock:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                self.wait_parent()
                self.control() if self.arm == 'C' else self.refreshed()
                self.status('complete')
            except BaseException as error:
                self.status('failed',error=repr(error))
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['C','R'],required=True)
    parser.add_argument('--gpus',type=int,nargs='+',required=True)
    args = parser.parse_args()
    assert len(args.gpus) == (1 if args.arm == 'C' else 2) and len(set(args.gpus)) == len(args.gpus)
    Runner(args.arm,args.gpus).run()
