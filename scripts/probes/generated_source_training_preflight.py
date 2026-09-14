"""Read-only GPU runtime and gradient checks of the exact ef163b2 training implementation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TRAINING_COMMIT = 'ef163b26e33f62c496ed0da8744ebb7bf1163873'
PARAMETERS_SHA = '16bf39236d6828dc9bc7ed7168f7e10092ccc9ea4868647f9f7c62b0d6a73975'
RUNTIME_SHA = 'b97bf55f679cb94805ef769b9e748f09a55182a8f36fb9dcb48c6fcfb7bdd1a0'


def selected_draws(bank):
    from vision_memory.training.latent_bank_unet import balanced_draw
    from scripts.train.generated_source_augmentation import source_variant_index
    selected = {}
    for index in range(19328):
        group, _, _, _ = balanced_draw(bank['groups'], 20260915, index, sampling_strategy='logical_condition')
        if group.get('source_kind') == 'sealed_rgb_1024':
            variant = source_variant_index(20260915, index, group['question_id'])
            selected.setdefault((group['source_state'], variant), index)
        if len(selected) == 27:
            break
    if set(selected) != {(state, index) for state in ('ambient', 'jazz', 'clear') for index in range(9)}:
        raise ValueError('Require all three source states and all nine source choices')
    return sorted(selected.values())


def clean(path, expected):
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip() != expected
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=path, text=True).strip()):
        raise ValueError('Require both exact immutable clean source roots')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-source-root', type=Path, required=True)
    p.add_argument('--expected-probe-commit', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    probe_root = Path(__file__).resolve().parents[2]
    clean(probe_root, a.expected_probe_commit)
    clean(a.training_source_root, TRAINING_COMMIT)
    # Resolve every training dependency from the immutable training checkout,
    # not from this reporting/probe checkout.
    sys.path[:0] = [str(a.training_source_root), str(a.training_source_root / 'src')]
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    if not a.worker:
        return subprocess.call([sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
            env={**os.environ, **REQUIRED_DETERMINISM_ENV, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    import math
    if not math.isfinite(a.deadline_unix) or time.time() + 10*60 > a.deadline_unix:
        raise ValueError('Reserve ten minutes for the bounded runtime probe')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('Existing GPU work must not be evicted')
    a.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (a.output / name).write_bytes((json.dumps(value, indent=2, sort_keys=True)+'\n').encode())
    def deadline(signum, frame):
        raise TimeoutError('Runtime probe deadline; no optimizer update was permitted')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(int(a.deadline_unix-time.time()))
    runs = Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/dreamlite-official-alignment')
    models = Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')
    from scripts.train import train_latent_bank_unet as training
    from scripts.experiments.generated_source_training_protocol import POOL_MANIFEST_SHA
    from scripts.experiments.generated_source_pool_protocol import PACKAGE_SHA
    from vision_memory.training.writer_initialization import initial_writer_binding, apply_initial_writer
    from vision_memory.training.latent_bank_unet import load_teacher_bank, file_sha256
    from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism
    import torch
    torch.set_num_threads(1)
    args = training.parser().parse_args(['--bank-manifest', str(runs/'84cdfdb-broader151-full4832/bank/manifest.json'),
        '--output-dir', str(a.output), '--model-variant', 'base', '--dreamlite', str(models/'DreamLite-base-a9a0f15-20260907'),
        '--teacher-dreamlite', str(models/'DreamLite-mobile'), '--reader-model', str(models/'Qwen3-VL-4B-Instruct'),
        '--official-source', str(runs.parents[1]/'Vision-Language-Memory/third_party/DreamLite'),
        '--base-manifest', str(runs/'base-complete-snapshot-seal.json'), '--base-guidance-scale', '1',
        '--dreamlite-device', 'cuda:0', '--reader-device', 'cuda:0', '--colocate-models', '--trainable-scope', 'full_unet',
        '--flow-protocol', 'official', '--prompt-style', 'native_base', '--seed', '20260915', '--steps', '4832', '--eval-seeds', '2',
        '--lr', '1e-5', '--gradient-accumulation-steps', '4', '--sampling-strategy', 'logical_condition',
        '--historical-wording-augmentation', '--generated-source-pool', str(runs/'90b41a2-generated-source-pool/manifest.json'),
        '--generated-source-pool-sha256', POOL_MANIFEST_SHA, '--initial-writer-package', str(runs/'e372f3c-logical-package'),
        '--initial-writer-package-sha256', PACKAGE_SHA, '--expected-commit', TRAINING_COMMIT, '--target-mode', 'single'])
    try:
        save('status.json', {'state': 'running', 'stage': 'load_exact_training_runtime', 'time_unix': time.time(),
            'pid': os.getpid(), 'training_commit': TRAINING_COMMIT, 'probe_commit': a.expected_probe_commit, 'optimizer_updates': 0})
        configure_strict_cuda_determinism(args.seed)
        bank, teachers = load_teacher_bank(args.bank_manifest)
        from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
        os.environ.update(snapshot_environment(bank))
        indices = selected_draws(bank)
        save('selected-draws.json', {'rule': 'First actual draw for each source-state/source-choice combination; fixed before loading model or observing losses.', 'indices': indices})
        runtime = training.load_runtime(args, bank)
        binding = {'snapshots': runtime['snapshots'], 'termination': runtime['termination'],
            'additional_protocol_binding': runtime.get('protocol_binding', {}),
            'scheduler_config': dict(runtime['pipe'].scheduler.config),
            'effective_inference_sigmas': {k: list(v['effective_sigmas']) for k,v in runtime['contexts'].items()},
            'condition_sha256': {k: canonical_tensor_sha256(v['condition'].prompt_embeds) for k,v in runtime['contexts'].items()}}
        training.write_json(a.output/'runtime.json', binding)
        if file_sha256(a.output/'runtime.json') != RUNTIME_SHA:
            raise ValueError('Source-cache preparation changed the canonical runtime')
        source_bindings = runtime['generated_source_augmentation_binding']
        save('training-source-augmentation.json', source_bindings)
        if len(source_bindings) != 108 or any(len(v) != 9 for v in source_bindings.values()):
            raise ValueError('Incomplete actual native source conditioning coverage')
        initial = initial_writer_binding(args)
        apply_initial_writer(args, runtime, initial)
        pipe, reader = runtime['pipe'], runtime['reader']
        frozen = training.frozen_versions(pipe, reader)
        def parameters_sha():
            digest = hashlib.sha256()
            for name, parameter in pipe.unet.named_parameters():
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()
        before = parameters_sha()
        if before != PARAMETERS_SHA:
            raise ValueError('Probe did not load the fixed4f parameters')
        rows = []
        for index in indices:
            for parameter in pipe.unet.parameters():
                parameter.grad = None
            row = training.flow_microbatch(args, runtime, bank['groups'], teachers, index)
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in pipe.unet.parameters()):
                raise ValueError('Actual full-U-Net source-variant gradient is missing or nonfinite')
            rows.append({'draw_index': index, **row})
            save('gradient-draws.json', rows)
        for parameter in pipe.unet.parameters():
            parameter.grad = None
        if parameters_sha() != before or training.frozen_versions(pipe, reader) != frozen:
            raise ValueError('Read-only runtime probe changed model parameters')
        runtime['verify_additional_bindings']()
        clean(a.training_source_root, TRAINING_COMMIT)
        result = {'status': 'preflight_passed', 'optimizer_updates': 0, 'actual_gradient_draws': len(rows),
            'source_conditions': 108, 'source_condition_pairs': 972, 'canonical_runtime_sha256': RUNTIME_SHA,
            'training_commit': TRAINING_COMMIT, 'probe_commit': a.expected_probe_commit, 'parameters_before_after_sha256': before,
            'source_seal_sha256': file_sha256(a.output/'training-source-augmentation.json'),
            'gradient_draws_sha256': file_sha256(a.output/'gradient-draws.json'),
            'scope': 'One-GPU exact training-runtime/source-encoding and finite-gradient probe. No optimizer, no baseline generation, no four-rank parity or functional acceptance claim.'}
        save('result.json', result)
        save('status.json', {'state': 'completed', 'time_unix': time.time(), **result})
    except BaseException as error:
        save('status.json', {'state': 'failed', 'time_unix': time.time(), 'error': repr(error), 'optimizer_updates': 0})
        raise
    finally:
        signal.alarm(0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
