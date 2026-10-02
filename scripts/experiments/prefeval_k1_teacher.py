"""Paired K1 target latents using official answers or permuted official MCQ."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from PIL import Image
from diffusers import AutoencoderTiny
from diffusers.image_processor import VaeImageProcessor
from scripts.experiments.prefeval_k1_data import load_training_records, official_mcq, option_order, sha, REPORT
from scripts.eval.prefeval_rgb import load_reader, read_png, append
from scripts.train.latent_r11_vae_oracle import VAELatentOracle, _save_image
from vision_memory.reader.qwen3vl import qwen3vl_target_only_ce, R3_QWEN_READER_RESIZE_CONTRACT
from vision_memory.reader.open_eos import assistant_termination_contract
from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.repro import configure_strict_cuda_determinism

def save_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)

def atomic_save(path, value):
    temp = path.with_suffix('.pt.tmp')
    torch.save(value, temp)
    temp.replace(path)

def query_target(row, arm, step, mcq):
    question = row['forms'][f'T{step % 3 + 1}']
    if arm == 'A':
        return question + ' (Please respond within 300 words.)', row['target']['content'], None
    order, correct = option_order(row['base_pair_id'], step)
    query = question + mcq['get_mcq_question_format']([row['options'][i] for i in order])
    return query, f'<choice>{"ABCD"[correct]}</choice>', order

def main(args):
    configure_strict_cuda_determinism(0)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_training_records(args.split, args.exclude_pilot)[args.shard::args.shards]
    if args.limit:
        rows = rows[:args.limit]
    execution_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    recovery_commit = args.boundary_recovery_from
    if recovery_commit:
        assert args.arm == 'A' and args.split == 'train' and args.exclude_pilot and args.steps == 288
        assert len(recovery_commit) == 40 and all(c in '0123456789abcdef' for c in recovery_commit)
        assert (args.output / f'identity-{args.shard}.json').exists(), 'Recovery requires an existing registered identity'
    binding = {'arm': args.arm, 'steps': args.steps, 'lr': .05,
        'forms_sha256': sha(REPORT / ('train-question-forms.json' if args.split == 'train' else 'question-forms.json')),
        'split': args.split, 'exclude_pilot': args.exclude_pilot,
        'mcq_source_sha256': sha(args.prefeval / 'utils/utils_mcq.py'),
        'assignment': [r['base_pair_id'] for r in rows], 'shard': args.shard, 'shards': args.shards,
        'commit': recovery_commit or execution_commit,
        'loss': 'mean over answer tokens including one actual assistant terminator',
        'quantization': 'uint8-equivalent RGB forward with STE backward',
        'budget_class': 'technical_smoke' if args.steps != 288 else ('registered_train730' if args.split == 'train' else 'registered_pilot')}
    ident = args.output / f'identity-{args.shard}.json'
    if ident.exists():
        assert json.loads(ident.read_text()) == binding, 'Resume binding differs'
    else:
        save_json(ident, binding)
    execution = {'execution_commit': execution_commit, 'registered_binding_commit': binding['commit'],
                 'preserve_generation_prefix_on_retokenization': bool(recovery_commit)}
    save_json(args.output / f'execution-{args.shard}-{execution_commit[:7]}.json', execution)
    mcq = official_mcq(args.prefeval)
    vae = AutoencoderTiny.from_pretrained(args.base, subfolder='vae', local_files_only=True,
                                         torch_dtype=torch.float32).to(args.device).eval().requires_grad_(False)
    assert vae.config.latent_channels == 4 and vae.config.scaling_factor == 1 and vae.config.shift_factor == 0
    processor, reader = load_reader(args.reader, args.device)
    termination = assistant_termination_contract(reader, processor)
    save_json(args.output / f'termination-{args.shard}.json', termination)
    native = VaeImageProcessor(vae_scale_factor=8).preprocess(Image.new('RGB', (1024, 1024), (128, 128, 128)))
    with torch.no_grad():
        initial = vae.encode(native.to(device=args.device, dtype=torch.float32)).latents.detach()
    assert tuple(initial.shape) == (1, 4, 128, 128)
    for index, row in enumerate(rows):
        pid = row['base_pair_id']
        out = args.output / pid.replace(':', '_')
        out.mkdir(exist_ok=True)
        if (out / 'complete.json').exists():
            done = json.loads((out / 'complete.json').read_text())
            assert done['binding'] == binding and done['png_sha256'] == sha(out / 'memory.png')
            continue
        oracle = VAELatentOracle(vae=vae, initial_latent=initial, compute_dtype=torch.float32)
        optimizer = torch.optim.Adam([oracle.latent_fp32], lr=.05, betas=(.9,.999), eps=1e-8)
        checkpoint = out / 'resume.pt'
        start = 0
        if checkpoint.exists():
            ck = torch.load(checkpoint, map_location=args.device, weights_only=False)
            assert ck['binding'] == binding
            with torch.no_grad():
                oracle.latent_fp32.copy_(ck['latent'])
            optimizer.load_state_dict(ck['optimizer'])
            start = ck['step']
        started = time.monotonic()
        for step in range(start, args.steps):
            optimizer.zero_grad(set_to_none=True)
            pixels = oracle.image()
            quantized = pixels + (pixels.mul(255).round().div(255) - pixels).detach()
            question, target, order = query_target(row, args.arm, step, mcq)
            ce = qwen3vl_target_only_ce(model=reader, processor=processor, image=quantized[0],
                query=question, target=target + termination['assistant_end_token_text'], device=args.device,
                require_image_grad=True, reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
                deterministic_ce=True, preserve_generation_prefix=bool(recovery_commit))
            assert int(ce.target_ids[0, -1]) == termination['assistant_end_token_id']
            if not torch.isfinite(ce.loss):
                raise RuntimeError('Nonfinite CE')
            ce.loss.backward()
            grad = oracle.latent_fp32.grad
            if grad is None or not torch.isfinite(grad).all() or not torch.any(grad != 0):
                raise RuntimeError('Missing/nonfinite/zero latent gradient')
            grad_norm = float(grad.norm())
            optimizer.step()
            if not torch.isfinite(oracle.latent_fp32).all():
                raise RuntimeError('Nonfinite latent')
            record = {'pair_id': pid, 'arm': args.arm, 'step': step + 1,
                'family': f'T{step % 3 + 1}', 'ce': float(ce.loss.detach()),
                'answer_tokens_with_eos': ce.target_ids.numel(), 'grad_norm': grad_norm,
                'option_order': order, 'elapsed_seconds': time.monotonic() - started}
            append(out / 'optimization.jsonl', record)
            if (step + 1) % 24 == 0 or step + 1 == args.steps:
                atomic_save(checkpoint, {'latent': oracle.latent_fp32.detach(),
                    'optimizer': optimizer.state_dict(), 'step': step + 1, 'binding': binding, 'execution': execution})
                print(json.dumps({**record, 'item': index + 1, 'items': len(rows)}), flush=True)
        with torch.no_grad():
            _save_image(out / 'memory.png', oracle.image())
        atomic_save(out / 'latent.pt', oracle.latent_fp32.detach().cpu())
        # Confirm the actual artifact can be decoded; no OOD inference during training.
        assert tuple(read_png(out / 'memory.png').shape) == (3, 1024, 1024)
        save_json(out / 'complete.json', {'pair_id': pid, 'binding': binding, 'execution': execution,
            'png_sha256': sha(out / 'memory.png'), 'latent_sha256': sha(out / 'latent.pt'),
            'step': args.steps, 'status': 'optimized_not_yet_semantically_evaluated'})
        print(json.dumps({'completed': pid, 'arm': args.arm}), flush=True)
        del oracle, optimizer, ce, pixels, quantized
    save_json(args.output / f'finished-{args.shard}.json', {'binding': binding, 'execution': execution, 'status': 'completed'})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['A', 'B'], required=True)
    parser.add_argument('--split', choices=['pilot', 'train'], default='pilot')
    parser.add_argument('--exclude-pilot', action='store_true')
    for name in ['base', 'reader', 'prefeval', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=288)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--boundary-recovery-from', help='Registered full commit SHA for the diagnosed A/train prefix-boundary recovery; execution SHA is recorded separately')
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f'failed-{args.shard}.txt').write_text(traceback.format_exc())
        raise
