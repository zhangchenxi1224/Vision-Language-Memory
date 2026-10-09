"""Build the frozen, hard-MCQ teacher bank from the registered new T1/T2/T3."""
# ruff: noqa: E402 -- establish repository imports and deterministic CUDA env first.
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from scripts.experiments.prefeval_k1_data import official_mcq, option_order, sha
from scripts.experiments.prefeval_aug4_assets import verify_asset_seal


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def atomic_save(path, value):
    import torch
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, temp)
    temp.replace(path)


def bound_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError(f'Resume binding differs: {path}')
    else:
        save_json(path, value)


def safe_id(pid):
    if not isinstance(pid, str) or not pid or any(c in pid for c in '/\\') or pid in {'.', '..'}:
        raise ValueError(f'Unsafe preference identifier: {pid!r}')
    return pid.replace(':', '_')


def load_data(path, split):
    artifact = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = artifact[split]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'Empty or malformed split: {split}')
    ids, safe = set(), set()
    for row in rows:
        pid = row['base_pair_id']
        key = safe_id(pid)
        if pid in ids or key in safe:
            raise ValueError('Duplicate preference identifier or filename collision')
        ids.add(pid)
        safe.add(key)
        if len(row['options']) != 4 or any(not isinstance(x, str) or not x.strip() for x in row['options']):
            raise ValueError('Exactly four nonempty choices required; the registered correct choice is index zero')
        if row.get('correct_option_index', 0) != 0:
            raise ValueError('Correct source option must be index zero')
        for family in (('T1', 'T2', 'T3') if split == 'train' else ('T1',)):
            if not row['teacher_question_forms'][family].strip():
                raise ValueError('Empty question')
        for family in (('W0', 'W1', 'W2', 'W3') if split == 'train' else ('W0',)):
            if not row['writer_user_forms'][family].strip():
                raise ValueError('Empty writer input')
    return rows


def select_shard(rows, shard, shards, limit=0):
    if shards < 1 or shard not in range(shards) or limit < 0:
        raise ValueError('Invalid shard specification')
    selected = rows[shard::shards]
    return selected[:limit] if limit else selected


def query_target(row, step, mcq):
    family = f'T{step % 3 + 1}'
    order, correct = option_order(row['base_pair_id'], step)
    query = row['teacher_question_forms'][family] + mcq['get_mcq_question_format']([row['options'][i] for i in order])
    return query, f'<choice>{"ABCD"[correct]}</choice>', order, family


def main(args):
    asset_seal_sha256 = verify_asset_seal(args)
    import torch
    from PIL import Image
    from diffusers import AutoencoderTiny
    from diffusers.image_processor import VaeImageProcessor
    from scripts.eval.prefeval_rgb import load_reader, read_png, append
    from scripts.train.latent_r11_vae_oracle import VAELatentOracle, _save_image
    from vision_memory.reader.qwen3vl import qwen3vl_target_only_ce, R3_QWEN_READER_RESIZE_CONTRACT
    from vision_memory.reader.open_eos import assistant_termination_contract
    from vision_memory.repro import configure_strict_cuda_determinism

    configure_strict_cuda_determinism(0)
    torch.set_num_threads(1)
    if args.steps != 288 and not args.technical_smoke:
        raise ValueError('Registered teacher budget is 288; other budgets require --technical-smoke')
    rows = select_shard(load_data(args.data, 'train'), args.shard, args.shards, args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    binding = {'schema': 'prefeval-aug4-teacher/v1', 'data_sha256': sha(args.data),
               'asset_seal_sha256': asset_seal_sha256,
               'steps': args.steps, 'lr': 0.05, 'arm': 'B', 'split': 'train',
               'base': str(args.base), 'reader': str(args.reader),
               'teacher_source_sha256': sha(Path(__file__)),
               'mcq_source_sha256': sha(args.prefeval / 'utils/utils_mcq.py'),
               'questions': 'registered_data.teacher_question_forms.T1,T2,T3',
               'loss': 'hard MCQ target CE including actual assistant terminator',
               'quantization': 'uint8-equivalent RGB forward; STE backward',
               'technical_smoke': bool(args.technical_smoke)}
    identity = {'binding': binding, 'shard': args.shard, 'shards': args.shards,
                'assignment': [r['base_pair_id'] for r in rows]}
    bound_json(args.output / f'identity-{args.shard}.json', identity)
    mcq = official_mcq(args.prefeval)
    vae = AutoencoderTiny.from_pretrained(args.base, subfolder='vae', local_files_only=True,
                                         torch_dtype=torch.float32).to(args.device).eval().requires_grad_(False)
    assert vae.config.latent_channels == 4 and vae.config.scaling_factor == 1 and vae.config.shift_factor == 0
    processor, reader = load_reader(args.reader, args.device)
    termination = assistant_termination_contract(reader, processor)
    bound_json(args.output / f'termination-{args.shard}.json', termination)
    native = VaeImageProcessor(vae_scale_factor=8).preprocess(Image.new('RGB', (1024, 1024), (128, 128, 128)))
    with torch.no_grad():
        initial = vae.encode(native.to(device=args.device, dtype=torch.float32)).latents.detach()
    assert tuple(initial.shape) == (1, 4, 128, 128)
    for index, row in enumerate(rows):
        pid = row['base_pair_id']
        out = args.output / safe_id(pid)
        out.mkdir(exist_ok=True)
        if (out / 'complete.json').exists():
            done = json.loads((out / 'complete.json').read_text(encoding='utf-8'))
            assert done['binding'] == binding and done['numeric_finite']
            assert done['png_sha256'] == sha(out / 'memory.png')
            assert done['latent_sha256'] == sha(out / 'latent.pt')
            continue
        oracle = VAELatentOracle(vae=vae, initial_latent=initial, compute_dtype=torch.float32)
        optimizer = torch.optim.Adam([oracle.latent_fp32], lr=.05, betas=(.9, .999), eps=1e-8)
        checkpoint, start = out / 'resume.pt', 0
        if checkpoint.exists():
            ck = torch.load(checkpoint, map_location=args.device, weights_only=False)
            assert ck['binding'] == binding and ck['pair_id'] == pid
            start = int(ck['step'])
            assert 0 <= start <= args.steps
            with torch.no_grad():
                oracle.latent_fp32.copy_(ck['latent'])
            optimizer.load_state_dict(ck['optimizer'])
        started = time.monotonic()
        for step in range(start, args.steps):
            optimizer.zero_grad(set_to_none=True)
            pixels = oracle.image()
            quantized = pixels + (pixels.mul(255).round().div(255) - pixels).detach()
            question, target, order, family = query_target(row, step, mcq)
            ce = qwen3vl_target_only_ce(model=reader, processor=processor, image=quantized[0],
                query=question, target=target + termination['assistant_end_token_text'], device=args.device,
                require_image_grad=True, reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
                deterministic_ce=True, preserve_generation_prefix=True)
            assert int(ce.target_ids[0, -1]) == termination['assistant_end_token_id']
            if not torch.isfinite(ce.loss):
                raise RuntimeError(f'Nonfinite teacher CE: {pid}, {step + 1}')
            ce.loss.backward()
            grad = oracle.latent_fp32.grad
            if grad is None or not torch.isfinite(grad).all() or not torch.any(grad != 0):
                raise RuntimeError(f'Missing/nonfinite/zero teacher latent gradient: {pid}')
            grad_norm = float(grad.norm())
            optimizer.step()
            if not torch.isfinite(oracle.latent_fp32).all():
                raise RuntimeError('Nonfinite teacher latent')
            record = {'pair_id': pid, 'step': step + 1, 'family': family, 'ce': float(ce.loss.detach()),
                      'grad_norm': grad_norm, 'option_order': order, 'elapsed_seconds': time.monotonic() - started}
            append(out / 'optimization.jsonl', record)
            if (step + 1) % 24 == 0 or step + 1 == args.steps:
                atomic_save(checkpoint, {'pair_id': pid, 'latent': oracle.latent_fp32.detach(),
                                        'optimizer': optimizer.state_dict(), 'step': step + 1, 'binding': binding})
                print(json.dumps({**record, 'item': index + 1, 'items': len(rows)}), flush=True)
        with torch.no_grad():
            pixels = oracle.image()
            if not torch.isfinite(pixels).all() or not torch.isfinite(oracle.latent_fp32).all():
                raise RuntimeError('Nonfinite teacher final artifact')
            _save_image(out / 'memory.png', pixels)
        atomic_save(out / 'latent.pt', oracle.latent_fp32.detach().cpu())
        assert tuple(read_png(out / 'memory.png').shape) == (3, 1024, 1024)
        save_json(out / 'complete.json', {'pair_id': pid, 'binding': binding,
                  'png_sha256': sha(out / 'memory.png'), 'latent_sha256': sha(out / 'latent.pt'),
                  'step': args.steps, 'numeric_finite': True, 'status': 'optimized_pending_png_semantic_readback'})
        print(json.dumps({'completed': pid}), flush=True)
        del oracle, optimizer, pixels
    save_json(args.output / f'finished-{args.shard}.json', {**identity, 'status': 'completed', 'items': len(rows)})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'base', 'reader', 'prefeval', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--steps', type=int, default=288)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=8)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--technical-smoke', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    try:
        main(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f'failed-{args.shard}.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
