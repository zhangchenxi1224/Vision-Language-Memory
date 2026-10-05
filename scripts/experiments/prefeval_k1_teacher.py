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
from scripts.experiments.prefeval_prompt_matching import (
    SUPERVISIONS, validate_options, supervision_binding, get_history_target,
)
from vision_memory.reader.prompt_matching import (
    qwen3vl_continuation_logits, soft_target_cross_entropy, soft_target_kl_divergence,
)

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
    supervision = validate_options(args)
    if args.shards <= 0 or args.shard not in range(args.shards) or args.steps <= 0:
        raise ValueError('Invalid teacher shard or optimization budget')
    configure_strict_cuda_determinism(0)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_training_records(args.split, args.exclude_pilot)[args.shard::args.shards]
    if args.limit:
        rows = rows[:args.limit]
    if getattr(args, 'ids_file', None):
        from scripts.experiments.prefeval_multitarget_bank import select_rows
        # Select before sharding, so changing shard count cannot alter the population.
        all_rows = select_rows(load_training_records(args.split, args.exclude_pilot), args.ids_file)
        rows = all_rows[args.shard::args.shards]
        if args.limit:
            rows = rows[:args.limit]
    context_suite = getattr(args, 'context_suite', 'original')
    snapshot_steps = sorted(set(int(x) for x in getattr(args, 'snapshot_steps', '').split(',') if x))
    if any(s <= 0 or s > args.steps or s % 24 for s in snapshot_steps):
        raise ValueError('Snapshots must be positive checkpoint boundaries (multiples of 24) within budget')
    if context_suite != 'original' and (supervision == 'hard_ce' or args.arm != 'B'):
        raise ValueError('Diverse contexts require B format and a history teacher')
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
    binding.update(supervision_binding(args, rows))
    if context_suite != 'original' or snapshot_steps or getattr(args, 'ids_file', None):
        from scripts.experiments.prefeval_context_coverage import context_manifest
        binding['context_suite'] = context_suite
        binding['snapshot_steps'] = snapshot_steps
        binding['ids_sha256'] = sha(args.ids_file) if args.ids_file else None
        if context_suite != 'original':
            binding['contexts'] = context_manifest(rows, args.steps, official_mcq(args.prefeval))
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
    reference = torch.full((3, 1024, 1024), 128 / 255, dtype=torch.float32, device=args.device)
    cache_root = getattr(args, 'teacher_cache', None) or args.output / 'teacher-distributions'
    for index, row in enumerate(rows):
        pid = row['base_pair_id']
        out = args.output / pid.replace(':', '_')
        out.mkdir(exist_ok=True)
        if (out / 'complete.json').exists():
            done = json.loads((out / 'complete.json').read_text())
            assert done['binding'] == binding and done['png_sha256'] == sha(out / 'memory.png')
            if supervision != 'hard_ce':
                if done['latent_sha256'] != sha(out / 'latent.pt') or done['teacher_targets_sha256'] != sha(out / 'teacher-targets.json'):
                    raise ValueError('Completed history teacher artifacts changed')
            continue
        oracle = VAELatentOracle(vae=vae, initial_latent=initial, compute_dtype=torch.float32)
        history_targets = {}
        receipts_path = out / 'teacher-targets.json'
        target_receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
        if supervision != 'hard_ce':
            for name, receipt in target_receipts.items():
                if Path(name).name != name or sha(cache_root / name) != receipt['cache_sha256']:
                    raise ValueError('Previously used teacher cache changed before resume')
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
        ce = pixels = quantized = objective = None
        started = time.monotonic()
        for step in range(start, args.steps):
            optimizer.zero_grad(set_to_none=True)
            pixels = oracle.image()
            quantized = pixels + (pixels.mul(255).round().div(255) - pixels).detach()
            family = f'T{step % 3 + 1}'
            if context_suite == 'original':
                question, target, order = query_target(row, args.arm, step, mcq)
            else:
                from scripts.experiments.prefeval_context_coverage import training_context
                question, family, order = training_context(row, step, mcq)
                target = None
            diagnostics = {}
            if supervision == 'hard_ce':
                ce = qwen3vl_target_only_ce(model=reader, processor=processor, image=quantized[0],
                    query=question, target=target + termination['assistant_end_token_text'], device=args.device,
                    require_image_grad=True, reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
                    deterministic_ce=True, preserve_generation_prefix=bool(recovery_commit))
                objective = ce.loss
            else:
                if question not in history_targets:
                    teacher, teacher_path = get_history_target(
                        cache_root=cache_root, binding=binding, row=row, query=question,
                        model=reader, processor=processor, reference=reference, device=args.device,
                        assistant_end_token_id=termination['assistant_end_token_id'])
                    history_targets[question] = teacher
                    receipt = {'cache_sha256': sha(teacher_path), 'target_ids': teacher['target_ids'],
                               'logits_sha256': teacher['logits_sha256'], 'binding': teacher['binding']}
                    if teacher_path.name in target_receipts and target_receipts[teacher_path.name] != receipt:
                        raise ValueError('Teacher cache changed during resume')
                    target_receipts[teacher_path.name] = receipt
                    save_json(receipts_path, target_receipts)
                teacher = history_targets[question]
                ce = qwen3vl_continuation_logits(
                    model=reader, processor=processor, image=quantized[0], query=question,
                    target_ids=torch.tensor([teacher['target_ids']], dtype=torch.long),
                    device=args.device, require_image_grad=True,
                    reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
                if supervision == 'prompt_matching':
                    teacher_logits = teacher['logits'].to(device=ce.target_logits.device)
                    objective = soft_target_cross_entropy(ce.target_logits, teacher_logits,
                                                          temperature=args.temperature)
                    with torch.no_grad():
                        diagnostics['teacher_student_kl'] = float(soft_target_kl_divergence(
                            ce.target_logits.detach(), teacher_logits, temperature=args.temperature))
                    del teacher_logits
                else:
                    objective = ce.loss
                diagnostics.update(supervision=supervision, objective_loss=float(objective.detach()),
                                   generated_hard_ce=float(ce.loss.detach()))
            assert int(ce.target_ids[0, -1]) == termination['assistant_end_token_id']
            if not torch.isfinite(objective):
                raise RuntimeError('Nonfinite teacher objective')
            objective.backward()
            grad = oracle.latent_fp32.grad
            if grad is None or not torch.isfinite(grad).all() or not torch.any(grad != 0):
                raise RuntimeError('Missing/nonfinite/zero latent gradient')
            grad_norm = float(grad.norm())
            optimizer.step()
            if not torch.isfinite(oracle.latent_fp32).all():
                raise RuntimeError('Nonfinite latent')
            record = {'pair_id': pid, 'arm': args.arm, 'step': step + 1,
                'family': family, 'ce': float(ce.loss.detach()),
                'answer_tokens_with_eos': ce.target_ids.numel(), 'grad_norm': grad_norm,
                'option_order': order, 'elapsed_seconds': time.monotonic() - started}
            record.update(diagnostics)
            append(out / 'optimization.jsonl', record)
            if step + 1 in snapshot_steps:
                snap = out / 'snapshots' / f'step-{step + 1:04d}'
                snap.mkdir(parents=True, exist_ok=True)
                with torch.no_grad():
                    _save_image(snap / 'memory.png', oracle.image())
                atomic_save(snap / 'latent.pt', oracle.latent_fp32.detach().cpu())
                save_json(snap / 'snapshot.json', {'step': step + 1, 'pair_id': pid,
                    'binding': binding, 'png_sha256': sha(snap / 'memory.png'),
                    'latent_sha256': sha(snap / 'latent.pt')})
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
            'step': args.steps, 'status': 'optimized_not_yet_semantically_evaluated',
            **({'teacher_targets_sha256': sha(receipts_path)} if supervision != 'hard_ce' else {})})
        print(json.dumps({'completed': pid, 'arm': args.arm}), flush=True)
        del oracle, optimizer, ce, pixels, quantized, objective, history_targets
    save_json(args.output / f'finished-{args.shard}.json', {'binding': binding, 'execution': execution, 'status': 'completed'})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['A', 'B'], required=True)
    parser.add_argument('--supervision', choices=SUPERVISIONS, default='hard_ce')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--teacher-max-new-tokens', type=int, default=512)
    parser.add_argument('--teacher-cache', type=Path,
                        help='Shared history target cache for history_hard and prompt_matching')
    parser.add_argument('--split', choices=['pilot', 'train'], default='pilot')
    parser.add_argument('--exclude-pilot', action='store_true')
    for name in ['base', 'reader', 'prefeval', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--context-suite', choices=['original', 'diverse-v1'], default='original')
    parser.add_argument('--ids-file', type=Path)
    parser.add_argument('--snapshot-steps', default='')
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
