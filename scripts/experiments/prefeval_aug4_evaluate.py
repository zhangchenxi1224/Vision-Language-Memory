"""Sharded first-write PNG generation and real-PNG MCQ readback for AUG4x3."""
# ruff: noqa: E402 -- establish repository imports and deterministic CUDA env first.
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from scripts.experiments.prefeval_k1_data import official_mcq, option_order, sha
from scripts.experiments.prefeval_aug4_teacher import load_data, select_shard, safe_id, save_json, bound_json
from scripts.experiments.prefeval_aug4_assets import verify_asset_seal

OFFICIAL_UNET_SHA = 'f983bd1710344cb44f7d12d05c0ca961984b48391dcbb095ad710609cbde3246'
KEY_FIELDS = ('pair_id', 'seed', 'writer_family', 'question_family', 'control', 'correct_position')


def question_suites(kind, split):
    if kind == 'teacher':
        if split != 'train':
            raise ValueError('Teacher targets and evaluation are restricted to train730')
        return [('teacher', family) for family in ('T1', 'T2', 'T3')]
    if split == 'diagnostics':
        return [('Wstar', 'T1'), ('W0', 'Tstar'), ('Wstar', 'Tstar')]
    return [('W0', 'T1')]


def donor_map(rows):
    groups = defaultdict(list)
    meanings = {}
    for row in rows:
        pid = row['base_pair_id']
        groups[row['topic']].append(pid)
        meanings[pid] = ' '.join(row['writer_user_forms']['W0'].casefold().split())
    result = {}
    for topic, peers in groups.items():
        peers.sort()
        if len(peers) < 2:
            raise ValueError(f'No distinct same-topic mismatch donor: {topic}')
        for i, pid in enumerate(peers):
            donor = next((peers[(i + offset) % len(peers)] for offset in range(1, len(peers))
                          if meanings[peers[(i + offset) % len(peers)]] != meanings[pid]), None)
            if donor is None:
                raise ValueError(f'No distinct same-topic preference text for mismatch donor: {pid}')
            result[pid] = donor
    return result


def mismatch_donors(rows, split):
    if split == 'opposites':
        # Each counterfactual has one unambiguous same-topic donor: its original preference.
        result = {r['base_pair_id']: r['origin_id'] for r in rows}
        if any(pid == donor for pid, donor in result.items()):
            raise ValueError('Counterfactual and original identifiers must differ')
        return result
    return donor_map(rows)


def eval_order(pid, family, position):
    # Identical answer rotation across Writer families, seeds, checkpoints and arms.
    family_index = {'T1': 0, 'T2': 1, 'T3': 2, 'Tstar': 0}[family]
    return option_order(pid, position * 3 + family_index)


def event_text(row, family):
    return 'user: ' + row['writer_user_forms'][family] + '\nassistant: Understood.'


def image_path(root, pid, family, seed):
    return Path(root) / safe_id(pid) / family / f'seed-{seed}' / 'memory.png'


def read_rows(path):
    """Recover a interrupted final JSONL write; earlier malformed data is an error."""
    if not Path(path).exists():
        return []
    raw = Path(path).read_bytes()
    lines, result, good = raw.splitlines(keepends=True), [], 0
    for i, line in enumerate(lines):
        try:
            result.append(json.loads(line))
            good += len(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i != len(lines) - 1:
                raise
            Path(str(path) + f'.interrupted-tail-{os.getpid()}').write_bytes(line)
            with Path(path).open('r+b') as stream:
                stream.truncate(good)
    return result


def checkpoint_identity(args):
    if args.kind == 'teacher':
        return None
    if not args.checkpoint:
        raise ValueError('Student generation and readback require --checkpoint')
    ready_path = args.checkpoint.with_suffix('.ready.json')
    if not ready_path.exists():
        raise ValueError(f'Checkpoint has no atomic ready receipt: {ready_path}')
    ready = json.loads(ready_path.read_text(encoding='utf-8'))
    checkpoint_sha = sha(args.checkpoint)
    if ready['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Checkpoint differs from its ready receipt')
    if args.split == 'official' and int(ready['step']) != 93440:
        raise ValueError('Official180 is evaluated only at the fixed final 93440-step endpoint')
    return {'checkpoint_sha256': checkpoint_sha, 'optimizer_step': int(ready['step'])}


def image_binding(binding, pid, family, seed):
    return {'schema': 'prefeval-aug4-png/v1', 'data_sha256': binding['data_sha256'],
            'asset_seal_sha256': binding.get('asset_seal_sha256'),
            'checkpoint': binding['checkpoint'], 'pair_id': pid, 'writer_family': family, 'seed': seed,
            'inference_steps': 28, 'cfg': 1.0, 'assistant_acknowledgment': 'Understood.',
            'evaluation_source_sha256': binding['evaluation_source_sha256']}


def verified_student_image(root, pid, family, seed, binding):
    path = image_path(root, pid, family, seed)
    receipt = json.loads(path.with_name('complete.json').read_text(encoding='utf-8'))
    if receipt['binding'] != image_binding(binding, pid, family, seed) or receipt['png_sha256'] != sha(path):
        raise ValueError(f'PNG binding or content changed: {path}')
    return path, receipt['png_sha256']


def generate(args, rows, selected, binding):
    import torch
    from PIL import Image
    from scripts.experiments.prefeval_k1_writer import encode_source
    from scripts.experiments.prefeval_k1_init_ablation import isolated_rng
    from scripts.train.latent_r11_vae_oracle import _save_image
    from vision_memory.dreamlite.native_base import NativeBaseEditSampler
    from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
    from vision_memory.training.latent_bank_unet import stable_seed, OFFICIAL_REFERENCE_COMMIT
    from vision_memory.training.checkpoint import load_trainable_weights

    if args.kind != 'student' or not args.base or not args.official_source:
        raise ValueError('generate requires student kind, --base and --official-source')
    actual_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.official_source, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=args.official_source, text=True).strip()
    if actual_commit != OFFICIAL_REFERENCE_COMMIT or dirty:
        raise ValueError('Official DreamLite code differs from the registered reference')
    if sha(args.base / 'unet/diffusion_pytorch_model.safetensors') != OFFICIAL_UNET_SHA:
        raise ValueError('Official Base U-Net artifact differs from registration')
    sys.path.insert(0, str(args.official_source))
    from dreamlite import DreamLitePipelineLoRA
    pipe = DreamLitePipelineLoRA.from_pretrained(args.base, local_files_only=True, torch_dtype=torch.float32).to(args.device)
    for module in (pipe.unet, pipe.vae, pipe.text_encoder):
        module.eval().requires_grad_(False)
    # Strict complete parameter replacement prevents any residual pretrained weights in R arms.
    pipe.unet.requires_grad_(True)
    payload = load_trainable_weights(args.checkpoint, trainable_module=pipe.unet)
    manifest = payload['manifest']
    if manifest['schema'] != 'prefeval-aug4-writer/v1' or manifest['data_sha256'] != binding['data_sha256']:
        raise ValueError('Checkpoint does not belong to this augmented experiment')
    if manifest.get('asset_seal_sha256') != binding.get('asset_seal_sha256'):
        raise ValueError('Checkpoint and evaluation model asset seals differ')
    if manifest['official_unet_sha256'] != OFFICIAL_UNET_SHA or manifest['init_mode'] not in {'pretrained', 'random'}:
        raise ValueError('Unregistered Writer initialization source')
    if int(payload['optimizer_step']) != binding['checkpoint']['optimizer_step']:
        raise ValueError('Checkpoint cursor differs from ready receipt')
    del payload
    pipe.unet.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    families = sorted({w for w, _ in question_suites(args.kind, args.split)})
    completed, reused = 0, 0
    for row in selected:
        pid = row['base_pair_id']
        for family in families:
            for seed in args.seeds:
                if args.split == 'diagnostics' and family == 'W0' and args.canonical_images:
                    verified_student_image(args.canonical_images, pid, family, seed, binding)
                    completed += 1
                    reused += 1
                    continue
                path = image_path(args.images, pid, family, seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.with_name('complete.json').exists():
                    verified_student_image(args.images, pid, family, seed, binding)
                    completed += 1
                    continue
                text = event_text(row, family)
                # W0/Wstar and all four models use the same source noise for each preference/seed.
                noise_seed = stable_seed(20261009, f'aug4-eval:{pid}:seed:{seed}', 0)
                conditioning_seed = stable_seed(20261009, f'aug4-condition:{pid}:{family}', 0)
                with torch.no_grad(), isolated_rng(conditioning_seed):
                    source_image = Image.new('RGB', (1024, 1024), (128, 128, 128))
                    source = encode_source(pipe, source_image, args.device)
                    generator = torch.Generator(device=args.device).manual_seed(noise_seed)
                    noise = torch.randn(source.shape, generator=generator, device=args.device, dtype=source.dtype)
                    sampler = NativeBaseEditSampler(pipe, source_image=source_image, event_text=text, guidance_scale=1.)
                    generated = sampler(source_latents=source, noise_latents=noise, num_steps=28, return_trajectory=False)
                    if not torch.isfinite(generated.latents).all():
                        raise RuntimeError('Nonfinite generated latent')
                    pixels = decode_model_latents_unit_interval(pipe.vae, generated.latents, clamp=True)
                    if not torch.isfinite(pixels).all():
                        raise RuntimeError('Nonfinite generated PNG pixels')
                    temp = path.with_name(f'memory.{os.getpid()}.png')
                    _save_image(temp, pixels)
                    temp.replace(path)
                save_json(path.with_name('complete.json'), {'binding': image_binding(binding, pid, family, seed),
                          'png_sha256': sha(path), 'numeric_finite': True, 'noise_seed': noise_seed,
                          'conditioning_seed': conditioning_seed})
                completed += 1
                print(json.dumps({'generated': pid, 'writer_family': family, 'seed': seed}), flush=True)
                del source, noise, sampler, generated, pixels
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    save_json(args.output / f'generate-finished-{args.shard}.json',
              {'status': 'completed', 'binding': binding, 'items': completed, 'reused_canonical_pngs': reused})


def planned_read_keys(rows, args):
    controls = args.controls
    seeds = [0] if args.kind == 'teacher' else args.seeds
    for row in rows:
        for writer_family, question_family in question_suites(args.kind, args.split):
            for seed in seeds:
                for control in controls:
                    if control in {'blank', 'text'} and seed != seeds[0]:
                        continue
                    for position in range(4):
                        yield (row['base_pair_id'], seed, writer_family, question_family, control, position)


def readback(args, rows, selected, binding):
    from PIL import Image
    from scripts.eval.prefeval_rgb import load_reader, read_png, append
    from scripts.experiments.prefeval_k1_evaluate import text_generate
    from vision_memory.reader.open_answer import generate_short_answer

    if not args.reader or not args.prefeval:
        raise ValueError('read requires --reader and --prefeval')
    donors = mismatch_donors(rows, args.split) if 'mismatch' in args.controls else {}
    mcq = official_mcq(args.prefeval)
    processor, reader = load_reader(args.reader, args.device)
    dest = args.output / f'readback-{args.shard}.jsonl'
    previous = read_rows(dest)
    completed = {tuple(record[key] for key in KEY_FIELDS) for record in previous}
    expected = set(planned_read_keys(selected, args))
    if len(completed) != len(previous) or not completed <= expected:
        raise ValueError('Duplicate or unexpected readback rows')
    if any(record['binding_sha256'] != binding_digest(binding) for record in previous):
        raise ValueError('Readback log is from another evaluation binding')
    counts = defaultdict(lambda: {'correct': 0, 'total': 0, 'parse_failures': 0})
    def count(record):
        for key in (record['control'], '/'.join(record[k] for k in ('writer_family', 'question_family', 'control'))):
            counts[key]['correct'] += int(record['correct'])
            counts[key]['total'] += 1
            counts[key]['parse_failures'] += int(record['parse_failure'])
    for record in previous:
        count(record)
    gray_path = args.output / f'blank-{args.shard}.png'
    if not gray_path.exists():
        Image.new('RGB', (1024, 1024), (128, 128, 128)).save(gray_path)
    blank = read_png(gray_path)
    by_id = {r['base_pair_id']: r for r in rows}
    cached_images = {}
    for key in planned_read_keys(selected, args):
        if key in completed:
            continue
        pid, seed, writer_family, question_family, control, position = key
        row = by_id[pid]
        source_id = donors[pid] if control == 'mismatch' else pid
        png_path, png_sha = None, None
        if control in {'memory', 'mismatch'}:
            image_key = (source_id, writer_family, seed)
            if image_key not in cached_images:
                if args.kind == 'teacher':
                    png_path = args.images / safe_id(source_id) / 'memory.png'
                    done = json.loads(png_path.with_name('complete.json').read_text(encoding='utf-8'))
                    if done['binding']['data_sha256'] != binding['data_sha256'] or done['step'] != 288 or done['binding'].get('technical_smoke'):
                        raise ValueError('Teacher PNG is not a registered final target')
                    if done['binding'].get('asset_seal_sha256') != binding.get('asset_seal_sha256'):
                        raise ValueError('Teacher and Reader model asset seals differ')
                    png_sha = sha(png_path)
                    assert png_sha == done['png_sha256'] and done['numeric_finite']
                else:
                    image_root = args.images
                    if args.split == 'diagnostics' and writer_family == 'W0' and args.canonical_images:
                        image_root = args.canonical_images
                    if args.split == 'opposites' and control == 'mismatch':
                        if not args.canonical_images:
                            raise ValueError('Counterfactual mismatch requires --canonical-images pointing to train PNGs')
                        image_root = args.canonical_images
                    png_path, png_sha = verified_student_image(image_root, source_id, writer_family, seed, binding)
                if len(cached_images) > 8:
                    cached_images.clear()
                cached_images[image_key] = (read_png(png_path), png_path, png_sha)
            pixels, png_path, png_sha = cached_images[image_key]
        else:
            pixels = blank
            if control == 'blank':
                png_path, png_sha = gray_path, sha(gray_path)
        question = row['teacher_question_forms'][question_family]
        order, correct = eval_order(pid, question_family, position)
        query = question + mcq['get_mcq_question_format']([row['options'][i] for i in order])
        if control == 'text':
            family = 'W0' if writer_family == 'teacher' else writer_family
            messages = [{'role': 'user', 'content': row['writer_user_forms'][family]},
                        {'role': 'assistant', 'content': 'Understood.'}, {'role': 'user', 'content': query}]
            generated = text_generate(reader, processor, messages, args.device, 32)
        else:
            # No preference, acknowledgment, answer or Writer input is passed into image readback.
            generated = generate_short_answer(model=reader, processor=processor, image=pixels,
                                               query=query, device=args.device, max_new_tokens=32)
        predicted = mcq['extract_choice'](generated['raw'])
        record = dict(zip(KEY_FIELDS, key))
        record.update({'binding_sha256': binding_digest(binding), 'split': args.split, 'kind': args.kind,
                       'png_path': str(png_path) if png_path else None, 'png_sha256': png_sha,
                       'donor_pair_id': source_id if control == 'mismatch' else None,
                       'reader_query': query, 'generated': generated, 'option_order': order,
                       'correct_letter': 'ABCD'[correct], 'predicted_letter': predicted,
                       'correct': predicted == 'ABCD'[correct], 'parse_failure': predicted is None,
                       'counterfactual_pair_id': row.get('origin_id'),
                       'new_correct_option_original_index': row.get('new_correct_option_original_index'),
                       'original_option_order': row.get('original_option_order'),
                       'official_duplicate_of_train': row.get('official_duplicate_of_train')})
        append(dest, record)
        count(record)
        completed.add(key)
        if len(completed) % 48 == 0:
            print(json.dumps({'readback_rows': len(completed), 'expected': len(expected), 'shard': args.shard}), flush=True)
    assert completed == expected
    save_json(args.output / f'read-finished-{args.shard}.json',
              {'status': 'completed', 'binding': binding, 'items': len(completed),
               'expected_items': len(expected), 'counts': dict(counts), 'readback_sha256': sha(dest)})


def binding_digest(binding):
    import hashlib
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main(args):
    asset_seal_sha256 = verify_asset_seal(args)
    import torch
    from vision_memory.repro import configure_strict_cuda_determinism
    configure_strict_cuda_determinism(0)
    torch.set_num_threads(1)
    args.seeds = [int(x) for x in args.seeds.split(',')]
    args.controls = args.controls.split(',')
    if args.seeds != [0, 1] or len(set(args.controls)) != len(args.controls) or not set(args.controls) <= {'memory', 'mismatch', 'blank', 'text'}:
        raise ValueError('Registered evaluation requires seeds 0,1 and valid unique controls')
    rows = load_data(args.data, args.split)
    selected = select_shard(rows, args.shard, args.shards, args.limit)
    suites = question_suites(args.kind, args.split)
    for row in rows:
        for writer_family, question_family in suites:
            assert row['teacher_question_forms'][question_family].strip()
            if writer_family != 'teacher':
                assert row['writer_user_forms'][writer_family].strip()
    args.output.mkdir(parents=True, exist_ok=True)
    binding = {'schema': 'prefeval-aug4-evaluation/v1', 'data_sha256': sha(args.data),
               'asset_seal_sha256': asset_seal_sha256,
               'evaluation_source_sha256': sha(Path(__file__)), 'kind': args.kind,
               'split': args.split, 'checkpoint': checkpoint_identity(args),
               'seeds': args.seeds, 'controls': args.controls, 'suites': [list(x) for x in suites],
               'correct_positions': [0, 1, 2, 3], 'shard': args.shard, 'shards': args.shards,
               'mismatch_protocol': ('counterfactual origin train PNG' if args.split == 'opposites' else
                   'next sorted same-topic id excluding identical normalized W0; semantic near-duplicates not fully excluded'),
               'assignment': [r['base_pair_id'] for r in selected], 'reader': str(args.reader),
               'canonical_images': str(args.canonical_images) if args.canonical_images else None,
               'mcq_source_sha256': sha(args.prefeval / 'utils/utils_mcq.py') if args.prefeval else None}
    # mode is intentionally excluded: generate and read must agree on the exact same identity.
    bound_json(args.output / f'identity-{args.shard}.json', binding)
    if args.mode == 'generate':
        generate(args, rows, selected, binding)
    else:
        readback(args, rows, selected, binding)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['generate', 'read'])
    p.add_argument('--kind', choices=['teacher', 'student'], required=True)
    p.add_argument('--split', choices=['train', 'dev', 'official', 'diagnostics', 'opposites'], required=True)
    for name in ('data', 'images', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('base', 'official-source', 'checkpoint', 'reader', 'prefeval', 'canonical-images'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seeds', default='0,1')
    p.add_argument('--controls', default='memory,mismatch,blank,text')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    try:
        main(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f'{args.mode}-failed-{args.shard}.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
