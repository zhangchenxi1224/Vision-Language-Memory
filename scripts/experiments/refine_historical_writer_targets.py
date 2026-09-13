"""Refine all16 fixed hash-selected targets with the proven three-prompt recipe.

This is latent-only teacher optimization. It does not train a shared Writer.
Historical diagnostic queries are known, not untouched research holdouts.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
BANK_SHA = 'd54895adb15b91c3befd52c57f6248c76cf2abadb44916af830726a0736e688c'
TRAIN_PROMPTS = ('original_open', 'paraphrase_1', 'paraphrase_2')
FORMS = ('fp32_vae_decoded', 'rgb_uint8')
P = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
M = Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')


def selection(bank):
    from vision_memory.training.latent_bank_unet import member_split
    groups = sorted(bank['groups'], key=lambda group: group['historical_target_index'])
    if [group['historical_target_index'] for group in groups] != list(range(16)) or len(bank['teachers']) != 64:
        raise ValueError('Require all16 original questions and all64 retained teachers')
    teachers = {teacher['teacher_id']: teacher for teacher in bank['teachers']}
    result = []
    for group in groups:
        if len(group['teacher_ids']) != 4 or group['question_instruction_contract'] != 'historical-r11-five-prompts/v1':
            raise ValueError('Historical target membership or query protocol changed')
        teacher = teachers[member_split(group['teacher_ids'])[0][0]]
        result.append({'target_index': group['historical_target_index'], 'question_id': group['question_id'],
            'teacher_id': teacher['teacher_id'], 'latent_sha256': teacher['latent_sha256'],
            'historical_checkpoint_sha256': teacher['historical_checkpoint_sha256']})
    return result


def plan(bank):
    return {'schema': 'historical-fixed-target-refinement/v1', 'parent_bank_sha256': BANK_SHA,
        'selection_rule': 'existing member_split hash order; first training member for every question; no outcome substitution',
        'selected': selection(bank), 'questions': 16, 'old_teachers_preserved': 64,
        'initialization': 'fixed original step256 latent; fresh optimizer and seed0; not a fresh latent',
        'parent_latent_updates': 256, 'additional_updates_per_target': 256, 'cumulative_updates_per_target': 512,
        'optimizer': {'kind': 'Adam', 'lr': .05, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0.},
        'training_prompts': list(TRAIN_PROMPTS), 'prompt_schedule': 'round_robin_zero_based',
        'loss': 'mean_answer_ce + eos_ce', 'vae_dtype': 'float32', 'reader_dtype': 'bfloat16',
        'qualification': {'forms': list(FORMS), 'prompts': ['original_open', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3', 'paraphrase_4'],
            'generation': 'greedy raw32tokens; exact gold tokens followed immediately by EOS151645',
            'controls': ['blank', 'fixed_donor'], 'require_all16': True},
        'lane_assignment': 'target_index modulo2; independent targets; no gradient synchronization',
        'writer_updates': 0, 'scope': 'Previously observed diagnostic questions; no untouched holdout claim. Teacher qualification is not shared-Writer success.'}


def summarize(rows, group):
    expected = {(form, condition, prompt) for form in FORMS for condition in ('matched', 'blank', 'fixed_donor')
                for prompt in group['question_variants']}
    seen, matched = set(), []
    for row in rows:
        key = row['image_form'], row['condition'], row['prompt_id']
        if key not in expected or key in seen or row['query'] != group['question_variants'][key[2]]:
            raise ValueError('Unexpected, repeated or changed qualification cell')
        if row['target_index'] != group['historical_target_index'] or row['gold'] != group['answer'] or row['optimizer_step'] != 256:
            raise ValueError('Qualification target or fixed endpoint changed')
        seen.add(key)
        passed = row['generated_token_ids'] == row['scorer']['gold_token_ids'] + [151645]
        if passed != bool(row['scorer']['strict_correct'] and row['scorer']['answer_followed_immediately_by_eos']):
            raise ValueError('Raw tokens disagree with the strict score')
        if row['condition'] == 'matched':
            matched.append(passed)
    if seen != expected or len(expected) != 30 or len(matched) != 10:
        raise ValueError('Incomplete two-form five-query qualification')
    return {'raw_rows': len(rows), 'matched_rows': 10, 'correct_eos': sum(matched),
            'both_forms_all_five': all(matched), 'writer_success': False}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def clean_source(commit):
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()):
        raise ValueError('Require the exact clean refinement source')


def snapshot_bindings():
    from scripts.reporting.collect_transition_endpoint import sha, read
    from vision_memory.repro.hf_snapshot import verify_download_seal
    from scripts.inspire.model_snapshot_manifest import verify_snapshot_manifest
    base = M / 'DreamLite-base-a9a0f15-20260907'
    result = {'base': verify_download_seal(P / 'runs/dreamlite-official-alignment/base-complete-snapshot-seal.json', base)}
    if result['base']['revision'] != 'a9a0f151ffd99d3c37f3fd0472f5e8f1b31215aa':
        raise ValueError('Wrong Base revision')
    for key, folder, repo, revision, digest in (
        ('mobile', 'DreamLite-mobile', 'carlofkl/DreamLite-mobile', '6695c3f4be230f0493fa5dbf78be3bc4d3bb2ab4',
         '1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159'),
        ('reader', 'Qwen3-VL-4B-Instruct', 'Qwen/Qwen3-VL-4B-Instruct', 'ebb281ec70b05090aa6165b016eac8ec08e71b17',
         '159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c')):
        result[key] = verify_snapshot_manifest(manifest_path=M / folder / '.snapshot_manifest.json',
            model_dir=M / folder, expected_repo_id=repo, expected_revision=revision)
        if result[key]['manifest_sha256'] != digest:
            raise ValueError('Historical model manifest changed: ' + key)
    mobile = M / 'DreamLite-mobile'
    if sha(base / 'vae/diffusion_pytorch_model.safetensors') != sha(mobile / 'vae/diffusion_pytorch_model.safetensors'):
        raise ValueError('Base and historical VAE weights differ')
    configs = [{k: v for k, v in read(root / 'vae/config.json').items() if not k.startswith('_')} for root in (base, mobile)]
    if configs[0] != configs[1]:
        raise ValueError('Base and historical VAE coordinates differ')
    return result


def worker(a, bank):
    import torch
    from diffusers import AutoencoderTiny
    from PIL import Image
    from types import SimpleNamespace
    from scripts.experiments import direct_geometry_eos_training as oracle
    from scripts.train.r11_new_frozen_dreamlite_oracle import _load_reader
    from scripts.reporting.collect_transition_endpoint import sha, read, jsonl
    from vision_memory.training.latent_bank_unet import load_teacher_bank
    from vision_memory.repro import configure_strict_cuda_determinism, canonical_tensor_sha256
    from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
    from vision_memory.reader.open_eos import assistant_termination_contract
    configure_strict_cuda_determinism(0)
    if torch.cuda.device_count() != 1 or 'H200' not in torch.cuda.get_device_name(0) or torch.cuda.mem_get_info(0)[0] < 100 * 1024**3:
        raise ValueError('Each refinement lane requires its assigned free H200')
    snapshots = snapshot_bindings()
    loaded, tensors = load_teacher_bank(a.bank)
    if loaded != bank or sha(a.bank) != BANK_SHA:
        raise ValueError('Original bank changed')
    if (bank['snapshots']['dreamlite_mobile'] != snapshots['mobile']
            or bank['snapshots']['qwen_reader'] != snapshots['reader']):
        raise ValueError('Teacher bank and real frozen model identities differ')
    device = torch.device('cuda:0')
    vae = AutoencoderTiny.from_pretrained(M / 'DreamLite-base-a9a0f15-20260907', subfolder='vae',
        local_files_only=True, torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
    processor, reader = _load_reader(SimpleNamespace(reader=M / 'Qwen3-VL-4B-Instruct'), device, torch.bfloat16)
    termination = assistant_termination_contract(reader, processor)
    if termination['assistant_end_token_id'] != 151645:
        raise ValueError('Reader termination changed')

    def versions():
        result = {}
        for label, module in (('vae', vae), ('reader', reader)):
            if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise ValueError('Only target latent may be trainable')
            result.update({label + '.' + name: p._version for name, p in module.named_parameters()})
        return result

    frozen = versions()
    directory = a.output / 'lanes' / f'lane-{a.lane}'
    directory.mkdir(parents=True, exist_ok=False)
    selected = [item for item in selection(bank) if item['target_index'] % 2 == a.lane]
    write(directory / 'identity.json', {'commit': a.expected_commit, 'snapshots': snapshots,
        'selected': selected, 'plan_sha256': sha(a.output / 'preregistered-plan.json'),
        'visible_device': os.environ['CUDA_VISIBLE_DEVICES'], 'worker_pid': os.getpid(),
        'vae_dtype': 'float32', 'reader_dtype': 'bfloat16', 'deadline_unix': a.deadline_unix})
    results = {}
    for item in selected:
        if time.time() >= a.deadline_unix:
            raise TimeoutError('Refinement deadline reached before another target')
        group = next(g for g in bank['groups'] if g['question_id'] == item['question_id'])
        initial = tensors[item['teacher_id']].clone()
        if canonical_tensor_sha256(initial) != item['latent_sha256']:
            raise ValueError('Hash-selected initialization differs')
        donor = group['donor_control']
        if sha(donor['latent_path']) != donor['latent_file_sha256']:
            raise ValueError('Fixed donor changed')
        donor_latent = torch.load(donor['latent_path'], map_location='cpu', weights_only=True)
        reference = torch.load(group['source_latent_path'], map_location='cpu', weights_only=True)
        if canonical_tensor_sha256(donor_latent) != donor['latent_sha256'] or canonical_tensor_sha256(reference) != group['source_latent_sha256']:
            raise ValueError('Donor or gray reference tensor differs')
        with torch.no_grad():
            donor_pixels = decode_model_latents_unit_interval(vae, donor_latent.to(device), clamp=True).cpu()
        blank = torch.full((1, 3, 1024, 1024), 128 / 255., dtype=torch.float32)
        rt = {'vae': vae, 'reader': reader, 'processor': processor, 'vae_device': device, 'reader_device': device,
            'termination': termination, 'training_prompts': TRAIN_PROMPTS, 'deadline_unix': a.deadline_unix,
            'target': {'inputs': group['question_variants'], 'scorer_metadata': {'gold': group['answer']}},
            'controls': {'blank': blank, 'fixed_donor': donor_pixels}}
        spec = {**item, 'run_id': f"target-{item['target_index']:02d}", 'gold': group['answer'],
            'image_form': 'fp32_vae_decoded', 'parent_latent_updates': 256, 'additional_updates': 256,
            'fresh_start_meaning': 'fresh optimizer only; initialization is the recorded trained parent latent',
            'query_exposure': 'allfive already observed in diagnostics; heldout_prompts means not optimized here only'}
        terminal = oracle.run_one(spec=spec, initial=initial, reference=reference, runtime=rt, output_dir=directory)
        run = directory / 'runs' / spec['run_id']
        endpoint = torch.load(run / 'endpoint_raw.pt', map_location='cpu', weights_only=True)
        if endpoint['optimizer_step'] != 256 or canonical_tensor_sha256(endpoint['latent_fp32']) != terminal['endpoint_latent_sha256']:
            raise ValueError('Refinement endpoint binding differs')
        pixels = endpoint['image']
        def quantize(image):
            return (image.clamp(0, 1) * 255).round().byte().float() / 255.
        rgb = quantize(pixels)
        rgb_dir = run / 'rgb'
        rgb_dir.mkdir()
        Image.fromarray((rgb[0] * 255).round().byte().permute(1, 2, 0).numpy()).save(rgb_dir / 'endpoint.png')
        torch.save({'fp32_vae_decoded': pixels, 'rgb_uint8': rgb, 'blank': blank,
                    'fixed_donor_fp32': donor_pixels, 'fixed_donor_rgb': quantize(donor_pixels)}, run / 'qualification-images.pt')
        rgb_rt = {**rt, 'controls': {name: quantize(image) for name, image in rt['controls'].items()}}
        oracle.evaluate(rgb_rt, rgb, spec={**spec, 'image_form': 'rgb_uint8'}, step=256, directory=rgb_dir, all_prompts=True)
        rows = jsonl(run / 'generations.jsonl') + jsonl(rgb_dir / 'generations.jsonl')
        summary = summarize(rows, group)
        if versions() != frozen:
            raise ValueError('Frozen model changed during latent optimization')
        # Seal every saved step, checkpoint, image and raw read; retain failures.
        artifacts = {path.relative_to(run).as_posix(): sha(path) for path in sorted(run.rglob('*')) if path.is_file()}
        write(run / 'qualification.json', {'selected': item, 'summary': summary, 'artifact_hashes': artifacts})
        results[str(item['target_index'])] = {'run': str(run), **summary,
            'qualification_sha256': sha(run / 'qualification.json'), 'latent_sha256': terminal['endpoint_latent_sha256']}
        write(directory / 'progress.json', results)
        print(json.dumps({'target': item['target_index'], **summary}), flush=True)
    clean_source(a.expected_commit)
    if snapshots != snapshot_bindings() or sha(a.bank) != BANK_SHA or read(a.output / 'preregistered-plan.json') != plan(bank):
        raise ValueError('Models, parent or plan changed during refinement')
    write(directory / 'complete.json', {'state': 'completed', 'results': results,
        'all_passed': all(value['both_forms_all_five'] for value in results.values()), 'writer_success': False})


def assemble(a, bank):
    import torch
    from scripts.reporting.collect_transition_endpoint import sha, read, jsonl
    from vision_memory.training.latent_bank_unet import load_teacher_bank
    from vision_memory.repro import canonical_tensor_sha256
    results, teachers, groups = {}, [], []
    by_qid = {group['question_id']: group for group in bank['groups']}
    for item in selection(bank):
        run = a.output / 'lanes' / f"lane-{item['target_index'] % 2}" / 'runs' / f"target-{item['target_index']:02d}"
        qualified = read(run / 'qualification.json')
        if qualified['selected'] != item:
            raise ValueError('A target was replaced during refinement')
        for name, digest in qualified['artifact_hashes'].items():
            path = (run / name).resolve()
            if not path.is_relative_to(run.resolve()) or sha(path) != digest:
                raise ValueError('Refinement artifact changed')
        group = copy.deepcopy(by_qid[item['question_id']])
        summary = summarize(jsonl(run / 'generations.jsonl') + jsonl(run / 'rgb/generations.jsonl'), group)
        if summary != qualified['summary']:
            raise ValueError('Qualification summary differs from the raw records')
        results[str(item['target_index'])] = summary
        payload = torch.load(run / 'endpoint_raw.pt', map_location='cpu', weights_only=True)
        latent = payload['latent_fp32']
        if latent.dtype != torch.float32 or tuple(latent.shape) != (1, 4, 128, 128) or not torch.isfinite(latent).all():
            raise ValueError('Invalid refined model-space target')
        path = a.output / 'bank' / 'latents' / f"target-{item['target_index']:02d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latent, path)
        digest = canonical_tensor_sha256(latent)
        teacher_id = item['question_id'] + '-refined-' + digest[:16]
        teachers.append({'teacher_id': teacher_id, 'question_id': group['question_id'], 'answer': group['answer'],
            'endpoint_step': 256, 'parent_latent_updates': 256, 'cumulative_latent_updates': 512,
            'additional_updates': 256, 'gold_eos_appended': True,
            'target': {'answer': group['answer'], 'strict_correct': summary['both_forms_all_five']},
            'evaluation_generation': {'do_sample': False, 'max_new_tokens': 32}, 'latent_path': str(path),
            'latent_file_sha256': sha(path), 'latent_sha256': digest, 'source_run': str(run),
            'generation_file_sha256': sha(run / 'generations.jsonl'), 'qualification_sha256': sha(run / 'qualification.json'),
            'parent': item, 'qualification': 'both FP32 and RGB, allfive fixed queries with strict answer and immediateEOS'})
        group.update(teacher_ids=[teacher_id], planned_count=1, successful_run_count=int(summary['both_forms_all_five']))
        groups.append(group)
    passed = all(summary['both_forms_all_five'] for summary in results.values())
    manifest = a.output / 'bank/manifest.json'
    if passed:
        output_bank = {**copy.deepcopy(bank), 'teachers': teachers, 'groups': groups,
            'provenance': {'parent_bank': str(a.bank), 'parent_bank_sha256': BANK_SHA, 'all64_original_teachers_retained_in_parent': True,
                'refinement_commit': a.expected_commit, 'refinement_plan_sha256': sha(a.output / 'preregistered-plan.json'),
                'selection': plan(bank)['selection_rule'], 'scope': plan(bank)['scope']}}
        write(manifest, output_bank)
        load_teacher_bank(manifest)
    write(a.output / 'complete.json', {'state': 'completed', 'results': results, 'bank_sealed': passed,
        'bank_manifest_sha256': sha(manifest) if passed else None, 'questions': 16, 'matched_raw_rows': 160,
        'correct_eos': sum(summary['correct_eos'] for summary in results.values()),
        'additional_latent_updates': 4096, 'writer_updates': 0, 'writer_success': False,
        'plan_sha256': sha(a.output / 'preregistered-plan.json')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    p.add_argument('--devices', nargs=2, type=int, default=[2, 3])
    p.add_argument('--lane', type=int, choices=[0, 1])
    a = p.parse_args()
    from scripts.reporting.collect_transition_endpoint import sha, read
    if not math.isfinite(a.deadline_unix) or a.deadline_unix <= time.time() or sha(a.bank) != BANK_SHA:
        raise ValueError('Require the fixed historical bank and a future finite deadline')
    clean_source(a.expected_commit)
    bank = read(a.bank)
    if a.lane is not None:
        def terminate(signum, frame):
            raise TimeoutError('Owned oracle worker received termination; saved evidence retained')
        signal.signal(signal.SIGTERM, terminate)
        worker(a, bank)
        return 0
    if len(set(a.devices)) != 2 or a.deadline_unix - time.time() < 3600:
        raise ValueError('Require two distinct idle H200s and at least one hour')
    gpu = subprocess.check_output(['nvidia-smi', '-i', ','.join(map(str, a.devices)),
        '--query-gpu=index,name,memory.total,memory.used', '--format=csv,noheader,nounits'], text=True)
    rows = [row.split(',') for row in gpu.strip().splitlines()]
    if len(rows) != 2 or any('H200' not in row[1] or int(row[2]) < 140000 or int(row[3]) > 100 for row in rows):
        raise ValueError('Assigned GPUs must actually be idle; never evict existing validation')
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output / 'preregistered-plan.json', plan(bank))
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    children, logs = [], []
    try:
        for lane, device in enumerate(a.devices):
            log = (a.output / f'lane-{lane}.log').open('w')
            logs.append(log)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--bank', str(a.bank), '--output', str(a.output),
                '--expected-commit', a.expected_commit, '--deadline-unix', str(a.deadline_unix), '--lane', str(lane)]
            children.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, **REQUIRED_DETERMINISM_ENV, 'CUDA_VISIBLE_DEVICES': str(device),
                     'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}))
        write(a.output / 'status.json', {'state': 'running', 'worker_pids': [child.pid for child in children],
            'devices': a.devices, 'deadline_unix': a.deadline_unix, 'commit': a.expected_commit})
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError('A target-refinement lane failed; retain both lanes for diagnosis')
            if time.time() >= a.deadline_unix:
                raise TimeoutError('Refinement deadline reached')
            time.sleep(10)
        if any(child.returncode for child in children):
            raise RuntimeError('Refinement failed')
        clean_source(a.expected_commit)
        assemble(a, bank)
        write(a.output / 'status.json', {'state': 'completed', 'complete_sha256': sha(a.output / 'complete.json'),
            'teacher_qualification_requires_raw_review': True, 'writer_success': False})
    except BaseException as error:
        write(a.output / 'status.json', {'state': 'failed', 'error': str(error), 'writer_success': False})
        raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        for log in logs:
            log.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
