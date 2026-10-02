"""Read every preregistered historical EOS endpoint through FP32 VAE and RGB pixels."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
PANEL_SHA = '17f00e924ea8a2e66999fd63fb45db8848be8f1c91083155b4a21f9d011b45b5'
FORMS = ('fp32_vae_decoded', 'rgb_uint8')


def load_historical_latent(member):
    import torch
    from scripts.reporting.collect_transition_endpoint import sha
    from vision_memory.repro import canonical_tensor_sha256
    path = Path(member['checkpoint'])
    if sha(path) != member['checkpoint_sha256']:
        raise ValueError('Historical checkpoint changed')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    latent, binding = payload['latent_fp32'], payload['run_binding']
    if ((binding['target_index'], binding['seed'], binding['arm']) != (member['target_index'], member['seed'], 'B')
            or payload['step'] != 256 or latent.dtype != torch.float32 or tuple(latent.shape) != (1, 4, 128, 128)
            or not torch.isfinite(latent).all() or canonical_tensor_sha256(latent) != member['latent_sha256']):
        raise ValueError('Historical endpoint tensor or identity differs from the fixed panel')
    return latent


def summarize(rows, panel):
    targets = {t['target_index']: t for t in panel['targets']}
    expected = {(m['target_index'], m['seed'], 'matched', form, prompt)
                for m in panel['members'] for form in FORMS for prompt in targets[m['target_index']]['question_variants']}
    expected.update((i, None, 'blank', 'rgb_uint8', p) for i, t in targets.items() for p in t['question_variants'])
    seen, cells, member_passes = set(), {}, {}
    for row in rows:
        key = tuple(row[k] for k in ('target_index', 'seed', 'condition', 'image_form', 'prompt_id'))
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicated readback cell')
        seen.add(key)
        target = targets[row['target_index']]
        if row['query'] != target['question_variants'][row['prompt_id']] or row['gold'] != target['gold']:
            raise ValueError('Readback query or answer changed')
        passed = row['generated_token_ids'] == row['scorer']['gold_token_ids'] + [151645]
        if passed != bool(row['scorer']['strict_correct'] and row['scorer']['answer_followed_immediately_by_eos']):
            raise ValueError('Raw tokens disagree with score')
        cell = cells.setdefault(f"{row['condition']}/{row['image_form']}/{row['prompt_id']}", {'n': 0, 'correct_eos': 0})
        cell['n'] += 1
        cell['correct_eos'] += int(passed)
        if row['condition'] == 'matched':
            member_passes.setdefault((row['target_index'], row['seed']), []).append(passed)
    if len(expected) != 720 or seen != expected or len(member_passes) != 64:
        raise ValueError('Missing fixed sixteen-question readback coverage')
    per_target = {str(i): sum(len(v) == 10 and all(v) for (target, _), v in member_passes.items() if target == i) for i in targets}
    return {'raw_rows': len(rows), 'cells': cells, 'members_passing_both_forms_all_five': sum(per_target.values()),
            'passing_members_per_target': per_target, 'targets_passing_all_four_seeds': sum(n == 4 for n in per_target.values()),
            'all_matched_correct_eos': all(all(v) for v in member_passes.values()),
            'shared_writer_success': False, 'scope': 'Direct oracle compatibility readback only; no Writer trained or evaluated.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('panel', 'base-model', 'base-seal', 'mobile-model', 'reader-model', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    if not a.worker:
        return subprocess.call([sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
            env={**os.environ, **REQUIRED_DETERMINISM_ENV, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    from scripts.reporting.collect_transition_endpoint import sha, read
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != a.expected_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()):
        raise ValueError('Readback requires its exact clean source commit')
    if sha(a.panel) != PANEL_SHA:
        raise ValueError('Preregistered panel changed')
    panel = read(a.panel)
    if a.deadline_unix - time.time() < 2400:
        raise ValueError('Reserve at least forty minutes for the complete readback')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if any(line.strip() and int(line.strip()) != os.getpid() for line in active.splitlines()):
        raise RuntimeError('Require an idle GPU; do not compete with the fixed Writer training')
    import numpy as np
    from PIL import Image
    import torch
    from diffusers import AutoencoderTiny
    from types import SimpleNamespace
    from vision_memory.repro import configure_strict_cuda_determinism, canonical_tensor_sha256
    from vision_memory.repro.hf_snapshot import verify_download_seal
    from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
    from vision_memory.reader.open_answer import generate_short_answer
    from vision_memory.reader.open_eos import assistant_termination_contract, generation_diagnostics, qwen3vl_answer_eos_ce
    from vision_memory.reader.deterministic_resize import R3_QWEN_READER_RESIZE_CONTRACT
    from scripts.inspire.model_snapshot_manifest import verify_snapshot_manifest
    from scripts.train.r11_new_frozen_dreamlite_oracle import _load_reader
    configure_strict_cuda_determinism(0)
    device = torch.device('cuda:0')
    if torch.cuda.mem_get_info(device)[0] < 20 * 1024**3:
        raise ValueError('Require at least20GiB available GPU memory')

    def bindings():
        base = verify_download_seal(a.base_seal, a.base_model)
        if base['revision'] != 'a9a0f151ffd99d3c37f3fd0472f5e8f1b31215aa':
            raise ValueError('Wrong Base revision')
        result = {'base': base}
        for key, directory, repo, revision, manifest_sha in (
            ('mobile', a.mobile_model, 'carlofkl/DreamLite-mobile', '6695c3f4be230f0493fa5dbf78be3bc4d3bb2ab4',
             '1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159'),
            ('reader', a.reader_model, 'Qwen/Qwen3-VL-4B-Instruct', 'ebb281ec70b05090aa6165b016eac8ec08e71b17',
             '159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c')):
            result[key] = verify_snapshot_manifest(manifest_path=directory / '.snapshot_manifest.json',
                model_dir=directory, expected_repo_id=repo, expected_revision=revision)
            if result[key]['manifest_sha256'] != manifest_sha:
                raise ValueError('Historical snapshot manifest changed: ' + key)
        if sha(a.base_model / 'vae/diffusion_pytorch_model.safetensors') != sha(a.mobile_model / 'vae/diffusion_pytorch_model.safetensors'):
            raise ValueError('Base and historical Mobile VAE weights differ')
        configs = [read(root / 'vae/config.json') for root in (a.base_model, a.mobile_model)]
        if [{k: v for k, v in c.items() if not k.startswith('_')} for c in configs][0] != [{k: v for k, v in c.items() if not k.startswith('_')} for c in configs][1]:
            raise ValueError('Base and historical VAE latent coordinates differ')
        return result

    snapshots = bindings()
    a.output.mkdir(parents=True, exist_ok=False)
    vae = AutoencoderTiny.from_pretrained(a.base_model, subfolder='vae', local_files_only=True,
                                         torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
    processor, reader = _load_reader(SimpleNamespace(reader=a.reader_model), device, torch.bfloat16)
    termination = assistant_termination_contract(reader, processor)
    if termination['assistant_end_token_id'] != 151645:
        raise ValueError('Reader termination contract changed')

    def versions():
        values = {}
        for label, module in (('vae', vae), ('reader', reader)):
            if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise ValueError('Readback models are not frozen')
            values.update({label + '.' + name: int(p._version) for name, p in module.named_parameters()})
        return values

    frozen = versions()
    identity = {'commit': a.expected_commit, 'probe_sha256': sha(Path(__file__)), 'panel_sha256': PANEL_SHA,
                'snapshots': snapshots, 'vae_dtype': 'float32', 'reader_dtype': 'bfloat16',
                'optimizer_updates': 0, 'writer_calls': 0, 'reader_input': 'only image and query; event stream and gold are excluded from generation',
                'generation': {'do_sample': False, 'max_new_tokens': 32, 'stop_on_newline': False},
                'deadline_unix': a.deadline_unix, 'scope': panel['scope']}
    (a.output / 'identity.json').write_text(json.dumps(identity, indent=2, sort_keys=True) + '\n')
    targets = {t['target_index']: t for t in panel['targets']}
    rows = []

    def evaluate(target, pixels, *, seed, condition, form, artifact):
        for prompt, query in target['question_variants'].items():
            if time.time() >= a.deadline_unix:
                raise TimeoutError('Readback deadline reached; partial evidence retained')
            ce = qwen3vl_answer_eos_ce(model=reader, processor=processor, image=pixels[0].to(device),
                device=device, query=query, target=target['gold'], termination=termination,
                lambda_eos=1., require_image_grad=False, deterministic_ce=True,
                reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
            generation = generate_short_answer(model=reader, processor=processor, image=pixels.to(device),
                query=query, device=device, max_new_tokens=32, do_sample=False)
            row = {'target_index': target['target_index'], 'seed': seed, 'condition': condition,
                'image_form': form, 'image_artifact': artifact, 'image_sha256': canonical_tensor_sha256(pixels),
                'prompt_id': prompt, 'query': query, 'gold': target['gold'], **generation,
                'scorer': generation_diagnostics(generation, target['gold'], ce.target_ids[0, :ce.answer_token_count].cpu().tolist())}
            with (a.output / 'generations.jsonl').open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
            rows.append(row)

    with torch.no_grad():
        for member in panel['members']:
            latent = load_historical_latent(member)
            decoded = decode_model_latents_unit_interval(vae, latent.to(device), clamp=True).cpu()
            if decoded.dtype != torch.float32 or tuple(decoded.shape) != (1, 3, 1024, 1024) or not torch.isfinite(decoded).all():
                raise ValueError('Invalid FP32 VAE output')
            rgb = (decoded[0].permute(1, 2, 0) * 255).round().byte().numpy()
            pixels = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.
            label = f"target-{member['target_index']:03d}-seed-{member['seed']:02d}"
            Image.fromarray(rgb).save(a.output / (label + '.png'))
            torch.save({'latent': latent, 'fp32_vae_decoded': decoded, 'rgb_uint8': pixels}, a.output / (label + '.pt'))
            for form, image in zip(FORMS, (decoded, pixels), strict=True):
                evaluate(targets[member['target_index']], image, seed=member['seed'], condition='matched', form=form, artifact=label + '.pt')
            print(label, flush=True)
        blank_image = Image.new('RGB', (1024, 1024), (128, 128, 128))
        blank_image.save(a.output / 'blank.png')
        blank = torch.full((1, 3, 1024, 1024), 128 / 255., dtype=torch.float32)
        for target in panel['targets']:
            evaluate(target, blank, seed=None, condition='blank', form='rgb_uint8', artifact='blank.png')
    summary = summarize(rows, panel)
    if versions() != frozen or bindings() != snapshots or sha(a.panel) != PANEL_SHA or sha(Path(__file__)) != identity['probe_sha256']:
        raise ValueError('Models, panel or probe changed during readback')
    for member in panel['members']:
        if sha(Path(member['checkpoint'])) != member['checkpoint_sha256']:
            raise ValueError('Historical checkpoint changed during readback')
    complete = {'identity': identity, 'summary': summary,
                'artifact_hashes': {path.name: sha(path) for path in a.output.iterdir() if path.is_file()}}
    (a.output / 'complete.json').write_text(json.dumps(complete, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
