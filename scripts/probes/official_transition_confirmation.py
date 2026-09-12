"""Execute the preregistered fresh-write or RGB-chain test for the45-condition Writer."""
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
from scripts.probes.transition_validation_plan import plan

PARENT_COMMIT = '9628d7142db5a81a9d11a35b89d0515ef32d2e4f'
BANK_SHA = '962f02846ed1a1933e6c219604bc22ee520e28f2dfe2721e26f111dc36ea122e'
PROMPTS = ('original_open', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3', 'paraphrase_4')


def development_gate(rows, bank, seed=20260913):
    """Coverage/tokens must be valid; return functional failures without hiding them."""
    from vision_memory.training.latent_bank_unet import stable_seed
    groups = {g['question_id']: g for g in bank['groups']}
    expected = {(qid, stable_seed(seed, 'heldout-evaluation-noise', i), prompt)
                for qid in groups for i in range(4) for prompt in PROMPTS}
    seen, failures = set(), []
    for row in rows:
        if row['condition'] != 'matched':
            continue
        key = row['question_id'], row['noise_seed'], row['prompt_id']
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate development cell')
        seen.add(key)
        group = groups[row['question_id']]
        if row['query'] != group['question_variants'][row['prompt_id']] or row['gold'] != group['answer']:
            raise ValueError('Development query or answer differs from the bank')
        correct = row['generated_token_ids'] == row['scorer']['gold_token_ids'] + [group['termination_contract']['assistant_end_token_id']]
        if correct != bool(row['scorer']['strict_correct'] and row['scorer']['answer_followed_immediately_by_eos']):
            raise ValueError('Development raw tokens disagree with score')
        if not correct:
            failures.append(key)
    if seen != expected or len(expected) != 900:
        raise ValueError('Require complete45-group/four-noise/five-prompt development coverage')
    return {'matched_rows': len(seen), 'correct_eos': len(seen) - len(failures),
            'all_correct_eos': not failures, 'failed_cells': failures}


def resolve_events(registered, bank):
    """Original prompts are bound to exact training groups; new prompts stay literal."""
    original, noops = {}, set()
    for state in ('ambient', 'jazz', 'clear'):
        matches = [g for g in bank['groups'] if g['source_state'] == 'gray'
                   and g['target_state'] == state and g['wording_index'] == 0]
        if len(matches) != 1:
            raise ValueError('Missing unique original gray-source event')
        original[state] = matches[0]
    for group in bank['groups']:
        if group['operation'] == 'noop' and group['wording_index'] == 0:
            noops.add(group['event_text'])
    if len(noops) != 1:
        raise ValueError('Original no-op must use identical text for every source state')
    resolved = json.loads(json.dumps(registered))
    for case in resolved['single_writes']:
        case.setdefault('event_text', original[case['state']]['event_text'])
    for sequence in resolved['rgb_chains']:
        for step in sequence['steps']:
            step.setdefault('event_text', next(iter(noops)) if step['operation'] == 'noop'
                            else original[step['expected_state']]['event_text'])
    return original, resolved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('single_writes', 'rgb_chains'), required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    p.add_argument('--diagnostic', action='store_true', help='Explicit diagnostic if the complete development endpoint failed')
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    from scripts.train import train_latent_bank_unet as train
    from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    from vision_memory.training.latent_bank_unet import file_sha256, load_teacher_bank
    command = json.loads((a.parent_run / 'commands.json').read_text())['commands'][-1]
    args = train.parser().parse_args(command[3:])
    if (args.expected_commit != PARENT_COMMIT or file_sha256(args.bank_manifest) != BANK_SHA
            or args.steps != 2880 or args.eval_seeds != 4 or args.seed != 20260913
            or args.trainable_scope != 'full_unet' or args.model_variant != 'base'
            or args.base_guidance_scale != 1.0 or args.flow_protocol != 'official'):
        raise ValueError('Require the fixed45-group transition training design')
    bank, _ = load_teacher_bank(args.bank_manifest)
    registered = plan(args.seed)
    if json.loads((ROOT / 'reports/official-transition-validation-plan-20260913.json').read_text()) != registered:
        raise ValueError('Validation plan changed from its preregistration')
    original, resolved = resolve_events(registered, bank)
    variants = original['ambient']['question_variants']
    if set(variants) != set(PROMPTS) or any(g['question_variants'] != variants for g in bank['groups']):
        raise ValueError('Require the same five queries for the one semantic question')
    if not a.worker:
        env = {**os.environ, **snapshot_environment(bank), **REQUIRED_DETERMINISM_ENV,
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--parent-run', str(a.parent_run),
               '--output', str(a.output), '--mode', a.mode, '--deadline-unix', str(a.deadline_unix), '--worker']
        if a.diagnostic:
            cmd.append('--diagnostic')
        return subprocess.call(cmd, env=env)
    result_path = a.parent_run / 'train/result.json'
    result_sha = file_sha256(result_path)
    terminal = json.loads((a.parent_run / 'terminal.json').read_text())
    if terminal.get('state') != 'completed' or terminal['training_result_sha256'] != result_sha:
        raise ValueError('Require a completed sealed parent endpoint')
    result = json.loads(result_path.read_text())
    phase = a.parent_run / 'train/trained'
    complete = json.loads((phase / 'complete.json').read_text())
    for name, digest in complete['artifact_hashes'].items():
        if Path(name).name != name or file_sha256(phase / name) != digest:
            raise ValueError('Parent evaluation artifact changed')
    gate = development_gate([json.loads(line) for line in (phase / 'generations.jsonl').read_text().splitlines()], bank)
    if not gate['all_correct_eos'] and not a.diagnostic:
        raise ValueError(f"Development failed: {gate['correct_eos']}/900; only an explicitly labelled diagnostic may proceed")
    checkpoint = a.parent_run / 'train/checkpoint-final.pt'
    if file_sha256(checkpoint) != result['checkpoint_sha256']:
        raise ValueError('Parent checkpoint changed')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Validation requires immutable clean source')
    import torch
    from PIL import Image
    from vision_memory.repro import configure_strict_cuda_determinism, canonical_tensor_sha256
    from vision_memory.training.checkpoint import load_trainable_weights
    from vision_memory.dreamlite.native_base import NativeBaseEditSampler
    from vision_memory.dreamlite.rgb_memory import OfficialRGBMemory
    from scripts.train.official_base_runtime import audit_inference_only_runtime
    configure_strict_cuda_determinism(args.seed)
    if torch.cuda.mem_get_info(torch.device(args.dreamlite_device))[0] < 70 * 1024**3:
        raise RuntimeError('Require70GiB available GPU memory; do not evict other work')
    a.output.mkdir(parents=True, exist_ok=False)
    runtime = train.load_runtime(args, bank)
    parent_runtime = json.loads((a.parent_run / 'train/runtime.json').read_text())
    parent_identity = json.loads((a.parent_run / 'train/identity.json').read_text())
    if (runtime['snapshots'] != parent_runtime['snapshots']
            or runtime['protocol_binding'] != parent_runtime['additional_protocol_binding']
            or {k: canonical_tensor_sha256(v['condition'].prompt_embeds) for k, v in runtime['contexts'].items()}
               != parent_runtime['condition_sha256']):
        raise ValueError('Runtime differs from the trained model bindings')
    loaded = load_trainable_weights(checkpoint, trainable_module=runtime['pipe'].unet)
    if loaded['optimizer_step'] != 2880 or loaded['manifest'] != {**parent_identity, **parent_runtime}:
        raise ValueError('Checkpoint manifest or optimizer cursor differs')
    del loaded
    for module in (runtime['pipe'].unet, runtime['pipe'].vae, runtime['pipe'].text_encoder, runtime['reader']):
        module.eval().requires_grad_(False)
    frozen = train.frozen_versions(runtime['pipe'], runtime['reader'])
    identity = {'probe_commit': commit, 'probe_file_sha256': file_sha256(Path(__file__)),
        'plan_file_sha256': file_sha256(ROOT / 'reports/official-transition-validation-plan-20260913.json'),
        'source_hashes': train.source_hashes(), 'parent_commit': PARENT_COMMIT, 'parent_result_sha256': result_sha,
        'checkpoint_sha256': result['checkpoint_sha256'], 'bank_sha256': BANK_SHA,
        'registered_plan': registered, 'resolved_plan': resolved, 'mode': a.mode, 'development_gate': gate,
        'interpretation': 'diagnostic_after_development_failure' if not gate['all_correct_eos'] else 'fresh_confirmation',
        'optimizer_updates': 0, 'guidance_scale': 1., 'image_guidance_scale': 1., 'native_steps': 28,
        'reader_input': 'actual RGB uint8 PNG pixels' if a.mode == 'rgb_chains' else 'decoded FP32 image as in development; PNG is a quantized visualization',
        'deadline_unix': a.deadline_unix, 'scope': registered['scope']}
    train.write_json(a.output / 'identity.json', identity)
    cells = {}

    def deadline():
        if time.time() >= a.deadline_unix:
            raise TimeoutError('Validation lease expired; partial evidence retained')

    def read_image(meta, pixels):
        for prompt_id, query in variants.items():
            deadline()
            ce = train.qwen3vl_answer_eos_ce(model=runtime['reader'], processor=runtime['processor'],
                image=pixels[0].to(runtime['reader_device']), device=runtime['reader_device'], query=query,
                target=meta['gold'], termination=runtime['termination'], lambda_eos=1., require_image_grad=False,
                deterministic_ce=True, reader_resize_contract=train.R3_QWEN_READER_RESIZE_CONTRACT)
            generation = train.generate_short_answer(model=runtime['reader'], processor=runtime['processor'],
                image=pixels.to(runtime['reader_device']), query=query, device=runtime['reader_device'], max_new_tokens=32, do_sample=False)
            score = train.generation_diagnostics(generation, meta['gold'], ce.target_ids[0, :ce.answer_token_count].cpu().tolist())
            train.append_jsonl(a.output / 'generations.jsonl', {**meta, 'prompt_id': prompt_id, 'query': query,
                'image_sha256': canonical_tensor_sha256(pixels), 'answer_ce': float(ce.answer_loss),
                'eos_ce': float(ce.eos_loss), **generation, 'scorer': score})
            cell = cells.setdefault(meta['case'] + '/' + prompt_id, {'n': 0, 'correct_eos': 0})
            cell['n'] += 1
            cell['correct_eos'] += int(score['strict_correct'] and score['answer_followed_immediately_by_eos'])

    expected = {}
    with torch.no_grad():
        if a.mode == 'single_writes':
            for case in resolved['single_writes']:
                group = original[case['state']]
                context = runtime['contexts'][group['question_id']]
                name = case['state'] + '_' + case['style']
                sampler = NativeBaseEditSampler(runtime['pipe'], source_image=Image.new('RGB', (1024, 1024), (128, 128, 128)),
                                               event_text=case['event_text'], guidance_scale=1.)
                for index, seed in enumerate(case['noise_seeds']):
                    deadline()
                    noise = torch.randn(context['source'].shape, generator=torch.Generator().manual_seed(seed)).to(runtime['vae_device'])
                    out = sampler(source_latents=context['source'], noise_latents=noise, num_steps=28)
                    pixels = train.decode_model_latents_unit_interval(runtime['pipe'].vae, out.latents, clamp=True).cpu()
                    label = name + f'-seed-{index:02d}'
                    train.atomic_tensor(a.output / (label + '.pt'), {'noise_seed': seed, 'noise': noise.cpu(),
                        'latent': out.latents.cpu(), 'image': pixels, 'trajectory': [x.cpu() for x in out.trajectory]})
                    Image.fromarray((pixels[0].permute(1, 2, 0) * 255).round().byte().numpy()).save(a.output / (label + '.png'))
                    read_image({'case': name, 'state': case['state'], 'gold': case['gold'], 'event_text': case['event_text'],
                                'noise_seed': seed, 'image_artifact': label + '.pt'}, pixels)
                    print(label, flush=True)
                expected.update({name + '/' + prompt: len(case['noise_seeds']) for prompt in PROMPTS})
            for state, group in original.items():
                for control in ('blank', 'donor'):
                    label = state + '_' + control
                    pixels = runtime['contexts'][group['question_id']][control].cpu()
                    train.atomic_tensor(a.output / (label + '.pt'), {'image': pixels})
                    read_image({'case': label, 'state': state, 'gold': group['answer'], 'event_text': None,
                                'noise_seed': None, 'image_artifact': label + '.pt'}, pixels)
                    expected.update({label + '/' + prompt: 1 for prompt in PROMPTS})
        else:
            for sequence in resolved['rgb_chains']:
                memory = OfficialRGBMemory(runtime['pipe'], guidance_scale=1.)
                previous = None
                for step in sequence['steps']:
                    deadline()
                    out = memory.write(step['event_text'], seed=step['noise_seed'])
                    label = f"sequence-{sequence['sequence']}-rep-{sequence['repetition']}-step-{step['step']}"
                    out.image.save(a.output / (label + '.png'))
                    train.atomic_tensor(a.output / (label + '.pt'), {'source': out.source_latent, 'noise': out.noise,
                        'latent': out.latent, 'image': out.pixels, 'trajectory': out.trajectory})
                    before = memory.image.tobytes()
                    read_image({'case': label, 'sequence': sequence['sequence'], 'repetition': sequence['repetition'], **step,
                        'source_artifact': previous, 'image_artifact': label + '.png',
                        'image_file_sha256': file_sha256(a.output / (label + '.png'))}, out.pixels)
                    if memory.image.tobytes() != before:
                        raise RuntimeError('Reader queries mutated RGB memory')
                    previous = label + '.png'
                    expected.update({label + '/' + prompt: 1 for prompt in PROMPTS})
                    print(label, flush=True)
    if {k: v['n'] for k, v in cells.items()} != expected:
        raise RuntimeError('Missing registered validation coverage')
    audit_inference_only_runtime(runtime, frozen)
    runtime['verify_additional_bindings']()
    if (file_sha256(result_path) != result_sha or file_sha256(checkpoint) != result['checkpoint_sha256']
            or file_sha256(args.bank_manifest) != BANK_SHA or train.source_hashes() != identity['source_hashes']
            or file_sha256(Path(__file__)) != identity['probe_file_sha256']
            or file_sha256(ROOT / 'reports/official-transition-validation-plan-20260913.json') != identity['plan_file_sha256']):
        raise RuntimeError('Parent weights, result, bank, or validation source changed')
    train.write_json(a.output / 'complete.json', {'identity': identity, 'cells': cells,
        'all_generated_correct_eos': all(c['correct_eos'] == c['n'] for k, c in cells.items()
            if not any(k.endswith('_' + control + '/' + prompt) for control in ('blank', 'donor') for prompt in PROMPTS)),
        'artifact_hashes': {path.name: file_sha256(path) for path in a.output.iterdir() if path.is_file()}})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
