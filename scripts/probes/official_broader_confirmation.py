"""Execute the fixed151-condition Writer's independent transition/prefix cases."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.reporting.collect_broader_endpoint import parent_binding, phase_summary, strict_pass, COMMIT, BANK_SHA, PLAN_SHA
from scripts.reporting.collect_transition_endpoint import read, jsonl, sha
from scripts.probes.official_transition_confirmation import resolve_events


def selected_cases(registered, mode, lane):
    if mode == 'historical_prefixes':
        if lane not in (0, 1):
            raise ValueError('Historical prefixes require one of two complete target lanes')
        return [case for case in registered['prefix_validation']['cases'] if case['target_index'] % 2 == lane]
    if lane is not None:
        raise ValueError('Only historical-prefix validation uses a lane')
    return registered['transition_validation'][mode]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('single_writes', 'rgb_chains', 'historical_prefixes'), required=True)
    parser.add_argument('--prefix-lane', type=int, choices=(0, 1))
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--worker', action='store_true')
    a = parser.parse_args()
    if not math.isfinite(a.deadline_unix) or a.deadline_unix <= time.time():
        raise ValueError('A finite future validation deadline is required')
    from scripts.train import train_latent_bank_unet as train
    from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    command = read(a.parent_run / 'commands.json')['commands'][-1]
    indices = [index for index, value in enumerate(command) if Path(value).name == 'train_latent_bank_unet.py']
    if len(indices) != 1:
        raise ValueError('Missing unique training entry point')
    args = train.parser().parse_args(command[indices[0] + 1:])
    bank, parent_identity, parent_result = parent_binding(a.parent_run, args.bank_manifest)
    if (args.expected_commit != COMMIT or args.steps != 4832 or args.seed != 20260915 or args.eval_seeds != 2
            or not args.data_parallel or args.gradient_accumulation_steps != 4):
        raise ValueError('Broader training command differs from the fixed plan')
    registered = read(a.parent_run / 'preregistered-experiment.json')
    selected_cases(registered, a.mode, a.prefix_lane)
    if not a.worker:
        return subprocess.call([sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
            env={**os.environ, **snapshot_environment(bank), **REQUIRED_DETERMINISM_ENV,
                 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Independent validation requires a clean immutable checkout')
    after = a.parent_run / 'train/trained'
    complete = read(after / 'complete.json')
    for name in ('generations.jsonl', 'summary.json'):
        if sha(after / name) != complete['artifact_hashes'][name]:
            raise ValueError('Parent raw evaluation changed')
    development, _ = phase_summary(jsonl(after / 'generations.jsonl'), bank, 'trained')
    passed = development['correct_eos'] == 1510
    if not passed and not a.diagnostic:
        raise ValueError('Development failed; explicit diagnostic label required')
    checkpoint = a.parent_run / 'train/checkpoint-final.pt'
    if sha(checkpoint) != parent_result['checkpoint_sha256']:
        raise ValueError('Fixed parent checkpoint changed')
    import torch
    from PIL import Image
    from vision_memory.repro import configure_strict_cuda_determinism, canonical_tensor_sha256
    from vision_memory.training.checkpoint import load_trainable_weights
    from vision_memory.dreamlite.native_base import NativeBaseEditSampler
    from vision_memory.dreamlite.rgb_memory import OfficialRGBMemory
    from scripts.train.official_base_runtime import audit_inference_only_runtime
    configure_strict_cuda_determinism(args.seed)
    if a.device not in range(torch.cuda.device_count()) or torch.cuda.mem_get_info(a.device)[0] < 70 * 1024**3:
        raise ValueError('Requested validation device must have70GiB free; do not evict work')
    args.dreamlite_device = args.reader_device = f'cuda:{a.device}'
    args.colocate_models = True
    a.output.mkdir(parents=True, exist_ok=False)
    runtime = train.load_runtime(args, bank)
    parent_runtime = read(a.parent_run / 'train/runtime.json')
    if (runtime['snapshots'] != parent_runtime['snapshots']
            or runtime['protocol_binding'] != parent_runtime['additional_protocol_binding']
            or {key: canonical_tensor_sha256(value['condition'].prompt_embeds) for key, value in runtime['contexts'].items()}
                != parent_runtime['condition_sha256']):
        raise ValueError('Actual validation runtime differs from the trained model bindings')
    loaded = load_trainable_weights(checkpoint, trainable_module=runtime['pipe'].unet)
    if loaded['optimizer_step'] != 4832 or loaded['manifest'] != {**parent_identity, **parent_runtime}:
        raise ValueError('Checkpoint cursor or complete manifest differs')
    del loaded
    for module in (runtime['pipe'].unet, runtime['pipe'].vae, runtime['pipe'].text_encoder, runtime['reader']):
        module.eval().requires_grad_(False)
    frozen = train.frozen_versions(runtime['pipe'], runtime['reader'])
    transitions = {**bank, 'groups': [group for group in bank['groups'] if 'source_state' in group]}
    original, resolved = resolve_events(registered['transition_validation'], transitions)
    cases = selected_cases({**registered, 'transition_validation': resolved}, a.mode, a.prefix_lane)
    identity = {'probe_commit': commit, 'probe_file_sha256': sha(Path(__file__)), 'source_hashes': train.source_hashes(),
        'parent_commit': COMMIT, 'parent_result_sha256': sha(a.parent_run / 'train/result.json'),
        'checkpoint_sha256': parent_result['checkpoint_sha256'], 'bank_sha256': BANK_SHA, 'plan_file_sha256': PLAN_SHA,
        'registered_plan': registered, 'selected_cases': cases, 'mode': a.mode, 'prefix_lane': a.prefix_lane,
        'development_correct_eos': development['correct_eos'],
        'interpretation': 'fresh_confirmation' if passed else 'diagnostic_after_development_failure',
        'optimizer_updates': 0, 'guidance_scale': 1., 'native_steps': 28, 'deadline_unix': a.deadline_unix,
        'reader_input': 'actual stored RGB uint8 pixels' if a.mode == 'rgb_chains' else 'decoded FP32 image; PNG is a quantized visualization',
        'scope': registered['scope']}
    train.write_json(a.output / 'identity.json', identity)
    cells, expected = {}, {}
    def deadline():
        if time.time() >= a.deadline_unix:
            raise TimeoutError('Validation deadline reached; partial evidence retained')
    def read_image(meta, pixels, variants):
        for prompt, query in variants.items():
            deadline()
            ce = train.qwen3vl_answer_eos_ce(model=runtime['reader'], processor=runtime['processor'],
                image=pixels[0].to(runtime['reader_device']), device=runtime['reader_device'], query=query, target=meta['gold'],
                termination=runtime['termination'], lambda_eos=1., require_image_grad=False, deterministic_ce=True,
                reader_resize_contract=train.R3_QWEN_READER_RESIZE_CONTRACT)
            generated = train.generate_short_answer(model=runtime['reader'], processor=runtime['processor'],
                image=pixels.to(runtime['reader_device']), query=query, device=runtime['reader_device'], max_new_tokens=32, do_sample=False)
            scorer = train.generation_diagnostics(generated, meta['gold'], ce.target_ids[0, :ce.answer_token_count].cpu().tolist())
            row = {**meta, 'prompt_id': prompt, 'query': query, 'image_sha256': canonical_tensor_sha256(pixels),
                   **generated, 'scorer': scorer}
            correct = strict_pass(row, meta['gold'])
            train.append_jsonl(a.output / 'generations.jsonl', row)
            cell = cells.setdefault(meta['case'] + '/' + prompt, {'n': 0, 'correct_eos': 0, 'condition': meta['condition']})
            cell['n'] += 1
            cell['correct_eos'] += int(correct)
    def single(case, group, label):
        if case['gold'] != group['answer'] or ('question_variants' in case
                and case['question_variants'] != group['question_variants']):
            raise ValueError('Registered prefix query or answer differs from the bank')
        context = runtime['contexts'][group['question_id']]
        sampler = NativeBaseEditSampler(runtime['pipe'], source_image=Image.new('RGB', (1024, 1024), (128, 128, 128)),
                                       event_text=case['event_text'], guidance_scale=1.)
        for index, seed in enumerate(case['noise_seeds']):
            deadline()
            noise = torch.randn(context['source'].shape, generator=torch.Generator().manual_seed(seed)).to(runtime['vae_device'])
            out = sampler(source_latents=context['source'], noise_latents=noise, num_steps=28)
            pixels = train.decode_model_latents_unit_interval(runtime['pipe'].vae, out.latents, clamp=True).cpu()
            name = label + f'-seed-{index:02d}'
            train.atomic_tensor(a.output / (name + '.pt'), {'noise': noise.cpu(), 'latent': out.latents.cpu(),
                'image': pixels, 'trajectory': [state.cpu() for state in out.trajectory]})
            Image.fromarray((pixels[0].permute(1, 2, 0) * 255).round().byte().numpy()).save(a.output / (name + '.png'))
            read_image({'case': label, 'question_id': group['question_id'], 'condition': 'matched', 'gold': group['answer'],
                'event_text': case['event_text'], 'noise_seed': seed, 'image_artifact': name + '.pt'}, pixels, group['question_variants'])
            print(name, flush=True)
        expected.update({label + '/' + prompt: len(case['noise_seeds']) for prompt in group['question_variants']})
    def controls(group, label):
        for condition in ('blank', 'donor'):
            name = label + '_' + condition
            pixels = runtime['contexts'][group['question_id']][condition].cpu()
            train.atomic_tensor(a.output / (name + '.pt'), {'image': pixels})
            read_image({'case': name, 'question_id': group['question_id'], 'condition': condition, 'gold': group['answer'],
                'event_text': None, 'noise_seed': None, 'image_artifact': name + '.pt'}, pixels, group['question_variants'])
            expected.update({name + '/' + prompt: 1 for prompt in group['question_variants']})
    with torch.no_grad():
        if a.mode == 'single_writes':
            for case in cases:
                single(case, original[case['state']], case['state'] + '_' + case['style'])
            for state, group in original.items():
                controls(group, state)
        elif a.mode == 'historical_prefixes':
            groups = {group['question_id']: group for group in bank['groups']}
            for case in cases:
                single(case, groups[case['question_id']], case['case'])
            for qid in sorted({case['question_id'] for case in cases}):
                controls(groups[qid], 'target-' + str(groups[qid]['historical_target_index']))
        else:
            variants = original['ambient']['question_variants']
            for sequence in cases:
                memory = OfficialRGBMemory(runtime['pipe'], guidance_scale=1.)
                previous = None
                for step in sequence['steps']:
                    deadline()
                    out = memory.write(step['event_text'], seed=step['noise_seed'])
                    name = f"sequence-{sequence['sequence']}-rep-{sequence['repetition']}-step-{step['step']}"
                    out.image.save(a.output / (name + '.png'))
                    train.atomic_tensor(a.output / (name + '.pt'), {'source': out.source_latent, 'noise': out.noise,
                        'latent': out.latent, 'image': out.pixels, 'trajectory': out.trajectory})
                    before = memory.image.tobytes()
                    read_image({'case': name, 'sequence': sequence['sequence'], 'repetition': sequence['repetition'], **step,
                        'condition': 'matched', 'source_artifact': previous, 'image_artifact': name + '.png',
                        'image_file_sha256': sha(a.output / (name + '.png'))}, out.pixels, variants)
                    if memory.image.tobytes() != before:
                        raise ValueError('Queries mutated persistent RGB memory')
                    previous = name + '.png'
                    expected.update({name + '/' + prompt: 1 for prompt in variants})
                    print(name, flush=True)
    if {key: value['n'] for key, value in cells.items()} != expected:
        raise ValueError('Independent validation coverage differs from the fixed plan')
    audit_inference_only_runtime(runtime, frozen)
    runtime['verify_additional_bindings']()
    if (sha(checkpoint) != parent_result['checkpoint_sha256'] or sha(args.bank_manifest) != BANK_SHA
            or sha(a.parent_run / 'preregistered-experiment.json') != PLAN_SHA
            or sha(a.parent_run / 'train/result.json') != identity['parent_result_sha256']
            or train.source_hashes() != identity['source_hashes'] or sha(Path(__file__)) != identity['probe_file_sha256']):
        raise ValueError('Model, plan, bank or source changed during validation')
    train.write_json(a.output / 'complete.json', {'identity': identity, 'cells': cells,
        'all_generated_correct_eos': all(value['n'] == value['correct_eos'] for value in cells.values() if value['condition'] == 'matched'),
        'artifact_hashes': {path.name: sha(path) for path in a.output.iterdir() if path.is_file()}})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
