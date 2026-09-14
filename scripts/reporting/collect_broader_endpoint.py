"""Verify all151 conditions,604 evaluation tensors and19328 actual training draws."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.reporting.collect_transition_endpoint import sha, read, jsonl
from scripts.experiments.broader_writer_protocol import training_plan, SEED, PARENT_PACKAGE, PARENT_CHECKPOINT
from vision_memory.training.latent_bank_unet import stable_seed, balanced_draw

COMMIT = '84cdfdb58ace96954243de5caf427948717c9abf'
BANK_SHA = 'c27cd65dab809deabb5f2cb08891517d3590244651d08a8c6763c84fea901592'
PLAN_SHA = 'c4a6986edf1330e27af5b91e5105ad6e3806040f01d562f791edbd392bd16834'
NATIVE_CONDITION_COMMIT = '03f8467e5a1201c2dbd9d12484bf2338d7837727'
HISTORICAL_WORDING_COMMIT = 'b9f90e956eea7bda15f638c8877919941ce4fec5'
CLEAR_RETENTION_COMMIT = '4fbc85725d78427235757ace2661d086b896a97f'
# Independently observed in the complete720-row historical readback and the
# sealed transition positive controls, not inferred from each scored output.
GOLD_IDS = {'ambient': [59614], 'jazz': [73, 9802], 'no active preference': [2152, 4541, 21933],
            'green': [13250], 'juice': [8613, 558], 'linen': [3732, 268], 'pasta': [79, 14300]}


def strict_pass(row, gold):
    if row['scorer']['gold_token_ids'] != GOLD_IDS[gold]:
        raise ValueError('Gold tokenization differs from the sealed actual Reader controls')
    passed = row['generated_token_ids'] == GOLD_IDS[gold] + [151645]
    if passed != bool(row['scorer']['strict_correct'] and row['scorer']['answer_followed_immediately_by_eos']):
        raise ValueError('Raw output differs from the strict score')
    return passed


def phase_summary(rows, bank, phase):
    groups = {group['question_id']: group for group in bank['groups']}
    expected = {(qid, condition, noise, prompt) for qid, group in groups.items() for prompt in group['question_variants']
        for condition, noise in [('blank', None), ('donor', None),
            *(('matched', stable_seed(SEED, 'heldout-evaluation-noise', i)) for i in range(2))]}
    seen, cells, images, pairs, partitions = set(), {}, {}, {}, {}
    for row in rows:
        key = row['question_id'], row['condition'], row['noise_seed'], row['prompt_id']
        if key not in expected or key in seen or row['phase'] != phase:
            raise ValueError('Unexpected, repeated or wrong-phase evaluation cell')
        group = groups[key[0]]
        if row['query'] != group['question_variants'][key[3]] or row['gold'] != group['answer']:
            raise ValueError('Original question or answer changed')
        seen.add(key)
        passed = strict_pass(row, group['answer'])
        image = images.setdefault(key[:-1], {'hash': row['image_sha256'], 'passes': []})
        if image['hash'] != row['image_sha256']:
            raise ValueError('Query variants were evaluated on different images')
        image['passes'].append(passed)
        if key[1] == 'matched':
            cell = cells.setdefault(key[0], {'n': 0, 'correct_eos': 0, 'raw': Counter()})
            cell['n'] += 1
            cell['correct_eos'] += int(passed)
            cell['raw'][row['raw']] += 1
            label = 'historical_prefixes' if 'historical_target_index' in group else 'state_transitions'
            part = partitions.setdefault(label, {'n': 0, 'correct_eos': 0})
            part['n'] += 1
            part['correct_eos'] += int(passed)
        pairs[key] = {'passed': passed, 'image_sha256': row['image_sha256'], 'tokens': row['generated_token_ids']}
    if (len(groups) != 151 or len(expected) != 3020 or seen != expected or len(cells) != 151
            or any(cell['n'] != 10 for cell in cells.values())):
        raise ValueError('Incomplete151-condition/two-noise/five-query evaluation')
    matched = [value for key, value in images.items() if key[1] == 'matched']
    return {'phase': phase, 'raw_rows': len(rows), 'matched_rows': 1510, 'generated_images': len(matched),
        'correct_eos': sum(cell['correct_eos'] for cell in cells.values()),
        'images_passing_all_five': sum(len(image['passes']) == 5 and all(image['passes']) for image in matched),
        'partitions': partitions, 'cells': cells}, pairs


def registered_protocol(bank, logical_sampling_commit=None):
    if logical_sampling_commit is None:
        return COMMIT, training_plan(BANK_SHA, COMMIT), PLAN_SHA
    # The explicit training commit selects the preregistered condition control.
    # Never infer a different protocol from mutable run metadata or its score.
    if logical_sampling_commit == CLEAR_RETENTION_COMMIT:
        from scripts.experiments.clear_retention_protocol import plan
    elif logical_sampling_commit == HISTORICAL_WORDING_COMMIT:
        from scripts.experiments.historical_wording_protocol import plan
    elif logical_sampling_commit == NATIVE_CONDITION_COMMIT:
        from scripts.experiments.native_condition_protocol import plan
    else:
        from scripts.experiments.logical_sampling_protocol import plan
    registered = plan(bank, logical_sampling_commit)
    digest = hashlib.sha256((json.dumps(registered, indent=2, sort_keys=True) + '\n').encode()).hexdigest()
    return logical_sampling_commit, registered, digest


def parent_binding(run, bank_path, *, logical_sampling_commit=None):
    bank = read(bank_path)
    commit, registered, plan_sha = registered_protocol(bank, logical_sampling_commit)
    if sha(bank_path) != BANK_SHA or sha(run / 'preregistered-experiment.json') != plan_sha:
        raise ValueError('Fixed broader bank or plan changed')
    if read(run / 'preregistered-experiment.json') != registered:
        raise ValueError('Fixed4832-update protocol differs')
    identity = read(run / 'train/identity.json')
    continuation = commit == CLEAR_RETENTION_COMMIT
    native_condition = commit in (NATIVE_CONDITION_COMMIT, HISTORICAL_WORDING_COMMIT, CLEAR_RETENTION_COMMIT)
    for key, expected in {'git_commit': commit, 'steps': 4832, 'seed': SEED, 'eval_seeds': 2,
            'bank_manifest_sha256': BANK_SHA, 'model_variant': 'base', 'flow_protocol': 'official',
            'trainable_scope': 'full_unet', 'gradient_accumulation_steps': 4,
            'lr': registered['optimizer']['lr'], 'weight_decay': registered['optimizer']['weight_decay']}.items():
        if identity.get(key) != expected:
            raise ValueError('Unexpected broader training identity: ' + key)
    if identity.get('prompt_style') != ('native_base' if native_condition else 'official_raw'):
        raise ValueError('Training condition protocol differs from the explicit registered commit')
    if logical_sampling_commit:
        sampling = registered['sampling']
        expected = {'strategy': 'logical_condition', 'strata': sampling['strata'],
            'within_stratum': 'balanced seeded expression cycles', 'writer_input': False}
        if identity.get('sampling') != expected:
            raise ValueError('Logical sampling identity changed')
        check = read(run / 'train/baseline-reference-check.json')
        binding = identity['initial_baseline_match']
        if continuation and (binding.get('reference_phase') != 'trained' or check.get('reference_phase') != 'trained'
                or check.get('raw_comparison_exclusion') != 'phase label only: baseline versus trained'):
            raise ValueError('Continuation must compare the measured baseline to the trained parent')
        if not continuation and ('reference_phase' in binding or 'reference_phase' in check):
            raise ValueError('Unregistered trained-parent comparison')
        if commit == NATIVE_CONDITION_COMMIT and (binding.get('training_condition_control') != 'official_raw -> native_base'
                or check.get('explicit_training_condition_change') !=
                    'official_raw -> native_base; only encoder hashes and train_prompt metadata may differ'):
            raise ValueError('Missing explicit native-condition baseline control')
        if (check['reference_result_sha256'] != registered['reference_result_sha256']
                or binding['result_sha256'] != registered['reference_result_sha256']
                or check['reference'] != binding['reference'] or binding['baseline_is_measured_again'] is not True
                or any(check.get(key) is not True for key in ('bitwise_latents_and_images', 'bitwise_trajectories', 'identical_raw_generation_records'))
                or len(check['samples']) != 302 or len(set(check['samples'])) != 302):
            raise ValueError('Missing full paired baseline gate')
    elif 'sampling' in identity or 'initial_baseline_match' in identity:
        raise ValueError('Original condition-uniform experiment changed')
    parallel = identity['data_parallel']
    if (parallel['world_size'], parallel['global_microbatches_per_update'], parallel['local_microbatches_per_update']) != (4, 4, 1):
        raise ValueError('Parallel/global batch protocol differs')
    initial = identity['initial_writer']
    if (initial['manifest_sha256'] != registered['initial_package_manifest_sha256']
            or initial['parent_checkpoint_sha256'] != registered['parent_checkpoint_sha256']
            or initial['parent_optimizer_steps'] != (4832 if continuation else 2880)):
        raise ValueError('Parameter-only initialization differs')
    if continuation and (initial['parent_commit'] != registered['reference_commit']
            or initial['parent_result_sha256'] != registered['reference_result_sha256']):
        raise ValueError('Continuation parameter export has different trained-parent lineage')
    runtime = read(run / 'train/runtime.json')['additional_protocol_binding']
    if commit in (HISTORICAL_WORDING_COMMIT, CLEAR_RETENTION_COMMIT):
        from scripts.reporting.verify_historical_condition_draws import verify_seal
        verify_seal(bank, identity, read(run / 'train/runtime.json'),
            read(run / 'train/training-condition-augmentation.json'), registered)
        if 'training_condition_control' in identity['initial_baseline_match']:
            raise ValueError('Wording comparison must preserve native training condition and exact baseline runtime')
    elif 'training_augmentation' in identity:
        raise ValueError('Unregistered training condition augmentation')
    expected_prompt = ('native Base edit, conditional row of upstream three-branch encoding'
        if native_condition else 'raw event, upstream LoRA example')
    if runtime.get('train_prompt') != expected_prompt:
        raise ValueError('Actual training prompt encoding differs from the registered protocol')
    if runtime['inference_guidance_scale'] != 1. or runtime['inference_steps'] != 28:
        raise ValueError('Native inference protocol changed')
    if set(runtime['source_bindings']) != {group['question_id'] for group in bank['groups']}:
        raise ValueError('Incomplete official source encodings')
    for group in bank['groups']:
        binding = runtime['source_bindings'][group['question_id']]
        if binding['bank_source_sha256'] != group['source_latent_sha256']:
            raise ValueError('Source latent identity changed')
        if group['source_kind'] == 'sealed_rgb_1024' and (binding['rms_difference'] != 0
                or binding['official_source_sha256'] != group['source_latent_sha256']
                or binding['source_image_file_sha256'] != group['source_image_file_sha256']):
            raise ValueError('RGB source differs from official encoding')
    result = read(run / 'train/result.json')
    terminal = read(run / 'terminal.json')
    if (terminal['state'] != 'completed' or terminal['training_result_sha256'] != sha(run / 'train/result.json')
            or result['status'] != 'completed' or result['optimizer_steps'] != 4832):
        raise ValueError('Require the complete fixed endpoint')
    return bank, identity, result


def collect(run, bank_path, *, text_only=False, logical_sampling_commit=None):
    run, bank_path = Path(run), Path(bank_path)
    bank, identity, result = parent_binding(run, bank_path, logical_sampling_commit=logical_sampling_commit)
    omitted, proof = [], {}
    def verify(path, digest):
        if text_only and path.suffix == '.pt' and not path.exists():
            omitted.append(path.relative_to(run).as_posix())
        elif sha(path) != digest:
            raise ValueError('Changed artifact: ' + str(path))
    verify(run / 'train/checkpoint-final.pt', result['checkpoint_sha256'])
    for name in ('parallel-initial-parameters.json', 'parallel-gradient-preflight.json',
                 'parallel-parameters-step-000001.json', 'parallel-parameters-step-004832.json'):
        path = run / 'train' / name
        value = read(path)
        if name == 'parallel-gradient-preflight.json':
            if (value['passed'] is not True or not value['identical_draws_and_losses']
                    or not all(math.isfinite(value[key]) for key in ('gradient_relative_l2_error', 'gradient_relative_max_error'))
                    or value['gradient_relative_l2_error'] > 2e-6 or value['gradient_relative_max_error'] > 2e-6):
                raise ValueError('Actual full-U-Net gradient parity failed')
        elif (not value['bitwise_rank_agreement'] or len(value['parameter_sha256_by_rank']) != 4
                or len(set(value['parameter_sha256_by_rank'])) != 1):
            raise ValueError('Four ranks have different parameters')
        expected_initial = ('4a41876c30d6e8d8b5de5ac71af91fee97ae23f77a1299863dde1d32572fce90'
            if logical_sampling_commit == CLEAR_RETENTION_COMMIT else
            '0025dd0c573218179857beaf7e48a4dc7f9d86af5c07056962bea34fb3f6294d')
        if name == 'parallel-initial-parameters.json' and set(value['parameter_sha256_by_rank']) != {expected_initial}:
            raise ValueError('Initial parameters differ from the verified warm endpoint')
        proof[name] = sha(path)
    phases, pairs = {}, {}
    for phase in ('baseline', 'trained'):
        directory = run / 'train' / phase
        complete = read(directory / 'complete.json')
        expected = {'generations.jsonl', 'summary.json'} | {
            hashlib.sha256(group['question_id'].encode()).hexdigest()[:16] + f'-seed-{i:02d}.pt'
            for group in bank['groups'] for i in range(2)}
        if complete['generation_rows'] != 3020 or set(complete['artifact_hashes']) != expected:
            raise ValueError('Missing complete evaluation artifacts')
        for name, digest in complete['artifact_hashes'].items():
            verify(directory / name, digest)
        phases[phase], pairs[phase] = phase_summary(jsonl(directory / 'generations.jsonl'), bank, phase)
        phases[phase]['complete_sha256'] = sha(directory / 'complete.json')
    changes = Counter()
    for key, before in pairs['baseline'].items():
        after = pairs['trained'][key]
        if key[1] == 'matched':
            changes[f"{int(before['passed'])}->{int(after['passed'])}"] += 1
        elif before != after:
            raise ValueError('A fixed negative control changed')
    rows, draws = jsonl(run / 'train/training.jsonl'), []
    wording = logical_sampling_commit in (HISTORICAL_WORDING_COMMIT, CLEAR_RETENTION_COMMIT)
    augmentation_counts, augmentation_seal = {}, None
    if wording:
        from scripts.reporting.verify_historical_condition_draws import verify_draw
        augmentation_seal = read(run / 'train/training-condition-augmentation.json')
    if len(rows) != 4832:
        raise ValueError('Wrong fixed optimizer update count')
    for index, row in enumerate(rows):
        if row['optimizer_step'] != index + 1 or len(row['microbatches']) != 4:
            raise ValueError('Wrong optimizer sequence or global batch')
        for micro, draw in enumerate(row['microbatches']):
            group, teacher, noise, sigma = balanced_draw(bank['groups'], SEED, index * 4 + micro,
                sampling_strategy='logical_condition' if logical_sampling_commit else 'condition')
            if (draw['question_id'], draw['teacher_id'], draw['noise_seed'], draw['effective_sigma']) != (group['question_id'], teacher, noise, sigma):
                raise ValueError('Draw does not match deterministic replay')
            if not math.isfinite(draw['flow_matching_mse']):
                raise ValueError('Nonfinite training loss')
            if wording:
                selected = verify_draw(group, draw, index * 4 + micro, augmentation_seal)
                if selected is not None:
                    qid, expression = selected
                    augmentation_counts.setdefault(qid, Counter())[expression] += 1
            elif 'training_condition_variant' in draw:
                raise ValueError('Unregistered condition change in a training draw')
            draws.append(draw)
    counts = Counter(draw['question_id'] for draw in draws)
    expected_counts = (read(run / 'preregistered-experiment.json')['sampling']['exact_draws_per_group']
        if logical_sampling_commit else {group['question_id']: 128 for group in bank['groups']})
    if counts != expected_counts:
        raise ValueError('Missing or unbalanced condition exposure')
    if wording and augmentation_counts != read(run / 'preregistered-experiment.json')['training_augmentation']['exact_draws_per_expression']:
        raise ValueError('Historical expression exposure differs from the complete registered draw stream')
    if logical_sampling_commit and not text_only:
        from scripts.train.train_latent_bank_unet import verify_initialized_baseline_reference
        # Recheck actual tensors/raws against the original initialization at
        # collection time; a stored pass flag alone is insufficient.
        gate = identity['initial_baseline_match']
        verify_initialized_baseline_reference(run / 'train', Path(gate['reference']), gate['result_sha256'],
            native_condition_control=logical_sampling_commit == NATIVE_CONDITION_COMMIT,
            reference_phase='trained' if logical_sampling_commit == CLEAR_RETENTION_COMMIT else 'baseline')
    extra = ({'training_augmentation_evidence': {
        'seal_sha256': sha(run / 'train/training-condition-augmentation.json'),
        'exact_draws_per_expression': augmentation_counts,
        'scope': 'All recorded event/embedding/mask bindings checked against the fixed plan and each actual draw. Embedding tensors are not independently re-encoded in this CPU collector.'}} if wording else {})
    return {**extra, 'identity': identity, 'result_sha256': sha(run / 'train/result.json'), 'checkpoint_sha256': result['checkpoint_sha256'],
        'phases': phases, 'matched_pairs': changes, 'optimizer_steps': 4832, 'exact_draws_replayed': len(draws),
        'draws_per_group': counts, 'sigma_min': min(draw['effective_sigma'] for draw in draws),
        'sigma_max': max(draw['effective_sigma'] for draw in draws), 'sigma_above_half': sum(draw['effective_sigma'] > .5 for draw in draws),
        'parallel_evidence_sha256': proof, 'artifacts_omitted_locally': omitted, 'all_remote_artifacts_verified_here': not omitted,
        'development_all_correct_eos': phases['trained']['correct_eos'] == 1510,
        'scope': '17 seen semantic questions; development is not independent prefix/chain validation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'bank', 'output-prefix'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--text-only', action='store_true')
    parser.add_argument('--logical-sampling-commit')
    a = parser.parse_args()
    summary = collect(a.run, a.bank, text_only=a.text_only, logical_sampling_commit=a.logical_sampling_commit)
    path = Path(str(a.output_prefix) + '-summary.json')
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    if not a.text_only:
        with tarfile.open(str(a.output_prefix) + '-evidence.tgz', 'w:gz') as archive:
            for item in sorted(a.run.rglob('*')):
                if item.is_file() and item.suffix in ('.json', '.jsonl', '.log'):
                    archive.add(item, arcname=item.relative_to(a.run).as_posix())
            archive.add(path, arcname='verified-summary.json')
    print(json.dumps({key: value for key, value in summary.items() if key not in ('identity', 'phases', 'draws_per_group', 'artifacts_omitted_locally')}))
