from collections import Counter
import copy
import json
from pathlib import Path
import pytest
import torch

from vision_memory.training.latent_bank_unet import balanced_draw, logical_condition_strata
from vision_memory.training.synchronous_parallel import microbatch_indices
from scripts.train.train_latent_bank_unet import verify_initialized_baseline_reference, write_json, file_sha256


def actual_groups():
    path = Path(__file__).resolve().parents[1] / 'reports/official-alignment-results-20260913/broader151-bank-manifest.json'
    return json.loads(path.read_bytes())['groups']


def test_augmentation_does_not_multiply_logical_task_weight_and_stream_is_replayable():
    groups = actual_groups()
    strata = logical_condition_strata(groups)
    assert len(strata) == 31 and Counter(map(len, strata.values())) == {9: 15, 1: 16}
    lookup = {groups[index]['question_id']: key for key, indices in strata.items() for index in indices}
    count = 31 * 9 * 4
    draws = [balanced_draw(groups, 20260915, index, sampling_strategy='logical_condition') for index in range(count)]
    assert Counter(lookup[group['question_id']] for group, _, _, _ in draws) == {key: 36 for key in strata}
    group_counts = Counter(group['question_id'] for group, _, _, _ in draws)
    assert all(group_counts[group['question_id']] == (36 if 'historical_target_index' in group else 4) for group in groups)
    # Group ordering in a JSON manifest cannot silently change the hierarchy.
    reversed_groups = list(reversed(groups))
    for index in range(100):
        assert balanced_draw(reversed_groups, 20260915, index, sampling_strategy='logical_condition') == draws[index]
    # Single-target sampling leaves noise and sigma generation independent of
    # the chosen group. The changed experiment changes data weighting only.
    assert [draw[2:] for draw in draws] == [balanced_draw(groups, 20260915, index)[2:] for index in range(count)]
    assert min(draw[3] for draw in draws) < .01 and max(draw[3] for draw in draws) > .99
    restored = []
    for step in range(count // 4):
        per_rank = [[(micro, draws[step * 4 + micro]) for micro in microbatch_indices(4, 4, rank)] for rank in range(4)]
        restored.extend(draw for _, draw in sorted(item for lane in per_rank for item in lane))
    assert restored == draws
    assert [balanced_draw(groups, 20260915, index, sampling_strategy='logical_condition') for index in range(83, 200)] == draws[83:200]


def test_logical_condition_cannot_merge_different_source_or_query_semantics():
    groups = actual_groups()
    with pytest.raises(ValueError, match='explicit'):
        logical_condition_strata([{'question_id': 'no-provenance'}])
    original = next(group for group in groups if 'parent_transition_group' in group)
    for name, replacement in [('source_latent_sha256', 'wrong-source'), ('target_state', 'jazz'),
                              ('question_variants', {'original_open': 'another question'})]:
        bad = copy.deepcopy(original)
        bad['question_id'] += '-changed'
        bad[name] = replacement
        with pytest.raises(ValueError, match='mixes different'):
            logical_condition_strata([original, bad])
    with pytest.raises(ValueError, match='Repeated'):
        logical_condition_strata([original, original])


def test_registered_comparison_preserves_budget_and_marks_observed_validation():
    from scripts.experiments.logical_sampling_protocol import plan, REFERENCE_RESULT
    value = plan({'groups': actual_groups()}, 'a' * 40)
    sampling = value['sampling']
    assert value['optimizer_steps'] == 4832 and value['draws'] == 19328
    assert value['reference_result_sha256'] == REFERENCE_RESULT
    assert set(sampling['exact_draws_per_stratum'].values()) == {623, 624}
    assert sum(sampling['exact_draws_per_group'].values()) == 19328
    assert 'draws_per_group' not in value
    assert 'observed' in value['validation_exposure'] and 'insufficient' in value['success_policy']


def matched_baselines(root):
    current, reference = root / 'current/train', root / 'reference'
    identity = {key: 'same-' + key for key in ('bank_manifest_sha256', 'seed', 'eval_seeds', 'model_variant', 'trainable_scope',
        'flow_protocol', 'prompt_style', 'steps', 'lr', 'gradient_accumulation_steps', 'weight_decay')}
    identity['initial_writer'] = {'manifest_sha256': 'same-initial-export'}
    for directory in (current, reference / 'train'):
        write_json(directory / 'identity.json', identity)
        write_json(directory / 'runtime.json', {'models_and_protocol': 'same'})
        phase = directory / 'baseline'
        phase.mkdir()
        x = torch.ones(1, 4, 2, 2)
        torch.save({'image': x, 'latent': x, 'noise_seed': 7, 'trajectory': [x, x]}, phase / 'sample.pt')
        (phase / 'generations.jsonl').write_text(json.dumps({'raw': 'green'}) + '\n')
        write_json(phase / 'complete.json', {'artifact_hashes': {path.name: file_sha256(path) for path in phase.iterdir()}})
    write_json(reference / 'train/result.json', {'status': 'completed'})
    digest = file_sha256(reference / 'train/result.json')
    write_json(reference / 'terminal.json', {'state': 'completed', 'training_result_sha256': digest})
    return current, reference, identity, digest


def test_sampler_comparison_requires_identical_initialization_and_actual_new_baseline(tmp_path):
    current, reference, identity, digest = matched_baselines(tmp_path)
    assert verify_initialized_baseline_reference(current, reference, digest)['bitwise_trajectories']
    for key in ('initial_writer', 'seed', 'eval_seeds', 'bank_manifest_sha256', 'steps'):
        changed = {**identity, key: 'different'}
        write_json(current / 'identity.json', changed)
        with pytest.raises(ValueError, match='differs'):
            verify_initialized_baseline_reference(current, reference, digest)
    write_json(current / 'identity.json', identity)
    phase = current / 'baseline'
    payload = torch.load(phase / 'sample.pt', weights_only=True)
    payload['latent'] = payload['latent'] + .01
    torch.save(payload, phase / 'sample.pt')
    complete = json.loads((phase / 'complete.json').read_bytes())
    complete['artifact_hashes']['sample.pt'] = file_sha256(phase / 'sample.pt')
    write_json(phase / 'complete.json', complete)
    with pytest.raises(RuntimeError, match='not bitwise'):
        verify_initialized_baseline_reference(current, reference, digest)
