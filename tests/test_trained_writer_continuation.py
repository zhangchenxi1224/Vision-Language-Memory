import json

import pytest
import torch

from test_logical_condition_sampling import matched_baselines
from scripts.train.train_latent_bank_unet import verify_initialized_baseline_reference, write_json, file_sha256


def continuation(root):
    current, reference, identity, _ = matched_baselines(root)
    prior = {**identity, 'git_commit': 'parent-commit'}
    write_json(reference / 'train/identity.json', prior)
    (reference / 'train/baseline').rename(reference / 'train/trained')
    for directory, label in ((current / 'baseline', 'baseline'), (reference / 'train/trained', 'trained')):
        (directory / 'generations.jsonl').write_text(json.dumps({'phase': label, 'raw': 'green', 'output_ids': [13250, 151645]}) + '\n')
        write_json(directory / 'complete.json', {'artifact_hashes': {
            path.name: file_sha256(path) for path in directory.iterdir() if path.name != 'complete.json'}})
    checkpoint = reference / 'train/checkpoint-final.pt'
    checkpoint.write_bytes(b'sealed parent parameters')
    result = {'status': 'completed', 'optimizer_steps': 4832, 'checkpoint_sha256': file_sha256(checkpoint)}
    write_json(reference / 'train/result.json', result)
    digest = file_sha256(reference / 'train/result.json')
    write_json(reference / 'terminal.json', {'state': 'completed', 'training_result_sha256': digest})
    identity['initial_writer'] = {'parent_commit': 'parent-commit', 'parent_result_sha256': digest,
        'parent_checkpoint_sha256': result['checkpoint_sha256'], 'parent_optimizer_steps': 4832}
    identity['lr'] = 1e-5
    write_json(current / 'identity.json', identity)
    return current, reference, identity, digest


def test_restart_matches_trained_tensors_and_raws(tmp_path):
    current, reference, _, digest = continuation(tmp_path)
    report = verify_initialized_baseline_reference(current, reference, digest, reference_phase='trained')
    assert report['reference_phase'] == 'trained'
    assert report['bitwise_trajectories'] and report['identical_raw_generation_records']
    with pytest.raises(ValueError, match='differs'):
        verify_initialized_baseline_reference(current, reference, digest)


@pytest.mark.parametrize('field', ['parent_commit', 'parent_checkpoint_sha256', 'parent_result_sha256', 'parent_optimizer_steps'])
def test_restart_rejects_wrong_parameter_lineage(tmp_path, field):
    current, reference, identity, digest = continuation(tmp_path)
    identity['initial_writer'][field] = 'changed'
    write_json(current / 'identity.json', identity)
    with pytest.raises(ValueError, match='do not bind'):
        verify_initialized_baseline_reference(current, reference, digest, reference_phase='trained')


@pytest.mark.parametrize('mutation', ['checkpoint', 'latent', 'raw', 'runtime', 'seed'])
def test_restart_rejects_actual_payload_or_protocol_changes(tmp_path, mutation):
    current, reference, identity, digest = continuation(tmp_path)
    phase = current / 'baseline'
    if mutation == 'checkpoint':
        (reference / 'train/checkpoint-final.pt').write_bytes(b'changed')
    elif mutation == 'runtime':
        write_json(current / 'runtime.json', {'models_and_protocol': 'different'})
    elif mutation == 'seed':
        write_json(current / 'identity.json', {**identity, 'seed': 99})
    else:
        if mutation == 'latent':
            payload = torch.load(phase / 'sample.pt', weights_only=True)
            payload['latent'] = payload['latent'] + .01
            torch.save(payload, phase / 'sample.pt')
        else:
            (phase / 'generations.jsonl').write_text(json.dumps({'phase': 'baseline', 'raw': 'jazz', 'output_ids': [73, 9802, 151645]}) + '\n')
        write_json(phase / 'complete.json', {'artifact_hashes': {
            path.name: file_sha256(path) for path in phase.iterdir() if path.name != 'complete.json'}})
    with pytest.raises((ValueError, RuntimeError)):
        verify_initialized_baseline_reference(current, reference, digest, reference_phase='trained')
