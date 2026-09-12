import json
import pytest
from scripts.probes.rgb_package_parity import FIELDS, sha, verify, write_json


def replay_fixture(root):
    prepared, inference = root / 'prepared', root / 'inference'
    prepared.mkdir()
    inference.mkdir()
    commands, rows, reads, images = [], [], [], []
    for step in range(6):
        commands.append({'op': 'write', 'event': f'event {step}', 'seed': step})
        image = inference / f'memory-{step + 1:04d}.png'
        image.write_bytes(b'fixed-image-content-' + bytes([step]))
        images.append(sha(image))
        rows.append({'op': 'write', 'index': len(rows), 'memory_image': image.name, 'image_file_sha256': sha(image)})
        for prompt in range(5):
            target = {key: [1, 2] if 'ids' in key else 'raw-value' for key in FIELDS}
            target['query'] = f'query {prompt}'
            commands.append({'op': 'read', 'query': target['query']})
            reads.append(target)
            rows.append({'op': 'read', 'index': len(rows), **target})
    (inference / 'memory-final.png').write_bytes(image.read_bytes())
    (prepared / 'commands.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in commands))
    write_json(prepared / 'expected.json', {'reads': reads, 'png_sha256': images})
    binding = {'expected_sha256': sha(prepared / 'expected.json'), 'command_file_sha256': sha(prepared / 'commands.jsonl'),
               'package_manifest_sha256': 'bound-package', 'reference_functional_pass': False}
    write_json(prepared / 'prepared.json', binding)
    write_json(inference / 'complete.json', {**binding, 'commands': 36, 'writes': 6, 'reads': 30, 'final_image_sha256': images[-1]})
    (inference / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return prepared, inference, rows


def test_exact_replay_does_not_turn_reference_failure_into_functional_success(tmp_path):
    prepared, inference, rows = replay_fixture(tmp_path)
    result = verify(prepared, inference)
    assert result['parity_pass'] and result['reference_functional_pass'] is False
    # A one-token deviation must fail even when decoded strings are unchanged.
    rows[-1]['generated_token_ids'] = [1, 3]
    (inference / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    result = verify(prepared, inference)
    assert result['mismatches'] == [{'read': 29, 'field': 'generated_token_ids'}]
    assert not result['parity_pass']


def test_missing_read_or_mutated_persistent_image_is_rejected(tmp_path):
    prepared, inference, rows = replay_fixture(tmp_path)
    (inference / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows[:-1]))
    with pytest.raises(ValueError, match='Missing'):
        verify(prepared, inference)
    (inference / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (inference / 'memory-0003.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        verify(prepared, inference)
