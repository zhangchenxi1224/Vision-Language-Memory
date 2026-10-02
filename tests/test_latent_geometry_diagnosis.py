import hashlib
import json
import pytest
import torch
from scripts.reporting.diagnose_broader_latent_geometry import diagnose, sha


def test_geometry_is_separate_from_reader_success_and_rejects_changed_tensors(tmp_path):
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    target = tmp_path / 'target.pt'
    torch.save(torch.zeros(1, 4, 2, 2), target)
    bank = tmp_path / 'bank.json'
    write(bank, {'teachers': [{'teacher_id': 't', 'answer': 'green', 'latent_sha256': 'target',
        'latent_file_sha256': sha(target), 'latent_path': str(target)}],
        'groups': [{'question_id': 'q', 'teacher_ids': ['t'], 'answer': 'green'}]})
    write(tmp_path / 'train/result.json', {'checkpoint_sha256': 'checkpoint'})
    result_sha = sha(tmp_path / 'train/result.json')
    write(tmp_path / 'terminal.json', {'state': 'completed', 'training_result_sha256': result_sha})
    for phase in ('baseline', 'trained'):
        directory = tmp_path / 'train' / phase
        directory.mkdir()
        rows = []
        for seed in range(2):
            name = hashlib.sha256(b'q').hexdigest()[:16] + f'-seed-{seed:02d}.pt'
            torch.save({'latent': torch.ones(1, 4, 2, 2) * 2, 'noise_seed': seed}, directory / name)
            rows += [{'question_id': 'q', 'condition': 'matched', 'noise_seed': seed, 'prompt_id': str(prompt),
                'generated_token_ids': [99, 151645], 'scorer': {'gold_token_ids': [13250]}} for prompt in range(5)]
        (directory / 'generations.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
        write(directory / 'complete.json', {'artifact_hashes': {p.name: sha(p) for p in directory.iterdir()}})
    value = diagnose(tmp_path, bank, sha(bank), result_sha)
    assert value['phases']['trained']['counts'] == {'images': 2, 'nearest_is_intended': 2, 'correct_eos_rows': 0}
    assert value['phases']['trained']['cells'][0]['intended_rms_error'] == 2
    torch.save(torch.ones(1, 4, 2, 2), target)
    with pytest.raises(ValueError, match='teacher tensor changed'):
        diagnose(tmp_path, bank, sha(bank), result_sha)
