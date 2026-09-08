"""Exercise the real 256-update loop with a tiny differentiable CPU reader."""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.experiments import direct_geometry_eos_training as training
from scripts.experiments.run_direct_latent_geometry import verify_run
from vision_memory.training.direct_latent_geometry import TRAIN_PROMPTS, HELDOUT_PROMPTS, panel, question_prompts


def test_heldout_gradient_request_is_rejected_before_reader_access():
    for prompt in HELDOUT_PROMPTS:
        with pytest.raises(ValueError, match='Held-out'):
            training.teacher({}, None, prompt, True)


def test_real_loop_rotates_one_gradient_per_update_and_audits_receipts(tmp_path, monkeypatch):
    calls = []
    prompt_text = question_prompts()
    names = {v: k for k, v in prompt_text.items()}

    class TinyOracle(torch.nn.Module):
        def __init__(self, *, initial_latent, **kwargs):
            super().__init__()
            self.latent_fp32 = torch.nn.Parameter(initial_latent.clone())

        def image(self):
            return self.latent_fp32.sigmoid()

    def tiny_reader(**kwargs):
        calls.append((names[kwargs['query']], kwargs['require_image_grad']))
        assert kwargs['lambda_eos'] == 1.0
        image = kwargs['image']
        answer = image.square().mean()
        eos = image.mean()
        return SimpleNamespace(loss=answer+eos, answer_loss=answer, eos_loss=eos,
                               target_ids=torch.tensor([[1, 9]]), answer_token_count=1)

    monkeypatch.setattr(training, 'VAELatentOracle', TinyOracle)
    monkeypatch.setattr(training, 'configure_strict_cuda_determinism', lambda seed: None)
    monkeypatch.setattr(training, 'qwen3vl_answer_eos_ce', tiny_reader)
    monkeypatch.setattr(training.replay, 'frozen_audit', lambda *args: {})
    monkeypatch.setattr(training, 'generate_short_answer', lambda **kwargs: {
        'raw': 'ambient', 'generated_token_ids': [1, 9], 'eos_token_ids': [9]})
    initial = torch.ones(1, 3, 2, 2)
    runtime = {'vae': None, 'reader': None, 'processor': None, 'vae_device': 'cpu',
               'reader_device': 'cpu', 'termination': {},
               'controls': {'blank': initial.clone(), 'fixed_donor': initial.clone()},
               'target': {'inputs': prompt_text, 'scorer_metadata': {'gold': 'ambient'}}}
    spec = panel()[0]
    terminal = training.run_one(spec=spec, initial=initial, reference=initial,
                                runtime=runtime, output_dir=tmp_path)
    assert terminal['optimizer_steps'] == 256
    gradients = [p for p, grad in calls if grad]
    assert gradients[:2] == ['original_open'] * 2  # Unchanged reproducibility gate.
    updates = gradients[2:]
    assert updates == ['original_open', 'paraphrase_1', 'paraphrase_2'] * 85 + ['original_open']
    assert Counter(updates) == {'original_open': 86, 'paraphrase_1': 85, 'paraphrase_2': 85}
    assert not set(gradients) & set(HELDOUT_PROMPTS)
    directory = tmp_path / 'runs' / spec['run_id']
    verified = verify_run(directory, spec)
    assert verified['train_qa_pass'] and verified['heldout_qa_pass'] and verified['robust_qa_pass']
    generations = [json.loads(x) for x in (directory/'generations.jsonl').read_text().splitlines()]
    assert len(generations) == 15
    assert all(r['question_trained'] == (r['prompt_id'] in TRAIN_PROMPTS) for r in generations)
    # A result from the old single-prompt protocol cannot be resumed as this experiment.
    manifest_path = directory / 'manifest.json'
    original_manifest = manifest_path.read_text()
    manifest = json.loads(original_manifest)
    manifest['training_prompts'] = ['original_open']
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='different prompt-training protocol'):
        verify_run(directory, spec)
    manifest_path.write_text(original_manifest)
    metrics_path = directory / 'metrics.jsonl'
    metrics = [json.loads(x) for x in metrics_path.read_text().splitlines()]
    metrics[100]['training_prompt_id'] = 'paraphrase_4'
    metrics_path.write_text(''.join(json.dumps(r)+'\n' for r in metrics))
    with pytest.raises(RuntimeError, match='optimizer prompt receipts'):
        verify_run(directory, spec)


def test_config_preserves_initialization_and_question_text():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'configs/experiments/direct_latent_geometry.json').read_text())
    assert config['runs'] == panel()
    assert config['target']['inputs'] == question_prompts()
    assert config['training'] == {'optimizer': 'Adam', 'steps': 256, 'lr': .05, 'lambda_eos': 1.,
                                  'prompts': list(TRAIN_PROMPTS), 'prompt_schedule': 'round_robin_zero_based'}
    assert config['heldout_prompts'] == list(HELDOUT_PROMPTS)
