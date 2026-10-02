import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.train.generated_source_augmentation import source_variant_index, source_pool_binding
from scripts.train import train_latent_bank_unet as training
from vision_memory.training.latent_bank_unet import balanced_draw


def test_all_actual_music_draws_balance_canonical_and_all_generated_sources():
    path = Path(__file__).resolve().parents[1] / 'reports/official-alignment-results-20260913/broader151-bank-manifest.json'
    groups = json.loads(path.read_bytes())['groups']
    counts = {g['question_id']: [0] * 9 for g in groups if g.get('source_kind') == 'sealed_rgb_1024'}
    assert len(counts) == 108
    for draw_index in range(19328):
        group, _, _, _ = balanced_draw(groups, 20260915, draw_index, sampling_strategy='logical_condition')
        if group['question_id'] in counts:
            counts[group['question_id']][source_variant_index(20260915, draw_index, group['question_id'])] += 1
    assert all(min(count) > 0 and max(count) - min(count) <= 1 for count in counts.values())


def test_actual_flow_bridge_excludes_selected_source_and_uses_its_condition(monkeypatch):
    condition = lambda value: SimpleNamespace(prompt_embeds=torch.tensor([value]), attention_mask=torch.ones(1))
    canonical, changed = condition(2.), condition(9.)
    context = {'source': torch.full((1, 1, 1, 1), 3.), 'condition': canonical,
        'training_source_variants': [(torch.full((1, 1, 1, 1), 100.), changed)],
        'training_source_variant_binding': [{'index': 0, 'verified_pair': True}]}
    group = {'question_id': 'case', 'source_kind': 'sealed_rgb_1024'}
    monkeypatch.setattr(training, 'balanced_draw', lambda *a, **kw: (group, 'teacher', 456, .75))
    monkeypatch.setattr('scripts.train.generated_source_augmentation.source_variant_index', lambda *a: 0)
    parameter = torch.tensor(0., requires_grad=True)
    seen = []
    def predict(sampler, state, source, sigma, prompt, mask, **kwargs):
        seen.append((state.clone(), source.clone(), prompt.clone(), kwargs))
        return torch.zeros_like(state) + parameter
    monkeypatch.setattr(training, 'predict_velocity', predict)
    runtime = {'contexts': {'case': context}, 'vae_device': 'cpu', 'sampler': None}
    args = SimpleNamespace(flow_protocol='official', seed=20260915, gradient_accumulation_steps=1, generated_source_pool=None)
    teachers = {'teacher': torch.full((1, 1, 1, 1), 7.)}
    old = training.flow_microbatch(args, runtime, [group], teachers, 0)
    args.generated_source_pool = Path('qualified/manifest.json')
    new = training.flow_microbatch(args, runtime, [group], teachers, 0)
    assert torch.equal(seen[0][0], seen[1][0])
    noise = torch.randn((1, 1, 1, 1), generator=torch.Generator().manual_seed(456))
    assert torch.equal(seen[1][0], .25 * teachers['teacher'] + .75 * noise)
    assert old['flow_matching_mse'] == new['flow_matching_mse'] == float((noise - teachers['teacher']).square().mean())
    assert seen[1][1].item() == 100. and seen[1][2].item() == 9.
    assert seen[1][3]['integer_timestep'] is True
    assert new['training_source_variant'] == context['training_source_variant_binding'][0]
    assert context['source'].item() == 3. and context['condition'] is canonical


def test_pool_flag_requires_explicit_integrity_binding():
    assert source_pool_binding(SimpleNamespace()) is None
    with pytest.raises(ValueError, match='explicit manifest'):
        source_pool_binding(SimpleNamespace(generated_source_pool=Path('pool/manifest.json')))


def test_installed_source_registry_supports_actual_file_integrity_recheck(tmp_path, monkeypatch):
    from PIL import Image
    from scripts.train.generated_source_augmentation import install_source_variants
    from vision_memory.training.latent_bank_unet import file_sha256
    records = []
    latent = torch.ones(1, 4, 2, 2)
    for state in ('ambient', 'jazz', 'clear'):
        for index in range(8):
            job = f'{state}-{index:02d}'
            png, tensor = tmp_path/(job+'.png'), tmp_path/(job+'.pt')
            Image.new('RGB', (16, 16), (index, 128, 128)).save(png)
            torch.save({'source_latent': latent}, tensor)
            records.append({'state': state, 'job': job, 'png': png.name,
                'tensor': tensor.name, 'png_sha256': file_sha256(png)})
    manifest = {'records': records}
    manifest_path = tmp_path/'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr('scripts.train.generated_source_augmentation.source_pool_binding',
        lambda args: {'manifest': str(manifest_path)})
    monkeypatch.setattr('scripts.train.generated_source_augmentation.plan', lambda: {})
    monkeypatch.setattr('scripts.probes.generate_training_source_pool.collect', lambda *args: manifest)
    condition = SimpleNamespace(prompt_embeds=torch.ones(1, 2, 3), attention_mask=torch.ones(1, 2))
    monkeypatch.setattr('vision_memory.dreamlite.conditioning.encode_native_base_edit_condition', lambda *args, **kwargs: condition)
    pipe = SimpleNamespace(image_processor=SimpleNamespace(preprocess=lambda image: image),
        prepare_image_latents=lambda *args, **kwargs: latent)
    groups = [{'question_id': f'{state}-{i}', 'source_state': state, 'source_kind': 'sealed_rgb_1024',
        'wording_index': i % 9, 'event_text': 'fixture', 'source_image_file_sha256': 'a'*64}
        for state in ('ambient', 'jazz', 'clear') for i in range(36)]
    contexts = {g['question_id']: {'source': latent, 'condition': condition} for g in groups}
    files = {}
    bindings = install_source_variants(SimpleNamespace(dreamlite_device='cpu'), {'groups': groups}, pipe, contexts, files)
    assert len(bindings) == 108 and len(files) == 24
    # This is the actual consumer contract in official_base_runtime.verify_extra.
    assert all(file_sha256(path) == digest for path, digest in files.items())
    changed = next(iter(files))
    changed.write_bytes(b'changed source PNG')
    assert file_sha256(changed) != files[changed]


def test_complete_training_plan_binds_4f_and_keeps_full_budget():
    from scripts.experiments.generated_source_training_protocol import plan, REFERENCE_COMMIT, REFERENCE_RESULT
    path = Path(__file__).resolve().parents[1] / 'reports/official-alignment-results-20260913/broader151-bank-manifest.json'
    value = plan(json.loads(path.read_bytes()), 'a' * 40)  # Fixture source revision; never dispatched.
    assert value['reference_commit'] == value['parent_training_commit'] == REFERENCE_COMMIT
    assert value['reference_result_sha256'] == value['parent_result_sha256'] == REFERENCE_RESULT
    assert value['optimizer_steps'] == 4832 and value['draws'] == 19328
    assert value['optimizer']['lr'] == 1e-5
    assert value['generated_source_pool']['augmented_conditions'] == 108
    assert len(value['training_augmentation']['events']) == 16
    assert len(value['sampling']['strata']) == 31
    assert value['reference_phase'] == 'trained'
