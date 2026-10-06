import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.inspire import run_context_exposure as m
from vision_memory.training.checkpoint import save_training_checkpoint, load_training_checkpoint


def test_exact_native_adam_rng_continuation_matches_uninterrupted(tmp_path):
    torch.manual_seed(19)
    model = torch.nn.Linear(2, 1); optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    def updates():
        for _ in range(128):
            optimizer.zero_grad(); loss = model(torch.randn(3, 2)).square().mean()
            loss.backward(); optimizer.step()
    updates()
    source = tmp_path/'source.pt'; manifest = dict(steps=128, targets={'frozen': 'sha'}, seed=20260924)
    save_training_checkpoint(source, trainable_module=model, optimizer=optimizer, epoch=0,
        episode_cursor=512, optimizer_step=128, manifest=manifest)
    original = torch.load(source, weights_only=False); changed = m.extended_payload(original)
    assert original['manifest']['steps'] == 128
    for k in original:
        if k != 'manifest': assert m.equivalent(original[k], changed[k])
    assert changed['manifest'] == dict(manifest, steps=256)
    updates(); expected = copy.deepcopy(model.state_dict()); expected_optimizer = copy.deepcopy(optimizer.state_dict())
    clone = torch.nn.Linear(2, 1); opt = torch.optim.AdamW(clone.parameters(), lr=5e-5)
    dest = tmp_path/'extended.pt'; torch.save(changed, dest)
    load_training_checkpoint(dest, trainable_module=clone, optimizer=opt, expected_manifest=dict(manifest, steps=256))
    model, optimizer = clone, opt; updates()
    assert m.equivalent(model.state_dict(), expected)
    assert m.equivalent(optimizer.state_dict(), expected_optimizer)


@pytest.mark.parametrize('field,value', [('optimizer_step', 127), ('episode_cursor', 511), ('rng_state', {}), ('optimizer', {'state': {0: {'step': 127}}})])
def test_bad_resume_cursor_or_state_rejected(field, value):
    payload = dict(schema_version=1, optimizer_step=128, episode_cursor=512, epoch=0, manifest={'steps': 128},
        rng_state=dict(python=None, numpy=None, torch_cpu=None), optimizer={'state': {0: {'step': 128}}})
    payload[field] = value
    with pytest.raises(ValueError): m.extended_payload(payload)


def test_denominators_prefix_pairing_and_stratified_effect():
    spec = m.read(m.p.PROTOCOL)
    ids = [r['base_pair_id'] for r in m.p.population()[0]]
    old = []
    for key in m.p.expected_keys(ids, spec):
        pid, qid, e, c, n = key
        value = dict(zip(m.KEYS, key))
        value.update(kl={'writer16': 3., 'writer32': 2., 'blank': 4., 'text': 0.}[e]+(c == 'mismatch'),
            teacher_target=pid+qid, target_ids=[1, 2], teacher_logits_sha256=pid+qid)
        old.append(value)
    new = [dict(r, endpoint=m.ENDPOINT, kl=r['kl']-.5) for r in old if r['endpoint'] == 'writer32']
    result = m.summarize(old, new, spec)
    assert (result['combined_rows'], result['new_rows'], result['reused_rows']) == (5376, 1536, 3840)
    for group in result['strata'].values():
        assert group['independent_n'] == 16
        for f in group['families'].values(): assert f['updates128_minus256']['mean'] == .5
    for bad in (new[:-1], new+[new[0]], [dict(new[0], teacher_target='foreign')]+new[1:]):
        with pytest.raises(ValueError): m.summarize(old, bad, spec)


def test_prior_population_and_failed_continuations_count_once(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'RUN', tmp_path)
    for name, cost in [('context-population32-v1', 1900), ('context-population-readout-v1', 1600), ('context-exposure-v1', 300)]:
        folder = tmp_path/name/'attempts'; folder.mkdir(parents=True)
        (folder/'attempt.json').write_text(json.dumps(dict(started=1, finished=cost+1, exit_code=124)))
        receipts = folder.parent/'receipts'; receipts.mkdir(); (receipts/'success.json').write_text(json.dumps(dict(started=1, finished=cost+1)))
    assert m.budget(tmp_path/'context-exposure-v1') == (5100, 3500, 300)
    (tmp_path/'context-exposure-v1/attempts/live.json').write_text(json.dumps(dict(started=1)))
    with pytest.raises(ValueError, match='Unsettled'): m.budget(tmp_path/'context-exposure-v1')


def test_jobs_keep_native_parent_and_only_raise_total_steps(tmp_path):
    args = SimpleNamespace(output=tmp_path, base=Path('/base'), official_source=Path('/official'))
    job = m.jobs(args, 'train')[0]; cmd = job['command']
    assert job['gpu'] == 0 and cmd[cmd.index('--steps')+1] == '256'
    assert cmd[cmd.index('--checkpoint')+1] == str(m.OLD/'train/checkpoint-final.pt')
    assert cmd[cmd.index('--teachers')+1] == str(m.p.SOURCE/'bank')
    assert '--snapshot-steps' not in cmd and '--initial-variants' not in cmd
