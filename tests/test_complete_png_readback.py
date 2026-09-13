import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.experiments.png_readback_protocol import LANES, SOURCES, PARENT_COMMIT, plan, png_pixels, compare
from scripts.experiments.historical_wording_protocol import plan as training_plan
from scripts.experiments.fresh_wording_validation import plan as fresh_plan
from scripts.reporting.collect_broader_endpoint import GOLD_IDS
from scripts.reporting.collect_broader_validation import expected_rows

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def matrices():
    bank = json.loads((ROOT / 'reports/official-alignment-results-20260913/broader151-bank-manifest.json').read_bytes())
    training = training_plan(bank, PARENT_COMMIT)
    return bank, {'registered': training, 'fresh_wording_v1': fresh_plan(training, bank)}


def rows_for(registered, bank, mode, lane):
    expected, _, _ = expected_rows(registered, bank, mode, lane)
    return [{**meta, 'raw': meta['gold'], 'generated_token_ids': GOLD_IDS[meta['gold']] + [151645],
        'image_sha256': hashlib.sha256(meta['image_artifact'].encode()).hexdigest(),
        'scorer': {'gold_token_ids': GOLD_IDS[meta['gold']], 'strict_correct': True,
                   'answer_followed_immediately_by_eos': True}} for meta in expected.values()]


@pytest.mark.parametrize('validation_set', SOURCES)
@pytest.mark.parametrize('label', LANES)
def test_complete_matrix_paired_regression_and_chain_parity(matrices, validation_set, label):
    bank, plans = matrices
    mode, lane, raw_count, matched, image_count = LANES[label]
    original = rows_for(plans[validation_set], bank, mode, lane)
    rows = [{**copy.deepcopy(row), 'source_image_sha256': row['image_sha256']} for row in original]
    result = compare(rows, original, plans[validation_set], bank, mode, lane)
    assert len(rows) == raw_count
    assert len({row['image_artifact'] for row in rows}) == image_count
    assert result['png']['matched_correct_eos'] == matched
    assert result['chain_parity_passed']
    changed = next(row for row in rows if row['condition'] == 'matched')
    changed.update(raw='incorrect', generated_token_ids=[0, 151645],
        scorer={**changed['scorer'], 'strict_correct': False, 'answer_followed_immediately_by_eos': False})
    result = compare(rows, original, plans[validation_set], bank, mode, lane)
    assert result['paired_matched']['regressed'] == 1
    assert result['png']['matched_correct_eos'] == matched - 1
    assert result['chain_parity_passed'] == (mode != 'rgb_chains')
    with pytest.raises(ValueError, match='Incomplete'):
        compare(rows[:-1], original, plans[validation_set], bank, mode, lane)
    with pytest.raises(ValueError, match='duplicated'):
        compare(rows + [rows[0]], original, plans[validation_set], bank, mode, lane)


def test_actual_saved_png_matches_cli_conversion_and_exposes_quantization(tmp_path):
    pixels = torch.linspace(0, 1, 1024 * 1024).reshape(1, 1, 1024, 1024).repeat(1, 3, 1, 1)
    path = tmp_path / 'memory.png'
    Image.fromarray((pixels[0].permute(1, 2, 0) * 255).round().byte().numpy()).save(path)
    with Image.open(path) as image:
        actual = png_pixels(image)
        cli = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.
    assert torch.equal(actual, cli)
    assert not torch.equal(actual, pixels)
    assert (actual - pixels).abs().max() <= 0.5 / 255 + 1e-7
    with pytest.raises(ValueError):
        png_pixels(Image.new('L', (1024, 1024)))


def test_plan_has_complete_totals_and_entrypoints_work_outside_checkout(tmp_path):
    value = plan()
    assert sum(v[2] for v in LANES.values()) * 2 == value['total_raw_rows'] == 3980
    assert sum(v[3] for v in LANES.values()) * 2 == value['total_matched_rows'] == 3600
    assert sum(v[4] for v in LANES.values()) * 2 == value['total_images'] == 796
    for file in ('scripts/probes/complete_png_readback.py', 'scripts/inspire/run_png_readback_suite.py',
                 'scripts/reporting/collect_png_readback.py'):
        result = subprocess.run([sys.executable, '-I', str(ROOT / file), '--help'], cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_collector_recounts_actual_pngs_and_rejects_altered_pixels(tmp_path, matrices):
    from scripts.reporting.collect_png_readback import collect
    from scripts.reporting.collect_transition_endpoint import sha
    from vision_memory.repro import canonical_tensor_sha256
    bank, plans = matrices
    registered = plans['registered']
    original = rows_for(registered, bank, 'single_writes', None)
    run, source = tmp_path / 'readback', tmp_path / 'source'
    run.mkdir()
    source.mkdir()
    commit = 'a' * 40
    def write(path, value):
        path.write_bytes((json.dumps(value, indent=2) + '\n').encode())
    image = Image.new('RGB', (1024, 1024), (128, 128, 128))
    pixel_sha = canonical_tensor_sha256(png_pixels(image))
    for row in original:
        row['image_sha256'] = pixel_sha
    (run / 'source-generations.jsonl').write_bytes(('\n'.join(json.dumps(row) for row in original) + '\n').encode())
    hashes = {'generations.jsonl': sha(run / 'source-generations.jsonl')}
    checks = {}
    for row in original:
        name = row['image_artifact']
        if name in checks:
            continue
        png_name = str(Path(name).with_suffix('.png'))
        matched = row['condition'] == 'matched'
        png_path = (source if matched else run) / png_name
        image.save(png_path)
        hashes[name] = 'b' * 64
        if matched:
            hashes[png_name] = sha(png_path)
        checks[name] = {'source_image_sha256': pixel_sha, 'source_pt_sha256': hashes[name],
            'png_file_sha256': sha(png_path), 'png_name': png_name,
            'png_location': 'source' if matched else 'readback', 'image_sha256': pixel_sha,
            'pixel_max_abs_change': 0.0}
    source_identity = {'probe_commit': SOURCES['registered'][0], 'checkpoint_sha256': 'c' * 64,
        'registered_plan': registered, 'parent_commit': PARENT_COMMIT, 'mode': 'single_writes', 'prefix_lane': None}
    write(run / 'source-complete.json', {'identity': source_identity, 'artifact_hashes': hashes})
    identity = {'readback_commit': commit, 'plan': plan(), 'validation_set': 'registered', 'lane': 'confirmation',
        'source_complete_sha256': sha(run / 'source-complete.json'),
        'source_generations_sha256': hashes['generations.jsonl'], 'checkpoint_sha256': 'c' * 64}
    write(run / 'identity.json', identity)
    write(run / 'image-checks.json', checks)
    (run / 'bank.json').write_bytes((ROOT / 'reports/official-alignment-results-20260913/broader151-bank-manifest.json').read_bytes())
    rows = [{**row, **checks[row['image_artifact']]} for row in original]
    (run / 'generations.jsonl').write_bytes(('\n'.join(json.dumps(row) for row in rows) + '\n').encode())
    write(run / 'comparison.json', compare(rows, original, registered, bank, 'single_writes', None))
    write(run / 'complete.json', {'identity': identity, 'raw_rows': 390, 'images_checked': 78,
        'all_matched_correct_eos': True, 'chain_parity_passed': True,
        'artifact_hashes': {path.name: sha(path) for path in run.iterdir()}})
    result = collect(run, expected_commit=commit, source=source)
    assert result['all_png_pixels_verified_here']
    assert result['png_images_verified_here'] == 78
    assert result['png']['matched_correct_eos'] == 360
    path = next(source.glob('*.png'))
    Image.new('RGB', (1024, 1024), (127, 128, 128)).save(path)
    with pytest.raises(ValueError, match='Actual PNG differs'):
        collect(run, expected_commit=commit, source=source)
