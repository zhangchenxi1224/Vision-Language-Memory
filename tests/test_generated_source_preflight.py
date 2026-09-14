import json
from pathlib import Path

from scripts.probes.generated_source_training_preflight import selected_draws
from scripts.train.generated_source_augmentation import source_variant_index
from vision_memory.training.latent_bank_unet import balanced_draw


def test_gpu_preflight_selection_covers_every_source_state_and_choice():
    root = Path(__file__).resolve().parents[1]
    bank = json.loads((root / 'reports/official-alignment-results-20260913/broader151-bank-manifest.json').read_bytes())
    indices = selected_draws(bank)
    assert len(indices) == len(set(indices)) == 27
    covered = set()
    for index in indices:
        group, _, _, _ = balanced_draw(bank['groups'], 20260915, index, sampling_strategy='logical_condition')
        covered.add((group['source_state'], source_variant_index(20260915, index, group['question_id'])))
    assert covered == {(state, i) for state in ('ambient', 'jazz', 'clear') for i in range(9)}
