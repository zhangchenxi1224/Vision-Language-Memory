import json
from pathlib import Path

from scripts.experiments.augment_official_transition_bank import assemble
from scripts.probes.transition_validation_plan import events, NOOPS, plan


def test_new_wordings_are_held_out_from_training_and_prior_confirmation():
    root = Path(__file__).resolve().parents[1]
    bank = json.loads((root / 'reports/official-alignment-results-20260913/source-transitions/manifest.json').read_text())
    trained = {g['event_text'] for g in assemble(bank)['groups']}
    observed = {c['event'] for c in json.loads((root / 'reports/official-cfg1-confirmation-plan-20260913.json').read_text())}
    fresh = {e for state in ('ambient', 'jazz', 'clear') for e in events(state)} | set(NOOPS)
    assert len(fresh) == 8 and not fresh.intersection(trained | observed)
    current = plan()
    assert current == plan()
    assert sum(len(c['noise_seeds']) for c in current['single_writes']) == 72
    assert len(current['rgb_chains']) == 16
    assert sum(len(c['steps']) for c in current['rgb_chains']) == 96
    assert all([s['operation'] for s in c['steps']] == ['write', 'noop'] * 3 for c in current['rgb_chains'])
