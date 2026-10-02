import copy
from collections import Counter
import json
from pathlib import Path

from scripts.experiments.augment_official_transition_bank import assemble
from vision_memory.training.latent_bank_unet import balanced_draw

ROOT = Path(__file__).resolve().parents[1]


def bank():
    return json.loads((ROOT / 'reports/official-alignment-results-20260913/source-transitions/manifest.json').read_text())


def test_wordings_preserve_source_target_bindings_and_source_dependent_noop():
    parent = bank()
    before = copy.deepcopy(parent)
    result = assemble(parent)
    assert parent == before
    assert len(result['groups']) == len(result['teachers']) == 45
    assert len({g['semantic_question_id'] for g in result['groups']}) == 1
    originals = {g['question_id']: g for g in parent['groups']}
    teachers = {t['teacher_id']: t for t in parent['teachers']}
    for group in result['groups']:
        original = originals[group['parent_transition_group']]
        assert {k: v for k, v in group.items() if k.startswith('source_')} == {
            k: v for k, v in original.items() if k.startswith('source_')}
        assert group['answer'] == original['answer']
        assert group['question_variants'] == original['question_variants']
    for teacher in result['teachers']:
        original = teachers[teacher['parent_transition_teacher']]
        assert {k: v for k, v in teacher.items() if k.startswith('latent_')} == {
            k: v for k, v in original.items() if k.startswith('latent_')}
    for index in range(3):
        noops = [g for g in result['groups'] if g['operation'] == 'noop' and g['wording_index'] == index]
        assert len(noops) == 3 and len({g['event_text'] for g in noops}) == 1
        assert len({g['answer'] for g in noops}) == 3
    old_cases = json.loads((ROOT / 'reports/official-cfg1-confirmation-plan-20260913.json').read_text())
    old_rewordings = {case['event'] for case in old_cases if case['style'] != 'trained'}
    assert not old_rewordings.intersection(g['event_text'] for g in result['groups'])


def test_full_registered_draw_budget_covers_every_condition_equally():
    groups = assemble(bank())['groups']
    counts = Counter()
    for draw in range(2880 * 4):
        selected, _, _, _ = balanced_draw(groups, seed=20260913, step=draw)
        counts[selected['question_id']] += 1
    assert len(counts) == 45 and set(counts.values()) == {256}
