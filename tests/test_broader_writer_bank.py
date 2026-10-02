import copy
import json
from pathlib import Path

import pytest

from scripts.experiments.build_broader_writer_bank import merge
from scripts.experiments.refine_historical_writer_targets import selection


def test_merge_preserves_every_original_event_source_and_query():
    directory = Path(__file__).resolve().parents[1] / 'reports/official-alignment-results-20260913'
    transition = json.loads((directory / 'transition-wording-bank-manifest.json').read_bytes())
    historical = json.loads((directory / 'historical-writer-bank-manifest.json').read_bytes())
    # Metadata-only fixture; it makes no claim that unrun refinement passed.
    refined = copy.deepcopy(historical)
    selected = {item['question_id']: item['teacher_id'] for item in selection(historical)}
    refined['teachers'] = [teacher for teacher in refined['teachers'] if teacher['teacher_id'] in selected.values()]
    for group in refined['groups']:
        group['teacher_ids'] = [selected[group['question_id']]]
    result = merge(transition, refined, {'fixture': True})
    assert result['groups'][:45] == transition['groups']
    assert result['groups'][45:] == refined['groups']
    assert result['teachers'] == transition['teachers'] + refined['teachers']
    assert result['semantic_question_count'] == 17
    changed = copy.deepcopy(refined)
    changed['groups'].pop()
    with pytest.raises(ValueError):
        merge(transition, changed, {})
    changed = copy.deepcopy(refined)
    changed['snapshots']['qwen_reader']['revision'] = 'other'
    with pytest.raises(ValueError):
        merge(transition, changed, {})
