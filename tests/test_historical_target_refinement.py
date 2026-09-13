import copy
import json
from pathlib import Path
import tarfile

import pytest

from scripts.experiments.refine_historical_writer_targets import selection, plan, summarize

RESULTS = Path(__file__).resolve().parents[1] / 'reports/official-alignment-results-20260913'


def test_selection_matches_previously_audited_actual_targets_and_keeps_all_questions():
    bank = json.loads((RESULTS / 'historical-writer-bank-manifest.json').read_bytes())
    audit = json.loads((RESULTS / 'historical-fixed-target-readback.json').read_bytes())
    selected = selection(bank)
    assert selected == [{key: row[key] for key in selected[0]} for row in audit['selected']]
    assert {item['target_index'] for item in selected} == set(range(16))
    assert [len([item for item in selected if item['target_index'] % 2 == lane]) for lane in (0, 1)] == [8, 8]
    value = plan(bank)
    assert value['training_prompts'] == ['original_open', 'paraphrase_1', 'paraphrase_2']
    assert value['additional_updates_per_target'] == 256
    assert value['cumulative_updates_per_target'] == 512
    five = plan(bank, all_five=True)
    assert five['training_prompts'] == ['original_open', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3', 'paraphrase_4']
    assert five['selected'] == value['selected']
    assert five['additional_updates_per_target'] == value['additional_updates_per_target']
    assert five['optimizer'] == value['optimizer']
    changed = copy.deepcopy(bank)
    changed['groups'].pop(3)
    with pytest.raises(ValueError):
        selection(changed)


def test_actual_selected_readback_failures_are_not_hidden_by_qualification():
    bank = json.loads((RESULTS / 'historical-writer-bank-manifest.json').read_bytes())
    with tarfile.open(RESULTS / 'historical-fp32-readback-evidence.tgz') as stream:
        original = [json.loads(line) for line in stream.extractfile('generations.jsonl').read().splitlines()]
    totals = []
    for item in selection(bank):
        group = next(group for group in bank['groups'] if group['question_id'] == item['question_id'])
        rows = [dict(row, optimizer_step=256) for row in original if row['condition'] == 'matched'
                and row['target_index'] == item['target_index'] and f"-seed-{row['seed']}-" in item['teacher_id']]
        # Synthetic control fixtures only; actual matched outputs above are untouched.
        # Controls must not change the matched-target pass/fail decision.
        rows += [dict(row, condition=condition) for row in rows[:] for condition in ('blank', 'fixed_donor')]
        summary = summarize(rows, group)
        totals.append(summary)
        with pytest.raises(ValueError):
            summarize(rows[:-1], group)
        tampered = copy.deepcopy(rows)
        tampered[0]['generated_token_ids'] += [0]
        with pytest.raises(ValueError):
            summarize(tampered, group)
    assert sum(row['correct_eos'] for row in totals) == 154
    assert sum(row['both_forms_all_five'] for row in totals) == 14
