from pathlib import Path
import json
import pytest
from vision_memory.training.frozen_oracle_eos import question_prompts, endpoint_gate, INSTRUCTIONS
from vision_memory.training.frozen_oracle_geometry import canonical_hash


@pytest.mark.parametrize('time', ['At this later check', 'After applying this update first', 'Before any later update'])
def test_question_changes_preserve_instructions_time_entity_and_no_choices(time):
    query=f'R3 Train Standard Templates 03: {time}, what is the current music preference for the indigo desk train 001123? Choose exactly one option.'
    prompts=question_prompts(query)
    assert len(prompts)==5 and len(set(prompts.values()))==5
    for prompt in prompts.values():
        assert prompt.splitlines()[1:]==INSTRUCTIONS.splitlines()
        assert 'indigo desk train 001123' in prompt and 'music' in prompt
        assert time.lower() in prompt.lower()
        assert 'ambient' not in prompt and 'Choose' not in prompt
    assert prompts['original_open'].splitlines()[0]==query.removesuffix(' Choose exactly one option.')


def test_original_success_is_not_all_paraphrases_and_prefix_is_not_exact():
    rows=[{'condition': 'matched', 'prompt_id': p, 'strict_correct': i!=1,
           'answer_prefix_token_exact': True, 'overgeneration': i==1}
          for i,p in enumerate(['original_open','paraphrase_1','paraphrase_2','paraphrase_3','paraphrase_4'])]
    result=endpoint_gate(rows)
    assert result['qa_pass'] and not result['paraphrase_all_correct']
    rows[0].update(strict_correct=False,overgeneration=True)
    assert not endpoint_gate(rows)['qa_pass']
    with pytest.raises(ValueError): endpoint_gate(rows[:-1])


def test_full_manifest_is_fresh_eos_and_choice_path_absent():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/experiments/frozen_oracle_eos.json').read_text())
    man=json.loads((root/'configs/experiments/frozen_oracle_eos_manifest.json').read_text())
    assert cfg['manifest_sha256']==canonical_hash(man)
    assert len(man['runs'])==158 and len({r['run_id'] for r in man['runs']})==158
    assert all(r['run_id'].startswith('eos-') for r in man['runs'])
    worker=(root/'scripts/experiments/run_frozen_oracle_eos_worker.py').read_text()
    assert 'choice_reader_callable' not in worker and 'format_mcq_query' not in worker
    assert cfg['open_supervision']['train_prompts']==['original_open']
    assert cfg['open_supervision']['lambda_eos']==1.0
