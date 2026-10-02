from collections import Counter
import pytest
from scripts.experiments.direct_geometry_eos_training import training_prompt, teacher


def test_oracle_replays_proven_round_robin_and_blocks_heldout_gradients():
    runtime={'training_prompts':['original_open','paraphrase_1','paraphrase_2'],
             'target':{'inputs':{p:p for p in ['original_open','paraphrase_1','paraphrase_2','paraphrase_3','paraphrase_4']}}}
    assert Counter(training_prompt(runtime,s) for s in range(256))=={'original_open':86,'paraphrase_1':85,'paraphrase_2':85}
    with pytest.raises(ValueError,match='Held-out'):
        teacher(runtime,None,'paraphrase_3',require_grad=True)
    del runtime['training_prompts']
    assert all(training_prompt(runtime,s)=='original_open' for s in range(256))
