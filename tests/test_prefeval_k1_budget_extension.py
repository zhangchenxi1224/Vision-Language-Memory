from copy import deepcopy

import pytest

from scripts.experiments.prefeval_k1_budget_extension import validate_extension


def manifests():
    old = dict(arm='B', stage='write', split='train', effective_batch=4,
               steps=23360, snapshot_steps=[2048], implementation_sha256='old',
               targets={'preference':'teacher-hash'}, initial_variants_sha256='v0-v1',
               seed=20260924, parent_sha256='original-parent', training_variant_indices=[0,1])
    new = deepcopy(old)
    new.update(steps=93440, snapshot_steps=[46720,70080], implementation_sha256='new',
               continued_from_sha256='full-resume-hash')
    return old, new


def test_only_prospective_bookkeeping_changes_are_allowed():
    old, new = manifests()
    validate_extension(old, new, step=23360, cursor=93440)


@pytest.mark.parametrize('key,value', [('targets',{'preference':'different'}),
    ('initial_variants_sha256','different'), ('seed',0), ('effective_batch',16),
    ('snapshot_steps',[50000]), ('steps',100000), ('parent_sha256','different')])
def test_rejects_scientific_input_or_budget_changes(key,value):
    old, new = manifests()
    new[key] = value
    with pytest.raises(ValueError):
        validate_extension(old, new, step=23360, cursor=93440)


def test_rejects_reset_or_incomplete_optimizer_cursor():
    old, new = manifests()
    with pytest.raises(ValueError):
        validate_extension(old, new, step=0, cursor=0)
