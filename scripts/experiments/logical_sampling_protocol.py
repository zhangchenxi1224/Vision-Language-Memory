"""Preregister a fixed-budget, same-initialization comparison of data weighting."""
from collections import Counter
import copy

from scripts.experiments.broader_writer_protocol import training_plan, SEED
from vision_memory.training.latent_bank_unet import balanced_draw, logical_condition_strata

REFERENCE_COMMIT = '84cdfdb58ace96954243de5caf427948717c9abf'
REFERENCE_RESULT = 'c230451494fd8809621757b747bf3635d80fda5bc998e208a757128167147f82'
BANK_SHA = 'c27cd65dab809deabb5f2cb08891517d3590244651d08a8c6763c84fea901592'


def plan(bank, commit):
    if len(commit) != 40 or any(char not in '0123456789abcdef' for char in commit):
        raise ValueError('Require the exact training source revision')
    value = copy.deepcopy(training_plan(BANK_SHA, commit))
    strata = logical_condition_strata(bank['groups'])
    membership = {bank['groups'][index]['question_id']: key for key, indices in strata.items() for index in indices}
    counts, logical_counts = Counter(), Counter()
    for index in range(value['draws']):
        group, teacher, noise, sigma = balanced_draw(bank['groups'], SEED, index, sampling_strategy='logical_condition')
        # This bank has one teacher per condition. The noise/sigma stream must
        # stay identical to the paired experiment, even though conditions change.
        if (noise, sigma) != balanced_draw(bank['groups'], SEED, index)[2:]:
            raise ValueError('Sampling comparison changed the Gaussian/time stream')
        counts[group['question_id']] += 1
        logical_counts[membership[group['question_id']]] += 1
    if len(strata) != 31 or len(counts) != 151 or max(logical_counts.values()) - min(logical_counts.values()) != 1:
        raise ValueError('Unexpected logical-condition coverage')
    value.pop('draws_per_group')
    value.update(schema='logical-condition-sampling-comparison/v1',
        sampling={'strategy': 'logical_condition', 'strata': {key: [bank['groups'][index]['question_id'] for index in indices]
            for key, indices in strata.items()}, 'exact_draws_per_group': dict(sorted(counts.items())),
            'exact_draws_per_stratum': dict(sorted(logical_counts.items())), 'writer_input': False},
        reference_commit=REFERENCE_COMMIT, reference_result_sha256=REFERENCE_RESULT,
        baseline_requirement='Measure all302 images and3020 raw rows again; require bitwise equality of latent, RGB, all29 trajectory states and every raw Reader record before optimization.',
        budget_change='Same4832 updates, global4, seed20260915, original warm package, bank and fresh AdamW. Only logical-condition/expression sampling changes.',
        endpoint_selection='Fixed4832; retain every failure, no checkpoint or case selection.',
        validation_exposure='The c2ec407 expressions/noises have now been observed. Reuse all of them as a registered paired diagnostic comparison, never claim fresh holdouts.',
        success_policy='Require all1510 development,360 single-write,480 chain and960 prefix strict answer-plus-EOS cells; CLI replay must agree. Passing these observed cases alone is insufficient for unseen-entity or multi-fact usability.')
    return value
