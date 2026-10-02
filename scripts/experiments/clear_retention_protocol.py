"""Fixed repair experiment continuing the native Writer with smaller updates."""
from scripts.experiments.historical_wording_protocol import plan as wording_plan, REFERENCE_COMMIT, REFERENCE_RESULT

PARENT_PACKAGE = 'ef4d4a91b99d0b05a1561e05875b2d7a7d772267e81c3bf9174dc861f52a8314'
PARENT_CHECKPOINT = 'd473825a403ad5c219681a09fc9e8a4270a63133393fc8ee30c0baac4c48d539'
SOURCE_SWAP_RESULT = '7d250091c39613eaf6d5b8848a3cd95a0f3851ddafc32caaff0256b0350c86a9'
LEARNING_RATE = 1e-5


def plan(bank, commit):
    value = wording_plan(bank, commit)
    value['optimizer']['lr'] = LEARNING_RATE
    value.update(schema='native-clear-retention-continuation/v1',
        parent_checkpoint_sha256=PARENT_CHECKPOINT, initial_package_manifest_sha256=PARENT_PACKAGE,
        reference_phase='trained', source_swap_result_sha256=SOURCE_SWAP_RESULT,
        parent_optimizer_steps=4832,
        budget_change='Continue03 trained parameters with fresh AdamW1e-5,4832 additional updates and19328 draws. Initialization and learning rate jointly change; not a one-factor ablation. Preserve the complete historical-nine-expressions-v1 augmentation, logical31 sampling and native condition training.',
        baseline_requirement='Recompute all302 images, trajectories and3020 raw reads before optimization; require equality to03 trained endpoint, excluding only the administrative baseline/trained raw-record phase label.',
        endpoint_selection='Fixed4832 additional updates. Full1510 development,1800 original functional and1800 already-observed fresh-wording-v1 regression cells, real CLI replay and PNG readback. Preserve every failure; no best-checkpoint selection.',
        validation_exposure='Both original and fresh-wording-v1 suites have been observed and are regression tests for this continuation. No new holdout or unseen-entity claim.',
        success_policy='Require every registered development and both functional matrices to pass strict answer-plus-immediate-EOS and real CLI/PNG parity. Do not call a higher aggregate usable while clear/noop regressions remain.')
    return value
