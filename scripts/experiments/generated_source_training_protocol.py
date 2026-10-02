"""Fixed continuation from4f with explicit training-only generated source variation."""
from scripts.experiments.clear_retention_protocol import plan as retention_plan
from scripts.experiments.generated_source_pool_protocol import PACKAGE_SHA, CHECKPOINT_SHA

POOL_MANIFEST_SHA = '9431b9a11eac1c4b956d8178447223f92c59058ef8a851d75b575abb41b60858'
POOL_PLAN_SHA = '7603bd62e10241e0542e6b5b30168a5c2ed6d22159e7113d7754c8860b95c2f3'
REFERENCE_RESULT = '2ee5290101c83f1b53088fba8ae294e5fb71cd42f223580eb588c7ad2001a605'
REFERENCE_COMMIT = '4fbc85725d78427235757ace2661d086b896a97f'


def plan(bank, commit):
    value = retention_plan(bank, commit)
    value.pop('source_swap_result_sha256')
    value.update(schema='generated-source-continuation/v1',
        parent_checkpoint_sha256=CHECKPOINT_SHA, initial_package_manifest_sha256=PACKAGE_SHA,
        reference_phase='trained', parent_training_commit=REFERENCE_COMMIT,
        reference_commit=REFERENCE_COMMIT, reference_result_sha256=REFERENCE_RESULT,
        parent_result_sha256=REFERENCE_RESULT, parent_optimizer_steps=4832,
        cumulative_parent_unet_updates=9664,
        budget_change='Continue4f trained parameters with fresh AdamW1e-5 and4832 additional updates. Preserve complete historical-nine-expression augmentation, logical31 strata and the19328 target/noise/sigma draws. On108 existing music source conditions, choose among canonical and all eight generated source PNGs with their matching native event encodings. No source in the FM bridge. Additional optimization and source variation jointly change; not a one-factor ablation.',
        baseline_requirement='Recompute all302 canonical-source images, trajectories and3020 raw reads before optimization; require equality to4f trained endpoint, excluding only the administrative baseline/trained raw-record phase label.',
        generated_source_pool={'manifest_sha256': POOL_MANIFEST_SHA, 'plan_sha256': POOL_PLAN_SHA,
            'generation_commit': '90b41a2f1e7e4e0a8d41e709cedb69c4b32e3231',
            'policy': 'canonical-plus-eight-generated-source-pngs/v1',
            'pool_images': 24, 'qualification_raw_reads': 120,
            'canonical_conditions_unchanged': 151, 'augmented_conditions': 108,
            'source_choices_per_augmented_condition': 9,
            'selection': 'Per-question balanced seeded nine-choice cycles; occurrence=(draw_index//31)//9.',
            'source_and_condition': 'Actual selected PNG official VAE encoding and native Base event encoding; no label/query/target metadata enters the Writer.',
            'qualification': 'Require the complete pool to replay all24 tensors and120 reads; reject partial, changed or failed pool.'},
        endpoint_selection='Fixed4832 additional updates. Require full1510 development,1800 original functional and1800 previously observed expression regression cells, real CLI replay and all PNG readback. Preserve every failure; no best-checkpoint selection.',
        validation_exposure='Both functional suites are previously observed regressions. The24 training source images use independent newly fixed training noise and are not holdout data.',
        success_policy='All registered development, both complete functional matrices and CLI/PNG checks must pass strict original tokens plus immediate EOS. Aggregate improvement with remaining clear/noop errors is not complete usability.')
    return value
