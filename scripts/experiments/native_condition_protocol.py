"""Same-budget logical sampling; train on the native Base conditional encoder row."""
from scripts.experiments.logical_sampling_protocol import plan as logical_plan, BANK_SHA

REFERENCE_COMMIT='bb34092ab0d1292c87d16d9632716b218f54054b'
REFERENCE_RESULT='717522c160bfeffc13fd1789ef6b6e6694ed3d8778dbfd047eb6f10e5bd2f4ad'
FIRST_STEP_EVIDENCE='24cdf7bb88dc23298167a165d3c011cde066a1c866bfdfe0944ee2d9149ec611'


def plan(bank,commit):
    value=logical_plan(bank,commit)
    value.update(schema='native-base-training-condition-comparison/v1',
        reference_commit=REFERENCE_COMMIT,reference_result_sha256=REFERENCE_RESULT,
        first_step_evidence_sha256=FIRST_STEP_EVIDENCE,training_prompt_style='native_base',
        training_condition='Upstream Base edit prompts [empty,empty,diptych(event)] with the original source image; cache only conditional row2 and its exact mask. No target/query/teacher metadata in condition.',
        budget_change='Same4832 updates, global4, logical sampling, seed20260915, all19328 condition/teacher/Gaussian/sigma draws, original warm package, bank and fresh AdamW. Only training condition encoding changes from raw-event to native Base conditional row.',
        baseline_requirement='Remeasure all302 native baseline images and3020 raw Reader rows. Require bitwise latent/RGB/all29 states and identical raw records. Runtime whitelist permits only training condition hashes and train_prompt; every inference/model/source/schedule field remains equal.',
        alignment_difference='Official target/noise FM and native Base inference remain unchanged. Explicit departure from upstream LoRA raw-event training text to match upstream native Base edit encoding; not an unmodified reproduction of the LoRA example.',
        endpoint_selection='Fixed4832; no checkpoint or case selection. Keep original full independent matrices and CLI replay.')
    return value
