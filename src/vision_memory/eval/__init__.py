"""Evaluation metrics and paired statistical tests."""

from .io import read_records, write_jsonl
from .metrics import compute_prefeval_metrics, correctness, diagnostic_metrics, topic_form_metrics
from .r3_micro import (
    R3_MICRO_ARTIFACT_PROVENANCE_FIELDS,
    R3_MICRO_ARTIFACT_PROVENANCE_SCHEMA,
    read_prediction_jsonl,
    score_r3_micro,
    score_set8,
    score_transition16,
    validate_r3_micro_artifact_provenance,
)
from .r3_teacher_attribution import (
    R3_TEACHER_ATTRIBUTION_SCHEMA,
    TEACHER_CONTROLS,
    score_r3_teacher_attribution,
)
from .statistics import (
    DEFAULT_PAIR_FIELDS,
    DEFAULT_STRATA_FIELDS,
    filter_preregistered_records,
    holm_correction,
    paired_hierarchical_bootstrap,
    seeded_stratified_accuracy,
)
from .synthetic import compute_synthetic_metrics

__all__ = [
    "DEFAULT_PAIR_FIELDS",
    "DEFAULT_STRATA_FIELDS",
    "R3_TEACHER_ATTRIBUTION_SCHEMA",
    "R3_MICRO_ARTIFACT_PROVENANCE_FIELDS",
    "R3_MICRO_ARTIFACT_PROVENANCE_SCHEMA",
    "TEACHER_CONTROLS",
    "compute_prefeval_metrics",
    "compute_synthetic_metrics",
    "correctness",
    "diagnostic_metrics",
    "filter_preregistered_records",
    "holm_correction",
    "paired_hierarchical_bootstrap",
    "read_records",
    "read_prediction_jsonl",
    "score_r3_micro",
    "score_r3_teacher_attribution",
    "score_set8",
    "score_transition16",
    "validate_r3_micro_artifact_provenance",
    "seeded_stratified_accuracy",
    "topic_form_metrics",
    "write_jsonl",
]
