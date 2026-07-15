"""Evaluation metrics and paired statistical tests."""

from .io import read_records, write_jsonl
from .metrics import compute_prefeval_metrics, correctness, diagnostic_metrics, topic_form_metrics
from .statistics import DEFAULT_PAIR_FIELDS, holm_correction, paired_hierarchical_bootstrap
from .synthetic import compute_synthetic_metrics

__all__ = [
    "DEFAULT_PAIR_FIELDS",
    "compute_prefeval_metrics",
    "compute_synthetic_metrics",
    "correctness",
    "diagnostic_metrics",
    "holm_correction",
    "paired_hierarchical_bootstrap",
    "read_records",
    "topic_form_metrics",
    "write_jsonl",
]
