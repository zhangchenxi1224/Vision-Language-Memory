"""Reproducibility helpers shared by the real GPU probes."""

from .probes import (
    DETERMINISTIC_FIXTURE_ID,
    DETERMINISTIC_FIXTURE_RGB_SHA256_1024,
    assert_no_frozen_parameter_grads,
    canonical_json_sha256,
    cuda_peak_memory_report,
    emit_json_report,
    load_source_image,
    load_initial_image,
    lora_trainable_parameters,
    probe_provenance,
    reset_cuda_peak_memory,
    seed_adapter_initialization,
    validate_e2e_pair_reports,
)

__all__ = [
    "DETERMINISTIC_FIXTURE_ID",
    "DETERMINISTIC_FIXTURE_RGB_SHA256_1024",
    "assert_no_frozen_parameter_grads",
    "canonical_json_sha256",
    "cuda_peak_memory_report",
    "emit_json_report",
    "load_source_image",
    "load_initial_image",
    "lora_trainable_parameters",
    "probe_provenance",
    "reset_cuda_peak_memory",
    "seed_adapter_initialization",
    "validate_e2e_pair_reports",
]
