"""Episode schemas, deterministic synthetic data, and validation helpers."""

from .episode import ChoiceReader, EpisodeRunOutput, EpisodeUpdater, ReaderOutput, run_episode
from .generator import DatasetSizes, generate_dataset
from .schema import (
    DistractorVariant,
    EventKind,
    Episode,
    QuerySpec,
    Turn,
    TurnType,
    read_jsonl,
    surface_template_signatures,
    write_jsonl,
)
from .validation import DatasetValidationError, ValidationReport, validate_dataset

__all__ = [
    "DatasetSizes",
    "DatasetValidationError",
    "ChoiceReader",
    "Episode",
    "EpisodeRunOutput",
    "EpisodeUpdater",
    "DistractorVariant",
    "EventKind",
    "QuerySpec",
    "ReaderOutput",
    "Turn",
    "TurnType",
    "ValidationReport",
    "generate_dataset",
    "run_episode",
    "read_jsonl",
    "surface_template_signatures",
    "validate_dataset",
    "write_jsonl",
]
