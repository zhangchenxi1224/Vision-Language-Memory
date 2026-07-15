"""Episode schemas, deterministic synthetic data, and validation helpers."""

from .episode import ChoiceReader, EpisodeRunOutput, EpisodeUpdater, ReaderOutput, run_episode
from .generator import DatasetSizes, generate_dataset
from .schema import EventKind, Episode, QuerySpec, Turn, TurnType, read_jsonl, write_jsonl
from .validation import DatasetValidationError, ValidationReport, validate_dataset

__all__ = [
    "DatasetSizes",
    "DatasetValidationError",
    "ChoiceReader",
    "Episode",
    "EpisodeRunOutput",
    "EpisodeUpdater",
    "EventKind",
    "QuerySpec",
    "ReaderOutput",
    "Turn",
    "TurnType",
    "ValidationReport",
    "generate_dataset",
    "run_episode",
    "read_jsonl",
    "validate_dataset",
    "write_jsonl",
]
