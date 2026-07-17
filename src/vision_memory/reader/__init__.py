"""Frozen Reader helpers that preserve gradients with respect to image inputs."""

from .deterministic_resize import (
    R3_QWEN_READER_GRID_THW,
    R3_QWEN_READER_INPUT_HW,
    R3_QWEN_READER_OUTPUT_HW,
    R3_QWEN_READER_PIXEL_VALUES_SHAPE,
    R3_QWEN_READER_RESIZE_CONTRACT,
    deterministic_qwen_reader_resize,
)
from .qwen3vl import (
    ChoiceScoreOutput,
    ListwiseChoiceLossOutput,
    ReaderLossOutput,
    VisualFeatureOutput,
    qwen3vl_choice_nll,
    qwen3vl_listwise_choice_ce,
    qwen3vl_query_free_visual_features,
    qwen3vl_target_only_ce,
)

__all__ = [
    "ChoiceScoreOutput",
    "ListwiseChoiceLossOutput",
    "R3_QWEN_READER_GRID_THW",
    "R3_QWEN_READER_INPUT_HW",
    "R3_QWEN_READER_OUTPUT_HW",
    "R3_QWEN_READER_PIXEL_VALUES_SHAPE",
    "R3_QWEN_READER_RESIZE_CONTRACT",
    "ReaderLossOutput",
    "VisualFeatureOutput",
    "deterministic_qwen_reader_resize",
    "qwen3vl_choice_nll",
    "qwen3vl_listwise_choice_ce",
    "qwen3vl_query_free_visual_features",
    "qwen3vl_target_only_ce",
]
