"""Frozen-scale normalized distillation loss for full-state teacher tensors."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .cache import TeacherState
from .state import canonical_json_bytes


TEACHER_LOSS_SCHEMA = "vision_memory.teacher-loss.smoothl1-cosine-normalized.v1"
LATENT_NORMALIZATION_EPSILON = 1e-6
CALIBRATION_DENOMINATOR_EPSILON = 1e-6


@dataclass(frozen=True)
class FrozenTeacherLossCalibration:
    """Three positive denominators frozen before training begins."""

    latent_scale: float
    image_scale: float
    feature_scale: float
    schema: str = TEACHER_LOSS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TEACHER_LOSS_SCHEMA:
            raise ValueError(f"Unsupported teacher loss schema: {self.schema!r}.")
        for field in ("latent_scale", "image_scale", "feature_scale"):
            value = getattr(self, field)
            if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be a positive finite frozen scalar.")
            object.__setattr__(self, field, float(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "latent_metric": "smooth-l1-after-independent-per-channel-standardization",
            "latent_normalization_epsilon": LATENT_NORMALIZATION_EPSILON,
            "image_metric": "smooth-l1-on-decoded-reader-image",
            "feature_metric": "mean-tokenwise-one-minus-cosine-over-last-dimension",
            "normalization": "raw-component-divided-by-frozen-calibration-scale",
            "calibration_denominator_epsilon": CALIBRATION_DENOMINATOR_EPSILON,
            "aggregation": "exact-unweighted-mean-of-three-components",
            "latent_scale": self.latent_scale,
            "image_scale": self.image_scale,
            "feature_scale": self.feature_scale,
        }

    @property
    def contract_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenTeacherLossCalibration":
        expected = {
            "schema",
            "latent_metric",
            "latent_normalization_epsilon",
            "image_metric",
            "feature_metric",
            "normalization",
            "aggregation",
            "calibration_denominator_epsilon",
            "latent_scale",
            "image_scale",
            "feature_scale",
        }
        if set(value) != expected:
            raise ValueError("Teacher loss calibration fields differ from the locked schema.")
        calibration = cls(
            schema=value["schema"],
            latent_scale=value["latent_scale"],
            image_scale=value["image_scale"],
            feature_scale=value["feature_scale"],
        )
        if dict(value) != calibration.to_dict():
            raise ValueError("Teacher loss calibration metric metadata drifted from the locked contract.")
        return calibration


@dataclass(frozen=True)
class TeacherDistillationLossOutput:
    loss: Tensor
    teacher_state_id: str
    latent_raw: Tensor
    image_raw: Tensor
    feature_raw: Tensor
    latent_normalized: Tensor
    image_normalized: Tensor
    feature_normalized: Tensor
    student_image: Tensor
    student_feature: Tensor


def _validated_student(student: Tensor, teacher: Tensor, *, name: str) -> tuple[Tensor, Tensor]:
    if not isinstance(student, Tensor):
        raise TypeError(f"student_{name} must be a torch.Tensor.")
    if not student.is_floating_point() or not torch.isfinite(student).all():
        raise ValueError(f"student_{name} must contain finite floating-point values.")
    if tuple(student.shape) != tuple(teacher.shape):
        raise ValueError(
            f"student_{name} shape {tuple(student.shape)} differs from teacher shape {tuple(teacher.shape)}."
        )
    target = teacher.detach().to(device=student.device, dtype=student.dtype)
    return student, target


def normalize_latent_per_channel(latent: Tensor) -> Tensor:
    """Deterministically standardize each batch/channel over all remaining axes."""

    if not isinstance(latent, Tensor) or latent.ndim < 3 or not latent.is_floating_point():
        raise ValueError("Latent normalization requires a floating tensor with shape [B,C,...].")
    if not torch.isfinite(latent).all():
        raise ValueError("Latent normalization received a non-finite value.")
    dimensions = tuple(range(2, latent.ndim))
    centered = latent - latent.mean(dim=dimensions, keepdim=True)
    variance = centered.square().mean(dim=dimensions, keepdim=True)
    return centered * torch.rsqrt(variance + LATENT_NORMALIZATION_EPSILON)


def composite_teacher_distillation_loss(
    *,
    student_latent: Tensor,
    student_image: Tensor,
    student_feature: Tensor,
    teacher: TeacherState,
    calibration: FrozenTeacherLossCalibration,
) -> TeacherDistillationLossOutput:
    """Compute the locked equal-third latent/image/feature teacher objective."""

    if not isinstance(teacher, TeacherState):
        raise TypeError("teacher must be a validated TeacherState.")
    if not isinstance(calibration, FrozenTeacherLossCalibration):
        raise TypeError("calibration must be FrozenTeacherLossCalibration.")
    student_latent, latent_target = _validated_student(student_latent, teacher.latent, name="latent")
    student_image, image_target = _validated_student(student_image, teacher.image, name="image")
    student_feature, feature_target = _validated_student(student_feature, teacher.feature, name="feature")

    latent_raw = F.smooth_l1_loss(
        normalize_latent_per_channel(student_latent),
        normalize_latent_per_channel(latent_target),
    )
    image_raw = F.smooth_l1_loss(student_image, image_target)
    feature_raw = (1.0 - F.cosine_similarity(student_feature, feature_target, dim=-1)).mean()
    latent_normalized = latent_raw / (calibration.latent_scale + CALIBRATION_DENOMINATOR_EPSILON)
    image_normalized = image_raw / (calibration.image_scale + CALIBRATION_DENOMINATOR_EPSILON)
    feature_normalized = feature_raw / (calibration.feature_scale + CALIBRATION_DENOMINATOR_EPSILON)
    loss = (latent_normalized + image_normalized + feature_normalized) / 3.0
    if not torch.isfinite(loss):
        raise RuntimeError("Composite teacher distillation loss is non-finite.")
    return TeacherDistillationLossOutput(
        loss=loss,
        teacher_state_id=teacher.state_id,
        latent_raw=latent_raw,
        image_raw=image_raw,
        feature_raw=feature_raw,
        latent_normalized=latent_normalized,
        image_normalized=image_normalized,
        feature_normalized=feature_normalized,
        student_image=student_image,
        student_feature=student_feature,
    )


__all__ = [
    "TEACHER_LOSS_SCHEMA",
    "LATENT_NORMALIZATION_EPSILON",
    "CALIBRATION_DENOMINATOR_EPSILON",
    "FrozenTeacherLossCalibration",
    "TeacherDistillationLossOutput",
    "composite_teacher_distillation_loss",
    "normalize_latent_per_channel",
]
