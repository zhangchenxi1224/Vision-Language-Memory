"""Query-free construction of full-state image, latent, and Qwen features."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .cache import TeacherState
from .renderer import FullStateCardRenderer
from .state import SemanticState, canonical_json_bytes


TEACHER_BUILD_SCHEMA = "vision_memory.full-state-teacher-build.v1"
ImageTensorCallback = Callable[[Tensor], Tensor]


def _contract_text(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string.")
    return value


@dataclass(frozen=True)
class TeacherBuildContract:
    """Version every frozen component used to derive cache tensors."""

    latent_callback_id: str
    decode_callback_id: str
    feature_callback_id: str
    vae_revision: str
    reader_revision: str
    schema: str = TEACHER_BUILD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TEACHER_BUILD_SCHEMA:
            raise ValueError(f"Unsupported teacher build schema: {self.schema!r}.")
        for field in (
            "latent_callback_id",
            "decode_callback_id",
            "feature_callback_id",
            "vae_revision",
            "reader_revision",
        ):
            object.__setattr__(self, field, _contract_text(getattr(self, field), field=field))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "latent_callback_id": self.latent_callback_id,
            "decode_callback_id": self.decode_callback_id,
            "feature_callback_id": self.feature_callback_id,
            "vae_revision": self.vae_revision,
            "reader_revision": self.reader_revision,
            "callback_boundary": (
                "card->VAE posterior-mean latent->VAE decoded RGB->Qwen visual feature; "
                "no query/options/target arguments"
            ),
        }

    @property
    def contract_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def _detached_finite_tensor(value: Any, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} callback must return a torch.Tensor.")
    result = value.detach().cpu().contiguous()
    if result.numel() == 0 or result.ndim < 2 or result.shape[0] != 1:
        raise ValueError(f"{name} callback must return a non-empty batch-one tensor.")
    if not result.is_floating_point():
        raise TypeError(f"{name} callback must return floating-point values.")
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} callback returned a non-finite value.")
    return result


def build_teacher_card(state: SemanticState, *, renderer: FullStateCardRenderer) -> Tensor:
    """Render the raw, query-independent full-state card fed to the VAE."""

    image = renderer.render_tensor(state).detach().cpu().contiguous()
    if tuple(image.shape) != (1, 3, 1024, 1024):
        raise RuntimeError("Full-state renderer violated its locked [1,3,1024,1024] contract.")
    if not torch.isfinite(image).all() or float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise RuntimeError("Full-state renderer returned invalid RGB values.")
    return image


def build_teacher_latent(card: Tensor, *, encode_image: ImageTensorCallback) -> Tensor:
    """Invoke a frozen VAE posterior-mean callback on the raw state card."""

    if not callable(encode_image):
        raise TypeError("encode_image must be callable.")
    with torch.no_grad():
        value = encode_image(card)
    return _detached_finite_tensor(value, name="latent")


def build_teacher_image(latent: Tensor, *, decode_latent: ImageTensorCallback) -> Tensor:
    """Decode the posterior-mean latent into the reader-facing RGB teacher image."""

    if not callable(decode_latent):
        raise TypeError("decode_latent must be callable.")
    with torch.no_grad():
        value = decode_latent(latent)
    image = _detached_finite_tensor(value, name="decoded image")
    if tuple(image.shape) != (1, 3, 1024, 1024):
        raise ValueError("Decoded teacher image must have shape [1, 3, 1024, 1024].")
    if float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise ValueError("Decoded teacher image values must lie in [0, 1].")
    return image


def build_teacher_query_free_feature(image: Tensor, *, encode_visual_feature: ImageTensorCallback) -> Tensor:
    """Invoke a frozen Qwen visual callback with no query-side API surface."""

    if not callable(encode_visual_feature):
        raise TypeError("encode_visual_feature must be callable.")
    with torch.no_grad():
        value = encode_visual_feature(image)
    return _detached_finite_tensor(value, name="feature")


def build_teacher_state(
    state: SemanticState,
    *,
    renderer: FullStateCardRenderer,
    contract: TeacherBuildContract,
    encode_image: ImageTensorCallback,
    decode_latent: ImageTensorCallback,
    encode_visual_feature: ImageTensorCallback,
) -> TeacherState:
    """Materialize one train-only, path-invariant teacher artifact bundle."""

    card = build_teacher_card(state, renderer=renderer)
    latent = build_teacher_latent(card, encode_image=encode_image)
    image = build_teacher_image(latent, decode_latent=decode_latent)
    feature = build_teacher_query_free_feature(image, encode_visual_feature=encode_visual_feature)
    teacher_key_payload = {
        "schema": "vision_memory.teacher-key.v1",
        "state_id": state.state_id,
        "semantic_state_sha256": state.canonical_sha256,
        "teacher_contract_sha256": contract.contract_sha256,
        "renderer_contract_sha256": renderer.contract_sha256,
    }
    teacher_key = hashlib.sha256(canonical_json_bytes(teacher_key_payload)).hexdigest()
    return TeacherState(
        state_id=state.state_id,
        teacher_key=teacher_key,
        semantic_state_sha256=state.canonical_sha256,
        teacher_contract_sha256=contract.contract_sha256,
        renderer_contract_sha256=renderer.contract_sha256,
        image=image,
        latent=latent,
        feature=feature,
    )


__all__ = [
    "TEACHER_BUILD_SCHEMA",
    "ImageTensorCallback",
    "TeacherBuildContract",
    "build_teacher_card",
    "build_teacher_image",
    "build_teacher_latent",
    "build_teacher_query_free_feature",
    "build_teacher_state",
]
