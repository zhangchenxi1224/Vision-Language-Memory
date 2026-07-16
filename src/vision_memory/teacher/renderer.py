"""Deterministic 1024x1024 RGB renderer for complete semantic states."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION
from torch import Tensor

from .state import SEMANTIC_STATE_SCHEMA, SemanticState, canonical_json_bytes, require_sha256


FULL_STATE_RENDERER_SCHEMA = "vision_memory.full-state-card.v1"
FULL_STATE_RESOLUTION = 1024
_LAYOUT_VERSION = "four-column-four-row-compact-v1"
_BACKGROUND = (238, 242, 247)
_HEADER = (24, 38, 58)
_CARD = (255, 255, 255)
_BORDER = (116, 132, 151)
_TEXT = (20, 28, 40)
_MUTED = (73, 87, 105)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FixedFontContract:
    """Font bytes and Pillow version are mandatory parts of deterministic rendering."""

    font_id: str
    path: Path
    sha256: str
    pillow_version: str
    header_size: int = 40
    body_size: int = 18

    def __post_init__(self) -> None:
        if not isinstance(self.font_id, str) or not self.font_id.strip() or self.font_id != self.font_id.strip():
            raise ValueError("font_id must be a non-empty trimmed string.")
        require_sha256(self.sha256, field="font.sha256")
        if not isinstance(self.pillow_version, str) or not self.pillow_version:
            raise ValueError("pillow_version must be non-empty.")
        for field in ("header_size", "body_size"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer.")
        object.__setattr__(self, "path", Path(self.path))

    def contract_dict(self) -> dict[str, Any]:
        # The host path is deliberately excluded; identical font bytes may live at
        # different absolute locations on local and cluster machines.
        return {
            "font_id": self.font_id,
            "sha256": self.sha256,
            "pillow_version": self.pillow_version,
            "header_size": self.header_size,
            "body_size": self.body_size,
            "layout_engine": "Pillow-BASIC",
        }

    @property
    def contract_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.contract_dict())).hexdigest()

    def load(self) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
        if PILLOW_VERSION != self.pillow_version:
            raise RuntimeError(
                f"Pillow version drifted: contract={self.pillow_version!r}, runtime={PILLOW_VERSION!r}."
            )
        try:
            resolved = self.path.expanduser().resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"Locked teacher font is unavailable: {self.path}") from exc
        if not resolved.is_file():
            raise RuntimeError(f"Locked teacher font is not a file: {resolved}")
        actual_sha256 = file_sha256(resolved)
        if actual_sha256 != self.sha256:
            raise RuntimeError(
                f"Locked teacher font SHA256 drifted: expected {self.sha256}, got {actual_sha256}."
            )
        try:
            header = ImageFont.truetype(
                str(resolved), self.header_size, layout_engine=ImageFont.Layout.BASIC
            )
            body = ImageFont.truetype(str(resolved), self.body_size, layout_engine=ImageFont.Layout.BASIC)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Locked teacher font could not be loaded: {resolved}") from exc
        return header, body


class FullStateCardRenderer:
    """Render one path-independent card without query, option, or answer fields."""

    resolution = FULL_STATE_RESOLUTION
    max_entries = 16

    def __init__(self, font: FixedFontContract) -> None:
        self.font_contract = font
        self._header_font, self._body_font = font.load()

    @property
    def contract_dict(self) -> dict[str, Any]:
        return {
            "schema": FULL_STATE_RENDERER_SCHEMA,
            "semantic_state_schema": SEMANTIC_STATE_SCHEMA,
            "layout_version": _LAYOUT_VERSION,
            "resolution": [self.resolution, self.resolution],
            "mode": "RGB",
            "max_entries": self.max_entries,
            "font": self.font_contract.contract_dict(),
            "colors": {
                "background": list(_BACKGROUND),
                "header": list(_HEADER),
                "card": list(_CARD),
                "border": list(_BORDER),
                "text": list(_TEXT),
                "muted": list(_MUTED),
            },
        }

    @property
    def contract_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.contract_dict)).hexdigest()

    def _text_width(self, text: str) -> int:
        left, _top, right, _bottom = self._body_font.getbbox(text)
        return right - left

    def _wrap(self, text: str, *, width: int) -> list[str]:
        words = text.split(" ")
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if self._text_width(candidate) <= width:
                current = candidate
                continue
            if not current:
                raise ValueError("Semantic-state text contains a token too wide for the locked card layout.")
            lines.append(current)
            current = word
            if self._text_width(current) > width:
                raise ValueError("Semantic-state text contains a token too wide for the locked card layout.")
        if current:
            lines.append(current)
        return lines

    def _entry_lines(self, state: SemanticState, index: int, *, width: int) -> list[tuple[str, tuple[int, int, int]]]:
        entry = state.sorted_entries[index]
        if entry.status == "active":
            assert entry.value_text is not None
            rendered_value = entry.value_text
        elif entry.status == "cleared":
            rendered_value = "<CLEARED>"
        else:
            rendered_value = "<UNSET>"
        fields = (
            (f"ENTITY: {entry.entity_text}", _TEXT),
            (f"SLOT: {entry.slot_text}", _MUTED),
            (f"STATE: {rendered_value}", _TEXT),
        )
        return [(line, color) for text, color in fields for line in self._wrap(text, width=width)]

    def render(self, state: SemanticState) -> Image.Image:
        if not isinstance(state, SemanticState):
            raise TypeError("FullStateCardRenderer requires SemanticState.")
        if len(state.entries) > self.max_entries:
            raise ValueError(f"Semantic state exceeds the locked {self.max_entries}-entry card capacity.")

        image = Image.new("RGB", (self.resolution, self.resolution), color=_BACKGROUND)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.resolution - 1, 134), fill=_HEADER)
        draw.text((48, 28), "SEMANTIC STATE", font=self._header_font, fill=(255, 255, 255))
        draw.text(
            (49, 88),
            "CURRENT VALUES - QUERY INDEPENDENT",
            font=self._body_font,
            fill=(211, 220, 232),
        )

        if not state.entries:
            draw.rounded_rectangle((48, 178, 975, 308), radius=16, fill=_CARD, outline=_BORDER, width=2)
            draw.text((78, 222), "NO SAVED SEMANTIC ENTRIES", font=self._body_font, fill=_MUTED)
            return image

        margin = 40
        gap = 14
        content_top = 154
        content_bottom = 980
        columns = 4
        rows = 4
        card_width = (self.resolution - 2 * margin - gap) // columns
        row_gap = 12
        card_height = (content_bottom - content_top - row_gap * (rows - 1)) // rows
        inner_margin = 12
        line_bbox = self._body_font.getbbox("Ag")
        line_height = line_bbox[3] - line_bbox[1] + 5

        for index, _entry in enumerate(state.sorted_entries):
            row, column = divmod(index, columns)
            left = margin + column * (card_width + gap)
            top = content_top + row * (card_height + row_gap)
            right = left + card_width
            bottom = top + card_height
            lines = self._entry_lines(state, index, width=card_width - 2 * inner_margin)
            required_height = len(lines) * line_height
            if required_height > card_height - 2 * inner_margin:
                raise ValueError("Semantic-state entry overflows the locked card cell; truncation is forbidden.")
            draw.rounded_rectangle((left, top, right, bottom), radius=12, fill=_CARD, outline=_BORDER, width=2)
            y = top + inner_margin
            for line, color in lines:
                draw.text((left + inner_margin, y), line, font=self._body_font, fill=color)
                y += line_height
        return image

    def render_tensor(self, state: SemanticState) -> Tensor:
        image = self.render(state)
        buffer = bytearray(image.tobytes())
        tensor = torch.frombuffer(buffer, dtype=torch.uint8).clone()
        return tensor.reshape(self.resolution, self.resolution, 3).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def rgb_sha256(image: Image.Image) -> str:
    if not isinstance(image, Image.Image):
        raise TypeError("rgb_sha256 requires a PIL image.")
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(b"vision-memory-full-state-rgb-v1\0")
    digest.update(f"{rgb.width}x{rgb.height}\0".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


__all__ = [
    "FULL_STATE_RENDERER_SCHEMA",
    "FULL_STATE_RESOLUTION",
    "FixedFontContract",
    "FullStateCardRenderer",
    "file_sha256",
    "rgb_sha256",
]
