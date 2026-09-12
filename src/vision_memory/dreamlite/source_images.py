"""Load an exact RGB source image for official edit conditioning."""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_sealed_source_image(group: dict, bank_manifest: Path):
    """The bank binds file bytes; no resizing, recoloring, or latent decoding here."""
    from vision_memory.training.latent_bank_unet import file_sha256
    if group.get("source_kind") != "sealed_rgb_1024":
        raise ValueError("Expected a sealed RGB source image")
    path = Path(group["source_image_path"])
    if not path.is_absolute():
        path = bank_manifest.parent / path
    if file_sha256(path) != group["source_image_file_sha256"]:
        raise ValueError("Sealed source image file changed")
    with Image.open(path) as opened:
        if opened.format != "PNG" or opened.mode != "RGB" or opened.size != (1024, 1024):
            raise ValueError("Sealed source must be an RGB 1024x1024 PNG")
        image = opened.copy()
    return image, path
