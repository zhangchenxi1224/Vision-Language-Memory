"""Image-only recurrent memory using the official Base editing pipeline."""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
import torch

from .latent_codec import decode_model_latents_unit_interval
from .native_base import NativeBaseEditSampler


@dataclass(frozen=True)
class MemoryWrite:
    source_latent: torch.Tensor
    noise: torch.Tensor
    latent: torch.Tensor
    pixels: torch.Tensor
    image: Image.Image
    trajectory: tuple[torch.Tensor, ...]


class OfficialRGBMemory:
    """Persist only generated RGB pixels between edits, never an oracle/answer ledger.

    Supply an already loaded, frozen FP32 official Base pipeline. The caller can
    read ``image`` any number of times without calling the Writer. A new event
    always invokes a native edit, including a no-op or distractor event.
    """

    def __init__(self, pipeline, *, image: Image.Image | None = None, guidance_scale: float = 7.5):
        self.pipeline = pipeline
        self.guidance_scale = guidance_scale
        self._image = (image if image is not None else Image.new("RGB", (1024, 1024), (128, 128, 128))).copy()
        if self._image.mode != "RGB" or self._image.size != (1024, 1024):
            raise ValueError("Memory must be an exact RGB 1024x1024 image")
        if pipeline.vae.dtype != torch.float32 or pipeline.text_encoder.dtype != torch.float32:
            raise ValueError("This verified Base memory protocol requires FP32")
        self._versions = self._frozen_versions()

    def _frozen_versions(self):
        versions = {}
        for label in ("unet", "vae", "text_encoder"):
            module = getattr(self.pipeline, label)
            if module.training:
                raise ValueError("Memory inference requires evaluation mode")
            for name, parameter in module.named_parameters():
                if parameter.requires_grad or parameter.grad is not None:
                    raise ValueError("Memory inference requires frozen models without gradients")
                versions[label + "." + name] = int(parameter._version)
        return versions

    @property
    def image(self):
        # Reader or saving code cannot accidentally mutate the session's state.
        return self._image.copy()

    @torch.no_grad()
    def write(self, event: str, *, seed: int) -> MemoryWrite:
        if not isinstance(event, str) or not event.strip():
            raise ValueError("A nonempty event-only prompt is required")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
            raise ValueError("Supply an explicit nonnegative 63-bit noise seed")
        if self._frozen_versions() != self._versions:
            raise RuntimeError("Writer weights changed between events")
        pipe = self.pipeline
        source_image = self.image
        device = next(pipe.vae.parameters()).device
        source = pipe.prepare_image_latents(pipe.image_processor.preprocess(source_image), dtype=torch.float32, device=device)
        noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32).to(device)
        sampler = NativeBaseEditSampler(pipe, source_image=source_image, event_text=event, guidance_scale=self.guidance_scale)
        output = sampler(source_latents=source, noise_latents=noise, num_steps=28)
        decoded = decode_model_latents_unit_interval(pipe.vae, output.latents, clamp=True).cpu()
        if decoded.shape != (1, 3, 1024, 1024) or not torch.isfinite(decoded).all():
            raise RuntimeError("Invalid generated memory image")
        rgb = (decoded[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy()
        image = Image.fromarray(rgb)
        # Read the same quantized pixels that will be saved and used next time.
        pixels = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.
        if self._frozen_versions() != self._versions:
            raise RuntimeError("Writer weights changed during an event")
        result = MemoryWrite(source.cpu(), noise.cpu(), output.latents.cpu(), pixels, image.copy(),
                             tuple(value.cpu() for value in output.trajectory))
        # Failed edits leave the prior memory intact.
        self._image = image
        return result
