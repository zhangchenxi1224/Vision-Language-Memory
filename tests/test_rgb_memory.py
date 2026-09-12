from types import SimpleNamespace

import pytest
import torch

from vision_memory.dreamlite import rgb_memory


def pipeline():
    modules = {name: torch.nn.Linear(1, 1).requires_grad_(False).eval() for name in ("unet", "vae", "text_encoder")}
    modules["vae"].dtype = modules["text_encoder"].dtype = torch.float32
    return SimpleNamespace(**modules,
        image_processor=SimpleNamespace(preprocess=lambda image: torch.full((1, 4, 2, 2), image.getpixel((0, 0))[0] / 255.)),
        prepare_image_latents=lambda image, **kw: image.clone())


def test_recurrent_writer_uses_previous_generated_rgb_and_reads_do_not_edit(monkeypatch):
    edits = []
    class Sampler:
        def __init__(self, pipe, **kwargs):
            self.kwargs = kwargs
        def __call__(self, **kwargs):
            edits.append((self.kwargs, kwargs))
            return SimpleNamespace(latents=torch.ones(1, 4, 2, 2), trajectory=(kwargs["noise_latents"],))
    monkeypatch.setattr(rgb_memory, "NativeBaseEditSampler", Sampler)
    monkeypatch.setattr(rgb_memory, "decode_model_latents_unit_interval", lambda *a, **k: torch.full((1, 3, 1024, 1024), .25))
    memory = rgb_memory.OfficialRGBMemory(pipeline())
    first = memory.write("set a value", seed=1)
    assert first.image.getpixel((0, 0)) == (64, 64, 64)
    assert first.pixels[0, 0, 0, 0] == 64 / 255.
    read_image = memory.image
    read_image.putpixel((0, 0), (0, 0, 0))
    assert len(edits) == 1 and memory.image.getpixel((0, 0)) == (64, 64, 64)
    memory.write("keep the value unchanged", seed=2)
    assert len(edits) == 2
    assert edits[0][0]["source_image"].getpixel((0, 0)) == (128, 128, 128)
    assert edits[1][0]["source_image"].getpixel((0, 0)) == (64, 64, 64)
    torch.testing.assert_close(edits[1][1]["source_latents"], torch.full((1, 4, 2, 2), 64 / 255.), rtol=0, atol=0)
    assert edits[1][0]["event_text"] == "keep the value unchanged"


def test_failed_edit_and_model_mutation_cannot_silently_change_memory(monkeypatch):
    class FailedSampler:
        def __init__(self, *a, **k):
            pass
        def __call__(self, **k):
            raise RuntimeError("sampling failed")
    monkeypatch.setattr(rgb_memory, "NativeBaseEditSampler", FailedSampler)
    pipe = pipeline()
    memory = rgb_memory.OfficialRGBMemory(pipe)
    with pytest.raises(RuntimeError, match="sampling failed"):
        memory.write("update", seed=1)
    assert memory.image.getpixel((0, 0)) == (128, 128, 128)
    with torch.no_grad():
        pipe.unet.weight.add_(1.)
    with pytest.raises(RuntimeError, match="weights changed"):
        memory.write("update", seed=1)
