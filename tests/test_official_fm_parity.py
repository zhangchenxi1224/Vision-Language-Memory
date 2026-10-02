"""Execute the pinned upstream arithmetic, not a second handwritten FM formula."""
import ast
from pathlib import Path
import subprocess

import pytest
import torch

from vision_memory.training.latent_bank_unet import OFFICIAL_REFERENCE_COMMIT, official_flow_bridge


def test_bridge_and_loss_gradient_match_pinned_official_source():
    root = Path(__file__).resolve().parents[1] / "third_party" / "DreamLite"
    path = root / "lora" / "train_edit_lora.py"
    if not path.exists():
        pytest.skip("Pinned official checkout required for source parity")
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() == OFFICIAL_REFERENCE_COMMIT
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"noisy_latents", "target", "timesteps"}:
                assert name not in selected
                selected[name] = node
    code = compile(ast.Module(body=[selected[k] for k in ("noisy_latents", "target", "timesteps")], type_ignores=[]),
                   str(path), "exec")
    torch.manual_seed(915)
    for sigma in (0., .1259, .5, .8769, 1.):
        target, noise = torch.randn(2, 1, 4, 3, 5)
        values = dict(torch=torch, latents=target, noise=noise,
                      sigmas=torch.tensor([sigma]), sigmas_expanded=torch.tensor([sigma]).view(1,1,1,1))
        exec(code, values)
        state, velocity = official_flow_bridge(noise, target, sigma)
        torch.testing.assert_close(state, values["noisy_latents"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(velocity, values["target"], rtol=0, atol=0)
        gain = torch.tensor(.3, requires_grad=True)
        reference_loss = ((gain * values["noisy_latents"] - values["target"]).float()**2).mean()
        local_loss = ((gain * state - velocity).float()**2).mean()
        torch.testing.assert_close(local_loss, reference_loss)
        torch.testing.assert_close(torch.autograd.grad(local_loss, gain)[0], torch.autograd.grad(reference_loss, gain)[0])


def test_raw_condition_uses_original_image_and_no_question_or_diptych_wrapper():
    from vision_memory.dreamlite.conditioning import encode_image_edit_condition
    class Pipe:
        def encode_prompt(self, **kwargs):
            self.observed = kwargs
            return torch.ones(1,3,4), torch.ones(1,3)
    pipe, original_image = Pipe(), object()
    result = encode_image_edit_condition(pipe, original_image, "make the room blue", device="cpu", dtype=torch.float32)
    assert pipe.observed["image"] is original_image
    assert pipe.observed["prompts"] == ["make the room blue"]
    assert not result.prompt_embeds.requires_grad
