"""Numerical derivatives with frozen inputs and a nonlinear time-dependent U-Net."""
from pathlib import Path
import sys
import unittest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from vision_memory.dreamlite import DifferentiableDreamLiteMobileSampler
from test_differentiable_mobile import MockFlowScheduler


class NonlinearTimeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameters_vector = nn.Parameter(torch.tensor([.3, -.1], dtype=torch.float64))

    def forward(self, x, *, timestep, **kwargs):
        a, b = self.parameters_vector
        return (a*x.tanh() + b*(timestep[:, None, None, None]/1000).sin(),)


class SingleTargetGradientAudit(unittest.TestCase):
    def test_checkpoint_and_finite_difference_with_all_inputs_frozen(self):
        torch.manual_seed(9)
        model = NonlinearTimeUNet()
        sampler = DifferentiableDreamLiteMobileSampler(unet=model, scheduler=MockFlowScheduler())
        source = torch.randn(1, 4, 8, 8, dtype=torch.float64)
        noise = torch.randn_like(source)
        target = torch.randn_like(source)
        args = dict(source_latents=source, noise_latents=noise,
            prompt_embeds=torch.ones(1, 3, 4, dtype=torch.float64),
            prompt_attention_mask=torch.ones(1, 3, dtype=torch.long), edit_start_sigma=.5)
        def loss():
            return (sampler(**args).latents-target).square().mean()
        gradients = []
        for checkpoint in [False, True]:
            sampler.checkpoint_unet = checkpoint
            gradients.append(torch.autograd.grad(loss(), model.parameters_vector)[0])
        torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-10, atol=1e-12)
        self.assertGreater(float(gradients[0].norm()), 0)
        initial = model.parameters_vector.detach().clone()
        eps = 1e-5
        for i in range(2):
            values = []
            with torch.no_grad():
                for sign in [-1, 1]:
                    model.parameters_vector.copy_(initial)
                    model.parameters_vector[i] += sign*eps
                    values.append(float(loss()))
                model.parameters_vector.copy_(initial)
            self.assertAlmostEqual((values[1]-values[0])/(2*eps), float(gradients[0][i]), places=7)
        # One small actual negative-gradient update must lower the same objective.
        before = float(loss().detach())
        with torch.no_grad(): model.parameters_vector.add_(gradients[0], alpha=-.01)
        self.assertLess(float(loss().detach()), before)


if __name__ == "__main__":
    unittest.main()
