"""Four-step reference trajectories and differentiable student rollout loss."""
from __future__ import annotations
import torch
from .latent_bank_unet import EFFECTIVE_SIGMAS


@torch.no_grad()
def reference_trajectory(flow, source, noise):
    if source.shape != noise.shape:
        raise ValueError('Source/noise shape mismatch')
    state=.5*source+.5*noise
    path=[state]
    for sigma in EFFECTIVE_SIGMAS:
        velocity,_,_=flow.evaluate(state,sigma)
        state=(state.double()-.125*velocity).to(source.dtype)
        path.append(state)
    return tuple(path)


def rollout_distillation_loss(student, reference):
    if len(student)!=5 or len(reference)!=5:
        raise ValueError('Require start and four actual rollout states')
    if any(x.shape != y.shape for x,y in zip(student,reference)):
        raise ValueError('Trajectory shape mismatch')
    errors=[(x.float()-y.detach().float()).square().mean() for x,y in zip(student[1:],reference[1:])]
    endpoint=errors[-1]
    path=torch.stack(errors[:3]).mean()
    return endpoint+path,endpoint,path
