"""Four-step reference trajectories and differentiable student rollout loss."""
from __future__ import annotations
import torch
from .latent_bank_unet import EFFECTIVE_SIGMAS


@torch.no_grad()
def reference_trajectory(flow, source, noise, *, sigmas=EFFECTIVE_SIGMAS):
    if source.shape != noise.shape:
        raise ValueError('Source/noise shape mismatch')
    sigmas=tuple(float(s) for s in sigmas)
    if len(sigmas)!=4 or not all(a>b for a,b in zip(sigmas,(*sigmas[1:],0.))):
        raise ValueError('Require four positive descending scheduler sigmas')
    if flow.start_sigma!=sigmas[0]:
        raise ValueError('Reference field and sampler start sigma differ')
    # Match the actual post-shift scheduler and sampler arithmetic exactly.
    state=source.mul(1-sigmas[0]).add(noise,alpha=sigmas[0])
    path=[state]
    for sigma,next_sigma in zip(sigmas,(*sigmas[1:],0.)):
        velocity,_,_=flow.evaluate(state,sigma)
        state=(state.double()+(next_sigma-sigma)*velocity).to(source.dtype)
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
