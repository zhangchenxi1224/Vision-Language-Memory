"""Analytic diagnostic for a finite, uniform teacher bank and independent noise.

For the existing bridge, X_sigma | Z_j is Gaussian with mean
sigma * source + (1 - 2*sigma) * Z_j and covariance sigma**2 I.
The conditional mean velocity is (X_sigma - E[Z|X_sigma]) / sigma.
This evaluator uses the entire bank at inference; it is NOT a trained Writer.
"""
from __future__ import annotations
import torch


class EmpiricalBankFlow:
    def __init__(self, source: torch.Tensor, teachers: torch.Tensor):
        if teachers.ndim != source.ndim + 1 or teachers.shape[1:] != source.shape or not len(teachers):
            raise ValueError("Teachers must be [N, *source.shape]")
        self.source = source.detach().double()
        self.teachers = teachers.detach().to(source.device).double()
        self.flat = self.teachers.flatten(1)
        self.norm2 = self.flat.square().sum(1)
        self.mean = self.teachers.mean(0)

    def evaluate(self, state: torch.Tensor, sigma: float):
        if not 0 < sigma <= .5 or state.shape != self.source.shape:
            raise ValueError("Require matching state and 0 < sigma <= .5")
        x = state.double()
        residual = (x - sigma*self.source).flatten()
        a = 1 - 2*sigma
        # The shared ||residual||**2 term cancels in posterior normalization.
        logits = (2*a*(self.flat @ residual) - a*a*self.norm2) / (2*sigma*sigma)
        weights = logits.softmax(0)
        mean = (weights @ self.flat).reshape_as(x)
        velocity = (x-mean)/sigma
        conditional_variance = (weights * (self.flat-mean.flatten()).square().mean(1)).sum() / (sigma*sigma)
        return velocity, weights, conditional_variance
