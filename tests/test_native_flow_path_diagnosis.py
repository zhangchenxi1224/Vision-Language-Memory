import pytest
import torch
from scripts.reporting.diagnose_native_flow_path import clean_estimates


def test_euler_estimates_recover_official_clean_target_on_shifted_schedule():
    target = torch.tensor([.2, -.6], dtype=torch.float64)
    noise = torch.tensor([1.4, -.1], dtype=torch.float64)
    sigmas = [1., .97, .81, .45, .1]
    path = [(1 - sigma) * target + sigma * noise for sigma in [*sigmas, 0.]]
    assert all(torch.allclose(value, target, rtol=0, atol=1e-13) for value in clean_estimates(path, sigmas))
    for invalid in ([1., 1., .5, .2, .1], [.5, .4, .3, .2, .1], [1., .9, .7, .5, 0.]):
        with pytest.raises(ValueError):
            list(clean_estimates(path, invalid))
