import pytest
import torch

from scripts.reporting.collect_native_endpoint_tensors import check_payload
from vision_memory.repro import canonical_tensor_sha256


@pytest.fixture
def payload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    seed = 20260915
    noise = torch.randn((1, 4, 128, 128), generator=torch.Generator().manual_seed(seed))
    latent = torch.zeros_like(noise)
    image = torch.zeros(1, 3, 1024, 1024)
    value = {'question_id': 'synthetic-test', 'noise_seed': seed, 'latent': latent, 'image': image,
        'trajectory': [noise]+[latent.clone() for _ in range(28)]}
    yield value, canonical_tensor_sha256(image)
    torch.set_num_threads(previous)


def test_actual_noise_pixels_and_all_states_are_checked(payload):
    value, digest = payload
    result = check_payload(value, 'synthetic-test', 20260915, digest)
    assert result['finite_fp32_states'] == 29
    assert result['image_sha256'] == digest


@pytest.mark.parametrize('mutation', ['nonfinite_middle', 'source_in_noise', 'wrong_pixels', 'half_precision', 'wrong_seed'])
def test_tensor_protocol_violations_are_rejected(payload, mutation):
    value, digest = payload
    if mutation == 'nonfinite_middle':
        value['trajectory'][14][0, 0, 0, 0] = float('nan')
    elif mutation == 'source_in_noise':
        value['trajectory'][0] += .5
    elif mutation == 'wrong_pixels':
        value['image'][0, 0, 0, 0] = 1.
    elif mutation == 'half_precision':
        value['trajectory'][14] = value['trajectory'][14].half()
    else:
        value['noise_seed'] += 1
    with pytest.raises(ValueError):
        check_payload(value, 'synthetic-test', 20260915, digest)
