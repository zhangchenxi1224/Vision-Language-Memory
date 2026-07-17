"""Protocol-preserving deterministic resize for the frozen Qwen Reader.

The R3 Reader contract presents a 1024x1024 differentiable RGB tensor to the
Hugging Face fast image processor configured with ``min_pixels=max_pixels=256^2``.
That processor uses torchvision's bicubic, antialiased 1024->256 CUDA resize.
Its forward is deterministic, but its CUDA backward is rejected by
``torch.use_deterministic_algorithms(True)``.

This module keeps the exact torchvision CUDA forward and supplies the adjoint
of the same linear resize operator from a deterministic CPU autograd replay.
This is neither an STE nor a surrogate gradient: bicubic interpolation is
linear in the source pixels, so its Jacobian is independent of pixel values.
The HF path resizes in the decoded tensor's original dtype and only then
normalizes Reader pixels in float32. The locked forward preserves that order.
Half and bfloat16 gradients are replayed in float32 on CPU and cast back to the
source dtype, matching the usual mixed-precision accumulation contract.
This preserves the mathematical adjoint, not the reduction order of the
unavailable strict CUDA backward; R3-R0 audits that numerical boundary against
three isolated native CUDA references under fixed prospective tolerances.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tv_functional


R3_QWEN_READER_RESIZE_CONTRACT = "r3-qwen-reader-1024-to-256-bicubic-antialias-cpu-adjoint.v1"
R3_QWEN_READER_INPUT_HW = (1024, 1024)
R3_QWEN_READER_OUTPUT_HW = (256, 256)
R3_QWEN_READER_GRID_THW = (1, 16, 16)
R3_QWEN_READER_PIXEL_VALUES_SHAPE = (256, 1536)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _torchvision_bicubic_antialias_resize(image: Tensor) -> Tensor:
    """Run the same batched torchvision forward used by the HF fast processor."""

    return tv_functional.resize(
        torch.stack((image,), dim=0),
        list(R3_QWEN_READER_OUTPUT_HW),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )[0]


class _DeterministicBicubicAntialiasResize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image: Tensor) -> Tensor:  # type: ignore[override]
        ctx.source_device = image.device
        ctx.source_dtype = image.dtype
        return _torchvision_bicubic_antialias_resize(image)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:  # type: ignore[override]
        # The resize is linear, hence its Jacobian depends only on the locked
        # input/output geometry. Replaying the identical torchvision operator
        # on a zero-valued CPU source computes its exact adjoint deterministically.
        if tuple(grad_output.shape) != (3, *R3_QWEN_READER_OUTPUT_HW):
            raise RuntimeError(
                "R3 Qwen Reader resize received an invalid output gradient shape: "
                f"{tuple(grad_output.shape)}."
            )
        if grad_output.dtype != ctx.source_dtype or grad_output.device != ctx.source_device:
            raise RuntimeError(
                "R3 Qwen Reader resize output gradient changed dtype/device: "
                f"expected {ctx.source_dtype}/{ctx.source_device}, "
                f"got {grad_output.dtype}/{grad_output.device}."
            )
        if not torch.isfinite(grad_output).all():
            raise RuntimeError("R3 Qwen Reader resize output gradient contains NaN or Inf.")
        cpu_grad_output = grad_output.detach().to(device="cpu", dtype=torch.float32)
        with torch.enable_grad():
            cpu_source = torch.zeros(
                (3, *R3_QWEN_READER_INPUT_HW),
                device="cpu",
                dtype=torch.float32,
                requires_grad=True,
            )
            cpu_output = _torchvision_bicubic_antialias_resize(cpu_source)
            (cpu_source_grad,) = torch.autograd.grad(
                outputs=cpu_output,
                inputs=cpu_source,
                grad_outputs=cpu_grad_output,
                create_graph=False,
                retain_graph=False,
            )
        return (cpu_source_grad.to(device=ctx.source_device, dtype=ctx.source_dtype),)


def deterministic_qwen_reader_resize(
    image: Tensor,
    *,
    contract: str = R3_QWEN_READER_RESIZE_CONTRACT,
) -> Tensor:
    """Resize one locked R3 RGB state while preserving strict autograd determinism."""

    if contract != R3_QWEN_READER_RESIZE_CONTRACT:
        raise ValueError(f"Unknown Qwen Reader resize contract: {contract!r}.")
    if not isinstance(image, Tensor):
        raise TypeError("Qwen Reader resize input must be a torch.Tensor.")
    if image.ndim != 3 or tuple(image.shape) != (3, *R3_QWEN_READER_INPUT_HW):
        raise ValueError(
            "R3 Qwen Reader resize requires one RGB tensor with shape "
            f"[3,{R3_QWEN_READER_INPUT_HW[0]},{R3_QWEN_READER_INPUT_HW[1]}], "
            f"got {tuple(image.shape)}."
        )
    if image.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "R3 Qwen Reader resize requires float16, bfloat16, or float32 input; "
            f"got {image.dtype}."
        )
    if image.device.type not in {"cpu", "cuda"}:
        raise ValueError(f"R3 Qwen Reader resize supports CPU/CUDA tensors only, got {image.device}.")
    if not torch.isfinite(image).all():
        raise ValueError("R3 Qwen Reader resize input contains NaN or Inf.")
    if bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError("R3 Qwen Reader resize input must lie in the closed [0, 1] RGB range.")

    resized = _DeterministicBicubicAntialiasResize.apply(image)
    if tuple(resized.shape) != (3, *R3_QWEN_READER_OUTPUT_HW):
        raise RuntimeError(f"R3 Qwen Reader resize returned an invalid shape: {tuple(resized.shape)}.")
    if resized.dtype != image.dtype or resized.device != image.device:
        raise RuntimeError(
            "R3 Qwen Reader resize changed dtype/device: "
            f"expected {image.dtype}/{image.device}, got {resized.dtype}/{resized.device}."
        )
    if not torch.isfinite(resized).all():
        raise RuntimeError("R3 Qwen Reader resize output contains NaN or Inf.")
    return resized


__all__ = [
    "R3_QWEN_READER_GRID_THW",
    "R3_QWEN_READER_INPUT_HW",
    "R3_QWEN_READER_OUTPUT_HW",
    "R3_QWEN_READER_PIXEL_VALUES_SHAPE",
    "R3_QWEN_READER_RESIZE_CONTRACT",
    "deterministic_qwen_reader_resize",
]
