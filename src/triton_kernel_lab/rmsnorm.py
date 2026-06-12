"""Fused Triton RMSNorm kernel and a high-precision PyTorch oracle."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    hidden_size: tl.constexpr,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, block_size)
    mask = offsets < hidden_size

    input_row = input_ptr + row * input_row_stride
    values = tl.load(input_row + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / hidden_size
    inverse_rms = tl.rsqrt(variance + epsilon)
    weights = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    output_row = output_ptr + row * output_row_stride
    tl.store(output_row + offsets, values * inverse_rms * weights, mask=mask)


def rmsnorm_triton(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Run fused RMSNorm over the final dimension of a contiguous 2D tensor."""
    if not inputs.is_cuda or not weight.is_cuda:
        raise ValueError("inputs and weight must be CUDA tensors")
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape [rows, hidden_size]")
    if weight.ndim != 1 or weight.shape[0] != inputs.shape[1]:
        raise ValueError("weight must have shape [hidden_size]")
    if inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes are float16, bfloat16, and float32")
    if weight.dtype != inputs.dtype:
        raise ValueError("inputs and weight must use the same dtype")
    if not inputs.is_contiguous() or not weight.is_contiguous():
        raise ValueError("inputs and weight must be contiguous")

    rows, hidden_size = inputs.shape
    block_size = triton.next_power_of_2(hidden_size)
    if block_size > 65536:
        raise ValueError("hidden_size is too large for this single-program kernel")

    output = torch.empty_like(inputs)
    num_warps = 4 if block_size <= 2048 else 8
    _rmsnorm_kernel[(rows,)](
        inputs,
        weight,
        output,
        inputs.stride(0),
        output.stride(0),
        hidden_size=hidden_size,
        epsilon=epsilon,
        block_size=block_size,
        num_warps=num_warps,
    )
    return output


def rmsnorm_reference(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute an FP32-accumulating PyTorch oracle."""
    values = inputs.float()
    variance = values.square().mean(dim=-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + epsilon)
    return (normalized * weight.float()).to(inputs.dtype)

