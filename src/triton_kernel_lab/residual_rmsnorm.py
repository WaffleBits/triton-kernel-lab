"""Fused residual addition and RMSNorm with dual outputs."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _residual_rmsnorm_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    normalized_ptr,
    combined_ptr,
    input_row_stride,
    residual_row_stride,
    normalized_row_stride,
    combined_row_stride,
    hidden_size: tl.constexpr,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, block_size)
    mask = offsets < hidden_size
    inputs = tl.load(
        input_ptr + row * input_row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + row * residual_row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    combined = inputs + residual
    variance = tl.sum(combined * combined, axis=0) / hidden_size
    inverse_rms = tl.rsqrt(variance + epsilon)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        combined_ptr + row * combined_row_stride + offsets,
        combined,
        mask=mask,
    )
    tl.store(
        normalized_ptr + row * normalized_row_stride + offsets,
        combined * inverse_rms * weight,
        mask=mask,
    )


def residual_rmsnorm_triton(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized output and updated residual in one kernel."""
    if not inputs.is_cuda or not residual.is_cuda or not weight.is_cuda:
        raise ValueError("inputs, residual, and weight must be CUDA tensors")
    if inputs.ndim != 2 or residual.shape != inputs.shape:
        raise ValueError("inputs and residual must have identical 2D shapes")
    if weight.ndim != 1 or weight.shape[0] != inputs.shape[1]:
        raise ValueError("weight must have shape [hidden_size]")
    if inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes are float16, bfloat16, and float32")
    if residual.dtype != inputs.dtype or weight.dtype != inputs.dtype:
        raise ValueError("all tensors must use the same dtype")
    if not inputs.is_contiguous() or not residual.is_contiguous() or not weight.is_contiguous():
        raise ValueError("all tensors must be contiguous")

    rows, hidden_size = inputs.shape
    block_size = triton.next_power_of_2(hidden_size)
    if block_size > 65536:
        raise ValueError("hidden_size is too large for this single-program kernel")
    normalized = torch.empty_like(inputs)
    combined = torch.empty_like(inputs)
    _residual_rmsnorm_kernel[(rows,)](
        inputs,
        residual,
        weight,
        normalized,
        combined,
        inputs.stride(0),
        residual.stride(0),
        normalized.stride(0),
        combined.stride(0),
        hidden_size=hidden_size,
        epsilon=epsilon,
        block_size=block_size,
        num_warps=4 if block_size <= 2048 else 8,
    )
    return normalized, combined


def residual_rmsnorm_reference(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the dual-output FP32 PyTorch oracle."""
    combined = inputs.float() + residual.float()
    variance = combined.square().mean(dim=-1, keepdim=True)
    normalized = combined * torch.rsqrt(variance + epsilon) * weight.float()
    return normalized.to(inputs.dtype), combined.to(inputs.dtype)

