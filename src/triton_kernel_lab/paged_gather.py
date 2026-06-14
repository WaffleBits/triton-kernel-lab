"""Triton paged KV row gather and PyTorch oracle."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_gather_kernel(
    source_ptr,
    indices_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    output_elements: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * block_size + tl.arange(0, block_size)
    mask = offsets < output_elements
    output_row = offsets // hidden_size
    column = offsets % hidden_size
    source_row = tl.load(indices_ptr + output_row, mask=mask, other=0)
    values = tl.load(
        source_ptr + source_row * hidden_size + column,
        mask=mask,
        other=0.0,
    )
    tl.store(output_ptr + offsets, values, mask=mask)


def paged_gather_triton(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather selected contiguous KV rows from a larger cache."""
    if not source.is_cuda or not indices.is_cuda:
        raise ValueError("source and indices must be CUDA tensors")
    if source.ndim != 2 or indices.ndim != 1:
        raise ValueError("source must be 2D and indices must be 1D")
    if source.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported source dtypes are float16, bfloat16, and float32")
    if indices.dtype != torch.int32:
        raise ValueError("indices must use int32")
    if not source.is_contiguous() or not indices.is_contiguous():
        raise ValueError("source and indices must be contiguous")

    output = torch.empty(
        (indices.numel(), source.shape[1]),
        device=source.device,
        dtype=source.dtype,
    )
    output_elements = output.numel()
    block_size = 256
    _paged_gather_kernel[(triton.cdiv(output_elements, block_size),)](
        source,
        indices,
        output,
        hidden_size=source.shape[1],
        output_elements=output_elements,
        block_size=block_size,
        num_warps=4,
    )
    return output


def paged_gather_reference(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Compute the PyTorch row-gather oracle."""
    return source.index_select(0, indices.long())

