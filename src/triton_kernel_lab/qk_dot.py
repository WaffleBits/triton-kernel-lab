"""Triton query-key row dot product for decode block scoring."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_dot_kernel(
    query_ptr,
    key_ptr,
    output_ptr,
    query_row_stride,
    key_row_stride,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, block_size)
    mask = offsets < head_dim
    query = tl.load(
        query_ptr + row * query_row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    key = tl.load(
        key_ptr + row * key_row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(output_ptr + row, tl.sum(query * key, axis=0))


def qk_dot_triton(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Compute one FP32 score for every aligned query/key row."""
    if not query.is_cuda or not key.is_cuda:
        raise ValueError("query and key must be CUDA tensors")
    if query.ndim != 2 or key.ndim != 2 or query.shape != key.shape:
        raise ValueError("query and key must have identical [rows, head_dim] shapes")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes are float16, bfloat16, and float32")
    if query.dtype != key.dtype:
        raise ValueError("query and key must use the same dtype")
    if not query.is_contiguous() or not key.is_contiguous():
        raise ValueError("query and key must be contiguous")

    rows, head_dim = query.shape
    block_size = triton.next_power_of_2(head_dim)
    if block_size > 65536:
        raise ValueError("head_dim is too large for this single-program kernel")
    output = torch.empty(rows, device=query.device, dtype=torch.float32)
    _qk_dot_kernel[(rows,)](
        query,
        key,
        output,
        query.stride(0),
        key.stride(0),
        head_dim=head_dim,
        block_size=block_size,
        num_warps=4 if block_size <= 2048 else 8,
    )
    return output


def qk_dot_reference(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Compute the FP32 PyTorch oracle."""
    return (query.float() * key.float()).sum(dim=-1)

