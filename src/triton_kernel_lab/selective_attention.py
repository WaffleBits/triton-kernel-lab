"""Fused Triton attention over an indirect list of selected KV rows."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _selective_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    indices_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    selected_rows: tl.constexpr,
    row_block: tl.constexpr,
    column_block: tl.constexpr,
):
    row_offsets = tl.arange(0, row_block)
    column_offsets = tl.arange(0, column_block)
    rows = row_offsets[:, None]
    columns = column_offsets[None, :]
    row_mask = rows < selected_rows
    column_mask = columns < hidden_size
    selected_indices = tl.load(indices_ptr + rows, mask=row_mask, other=0)
    query = tl.load(
        query_ptr + columns,
        mask=column_mask,
        other=0.0,
    ).to(tl.float32)
    keys = tl.load(
        key_ptr + selected_indices * hidden_size + columns,
        mask=row_mask & column_mask,
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(keys * query, axis=1) / math.sqrt(hidden_size)
    scores = tl.where(row_offsets < selected_rows, scores, -float("inf"))
    weights = tl.exp(scores - tl.max(scores, axis=0))
    weights = weights / tl.sum(weights, axis=0)
    values = tl.load(
        value_ptr + selected_indices * hidden_size + columns,
        mask=row_mask & column_mask,
        other=0.0,
    ).to(tl.float32)
    output = tl.sum(values * weights[:, None], axis=0)
    tl.store(output_ptr + column_offsets, output, mask=column_offsets < hidden_size)


def selective_attention_triton(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute exact attention over only the selected indirect KV rows."""
    if not all(tensor.is_cuda for tensor in (query, keys, values, selected_indices)):
        raise ValueError("all inputs must be CUDA tensors")
    if query.ndim != 1 or keys.ndim != 2 or values.shape != keys.shape:
        raise ValueError("expected query [hidden] and matching keys/values [rows, hidden]")
    if keys.shape[1] != query.shape[0] or selected_indices.ndim != 1:
        raise ValueError("input dimensions are inconsistent")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes are float16, bfloat16, and float32")
    if keys.dtype != query.dtype or values.dtype != query.dtype:
        raise ValueError("query, keys, and values must use the same dtype")
    if selected_indices.dtype != torch.int32:
        raise ValueError("selected_indices must use int32")
    if selected_indices.numel() <= 0:
        raise ValueError("at least one KV row must be selected")
    if not all(
        tensor.is_contiguous() for tensor in (query, keys, values, selected_indices)
    ):
        raise ValueError("all inputs must be contiguous")

    hidden_size = query.numel()
    selected_rows = selected_indices.numel()
    row_block = triton.next_power_of_2(selected_rows)
    column_block = triton.next_power_of_2(hidden_size)
    if row_block * column_block > 65536:
        raise ValueError("selected attention tile is too large for one Triton program")
    output = torch.empty_like(query)
    _selective_attention_kernel[(1,)](
        query,
        keys,
        values,
        selected_indices,
        output,
        hidden_size=hidden_size,
        selected_rows=selected_rows,
        row_block=row_block,
        column_block=column_block,
        num_warps=8 if row_block * column_block >= 16384 else 4,
    )
    return output


def selective_attention_reference(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute the PyTorch oracle over the same selected rows."""
    selected_keys = keys.index_select(0, selected_indices.long())
    selected_values = values.index_select(0, selected_indices.long())
    scores = torch.mv(selected_keys.float(), query.float()) / math.sqrt(query.numel())
    weights = torch.softmax(scores, dim=0)
    return torch.mv(selected_values.float().transpose(0, 1), weights).to(query.dtype)
