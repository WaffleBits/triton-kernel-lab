"""INT4 weight dequantization fused with matrix-vector multiplication."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _int4_gemv_kernel(
    input_ptr,
    packed_weight_ptr,
    scale_ptr,
    output_ptr,
    packed_row_stride,
    input_size: tl.constexpr,
    block_size: tl.constexpr,
):
    output_row = tl.program_id(axis=0)
    offsets = tl.arange(0, block_size)
    mask = offsets < input_size
    packed_offsets = offsets // 2
    packed = tl.load(
        packed_weight_ptr + output_row * packed_row_stride + packed_offsets,
        mask=mask,
        other=0,
    ).to(tl.int32)
    nibble = tl.where(
        offsets % 2 == 0,
        packed & 0xF,
        (packed >> 4) & 0xF,
    )
    signed_weight = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
    inputs = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + output_row).to(tl.float32)
    tl.store(output_ptr + output_row, tl.sum(inputs * signed_weight, axis=0) * scale)


def pack_int4(weights: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 rows into two nibbles per byte."""
    if weights.dtype != torch.int8 or weights.ndim != 2:
        raise ValueError("weights must be a 2D int8 tensor")
    if weights.shape[1] % 2:
        raise ValueError("input dimension must be even")
    if bool(((weights < -8) | (weights > 7)).any()):
        raise ValueError("weights must be in the signed INT4 range [-8, 7]")
    low = (weights[:, 0::2].to(torch.int16) & 0xF).to(torch.uint8)
    high = ((weights[:, 1::2].to(torch.int16) & 0xF) << 4).to(torch.uint8)
    return (low | high).contiguous()


def unpack_int4(packed: torch.Tensor, input_size: int) -> torch.Tensor:
    """Unpack bytes into an int8 matrix for the reference path."""
    low = (packed & 0xF).to(torch.int8)
    high = ((packed >> 4) & 0xF).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)[:, :input_size]


def int4_gemv_triton(
    inputs: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Dequantize per-row INT4 weights inside the GEMV kernel."""
    if not inputs.is_cuda or not packed_weight.is_cuda or not scales.is_cuda:
        raise ValueError("all tensors must be CUDA tensors")
    if inputs.ndim != 1 or packed_weight.ndim != 2 or scales.ndim != 1:
        raise ValueError("expected input [K], packed weights [N,K/2], and scales [N]")
    if packed_weight.dtype != torch.uint8 or scales.dtype != torch.float32:
        raise ValueError("packed weights must be uint8 and scales must be float32")
    if inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported input dtypes are float16, bfloat16, and float32")
    if packed_weight.shape[0] != scales.shape[0]:
        raise ValueError("weight rows and scale count must match")
    if packed_weight.shape[1] * 2 < inputs.numel():
        raise ValueError("packed weights do not cover the input dimension")
    if (
        not inputs.is_contiguous()
        or not packed_weight.is_contiguous()
        or not scales.is_contiguous()
    ):
        raise ValueError("all tensors must be contiguous")

    output_rows = packed_weight.shape[0]
    input_size = inputs.numel()
    block_size = triton.next_power_of_2(input_size)
    if block_size > 65536:
        raise ValueError("input dimension is too large for this single-program kernel")
    output = torch.empty(output_rows, device=inputs.device, dtype=torch.float32)
    _int4_gemv_kernel[(output_rows,)](
        inputs,
        packed_weight,
        scales,
        output,
        packed_weight.stride(0),
        input_size=input_size,
        block_size=block_size,
        num_warps=4 if block_size <= 2048 else 8,
    )
    return output


def int4_gemv_reference(
    inputs: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Compute the explicit dequantization plus FP32 GEMV oracle."""
    unpacked = unpack_int4(packed_weight, inputs.numel()).float()
    return torch.matmul(unpacked * scales[:, None], inputs.float())
