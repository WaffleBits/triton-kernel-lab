"""Fused Triton SwiGLU kernel and a high-precision PyTorch oracle."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"block_size": 256}, num_warps=4),
        triton.Config({"block_size": 512}, num_warps=4),
        triton.Config({"block_size": 1024}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _swiglu_kernel(
    gate_ptr,
    up_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = (tl.program_id(axis=0) * block_size) + tl.arange(0, block_size)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    activated = gate * tl.sigmoid(gate)
    tl.store(output_ptr + offsets, activated * up, mask=mask)


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Run fused SwiGLU over two contiguous CUDA tensors with identical shape."""
    if not gate.is_cuda or not up.is_cuda:
        raise ValueError("gate and up must be CUDA tensors")
    if gate.ndim != 2 or up.ndim != 2:
        raise ValueError("gate and up must have shape [tokens, hidden_size]")
    if gate.shape != up.shape:
        raise ValueError("gate and up must have identical shapes")
    if gate.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes are float16, bfloat16, and float32")
    if gate.dtype != up.dtype:
        raise ValueError("gate and up must use the same dtype")
    if not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("gate and up must be contiguous")

    output = torch.empty_like(gate)
    n_elements = gate.numel()

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(n_elements, meta["block_size"]),)

    _swiglu_kernel[grid](
        gate,
        up,
        output,
        n_elements=n_elements,
    )
    return output


def swiglu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute an FP32 PyTorch SwiGLU oracle."""
    gate_fp32 = gate.float()
    activated = gate_fp32 * torch.sigmoid(gate_fp32)
    return (activated * up.float()).to(gate.dtype)
