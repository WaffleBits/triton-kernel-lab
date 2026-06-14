"""Launch one custom Triton kernel repeatedly for Nsight Compute filtering."""

from __future__ import annotations

import argparse

from .benchmark import parse_shape


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        required=True,
        choices=(
            "qk-dot",
            "paged-gather",
            "selective-attention",
            "residual-rmsnorm",
            "int4-gemv",
        ),
    )
    parser.add_argument("--shape", required=True, type=parse_shape)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args(argv)

    import torch

    torch.manual_seed(17)
    dtype = getattr(torch, args.dtype)
    rows, hidden = args.shape

    if args.kernel == "qk-dot":
        from triton_kernel_lab.qk_dot import qk_dot_triton

        left = torch.randn(args.shape, device="cuda", dtype=dtype)
        right = torch.randn(args.shape, device="cuda", dtype=dtype)
        def operation() -> object:
            return qk_dot_triton(left, right)

    elif args.kernel == "paged-gather":
        from triton_kernel_lab.paged_gather import paged_gather_triton

        source = torch.randn((rows * 4, hidden), device="cuda", dtype=dtype)
        indices = torch.randperm(rows * 4, device="cuda")[:rows].to(torch.int32)
        def operation() -> object:
            return paged_gather_triton(source, indices)

    elif args.kernel == "residual-rmsnorm":
        from triton_kernel_lab.residual_rmsnorm import residual_rmsnorm_triton

        inputs = torch.randn(args.shape, device="cuda", dtype=dtype)
        residual = torch.randn(args.shape, device="cuda", dtype=dtype)
        weight = torch.randn(hidden, device="cuda", dtype=dtype)
        def operation() -> object:
            return residual_rmsnorm_triton(inputs, residual, weight)

    elif args.kernel == "selective-attention":
        from triton_kernel_lab.selective_attention import selective_attention_triton

        query = torch.randn(hidden, device="cuda", dtype=dtype)
        keys = torch.randn((rows * 4, hidden), device="cuda", dtype=dtype)
        values = torch.randn((rows * 4, hidden), device="cuda", dtype=dtype)
        indices = torch.randperm(rows * 4, device="cuda")[:rows].to(torch.int32)

        def operation() -> object:
            return selective_attention_triton(query, keys, values, indices)

    else:
        from triton_kernel_lab.int4_gemv import int4_gemv_triton, pack_int4

        inputs = torch.randn(hidden, device="cuda", dtype=dtype)
        weights = torch.randint(
            -8, 8, (rows, hidden), device="cuda", dtype=torch.int8
        )
        packed = pack_int4(weights)
        scales = torch.rand(rows, device="cuda", dtype=torch.float32) * 0.2 + 0.01
        def operation() -> object:
            return int4_gemv_triton(inputs, packed, scales)

    for _ in range(args.warmup):
        operation()
    torch.cuda.synchronize()
    operation()
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
