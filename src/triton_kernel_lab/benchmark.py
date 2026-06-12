"""CLI for correctness checks, GPU timing, and baseline regression gates."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from triton_kernel_lab.report import compare_with_baseline, summarize_timings


def parse_shape(value: str) -> tuple[int, int]:
    """Parse ROWSxHIDDEN into a positive shape."""
    try:
        rows_text, hidden_text = value.lower().split("x", maxsplit=1)
        rows, hidden = int(rows_text), int(hidden_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("shape must look like 512x4096") from exc
    if rows <= 0 or hidden <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return rows, hidden


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a fused Triton RMSNorm kernel against a PyTorch oracle."
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        dest="shapes",
        help="Benchmark shape ROWSxHIDDEN. Repeat for multiple cases.",
    )
    parser.add_argument(
        "--dtype",
        action="append",
        choices=("float16", "bfloat16", "float32"),
        dest="dtypes",
        help="Tensor dtype. Repeat for multiple dtypes.",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("artifacts/latest.json"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-regression-percent", type=float, default=10.0)
    return parser


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _measure_cuda_ms(torch: Any, operation: Any, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        operation()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def _effective_bandwidth_gbps(shape: tuple[int, int], element_size: int, p50_ms: float) -> float:
    rows, hidden = shape
    logical_bytes = ((2 * rows * hidden) + hidden) * element_size
    return logical_bytes / (p50_ms / 1000.0) / 1e9


def _run_case(
    torch: Any,
    triton: Any,
    shape: tuple[int, int],
    dtype_name: str,
    epsilon: float,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    from triton_kernel_lab.rmsnorm import rmsnorm_reference, rmsnorm_triton

    dtype = getattr(torch, dtype_name)
    inputs = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn((shape[1],), device="cuda", dtype=dtype)

    expected = rmsnorm_reference(inputs, weight, epsilon)
    actual = rmsnorm_triton(inputs, weight, epsilon)
    compiled_reference = torch.compile(rmsnorm_reference, fullgraph=True)
    compiled_actual = compiled_reference(inputs, weight, epsilon)
    torch.cuda.synchronize()
    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1e-6)
    max_abs_error = difference.max().item()
    max_rel_error = (difference / denominator).max().item()

    atol = 2e-2 if dtype_name in {"float16", "bfloat16"} else 1e-5
    rtol = 2e-2 if dtype_name in {"float16", "bfloat16"} else 1e-5
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise RuntimeError(
            f"correctness check failed for {shape} {dtype_name}: "
            f"max_abs={max_abs_error:.6g}, max_rel={max_rel_error:.6g}"
        )
    if not torch.allclose(compiled_actual, expected, atol=atol, rtol=rtol):
        raise RuntimeError(
            f"torch.compile correctness check failed for {shape} {dtype_name}"
        )

    triton_samples = _measure_cuda_ms(
        torch,
        lambda: rmsnorm_triton(inputs, weight, epsilon),
        warmup,
        repeats,
    )
    compiled_samples = _measure_cuda_ms(
        torch,
        lambda: compiled_reference(inputs, weight, epsilon),
        warmup,
        repeats,
    )
    eager_samples = _measure_cuda_ms(
        torch,
        lambda: rmsnorm_reference(inputs, weight, epsilon),
        warmup,
        repeats,
    )
    triton_summary = summarize_timings(triton_samples)
    compiled_summary = summarize_timings(compiled_samples)
    eager_summary = summarize_timings(eager_samples)
    triton_p50 = float(triton_summary["p50_ms"])
    compiled_p50 = float(compiled_summary["p50_ms"])
    eager_p50 = float(eager_summary["p50_ms"])

    return {
        "kernel": "rmsnorm",
        "shape": list(shape),
        "dtype": dtype_name,
        "epsilon": epsilon,
        "correctness": {
            "passed": True,
            "atol": atol,
            "rtol": rtol,
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
        },
        "triton": {
            **triton_summary,
            "samples_ms": triton_samples,
            "effective_bandwidth_gbps": _effective_bandwidth_gbps(
                shape, inputs.element_size(), triton_p50
            ),
        },
        "torch_compile": {
            **compiled_summary,
            "samples_ms": compiled_samples,
            "effective_bandwidth_gbps": _effective_bandwidth_gbps(
                shape, inputs.element_size(), compiled_p50
            ),
        },
        "pytorch_eager": {
            **eager_summary,
            "samples_ms": eager_samples,
            "effective_bandwidth_gbps": _effective_bandwidth_gbps(
                shape, inputs.element_size(), eager_p50
            ),
        },
        "speedup_vs_torch_compile": compiled_p50 / triton_p50,
        "speedup_vs_pytorch_eager": eager_p50 / triton_p50,
    }


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        import triton
    except ImportError as exc:
        raise RuntimeError(
            "GPU dependencies are missing. Install with: pip install -e '.[gpu]'"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    torch.manual_seed(args.seed)
    shapes = args.shapes or [(128, 1024), (512, 4096), (2048, 4096)]
    dtypes = args.dtypes or ["float16", "bfloat16"]
    properties = torch.cuda.get_device_properties(0)

    cases = [
        _run_case(
            torch=torch,
            triton=triton,
            shape=shape,
            dtype_name=dtype_name,
            epsilon=args.epsilon,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for dtype_name in dtypes
        for shape in shapes
    ]

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_memory_bytes": properties.total_memory,
            "cuda_runtime": torch.version.cuda,
            "torch": torch.__version__,
            "triton": triton.__version__,
        },
        "methodology": {
            "warmup_iterations": args.warmup,
            "timed_iterations": args.repeats,
            "seed": args.seed,
            "timing": "CUDA events around each queued operation, synchronized after the batch",
            "oracle": "PyTorch implementation with FP32 accumulation",
            "bandwidth": "Logical input read + weight read + output write bytes divided by p50",
        },
        "cases": cases,
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(f"GPU: {report['system']['gpu']}")
    print("shape       dtype      Triton p50   compile p50   vs compile   bandwidth")
    for case in report["cases"]:
        rows, hidden = case["shape"]
        print(
            f"{rows:>4}x{hidden:<5} "
            f"{case['dtype']:<10} "
            f"{case['triton']['p50_ms']:>9.4f} ms "
            f"{case['torch_compile']['p50_ms']:>9.4f} ms "
            f"{case['speedup_vs_torch_compile']:>9.2f}x "
            f"{case['triton']['effective_bandwidth_gbps']:>8.1f} GB/s"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = _build_report(args)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_summary(report)
    print(f"report: {args.output}")

    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        regressions = compare_with_baseline(
            report["cases"],
            baseline["cases"],
            args.max_regression_percent,
        )
        if regressions:
            print("performance regression gate failed:", file=sys.stderr)
            for regression in regressions:
                print(
                    f"  {regression['case']}: {regression['regression_percent']:.2f}% "
                    f"({regression['baseline_p50_ms']:.4f} -> "
                    f"{regression['current_p50_ms']:.4f} ms)",
                    file=sys.stderr,
                )
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
