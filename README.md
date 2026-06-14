# Triton Kernel Lab

Correctness-first GPU kernel work for inference performance engineering.

The repository contains fused RMSNorm and SwiGLU kernels written in Triton,
high-precision PyTorch oracles, PyTorch eager and `torch.compile` baselines, raw
latency samples, effective-bandwidth modeling, machine-readable reports, and a
baseline regression gate.

## What It Demonstrates

- Triton GPU kernel development with FP32 reduction, fused normalization, and
  fused gated activation.
- Shape-aware SwiGLU launch autotuning across block sizes and warp counts.
- Correctness validation across shapes and low-precision dtypes before timing,
  including validation of the `torch.compile` baseline against the FP32 oracle.
- GPU benchmarking with warmup, CUDA events, p50/p95/p99/max latency, explicit
  cache state, and reproducible environment metadata.
- Performance reasoning that distinguishes a logical bandwidth model from
  hardware-counter evidence.
- CPU-only CI for parsers, statistics, report contracts, linting, and CLI shape.
- A lab-oriented regression gate for controlled GPU runners.

## Quick Start

Linux or WSL with an NVIDIA GPU and a compatible driver is required for GPU runs.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[gpu,dev]"

triton-kernel-lab \
  --kernel rmsnorm \
  --kernel swiglu \
  --shape 128x1024 \
  --shape 512x4096 \
  --shape 2048x4096 \
  --dtype float16 \
  --dtype bfloat16 \
  --output artifacts/latest.json
```

The benchmark refuses to publish timing data when a correctness case fails.
Cache-cold timing is the default; use `--cache-mode hot` only when a
resident-working-set result is intentionally required.

## Measured Result

Cache-cold run on June 14, 2026 using an RTX 5070 Ti, CUDA 13.0, PyTorch 2.12,
and Triton 3.7. Each case used 100 warmups, 500 timed samples, and a 256 MiB
cache-eviction buffer outside the timed region. The table shows FP16; the
artifact also includes BF16.

| Kernel | Shape | Triton p50 | `torch.compile` p50 | Speedup |
|---|---:|---:|---:|---:|
| RMSNorm | 128 x 1024 | 0.0058 ms | 0.0127 ms | 2.21x |
| RMSNorm | 512 x 4096 | 0.0135 ms | 0.0235 ms | 1.74x |
| RMSNorm | 2048 x 4096 | 0.0427 ms | 0.0532 ms | 1.25x |
| SwiGLU | 128 x 1024 | 0.0061 ms | 0.0126 ms | 2.07x |
| SwiGLU | 512 x 4096 | 0.0184 ms | 0.0249 ms | 1.35x |
| SwiGLU | 2048 x 4096 | 0.0678 ms | 0.0728 ms | 1.07x |

The full environment record, correctness errors, p95/p99 tails, and timing
samples are in
[artifacts/rtx-5070-ti-rmsnorm-swiglu.json](artifacts/rtx-5070-ti-rmsnorm-swiglu.json).
These results are specific to this hardware and software stack.

The reported bandwidth remains a logical traffic model. Nsight Compute was not
available in this environment, so the repository does not claim measured DRAM
transactions, cache hit rates, occupancy, or tensor-core utilization.

## Regression Gate

```bash
triton-kernel-lab \
  --baseline artifacts/rtx-5070-ti-rmsnorm-swiglu.json \
  --max-regression-percent 10 \
  --output artifacts/candidate.json
```

Matching cases are identified by kernel, shape, and dtype, so older RMSNorm-only
baselines remain valid. Exit code `2` means a Triton p50 regression exceeded the
configured threshold.

## Report Shape

Each JSON report includes:

- GPU model, memory, compute capability, CUDA runtime, Python, PyTorch, and Triton.
- Git commit, seed, warmup count, timed iteration count, and timing method.
- Correctness tolerances plus maximum absolute and relative error.
- Triton, `torch.compile`, and PyTorch eager timing samples plus
  min/p50/p95/p99/max latency.
- Speedup against both PyTorch baselines and transparent logical
  effective-bandwidth estimates.

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for assumptions and limitations.

## Development

CPU-side validation does not install the GPU stack:

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python -m compileall -q src tests
```

## Scope

This is a focused kernel and measurement lab, not a claim of production-scale
GPU infrastructure. Planned extensions are fused gated activations,
hardware-counter capture with Nsight Compute, and shape-specific autotuning.
