# Results and tooling

## What the lab covers

- Triton GPU kernel development with FP32 reduction, fused normalization, and
  fused gated activation.
- Decode-oriented kernels for attention scoring, non-contiguous KV movement,
  residual normalization, and packed signed INT4 weight-only projection.
- Shape-aware SwiGLU launch autotuning across block sizes and warp counts.
- Correctness validation across shapes and low-precision dtypes before timing,
  including validation of the `torch.compile` baseline against the FP32 oracle.
- GPU benchmarking with warmup, CUDA events, p50/p95/p99/max latency, explicit
  cache state, and reproducible environment metadata.
- CPU-only validation that recomputes every committed timing summary and derived
  roofline case from the raw JSON evidence.

## Measured run

Cache-cold run on June 14, 2026 using an RTX 5070 Ti, CUDA 13.0, PyTorch 2.12,
and Triton 3.7. Each case used 100 warmups, 500 timed samples, and a 256 MiB
cache-eviction buffer outside the timed region. The README table shows FP16; the
artifact also includes BF16.

## Decode kernels

Serial cache-cold measurements on the same RTX 5070 Ti stack:

| Kernel | Shape | Dtype | Triton p50 | `torch.compile` p50 | Speedup | Logical BW |
|---|---:|---:|---:|---:|---:|---:|
| QK dot | 512 x 128 | FP16 | 0.0055 ms | 0.0127 ms | 2.32x | 48.4 GB/s |
| QK dot | 2,048 x 128 | FP16 | 0.0078 ms | 0.0141 ms | 1.81x | 135.9 GB/s |
| Paged KV gather | 512 x 128 | FP16 | 0.0057 ms | 0.0163 ms | 2.85x | 46.4 GB/s |
| Selected attention | 64 x 128 | BF16 | 0.0064 ms | 0.0185 ms | 2.90x | 5.3 GB/s |
| Selected attention | 256 x 128 | BF16 | 0.0115 ms | 0.0185 ms | 1.60x | 11.5 GB/s |
| Residual RMSNorm | 4 x 1,536 | BF16 | 0.0051 ms | 0.0121 ms | 2.37x | 10.3 GB/s |
| INT4 GEMV | 1,536 x 1,536 | BF16 | 0.0077 ms | 0.0208 ms | 2.70x | 154.6 GB/s |
| INT4 GEMV | 8,960 x 1,536 | BF16 | 0.0205 ms | 0.0449 ms | 2.19x | 339.4 GB/s |

The checked roofline report uses the RTX 5070 Ti specification of 896 GB/s and
labels all ten decode cases memory-bound or data-movement-only. Its best
logical bandwidth fraction is 37.9% for the 8,960 x 1,536 INT4 GEMV. This is a
specification-based projection, not a physical-traffic measurement.

Nsight Compute 2026.2 is installed and the repository includes a filtered
capture command. The current host returns `ERR_NVGPUCTRPERM` because Windows
performance-counter access is disabled. No measured DRAM transactions, cache
hit rates, occupancy, or execution-pipeline utilization are claimed.

```bash
triton-kernel-nsight \
  --kernel int4-gemv \
  --shape 8960x1536 \
  --dtype bfloat16 \
  --output artifacts/nsight/int4-gemv
```

## Regression gate

```bash
triton-kernel-lab \
  --baseline artifacts/rtx-5070-ti-rmsnorm-swiglu.json \
  --max-regression-percent 10 \
  --output artifacts/candidate.json
```

Matching cases are identified by kernel, shape, and dtype, so older RMSNorm-only
baselines remain valid. Exit code `2` means a Triton p50 regression exceeded the
configured threshold.

## Report shape

Each JSON report includes:

- GPU model, memory, compute capability, CUDA runtime, Python, PyTorch, and Triton.
- Git commit, seed, warmup count, timed iteration count, and timing method.
- Correctness tolerances plus maximum absolute and relative error.
- Triton, `torch.compile`, and PyTorch eager timing samples plus
  min/p50/p95/p99/max latency.
- Speedup against both PyTorch baselines and transparent logical
  effective-bandwidth estimates.

Committed artifacts have an additional CPU-only integrity gate. It rejects
non-finite JSON, sensitive fields, failed correctness records, duplicate cases,
mismatched raw sample counts, stale percentile/speedup/bandwidth summaries, and
roofline cases that no longer reproduce from their named source artifacts. It
validates the published records; it does not rerun a GPU measurement.

## Development

CPU-side validation does not install the GPU stack:

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
triton-kernel-artifacts
python -m compileall -q src tests
```
