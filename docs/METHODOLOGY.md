# Benchmark Methodology

The benchmark treats correctness and reproducibility as release criteria, not as
notes around a headline speedup.

## Kernel

The first workload is RMSNorm, an inference-relevant reduction and elementwise
operation. One Triton program handles one row:

1. Load the row and accumulate squared values in FP32.
2. Compute reciprocal RMS with the configured epsilon.
3. Load the learned weight vector.
4. Normalize, scale, and write the output in one fused kernel.

The PyTorch oracle also accumulates in FP32 before converting back to the input
dtype. Every benchmark case validates both the custom kernel and a
`torch.compile` version against that oracle before timing.

## Timing

- Warmup iterations compile the Triton kernel and stabilize allocator/runtime state.
- CUDA events surround each queued operation.
- The benchmark synchronizes after the timed batch, then reports min, p50, p95,
  p99, and max instead of relying on a single best result.
- PyTorch eager, `torch.compile`, and Triton operate on the same preallocated
  tensors.
- The report records GPU model, compute capability, CUDA runtime, PyTorch,
  Triton, Python, seed, shapes, dtypes, and the current git commit.
- The JSON preserves the timing samples as well as min, p50, p95, p99, and max.

## Effective Bandwidth

The reported bandwidth is a transparent logical model:

`input read + weight read + output write`

It does not claim to be a hardware-counter measurement. Weight caching, compiler
behavior, and memory transactions can make physical traffic differ from the
logical byte count. Nsight Compute should be used when hardware-counter evidence
is required.

## Regression Gate

Passing `--baseline previous.json --max-regression-percent 10` compares matching
kernel, shape, and dtype cases by Triton p50. The process exits with status `2`
when a matching case exceeds the allowed regression.

Hardware, driver, thermal, clock, and background-load changes can move results.
Keep baselines specific to a controlled machine or runner.
