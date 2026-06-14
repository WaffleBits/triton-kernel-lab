# Benchmark Methodology

The benchmark treats correctness and reproducibility as release criteria, not as
notes around a headline speedup.

## Kernels

RMSNorm is an inference-relevant reduction and elementwise operation. One
Triton program handles one row:

1. Load the row and accumulate squared values in FP32.
2. Compute reciprocal RMS with the configured epsilon.
3. Load the learned weight vector.
4. Normalize, scale, and write the output in one fused kernel.

The PyTorch oracle also accumulates in FP32 before converting back to the input
dtype. Every benchmark case validates both the custom kernel and a
`torch.compile` version against that oracle before timing.

SwiGLU is an inference-relevant gated activation. The Triton implementation
loads the gate and up-projection tensors, computes SiLU in FP32, multiplies by
the up projection, and writes the result in one kernel. Triton autotuning
compares three launch configurations per tensor size:

- 256 elements per program with 4 warps.
- 512 elements per program with 4 warps.
- 1024 elements per program with 8 warps.

Autotuning occurs before timed samples. The PyTorch oracle computes the same
expression in FP32 before converting back to the input dtype.

The decode suite adds:

- QK dot-product with one program per KV position and FP32 accumulation.
- Paged KV gather with explicit logical-to-physical page mapping.
- Fused selected-row attention that loads indirect K/V rows, computes QK scores
  and softmax in FP32, and accumulates the selected values in one Triton program.
- Fused residual addition and RMSNorm with both the combined residual and
  normalized output returned.
- Signed INT4 GEMV with two weights packed per byte, unpacked in-kernel, and
  accumulated in FP32.

Every implementation is checked against a PyTorch oracle before timing.

## Timing

- Warmup iterations compile the Triton kernel and stabilize allocator/runtime state.
- CUDA events surround each queued operation.
- The benchmark synchronizes after the timed batch, then reports min, p50, p95,
  p99, and max instead of relying on a single best result.
- Cache-cold mode is the default. A 256 MiB eviction buffer is touched on the
  same CUDA stream before each start event, so eviction is ordered before but
  excluded from the timed region. `--cache-mode hot` explicitly selects a
  resident-working-set measurement.
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
logical byte count.

`triton-kernel-roofline` combines checked benchmark artifacts with explicit
peak-bandwidth and peak-compute assumptions. It reports arithmetic intensity,
the specification-derived ceiling, and logical efficiency. Cases with no
logical arithmetic, such as paged gather, are classified as data movement only.

Nsight Compute 2026.2 is installed at
`/opt/nvidia/nsight-compute/2026.2.0/ncu` in WSL2. The
`triton-kernel-nsight` command filters one custom Triton kernel and requests
SpeedOfLight, memory-workload, occupancy, and compute-workload sections. The
current Windows host denies performance-counter access with
`ERR_NVGPUCTRPERM`; no counter-derived evidence is checked in until the user
enables that host security setting.

## Regression Gate

Passing `--baseline previous.json --max-regression-percent 10` compares matching
kernel, shape, and dtype cases by Triton p50. The process exits with status `2`
when a matching case exceeds the allowed regression.

Hardware, driver, thermal, clock, and background-load changes can move results.
Keep baselines specific to a controlled machine or runner.
