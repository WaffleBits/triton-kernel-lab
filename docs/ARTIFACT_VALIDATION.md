# Committed artifact validation plan

## Market-backed gap

Inference performance roles repeatedly ask for correctness gates, reproducible measurements, root-causeable telemetry, and explicit model boundaries. This repository publishes raw benchmark and derived roofline JSON, but CI currently checks only Python contracts. It does not prove that committed summaries still match their raw samples or that the roofline report can be reproduced from its named sources. The latest public workflow is also red because the chart generator does not pass the repository lint gate.

## Selected change

Add one CPU-only artifact validation path to the existing kernel lab:

1. Parse every committed JSON artifact without accepting non-finite values or
   fields that could carry prompts, credentials, private keys, or response bodies.
2. Validate benchmark identity, environment, methodology, correctness, unique case keys, raw sample counts, recomputed min/p50/p95/p99/max summaries, bandwidth values, and speedup ratios.
3. Rebuild the committed roofline cases from the named source artifacts and compare the derived values and claim boundary.
4. Expose the validator as `triton-kernel-artifacts` and run it in CI.
5. Repair the existing chart-script lint failure without changing the chart or benchmark values.

## Verification boundary

The validator checks internal integrity and reproducibility of already committed RTX 5070 Ti artifacts on a CPU runner. It does not rerun GPU kernels, create new performance measurements, or convert logical-bandwidth projections into hardware-counter evidence.
