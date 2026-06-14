"""Build an assumption-labeled roofline report from benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def analyze_case(
    case: dict[str, Any],
    peak_bandwidth_gbps: float,
    peak_compute_tflops: float,
) -> dict[str, Any]:
    logical_bytes = int(case["logical_bytes"])
    logical_flops = int(case.get("logical_flops", 0))
    p50_ms = float(case["triton"]["p50_ms"])
    achieved_bandwidth = logical_bytes / (p50_ms / 1000) / 1e9

    result = {
        "kernel": case["kernel"],
        "shape": case["shape"],
        "dtype": case["dtype"],
        "p50_ms": p50_ms,
        "logical_bytes": logical_bytes,
        "logical_flops": logical_flops,
        "achieved_logical_bandwidth_gbps": achieved_bandwidth,
        "bandwidth_fraction_of_spec": achieved_bandwidth / peak_bandwidth_gbps,
    }
    if logical_flops <= 0:
        return {
            **result,
            "classification": "data-movement-only",
            "arithmetic_intensity_flops_per_byte": 0.0,
        }

    intensity = logical_flops / logical_bytes
    achieved_gflops = logical_flops / (p50_ms / 1000) / 1e9
    memory_ceiling_gflops = peak_bandwidth_gbps * intensity
    compute_ceiling_gflops = peak_compute_tflops * 1000
    roofline_ceiling = min(memory_ceiling_gflops, compute_ceiling_gflops)
    return {
        **result,
        "classification": (
            "memory-bound"
            if memory_ceiling_gflops < compute_ceiling_gflops
            else "compute-bound"
        ),
        "arithmetic_intensity_flops_per_byte": intensity,
        "achieved_gflops": achieved_gflops,
        "memory_ceiling_gflops": memory_ceiling_gflops,
        "compute_ceiling_gflops": compute_ceiling_gflops,
        "roofline_efficiency": achieved_gflops / roofline_ceiling,
    }


def build_report(
    artifacts: list[Path],
    peak_bandwidth_gbps: float,
    peak_compute_tflops: float,
) -> dict[str, Any]:
    if peak_bandwidth_gbps <= 0 or peak_compute_tflops <= 0:
        raise ValueError("peak bandwidth and compute must be positive")
    source_reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in artifacts
    ]
    cases = [
        analyze_case(case, peak_bandwidth_gbps, peak_compute_tflops)
        for report in source_reports
        for case in report["cases"]
    ]
    return {
        "schema_version": 1,
        "source_artifacts": [str(path) for path in artifacts],
        "assumptions": {
            "peak_memory_bandwidth_gbps": peak_bandwidth_gbps,
            "peak_compute_tflops": peak_compute_tflops,
            "traffic": "logical payload bytes, not measured DRAM transactions",
            "compute": "logical FLOPs, excluding address and conversion instructions",
        },
        "cases": cases,
        "claim_boundary": (
            "This is a specification-based roofline projection. Nsight Compute "
            "counters are required for physical memory-traffic and utilization claims."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--peak-bandwidth-gbps", type=float, default=896.0)
    parser.add_argument("--peak-compute-tflops", type=float, default=43.9)
    parser.add_argument("--output", type=Path, default=Path("artifacts/roofline.json"))
    args = parser.parse_args(argv)

    report = build_report(
        args.artifacts,
        args.peak_bandwidth_gbps,
        args.peak_compute_tflops,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(report['cases'])} roofline cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

