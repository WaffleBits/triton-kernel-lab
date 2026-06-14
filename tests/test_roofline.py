from __future__ import annotations

import pytest

from triton_kernel_lab.roofline import analyze_case


def test_roofline_classifies_low_intensity_case_as_memory_bound() -> None:
    result = analyze_case(
        {
            "kernel": "qk-dot",
            "shape": [512, 128],
            "dtype": "float16",
            "logical_bytes": 264_192,
            "logical_flops": 131_072,
            "triton": {"p50_ms": 0.006},
        },
        peak_bandwidth_gbps=896,
        peak_compute_tflops=43.9,
    )
    assert result["classification"] == "memory-bound"
    assert result["arithmetic_intensity_flops_per_byte"] == pytest.approx(
        131_072 / 264_192
    )


def test_copy_kernel_is_data_movement_only() -> None:
    result = analyze_case(
        {
            "kernel": "paged-gather",
            "shape": [128, 128],
            "dtype": "float16",
            "logical_bytes": 66_048,
            "logical_flops": 0,
            "triton": {"p50_ms": 0.005},
        },
        peak_bandwidth_gbps=896,
        peak_compute_tflops=43.9,
    )
    assert result["classification"] == "data-movement-only"

