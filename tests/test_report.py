from __future__ import annotations

import pytest

from triton_kernel_lab.report import (
    case_key,
    compare_with_baseline,
    percentile,
    summarize_timings,
)


def benchmark_case(p50_ms: float) -> dict:
    return {
        "kernel": "rmsnorm",
        "shape": [512, 4096],
        "dtype": "float16",
        "triton": {"p50_ms": p50_ms},
    }


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.50) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_percentile_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        percentile([], 0.50)
    with pytest.raises(ValueError):
        percentile([1.0], 1.01)


def test_summary_preserves_tail_metrics() -> None:
    summary = summarize_timings([0.1, 0.2, 0.3, 1.0])
    assert summary["samples"] == 4
    assert summary["min_ms"] == pytest.approx(0.1)
    assert summary["p50_ms"] == pytest.approx(0.25)
    assert summary["p99_ms"] > summary["p95_ms"]
    assert summary["max_ms"] == pytest.approx(1.0)


def test_case_key_is_stable() -> None:
    assert case_key(benchmark_case(0.1)) == "rmsnorm:512x4096:float16"


def test_regression_gate_reports_only_matching_regressions() -> None:
    regressions = compare_with_baseline(
        current_cases=[benchmark_case(1.12)],
        baseline_cases=[benchmark_case(1.0)],
        max_regression_percent=10.0,
    )
    assert len(regressions) == 1
    assert regressions[0]["regression_percent"] == pytest.approx(12.0)

    assert (
        compare_with_baseline(
            current_cases=[benchmark_case(1.05)],
            baseline_cases=[benchmark_case(1.0)],
            max_regression_percent=10.0,
        )
        == []
    )

