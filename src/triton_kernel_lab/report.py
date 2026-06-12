"""Pure-Python benchmark statistics and regression checks."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def percentile(values: Iterable[float], quantile: float) -> float:
    """Return an interpolated percentile for a non-empty sample."""
    samples = sorted(float(value) for value in values)
    if not samples:
        raise ValueError("percentile requires at least one sample")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")

    position = (len(samples) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return samples[lower]

    weight = position - lower
    return samples[lower] * (1.0 - weight) + samples[upper] * weight


def summarize_timings(samples_ms: Iterable[float]) -> dict[str, float | int]:
    """Summarize raw GPU timings without hiding the tail."""
    samples = [float(value) for value in samples_ms]
    if not samples:
        raise ValueError("timing summary requires at least one sample")
    if any(value <= 0.0 for value in samples):
        raise ValueError("timings must be positive")

    return {
        "samples": len(samples),
        "min_ms": min(samples),
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "p99_ms": percentile(samples, 0.99),
        "max_ms": max(samples),
    }


def case_key(case: dict[str, Any]) -> str:
    """Build a stable identity for one benchmark case."""
    rows, hidden = case["shape"]
    return f"{case['kernel']}:{rows}x{hidden}:{case['dtype']}"


def compare_with_baseline(
    current_cases: list[dict[str, Any]],
    baseline_cases: list[dict[str, Any]],
    max_regression_percent: float,
) -> list[dict[str, float | str]]:
    """Return cases whose Triton median exceeds the allowed regression."""
    if max_regression_percent < 0.0:
        raise ValueError("max_regression_percent must be non-negative")

    baseline_by_key = {case_key(case): case for case in baseline_cases}
    regressions: list[dict[str, float | str]] = []

    for current in current_cases:
        key = case_key(current)
        baseline = baseline_by_key.get(key)
        if baseline is None:
            continue

        current_ms = float(current["triton"]["p50_ms"])
        baseline_ms = float(baseline["triton"]["p50_ms"])
        regression_percent = ((current_ms / baseline_ms) - 1.0) * 100.0
        if regression_percent > max_regression_percent:
            regressions.append(
                {
                    "case": key,
                    "baseline_p50_ms": baseline_ms,
                    "current_p50_ms": current_ms,
                    "regression_percent": regression_percent,
                }
            )

    return regressions

