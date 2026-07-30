from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from triton_kernel_lab.artifacts import (
    ArtifactValidationError,
    validate_artifact_path,
)
from triton_kernel_lab.report import summarize_timings


def _benchmark_report() -> dict:
    samples_by_backend = {
        "triton": [0.10, 0.20, 0.30, 0.40],
        "torch_compile": [0.20, 0.30, 0.40, 0.50],
        "pytorch_eager": [0.30, 0.40, 0.50, 0.60],
    }
    case = {
        "kernel": "rmsnorm",
        "shape": [2, 16],
        "dtype": "float16",
        "logical_bytes": 160,
        "correctness": {
            "passed": True,
            "atol": 0.02,
            "rtol": 0.02,
            "max_abs_error": 0.001,
            "max_rel_error": 0.002,
        },
    }
    for backend, samples in samples_by_backend.items():
        summary = summarize_timings(samples)
        case[backend] = {
            **summary,
            "samples_ms": samples,
            "effective_bandwidth_gbps": 160 / (summary["p50_ms"] / 1000) / 1e9,
        }
    case["speedup_vs_torch_compile"] = (
        case["torch_compile"]["p50_ms"] / case["triton"]["p50_ms"]
    )
    case["speedup_vs_pytorch_eager"] = (
        case["pytorch_eager"]["p50_ms"] / case["triton"]["p50_ms"]
    )
    return {
        "schema_version": 1,
        "generated_at": "2026-07-30T10:00:00+00:00",
        "git_commit": "a" * 40,
        "system": {
            "platform": "Linux-test",
            "python": "3.12.0",
            "gpu": "fixture GPU",
            "compute_capability": [12, 0],
            "gpu_memory_bytes": 1024,
            "cuda_runtime": "13.0",
            "torch": "2.12.0",
            "triton": "3.7.0",
        },
        "methodology": {
            "warmup_iterations": 2,
            "timed_iterations": 4,
            "seed": 17,
            "timing": "fixture",
            "cache_mode": "cold",
            "cache_flush_mib": 1,
            "oracle": "fixture",
            "bandwidth": "logical bytes divided by p50",
            "autotuning": "fixture",
        },
        "cases": [case],
    }


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_accepts_self_consistent_benchmark_report(tmp_path: Path) -> None:
    result = validate_artifact_path(_write(tmp_path / "benchmark.json", _benchmark_report()))
    assert result.kind == "benchmark"
    assert result.case_count == 1


def test_rejects_tampered_summary(tmp_path: Path) -> None:
    report = _benchmark_report()
    report["cases"][0]["triton"]["p50_ms"] = 99.0

    with pytest.raises(ArtifactValidationError, match="triton.p50_ms"):
        validate_artifact_path(_write(tmp_path / "tampered.json", report))


def test_rejects_failed_correctness_evidence(tmp_path: Path) -> None:
    report = _benchmark_report()
    report["cases"][0]["correctness"]["passed"] = False

    with pytest.raises(ArtifactValidationError, match="correctness.passed"):
        validate_artifact_path(_write(tmp_path / "failed.json", report))


def test_rejects_non_finite_json_number(tmp_path: Path) -> None:
    report = _benchmark_report()
    report["cases"][0]["triton"]["samples_ms"][0] = float("nan")

    with pytest.raises(ArtifactValidationError, match="non-finite"):
        validate_artifact_path(_write(tmp_path / "nan.json", report))


def test_rejects_duplicate_case_identity(tmp_path: Path) -> None:
    report = _benchmark_report()
    report["cases"].append(copy.deepcopy(report["cases"][0]))

    with pytest.raises(ArtifactValidationError, match="duplicate case"):
        validate_artifact_path(_write(tmp_path / "duplicate.json", report))


def test_rejects_sensitive_fields_from_shareable_artifacts(tmp_path: Path) -> None:
    report = _benchmark_report()
    report["system"]["api_key"] = "must-not-be-published"

    with pytest.raises(ArtifactValidationError, match="sensitive field"):
        validate_artifact_path(_write(tmp_path / "secret.json", report))


def test_all_committed_artifacts_are_internally_consistent() -> None:
    root = Path(__file__).parents[1]
    paths = sorted((root / "artifacts").glob("*.json"))
    results = [validate_artifact_path(path, repository_root=root) for path in paths]

    assert len(results) == 8
    assert {result.kind for result in results} == {"benchmark", "roofline"}
    assert sum(result.case_count for result in results) == 38
