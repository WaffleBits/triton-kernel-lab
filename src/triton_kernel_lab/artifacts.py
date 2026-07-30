"""Validate committed benchmark and roofline artifacts without a GPU."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from triton_kernel_lab.report import case_key, summarize_timings
from triton_kernel_lab.roofline import build_report as build_roofline_report

_BACKENDS = ("triton", "torch_compile", "pytorch_eager")
_SUMMARY_FIELDS = ("samples", "min_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SENSITIVE_FIELDS = {
    "access_token",
    "api_key",
    "authorization",
    "environment",
    "output_text",
    "password",
    "private_key",
    "prompt",
    "refresh_token",
    "response_body",
    "secret",
}


class ArtifactValidationError(ValueError):
    """Raised when a published artifact is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class ArtifactValidationResult:
    """A compact success record suitable for CLI and CI output."""

    path: Path
    kind: str
    case_count: int


def _fail(path: Path, field: str, message: str) -> None:
    raise ArtifactValidationError(f"{path}: {field}: {message}")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactValidationError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        _fail(path, "$", "top-level value must be an object")
    return data


def _reject_sensitive_fields(value: Any, path: Path, field: str = "$") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = key.lower().replace("-", "_")
            nested_field = f"{field}.{key}"
            if normalized_key in _SENSITIVE_FIELDS:
                _fail(path, nested_field, "sensitive field is not allowed in a public artifact")
            _reject_sensitive_fields(nested, path, nested_field)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_sensitive_fields(nested, path, f"{field}[{index}]")


def _require_mapping(value: Any, path: Path, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, field, "must be an object")
    return value


def _require_nonempty_string(value: Any, path: Path, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, field, "must be a non-empty string")
    return value


def _require_positive_int(value: Any, path: Path, field: str, *, allow_zero: bool = False) -> int:
    lower_bound = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < lower_bound:
        qualifier = "non-negative" if allow_zero else "positive"
        _fail(path, field, f"must be a {qualifier} integer")
    return value


def _require_finite_number(
    value: Any,
    path: Path,
    field: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, field, "must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        _fail(path, field, "must be finite")
    if positive and numeric <= 0:
        _fail(path, field, "must be positive")
    if non_negative and numeric < 0:
        _fail(path, field, "must be non-negative")
    return numeric


def _assert_close(actual: Any, expected: float, path: Path, field: str) -> None:
    numeric = _require_finite_number(actual, path, field)
    if not math.isclose(numeric, expected, rel_tol=1e-9, abs_tol=1e-12):
        _fail(path, field, f"declared {numeric!r}, recomputed {expected!r}")


def _validate_correctness(value: Any, path: Path, field: str) -> None:
    evidence = _require_mapping(value, path, field)
    if evidence.get("passed") is not True:
        _fail(path, f"{field}.passed", "must be true before measurements are published")

    for key, nested in evidence.items():
        nested_field = f"{field}.{key}"
        if isinstance(nested, dict):
            _validate_correctness(nested, path, nested_field)
        elif key in {"atol", "rtol"}:
            _require_finite_number(nested, path, nested_field, positive=True)
        elif key in {"max_abs_error", "max_rel_error"}:
            _require_finite_number(nested, path, nested_field, non_negative=True)


def _validate_backend(
    backend: Any,
    path: Path,
    field: str,
    timed_iterations: int,
    logical_bytes: int | None,
) -> float:
    measurement = _require_mapping(backend, path, field)
    samples = measurement.get("samples_ms")
    if not isinstance(samples, list) or not samples:
        _fail(path, f"{field}.samples_ms", "must be a non-empty array")
    checked_samples = [
        _require_finite_number(sample, path, f"{field}.samples_ms[{index}]", positive=True)
        for index, sample in enumerate(samples)
    ]
    if len(checked_samples) != timed_iterations:
        _fail(
            path,
            f"{field}.samples_ms",
            f"contains {len(checked_samples)} values; methodology declares {timed_iterations}",
        )

    recomputed = summarize_timings(checked_samples)
    for summary_field in _SUMMARY_FIELDS:
        declared = measurement.get(summary_field)
        expected = recomputed[summary_field]
        if summary_field == "samples":
            if declared != expected:
                _fail(
                    path,
                    f"{field}.{summary_field}",
                    f"declared {declared!r}, counted {expected}",
                )
        else:
            _assert_close(declared, float(expected), path, f"{field}.{summary_field}")

    bandwidth = _require_finite_number(
        measurement.get("effective_bandwidth_gbps"),
        path,
        f"{field}.effective_bandwidth_gbps",
        positive=True,
    )
    if logical_bytes is not None:
        p50_ms = float(recomputed["p50_ms"])
        expected_bandwidth = logical_bytes / (p50_ms / 1000.0) / 1e9
        if not math.isclose(bandwidth, expected_bandwidth, rel_tol=1e-9, abs_tol=1e-12):
            _fail(
                path,
                f"{field}.effective_bandwidth_gbps",
                f"declared {bandwidth!r}, recomputed {expected_bandwidth!r}",
            )
    return float(recomputed["p50_ms"])


def _validate_benchmark(data: dict[str, Any], path: Path) -> ArtifactValidationResult:
    if data.get("schema_version") != 1:
        _fail(path, "schema_version", "only schema version 1 is supported")

    generated_at = _require_nonempty_string(data.get("generated_at"), path, "generated_at")
    try:
        parsed_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactValidationError(f"{path}: generated_at: invalid ISO-8601 timestamp") from exc
    if parsed_at.tzinfo is None:
        _fail(path, "generated_at", "must include a timezone")

    commit = _require_nonempty_string(data.get("git_commit"), path, "git_commit")
    if not _SHA1.fullmatch(commit):
        _fail(path, "git_commit", "must be a lowercase 40-character Git SHA-1")

    system = _require_mapping(data.get("system"), path, "system")
    for key in ("platform", "python", "gpu", "cuda_runtime", "torch", "triton"):
        _require_nonempty_string(system.get(key), path, f"system.{key}")
    capability = system.get("compute_capability")
    if not isinstance(capability, list) or len(capability) != 2:
        _fail(path, "system.compute_capability", "must contain major and minor versions")
    for index, value in enumerate(capability):
        _require_positive_int(
            value,
            path,
            f"system.compute_capability[{index}]",
            allow_zero=True,
        )
    _require_positive_int(system.get("gpu_memory_bytes"), path, "system.gpu_memory_bytes")

    methodology = _require_mapping(data.get("methodology"), path, "methodology")
    _require_positive_int(
        methodology.get("warmup_iterations"),
        path,
        "methodology.warmup_iterations",
        allow_zero=True,
    )
    timed_iterations = _require_positive_int(
        methodology.get("timed_iterations"), path, "methodology.timed_iterations"
    )
    _require_positive_int(
        methodology.get("cache_flush_mib"),
        path,
        "methodology.cache_flush_mib",
        allow_zero=True,
    )
    if methodology.get("cache_mode") not in {"cold", "hot"}:
        _fail(path, "methodology.cache_mode", "must be cold or hot")
    for key in ("timing", "oracle", "bandwidth"):
        _require_nonempty_string(methodology.get(key), path, f"methodology.{key}")

    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        _fail(path, "cases", "must be a non-empty array")
    identities: set[str] = set()
    for index, raw_case in enumerate(cases):
        field = f"cases[{index}]"
        case = _require_mapping(raw_case, path, field)
        _require_nonempty_string(case.get("kernel"), path, f"{field}.kernel")
        _require_nonempty_string(case.get("dtype"), path, f"{field}.dtype")
        shape = case.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            _fail(path, f"{field}.shape", "must contain two dimensions")
        for dimension, value in enumerate(shape):
            _require_positive_int(value, path, f"{field}.shape[{dimension}]")

        identity = case_key(case)
        if identity in identities:
            _fail(path, field, f"duplicate case identity {identity}")
        identities.add(identity)

        logical_bytes_value = case.get("logical_bytes")
        logical_bytes = None
        if logical_bytes_value is not None:
            logical_bytes = _require_positive_int(
                logical_bytes_value, path, f"{field}.logical_bytes"
            )
        if "logical_flops" in case:
            _require_positive_int(
                case["logical_flops"], path, f"{field}.logical_flops", allow_zero=True
            )

        _validate_correctness(case.get("correctness"), path, f"{field}.correctness")
        medians = {
            backend: _validate_backend(
                case.get(backend),
                path,
                f"{field}.{backend}",
                timed_iterations,
                logical_bytes,
            )
            for backend in _BACKENDS
        }
        triton_p50 = medians["triton"]
        _assert_close(
            case.get("speedup_vs_torch_compile"),
            medians["torch_compile"] / triton_p50,
            path,
            f"{field}.speedup_vs_torch_compile",
        )
        _assert_close(
            case.get("speedup_vs_pytorch_eager"),
            medians["pytorch_eager"] / triton_p50,
            path,
            f"{field}.speedup_vs_pytorch_eager",
        )

    return ArtifactValidationResult(path=path, kind="benchmark", case_count=len(cases))


def _compare_derived(actual: Any, expected: Any, path: Path, field: str) -> None:
    if isinstance(expected, dict):
        actual_mapping = _require_mapping(actual, path, field)
        if set(actual_mapping) != set(expected):
            _fail(
                path,
                field,
                f"keys differ: declared {sorted(actual_mapping)}, expected {sorted(expected)}",
            )
        for key, expected_value in expected.items():
            _compare_derived(actual_mapping[key], expected_value, path, f"{field}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            _fail(path, field, "array length differs from recomputed report")
        for index, expected_value in enumerate(expected):
            _compare_derived(actual[index], expected_value, path, f"{field}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        _assert_close(actual, float(expected), path, field)
        return
    if actual != expected:
        _fail(path, field, f"declared {actual!r}, recomputed {expected!r}")


def _resolve_source(source: Any, artifact_path: Path, repository_root: Path) -> Path:
    source_text = _require_nonempty_string(source, artifact_path, "source_artifacts[]")
    normalized = Path(source_text.replace("\\", "/"))
    candidate = (repository_root / normalized).resolve()
    try:
        candidate.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise ArtifactValidationError(
            f"{artifact_path}: source_artifacts: path escapes repository: {source_text}"
        ) from exc
    if not candidate.is_file():
        _fail(artifact_path, "source_artifacts", f"source does not exist: {source_text}")
    return candidate


def _validate_roofline(
    data: dict[str, Any], path: Path, repository_root: Path
) -> ArtifactValidationResult:
    if data.get("schema_version") != 1:
        _fail(path, "schema_version", "only schema version 1 is supported")
    sources = data.get("source_artifacts")
    if not isinstance(sources, list) or not sources:
        _fail(path, "source_artifacts", "must be a non-empty array")
    source_paths = [_resolve_source(source, path, repository_root) for source in sources]
    if len(set(source_paths)) != len(source_paths):
        _fail(path, "source_artifacts", "contains duplicate paths")
    for source_path in source_paths:
        source_data = _load_json(source_path)
        _reject_sensitive_fields(source_data, source_path)
        _validate_benchmark(source_data, source_path)

    assumptions = _require_mapping(data.get("assumptions"), path, "assumptions")
    peak_bandwidth = _require_finite_number(
        assumptions.get("peak_memory_bandwidth_gbps"),
        path,
        "assumptions.peak_memory_bandwidth_gbps",
        positive=True,
    )
    peak_compute = _require_finite_number(
        assumptions.get("peak_compute_tflops"),
        path,
        "assumptions.peak_compute_tflops",
        positive=True,
    )
    expected = build_roofline_report(source_paths, peak_bandwidth, peak_compute)
    _compare_derived(data.get("assumptions"), expected["assumptions"], path, "assumptions")
    _compare_derived(data.get("cases"), expected["cases"], path, "cases")
    _compare_derived(
        data.get("claim_boundary"), expected["claim_boundary"], path, "claim_boundary"
    )
    cases = data.get("cases")
    assert isinstance(cases, list)
    return ArtifactValidationResult(path=path, kind="roofline", case_count=len(cases))


def validate_artifact_path(
    path: Path | str, *, repository_root: Path | str | None = None
) -> ArtifactValidationResult:
    """Validate one benchmark or derived roofline artifact."""
    artifact_path = Path(path)
    root = Path(repository_root) if repository_root is not None else artifact_path.parent.parent
    data = _load_json(artifact_path)
    _reject_sensitive_fields(data, artifact_path)
    if "system" in data and "methodology" in data:
        return _validate_benchmark(data, artifact_path)
    if "source_artifacts" in data and "assumptions" in data:
        return _validate_roofline(data, artifact_path, root)
    _fail(artifact_path, "$", "unrecognized artifact kind")
    raise AssertionError("unreachable")


def _expand_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.json")))
        else:
            expanded.append(path)
    return expanded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("artifacts")])
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    paths = _expand_paths(args.paths)
    if not paths:
        print("error: no JSON artifacts found", file=sys.stderr)
        return 1

    failed = False
    for path in paths:
        try:
            result = validate_artifact_path(path, repository_root=args.repository_root)
        except ArtifactValidationError as exc:
            failed = True
            print(f"error: {exc}", file=sys.stderr)
        else:
            print(f"ok: {result.kind:<9} {result.case_count:>2} cases  {result.path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
