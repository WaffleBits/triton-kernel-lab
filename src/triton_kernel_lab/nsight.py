"""Capture one filtered custom kernel with NVIDIA Nsight Compute."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .benchmark import parse_shape

KERNEL_FILTERS = {
    "qk-dot": "_qk_dot_kernel",
    "paged-gather": "_paged_gather_kernel",
    "selective-attention": "_selective_attention_kernel",
    "residual-rmsnorm": "_residual_rmsnorm_kernel",
    "int4-gemv": "_int4_gemv_kernel",
}


def find_ncu() -> str:
    candidates = [
        shutil.which("ncu"),
        "/opt/nvidia/nsight-compute/2026.2.0/ncu",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("ncu was not found; install NVIDIA Nsight Compute in WSL2")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True, choices=tuple(KERNEL_FILTERS))
    parser.add_argument("--shape", required=True, type=parse_shape)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/nsight/decode-kernel"),
        help="Output base path without extension.",
    )
    args = parser.parse_args(argv)

    ncu = find_ncu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.output.with_suffix(".ncu-rep")
    shape = f"{args.shape[0]}x{args.shape[1]}"
    command = [
        ncu,
        "--target-processes",
        "all",
        "--kernel-name",
        f"regex:{KERNEL_FILTERS[args.kernel]}",
        "--launch-skip",
        str(args.warmup),
        "--launch-count",
        "1",
        "--section",
        "SpeedOfLight",
        "--section",
        "MemoryWorkloadAnalysis_Tables",
        "--section",
        "Occupancy",
        "--section",
        "ComputeWorkloadAnalysis",
        "--force-overwrite",
        "--export",
        str(report_path),
        sys.executable,
        "-m",
        "triton_kernel_lab.profile_target",
        "--kernel",
        args.kernel,
        "--shape",
        shape,
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        combined = f"{result.stdout}\n{result.stderr}"
        if "ERR_NVGPUCTRPERM" in combined:
            print(
                "error: NVIDIA performance counters are disabled on the Windows host. "
                "Enable Developer > Manage GPU Performance Counters > Allow access "
                "to all users in NVIDIA Control Panel, then rerun.",
                file=sys.stderr,
            )
            return 2
        print(combined, file=sys.stderr)
        return result.returncode

    csv_path = args.output.with_suffix(".csv")
    imported = subprocess.run(
        [ncu, "--import", str(report_path), "--csv", "--page", "raw"],
        text=True,
        capture_output=True,
        check=True,
    )
    csv_path.write_text(imported.stdout, encoding="utf-8")
    print(f"report: {report_path}")
    print(f"csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
