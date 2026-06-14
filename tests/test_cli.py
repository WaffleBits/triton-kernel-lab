from __future__ import annotations

import argparse

import pytest

from triton_kernel_lab.benchmark import build_parser, parse_shape


def test_parse_shape() -> None:
    assert parse_shape("512x4096") == (512, 4096)
    assert parse_shape("8X1024") == (8, 1024)


@pytest.mark.parametrize("value", ["4096", "0x10", "4x-1", "axb"])
def test_parse_shape_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shape(value)


def test_parser_accepts_repeated_cases() -> None:
    args = build_parser().parse_args(
        [
            "--kernel",
            "rmsnorm",
            "--kernel",
            "swiglu",
            "--shape",
            "128x1024",
            "--shape",
            "512x4096",
            "--dtype",
            "float16",
            "--dtype",
            "bfloat16",
        ]
    )
    assert args.kernels == ["rmsnorm", "swiglu"]
    assert args.shapes == [(128, 1024), (512, 4096)]
    assert args.dtypes == ["float16", "bfloat16"]
