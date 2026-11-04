"""BackendBench data parsing must never execute dataset-provided Python."""

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from ojasx2.tensor_specs import parse_tensor_spec_args


@dataclass
class Spec:
    shape: object
    dtype: object = torch.float32
    stride: object = None


def test_safe_parser_supports_the_backendbench_grammar():
    result = parse_tensor_spec_args(
        "((T([2, 3], f16), T([3, 4], torch.float32, stride=[4, 1]), -2, True, None), {'layout': torch.strided})",
        Spec,
    )
    args, kwargs = result
    assert args[0].shape == [2, 3] and args[0].dtype == torch.float16
    assert args[1].dtype == torch.float32 and args[1].stride == [4, 1]
    assert args[2:] == (-2, True, None)
    assert kwargs == {"layout": torch.strided}


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('touch /tmp/ojasx-must-not-exist')",
        "open('/tmp/ojasx-must-not-exist', 'w')",
        "torch.__dict__",
        "T([1], f16).__class__",
        "(lambda: 1)()",
        "[x for x in [1, 2]]",
        "T([1], f16, malicious=True)",
        "1 + 2",
    ],
)
def test_safe_parser_rejects_executable_or_unsupported_syntax(source, tmp_path):
    marker = tmp_path / "must-not-exist"
    source = source.replace("/tmp/ojasx-must-not-exist", str(marker))
    with pytest.raises((ValueError, SyntaxError)):
        parse_tensor_spec_args(source, Spec)
    assert not marker.exists()


def test_every_checked_in_backendbench_expression_matches_the_safe_grammar():
    pd = pytest.importorskip("pandas")
    parquet = Path(__file__).resolve().parents[1] / "third_party" / "backend_bench_problems.parquet"
    if not parquet.exists():
        pytest.skip("BackendBench parquet is not included in this source distribution")
    frame = pd.read_parquet(parquet, columns=["args"])
    for row_number, source in enumerate(frame["args"]):
        try:
            parse_tensor_spec_args(source, Spec)
        except Exception as exc:  # pragma: no cover - assertion adds the exact incompatible row
            raise AssertionError(f"Safe parser rejected BackendBench row {row_number}: {source}") from exc
    assert len(frame) == 19_171
