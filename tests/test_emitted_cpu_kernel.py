from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx2.math.lattice import E8LatticeLinear
from ojasx2.tile_fallback import _emit_universal_source_fallback, emit_gemm_kernel_source
from ojasx2.tilelang_kernels import (
    TileLangHardwareAutoTuner,
    compile_or_emit_universal_kernel,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("g++") is None, reason="C++20 compiler unavailable for emitted-source qualification")
def test_generated_cpu_source_executes_gemm_relu2_and_global_mask(tmp_path: Path) -> None:
    source = _emit_universal_source_fallback("cpu", 4, 40, 7, 4)
    source_path = tmp_path / "generated_cpu.cpp"
    library_path = tmp_path / "generated_cpu.so"
    source_path.write_text(source)
    result = subprocess.run(
        [
            "g++", "-std=c++20", "-O2", "-shared", "-fPIC", "-Wall", "-Wextra",
            f"-I{ROOT / 'third_party' / 'tilelang' / 'src'}",
            str(source_path), "-o", str(library_path),
        ],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    library = ctypes.CDLL(str(library_path))
    function = library.ojasx_tilelang_cpu_gemm_relu2_ballot_kernel
    float_pointer = ctypes.POINTER(ctypes.c_float)
    int_pointer = ctypes.POINTER(ctypes.c_int32)
    function.argtypes = [
        float_pointer, float_pointer, float_pointer, float_pointer, int_pointer,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_bool,
    ]
    function.restype = None
    rng = np.random.default_rng(90210)

    for rows, columns, reduction, use_bias in ((5, 75, 5, True), (2, 33, 0, False)):
        a = np.ascontiguousarray(rng.standard_normal((rows, reduction)), dtype=np.float32)
        b = np.ascontiguousarray(rng.standard_normal((columns, reduction)), dtype=np.float32)
        bias = np.ascontiguousarray(rng.standard_normal(columns), dtype=np.float32)
        output = np.full((rows, columns), -777.0, dtype=np.float32)
        words = (columns + 31) // 32
        packed = np.full((rows, words), -1, dtype=np.int32)
        function(
            a.ctypes.data_as(float_pointer),
            b.ctypes.data_as(float_pointer),
            bias.ctypes.data_as(float_pointer),
            output.ctypes.data_as(float_pointer),
            packed.ctypes.data_as(int_pointer),
            rows,
            columns,
            reduction,
            use_bias,
        )

        pre_activation = a @ b.T
        if use_bias:
            pre_activation = pre_activation + bias
        expected_output = np.where(pre_activation > 0, np.square(pre_activation), 0.0)
        np.testing.assert_allclose(output, expected_output, rtol=2e-6, atol=2e-6)

        expected_packed = np.zeros((rows, words), dtype=np.uint32)
        for row, column in np.argwhere(pre_activation > 0):
            word, bit = divmod(int(column), 32)
            expected_packed[row, word] |= np.uint32(1) << np.uint32(bit)
        np.testing.assert_array_equal(packed.view(np.uint32), expected_packed)


@pytest.mark.parametrize("target", ["cuda", "rocm", "sycl"])
def test_accelerator_source_fallback_contains_complete_bounded_epilogue(target: str) -> None:
    source = _emit_universal_source_fallback(target, 32, 32, 16, 4)
    assert "for (int k = 0; k < K; ++k)" in source
    assert "Out[" in source and "PackedMask[" in source
    assert "mask_word |= (uint32_t(1) << bit)" in source
    assert "populated by the target GEMM epilogue" not in source


def test_metal_source_fallback_computes_all_k_and_packs_ballot() -> None:
    source = _emit_universal_source_fallback("metal", 32, 32, 8, 4)
    assert "for (uint k = 0u; k < params.K; ++k)" in source
    assert "simd_ballot(is_pos)" in source
    assert "Out[gm * params.N + gn]" in source
    assert "PackedMask[gm * words_per_row + word_col]" in source
    assert "simdgroup_multiply_accumulate(acc[0], acc[0], acc[0], acc[0])" not in source


@pytest.mark.parametrize(
    "arguments",
    [
        ("cpu", 0, 32, 16, 4),
        ("cuda", 32, -1, 16, 4),
        ("rocm", 32, 32, 0, 4),
        ("sycl", 32, 32, 16, 3),
        ("ascend", 24, 32, 16, 4),
    ],
)
def test_emitters_reject_invalid_tile_contracts(arguments: tuple) -> None:
    with pytest.raises(RuntimeError):
        _emit_universal_source_fallback(*arguments)


@pytest.mark.parametrize(
    "device,M,N,K,dtype,expected_tile,expected_arch",
    [
        ("metal", 8, 128, 128, torch.float16, (8, 32, 16), "metal"),
        ("mps", 64, 128, 128, torch.float16, (32, 32, 8), "metal"),
        ("cuda", 32, 128, 128, torch.float16, (64, 64, 32), "cuda"),
        ("cuda", 256, 256, 256, torch.float16, (128, 128, 32), "cuda"),
        ("rocm", 32, 128, 128, torch.float16, (64, 64, 32), "rocm"),
        ("rocm", 256, 256, 256, torch.bfloat16, (128, 128, 32), "rocm"),
        ("sycl", 16, 64, 64, torch.float16, (32, 32, 16), "sycl"),
        ("xpu", 128, 128, 128, torch.float16, (64, 64, 16), "sycl"),
        ("ascend", 16, 64, 64, torch.float16, (32, 32, 16), "ascend"),
        ("ascend", 128, 128, 128, torch.float16, (64, 64, 32), "ascend"),
        ("cpu", 64, 128, 128, torch.float32, (32, 32, 16), "cpu"),
    ],
)
def test_hardware_autotuner_all_architectures(
    device: str,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    expected_tile: tuple,
    expected_arch: str,
) -> None:
    block_m, block_n, block_k, panel_size, target_arch = (
        TileLangHardwareAutoTuner.select_tile_config(M, N, K, device=device, dtype=dtype)
    )
    assert (block_m, block_n, block_k) == expected_tile
    assert target_arch == expected_arch
    assert panel_size in (2, 4, 8)
    emitted = emit_gemm_kernel_source(target_arch, block_m, block_n, block_k, panel_size)
    assert len(emitted) > 64


@pytest.mark.parametrize("arch", ["metal", "cuda", "rocm", "sycl", "ascend", "cpu"])
def test_compile_or_emit_universal_kernel_all_architectures_and_parity(arch: str) -> None:
    torch.manual_seed(7)
    M, N, K = 32, 64, 64
    kernel = compile_or_emit_universal_kernel(
        M, N, K, device=arch, bitrate=2.0, group_size=32, dtype=torch.float32
    )
    assert kernel.target_arch == arch
    assert isinstance(kernel.source, str) and len(kernel.source) > 64

    # Verify numerical parity for dense linear and ReLU2+ballot execution
    x = torch.randn(M, K, dtype=torch.float32)
    w = torch.randn(N, K, dtype=torch.float32)
    b = torch.randn(N, dtype=torch.float32)

    out_dense = kernel(x, w, b)
    expected_dense = F.linear(x, w, b)
    torch.testing.assert_close(out_dense, expected_dense, rtol=1e-5, atol=1e-5)

    out_relu2, mask = kernel(x, w, b, relu2_ballot=True)
    expected_relu2 = F.relu(expected_dense).square()
    torch.testing.assert_close(out_relu2, expected_relu2, rtol=1e-5, atol=1e-5)
    assert mask.shape == (M, (N + 31) // 32)

    # Verify numerical parity for packed E8 2.0-bit weights
    ref_linear = nn.Linear(K, N, bias=True)
    e8_layer = E8LatticeLinear.from_linear(ref_linear, rank=0, bitrate=2.0, group_size=32)
    out_e8 = kernel(x, e8_layer.packed_weight, e8_layer.block_scales, e8_layer.bias)
    expected_e8 = F.linear(x, e8_layer.unpack_weights(dtype=x.dtype), e8_layer.bias)
    torch.testing.assert_close(out_e8, expected_e8, rtol=1e-4, atol=1e-4)

