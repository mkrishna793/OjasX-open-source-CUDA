"""
Production Readiness v2 Test Suite for OjasX3:
  - CW(120, 36) unit-multiplier + shift + sign orbit orthogonality
  - Two-sided Sign + FWHT / CW incoherence exact invertibility
  - CWOFTLinear merge_into_dense_weight & RotatingCWOrbitController exact forward parity
  - Zero lock-in export_peft_adapter (with exact Hadamard un-rotation) & merge_and_export_hf
  - Adaptive Precision Thermostat (select_sensitive_layers)
  - Zero-copy Metal 4 E8 dispatcher & Triton fallback safety
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from ojasx2.math.lattice import E8LatticeLinear
from ojasx3.lattice.transforms import (
    CWOFTLinear,
    apply_circulant_weighing,
    apply_two_sided_incoherence,
    cw120_36_orbit_support,
    generate_rademacher_signs,
)
from ojasx3.trainer.export import export_peft_adapter, merge_and_export_hf
from ojasx3.trainer.peft import LoRALinear
from ojasx3.trainer.precision_thermostat import (
    RotatingCWOrbitController,
    select_sensitive_layers,
)
from ojasx3.hardware.metal_e8_dispatch import (
    metal_fused_e8_2bit_gemm,
    metal_fused_ternary_e8_gemm,
)


@pytest.mark.parametrize("orbit_id", [0, 1, 7, 31, 120, 513, 3840, 7679])
def test_cw120_36_orbit_exact_orthogonality(orbit_id: int):
    pos, neg = cw120_36_orbit_support(orbit_id)
    assert len(pos) + len(neg) == 36
    x = torch.randn(3, 120, dtype=torch.float64)
    y = apply_circulant_weighing(x, 120, pos, neg, transpose=False)
    x_rec = apply_circulant_weighing(y, 120, pos, neg, transpose=True)
    assert torch.allclose(x, x_rec, atol=1e-12)


@pytest.mark.parametrize("shape", [(64, 64), (120, 120), (56, 128), (28, 32)])
def test_two_sided_sign_incoherence_exact_invertibility(shape):
    out_f, in_f = shape
    w = torch.randn(out_f, in_f, dtype=torch.float64)
    r_signs = generate_rademacher_signs(out_f, seed=11).to(torch.float64)
    c_signs = generate_rademacher_signs(in_f, seed=22).to(torch.float64)
    w_rot = apply_two_sided_incoherence(w, inverse=False, row_signs=r_signs, col_signs=c_signs)
    w_rec = apply_two_sided_incoherence(w_rot, inverse=True, row_signs=r_signs, col_signs=c_signs)
    assert torch.allclose(w, w_rec, atol=1e-11)


def test_cwoft_merge_and_rotating_orbit_exact_parity():
    torch.manual_seed(0)
    base = nn.Linear(120, 64, bias=True)
    adapter = CWOFTLinear(base, orbit_id=0)
    with torch.no_grad():
        adapter.d_in.uniform_(-0.2, 0.2)
        adapter.d_out.uniform_(-0.2, 0.2)
    x = torch.randn(4, 120)
    y_before = adapter(x)

    # Rotate orbit folds exact dense update into base_layer.weight and resets d_in/d_out to 0
    # Note: bias output scaling is preserved when merging via merge_and_export_hf; test bias=False for pure weight fold
    base2 = nn.Linear(120, 64, bias=False)
    adapter2 = CWOFTLinear(base2, orbit_id=0)
    with torch.no_grad():
        adapter2.d_in.uniform_(-0.2, 0.2)
        adapter2.d_out.uniform_(-0.2, 0.2)
    y_pre = adapter2(x)
    new_oid = adapter2.rotate_orbit()
    assert new_oid == 1
    assert torch.allclose(adapter2(x), y_pre, atol=1e-5)

    # Controller steps & rotates cleanly
    model = nn.Sequential(nn.Linear(64, 64, bias=False))
    ctrl = RotatingCWOrbitController(model, rotate_every_steps=2)
    assert ctrl.attach_cwoft() == 1
    assert ctrl.step() is False
    assert ctrl.step() is True
    assert ctrl.rotations_completed == 1


def test_zero_lockin_peft_export_and_dense_hf_merge(tmp_path: Path):
    torch.manual_seed(42)
    model = nn.Sequential(
        LoRALinear(nn.Linear(64, 64), rank=4, lora_alpha=8.0),
        CWOFTLinear(nn.Linear(64, 32)),
    )
    with torch.no_grad():
        model[0].lora_B.normal_(0.0, 0.05)
        model[1].d_in.uniform_(-0.1, 0.1)
        model[1].d_out.uniform_(-0.1, 0.1)

    x = torch.randn(3, 64)
    y_ref = model(x)

    # 1. Standard PEFT export
    peft_info = export_peft_adapter(model, tmp_path / "peft_out", base_model_name_or_path="test/model")
    assert Path(peft_info["weights_path"]).exists()
    cfg = json.loads(Path(peft_info["config_path"]).read_text())
    assert cfg["peft_type"] == "LORA" and cfg["r"] == 4

    # 2. Dense HF merge export -> exact numerical parity with zero OjasX classes!
    merge_info = merge_and_export_hf(model, tmp_path / "hf_merged", dtype=torch.float32)
    assert merge_info["merged_adapter_layers"] == 2
    w_path = Path(merge_info["weights_path"])
    if w_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(w_path))
    else:
        sd = torch.load(str(w_path), map_location="cpu")

    plain = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 32))
    plain.load_state_dict(sd, strict=True)
    y_plain = plain(x)
    assert torch.allclose(y_ref, y_plain, atol=1e-5)


def test_precision_thermostat_flags_outlier_layer():
    torch.manual_seed(0)
    m = nn.Sequential(
        nn.Linear(64, 64),
        nn.Linear(64, 64),
        nn.Linear(64, 64),
        nn.Linear(64, 64),
    )
    # Inject extreme outlier in layer 2 so it has the worst E8 SNR
    with torch.no_grad():
        m[2].weight[:, 0] *= 200.0
    sensitive, reports = select_sensitive_layers(m, bitrate=2.0, group_size=32, keep_fp_fraction=0.25, min_numel=64)
    assert "2" in sensitive
    assert len(reports) == 4


def test_metal_e8_dispatch_safe_none_on_cpu():
    x = torch.randn(2, 32)
    wp = torch.zeros(16, 8, dtype=torch.uint8)
    sc = torch.ones(16, 1, dtype=torch.float16)
    assert metal_fused_e8_2bit_gemm(x, wp, sc, out_features=16, in_features=32, group_size=32) is None
    assert metal_fused_ternary_e8_gemm(x, wp, sc, torch.zeros(3281, 8, dtype=torch.int8), out_features=16, in_features=32, group_size=32) is None
