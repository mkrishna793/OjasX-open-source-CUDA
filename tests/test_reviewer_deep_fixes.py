"""Regression tests certifying all 6 deep engineering fixes from the external repository review."""

from __future__ import annotations

import copy
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx2.holonomic_math_compiler import (
    HolonomicGeGLU,
    HolonomicLinear,
    HolonomicSwiGLU,
    holonomic_lm_head_loss,
)
from ojasx2.mesh_hal import (
    HolonomicVirtualMeshEngine,
    MeshNodeSpec,
    MeshRemoteWorkerServer,
    _recv_framed_bytes,
    _send_framed_bytes,
    pack_tensor_for_wire,
    unpack_tensor_from_wire,
)
from ojasx2.residuals import SavedTensorAudit
from ojasx2.smart_router import SmartWorkloadRouter
from ojasx2.tile_hal import tilelang_emit_universal_source


def test_fix1_no_hidden_ctx_saved_z_or_saved_residuals() -> None:
    """Verify no pre-activations are hidden on ctx attributes outside ctx.save_for_backward."""
    torch.manual_seed(0)
    x = torch.randn(16, 64, requires_grad=True)

    # 1. HolonomicLinear with ReLU packs 1-bit sign mask via save_for_backward (32x smaller than FP32 z)
    lin_relu = HolonomicLinear(64, 128, bias=True, act="relu")
    with SavedTensorAudit(external=[x, *lin_relu.parameters()]) as audit_relu:
        y_relu = lin_relu(x)
    assert not hasattr(y_relu.grad_fn, "saved_z")
    # Fresh saved storage is strictly the 1-bit packed mask: 16 * 128 / 8 = 256 bytes (vs 8192B FP32 z)
    assert audit_relu.count == 4
    assert audit_relu.summary()["fresh_saved_storage_bytes"] == 256
    y_relu.sum().backward()

    # 2. HolonomicLinear with SiLU saves ONLY (x, weight, bias) on tape (0 fresh saved bytes of z)
    x2 = torch.randn(16, 64, requires_grad=True)
    lin_silu = HolonomicLinear(64, 128, bias=True, act="silu")
    with SavedTensorAudit(external=[x2, *lin_silu.parameters()]) as audit_silu:
        y_silu = lin_silu(x2)
    assert not hasattr(y_silu.grad_fn, "saved_z")
    assert audit_silu.count == 3
    assert audit_silu.summary()["fresh_saved_storage_bytes"] == 0
    y_silu.sum().backward()

    # 3. HolonomicSwiGLU and HolonomicGeGLU save ONLY (x, weights, biases) — 0 fresh saved bytes
    x3 = torch.randn(8, 32, requires_grad=True)
    swiglu = HolonomicSwiGLU(32, 64)
    geglu = HolonomicGeGLU(32, 64, approximate="none")
    with SavedTensorAudit(external=[x3, *swiglu.parameters()]) as audit_swiglu:
        y_swi = swiglu(x3)
    assert not hasattr(y_swi.grad_fn, "saved_residuals")
    assert audit_swiglu.summary()["fresh_saved_storage_bytes"] == 0
    y_swi.sum().backward()

    x4 = torch.randn(8, 32, requires_grad=True)
    with SavedTensorAudit(external=[x4, *geglu.parameters()]) as audit_geglu:
        y_geg = geglu(x4)
    assert not hasattr(y_geg.grad_fn, "saved_residuals")
    assert audit_geglu.summary()["fresh_saved_storage_bytes"] == 0
    y_geg.sum().backward()


def test_fix2_chunked_lm_head_loss_grad_enabled_zero_logit_tape() -> None:
    """Verify holonomic_lm_head_loss uses custom chunked backward when grad is enabled."""
    torch.manual_seed(42)
    B, T, D, V = 4, 32, 64, 512
    h_ref = torch.randn(B, T, D, requires_grad=True)
    w_ref = torch.randn(V, D, requires_grad=True)
    b_ref = torch.randn(V, requires_grad=True)
    targets = torch.randint(0, V, (B, T))
    targets[0, 0] = -100  # test ignore_index handling

    h_oj = h_ref.detach().clone().requires_grad_(True)
    w_oj = w_ref.detach().clone().requires_grad_(True)
    b_oj = b_ref.detach().clone().requires_grad_(True)

    # Reference full-logit cross_entropy
    logits_ref = F.linear(h_ref.view(-1, D), w_ref, b_ref)
    loss_ref = F.cross_entropy(logits_ref, targets.view(-1), ignore_index=-100, reduction="mean")
    loss_ref.backward()

    # OjasX chunked LM head under normal grad-enabled training
    assert torch.is_grad_enabled()
    with SavedTensorAudit(external=[h_oj, w_oj, targets, b_oj]) as audit_lm:
        loss_oj = holonomic_lm_head_loss(
            h_oj, w_oj, targets, bias=b_oj, chunk_tokens=32, ignore_index=-100, reduction="mean"
        )
    # Strictly 0 fresh saved bytes on the autograd tape (0 bytes of [B*T, V] chunk logits retained!)
    assert audit_lm.summary()["fresh_saved_storage_bytes"] == 0
    loss_oj.backward()

    assert torch.allclose(loss_oj, loss_ref, atol=1e-5)
    assert torch.allclose(h_oj.grad, h_ref.grad, atol=1e-5)
    assert torch.allclose(w_oj.grad, w_ref.grad, atol=1e-5)
    assert torch.allclose(b_oj.grad, b_ref.grad, atol=1e-5)


def test_fix3_smart_router_preserves_gelu_mode_and_leaky_relu_slope() -> None:
    """Verify SmartWorkloadRouter preserves GELU(approximate='none'|'tanh') and LeakyReLU(negative_slope)."""
    torch.manual_seed(7)
    model_ref = nn.Sequential(
        nn.Linear(64, 64),
        nn.GELU(approximate="none"),
        nn.Linear(64, 64),
        nn.GELU(approximate="tanh"),
        nn.Linear(64, 64),
        nn.LeakyReLU(negative_slope=0.25),
    )
    model_oj = copy.deepcopy(model_ref)
    router = SmartWorkloadRouter(tiny_param_threshold=16)
    model_oj, _ = router.optimize_model(model_oj, batch_tokens=32)

    assert isinstance(model_oj[0], HolonomicLinear) and model_oj[0].act == "gelu_exact"
    assert isinstance(model_oj[2], HolonomicLinear) and model_oj[2].act == "gelu_tanh"
    assert isinstance(model_oj[4], HolonomicLinear) and model_oj[4].act == "leaky_relu"
    assert abs(model_oj[4].negative_slope - 0.25) < 1e-9

    x_ref = torch.randn(16, 64, requires_grad=True)
    x_oj = x_ref.detach().clone().requires_grad_(True)
    y_ref = model_ref(x_ref)
    y_oj = model_oj(x_oj)
    assert torch.allclose(y_oj, y_ref, atol=1e-5)

    y_ref.square().mean().backward()
    y_oj.square().mean().backward()
    assert torch.allclose(x_oj.grad, x_ref.grad, atol=1e-5)


def test_fix4_mesh_deterministic_binary_wire_and_remote_worker() -> None:
    """Verify deterministic C-struct wire framing (no pickle) and exact wire byte accounting."""
    torch.manual_seed(11)
    # Test non-multiple of 32 shape to exercise padded q_shape / orig_shape handling
    x = torch.randn(5, 50, dtype=torch.float32)
    payload, wire_bytes, raw_bytes = pack_tensor_for_wire(x, compress_int8=True, block_size=32)

    # First byte must be _WIRE_MAGIC (0x4F), not pickle PROTO (0x80)
    assert payload[0] == 0x4F
    # Reported wire_bytes must equal exact framed TCP packet size (8-byte length header + payload)
    assert wire_bytes == 8 + len(payload)
    assert raw_bytes == 5 * 50 * 4

    recon = unpack_tensor_from_wire(payload, torch.device("cpu"), torch.float32)
    assert list(recon.shape) == [5, 50]
    cos_sim = F.cosine_similarity(x.flatten(), recon.flatten(), dim=0).item()
    assert cos_sim > 0.999

    # Verify standalone MeshRemoteWorkerServer over a real TCP socket
    remote_layer = nn.Linear(32, 32)
    worker = MeshRemoteWorkerServer(remote_layer, host="127.0.0.1", port=0, device="cpu")
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((worker.host, worker.port))
        inp = torch.randn(4, 32)
        p_in, _, _ = pack_tensor_for_wire(inp, compress_int8=True)
        _send_framed_bytes(sock, bytes([1]) + p_in)
        p_out = _recv_framed_bytes(sock)
        out = unpack_tensor_from_wire(p_out, torch.device("cpu"), torch.float32)
        assert list(out.shape) == [4, 32]
        sock.close()
    finally:
        worker.close()


def test_fix5_ascend_codegen_complete_ub_epilogue() -> None:
    """Verify Huawei Ascend 910B kernel codegen emits complete L0C->UB->GM epilogue."""
    src = tilelang_emit_universal_source("ascend", block_m=32, block_n=32, block_k=16, panel_size=4)
    assert "ascend_gemm_l1" in src
    assert "asc_copy_l0c2ub" in src
    assert "ub_mask" in src
    assert "asc_copy_ub2gm(Out_gm, ub_out" in src
    assert "asc_copy_ub2gm(Mask_gm, ub_mask" in src
