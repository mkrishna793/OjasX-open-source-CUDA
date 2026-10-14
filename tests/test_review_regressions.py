"""Correctness regressions from the October 10 repository review."""
import copy
import os
import subprocess
import sys
import multiprocessing
import socket
import struct
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn

from ojasx2.holonomic_math_compiler import holonomic_lm_head_loss
from ojasx2.mesh_hal import (HolonomicVirtualMeshEngine, MeshNodeSpec, MeshRemoteWorkerServer, plan_mesh_partitions,
                           _recv_framed_bytes, pack_tensor_for_wire, unpack_tensor_from_wire)


@pytest.mark.parametrize("smoothing", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("layout", ["vd", "dv"])
def test_chunked_loss_smoothed_gradients(smoothing, reduction, layout):
    torch.manual_seed(42)
    h = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    if layout == "dv":
        w = w.t().detach().requires_grad_()
    b = torch.randn(7, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[1, -100, 3], [4, 2, 0]])
    ref_inputs = [t.detach().clone().requires_grad_() for t in (h, w, b)]
    hr, wr, br = ref_inputs
    logits = F.linear(hr.reshape(-1, 4), wr if layout == "vd" else wr.t(), br)
    expected = F.cross_entropy(logits, targets.flatten(), reduction=reduction, label_smoothing=smoothing)
    actual = holonomic_lm_head_loss(h, w, targets, b, chunk_rows=2, reduction=reduction, label_smoothing=smoothing)
    torch.testing.assert_close(actual, expected)
    grads = torch.autograd.grad(actual, (h, w, b))
    expected_grads = torch.autograd.grad(expected, ref_inputs)
    for actual_grad, expected_grad in zip(grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    torch.testing.assert_close(grads[2].sum(), b.new_zeros(()), atol=1e-12, rtol=0)


@pytest.mark.parametrize("composition", ["input_cotangent", "squared_loss"])
def test_chunked_loss_composed_higher_derivatives(composition):
    torch.manual_seed(17)
    inputs = [torch.randn(*shape, dtype=torch.float64, requires_grad=True) for shape in ((3, 4), (7, 4), (7,))]
    reference = [t.detach().clone().requires_grad_() for t in inputs]
    targets = torch.tensor([1, 5, 3])
    loss = holonomic_lm_head_loss(*inputs[:2], targets, bias=inputs[2], chunk_rows=2, label_smoothing=0.1)
    ref_loss = F.cross_entropy(F.linear(*reference), targets, label_smoothing=0.1)
    if composition == "squared_loss":
        loss, ref_loss = loss.square(), ref_loss.square()
        cotangent = ref_cotangent = None
    else:
        cotangent, ref_cotangent = inputs[0].sum(), reference[0].sum()
    grads = torch.autograd.grad(loss, inputs, grad_outputs=cotangent, create_graph=True)
    ref_grads = torch.autograd.grad(ref_loss, reference, grad_outputs=ref_cotangent, create_graph=True)
    for actual, expected in zip(grads, ref_grads):
        torch.testing.assert_close(actual, expected)
    second = torch.autograd.grad(sum(g.sum() for g in grads), inputs)
    ref_second = torch.autograd.grad(sum(g.sum() for g in ref_grads), reference)
    for actual, expected in zip(second, ref_second):
        torch.testing.assert_close(actual, expected)


def test_chunked_loss_gradcheck():
    torch.manual_seed(11)
    h = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    w = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([1, 4])
    fn = lambda h, w: holonomic_lm_head_loss(h, w, targets, chunk_rows=1, label_smoothing=0.2)
    assert torch.autograd.gradcheck(fn, (h, w))
    assert torch.autograd.gradgradcheck(fn, (h, w))


@pytest.mark.parametrize("corrupt", ["output", "input_gradient", "weight_gradient"])
def test_backendbench_rejects_wrong_results(corrupt):
    from tools.run_industrial_all_level_benchmark import qualify_backendbench_case
    torch.manual_seed(12)
    expected = [torch.randn(3, 4) for _ in range(3)]
    actual = [t.clone() for t in expected]
    index = {"output": 0, "input_gradient": 1, "weight_gradient": 2}[corrupt]
    actual[index] *= -1
    with pytest.raises(AssertionError):
        qualify_backendbench_case(expected[0], actual[0], expected[1], actual[1], expected[2], actual[2])
    assert qualify_backendbench_case(expected[0], expected[0], expected[1], expected[1], expected[2], expected[2]) > 0.999


def test_platform_candidates_and_failed_native_load(monkeypatch, tmp_path):
    import ojasx2.tile_hal as hal
    dylib, so = tmp_path / "native.dylib", tmp_path / "native.so"
    dylib.touch(); so.touch()
    monkeypatch.setattr(hal, "_TILE_LIB_CANDIDATES", (dylib, so))
    monkeypatch.setattr(hal.sys, "platform", "linux")
    assert hal._native_library_candidates() == (so,)
    monkeypatch.setattr(hal, "_TILE_KERNELS_READY", False)
    monkeypatch.setattr(hal, "_LOADED_TILE_DYLIB", False)
    monkeypatch.setattr(hal, "_NATIVE_TILELANG_LOADED", False)
    monkeypatch.setattr(hal, "_NATIVE_LOAD_ERRORS", [])
    def incompatible(path):
        raise OSError("incompatible ABI")
    monkeypatch.setattr(torch.ops, "load_library", incompatible)
    assert hal.ensure_tile_kernels_loaded() is False
    assert "incompatible ABI" in hal.tile_kernel_status()["load_errors"][0]


def test_portable_fallback_without_native_binary():
    script = '''
import torch
import torch.nn.functional as F
import ojasx2
from ojasx2.tile_hal import tile_kernel_status
assert not tile_kernel_status()['native_loaded']
x=torch.randn(3,4,requires_grad=True)
w=torch.randn(7,4,requires_grad=True)
y,_=ojasx2.amx_linear_relu2(x,w)
expected=F.relu(F.linear(x,w)).square()
torch.testing.assert_close(y,expected)
for a,b in zip(torch.autograd.grad(y.sum(),(x,w)),torch.autograd.grad(expected.sum(),(x,w))):
    torch.testing.assert_close(a,b)
gamma=torch.ones(4,requires_grad=True)
out,_=ojasx2.l1_rmsnorm(x,gamma)
torch.testing.assert_close(out,F.rms_norm(x,(4,),gamma,1e-6))
out.sum().backward()
from ojasx2.mesh_hal import pack_tensor_for_wire,unpack_tensor_from_wire
p,_,_=pack_tensor_for_wire(torch.randn(5,50))
assert unpack_tensor_from_wire(p,torch.device('cpu'),torch.float32).shape==(5,50)
cuda_src = ojasx2.tilelang_emit_universal_source('cuda')
assert 'cuda' in cuda_src.lower() or '__ballot_sync' in cuda_src
try:
    ojasx2.tilelang_emit_universal_source('unsupported_arch')
except RuntimeError as e:
    assert 'Unsupported TileLang target_arch' in str(e)
else:
    raise AssertionError('Unknown arch must be rejected')
'''
    result = subprocess.run([sys.executable, "-c", script], env={**os.environ, "OJASX_DISABLE_NATIVE": "1"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_remote_worker_fragmented_frame_and_protocol_error():
    layer = nn.Linear(4, 7)
    worker = MeshRemoteWorkerServer(layer, compress_int8=False)
    try:
        with socket.create_connection((worker.host, worker.port), timeout=3) as sock:
            x = torch.randn(2, 4)
            payload, _, _ = pack_tensor_for_wire(x, compress_int8=False)
            message = b"\x01" + payload
            frame = struct.pack("!Q", len(message)) + message
            for byte in frame:
                sock.sendall(bytes([byte]))
            result = unpack_tensor_from_wire(_recv_framed_bytes(sock), torch.device("cpu"), x.dtype)
            torch.testing.assert_close(result, layer(x))
            from ojasx2.mesh_hal import _send_framed_bytes
            _send_framed_bytes(sock, b"\x99")
            assert _recv_framed_bytes(sock).startswith(b"\xffUnknown mesh opcode")
            assert sock.recv(1) == b""
    finally:
        worker.close()
    assert not worker._thread.is_alive()


def test_remote_double_precision_lossless_mode_and_eval():
    torch.manual_seed(14)
    model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32), nn.Dropout(0.5)).double()
    reference = copy.deepcopy(model).eval()
    # Default worker compression is enabled, but an engine requesting lossless
    # transport must negotiate lossless replies too.
    worker = MeshRemoteWorkerServer(nn.Sequential(copy.deepcopy(model[1]), nn.Dropout(0.5)))
    mesh = None
    try:
        nodes = [MeshNodeSpec("local", "cpu", 1, 1), MeshNodeSpec("remote", "cpu", 1, 1, "cpu", worker.host, worker.port)]
        mesh = HolonomicVirtualMeshEngine(model, nodes, compress_boundary=False).eval()
        assert worker.sub_module.training is False
        x = torch.randn(8, 32, dtype=torch.float64, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        y, yr = mesh(x), reference(xr)
        assert y.dtype == torch.float64
        torch.testing.assert_close(y, yr)
        y.square().sum().backward(); yr.square().sum().backward()
        torch.testing.assert_close(x.grad, xr.grad)
    finally:
        if mesh is not None:
            mesh.close()
        worker.close()


def test_remote_partition_mismatch_is_rejected():
    worker = MeshRemoteWorkerServer(nn.Sequential(nn.Linear(4, 4)))
    try:
        nodes = [MeshNodeSpec("local", "cpu", 1, 1), MeshNodeSpec("remote", "cpu", 1, 1, "cpu", worker.host, worker.port)]
        with pytest.raises(ValueError, match="initial weights"):
            HolonomicVirtualMeshEngine(nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)), nodes)
    finally:
        worker.close()


def _worker_process(module, pipe):
    optimizer = torch.optim.SGD(module.parameters(), lr=0.05)
    worker = MeshRemoteWorkerServer(module, compress_int8=False, optimizer=optimizer)
    pipe.send((worker.host, worker.port))
    try:
        while True:
            command = pipe.recv()
            if command == "stop":
                break
            if command == "gradients":
                pipe.send([None if p.grad is None else p.grad.tolist() for p in module.parameters()])
            if command == "weights":
                pipe.send([p.detach().tolist() for p in module.parameters()])
    finally:
        worker.close()
        pipe.close()


def test_mesh_executes_in_independent_worker_and_updates_remote_weights():
    torch.manual_seed(9)
    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4))
    reference = copy.deepcopy(model)
    indices = plan_mesh_partitions(model, [MeshNodeSpec("local", "cpu", 1, 1), MeshNodeSpec("partner", "cpu", 1, 1)]).node_assignments["partner"]
    remote = nn.Sequential(*(copy.deepcopy(model[i]) for i in indices))
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_worker_process, args=(remote, child))
    process.start()
    child.close()
    mesh = None
    try:
        assert parent.poll(15), "Remote worker failed to start"
        host, port = parent.recv()
        # The remote node device is deliberately unavailable on the coordinator.
        nodes = [MeshNodeSpec("local", "cpu", 1, 1), MeshNodeSpec("partner", "cpu", 1, 1, "cuda:99", host, port)]
        mesh = HolonomicVirtualMeshEngine(model, nodes, compress_boundary=False, timeout=3)
        assert mesh.plan.node_assignments["partner"] == indices
        assert isinstance(mesh.layers[2], nn.Identity)
        assert len(list(mesh.parameters())) == 2  # Only local parameters belong to local optimizers.
        xs = [torch.randn(3, 4, requires_grad=True) for _ in range(2)]
        refs = [x.detach().clone().requires_grad_() for x in xs]
        ys, yr = [mesh(x) for x in xs], [reference(x) for x in refs]
        for actual, expected in zip(ys, yr):
            torch.testing.assert_close(actual, expected)
        # Multiple forwards followed by reverse-order backward must use their own tapes.
        for i in (1, 0):
            ys[i].square().sum().backward()
            yr[i].square().sum().backward()
            torch.testing.assert_close(xs[i].grad, refs[i].grad)
        parent.send("gradients")
        assert parent.poll(5)
        for actual, expected in zip(parent.recv(), reference[2].parameters()):
            torch.testing.assert_close(torch.tensor(actual), expected.grad)
        for actual, expected in zip(mesh.layers[0].parameters(), reference[0].parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)
        mesh.step_remote()
        parent.send("weights")
        assert parent.poll(5)
        for actual, expected in zip(parent.recv(), reference[2].parameters()):
            torch.testing.assert_close(torch.tensor(actual), expected.detach() - 0.05 * expected.grad)
        mesh.zero_remote_grad()
        parent.send("gradients")
        assert parent.poll(5)
        assert parent.recv() == [None, None]
        # No input gradient is needed, but worker-owned parameters still need backward.
        for parameter in mesh.parameters():
            parameter.requires_grad_(False)
        mesh(torch.randn(2, 4)).sum().backward()
        parent.send("gradients")
        assert parent.poll(5)
        assert all(g is not None for g in parent.recv())
        with torch.no_grad():
            mesh(torch.randn(2, 4))
        mesh.step_remote()  # Inference must not retain an outstanding tape.
        assert mesh.telemetry.forward_packets >= 2
        assert mesh.telemetry.backward_packets >= 2
    finally:
        if mesh is not None:
            mesh.close()
        if process.is_alive():
            parent.send("stop")
        process.join(5)
        if process.is_alive():
            process.terminate(); process.join(5)
        parent.close()
    assert process.exitcode == 0
