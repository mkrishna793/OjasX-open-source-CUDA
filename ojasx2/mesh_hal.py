"""Sequential model partitioning over local devices and preloaded TCP workers.

Plans use caller-supplied capacity estimates. Optional block-int8 compression is
lossy; telemetry counts application frames, not IP/TCP network overhead. Remote
execution supports first-order training and explicit worker-local optimizer steps.
"""

from __future__ import annotations

import hashlib
import json
import socket
import struct
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .tile_hal import ensure_tile_kernels_loaded

ensure_tile_kernels_loaded()

_WIRE_MAGIC = 0x4F  # 'O' for OjasX deterministic binary wire framing
_DTYPE_TO_CODE = {
    torch.float32: 1,
    torch.float16: 2,
    torch.bfloat16: 3,
    torch.float64: 4,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class MeshNodeSpec:
    node_id: Any = "0"
    chip_class: str = "generic_silicon"
    free_memory_gb: float = 16.0
    compute_tflops: float = 10.0
    device: str = "cpu"
    host: str = "127.0.0.1"
    port: int = 0  # 0: local device; positive port: preloaded remote partition worker
    is_local: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", str(self.node_id))
        if self.is_local:
            object.__setattr__(self, "port", 0)


@dataclass
class MeshTransferTelemetry:
    forward_packets: int = 0
    backward_packets: int = 0
    compressed_wire_bytes: int = 0
    uncompressed_fp32_bytes: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.compressed_wire_bytes == 0:
            return 1.0
        return self.uncompressed_fp32_bytes / float(self.compressed_wire_bytes)

    @property
    def bandwidth_saved_pct(self) -> float:
        if self.uncompressed_fp32_bytes == 0:
            return 0.0
        return 100.0 * (1.0 - (self.compressed_wire_bytes / float(self.uncompressed_fp32_bytes)))


@dataclass
class VirtualMeshPartitionPlan:
    """Capacity estimates supplied by the caller; not measured usable GPU memory."""
    pooled_memory_gb: float
    pooled_tflops: float
    node_assignments: Dict[str, List[int]] = field(default_factory=dict)
    node_param_counts: Dict[str, int] = field(default_factory=dict)
    node_memory_share_pct: Dict[str, float] = field(default_factory=dict)
    node_capacity_share_pct: Dict[str, float] = field(default_factory=dict)


def _send_framed_bytes(sock: socket.socket, payload: bytes) -> None:
    header = struct.pack("!Q", len(payload))
    sock.sendall(header + payload)


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    buf = bytearray()
    while len(buf) < num_bytes:
        chunk = sock.recv(num_bytes - len(buf))
        if not chunk:
            raise ConnectionError("P2P Mesh socket closed unexpectedly")
        buf.extend(chunk)
    return bytes(buf)


def _recv_framed_bytes(sock: socket.socket) -> bytes:
    header = _recv_exact(sock, 8)
    (length,) = struct.unpack("!Q", header)
    return _recv_exact(sock, length)


def pack_tensor_for_wire(
    x: torch.Tensor,
    compress_int8: bool = True,
    block_size: int = 32,
) -> Tuple[bytes, int, int]:
    """Serialize tensor for P2P wire transfer using deterministic C-struct binary framing.

    Eliminates Python pickle (`torch.save`) overhead completely so that the actual
    bytes transmitted over the OS TCP socket match `compressed_wire_bytes` exactly
    (including the 8-byte TCP frame length header and binary shape header).
    """
    x_cpu = x.detach().to("cpu").contiguous()
    raw_fp32_bytes = int(x_cpu.numel() * 4)
    orig_shape = list(x_cpu.shape)
    ndim = len(orig_shape)

    if compress_int8 and x_cpu.numel() >= block_size:
        q, scale = torch.ops.ojasx_hal.compress_boundary_int8(x_cpu, block_size)
        q_shape = list(q.shape)
        s_shape = list(scale.shape)
        q_bytes = q.contiguous().numpy().tobytes()
        s_bf16_bytes = scale.to(torch.bfloat16).view(torch.uint16).contiguous().numpy().tobytes()

        # Binary header: magic(B), mode=1(B), ndim(H), block_size(I), q_ndim(H), s_ndim(H), q_len(Q), s_len(Q)
        header = struct.pack(
            "!BBHIHHQQ",
            _WIRE_MAGIC,
            1,
            ndim,
            int(block_size),
            len(q_shape),
            len(s_shape),
            len(q_bytes),
            len(s_bf16_bytes),
        )
        dims_bin = struct.pack(f"!{ndim}q", *orig_shape)
        q_dims_bin = struct.pack(f"!{len(q_shape)}q", *q_shape)
        s_dims_bin = struct.pack(f"!{len(s_shape)}q", *s_shape)
        wire_payload = header + dims_bin + q_dims_bin + s_dims_bin + q_bytes + s_bf16_bytes
        # Exact wire bytes sent on the TCP socket = 8-byte frame header + payload
        actual_wire_bytes = 8 + len(wire_payload)
        return wire_payload, actual_wire_bytes, raw_fp32_bytes

    dtype_code = _DTYPE_TO_CODE.get(x_cpu.dtype, 1)
    if x_cpu.dtype == torch.bfloat16:
        raw_bytes = x_cpu.view(torch.uint16).numpy().tobytes()
    else:
        raw_bytes = x_cpu.to(torch.float32 if dtype_code == 1 else x_cpu.dtype).numpy().tobytes()
    header = struct.pack("!BBHI", _WIRE_MAGIC, 0, ndim, dtype_code)
    dims_bin = struct.pack(f"!{ndim}q", *orig_shape)
    wire_payload = header + dims_bin + raw_bytes
    actual_wire_bytes = 8 + len(wire_payload)
    return wire_payload, actual_wire_bytes, raw_fp32_bytes


def unpack_tensor_from_wire(
    payload: bytes,
    target_device: torch.device,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Deserialize deterministic C-struct binary frame back into a PyTorch tensor."""
    magic, mode = struct.unpack_from("!BB", payload, 0)
    if magic != _WIRE_MAGIC:
        raise ValueError(f"Invalid OjasX mesh wire magic byte: {magic:#x}")

    if mode == 0:
        _, _, ndim, dtype_code = struct.unpack_from("!BBHI", payload, 0)
        offset = 8
        orig_shape = list(struct.unpack_from(f"!{ndim}q", payload, offset))
        offset += 8 * ndim
        raw_buf = memoryview(payload)[offset:]
        dtype = _CODE_TO_DTYPE.get(dtype_code, torch.float32)
        if dtype == torch.bfloat16:
            arr = np.frombuffer(raw_buf, dtype=np.uint16).reshape(orig_shape).copy()
            t = torch.from_numpy(arr).view(torch.bfloat16)
        else:
            np_dtype = {
                torch.float32: np.float32,
                torch.float16: np.float16,
                torch.float64: np.float64,
            }[dtype]
            arr = np.frombuffer(raw_buf, dtype=np_dtype).reshape(orig_shape).copy()
            t = torch.from_numpy(arr)
        return t.to(device=target_device, dtype=target_dtype)

    _, _, ndim, _block_size, q_ndim, s_ndim, q_len, s_len = struct.unpack_from(
        "!BBHIHHQQ", payload, 0
    )
    offset = struct.calcsize("!BBHIHHQQ")
    orig_shape = list(struct.unpack_from(f"!{ndim}q", payload, offset))
    offset += 8 * ndim
    q_shape = list(struct.unpack_from(f"!{q_ndim}q", payload, offset))
    offset += 8 * q_ndim
    s_shape = list(struct.unpack_from(f"!{s_ndim}q", payload, offset))
    offset += 8 * s_ndim

    q_view = memoryview(payload)[offset : offset + q_len]
    offset += q_len
    s_view = memoryview(payload)[offset : offset + s_len]

    q_np = np.frombuffer(q_view, dtype=np.int8).reshape(q_shape).copy()
    s_u16 = np.frombuffer(s_view, dtype=np.uint16).reshape(s_shape).copy()
    q_t = torch.from_numpy(q_np)
    s_t = torch.from_numpy(s_u16).view(torch.bfloat16).to(torch.float32)
    recon = torch.ops.ojasx_hal.decompress_boundary_int8(
        q_t, s_t, orig_shape, target_dtype
    )
    return recon.to(device=target_device, dtype=target_dtype)


class _TCPLoopbackP2PChannel:
    """Real TCP socket channel supporting both local loopback and multi-host P2P links."""

    def __init__(self, bind_host: str = "127.0.0.1", bind_port: int = 0) -> None:
        self.bind_host = bind_host
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((bind_host, bind_port))
        self.server_sock.listen(1)
        self.port = self.server_sock.getsockname()[1]

        self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self.peer_sock: Optional[socket.socket] = None
        accept_thread = threading.Thread(target=self._accept, daemon=True)
        accept_thread.start()
        connect_host = "127.0.0.1" if bind_host in ("0.0.0.0", "") else bind_host
        self.client_sock.connect((connect_host, self.port))
        accept_thread.join(timeout=2.0)

    def _accept(self) -> None:
        conn, _ = self.server_sock.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.peer_sock = conn

    def transfer_tensor(
        self,
        x: torch.Tensor,
        target_device: torch.device,
        target_dtype: torch.dtype,
        telemetry: MeshTransferTelemetry,
        is_backward: bool = False,
        compress_int8: bool = True,
    ) -> torch.Tensor:
        assert self.peer_sock is not None
        payload, wire_bytes, raw_bytes = pack_tensor_for_wire(x, compress_int8=compress_int8)
        if is_backward:
            telemetry.backward_packets += 1
        else:
            telemetry.forward_packets += 1
        telemetry.compressed_wire_bytes += wire_bytes
        telemetry.uncompressed_fp32_bytes += raw_bytes

        # Send across real OS TCP socket from client -> peer
        _send_framed_bytes(self.client_sock, payload)
        received_payload = _recv_framed_bytes(self.peer_sock)
        return unpack_tensor_from_wire(received_payload, target_device, target_dtype)

    def close(self) -> None:
        for s in (self.client_sock, self.peer_sock, self.server_sock):
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass


def _partition_signature(module: nn.Module) -> str:
    """Check architecture and initial state without serializing executable Python objects."""
    digest = hashlib.sha256(repr(module).encode())
    digest.update(str([(name, p.requires_grad) for name, p in module.named_parameters()]).encode())
    digest.update(str([(name, m.training) for name, m in module.named_modules()]).encode())
    for name, tensor in module.state_dict().items():
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class MeshRemoteWorkerServer:
    """TCP worker owning a preloaded partition and its parameter gradients.

    The coordinator verifies the initial partition state before using it. An optional
    worker-local optimizer is stepped only on an explicit coordinator request.
    One coordinator connection is served at a time; multiple outstanding forwards
    are identified separately, so reverse-order backward is safe.
    """

    def __init__(
        self,
        sub_module: nn.Module,
        host: str = "127.0.0.1",
        port: int = 0,
        device: str = "cpu",
        compress_int8: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        timeout: float = 30.0,
    ) -> None:
        self.device = torch.device(device)
        self.sub_module = sub_module.to(self.device)
        self.compress_int8 = compress_int8
        self.optimizer = optimizer
        if optimizer is not None:
            owned = {id(p) for p in self.sub_module.parameters()}
            if any(id(p) not in owned for group in optimizer.param_groups for p in group["params"]):
                raise ValueError("Move the worker partition to its device before constructing its optimizer")
        self.timeout = timeout
        self.last_error: Optional[str] = None
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((host, port))
        self.server_sock.listen(1)
        self.server_sock.settimeout(0.2)
        self.host, self.port = self.server_sock.getsockname()
        self._stop = False
        self._conn: Optional[socket.socket] = None
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def _serve_loop(self) -> None:
        while not self._stop:
            try:
                conn, _ = self.server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._conn = conn
            graphs = {}
            response_compression = self.compress_int8
            try:
                conn.settimeout(self.timeout)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while not self._stop:
                    # TCP is a byte stream: both header and payload require exact reads.
                    msg = _recv_framed_bytes(conn)
                    if not msg:
                        raise ValueError("Empty mesh request")
                    op_code = msg[0]
                    if op_code == 8:  # Initial partition handshake
                        if len(msg) > 1:
                            response_compression = bool(msg[1])
                        metadata = {"signature": _partition_signature(self.sub_module),
                                    "parameters": sum(p.numel() for p in self.sub_module.parameters())}
                        _send_framed_bytes(conn, json.dumps(metadata).encode())
                        continue
                    if op_code in (5, 6, 9):  # Optimizer step / gradient reset / train mode
                        if graphs:
                            raise RuntimeError("Finish outstanding backwards before updating remote parameters")
                        if op_code == 5:
                            if self.optimizer is None:
                                raise RuntimeError("Remote worker has no configured optimizer")
                            self.optimizer.step()
                        elif op_code == 6:
                            self.sub_module.zero_grad(set_to_none=True)
                        else:
                            self.sub_module.train(bool(msg[1]))
                        _send_framed_bytes(conn, b"OK")
                        continue
                    if op_code not in (1, 2, 3, 4, 7):
                        raise ValueError(f"Unknown mesh opcode {op_code}")
                    request_id = 0 if op_code in (1, 2) else struct.unpack("!Q", msg[1:9])[0]
                    tensor_payload = msg[1:] if op_code in (1, 2) else msg[10:]
                    dtype = torch.float32 if op_code in (1, 2) else _CODE_TO_DTYPE[msg[9]]
                    if op_code in (1, 3, 7):
                        if request_id in graphs:
                            raise RuntimeError("Duplicate outstanding forward request")
                        inp = unpack_tensor_from_wire(tensor_payload, self.device, dtype).requires_grad_(op_code != 7)
                        with torch.set_grad_enabled(op_code != 7):
                            out = self.sub_module(inp)
                        if op_code != 7:
                            graphs[request_id] = (inp, out)
                        response = out.detach()
                    else:
                        if request_id not in graphs:
                            raise RuntimeError("Backward has no matching forward request")
                        inp, out = graphs.pop(request_id)
                        grad_out = unpack_tensor_from_wire(tensor_payload, self.device, out.dtype)
                        out.backward(grad_out)
                        if inp.grad is None:
                            raise RuntimeError("Remote partition did not produce an input gradient")
                        response = inp.grad.detach()
                    payload, _, _ = pack_tensor_for_wire(response, compress_int8=response_compression)
                    if op_code not in (1, 2):
                        payload = bytes([_DTYPE_TO_CODE[response.dtype]]) + payload
                    _send_framed_bytes(conn, payload)
            except (ConnectionError, OSError) as exc:
                if not self._stop:
                    self.last_error = str(exc)
            except Exception as exc:
                self.last_error = str(exc)
                try:
                    _send_framed_bytes(conn, b"\xff" + str(exc).encode())
                except OSError:
                    pass
            finally:
                graphs.clear()
                conn.close()
                self._conn = None

    def wait(self) -> None:
        """Keep a standalone worker process alive until it is closed or interrupted."""
        self._thread.join()

    def close(self) -> None:
        self._stop = True
        for sock in (self._conn, self.server_sock):
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()
        self._thread.join(timeout=1.0)


class _RemotePartitionClient:
    def __init__(self, node, expected_module, telemetry, compress_int8, timeout):
        self.telemetry = telemetry
        self.compress_int8 = compress_int8
        self.sock = socket.create_connection((node.host, node.port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.lock = threading.Lock()
        self.next_request_id = 1
        try:
            metadata = json.loads(self._exchange(b"\x08" + bytes([compress_int8])))
            if metadata["signature"] != _partition_signature(expected_module):
                raise ValueError(f"Remote partition {node.node_id} differs from its assigned architecture or initial weights")
        except Exception:
            self.close()
            raise

    def _exchange(self, message):
        with self.lock:
            _send_framed_bytes(self.sock, message)
            response = _recv_framed_bytes(self.sock)
        if response.startswith(b"\xff"):
            raise RuntimeError("Remote mesh worker: " + response[1:].decode(errors="replace"))
        return response

    def tensor_request(self, opcode, request_id, tensor):
        payload, _, raw_bytes = pack_tensor_for_wire(tensor, compress_int8=self.compress_int8)
        message = bytes([opcode]) + struct.pack("!Q", request_id) + bytes([_DTYPE_TO_CODE[tensor.dtype]]) + payload
        response = self._exchange(message)
        if opcode in (3, 7):
            self.telemetry.forward_packets += 2
        else:
            self.telemetry.backward_packets += 2
        self.telemetry.compressed_wire_bytes += 8 + len(message) + 8 + len(response)
        output = unpack_tensor_from_wire(response[1:], tensor.device, _CODE_TO_DTYPE[response[0]])
        self.telemetry.uncompressed_fp32_bytes += raw_bytes + output.numel() * 4
        return output

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


class _RemotePartitionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, anchor, client):
        ctx.client = client
        ctx.request_id = client.next_request_id
        client.next_request_id += 1
        return client.tensor_request(3, ctx.request_id, x)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        grad_in = ctx.client.tensor_request(4, ctx.request_id, grad_out)
        return grad_in if ctx.needs_input_grad[0] else None, None, None


class _MeshSocketBoundaryBridge(torch.autograd.Function):
    """Autograd bridge that streams forward boundary activations & backward cotangents over TCP."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        channel: _TCPLoopbackP2PChannel,
        target_device: torch.device,
        telemetry: MeshTransferTelemetry,
        compress_int8: bool,
    ) -> torch.Tensor:
        ctx.channel = channel
        ctx.src_device = x.device
        ctx.src_dtype = x.dtype
        ctx.telemetry = telemetry
        ctx.compress_int8 = compress_int8
        return channel.transfer_tensor(
            x,
            target_device=target_device,
            target_dtype=x.dtype,
            telemetry=telemetry,
            is_backward=False,
            compress_int8=compress_int8,
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_in = ctx.channel.transfer_tensor(
            grad_out,
            target_device=ctx.src_device,
            target_dtype=ctx.src_dtype,
            telemetry=ctx.telemetry,
            is_backward=True,
            compress_int8=ctx.compress_int8,
        )
        return grad_in, None, None, None, None


def plan_mesh_partitions(model, nodes) -> VirtualMeshPartitionPlan:
    """Plan contiguous layer assignments without moving modules or opening sockets.

    Use the same model/checkpoint and descriptors on both machines when preloading
    remote partitions. Descriptor capacity totals are estimates, not measured RAM.
    """
    if not nodes or not len(model) or len({n.node_id for n in nodes}) != len(nodes):
        raise ValueError("Partitioning requires layers and uniquely named nodes")
    if any(n.free_memory_gb <= 0 or n.compute_tflops <= 0 for n in nodes):
        raise ValueError("Node capacities must be positive")
    total_mem = sum(n.free_memory_gb for n in nodes)
    total_tflops = sum(n.compute_tflops for n in nodes)
    shares = [0.65 * n.free_memory_gb / total_mem + 0.35 * n.compute_tflops / total_tflops for n in nodes]
    assignments = {n.node_id: [] for n in nodes}
    counts = {n.node_id: 0 for n in nodes}
    weights = [max(1, sum(p.numel() for p in layer.parameters())) for layer in model]
    cumulative_weight = 0
    node_idx = 0
    target = shares[0]
    total_weight = sum(weights)
    for idx, weight in enumerate(weights):
        cumulative_weight += weight / total_weight
        while cumulative_weight > target + 1e-6 and node_idx < len(nodes) - 1:
            node_idx += 1
            target += shares[node_idx]
        node_id = nodes[node_idx].node_id
        assignments[node_id].append(idx)
        counts[node_id] += sum(p.numel() for p in model[idx].parameters())
    return VirtualMeshPartitionPlan(total_mem, total_tflops, assignments, counts,
        {n.node_id: 100 * n.free_memory_gb / total_mem for n in nodes},
        {n.node_id: 100 * share for n, share in zip(nodes, shares)})


class HolonomicVirtualMeshEngine(nn.Module):
    """Sequential partitioning across local devices and preloaded remote workers.

    A node with port=0 is local; port>0 connects to its host:port. Remote workers
    must own the corresponding Sequential partition with matching initial state.
    Their parameters are removed from the coordinator's module/optimizer. Capacity
    totals are planning estimates supplied by the caller, not measured GPU pooling.
    Remote execution supports first-order autograd; higher derivatives are rejected.
    """

    def __init__(
        self,
        model: Optional[Union[nn.Sequential, Sequence[nn.Module]]] = None,
        nodes: Optional[List[MeshNodeSpec]] = None,
        compress_boundary: Optional[bool] = None,
        timeout: float = 30.0,
        *,
        stages: Optional[Sequence[nn.Module]] = None,
        compress_wire: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if model is None and stages is not None:
            model = stages
        if isinstance(model, (list, tuple)):
            model = nn.Sequential(*model)
        if compress_wire is not None:
            compress_boundary = compress_wire
        if compress_boundary is None:
            compress_boundary = True

        if nodes is None or len(nodes) < 2:
            raise ValueError("HolonomicVirtualMeshEngine requires at least 2 MeshNodeSpec nodes")
        if len({n.node_id for n in nodes}) != len(nodes):
            raise ValueError("Mesh node IDs must be unique")
        if not isinstance(model, nn.Sequential) or len(model) == 0:
            raise ValueError("Mesh requires a nonempty nn.Sequential model")
        if any(n.port < 0 or n.free_memory_gb <= 0 or n.compute_tflops <= 0 for n in nodes):
            raise ValueError("Mesh node capacities must be positive and ports nonnegative")
        self.nodes = nodes
        self.compress_boundary = compress_boundary
        self.telemetry = MeshTransferTelemetry()
        self.channel = None
        self.remote_clients = {}
        # A non-parameter autograd anchor ensures remote weights receive gradients
        # even when the coordinator input and preceding local layers are frozen.
        self.register_buffer("_remote_anchor", torch.zeros((), requires_grad=True), persistent=False)
        self.layers = nn.ModuleList(list(model.children()))
        self.plan = self._build_capacity_weighted_plan()
        owners = {}
        for node in nodes:
            for i in self.plan.node_assignments[node.node_id]:
                for parameter in self.layers[i].parameters():
                    previous = owners.setdefault(id(parameter), node.node_id)
                    if previous != node.node_id:
                        raise ValueError("Parameters shared across mesh partitions are unsupported")
        for node in nodes:
            for i in self.plan.node_assignments[node.node_id]:
                if node.port == 0:
                    self.layers[i].to(torch.device(node.device))
        try:
            for node in nodes:
                indices = self.plan.node_assignments[node.node_id]
                if node.port > 0 and indices:
                    partition = nn.Sequential(*(self.layers[i] for i in indices))
                    self.remote_clients[node.node_id] = _RemotePartitionClient(
                        node, partition, self.telemetry, compress_boundary, timeout
                    )
                    for i in indices:
                        self.layers[i] = nn.Identity()
        except Exception:
            self.close()
            raise

    def _build_capacity_weighted_plan(self) -> VirtualMeshPartitionPlan:
        return plan_mesh_partitions(self.layers, self.nodes)

    def _owner_node_for_layer(self, layer_idx: int) -> MeshNodeSpec:
        for n in self.nodes:
            if layer_idx in self.plan.node_assignments[n.node_id]:
                return n
        return self.nodes[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        current_node = None
        idx = 0
        while idx < len(self.layers):
            node = self._owner_node_for_layer(idx)
            if node.port > 0:
                client = self.remote_clients[node.node_id]
                if torch.is_grad_enabled():
                    h = _RemotePartitionFunction.apply(h, self._remote_anchor, client)
                else:
                    h = client.tensor_request(7, 0, h)
                idx += len(self.plan.node_assignments[node.node_id])
                current_node = node
                continue
            if current_node is not None and current_node.port == 0 and node.node_id != current_node.node_id:
                if self.channel is None:
                    self.channel = _TCPLoopbackP2PChannel()
                h = _MeshSocketBoundaryBridge.apply(
                    h,
                    self.channel,
                    torch.device(node.device),
                    self.telemetry,
                    self.compress_boundary,
                )
            else:
                h = h.to(torch.device(node.device))
            h = self.layers[idx](h)
            current_node = node
            idx += 1
        return h

    def step_remote(self):
        """Explicitly step worker-local optimizers after completing backward."""
        for client in self.remote_clients.values():
            client._exchange(b"\x05")

    def train(self, mode=True):
        super().train(mode)
        for client in getattr(self, "remote_clients", {}).values():
            client._exchange(b"\x09" + bytes([mode]))
        return self

    def zero_remote_grad(self):
        for client in self.remote_clients.values():
            client._exchange(b"\x06")

    @property
    def stats(self) -> MeshTransferTelemetry:
        return self.telemetry

    def forward_backward(
        self, x: torch.Tensor, loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> torch.Tensor:
        out = self.forward(x)
        if loss_fn is not None:
            loss = loss_fn(out)
            loss.backward()
        return out

    def close(self) -> None:
        if self.channel is not None:
            self.channel.close()
            self.channel = None
        for client in self.remote_clients.values():
            client.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
