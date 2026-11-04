"""Executable security contracts for mesh framing and remote workers."""

from types import SimpleNamespace
import socket
import struct

import pytest
import torch
import torch.nn as nn

from tools.run_two_laptop_physical_mesh import _execution_scope
from ojasx2.mesh_hal import (
    MeshRemoteWorkerServer,
    MeshTransferTelemetry,
    _AUTH_FLAG,
    _MeshAuthenticator,
    _RemotePartitionClient,
    _recv_framed_bytes,
    pack_tensor_for_wire,
    unpack_tensor_from_wire,
)

KEY = b"mesh-test-key-32-bytes-minimum!!!"
OTHER_KEY = b"wrong-test-key-32-bytes-minimum!"
SESSION = b"0123456789abcdef0123456789abcdef"


def _raw_authenticated_frame(authenticator, payload):
    sequence, tag = authenticator.seal(payload)
    return (
        struct.pack("!Q", len(payload) | _AUTH_FLAG)
        + struct.pack("!Q", sequence)
        + payload
        + tag
    )


def test_frame_limit_rejects_header_before_reading_payload():
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack("!Q", 1025))
        with pytest.raises(ValueError, match="exceeds"):
            _recv_framed_bytes(right, max_frame_bytes=1024)
        left.sendall(struct.pack("!Q", 11) + b"hello")
        with pytest.raises(ValueError, match="authentication configuration"):
            _recv_framed_bytes(right, authenticator=_MeshAuthenticator(KEY, SESSION))
    finally:
        left.close()
        right.close()


def test_authenticated_frames_reject_tampering_replay_and_session_reuse():
    sender = _MeshAuthenticator(KEY, SESSION)
    raw = _raw_authenticated_frame(sender, b"optimizer-step")
    receiver = _MeshAuthenticator(KEY, SESSION)
    left, right = socket.socketpair()
    try:
        left.sendall(raw)
        assert _recv_framed_bytes(right, authenticator=receiver) == b"optimizer-step"
    finally:
        left.close()
        right.close()

    # Replaying the same valid bytes on the same ordered session is rejected.
    left, right = socket.socketpair()
    try:
        left.sendall(raw)
        with pytest.raises(ValueError, match="replay|reordering"):
            _recv_framed_bytes(right, authenticator=receiver)
    finally:
        left.close()
        right.close()

    tampered = bytearray(raw)
    tampered[17] ^= 1
    left, right = socket.socketpair()
    try:
        left.sendall(tampered)
        with pytest.raises(ValueError, match="authentication failed"):
            _recv_framed_bytes(right, authenticator=_MeshAuthenticator(KEY, SESSION))
    finally:
        left.close()
        right.close()

    left, right = socket.socketpair()
    try:
        left.sendall(raw)
        with pytest.raises(ValueError, match="authentication failed"):
            _recv_framed_bytes(right, authenticator=_MeshAuthenticator(KEY, SESSION[:-1] + b"x"))
    finally:
        left.close()
        right.close()


def test_raw_and_compressed_tensor_decoders_reject_malformed_metadata():
    x = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    for compress in (False, True):
        payload, _, _ = pack_tensor_for_wire(x, compress_int8=compress)
        decoded = unpack_tensor_from_wire(payload, torch.device("cpu"), torch.float32)
        assert decoded.shape == x.shape
        for malformed in (payload[:-1], payload + b"trailing"):
            with pytest.raises(ValueError):
                unpack_tensor_from_wire(malformed, torch.device("cpu"), torch.float32)

    raw, _, _ = pack_tensor_for_wire(x, compress_int8=False)
    bad_dtype = bytearray(raw)
    bad_dtype[4] = 255
    with pytest.raises(ValueError, match="dtype"):
        unpack_tensor_from_wire(bytes(bad_dtype), torch.device("cpu"), torch.float32)
    negative_dim = bytearray(raw)
    struct.pack_into("!q", negative_dim, 8, -1)
    with pytest.raises(ValueError, match="shape"):
        unpack_tensor_from_wire(bytes(negative_dim), torch.device("cpu"), torch.float32)

    compressed, _, _ = pack_tensor_for_wire(x, compress_int8=True)
    bad_length = bytearray(compressed)
    struct.pack_into("!Q", bad_length, 12, 2**40)
    with pytest.raises(ValueError, match="lengths"):
        unpack_tensor_from_wire(bytes(bad_length), torch.device("cpu"), torch.float32)


def test_authenticated_remote_worker_roundtrip_and_wrong_key_rejection():
    torch.manual_seed(8831)
    module = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
    server = MeshRemoteWorkerServer(module, compress_int8=False, timeout=2.0, auth_key=KEY)
    node = SimpleNamespace(host=server.host, port=server.port, node_id="remote")
    client = None
    try:
        client = _RemotePartitionClient(
            node, module, MeshTransferTelemetry(), False, 2.0, auth_key=KEY
        )
        x = torch.randn(3, 4)
        actual = client.tensor_request(7, 0, x)
        torch.testing.assert_close(actual, module(x))
    finally:
        if client is not None:
            client.close()
        server.close()

    server = MeshRemoteWorkerServer(module, compress_int8=False, timeout=1.0, auth_key=KEY)
    node = SimpleNamespace(host=server.host, port=server.port, node_id="remote")
    try:
        with pytest.raises((ValueError, RuntimeError, ConnectionError, OSError)):
            _RemotePartitionClient(
                node, module, MeshTransferTelemetry(), False, 1.0, auth_key=OTHER_KEY
            )
    finally:
        server.close()


def test_non_loopback_worker_requires_key_and_short_keys_fail_closed():
    with pytest.raises(ValueError, match="Non-loopback"):
        MeshRemoteWorkerServer(nn.Linear(2, 2), host="0.0.0.0")
    with pytest.raises(ValueError, match="32 bytes"):
        MeshRemoteWorkerServer(nn.Linear(2, 2), auth_key=b"short")


def test_mesh_cli_scope_never_promotes_loopback_to_physical_qualification():
    loopback = _execution_scope("127.0.0.1")
    assert loopback["loopback_execution"] is True
    assert "not two-physical-laptop qualification" in loopback["execution_scope"]
    remote = _execution_scope("192.0.2.10")
    assert remote["loopback_execution"] is False
    assert "not independently verified" in remote["execution_scope"]
    assert "scenario labels" in remote["hardware_identity_scope"]
