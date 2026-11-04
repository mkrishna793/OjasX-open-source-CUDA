"""M0/M1 structured-cotangent laws, topology, and execution regressions."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Mapping, Sequence, Tuple

import pytest
import torch

from ojasx2.core import (
    AccuracyMode,
    CORE_REGISTRY,
    DENSE_COTANGENT,
    UNIFORM_COTANGENT,
    CoreOperator,
    JointGraphBuilder,
    OperatorContract,
    ResidualBuild,
    ResidualPolicy,
    Value,
    analyze_cotangent_topology,
    execute_joint_graph,
    plan_memory,
)
from ojasx2.core.memory import graph_digest
from ojasx2.core.structured import add_uniform, clone_uniform, divide_uniform


def _run(build, tensors: Sequence[torch.Tensor], structured: bool):
    builder = JointGraphBuilder(structured_cotangents=structured)
    sources, requests, objective = build(builder)
    for source in requests:
        builder.request_gradient(source)
    builder.set_objective(objective)
    graph = builder.generate_reverse()
    certificate = plan_memory(graph)
    outputs = execute_joint_graph(graph, certificate, dict(zip(sources, tensors)))
    return graph, certificate, outputs


def _assert_bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> None:
    integer_dtype = {torch.float32: torch.int32, torch.float64: torch.int64}[left.dtype]
    assert left.shape == right.shape
    assert torch.equal(left.view(integer_dtype), right.view(integer_dtype))


def _compare_dense_and_structured(build, tensors: Sequence[torch.Tensor]):
    dense_graph, dense_certificate, dense = _run(build, tensors, False)
    graph, certificate, result = _run(build, tensors, True)
    for value_id, gradient in dense_graph.gradient_targets.items():
        _assert_bitwise_equal(result[graph.gradient_targets[value_id]], dense[gradient])
    assert all(value.encoding == DENSE_COTANGENT for value in dense_graph.values.values())
    return dense_graph, dense_certificate, graph, certificate, result


def _sum_graph(builder: JointGraphBuilder):
    x = builder.input("x", (3, 5), dtype=torch.float64, requires_grad=True)
    objective = builder.forward("sum", (x,))
    return (x,), (x,), objective


def test_sum_has_one_scalar_region_and_explicit_dense_output_cut():
    value = torch.randn(3, 5, dtype=torch.float64)
    dense_graph, dense_certificate, graph, certificate, result = _compare_dense_and_structured(
        _sum_graph, (value,)
    )
    topology = analyze_cotangent_topology(graph)
    assert len(topology.uniform_regions) == 1
    assert len(topology.uniform_values) == 2
    assert topology.logical_bytes == 128
    assert topology.managed_bytes == 16
    assert topology.saved_bytes == 112
    assert len(topology.materialization_nodes) == 1
    assert graph.nodes[topology.materialization_nodes[0]].kind == "cotangent_materialize"
    gradient = result[graph.gradient_targets[0]]
    assert gradient.is_contiguous()
    # The mandatory dense public output can dominate peak liveness; the
    # intermediate representation saving is certified separately above.
    assert certificate.peak_bytes <= dense_certificate.peak_bytes
    assert graph_digest(graph) != graph_digest(dense_graph)


@pytest.mark.parametrize("shape", [(), (2, 3), (2, 0, 3)])
def test_sum_and_mean_dense_fallback_cover_scalar_and_empty_shapes(shape):
    for operation in ("sum", "mean"):
        def build(builder):
            x = builder.input("x", shape, dtype=torch.float64, requires_grad=True)
            objective = builder.forward(operation, (x,))
            return (x,), (x,), objective

        value = torch.randn(shape, dtype=torch.float64)
        _, _, graph, _, _ = _compare_dense_and_structured(build, (value,))
        if shape == (2, 0, 3):
            assert all(
                value.encoding == DENSE_COTANGENT
                for value in graph.values.values()
                if value.kind == "cotangent" and value.shape == shape
            )


def test_mean_scales_once_and_materializes_bitwise_dense_gradient():
    def build(builder):
        x = builder.input("x", (4, 7), dtype=torch.float64, requires_grad=True)
        objective = builder.forward("mean", (x,))
        return (x,), (x,), objective

    value = torch.randn(4, 7, dtype=torch.float64)
    _, _, graph, _, result = _compare_dense_and_structured(build, (value,))
    topology = analyze_cotangent_topology(graph)
    assert topology.logical_bytes == 232
    assert topology.managed_bytes == 16
    assert result[graph.gradient_targets[0]].shape == value.shape


def test_reindexing_composes_one_uniform_region_across_reverse_views():
    def build(builder):
        x = builder.input("x", (2, 3, 4), dtype=torch.float32, requires_grad=True)
        reshaped = builder.forward("reshape", (x,), attrs={"shape": (4, 6)})
        transposed = builder.forward("transpose", (reshaped,), attrs={"dim0": 0, "dim1": 1})
        objective = builder.forward("sum", (transposed,))
        return (x,), (x,), objective

    value = torch.randn(2, 3, 4)
    dense_graph, dense_certificate, graph, certificate, _ = _compare_dense_and_structured(
        build, (value,)
    )
    topology = analyze_cotangent_topology(graph)
    assert len(topology.uniform_values) == 4
    assert len(topology.uniform_regions) == 1
    assert len(topology.uniform_regions[0]) == 4
    assert topology.managed_bytes == 16
    assert topology.saved_bytes == topology.logical_bytes - 16
    assert certificate.peak_bytes < dense_certificate.peak_bytes


def test_additive_fanout_adds_uniform_scalars_before_dense_cut():
    def build(builder):
        x = builder.input("x", (2, 3), dtype=torch.float64, requires_grad=True)
        doubled = builder.forward("add", (x, x))
        objective = builder.forward("sum", (doubled,))
        return (x,), (x,), objective

    value = torch.randn(2, 3, dtype=torch.float64)
    _, _, graph, _, result = _compare_dense_and_structured(build, (value,))
    topology = analyze_cotangent_topology(graph)
    assert len(topology.uniform_values) == 5
    assert len(topology.uniform_regions) == 1
    assert topology.managed_bytes == 40
    assert any(node.kind == "cotangent_add" for node in graph.nodes)
    _assert_bitwise_equal(result[graph.gradient_targets[0]], torch.full_like(value, 2.0))


def test_uniform_helpers_use_only_one_scalar_of_owned_storage():
    scalar = torch.tensor(3.0, dtype=torch.float32)
    uniform = clone_uniform(scalar, (5, 7))
    assert uniform.shape == (5, 7)
    assert uniform.stride() == (0, 0)
    assert uniform.untyped_storage().nbytes() == scalar.element_size()
    total = add_uniform(uniform, divide_uniform(uniform, 2, uniform.shape))
    assert total.untyped_storage().nbytes() == scalar.element_size()
    assert total[0, 0].item() == 4.5
    dense = total.clone(memory_format=torch.contiguous_format)
    assert dense.untyped_storage().nbytes() == dense.numel() * dense.element_size()
    _assert_bitwise_equal(dense, torch.full((5, 7), 4.5))


def test_uniform_value_storage_and_public_boundaries_are_validated():
    _, _, graph, _, _ = _compare_dense_and_structured(
        _sum_graph, (torch.randn(3, 5, dtype=torch.float64),)
    )
    uniform_id = next(
        value_id
        for value_id in analyze_cotangent_topology(graph).uniform_values
        if graph.values[value_id].nbytes > graph.values[value_id].storage_bytes
    )
    original = graph.values[uniform_id]
    graph.values[uniform_id] = replace(original, storage_bytes=original.nbytes)
    with pytest.raises(ValueError, match="Invalid uniform cotangent storage"):
        graph.validate()
    graph.values[uniform_id] = original
    graph.outputs.append(uniform_id)
    with pytest.raises(ValueError, match="Public graph outputs require dense"):
        graph.validate()


@pytest.mark.parametrize("kind", CORE_REGISTRY.names())
def test_every_builtin_pullback_matches_dense_bitwise_with_uniform_upstream(kind):
    shape = (2, 3)
    if kind == "linear":
        shapes, attrs = ((2, 3), (4, 3), (4,)), {}
    elif kind == "matmul":
        shapes, attrs = ((2, 3), (3, 4)), {}
    elif kind == "layer_norm":
        shapes, attrs = (shape, (3,), (3,)), {"eps": 1e-5}
    elif kind == "rms_norm":
        shapes, attrs = (shape, (3,)), {"eps": 1e-5}
    elif kind in {"add", "mul", "div"}:
        shapes, attrs = (shape, shape), {}
    else:
        shapes, attrs = (shape,), {}
    if kind == "activation_chain":
        attrs = {"operations": ("sigmoid", "tanh")}
    elif kind in {"softmax", "log_softmax"}:
        attrs = {"dim": -1}
    elif kind == "reshape":
        attrs = {"shape": (3, 2)}
    elif kind == "transpose":
        attrs = {"dim0": 0, "dim1": 1}

    def build(builder):
        sources = tuple(
            builder.input(f"source_{index}", source_shape, dtype=torch.float64, requires_grad=True)
            for index, source_shape in enumerate(shapes)
        )
        output = builder.forward(kind, sources, attrs=attrs)
        return sources, sources, output

    tensors = []
    for index, source_shape in enumerate(shapes):
        count = math.prod(source_shape)
        if kind in {"log", "sqrt"} or (kind == "div" and index == 1):
            tensor = torch.linspace(0.5, 1.5, count, dtype=torch.float64).reshape(source_shape)
        else:
            tensor = torch.linspace(-1.2 + index * 0.1, 1.3 + index * 0.1, count, dtype=torch.float64).reshape(source_shape)
        tensors.append(tensor)
    _compare_dense_and_structured(build, tensors)


class _DenseOnlyIdentity(CoreOperator):
    contract = OperatorContract(
        name="test_dense_only_identity",
        min_arity=1,
        max_arity=1,
        accuracy=AccuracyMode.REPRODUCIBLE,
        deterministic=True,
        max_derivative_order=1,
        derivative_domain="test identity",
        residual_policies=(
            ResidualPolicy(
                name="test_input",
                accuracy=AccuracyMode.REPRODUCIBLE,
                lossless=True,
                satisfies=("input",),
            ),
        ),
    )

    def output_shape(self, inputs: Sequence[Value], attrs: Mapping[str, object]) -> Tuple[int, ...]:
        return inputs[0].shape

    def residual_layout(self, inputs, demand, policy):
        return (ResidualBuild("input", "input", (0,), source_index=0),)

    def forward_workspace(self, inputs, demand, policy):
        return 0

    def backward_workspace(self, inputs, demand, policy):
        return 0

    def forward(self, tensors, attrs, policy):
        return tensors[0].clone(), {"input": tensors[0]}

    def backward(self, graph, node, upstream, residuals):
        return (upstream,)


def test_unqualified_consumer_receives_an_explicit_dense_materialization(monkeypatch):
    monkeypatch.setitem(CORE_REGISTRY._operators, "test_dense_only_identity", _DenseOnlyIdentity())

    def build(builder):
        x = builder.input("x", (2, 4), dtype=torch.float32, requires_grad=True)
        identity = builder.forward("test_dense_only_identity", (x,))
        objective = builder.forward("sum", (identity,))
        return (x,), (x,), objective

    value = torch.randn(2, 4)
    _, _, graph, _, result = _compare_dense_and_structured(build, (value,))
    materializations = [node for node in graph.nodes if node.kind == "cotangent_materialize"]
    assert len(materializations) == 1
    pullback = next(node for node in graph.nodes if node.op_name == "test_dense_only_identity" and node.kind == "pullback")
    assert pullback.inputs[0] == materializations[0].outputs[0]
    assert graph.values[pullback.inputs[0]].encoding == DENSE_COTANGENT
    _assert_bitwise_equal(result[graph.gradient_targets[0]], torch.ones_like(value))
