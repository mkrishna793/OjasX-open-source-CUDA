"""M2.5 replay fusion and joint residual-policy optimization tests."""

import json
import math

import pytest
import torch

from ojasx2.core import (
    JointGraphBuilder,
    MemoryBudgetError,
    execute_joint_graph,
    execute_jet_graph,
    fuse_activation_chains,
    plan_jet_memory,
    plan_m25_memory,
    plan_memory,
)
from ojasx2.core.activation_math import ACTIVATION_NAMES, activation_forward
from ojasx2.core.jet_execute import _LiveTensorLedger


def _split(operation):
    parts = operation.split(":")
    return parts[0], parts[1] if len(parts) == 2 else "none"


def _build(program, *, fused, policy="replay_streaming", dtype=torch.float64):
    builder = JointGraphBuilder()
    x = builder.input("x", (4,), dtype=dtype, requires_grad=True)
    if fused:
        output = builder.forward(
            "activation_chain",
            (x,),
            residual_policy=policy,
            attrs={"operations": tuple(program)},
        )
    else:
        output = x
        for operation in program:
            kind, approximate = _split(operation)
            attrs = {"approximate": approximate} if kind == "gelu" else None
            output = builder.forward(kind, (output,), attrs=attrs)
    builder.request_gradient(x)
    builder.set_objective(output)
    return builder, x


def _value(kind, dtype=torch.float64):
    if kind == "log":
        return torch.tensor([math.e**2, math.e**2.5, math.e**3, math.e**3.5], dtype=dtype)
    if kind == "sqrt":
        return torch.tensor([0.4, 0.7, 1.2, 2.0], dtype=dtype)
    if kind == "exp":
        return torch.tensor([-0.2, -0.1, 0.0, 0.1], dtype=dtype)
    if kind == "relu":
        return torch.tensor([0.2, 0.4, 0.7, 1.1], dtype=dtype)
    return torch.tensor([-0.7, -0.2, 0.3, 0.8], dtype=dtype)


@pytest.mark.parametrize("kind", ACTIVATION_NAMES)
@pytest.mark.parametrize("policy", ["replay_streaming", "replay_cached"])
def test_fused_pullback_is_bitwise_equal_to_unfused_and_matches_pytorch(kind, policy):
    program = [kind] * 3
    baseline_builder, baseline_x = _build(program, fused=False)
    fused_builder, fused_x = _build(program, fused=True, policy=policy)
    baseline = baseline_builder.generate_reverse()
    fused = fused_builder.generate_reverse()
    value = _value(kind)
    expected_baseline = execute_joint_graph(
        baseline,
        plan_memory(baseline),
        {baseline_x: value},
    )
    expected_fused = execute_joint_graph(
        fused,
        plan_memory(fused),
        {fused_x: value},
    )
    assert torch.equal(
        expected_fused[fused.objective],
        expected_baseline[baseline.objective],
    )
    assert torch.equal(
        expected_fused[fused.gradient_targets[fused_x]],
        expected_baseline[baseline.gradient_targets[baseline_x]],
    )

    reference = value.clone().requires_grad_(True)
    output = reference
    for _ in program:
        output = activation_forward(kind, output)
    gradient, = torch.autograd.grad(output.sum(), reference)
    torch.testing.assert_close(expected_fused[fused.objective], output.sum().detach())
    torch.testing.assert_close(
        expected_fused[fused.gradient_targets[fused_x]],
        gradient,
        rtol=2e-9,
        atol=2e-9,
    )


def test_gelu_tanh_fusion_is_bitwise_equal():
    program = ["sigmoid", "gelu:tanh", "tanh"]
    baseline_builder, baseline_x = _build(program, fused=False)
    fused_builder, fused_x = _build(program, fused=True)
    baseline = baseline_builder.generate_reverse()
    fused = fused_builder.generate_reverse()
    value = _value("sigmoid")
    left = execute_joint_graph(baseline, plan_memory(baseline), {baseline_x: value})
    right = execute_joint_graph(fused, plan_memory(fused), {fused_x: value})
    assert torch.equal(left[baseline.objective], right[fused.objective])
    assert torch.equal(
        left[baseline.gradient_targets[baseline_x]],
        right[fused.gradient_targets[fused_x]],
    )


def test_relu_nan_convention_survives_fusion():
    baseline_builder, baseline_x = _build(["relu", "sigmoid"], fused=False)
    fused_builder, fused_x = _build(["relu", "sigmoid"], fused=True)
    baseline = baseline_builder.generate_reverse()
    fused = fused_builder.generate_reverse()
    value = torch.tensor([float("nan"), -0.0, -1.0, 2.0], dtype=torch.float64)
    left = execute_joint_graph(baseline, plan_memory(baseline), {baseline_x: value})
    right = execute_joint_graph(fused, plan_memory(fused), {fused_x: value})
    torch.testing.assert_close(
        left[baseline.gradient_targets[baseline_x]],
        right[fused.gradient_targets[fused_x]],
        equal_nan=True,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "operations,message",
    [
        ((), "nonempty"),
        (("unknown",), "Unsupported"),
        (("gelu:bad",), "approximation"),
        (("relu:none",), "no approximation"),
        (("sigmoid:tanh:extra",), "Invalid"),
    ],
)
def test_activation_program_validation(operations, message):
    builder = JointGraphBuilder()
    x = builder.input("x", (4,), requires_grad=True)
    with pytest.raises(ValueError, match=message):
        builder.forward("activation_chain", (x,), attrs={"operations": operations})


def test_fusion_preserves_requested_intermediate_boundary():
    builder = JointGraphBuilder()
    x = builder.input("x", (4,), requires_grad=True)
    y = builder.forward("sigmoid", (x,))
    z = builder.forward("tanh", (y,))
    builder.request_gradient(y)
    builder.set_objective(z)
    fusion = fuse_activation_chains(builder)
    assert fusion.groups == ()
    assert all(node.op_name != "activation_chain" for node in fusion.builder.graph.nodes)


def test_fusion_does_not_cross_fanout():
    builder = JointGraphBuilder()
    x = builder.input("x", (4,), requires_grad=True)
    y = builder.forward("sigmoid", (x,))
    builder.forward("tanh", (y,))
    output = builder.forward("exp", (y,))
    builder.set_objective(output)
    assert fuse_activation_chains(builder).groups == ()


def test_m25_reduces_peak_and_preserves_bitwise_execution():
    program = ["sigmoid"] * 4
    builder, x = _build(program, fused=False)
    plan = plan_m25_memory(builder)
    assert plan.fusion_selected
    assert plan.certificate.peak_bytes < plan.baseline_certificate.peak_bytes
    assert plan.saved_bytes > 0
    assert plan.replay_evaluations == 10
    assert plan.optimal_for_considered_graphs
    assert json.dumps(plan.to_dict())
    value = _value("sigmoid")
    optimized = execute_joint_graph(plan.graph, plan.certificate, {x: value})
    baseline = execute_joint_graph(
        plan.baseline_graph,
        plan.baseline_certificate,
        {x: value},
    )
    assert torch.equal(optimized[plan.graph.objective], baseline[plan.baseline_graph.objective])
    assert torch.equal(
        optimized[plan.graph.gradient_targets[x]],
        baseline[plan.baseline_graph.gradient_targets[x]],
    )
    with pytest.raises(MemoryBudgetError):
        plan_m25_memory(builder, budget=plan.certificate.peak_bytes - 1)

    def observed_peak(graph, certificate):
        with torch.no_grad(), _LiveTensorLedger() as ledger:
            ledger._record(value)
            execute_joint_graph(graph, certificate, {x: value})
            return ledger.peak

    assert observed_peak(plan.graph, plan.certificate) < observed_peak(
        plan.baseline_graph,
        plan.baseline_certificate,
    )


def test_m25_negative_control_keeps_unfused_graph():
    builder, _ = _build(["sigmoid"] * 2, fused=False)
    plan = plan_m25_memory(builder)
    assert not plan.fusion_selected
    assert plan.saved_bytes == 0
    assert plan.certificate.peak_bytes == plan.baseline_certificate.peak_bytes
    assert all(node.op_name != "activation_chain" for node in plan.graph.nodes)


def test_joint_policy_optimizer_composes_structured_and_dense_residual_costs():
    plans = {}
    for structured in (False, True):
        builder = JointGraphBuilder(structured_cotangents=structured)
        x = builder.input("x", (64,), requires_grad=True)
        output = builder.forward("relu", (x,))
        builder.request_gradient(x)
        builder.set_objective(output)
        plans[structured] = plan_m25_memory(builder, fuse=False)
        assert plans[structured].replay_evaluations == 0
    dense, structured = plans[False], plans[True]
    assert dict(dense.policy_choices)[0] == "packed_sign"
    assert dense.certificate.peak_bytes < dense.baseline_certificate.peak_bytes
    assert dict(structured.policy_choices)[0] == "resident_input"
    assert structured.certificate.peak_bytes < dense.certificate.peak_bytes


def test_greedy_policy_fallback_is_not_marked_optimal():
    builder, _ = _build(["sigmoid"] * 4, fused=False)
    plan = plan_m25_memory(
        builder,
        exact_policy_node_limit=0,
        exact_node_limit=0,
    )
    assert not plan.optimal_for_considered_graphs
    assert plan.method == "deterministic_greedy_policy_search"


@pytest.mark.parametrize("kind", ACTIVATION_NAMES)
def test_fused_jets_match_unfused_through_order_four(kind):
    builders = [_build([kind, kind], fused=False), _build([kind, kind], fused=True)]
    results = []
    value = _value(kind)
    for builder, x in builders:
        # M2.5 first-order gradient requests are irrelevant to clean primal jets;
        # use a fresh primal-only clone without generating reverse mode.
        primal = JointGraphBuilder()
        source = primal.input("x", (4,), dtype=torch.float64, requires_grad=True)
        if builder.graph.nodes[0].op_name == "activation_chain":
            output = primal.forward(
                "activation_chain",
                (source,),
                attrs={"operations": (kind, kind)},
            )
        else:
            first = primal.forward(kind, (source,))
            output = primal.forward(kind, (first,))
        primal.set_objective(output)
        graph, certificate = plan_jet_memory(primal.graph, order=4)
        results.append(
            execute_jet_graph(graph, certificate, {source: value})[graph.objective]
        )
    fused, unfused = results
    torch.testing.assert_close(fused.value, unfused.value)
    for order in range(1, 5):
        torch.testing.assert_close(
            fused.derivative((order,)),
            unfused.derivative((order,)),
            rtol=2e-7,
            atol=2e-7,
        )
