import dataclasses
import json
from pathlib import Path

import pytest
import torch

from ojasx2.core import (
    JointGraphBuilder,
    MemoryBudgetError,
    execute_joint_graph,
    plan_m25_memory,
)
from ojasx2.core.egraph import EGraph, RewriteRule, UnprovedRewriteError
from ojasx2.core.m4 import (
    M4Config,
    M4Input,
    M4Problem,
    activation,
    activation_chain_expr,
    add,
    default_rewrite_rules,
    matmul,
    plan_m4,
    tensor_input,
    transpose,
)
from tools.verify_core_z3 import verify as verify_core_z3


def _chain_problem(kinds, shape=(8,)):
    spec = M4Input("x", shape, dtype=torch.float64)
    result = tensor_input(spec)
    for kind in kinds:
        result = activation(result, kind)
    return M4Problem((spec,), result)


def _execute_pair(plan, tensors):
    inputs = {index: value for index, value in enumerate(tensors)}
    actual = execute_joint_graph(plan.graph, plan.certificate, inputs)
    expected = execute_joint_graph(
        plan.baseline_graph,
        plan.baseline_certificate,
        inputs,
    )
    return actual, expected


def test_m4_discovers_segmented_schedule_beyond_m25_and_executes_bitwise():
    problem = _chain_problem(["sigmoid"] * 4)
    config = M4Config(
        max_iterations=8,
        beam_width=32,
        max_candidates=64,
        exact_node_limit=32,
    )
    plan = plan_m4(problem, config=config)
    assert plan.saturation.saturated
    assert plan.discovered
    assert plan.evaluated_candidates > 2
    assert plan.certificate.peak_bytes < plan.baseline_certificate.peak_bytes
    assert plan.proof_obligations
    assert "streaming" in plan.rewrite_families
    assert any(item.proofs for item in plan.proof_obligations)
    assert json.dumps(plan.to_dict())

    builder = JointGraphBuilder()
    x = builder.input("x", (8,), dtype=torch.float64, requires_grad=True)
    y = x
    for _ in range(4):
        y = builder.forward("sigmoid", (y,))
    builder.request_gradient(x)
    builder.set_objective(y)
    m25 = plan_m25_memory(builder, exact_node_limit=32)
    assert plan.certificate.peak_bytes < m25.certificate.peak_bytes

    value = torch.linspace(-0.3, 0.4, 8, dtype=torch.float64)
    actual, expected = _execute_pair(plan, [value])
    assert torch.equal(
        actual[plan.graph.objective],
        expected[plan.baseline_graph.objective],
    )
    assert torch.equal(
        actual[plan.graph.gradient_targets[0]],
        expected[plan.baseline_graph.gradient_targets[0]],
    )


def test_m4_is_deterministic_and_backward_costs_validate():
    problem = _chain_problem(["sigmoid"])
    # Use the public activation constructor for the GELU approximation.
    spec = problem.inputs[0]
    expression = tensor_input(spec)
    expression = activation(expression, "sigmoid")
    expression = activation(expression, "gelu", approximate="tanh")
    expression = activation(expression, "tanh")
    problem = M4Problem((spec,), expression)
    config = M4Config(max_iterations=8, beam_width=32, exact_node_limit=32)
    first = plan_m4(problem, config=config)
    second = plan_m4(problem, config=config)
    assert first.selected_expression == second.selected_expression
    assert first.certificate.graph_digest == second.certificate.graph_digest
    assert first.to_dict() == second.to_dict()
    certificate = first.certificate
    assert certificate.work_units == sum(step.work_units for step in certificate.steps)
    assert certificate.traffic_bytes == sum(step.traffic_bytes for step in certificate.steps)
    assert certificate.replay_evaluations == sum(
        step.replay_evaluations for step in certificate.steps
    )
    certificate.validate(first.graph)
    with pytest.raises(ValueError, match="liveness fields"):
        dataclasses.replace(certificate, traffic_bytes=certificate.traffic_bytes + 1).validate(
            first.graph
        )
    with pytest.raises(ValueError, match="liveness fields"):
        dataclasses.replace(certificate, work_units=certificate.work_units + 1).validate(
            first.graph
        )


def test_m4_relu_policy_accounts_for_uniform_upstream_and_preserves_nan_semantics():
    problem = _chain_problem(["relu"], shape=(64,))
    plan = plan_m4(problem, config=M4Config(exact_node_limit=16))
    assert plan.selected_expression.policy == "resident_input"
    assert plan.certificate.peak_bytes == plan.baseline_certificate.peak_bytes
    value = torch.linspace(-1, 1, 64, dtype=torch.float64)
    value[0] = torch.nan
    value[1] = -0.0
    actual, expected = _execute_pair(plan, [value])
    torch.testing.assert_close(
        actual[plan.graph.objective],
        expected[plan.baseline_graph.objective],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert torch.equal(
        actual[plan.graph.gradient_targets[0]],
        expected[plan.baseline_graph.gradient_targets[0]],
    )


def test_m4_replay_alternatives_have_distinct_backward_work():
    spec = M4Input("x", (8,), dtype=torch.float64)
    expression = activation_chain_expr(
        tensor_input(spec),
        ["sigmoid", "tanh", "sin"],
    )
    graph = EGraph()
    root = graph.add_expr(expression)
    for rule in default_rewrite_rules():
        if rule.name == "replay_policy":
            graph.register_rule(rule)
    graph.saturate(max_iterations=4)
    from ojasx2.core.m4 import _local_cost

    candidates, truncated = graph.extract(
        root,
        local_cost=lambda node: _local_cost(graph, node),
    )
    assert not truncated
    by_policy = {candidate.expr.policy: candidate for candidate in candidates}
    assert set(by_policy) == {"replay_streaming", "replay_cached"}
    assert by_policy["replay_cached"].work_units < by_policy["replay_streaming"].work_units
    assert by_policy["replay_cached"].traffic_bytes < by_policy["replay_streaming"].traffic_bytes
    assert all(candidate.obligations for candidate in candidates)


def test_m4_exact_real_contraction_is_gated_and_discovers_better_parenthesization():
    specs = (
        M4Input("a", (30, 2), dtype=torch.float64),
        M4Input("b", (2, 40), dtype=torch.float64),
        M4Input("c", (40, 3), dtype=torch.float64),
    )
    a, b, c = (tensor_input(spec) for spec in specs)
    problem = M4Problem(specs, matmul(matmul(a, b), c))
    blocked = plan_m4(problem, config=M4Config(exact_node_limit=32))
    assert not any(item.family == "contraction" for item in blocked.generated_obligations)
    assert blocked.selected_expression == blocked.baseline_expression

    plan = plan_m4(
        problem,
        config=M4Config(allow_exact_real=True, exact_node_limit=32),
    )
    assert any(item.family == "contraction" for item in plan.generated_obligations)
    assert plan.certificate.traffic_bytes < plan.baseline_certificate.traffic_bytes
    assert plan.selected_expression != plan.baseline_expression
    tensors = [
        torch.randn(30, 2, dtype=torch.float64),
        torch.randn(2, 40, dtype=torch.float64),
        torch.randn(40, 3, dtype=torch.float64),
    ]
    actual, expected = _execute_pair(plan, tensors)
    torch.testing.assert_close(
        actual[plan.graph.objective],
        expected[plan.baseline_graph.objective],
        rtol=2e-12,
        atol=2e-12,
    )
    for source in range(3):
        torch.testing.assert_close(
            actual[plan.graph.gradient_targets[source]],
            expected[plan.baseline_graph.gradient_targets[source]],
            rtol=2e-11,
            atol=2e-11,
        )


def test_m4_structured_adjoint_rules_and_involution():
    specs = (
        M4Input("a", (3, 4), dtype=torch.float64),
        M4Input("b", (4, 2), dtype=torch.float64),
    )
    a, b = (tensor_input(spec) for spec in specs)
    problem = M4Problem(specs, transpose(matmul(a, b)))
    plan = plan_m4(
        problem,
        config=M4Config(allow_exact_real=True, exact_node_limit=32),
    )
    assert any(item.family == "structured_adjoint" for item in plan.generated_obligations)
    actual, expected = _execute_pair(
        plan,
        [torch.randn(3, 4, dtype=torch.float64), torch.randn(4, 2, dtype=torch.float64)],
    )
    torch.testing.assert_close(
        actual[plan.graph.objective],
        expected[plan.baseline_graph.objective],
        rtol=2e-12,
        atol=2e-12,
    )

    involution_problem = M4Problem((specs[0],), transpose(transpose(a)))
    involution = plan_m4(involution_problem, config=M4Config(exact_node_limit=16))
    assert involution.selected_expression == a
    assert involution.certificate.work_units < involution.baseline_certificate.work_units
    assert "structured_adjoint" in involution.rewrite_families


def test_m4_fanout_accumulation_and_original_fallback():
    spec = M4Input("x", (12,), dtype=torch.float64)
    x = tensor_input(spec)
    shared = activation(x, "sigmoid")
    problem = M4Problem((spec,), add(shared, activation(shared, "tanh")))
    plan = plan_m4(problem, config=M4Config(exact_node_limit=32))
    value = torch.linspace(-0.4, 0.5, 12, dtype=torch.float64)
    actual, expected = _execute_pair(plan, [value])
    assert torch.equal(
        actual[plan.graph.objective],
        expected[plan.baseline_graph.objective],
    )
    assert torch.equal(
        actual[plan.graph.gradient_targets[0]],
        expected[plan.baseline_graph.gradient_targets[0]],
    )
    assert plan.certificate.peak_bytes <= plan.baseline_certificate.peak_bytes


def test_m4_root_package_exports_public_api():
    import ojasx2.core as core

    expected = {
        "M4Config", "M4Input", "M4Plan", "M4Problem", "plan_m4",
        "tensor_input", "activation", "activation_chain_expr", "matmul",
        "transpose", "add", "default_rewrite_rules", "EGraph", "ENode",
        "Expr", "ExtractedCandidate", "ProofObligation", "Rewrite",
        "RewriteRule", "SaturationReport", "UnprovedRewriteError",
    }
    assert expected <= set(core.__all__)
    assert all(getattr(core, name) is not None for name in expected)
    assert core.M4Config is M4Config
    assert core.plan_m4 is plan_m4


def test_m4_unproved_rewrite_is_never_admitted():
    spec = M4Input("x", (2,), dtype=torch.float64)
    graph = EGraph()
    graph.add_expr(tensor_input(spec))
    graph.register_rule(
        RewriteRule(
            "unproved_bad_rewrite",
            "test",
            "core_reproducible",
            (),
            lambda graph, class_id, node: (),
        )
    )
    with pytest.raises(UnprovedRewriteError, match="no registered proof"):
        graph.saturate()


def test_m4_square_output_residual_and_float_reassociation_counterexamples_remain():
    report = verify_core_z3()
    names = {item["name"] for item in report["counterexamples"]}
    assert "m4_square_output_only_residual_refuted" in names
    assert "m4_unconditional_bitwise_reassociation_refuted" in names
    assert all(item["refutation_verified"] for item in report["counterexamples"])

    x = torch.tensor(-2.0)
    assert x.square() == (-x).square()
    assert x != -x
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    c = torch.randn(5, 6)
    assert not torch.equal((a @ b) @ c, a @ (b @ c))


def test_m4_all_rule_proof_references_resolve():
    root = Path(__file__).resolve().parents[1]
    lean = (root / "lean").joinpath("OjasXMath.lean").read_text()
    lean += "\n".join(path.read_text() for path in (root / "lean/OjasXMath").glob("*.lean"))
    rocq = "\n".join(path.read_text() for path in (root / "rocq").glob("*.v"))
    z3_names = {item["name"] for item in verify_core_z3()["proofs"]}
    for rule in default_rewrite_rules():
        assert rule.proofs
        for proof in rule.proofs:
            if proof.system.value == "z3":
                assert proof.identifier in z3_names
            elif proof.system.value == "lean4":
                assert proof.identifier.split(".")[-1] in lean
            else:
                assert proof.identifier.split(".")[-1] in rocq


def test_m4_truncation_budget_and_signature_guards():
    problem = _chain_problem(["sigmoid"] * 4)
    limited = plan_m4(
        problem,
        config=M4Config(max_iterations=1, exact_node_limit=32),
    )
    assert not limited.optimal_for_searched_space
    with pytest.raises(MemoryBudgetError):
        plan_m4(
            problem,
            config=M4Config(
                max_iterations=8,
                exact_node_limit=32,
                budget=0,
            ),
        )
    with pytest.raises(ValueError, match="rank-two"):
        matmul(tensor_input(M4Input("v", (3,))), tensor_input(M4Input("m", (3, 2))))
    with pytest.raises(ValueError, match="inner|Incompatible"):
        matmul(
            tensor_input(M4Input("a", (2, 3))),
            tensor_input(M4Input("b", (4, 2))),
        )
