"""Global M4 objective, numerical-admission, and fallback invariants."""

import pytest
import torch

from ojasx2.core import (
    AccuracyMode,
    EGraph,
    ENode,
    ExecutionObjective,
    M4Config,
    M4Input,
    M4Problem,
    ProofRef,
    ProofSystem,
    Rewrite,
    RewriteRule,
    UnprovedRewriteError,
    activation,
    activation_chain_expr,
    plan_m4,
    tensor_input,
)


def _problem(operations=("sigmoid", "tanh", "sin")):
    spec = M4Input("x", (8,), dtype=torch.float64)
    expression = activation_chain_expr(tensor_input(spec), list(operations))
    return M4Problem((spec,), expression)


def test_original_is_retained_when_discovered_alternatives_are_not_better():
    spec = M4Input("x", (12,), dtype=torch.float64)
    expression = activation(activation(tensor_input(spec), "sigmoid"), "tanh")
    plan = plan_m4(M4Problem((spec,), expression), config=M4Config(exact_node_limit=32))
    assert plan.generated_obligations
    assert not plan.discovered
    assert plan.selected_expression == plan.baseline_expression
    assert plan.selected_cost.key(plan.objective) == plan.baseline_cost.key(plan.objective)
    assert not plan.proof_obligations


def test_work_and_traffic_objectives_select_different_qualified_schedules():
    problem = _problem()
    memory = plan_m4(problem, config=M4Config(exact_node_limit=32))
    work = plan_m4(
        problem,
        config=M4Config(
            exact_node_limit=32,
            objective=ExecutionObjective(primary="work"),
        ),
    )
    traffic = plan_m4(
        problem,
        config=M4Config(
            exact_node_limit=32,
            objective=ExecutionObjective(primary="traffic"),
        ),
    )
    assert not memory.discovered
    assert work.selected_expression.policy == "replay_cached"
    assert work.certificate.work_units < work.baseline_certificate.work_units
    assert work.certificate.peak_bytes > work.baseline_certificate.peak_bytes
    assert traffic.selected_expression.policy == "replay_cached"
    assert traffic.certificate.traffic_bytes < traffic.baseline_certificate.traffic_bytes
    assert traffic.selected_cost.key(traffic.objective) < traffic.baseline_cost.key(traffic.objective)


def _bounded_replay_rule(bound=0.0, proofs=None):
    def apply(graph, class_id, node):
        if node.op != "activation_chain":
            return ()
        replacement = ENode(
            node.op,
            node.args,
            node.attrs,
            "replay_cached",
            node.shape,
            node.dtype,
        )
        return (Rewrite(replacement, node.policy, "replay_cached"),)

    return RewriteRule(
        "test_bounded_replay",
        "test_bounded",
        "bounded_error",
        (
            ProofRef(ProofSystem.Z3, "m25_replay_cotangent_chain"),
        )
        if proofs is None
        else proofs,
        apply,
        error_bound=bound,
    )


def test_bounded_error_requires_a_bound_within_budget_and_proofs():
    spec = M4Input("x", (8,), dtype=torch.float64)
    expression = activation_chain_expr(tensor_input(spec), ["sigmoid", "tanh"])
    blocked = EGraph(
        objective=ExecutionObjective(
            accuracy=AccuracyMode.BOUNDED_ERROR,
            error_budget=0.1,
        )
    )
    root = blocked.add_expr(expression)
    blocked.register_rule(_bounded_replay_rule(bound=0.25))
    blocked.saturate(max_iterations=2)
    assert not blocked.generated_obligations

    admitted = EGraph(
        objective=ExecutionObjective(
            accuracy=AccuracyMode.BOUNDED_ERROR,
            error_budget=0.0,
        )
    )
    root = admitted.add_expr(expression)
    admitted.register_rule(_bounded_replay_rule(bound=0.0))
    admitted.saturate(max_iterations=2)
    assert admitted.generated_obligations
    assert admitted.generated_obligations[0].error_bound == 0.0
    assert admitted.extract(root, local_cost=lambda node: (1, 1, 1))[0]

    unproved = EGraph(
        objective=ExecutionObjective(
            accuracy=AccuracyMode.BOUNDED_ERROR,
            error_budget=1.0,
        )
    )
    unproved.add_expr(expression)
    unproved.register_rule(_bounded_replay_rule(bound=0.5, proofs=()))
    with pytest.raises(UnprovedRewriteError):
        unproved.saturate(max_iterations=2)


def test_accuracy_and_objective_validation_rejects_unspecified_relaxation():
    with pytest.raises(ValueError, match="finite nonnegative"):
        ExecutionObjective(accuracy=AccuracyMode.BOUNDED_ERROR)
    with pytest.raises(ValueError, match="only in bounded-error"):
        ExecutionObjective(error_budget=0.1)
    with pytest.raises(ValueError, match="memory, work, or traffic"):
        ExecutionObjective(primary="speed")
    with pytest.raises(ValueError, match="Exact-real"):
        EGraph(
            allow_exact_real=True,
            objective=ExecutionObjective(accuracy=AccuracyMode.STRICT),
        )


def test_global_plan_reports_representation_residual_layout_and_obligations():
    plan = plan_m4(
        _problem(),
        config=M4Config(
            exact_node_limit=32,
            objective=ExecutionObjective(primary="work"),
        ),
    )
    report = plan.to_dict()
    assert report["objective"]["primary"] == "work"
    assert report["objective"]["accuracy"] == "reproducible"
    assert "structured_topology" in report
    assert "residual_policies" in report
    assert "layout_view_nodes" in report
    assert set(report["planner_proofs"]) == {"original_fallback", "bounded_error_budget"}
    assert report["selected_cost"]["memory_bytes"] == plan.certificate.peak_bytes
    assert report["proof_obligations"]
    assert all(item["proofs"] for item in report["proof_obligations"])
