"""M2 alias-aware optimal scheduling and certificate-bound execution tests."""

from dataclasses import replace
import itertools
import json
import weakref

import pytest
import torch

from ojasx2.core import (
    CORE_REGISTRY,
    JointGraph,
    JointGraphBuilder,
    MemoryBudgetError,
    execute_joint_graph,
    plan_memory,
)


def _add_chain():
    builder = JointGraphBuilder()
    a = builder.input("a", (2, 3), requires_grad=True)
    b = builder.input("b", (2, 3), requires_grad=True)
    c = builder.input("c", (2, 3), requires_grad=True)
    ab = builder.forward("add", (a, b), name="ab")
    ac = builder.forward("add", (ab, c), name="ac")
    builder.request_input_gradients()
    builder.set_objective(ac)
    graph = builder.generate_reverse()
    tensors = {a: torch.randn(2, 3), b: torch.randn(2, 3), c: torch.randn(2, 3)}
    return graph, tensors


def _independent_schedule_peak(graph, order):
    """Direct use-count semantics, deliberately independent of core.memory."""

    consumers = {value_id: [] for value_id in graph.values}
    for node in graph.nodes:
        for value_id in node.inputs:
            consumers[value_id].append(node.id)
    total_uses = {
        value_id: len(consumers[value_id]) + int(value_id in graph.outputs)
        for value_id in graph.values
    }
    storage_bytes = {value.storage_id: value.storage_bytes for value in graph.values.values()}

    def live(completed):
        groups = set()
        for value_id, value in graph.values.items():
            consumed = sum(node_id in completed for node_id in consumers[value_id])
            produced = value.producer is None or value.producer in completed
            if produced and total_uses[value_id] - consumed > 0:
                groups.add(value.storage_id)
        return groups

    def size(groups):
        return sum(storage_bytes[group] for group in groups)

    completed = set()
    available = set(graph.inputs) | set(graph.seeds)
    peak = size(live(completed))
    for node_id in order:
        node = graph.node(node_id)
        if any(value_id not in available for value_id in node.inputs):
            return None
        before = live(completed)
        new_groups = {graph.values[value_id].storage_id for value_id in node.outputs} - before
        peak = max(peak, size(before) + size(new_groups) + node.workspace_bytes)
        available.update(node.outputs)
        completed.add(node_id)
        peak = max(peak, size(live(completed)))
    return peak


def test_subset_dp_matches_independent_exhaustive_optimality_check():
    graph, _ = _add_chain()
    certificate = plan_memory(graph, exact_node_limit=12)
    peaks = [
        peak
        for order in itertools.permutations(node.id for node in graph.nodes)
        if (peak := _independent_schedule_peak(graph, order)) is not None
    ]
    assert peaks
    assert certificate.optimal_for_managed_model
    assert certificate.peak_bytes == min(peaks)
    assert _independent_schedule_peak(graph, certificate.schedule) == certificate.peak_bytes

    greedy = plan_memory(graph, exact_node_limit=0)
    assert not greedy.optimal_for_managed_model
    assert greedy.method.startswith("deterministic_greedy")
    assert greedy.peak_bytes >= certificate.peak_bytes
    assert json.dumps(greedy.to_dict())


def test_exact_scheduler_has_no_finite_infinity_sentinel():
    huge = 1 << 80
    builder = JointGraphBuilder()
    left = builder.input("left", (huge, huge), requires_grad=True)
    right = builder.input("right", (huge, huge), requires_grad=True)
    output = builder.forward("add", (left, right))
    builder.request_input_gradients()
    builder.set_objective(output)
    certificate = plan_memory(builder.generate_reverse())
    assert certificate.peak_bytes > (1 << 120)


def test_memory_budget_is_checked_against_the_minimum_peak():
    graph, _ = _add_chain()
    certificate = plan_memory(graph)
    assert plan_memory(graph, budget=certificate.peak_bytes).schedule == certificate.schedule
    with pytest.raises(MemoryBudgetError) as error:
        plan_memory(graph, budget=certificate.peak_bytes - 1)
    assert error.value.required == certificate.peak_bytes
    assert error.value.budget == certificate.peak_bytes - 1
    assert error.value.optimal
    with pytest.raises(ValueError, match="non-negative"):
        plan_memory(graph, budget=-1)


def test_linear_residual_aliases_are_not_double_counted():
    builder = JointGraphBuilder()
    x = builder.input("x", (2, 3), requires_grad=True)
    weight = builder.input("w", (4, 3), requires_grad=True)
    bias = builder.input("bias", (4,), requires_grad=True)
    output = builder.forward("linear", (x, weight, bias))
    builder.request_gradient(weight)
    builder.set_objective(output)
    graph = builder.generate_reverse()
    certificate = plan_memory(graph)
    linear = graph.nodes[0]
    residual = graph.values[linear.residual_ids[0]]
    assert residual.role == "x"
    assert residual.storage_id == graph.values[x].storage_id
    assert certificate.steps[0].new_storage_bytes == graph.values[linear.outputs[0]].storage_bytes
    assert certificate.final_bytes == (
        graph.values[graph.objective].nbytes + graph.values[graph.gradient_targets[weight]].nbytes
    )


@pytest.mark.parametrize("request_both", [False, True])
def test_explicit_input_alias_group_is_counted_once(request_both):
    builder = JointGraphBuilder()
    left = builder.input("left", (2, 3), requires_grad=True)
    right = builder.input("right", (2, 3), requires_grad=True, alias_of=left)
    result = builder.forward("add", (left, right))
    builder.request_gradient(left)
    if request_both:
        builder.request_gradient(right)
    builder.set_objective(result)
    graph = builder.generate_reverse()
    certificate = plan_memory(graph)
    assert graph.values[left].storage_id == graph.values[right].storage_id
    assert certificate.steps[0].before_bytes == 24 + graph.values[graph.seeds[0]].nbytes
    tensor = torch.randn(2, 3)
    outputs = execute_joint_graph(graph, certificate, {left: tensor, right: tensor})
    reference = tensor.clone().requires_grad_(True)
    expected, = torch.autograd.grad((reference + reference).sum(), reference)
    torch.testing.assert_close(outputs[graph.gradient_targets[left]], expected)
    assert (right in graph.gradient_targets) == request_both
    if request_both:
        assert graph.gradient_targets[right] == graph.gradient_targets[left]
        torch.testing.assert_close(outputs[graph.gradient_targets[right]], expected)
    with pytest.raises(ValueError, match="declared as aliases"):
        execute_joint_graph(graph, certificate, {left: tensor, right: tensor.clone()})


def test_relu_workspace_accounts_for_explicit_mask_temporaries():
    for policy, backward_bytes in (("resident_input", 27), ("packed_sign", 32)):
        builder = JointGraphBuilder()
        x = builder.input("x", (9,), requires_grad=True)
        output = builder.forward("relu", (x,), residual_policy=policy)
        builder.request_gradient(x)
        builder.set_objective(output)
        graph = builder.generate_reverse()
        primal = graph.nodes[0]
        pullback = next(node for node in graph.nodes if node.kind == "pullback" and node.op_name == "relu")
        expected_forward = 0 if policy == "resident_input" else 68
        assert primal.workspace_bytes == expected_forward
        assert pullback.workspace_bytes == backward_bytes


def test_frozen_graph_has_no_reverse_or_residual_storage():
    builder = JointGraphBuilder()
    x = builder.input("x", (2, 3))
    weight = builder.input("w", (4, 3))
    output = builder.forward("linear", (x, weight))
    builder.set_objective(output)
    graph = builder.generate_reverse()
    certificate = plan_memory(graph)
    assert [node.kind for node in graph.nodes] == ["primal", "primal"]
    assert all(not node.residual_ids for node in graph.nodes)
    assert certificate.final_bytes == 4


def test_certificate_digest_and_schedule_are_checked_before_execution():
    graph, tensors = _add_chain()
    certificate = plan_memory(graph)
    with pytest.raises(ValueError, match="different graph"):
        execute_joint_graph(graph, replace(certificate, graph_digest="0" * 64), tensors)
    with pytest.raises(ValueError, match="not topological"):
        execute_joint_graph(graph, replace(certificate, schedule=tuple(reversed(certificate.schedule))), tensors)
    with pytest.raises(ValueError, match="liveness fields"):
        execute_joint_graph(graph, replace(certificate, peak_bytes=certificate.peak_bytes + 1), tensors)
    with pytest.raises(ValueError, match="liveness fields"):
        execute_joint_graph(graph, replace(certificate, steps=certificate.steps[:-1]), tensors)
    with pytest.raises(KeyError, match="Missing input"):
        execute_joint_graph(graph, certificate, {})


def test_semantic_graph_mutation_invalidates_certificate():
    graph, tensors = _add_chain()
    certificate = plan_memory(graph)
    graph.nodes[0].demand = (False, False)
    with pytest.raises(ValueError, match="different graph"):
        execute_joint_graph(graph, certificate, tensors)


def test_executor_builds_no_hidden_autograd_tape_and_checks_seed():
    graph, tensors = _add_chain()
    tensors = {value_id: tensor.requires_grad_(True) for value_id, tensor in tensors.items()}
    certificate = plan_memory(graph)
    outputs = execute_joint_graph(graph, certificate, tensors)
    assert all(not tensor.requires_grad for tensor in outputs.values())
    with pytest.raises(ValueError, match="scalar"):
        execute_joint_graph(graph, certificate, tensors, objective_upstream=torch.ones(2))
    with pytest.raises(ValueError, match="dtype"):
        execute_joint_graph(graph, certificate, tensors, objective_upstream=torch.ones((), dtype=torch.float64))


def test_executor_releases_last_use_before_later_node(monkeypatch):
    builder = JointGraphBuilder()
    x = builder.input("x", (7,))
    first = builder.forward("relu", (x,))
    second = builder.forward("relu", (first,))
    third = builder.forward("relu", (second,))
    builder.set_objective(third)
    graph = builder.generate_reverse()
    certificate = plan_memory(graph)

    operator = CORE_REGISTRY.get("relu")
    original = operator.forward
    references = []
    calls = 0

    def observed_forward(tensors, attrs, policy):
        nonlocal calls
        if calls == 2:
            assert references[0]() is None, "first activation survived its last use"
            assert references[1]() is not None, "current input was released too early"
        result = original(tensors, attrs, policy)
        references.append(weakref.ref(result[0]))
        calls += 1
        return result

    monkeypatch.setattr(operator, "forward", observed_forward)
    outputs = execute_joint_graph(graph, certificate, {x: torch.linspace(-1, 1, 7)})
    assert calls == 3
    assert outputs[graph.objective] is not None


def test_executor_releases_dead_primal_output_immediately(monkeypatch):
    builder = JointGraphBuilder()
    x = builder.input("x", (7,))
    builder.forward("relu", (x,), name="dead")
    live = builder.forward("relu", (x,), name="live")
    builder.set_objective(live)
    graph = builder.generate_reverse()
    certificate = plan_memory(graph, exact_node_limit=0)
    assert certificate.schedule[:2] == (0, 1)

    operator = CORE_REGISTRY.get("relu")
    original = operator.forward
    references = []
    calls = 0

    def observed_forward(tensors, attrs, policy):
        nonlocal calls
        if calls == 1:
            assert references[0]() is None, "dead primal output was retained"
        result = original(tensors, attrs, policy)
        references.append(weakref.ref(result[0]))
        calls += 1
        return result

    monkeypatch.setattr(operator, "forward", observed_forward)
    execute_joint_graph(graph, certificate, {x: torch.linspace(-1, 1, 7)})
    assert calls == 2


def test_empty_graph_is_rejected():
    with pytest.raises(ValueError):
        plan_memory(JointGraph())
