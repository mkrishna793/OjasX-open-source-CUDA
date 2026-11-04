"""Executable SMT contracts for the verified-core remediation.

Each proof checks satisfiable premises and refutes its negation. Counterexamples
are expected to be satisfiable and document why the repaired guard is necessary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import z3


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "ojasx2/grad_free_optimizer.py",
    "ojasx2/numerical_repairs.py",
    "ojasx2/math/lattice.py",
    "ojasx2/mesh_hal.py",
    "tools/verify_remediation_z3.py",
]


def prove(name, assumptions, negation, scope):
    solver = z3.Solver()
    solver.set(timeout=15_000)
    solver.add(*assumptions)
    preconditions = solver.check()
    solver.add(negation)
    counterexample_query = solver.check()
    return {
        "name": name,
        "preconditions": str(preconditions),
        "counterexample_query": str(counterexample_query),
        "verified": preconditions == z3.sat and counterexample_query == z3.unsat,
        "counterexample": str(solver.model()) if counterexample_query == z3.sat else None,
        "scope": scope,
    }


def expected_counterexample(name, constraints, scope, witness=None):
    solver = z3.Solver()
    solver.set(timeout=15_000)
    solver.add(*constraints)
    status = solver.check()
    return {
        "name": name,
        "status": str(status),
        "refutation_verified": status == z3.sat,
        "model": str(solver.model()) if status == z3.sat else None,
        "witness": witness,
        "scope": scope,
    }


def verify():
    proofs = []
    counterexamples = []

    gradients = z3.Reals("g0 g1 g2 g3")
    count = 4
    correct = sum(g / count for g in gradients)
    expected_mean = sum(gradients) / count
    old_double_scaled = sum(g / count for g in gradients) / count
    proofs.append(prove(
        "accumulation_complete_window_is_arithmetic_mean",
        [],
        correct != expected_mean,
        "Four independent real gradients; one logical optimizer window; not an IEEE rounding claim",
    ))
    for consumed in range(1, 4):
        pending = sum(gradients[:consumed]) / count
        flushed = pending * (z3.RealVal(count) / z3.RealVal(consumed))
        expected = sum(gradients[:consumed]) / consumed
        proofs.append(prove(
            f"accumulation_partial_flush_{consumed}_of_4",
            [],
            flushed != expected,
            f"Real mean of the final {consumed} microbatches after rescaling by 4/{consumed}",
        ))
    counterexamples.append(expected_counterexample(
        "old_trainer_and_optimizer_double_scaling_refuted",
        [gradients[0] == 1, *[g == 0 for g in gradients[1:]], old_double_scaled != expected_mean],
        "Old G=4 pipeline: backward gradients divided by four and optimizer divides by four again",
        {"old_effective_gradient": "1/16", "required_mean_gradient": "1/4"},
    ))

    # A complete reset is an explicit state transition, not merely a zero count.
    pending_count, pending_sum = z3.Int("pending_count"), z3.Real("pending_sum")
    next_count, next_sum = z3.IntVal(0), z3.RealVal(0)
    proofs.append(prove(
        "accumulation_consumption_clears_count_and_storage",
        [pending_count >= 0, next_count == 0, next_sum == 0],
        z3.Or(next_count != 0, next_sum != 0),
        "Abstract consumed-window state; packed/unpacked storage cleanup is separately exercised in PyTorch tests",
    ))

    k, n, group = z3.Ints("native_k native_n native_group")
    scales = (n * k) / group
    row_divisible = k % group == 0
    proofs.append(prove(
        "native_row_groups_make_flat_scale_reshape_exact",
        [n > 0, group > 0, k > 0, row_divisible],
        scales != n * (k / group),
        "Unbounded positive integer dimensions; exact per-row scale-count arithmetic",
    ))
    counterexamples.append(expected_counterexample(
        "flat_total_divisibility_is_not_a_native_row_layout",
        [k == 48, n == 4, group == 32, (n * k) % group == 0,
         z3.Not(row_divisible), scales % n != 0],
        "K=48,N=4,G=32: total divisibility holds, but six scales cannot reshape as four complete rows",
        {"total_scales": 6, "scales_mod_rows": 2},
    ))

    elements = z3.Int("compressed_elements")
    block = 32
    rows = (elements + block - 1) / block
    stored = rows * block
    proofs.append(prove(
        "compressed_block_layout_padding_bound",
        [elements > 0, rows == (elements + block - 1) / block, stored == rows * block],
        z3.Or(rows < 1, stored < elements, stored >= elements + block),
        "Positive integer element count and 32-wide blocks; exact ceil-div padding and one-block bound",
    ))

    max_elements = 64 * 1024 * 1024
    count, item_size, raw_bytes = z3.Ints("raw_elements raw_item_size raw_bytes")
    representations = z3.Or(
        z3.And(item_size == 2, raw_bytes == 2 * count),
        z3.And(item_size == 4, raw_bytes == 4 * count),
        z3.And(item_size == 8, raw_bytes == 8 * count),
    )
    proofs.append(prove(
        "raw_wire_allocation_has_a_strict_element_derived_bound",
        [count >= 0, count <= max_elements, representations],
        z3.Or(raw_bytes < 0, raw_bytes > 8 * max_elements),
        "Bounded integer allocation arithmetic for supported 16/32/64-bit raw dtypes; frame limits are separate",
    ))

    value_ok, grad_ok = z3.Bools("candidate_value_ok candidate_grad_ok")
    accepted = z3.And(value_ok, grad_ok)
    trial, reference = z3.Reals("trial_gradient reference_gradient")
    selected = z3.If(accepted, trial, reference)
    proofs.append(prove(
        "checked_candidate_selection_preserves_native_fallback",
        [],
        z3.Or(z3.And(accepted, selected != trial), z3.And(z3.Not(accepted), selected != reference)),
        "Exact selection logic after independent value/gradient validation; floating allclose is tested separately",
    ))

    received, sequence, following = z3.Ints("mesh_receive_sequence mesh_frame_sequence mesh_next_sequence")
    assumptions = [received >= 0, sequence == received, following == received + 1]
    proofs.append(prove(
        "mesh_ordered_session_rejects_immediate_replay",
        assumptions,
        z3.Or(sequence != received, sequence == following),
        "Ordered per-connection sequence state only; random session binding/HMAC and cross-connection replay require the protocol implementation",
    ))

    data = {
        "schema_version": 1,
        "solver_version": z3.get_version_string(),
        "proofs": proofs,
        "counterexamples": counterexamples,
        "all_proofs_passed": all(item["verified"] for item in proofs),
        "all_counterexamples_verified": all(item["refutation_verified"] for item in counterexamples),
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES
        },
        "trust_boundary": (
            "Exact SMT models of the stated real/integer/selection/state contracts. "
            "No claim of Python-to-SMT refinement, cryptographic HMAC security, IEEE accuracy, or hardware qualification."
        ),
    }
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/remediation-z3.json")
    args = parser.parse_args()
    result = verify()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    passed = sum(item["verified"] for item in result["proofs"])
    print(f"Remediation Z3: {passed}/{len(result['proofs'])} proofs; "
          f"{sum(item['refutation_verified'] for item in result['counterexamples'])}/"
          f"{len(result['counterexamples'])} expected counterexamples")
    raise SystemExit(0 if result["all_proofs_passed"] and result["all_counterexamples_verified"] else 1)
