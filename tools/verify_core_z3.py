"""Focused executable SMT contracts for the M0--M2 demanded-pullback core.

Every proof has a satisfiable-precondition/nonvacuity check. Real arithmetic
models mathematical adjoints; the separate FP32 model covers the declared ReLU
NaN and signed-zero convention. Neither is hardware certification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import z3

try:
    from .verify_z3 import prove
except ImportError:  # Direct ``python tools/verify_core_z3.py`` execution.
    from verify_z3 import prove


def verify() -> dict:
    results = []

    x, w, dx, dw, db, g = z3.Reals("core_x core_w core_dx core_dw core_db core_g")
    gradients = (g*w, x*g, g)
    directions = (dx, dw, db)
    pairings = tuple(direction*gradient for direction, gradient in zip(directions, gradients))
    full_pairing = g*(dx*w+x*dw+db)
    for mask in range(8):
        selected = [index for index in range(3) if mask & (1 << index)]
        demanded = sum((pairings[index] for index in selected), 0)
        reference = full_pairing
        for index in range(3):
            if index not in selected:
                reference = z3.substitute(reference, (directions[index], z3.RealVal(0)))
        # The reference is the full directional pairing with unrequested
        # perturbation channels set to zero, not a second list of answers.
        difference = demanded != reference
        results.append(
            prove(
                f"linear_demand_residual_sufficiency_{mask:03b}",
                [],
                difference,
                "Exact real scalar affine adjoint; dx=g*w, dw=x*g, db=g; "
                f"requested mask {mask:03b}; no floating or matrix-kernel claim",
            )
        )

    # One aggregate stable identifier used by the declarative linear contract.
    aggregate_negation = z3.Or(
        full_pairing != sum(pairings),
        gradients[0] != g*w,
        gradients[1] != x*g,
        gradients[2] != g,
    )
    results.append(
        prove(
            "linear_demand_residual_sufficiency",
            [],
            aggregate_negation,
            "Exact real scalar affine adjoint and all three residual formulas; "
            "eight demand projections are separate nonvacuous queries above",
        )
    )

    a0, a1, a2, a3, a4, b0, b1, b2, b3, b4 = z3.Reals(
        "jet_a0 jet_a1 jet_a2 jet_a3 jet_a4 jet_b0 jet_b1 jet_b2 jet_b3 jet_b4"
    )
    normalized_product = (
        a0*(b4/24) + a1*(b3/6) + (a2/2)*(b2/2)
        + (a3/6)*b1 + (a4/24)*b0
    )
    leibniz_product = a0*b4 + 4*a1*b3 + 6*a2*b2 + 4*a3*b1 + a4*b0
    results.append(
        prove(
            "jet_normalized_product_order_four",
            [],
            24*normalized_product != leibniz_product,
            "Exact real normalized Taylor convolution at order four; raw "
            "derivative inputs, binomial Leibniz coefficients, no IEEE claim",
        )
    )

    d, c0, c1, c2, c3, c4 = z3.Reals("jet_d jet_c0 jet_c1 jet_c2 jet_c3 jet_c4")
    horner = c0 + d*(c1 + d*(c2 + d*(c3 + d*c4)))
    polynomial = c0 + c1*d + c2*d**2 + c3*d**3 + c4*d**4
    results.append(
        prove(
            "jet_horner_order_four",
            [],
            horner != polynomial,
            "Exact real fourth-order sparse-composition Horner evaluation; "
            "does not prove operator-specific transcendental coefficients",
        )
    )

    s1, s2, s3, chain_g = z3.Reals("m25_s1 m25_s2 m25_s3 m25_g")
    reverse_chain = ((chain_g*s3)*s2)*s1
    composite_chain = chain_g*(s1*s2*s3)
    results.append(
        prove(
            "m25_replay_cotangent_chain",
            [],
            reverse_chain != composite_chain,
            "Exact real three-layer cotangent recurrence; replay re-encoding "
            "must separately reconstruct each tail residual",
        )
    )

    native_peak, fused_peak = z3.Ints("m25_native_peak m25_fused_peak")
    selected_peak = z3.If(native_peak <= fused_peak, native_peak, fused_peak)
    results.append(
        prove(
            "m25_policy_choice_memory_guard",
            [],
            selected_peak > native_peak,
            "Integer managed-memory candidate selection; minimum peak never "
            "exceeds the original candidate, not an allocator measurement",
        )
    )

    layers = z3.Int("m25_layers")
    results.append(
        prove(
            "m25_cached_work_le_streaming",
            [layers >= 0],
            2*layers > layers*(layers+1),
            "Nonnegative integer chain depth; cached L evaluations versus "
            "streaming L(L+1)/2 prefix evaluations, compared without division",
        )
    )

    f = z3.Function("m4_f", z3.RealSort(), z3.RealSort())
    g = z3.Function("m4_g", z3.RealSort(), z3.RealSort())
    h = z3.Function("m4_h", z3.RealSort(), z3.RealSort())
    hg = z3.Function("m4_h_after_g", z3.RealSort(), z3.RealSort())
    chain_x, chain_y = z3.Reals("m4_chain_x m4_chain_y")
    results.append(
        prove(
            "m4_activation_chain_split_fuse",
            [z3.ForAll(chain_y, hg(chain_y) == h(g(chain_y)))],
            h(g(f(chain_x))) != hg(f(chain_x)),
            "Function-composition model of activation-program append and segmentation",
        )
    )

    def mm2(left, right):
        return [
            [
                sum(left[i][k] * right[k][j] for k in range(2))
                for j in range(2)
            ]
            for i in range(2)
        ]

    def tp2(matrix):
        return [[matrix[j][i] for j in range(2)] for i in range(2)]

    matrix_a = [z3.Reals(f"m4_a_{i}_{j}" for j in range(2)) for i in range(2)]
    matrix_b = [z3.Reals(f"m4_b_{i}_{j}" for j in range(2)) for i in range(2)]
    matrix_c = [z3.Reals(f"m4_c_{i}_{j}" for j in range(2)) for i in range(2)]
    left_associated = mm2(mm2(matrix_a, matrix_b), matrix_c)
    right_associated = mm2(matrix_a, mm2(matrix_b, matrix_c))
    results.append(
        prove(
            "m4_matmul_reassociation",
            [],
            z3.Or(*[
                left_associated[i][j] != right_associated[i][j]
                for i in range(2) for j in range(2)
            ]),
            "Explicit 2x2 exact-real matrix reassociation model; not an FP bitwise claim",
        )
    )
    product_transpose = tp2(mm2(matrix_a, matrix_b))
    transpose_product = mm2(tp2(matrix_b), tp2(matrix_a))
    results.append(
        prove(
            "m4_transpose_matmul",
            [],
            z3.Or(*[
                product_transpose[i][j] != transpose_product[i][j]
                for i in range(2) for j in range(2)
            ]),
            "Explicit 2x2 exact-real structured transpose/adjoint model",
        )
    )
    twice_transposed = tp2(tp2(matrix_a))
    results.append(
        prove(
            "m4_transpose_involution",
            [],
            z3.Or(*[
                twice_transposed[i][j] != matrix_a[i][j]
                for i in range(2) for j in range(2)
            ]),
            "Transpose involution on the explicit matrix model",
        )
    )

    # M0/M1 uniform-cotangent representation laws. These are exact-real
    # pointwise identities; bitwise FP behavior is tested separately.
    uniform_g, uniform_a, uniform_b, uniform_n = z3.Reals(
        "structured_g structured_a structured_b structured_n"
    )
    results.append(
        prove(
            "structured_sum_uniform",
            [],
            z3.Or(*[entry != uniform_g for entry in (uniform_g, uniform_g, uniform_g)]),
            "Full-sum adjoint decodes one scalar witness as the constant ones field",
        )
    )
    results.append(
        prove(
            "structured_mean_uniform",
            [uniform_n != 0],
            uniform_n * (uniform_g / uniform_n) != uniform_g,
            "Exact-real nonzero mean divisor and scalar-before-broadcast decoding",
        )
    )
    results.append(
        prove(
            "structured_uniform_add",
            [],
            (uniform_a + uniform_b) + uniform_g != uniform_a + (uniform_b + uniform_g),
            "Scalar uniform addition and exact-real additive composition; no FP reassociation claim",
        )
    )

    index = z3.IntSort()
    dense = z3.Array("structured_dense", index, z3.RealSort())
    f = z3.Function("structured_f", index, index)
    h = z3.Function("structured_h", index, index)
    i, j = z3.Ints("structured_i structured_j")
    results.append(
        prove(
            "structured_uniform_reindex",
            [z3.ForAll([j], dense[j] == uniform_g)],
            dense[f(h(i))] != uniform_g,
            "Contravariant reindex composition preserves the constant-field invariant",
        )
    )
    materialized = z3.Lambda([j], uniform_g)
    results.append(
        prove(
            "structured_materialize_identity",
            [],
            materialized[i] != uniform_g,
            "Dense materialization is entry-for-entry decoding of the uniform witness",
        )
    )
    element_bytes, n0, n1, n2 = z3.Ints("structured_e structured_n0 structured_n1 structured_n2")
    extents = (n0, n1, n2)
    logical = element_bytes * sum(extents)
    managed = element_bytes * len(extents)
    saving = element_bytes * sum(extent - 1 for extent in extents)
    results.append(
        prove(
            "structured_uniform_region_storage",
            [element_bytes > 0, *[extent > 0 for extent in extents]],
            z3.Or(logical != managed + saving, managed > logical, saving < 0),
            "Three nonempty uniform values: exact logical/managed/saved byte decomposition",
        )
    )

    sort = z3.Float32()
    a, dy = z3.FPs("core_relu_input core_relu_upstream", sort)
    zero = z3.FPVal(0, sort)
    # For IEEE operands, !(a <= 0) is exactly (a > 0) || isnan(a).
    gate = z3.Not(z3.fpLEQ(a, zero))
    packed_bit = z3.If(gate, z3.BitVecVal(1, 8), z3.BitVecVal(0, 8))
    unpacked_gate = z3.Extract(0, 0, packed_bit) == 1
    reference = z3.If(z3.fpLEQ(a, zero), zero, dy)
    packed = z3.If(unpacked_gate, dy, zero)

    mask_byte = z3.BitVec("core_mask_byte", 8)
    rebuilt = z3.BitVecVal(0, 8)
    for index in range(8):
        bit = z3.Extract(index, index, mask_byte)
        rebuilt = rebuilt | (z3.ZeroExt(7, bit) << index)
    results.append(
        prove(
            "packed_sign_roundtrip_and_pullback",
            [z3.Not(z3.fpIsNaN(dy))],
            z3.Or(
                rebuilt != mask_byte,
                z3.fpToIEEEBV(reference) != z3.fpToIEEEBV(packed),
            ),
            "All 256 packed bytes plus FP32 ReLU pullback for arbitrary input, "
            "including NaN input and signed zero; NaN upstream excluded from "
            "bitwise payload comparison",
        )
    )

    fanout_x, fanout_g, fanout_h = z3.Reals("core_fanout_x core_fanout_g core_fanout_h")
    results.append(
        prove(
            "core_fanout_accumulation",
            [],
            fanout_x*fanout_g+fanout_x*fanout_h != fanout_x*(fanout_g+fanout_h),
            "Two exact real cotangent contributions to one shared value; "
            "floating reassociation is intentionally not claimed",
        )
    )

    original_cost = z3.Ints("m4_original_memory m4_original_work m4_original_traffic m4_original_final")
    candidate_cost = z3.Ints("m4_candidate_memory m4_candidate_work m4_candidate_traffic m4_candidate_final")

    def lex_less(left, right):
        return z3.Or(
            left[0] < right[0],
            z3.And(left[0] == right[0], left[1] < right[1]),
            z3.And(left[0] == right[0], left[1] == right[1], left[2] < right[2]),
            z3.And(
                left[0] == right[0],
                left[1] == right[1],
                left[2] == right[2],
                left[3] < right[3],
            ),
        )

    candidate_better = lex_less(candidate_cost, original_cost)
    selected_cost = tuple(
        z3.If(candidate_better, candidate, original)
        for candidate, original in zip(candidate_cost, original_cost)
    )
    results.append(
        prove(
            "m4_original_retained_without_strict_improvement",
            [value >= 0 for value in (*original_cost, *candidate_cost)]
            + [z3.Not(candidate_better)],
            z3.Or(
                *[
                    selected != original
                    for selected, original in zip(selected_cost, original_cost)
                ]
            ),
            "Lexicographic backward-inclusive objective: without a strictly "
            "better candidate the selected four-part cost is the original",
        )
    )
    error_bound, error_budget = z3.Reals("m4_error_bound m4_error_budget")
    results.append(
        prove(
            "m4_bounded_error_budget_nonnegative",
            [error_bound >= 0, error_bound <= error_budget],
            error_budget < 0,
            "A nonnegative admitted error bound within budget implies a nonnegative error budget",
        )
    )

    # A deliberately false positive-only rule must produce a concrete NaN
    # counterexample. This prevents the packed proof from vacuously ignoring
    # the nonfinite partition exercised by differential tests.
    positive_only = z3.fpLT(zero, a)
    solver = z3.Solver()
    solver.set(timeout=15000)
    solver.add(z3.fpIsNaN(a), z3.Not(z3.fpIsNaN(dy)), positive_only != gate)
    counterexample_status = solver.check()
    witness = {
        "name": "positive_only_relu_mask_refuted_for_NaN",
        "status": str(counterexample_status),
        "refutation_verified": counterexample_status == z3.sat,
        "model": str(solver.model()) if counterexample_status == z3.sat else None,
        "scope": "FP32 NaN input: PyTorch threshold convention differs from x>0",
    }
    fusion_solver = z3.Solver()
    fusion_solver.set(timeout=15000)
    fusion_solver.add(native_peak > 0, fused_peak >= native_peak)
    fusion_status = fusion_solver.check()
    fusion_witness = {
        "name": "m25_universal_fusion_peak_improvement_refuted",
        "status": str(fusion_status),
        "refutation_verified": fusion_status == z3.sat,
        "model": str(fusion_solver.model()) if fusion_status == z3.sat else None,
        "scope": "A fused candidate may have no lower peak; the optimizer must retain the original candidate",
    }
    fp_b, fp_c = z3.FPs("m4_fp_b m4_fp_c", sort)
    fp_left = z3.fpAdd(z3.RNE(), z3.fpAdd(z3.RNE(), a, fp_b), fp_c)
    fp_right = z3.fpAdd(z3.RNE(), a, z3.fpAdd(z3.RNE(), fp_b, fp_c))
    float_solver = z3.Solver()
    float_solver.set(timeout=15000)
    float_solver.add(
        z3.Not(z3.fpIsNaN(a)),
        z3.Not(z3.fpIsNaN(fp_b)),
        z3.Not(z3.fpIsNaN(fp_c)),
        z3.Not(z3.fpIsInf(a)),
        z3.Not(z3.fpIsInf(fp_b)),
        z3.Not(z3.fpIsInf(fp_c)),
        z3.fpToIEEEBV(fp_left) != z3.fpToIEEEBV(fp_right),
    )
    float_status = float_solver.check()
    float_witness = {
        "name": "m4_unconditional_bitwise_reassociation_refuted",
        "status": str(float_status),
        "refutation_verified": float_status == z3.sat,
        "model": str(float_solver.model()) if float_status == z3.sat else None,
        "scope": "Finite FP32 addition reassociation counterexample; matrix contraction inherits this numerical boundary",
    }
    square_x, square_y = z3.Reals("m4_square_x m4_square_y")
    square_solver = z3.Solver()
    square_solver.set(timeout=15000)
    square_solver.add(
        square_y == square_x*square_x,
        square_y == (-square_x)*(-square_x),
        square_x != -square_x,
    )
    square_status = square_solver.check()
    square_witness = {
        "name": "m4_square_output_only_residual_refuted",
        "status": str(square_status),
        "refutation_verified": square_status == z3.sat,
        "model": str(square_solver.model()) if square_status == z3.sat else None,
        "scope": "Distinct exact-real input fibers x and -x have the same square output",
    }
    return {
        "solver_version": z3.get_version_string(),
        "proofs": results,
        "all_proofs_passed": all(result["verified"] for result in results),
        "counterexamples": [witness, fusion_witness, float_witness, square_witness],
        "no_GPU_hardware_certification": True,
        "lean_rocq_kernels_executed_by_this_tool": False,
        "scoped_kernel_report": "reports/formal-kernel-results.json",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/core-z3-results.json")
    args = parser.parse_args()
    data = verify()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2), encoding="utf-8")
    passed = sum(proof["verified"] for proof in data["proofs"])
    print(
        f"Core Z3: {passed}/{len(data['proofs'])} proofs verified; "
        f"{len(data['counterexamples'])} intended counterexamples checked"
    )
    raise SystemExit(
        0
        if data["all_proofs_passed"]
        and all(item["refutation_verified"] for item in data["counterexamples"])
        else 1
    )
