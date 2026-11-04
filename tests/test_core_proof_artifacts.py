"""Resolve declarative proof targets and execute their available SMT checks."""

import json
from pathlib import Path
import re

from ojasx2.core import CORE_REGISTRY, PLANNER_PROOFS, REPRESENTATION_RULES, ProofSystem
from tools.verify_core_z3 import verify as verify_core_z3


def test_declared_proof_targets_exist_and_z3_queries_pass():
    root = Path(__file__).resolve().parents[1]
    lean = {
        f"OjasXMath.{name}"
        for path in (root / "lean/OjasXMath").glob("*.lean")
        for name in re.findall(r"^theorem\s+(\w+)", path.read_text(encoding="utf-8"), re.M)
    }
    rocq = {
        f"Ojasx.{path.stem}.{name}"
        for path in (root / "rocq").glob("*.v")
        for name in re.findall(r"^(?:Theorem|Lemma)\s+(\w+)", path.read_text(encoding="utf-8"), re.M)
    }
    report = verify_core_z3()
    assert report["all_proofs_passed"]
    assert all(result["preconditions"] == "sat" for result in report["proofs"])
    assert all(item["refutation_verified"] for item in report["counterexamples"])
    z3 = {result["name"] for result in report["proofs"]}

    references = [
        proof
        for rule in REPRESENTATION_RULES
        for proof in rule.proofs
    ]
    references.extend(
        proof
        for proofs in PLANNER_PROOFS.values()
        for proof in proofs
    )
    for contract in CORE_REGISTRY.snapshot().values():
        references.extend(contract.proofs)
        for policy in contract.residual_policies:
            references.extend(policy.proofs)
    assert references
    for reference in references:
        target = {
            ProofSystem.LEAN: lean,
            ProofSystem.ROCQ: rocq,
            ProofSystem.Z3: z3,
        }[reference.system]
        assert reference.identifier in target, f"Unresolved proof target {reference.identifier}"

    # Source resolution and SMT are not themselves Lean/Rocq kernel execution.
    # Scoped kernel runs are persisted by verify_formal_kernels.py separately.
    assert not report["lean_rocq_kernels_executed_by_this_tool"]
    assert report["scoped_kernel_report"] == "reports/formal-kernel-results.json"


def test_formal_kernel_results_report_validity():
    root = Path(__file__).resolve().parents[1]
    report_file = root / "reports/formal-kernel-results.json"
    assert report_file.is_file(), f"Missing formal kernel report at {report_file}"
    data = json.loads(report_file.read_text(encoding="utf-8"))
    assert data["status"] == "passed"
    assert data["scope"]["negative_controls"] is True
    assert data["scope"]["rocq_assumption_audit"] is True
    assert len(data["checks"]) >= 8
    assert all(c["passed"] for c in data["checks"])

    # Verify both Rocq and Lean negative controls are present and passed
    negatives = [c for c in data["checks"] if c.get("negative_control")]
    assert len(negatives) >= 2
    assert all(c["passed"] for c in negatives)
