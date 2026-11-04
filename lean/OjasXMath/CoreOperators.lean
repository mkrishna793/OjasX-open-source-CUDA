import OjasXMath.Residuals
import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

/-! Exact real-algebra contracts for the initial demanded-pullback core.
IEEE NaN/signed-zero behavior is deliberately delegated to the separate SMT
model and differential tests; these theorems make no floating-point claim. -/

namespace OjasXMath
open scoped BigOperators

/-- The scalar affine pairing exposes precisely the three linear cotangents. -/
theorem core_linear_pairing (x w dx dw db g : ℝ) :
    g*(dx*w+x*dw+db) = dx*(g*w)+dw*(x*g)+db*g := by ring

/-- Addition duplicates its upstream cotangent and preserves adjoint pairing. -/
theorem core_add_pullback (a b g : ℝ) :
    g*(a+b) = a*g+b*g := by ring

/-- A finite reduction distributes its upstream cotangent over every term. -/
theorem core_sum_pullback {I : Type*} [Fintype I] (x : I → ℝ) (g : ℝ) :
    g*(∑ i, x i) = ∑ i, x i*g := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i hi
  ring

/-- Multiple uses of one value add cotangents before its own pullback. -/
theorem core_fanout_accumulation (x g h : ℝ) :
    x*g+x*h = x*(g+h) := by ring

/-- Normalized Taylor convolution at order four is Leibniz' product rule.
The aᵢ and bᵢ are raw derivatives; division by factorials gives normalized
coefficients. This is an exact real identity, not a floating-point claim. -/
theorem core_normalized_product_order_four
    (a0 a1 a2 a3 a4 b0 b1 b2 b3 b4 : ℝ) :
    24 * (a0*(b4/24) + a1*(b3/6) + (a2/2)*(b2/2) +
      (a3/6)*b1 + (a4/24)*b0) =
      a0*b4 + 4*a1*b3 + 6*a2*b2 + 4*a3*b1 + a4*b0 := by ring

/-- The sparse composition evaluator's Horner form is the truncated polynomial. -/
theorem core_jet_horner_order_four (d c0 c1 c2 c3 c4 : ℝ) :
    c0 + d*(c1 + d*(c2 + d*(c3 + d*c4))) =
      c0 + c1*d + c2*d^2 + c3*d^3 + c4*d^4 := by ring

/-- Output alone cannot determine a weight cotangent on colliding fibers. -/
theorem core_weight_output_only_insufficient (g : ℝ) (hg : g ≠ 0) :
    ¬ ∃ h : ℝ → ℝ, ∀ p : ℝ×ℝ, h (p.1*p.2) = p.1*g := by
  apply collision_forbids_output_only
    (fun p : ℝ×ℝ => p.1*p.2) (fun p => p.1*g) (0,1) (1,0)
  · norm_num
  · simpa using Ne.symm hg

/-- Output alone likewise cannot determine an input cotangent. -/
theorem core_input_output_only_insufficient (g : ℝ) (hg : g ≠ 0) :
    ¬ ∃ h : ℝ → ℝ, ∀ p : ℝ×ℝ, h (p.1*p.2) = p.2*g := by
  apply collision_forbids_output_only
    (fun p : ℝ×ℝ => p.1*p.2) (fun p => p.2*g) (0,1) (1,0)
  · norm_num
  · simpa using hg

end OjasXMath
