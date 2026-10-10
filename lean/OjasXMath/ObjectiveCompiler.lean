import OjasXMath.AbstractEngine
import OjasXMath.Contraction
import Mathlib.Analysis.Calculus.Deriv.Polynomial

/-! Objective-aware schedules. Exact real arithmetic, arbitrary finite tensor
sizes, actual derivatives for finite objectives, and explicit cost premises.
No statement asserts GPU rounding, callback purity or a global optimum. -/
namespace OjasXMath
open scoped BigOperators

def frobenius {M N : Type*} [Fintype M] [Fintype N]
    (a b : Matrix M N ℝ) : ℝ := ∑ i, ∑ j, a i j * b i j

theorem objective_pair_mul {M K N : Type*} [Fintype M] [Fintype K] [Fintype N]
    (a : Matrix M K ℝ) (b : Matrix K N ℝ) (y : Matrix M N ℝ) :
    frobenius (a*b) y = frobenius b (a.transpose*y) := by
  simp only [frobenius, Matrix.mul_apply, Matrix.transpose_apply, Finset.sum_mul,
    Finset.mul_sum]
  calc
    _ = ∑ i, ∑ k, ∑ j, a i k * b k j * y i j := by
      apply Finset.sum_congr rfl; intro i hi; rw [Finset.sum_comm]
    _ = ∑ k, ∑ j, ∑ i, a i k * b k j * y i j := by
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl; intro k hk; rw [Finset.sum_comm]
    _ = _ := by
      apply Finset.sum_congr rfl; intro k hk
      apply Finset.sum_congr rfl; intro j hj
      apply Finset.sum_congr rfl; intro i hi; ring

theorem objective_gram_energy {M K N : Type*} [Fintype M] [Fintype K] [Fintype N]
    (h : Matrix M K ℝ) (w : Matrix K N ℝ) :
    frobenius (h*w) (h*w) = frobenius (h.transpose*h) (w*w.transpose) := by
  rw [objective_pair_mul, ← Matrix.mul_assoc]
  simp only [frobenius, Matrix.mul_apply, Matrix.transpose_apply, Finset.mul_sum,
    Finset.sum_mul]
  apply Finset.sum_congr rfl; intro k hk
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl; intro l hl
  apply Finset.sum_congr rfl; intro j hj
  apply Finset.sum_congr rfl; intro i hi; ring

theorem objective_square_expansion {M N : Type*} [Fintype M] [Fintype N]
    (a y : Matrix M N ℝ) :
    frobenius (a-y) (a-y) = frobenius a a - 2*frobenius a y + frobenius y y := by
  simp only [frobenius, Matrix.sub_apply, Finset.mul_sum, ← Finset.sum_sub_distrib,
    ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro i hi
  apply Finset.sum_congr rfl; intro j hj; ring

theorem objective_quadratic_gram_primal {M K N : Type*}
    [Fintype M] [Fintype K] [Fintype N]
    (h : Matrix M K ℝ) (w : Matrix K N ℝ) (y : Matrix M N ℝ) :
    frobenius (h*w-y) (h*w-y) =
      frobenius (h.transpose*h) (w*w.transpose) -
      2*frobenius w (h.transpose*y) + frobenius y y := by
  rw [objective_square_expansion, objective_gram_energy, objective_pair_mul h w y]

theorem objective_quadratic_input {M K N : Type*}
    [Fintype K] [Fintype N]
    (h : Matrix M K ℝ) (w : Matrix K N ℝ) (y : Matrix M N ℝ) :
    (h*w-y)*w.transpose = h*(w*w.transpose)-y*w.transpose := by
  rw [Matrix.sub_mul, Matrix.mul_assoc]

theorem objective_quadratic_weight {M K N : Type*}
    [Fintype M] [Fintype K]
    (h : Matrix M K ℝ) (w : Matrix K N ℝ) (y : Matrix M N ℝ) :
    h.transpose*(h*w-y) = (h.transpose*h)*w-h.transpose*y := by
  rw [Matrix.mul_sub, Matrix.mul_assoc]

/- Actual directional derivative of the full quadratic objective. h,w,y are
arbitrary differentiable matrix-valued paths, including a nonlinear feature
network h(theta); all original parameters may occur in these paths. -/
theorem objective_affine_residual_derivative {K : Type*} [Fintype K]
    (h w : K → ℝ → ℝ) (dh dw : K → ℝ) (y : ℝ → ℝ) (dy t : ℝ)
    (hh : ∀ k, HasDerivAt (h k) (dh k) t)
    (hw : ∀ k, HasDerivAt (w k) (dw k) t) (hy : HasDerivAt y dy t) :
    HasDerivAt (fun s => (∑ i : K, (h i s * w i s)) - y s)
      ((∑ i : K, (dh i * w i t + h i t * dw i)) - dy) t := by
  apply HasDerivAt.sub _ hy
  apply HasDerivAt.fun_sum; intro k hk
  exact (hh k).mul (hw k)

theorem objective_quadratic_actual_derivative {M N : Type*} [Fintype M] [Fintype N]
    (r : M → N → ℝ → ℝ) (dr : M → N → ℝ) (t : ℝ)
    (hr : ∀ i j, HasDerivAt (r i j) (dr i j) t) :
    HasDerivAt (fun s => ∑ i, ∑ j, (r i j s)^2)
      (∑ i, ∑ j, 2*r i j t*dr i j) t := by
  apply HasDerivAt.fun_sum; intro i hi
  apply HasDerivAt.fun_sum; intro j hj
  convert (hr i j).pow 2 using 1 <;> simp

theorem objective_quadratic_joint_dagger {M K N : Type*}
    [Fintype M] [Fintype K] [Fintype N]
    (h dh : Matrix M K ℝ) (w dw : Matrix K N ℝ) (y : Matrix M N ℝ) :
    (∑ i, ∑ j, (2*(h*w-y) i j)*((dh*w+h*dw) i j)) =
      (∑ i, ∑ k, dh i k * ((2:ℝ) • (h*(w*w.transpose)-y*w.transpose)) i k) +
      (∑ k, ∑ j, dw k j * ((2:ℝ) • ((h.transpose*h)*w-h.transpose*y)) k j) := by
  have hp := matrix_product_pullback h w dh dw ((2:ℝ) • (h*w-y))
  simpa only [Matrix.smul_apply, smul_eq_mul, Matrix.smul_mul, Matrix.mul_smul,
    objective_quadratic_input, objective_quadratic_weight] using hp

section Separable
variable {E F I : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

theorem objective_finite_derivative (s : Finset I) (f : I → E → F)
    (df : I → E →L[ℝ] F) (x : E)
    (h : ∀ i ∈ s, HasFDerivAt (f i) (df i) x) :
    HasFDerivAt (fun z => ∑ i ∈ s, f i z) (∑ i ∈ s, df i) x :=
  HasFDerivAt.fun_sum h

theorem objective_weighted_derivative (s : Finset I) (f : I → E → F)
    (df : I → E →L[ℝ] F) (weight : I → ℝ) (x : E)
    (h : ∀ i ∈ s, HasFDerivAt (f i) (df i) x) :
    HasFDerivAt (fun z => ∑ i ∈ s, weight i • f i z)
      (∑ i ∈ s, weight i • df i) x := by
  apply HasFDerivAt.fun_sum; intro i hi
  exact (h i hi).const_smul (weight i)
end Separable

theorem objective_streamed_pullback {E F I : Type*}
    [NormedAddCommGroup E] [NormedAddCommGroup F]
    [InnerProductSpace ℝ E] [InnerProductSpace ℝ F] [CompleteSpace E] [CompleteSpace F]
    (s : Finset I) (df : I → E →L[ℝ] F) (g : F) :
    (∑ i ∈ s, df i).adjoint g = ∑ i ∈ s, (df i).adjoint g := by
  simp only [map_sum, ContinuousLinearMap.sum_apply]

theorem objective_zero_weight_elision {I : Type*} [DecidableEq I]
    (s : Finset I) (w f : I → ℝ) :
    (∑ i ∈ s.filter (fun i => w i ≠ 0), w i*f i) = ∑ i ∈ s, w i*f i := by
  classical
  apply Finset.sum_subset (Finset.filter_subset _ _)
  intro i hi hnot
  have hz : w i = 0 := by simpa [Finset.mem_filter, hi] using hnot
  simp [hz]

theorem objective_partition_sum {I : Type*} [DecidableEq I]
    (a b : Finset I) (h : Disjoint a b) (f : I → ℝ) :
    (∑ i ∈ a ∪ b, f i) = (∑ i ∈ a, f i)+(∑ i ∈ b, f i) :=
  Finset.sum_union h

theorem objective_global_normalizer (a b denominator : ℝ) :
    a/denominator + b/denominator = (a+b)/denominator := by ring

theorem objective_optimizer_once {E S : Type*} (update : E → E → S → E × S)
    (theta full streamed : E) (state : S) (same : streamed=full) :
    update theta streamed state = update theta full state := by rw [same]

theorem objective_gram_cost (m k n : ℝ) :
    (6*m*k*n)-(4*m*k*n+4*m*k*k+4*k*k*n) =
      2*k*(m*n-2*k*(m+n)) := by ring

theorem objective_gram_cost_improves (m k n : ℝ) (hk : 0<k)
    (guard : 2*k*(m+n)<m*n) :
    4*m*k*n+4*m*k*k+4*k*k*n < 6*m*k*n := by
  have pos := mul_pos (by linarith : 0<2*k) (sub_pos.mpr guard)
  rw [← objective_gram_cost] at pos
  linarith

theorem objective_gram_quarter_saving (m k n : ℝ)
    (hm : 0≤m) (hk : 0≤k) (hn : 0≤n)
    (hkm : 16*k≤m) (hkn : 16*k≤n) :
    4*(4*m*k*n+4*m*k*k+4*k*k*n) ≤ 3*(6*m*k*n) := by
  have h1 := mul_le_mul_of_nonneg_right hkm (mul_nonneg hk hn)
  have h2 := mul_le_mul_of_nonneg_right hkn (mul_nonneg hm hk)
  nlinarith

theorem objective_complete_step_saving (feature optimizer native candidate : ℝ)
    (h : candidate≤native) :
    feature+candidate+optimizer ≤ feature+native+optimizer := by linarith

theorem objective_accumulation_error {I : Type*} (s : Finset I)
    (actual reference error : I → ℝ)
    (h : ∀ i ∈ s, |actual i-reference i|≤error i) :
    |(∑ i ∈ s, actual i)-(∑ i ∈ s, reference i)|≤∑ i ∈ s, error i := by
  rw [← Finset.sum_sub_distrib]
  exact (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum h)

theorem objective_stream_memory_bound (persistent perExample batch chunk : Nat)
    (h : chunk ≤ batch) :
    persistent+chunk*perExample ≤ persistent+batch*perExample := by
  exact Nat.add_le_add_left (Nat.mul_le_mul_right perExample h) persistent

theorem objective_zero_weight_work_bound (active batch perExample overhead : ℝ)
    (guard : overhead ≤ (batch-active)*perExample) :
    active*perExample+overhead ≤ batch*perExample := by nlinarith

end OjasXMath
