import OjasXMath.ObjectiveCompiler
import OjasXMath.Residuals
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-! Loss-directed backpropagation. These theorems use existing real analysis;
they introduce no axioms, floating-point claims or hardware speed claims. -/
namespace OjasXMath
open scoped BigOperators

theorem loss_partition_positive {I : Type*} [Fintype I] [Nonempty I] (z : I → ℝ) :
    0 < ∑ i, Real.exp (z i) := by
  exact Finset.sum_pos (fun i _ => Real.exp_pos (z i)) Finset.univ_nonempty

theorem loss_log_partition_derivative {I : Type*} [Fintype I] [Nonempty I]
    (z : I → ℝ → ℝ) (dz : I → ℝ) (t : ℝ)
    (hz : ∀ i, HasDerivAt (z i) (dz i) t) :
    HasDerivAt (fun x => Real.log (∑ i, Real.exp (z i x)))
      ((∑ i, Real.exp (z i t)*dz i)/(∑ i, Real.exp (z i t))) t := by
  have hs : HasDerivAt (fun x => ∑ i, Real.exp (z i x))
      (∑ i, Real.exp (z i t)*dz i) t := by
    apply HasDerivAt.fun_sum
    intro i hi
    exact (hz i).exp
  exact hs.log (ne_of_gt (loss_partition_positive (fun i => z i t)))

theorem loss_weighted_cross_entropy_primal {I : Type*} [Fintype I]
    (a z : I → ℝ) (partition : ℝ) :
    (∑ i, -(a i * (z i - partition))) =
      (∑ i, a i)*partition - ∑ i, a i*z i := by
  simp only [Finset.sum_mul, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i hi
  ring

theorem loss_weighted_cross_entropy_derivative {I : Type*} [Fintype I] [Nonempty I]
    (z : I → ℝ → ℝ) (dz a : I → ℝ) (t : ℝ)
    (hz : ∀ i, HasDerivAt (z i) (dz i) t) :
    HasDerivAt (fun x => (∑ i, a i)*Real.log (∑ i, Real.exp (z i x)) -
        ∑ i, a i*z i x)
      ((∑ i, a i)*((∑ i, Real.exp (z i t)*dz i)/(∑ i, Real.exp (z i t))) -
        ∑ i, a i*dz i) t := by
  apply HasDerivAt.sub ((loss_log_partition_derivative z dz t hz).const_mul _)
  apply HasDerivAt.fun_sum
  intro i hi
  exact (hz i).const_mul (a i)

theorem loss_cross_entropy_cotangent {I : Type*} [Fintype I]
    (p a dz : I → ℝ) :
    (∑ i, a i)*(∑ i, p i*dz i) - ∑ i, a i*dz i =
      ∑ i, ((∑ j, a j)*p i-a i)*dz i := by
  simp only [Finset.mul_sum, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i hi
  ring

theorem loss_cotangent_sum_zero {I : Type*} [Fintype I]
    (p a : I → ℝ) (hp : ∑ i, p i = 1) :
    (∑ i, ((∑ j, a j)*p i-a i)) = 0 := by
  rw [Finset.sum_sub_distrib, ← Finset.mul_sum, hp]
  ring

theorem loss_probability_cotangent_error (mass p phat a delta : ℝ)
    (he : |phat-p| ≤ delta) :
    |(mass*phat-a)-(mass*p-a)| ≤ |mass| * delta := by
  have hid : (mass*phat-a)-(mass*p-a) = mass*(phat-p) := by ring
  rw [hid, abs_mul]
  exact mul_le_mul_of_nonneg_left he (abs_nonneg mass)

theorem loss_dense_pullback_error {I : Type*} [Fintype I]
    (error weight : I → ℝ) (delta : ℝ) (he : ∀ i, |error i| ≤ delta) :
    |∑ i, error i*weight i| ≤ delta*(∑ i, |weight i|) := by
  calc
    _ ≤ ∑ i, |error i*weight i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, delta*|weight i| := by
      apply Finset.sum_le_sum
      intro i hi
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_right (he i) (abs_nonneg (weight i))
    _ = _ := (Finset.mul_sum _ _ _).symm

theorem loss_smoothed_weight_mass {I : Type*} [Fintype I] [DecidableEq I]
    (c : I → ℝ) (target : I) (epsilon invClasses : ℝ) :
    (∑ i, ((if i = target then (1-epsilon)*c target else 0) +
      epsilon*invClasses*c i)) =
      (1-epsilon)*c target + epsilon*invClasses*(∑ i, c i) := by
  simp [Finset.sum_add_distrib, Finset.mul_sum]

theorem loss_global_weight_normalizer (derivative normalizer : ℝ)
    (h : normalizer ≠ 0) :
    normalizer*(derivative/normalizer) = derivative := by
  field_simp

theorem loss_masked_joint_dagger {M K V : Type*}
    [Fintype M] [Fintype K] [Fintype V]
    (h dh : Matrix M K ℝ) (w dw : Matrix K V ℝ) (g : Matrix M V ℝ) :
    (∑ i, ∑ j, g i j*((dh*w+h*dw) i j)) =
      (∑ i, ∑ k, dh i k*(g*w.transpose) i k) +
      (∑ k, ∑ j, dw k j*(h.transpose*g) k j) :=
  matrix_product_pullback h w dh dw g

theorem loss_selected_cost (n a k v : ℝ) :
    6*n*k*v - 6*a*k*v = 6*(n-a)*k*v := by ring

theorem loss_replay_profitable (n a k v : ℝ) (hk : 0 < k) (hv : 0 < v)
    (ha : 4*a < 3*n) : 8*a*k*v < 6*n*k*v := by
  have h := mul_lt_mul_of_pos_right (mul_lt_mul_of_pos_right ha hk) hv
  nlinarith

theorem loss_replay_twice_work_reduction (n a k v : ℝ) (hk : 0 ≤ k) (hv : 0 ≤ v)
    (ha : 8*a ≤ 3*n) : 2*(8*a*k*v) ≤ 6*n*k*v := by
  have h := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right ha hk) hv
  nlinarith

theorem loss_chunk_storage (n b v : ℕ) (hb : b ≤ n) : b*v ≤ n*v :=
  Nat.mul_le_mul_right v hb

theorem loss_200b_bf16_parameter_bytes :
    (200000000000 : ℕ)*2 = 400000000000 := by norm_num

theorem loss_resident_state_lower_bound (parameters gradients moments master : ℕ) :
    parameters ≤ parameters+gradients+moments+master := by omega

end OjasXMath
