import OjasXMath.Nonlinear
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.Calculus.Deriv.Inv

/-! Small sufficient cores for radial nonlinearities. These are real-number
identities and analytic derivatives, not floating-point execution proofs. -/
namespace OjasXMath

theorem rms_moment_core {N K : Type*} [Fintype N] [Fintype K]
    (x : K → ℝ) (w : N → K → ℝ) (invn eps : ℝ) :
    invn * (∑ j, (∑ k, x k*w j k)^2) + eps =
      invn * (∑ k, ∑ l, x k*(∑ j, w j k*w j l)*x l) + eps := by
  rw [centered_weight_gram_moment]

theorem rms_forward_core {M K N P : Type*}
    [Fintype M] [Fintype K] [Fintype N]
    (x : Matrix M K ℝ) (a : Matrix K N ℝ) (b : Matrix N P ℝ)
    (r : Matrix M M ℝ) (gamma : Matrix N N ℝ) :
    (r*(x*a)*gamma)*b = (r*x)*(a*gamma*b) := by
  simp only [Matrix.mul_assoc]

theorem radial_normalization_derivative {q h : ℝ → ℝ} {q' h' t : ℝ}
    (hq : HasDerivAt q q' t) (hh : HasDerivAt h h' t) (hp : 0 < q t) :
    HasDerivAt (fun s => h s / Real.sqrt (q s))
      (h' / Real.sqrt (q t) - h t*q'/(2*(Real.sqrt (q t))^3)) t := by
  have hs : Real.sqrt (q t) ≠ 0 := ne_of_gt (Real.sqrt_pos.2 hp)
  convert hh.div (hq.sqrt (ne_of_gt hp)) hs using 1
  field_simp

theorem radial_rank_one_pullback {N : Type*} [Fintype N]
    (z v g : N → ℝ) (r c : ℝ) :
    (∑ i, (r*v i-c*z i*(∑ j, z j*v j))*g i) =
      ∑ i, v i*(r*g i-c*z i*(∑ j, z j*g j)) := by
  simp only [sub_mul, mul_sub, Finset.sum_sub_distrib]
  congr 1
  · apply Finset.sum_congr rfl; intro i _; ring
  · simp only [Finset.mul_sum, Finset.sum_mul]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl; intro i _
    apply Finset.sum_congr rfl; intro j _; ring

theorem rms_core_joint_leading_cost (m k n p : ℝ) :
    6*k*n*k+6*m*k*k+6*p*n*k+6*m*k*p =
      6*k*(k+p)*(n+m) := by ring

theorem rms_core_profitable_criterion (m k n p : ℝ)
    (hp : 0 < k+p) (hc : k*(n+m) < m*n) :
    6*k*(k+p)*(n+m) < 6*m*n*(k+p) := by
  have := mul_lt_mul_of_pos_right hc hp
  nlinarith

theorem paired_dagger_stream {M K N B : Type*}
    [Fintype K] [Fintype N]
    (x : Matrix M K ℝ) (g : Matrix B N ℝ) (w : Matrix K N ℝ) :
    (x*w, g*w.transpose).1 = x*w ∧
      (x*w, g*w.transpose).2 = g*w.transpose := by
  exact ⟨rfl, rfl⟩

end OjasXMath
