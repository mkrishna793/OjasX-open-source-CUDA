import Mathlib.Analysis.Calculus.Taylor
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-! Conditional real bounds; activation-specific analytic premises and IEEE
roundoff remain separate obligations. -/
namespace OjasXMath

theorem bounded_taylor_remainder (f : ℝ → ℝ) (x δ M : ℝ) (n : ℕ)
    (hx : 0 ≠ x) (hδ : 0 ≤ δ) (hM : 0 ≤ M) (habs : |x| ≤ δ)
    (hf : ContDiffOn ℝ (n+1) f (Set.uIcc 0 x))
    (hderiv : ∀ y ∈ Set.uIoo 0 x, |iteratedDeriv (n+1) f y| ≤ M) :
    |f x-taylorWithinEval f n (Set.uIcc 0 x) 0 x| ≤ M*δ^(n+1)/(Nat.factorial (n+1) : ℝ) := by
  obtain ⟨y, hy, heq⟩ := taylor_mean_remainder_lagrange_iteratedDeriv hx hf
  rw [heq, sub_zero, abs_div, abs_mul, abs_pow]
  have hn : (0:ℝ) < (Nat.factorial (n+1) : ℝ) := by positivity
  rw [abs_of_pos hn]
  apply div_le_div_of_nonneg_right _ (le_of_lt hn)
  exact mul_le_mul (hderiv y hy) (pow_le_pow_left₀ (abs_nonneg x) habs (n+1))
    (by positivity) hM

theorem quadratic_primal_remainder (f : ℝ → ℝ) (x δ M c : ℝ)
    (hx : 0 ≠ x) (hδ : 0 ≤ δ) (hM : 0 ≤ M) (habs : |x| ≤ δ)
    (hf : ContDiffOn ℝ 4 f (Set.uIcc 0 x))
    (hderiv : ∀ y ∈ Set.uIoo 0 x, |iteratedDeriv 4 f y| ≤ M)
    (hcoeff : taylorWithinEval f 3 (Set.uIcc 0 x) 0 x = x/2+c*x^2) :
    |f x-(x/2+c*x^2)| ≤ M*δ^4/24 := by
  have h := bounded_taylor_remainder f x δ M 3 hx hδ hM habs hf hderiv
  rw [hcoeff] at h
  norm_num [Nat.factorial] at h ⊢
  exact h

theorem quadratic_slope_remainder (f : ℝ → ℝ) (x δ M c : ℝ)
    (hx : 0 ≠ x) (hδ : 0 ≤ δ) (hM : 0 ≤ M) (habs : |x| ≤ δ)
    (hf : ContDiffOn ℝ 3 (deriv f) (Set.uIcc 0 x))
    (hderiv : ∀ y ∈ Set.uIoo 0 x, |iteratedDeriv 3 (deriv f) y| ≤ M)
    (hcoeff : taylorWithinEval (deriv f) 2 (Set.uIcc 0 x) 0 x = 1/2+2*c*x) :
    |deriv f x-(1/2+2*c*x)| ≤ M*δ^3/6 := by
  have h := bounded_taylor_remainder (deriv f) x δ M 2 hx hδ hM habs hf hderiv
  rw [hcoeff] at h
  norm_num [Nat.factorial] at h ⊢
  exact h

theorem weighted_residual_error {ι : Type*} (s : Finset ι)
    (actual approx weight : ι → ℝ) (ε : ℝ)
    (h : ∀ j ∈ s, |approx j-actual j| ≤ ε) :
    |∑ j ∈ s, (approx j-actual j)*weight j| ≤ ε*∑ j ∈ s, |weight j| := by
  calc
    _ ≤ ∑ j ∈ s, |(approx j-actual j)*weight j| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j ∈ s, ε*|weight j| := by
      apply Finset.sum_le_sum
      intro j hj
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_right (h j hj) (abs_nonneg _)
    _ = _ := by rw [Finset.mul_sum]

theorem sigmoid_factor_bounds (s : ℝ) (h0 : 0 ≤ s) (h1 : s ≤ 1) :
    |s*(1-s)*(1-6*(s*(1-s)))| ≤ 1/4 ∧
    |s*(1-s)*(1-2*s)*(1-12*(s*(1-s)))| ≤ 1/2 := by
  have ht0 : 0 ≤ s*(1-s) := mul_nonneg h0 (by linarith)
  have ht1 : s*(1-s) ≤ 1/4 := by nlinarith [sq_nonneg (s-1/2)]
  have ha : |1-6*(s*(1-s))| ≤ 1 := by rw [abs_le]; constructor <;> nlinarith
  have hb : |1-2*s| ≤ 1 := by rw [abs_le]; constructor <;> linarith
  have hc : |1-12*(s*(1-s))| ≤ 2 := by rw [abs_le]; constructor <;> nlinarith
  have ht : |s*(1-s)| ≤ 1/4 := by rwa [abs_of_nonneg ht0]
  constructor
  · rw [abs_mul]
    nlinarith [mul_le_mul ht ha (abs_nonneg _) (by norm_num : (0:ℝ) ≤ 1/4)]
  · have h := mul_le_mul ht hb (abs_nonneg _) (by norm_num : (0:ℝ) ≤ 1/4)
    have hh := mul_le_mul h hc (abs_nonneg _) (by norm_num : (0:ℝ) ≤ (1/4)*1)
    have result : |s*(1-s)| *|1-2*s| *|1-12*(s*(1-s))| ≤ 1/2 := by nlinarith
    simpa only [abs_mul] using result

theorem silu_fourth_polynomial_bound (s x δ : ℝ)
    (h0 : 0 ≤ s) (h1 : s ≤ 1) (hδ : 0 ≤ δ) (hx : |x| ≤ δ) :
    |4*(s*(1-s)*(1-6*(s*(1-s)))) +
      x*(s*(1-s)*(1-2*s)*(1-12*(s*(1-s))))| ≤ 1+δ/2 := by
  obtain ⟨ha,hb⟩ := sigmoid_factor_bounds s h0 h1
  calc
    _ ≤ |4*(s*(1-s)*(1-6*(s*(1-s))))| +
      |x*(s*(1-s)*(1-2*s)*(1-12*(s*(1-s))))| := abs_add_le _ _
    _ ≤ 4*(1/4)+δ*(1/2) := by
      have hfirst := mul_le_mul_of_nonneg_left ha (by norm_num : (0:ℝ) ≤ |4|)
      have hsecond := mul_le_mul hx hb (abs_nonneg _) hδ
      norm_num at hfirst
      simp only [abs_mul] at hfirst hsecond ⊢
      linarith
    _ = _ := by ring

end OjasXMath
