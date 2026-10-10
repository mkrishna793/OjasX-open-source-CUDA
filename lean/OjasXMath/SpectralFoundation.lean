import OjasXMath.AbstractEngine
import OjasXMath.GradientFactors
import OjasXMath.NonlinearBounds
import Mathlib.Analysis.Calculus.Deriv.Pow

namespace OjasXMath
noncomputable section

/- Approximation hypotheses are explicit obligations, never new axioms. -/
theorem spectral_polynomial_actual_derivative {S : Type*} [Fintype S]
    (coefficient : S → ℝ) (degree : S → ℕ) (z : ℝ) :
    HasDerivAt (fun x : ℝ => ∑ t, coefficient t*x^(degree t))
      (∑ t, coefficient t*(degree t : ℝ)*z^(degree t-1)) z := by
  have h (t : S) : HasDerivAt (fun x : ℝ => coefficient t*x^(degree t))
      (coefficient t*((degree t : ℝ)*z^(degree t-1))) z := by
    convert ((hasDerivAt_id z).pow (degree t)).const_mul (coefficient t) using 1 <;>
      (try funext x) <;> (try dsimp) <;> ring
  convert HasDerivAt.fun_sum (u := Finset.univ) (fun t _ => h t) using 1
  apply Finset.sum_congr rfl
  intro t _
  ring

theorem spectral_polynomial_tensor_forward {K N D : Type*}
    [Fintype K] [Fintype N] [Fintype D]
    (degree : D → ℕ) (coefficient : D → ℝ)
    (x : K → ℝ) (w : K → N → ℝ) (q : N → ℝ) :
    (∑ j, (∑ t, coefficient t*(∑ i, x i*w i j)^(degree t))*q j) =
      ∑ t, coefficient t*∑ f : Fin (degree t) → K,
        (∏ u, x (f u))*(∑ j, (∏ u, w (f u) j)*q j) := by
  simp_rw [Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro t _
  simp only [mul_assoc]
  rw [← Finset.mul_sum, homogeneous_feature_tensor]

theorem spectral_shared_projection {M N S R : Type*} [Fintype N] [Fintype S]
    (features : Matrix M S ℝ) (coefficients : Matrix S N ℝ) (q : Matrix N R ℝ) :
    (features*coefficients)*q = features*(coefficients*q) := Matrix.mul_assoc _ _ _

theorem spectral_shared_projection_gradient {M N S R : Type*} [Fintype M] [Fintype S]
    (features : Matrix M S ℝ) (coefficients : Matrix S N ℝ) (p : Matrix M R ℝ) :
    (features*coefficients).transpose*p = coefficients.transpose*(features.transpose*p) := by
  rw [Matrix.transpose_mul, Matrix.mul_assoc]

theorem spectral_bias_translation_derivative (f : ℝ → ℝ) (slope b w x : ℝ)
    (hf : HasDerivAt f slope (w*x+b)) :
    HasDerivAt (fun t : ℝ => f (w*x+t)) slope b := by
  convert hf.comp b ((hasDerivAt_const b (w*x)).add (hasDerivAt_id b)) using 1 <;>
    (try funext t) <;> (try dsimp) <;> ring

theorem spectral_projected_value_error {N : Type*} [Fintype N]
    (actual approx q : N → ℝ) (eps : ℝ)
    (h : ∀ j, |approx j-actual j| ≤ eps) :
    |∑ j, (approx j-actual j)*q j| ≤ eps*∑ j, |q j| :=
  weighted_residual_error Finset.univ actual approx q eps (fun j _ => h j)

theorem spectral_input_gradient_error {N : Type*} [Fintype N]
    (actual approx upstream w : N → ℝ) (eps : ℝ)
    (h : ∀ j, |approx j-actual j| ≤ eps) :
    |∑ j, (approx j-actual j)*(upstream j*w j)| ≤ eps*∑ j, |upstream j*w j| :=
  weighted_residual_error Finset.univ actual approx (fun j => upstream j*w j) eps (fun j _ => h j)

theorem spectral_weight_gradient_error {M : Type*} [Fintype M]
    (actual approx upstream x : M → ℝ) (eps : ℝ)
    (h : ∀ i, |approx i-actual i| ≤ eps) :
    |∑ i, (approx i-actual i)*(x i*upstream i)| ≤ eps*∑ i, |x i*upstream i| :=
  weighted_residual_error Finset.univ actual approx (fun i => x i*upstream i) eps (fun i _ => h i)

theorem spectral_bias_gradient_error {M : Type*} [Fintype M]
    (actual approx upstream : M → ℝ) (eps : ℝ)
    (h : ∀ i, |approx i-actual i| ≤ eps) :
    |∑ i, (approx i-actual i)*upstream i| ≤ eps*∑ i, |upstream i| :=
  weighted_residual_error Finset.univ actual approx upstream eps (fun i _ => h i)

theorem spectral_projection_gradient_error {M : Type*} [Fintype M]
    (actual approx upstream : M → ℝ) (eps : ℝ)
    (h : ∀ i, |approx i-actual i| ≤ eps) :
    |∑ i, (approx i-actual i)*upstream i| ≤ eps*∑ i, |upstream i| :=
  weighted_residual_error Finset.univ actual approx upstream eps (fun i _ => h i)

theorem spectral_zero_safe_monomial_pull (x : ℝ) (n : ℕ) :
    HasDerivAt (fun t : ℝ => t^n) ((n:ℝ)*x^(n-1)) x := by
  convert (hasDerivAt_id x).pow n using 1 <;> (try funext t) <;> simp

theorem spectral_loss_cotangent_error (g e p dp : ℝ) :
    (g+e)*(p+dp)-g*p=e*p+g*dp+e*dp := by ring

theorem spectral_relative_counterexample :
    (1:ℝ)/1000000 ≤ 1/10000 ∧
      (1:ℝ)/1000000 > (1/10000)*(1/100000000) := by norm_num

theorem spectral_feature_cost_guard (common candidate reference : ℕ) :
    common+(if candidate < reference then candidate else reference) ≤ common+reference := by
  split_ifs <;> omega

end
end OjasXMath
