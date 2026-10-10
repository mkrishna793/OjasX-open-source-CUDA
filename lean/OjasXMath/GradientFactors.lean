import OjasXMath.Contraction
import OjasXMath.CachedReverse
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma

/-! Exact-real, dimension-polymorphic activation/cotangent interaction.
Every factor is an explicit representation, not a rank-truncated approximation.
An actual primitive derivative is a hypothesis of the generic calculus rule.
ReLU's chosen derivative at zero is a separate numerical convention. -/
namespace OjasXMath
noncomputable section

theorem factor_activation_direction (φ : ℝ → ℝ) (a x b da dx db d : ℝ)
    (hφ : HasDerivAt φ d (a*x+b)) :
    HasDerivAt (fun t : ℝ => φ ((a+t*da)*(x+t*dx)+(b+t*db)))
      (d*(a*dx+da*x+db)) 0 :=
  cached_branch_direction_derivative φ a b x da db dx d hφ

theorem factor_activation_joint_dagger (a x da dx db d g : ℝ) :
    (d*(a*dx+da*x+db))*g = dx*(a*d*g)+da*(x*d*g)+db*(d*g) := by ring

theorem factor_hadamard_product {R S : Type*} [Fintype R] [Fintype S]
    (p q : R → ℝ) (s t : S → ℝ) :
    (∑ r, p r*q r)*(∑ h, s h*t h) =
      ∑ r, ∑ h, (p r*s h)*(q r*t h) := by
  simp only [Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro r _
  apply Finset.sum_congr rfl
  intro h _
  ring

/- Any finite batch and feature sizes: nonlinear derivative factors compose
with incoming gradient factors using a product index R × S. -/
theorem factor_nonlinear_matrix_closure {M N R S : Type*}
    [Fintype R] [Fintype S]
    (p : M → R → ℝ) (q : N → R → ℝ)
    (s : M → S → ℝ) (t : N → S → ℝ) (i : M) (j : N) :
    (∑ r, p i r*q j r)*(∑ h, s i h*t j h) =
      ∑ rh : R × S, (p i rh.1*s i rh.2)*(q j rh.1*t j rh.2) := by
  rw [Fintype.sum_prod_type]
  exact factor_hadamard_product _ _ _ _

theorem factor_square_slope {K : Type*} [Fintype K]
    (x w : K → ℝ) (b : ℝ) :
    2*((∑ k, x k*w k)+b) = (∑ k, x k*(2*w k))+2*b := by
  rw [mul_add, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro k _
  ring

theorem factor_bias_pullback {M N R : Type*} [Fintype M] [Fintype R]
    (p : M → R → ℝ) (q : N → R → ℝ) (j : N) :
    (∑ i, ∑ r, p i r*q j r) = ∑ r, (∑ i, p i r)*q j r := by
  rw [Finset.sum_comm]
  simp only [Finset.sum_mul]

theorem factor_input_pullback {M K N R : Type*} [Fintype N] [Fintype R]
    (w : Matrix K N ℝ) (p : Matrix M R ℝ) (q : Matrix N R ℝ) :
    (p*q.transpose)*w.transpose = p*(w*q).transpose :=
  factored_cotangent_input w p q

theorem factor_weight_pullback {M K N R : Type*} [Fintype M] [Fintype R]
    (x : Matrix M K ℝ) (p : Matrix M R ℝ) (q : Matrix N R ℝ) :
    x.transpose*(p*q.transpose) = (x.transpose*p)*q.transpose :=
  factored_cotangent_weight x p q

theorem factor_joint_linear_forward {M K N R : Type*} [Fintype K] [Fintype N]
    (x : Matrix M K ℝ) (w : Matrix K N ℝ) (q : Matrix N R ℝ)
    (bias : Matrix M N ℝ) :
    (x*w+bias)*q = x*(w*q)+bias*q := by
  rw [Matrix.add_mul, Matrix.mul_assoc]

theorem factor_joint_projection_weight {M K N R : Type*} [Fintype M] [Fintype K]
    (x : Matrix M K ℝ) (w : Matrix K N ℝ) (bias : Matrix M N ℝ)
    (p : Matrix M R ℝ) :
    (x*w+bias).transpose*p = w.transpose*(x.transpose*p)+bias.transpose*p := by
  rw [Matrix.transpose_add, Matrix.add_mul, Matrix.transpose_mul, Matrix.mul_assoc]

/- Algebraic work models are shape-conditioned, not GPU timing theorems. -/
theorem factor_width_product (r s : ℕ) : Fintype.card (Fin r × Fin s) = r*s := by simp

theorem factor_complete_cost_choice (common native candidate : ℕ) :
    common+(if candidate < native then candidate else native) ≤ common+native := by
  split_ifs with h <;> omega

theorem factor_shared_linear_cost_difference (m k n r : ℝ) :
    (2*m*n*r+4*m*n*k+2*m*n)-(4*k*r*(m+n)+(m+n)*r) =
    2*m*n*r+4*m*n*k+2*m*n-4*k*r*(m+n)-(m+n)*r := by ring

/- The explicit 2×2 rank-one impossibility witness used by the SMT search. -/
theorem factor_relu_rank_growth (a b c d : ℝ)
    (hac : a*c=1) (had : a*d=0) (hbc : b*c=0) (hbd : b*d=1) : False := by
  have he : (a*c)*(b*d)=(a*d)*(b*c) := by ring
  rw [hac, had, hbc, hbd] at he
  norm_num at he

end
end OjasXMath
