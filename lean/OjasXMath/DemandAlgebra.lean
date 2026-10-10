import OjasXMath.CotangentQuotient
namespace OjasXMath
open scoped BigOperators

theorem demand_linear_annihilator {E F : Type*}
    [AddCommGroup E] [Module ℝ E] [AddCommGroup F] [Module ℝ F]
    (pull : E →ₗ[ℝ] F) (project : E → E) (g : E)
    (h : project g = g) : pull (project g) = pull g := by rw [h]

theorem demand_support_pairing {I : Type*} [Fintype I] [DecidableEq I]
    (s : Finset I) (g dy : I → ℝ) (hz : ∀ i, i ∉ s → g i=0) :
    (∑ i, g i*dy i) = ∑ i ∈ s, g i*dy i := by
  exact defect_support_elision s g dy hz

theorem demand_zero_coordinate (slope cotangent : ℝ) (h : cotangent=0) :
    slope*cotangent=0 := by rw [h]; ring

theorem demand_nonlinear_zero_row {M N R : Type*} [Fintype R]
    (s : Matrix M N ℝ) (g : Matrix M R ℝ) (q : Matrix N R ℝ)
    (i : M) (hz : ∀ r, g i r=0) :
    ∀ j, s i j * (g*q.transpose) i j=0 := by
  intro j
  simp [Matrix.mul_apply,hz]

theorem demand_nonlinear_weight_sum {M : Type*} [Fintype M] [DecidableEq M]
    (support : Finset M) (x slope g : M → ℝ)
    (hz : ∀ i, i ∉ support → g i=0) :
    (∑ i, x i*(slope i*g i)) =
      ∑ i ∈ support, x i*(slope i*g i) := by
  symm
  apply Finset.sum_subset (Finset.subset_univ _)
  intro i hi hnot
  rw [hz i hnot]
  ring

theorem demand_actual_weighted_derivative {I : Type*} [Fintype I] [DecidableEq I]
    (support : Finset I) (f : I → ℝ → ℝ) (df g : I → ℝ) (x : ℝ)
    (hf : ∀ i, HasDerivAt (f i) (df i) x)
    (hz : ∀ i, i ∉ support → g i=0) :
    HasDerivAt (fun t => ∑ i, g i*f i t) (∑ i ∈ support, g i*df i) x := by
  rw [← demand_support_pairing support g df hz]
  apply HasDerivAt.fun_sum
  intro i hi
  exact (hf i).const_mul (g i)

theorem demand_two_sided_matrix {M R : Type*} [Fintype M] [Fintype R]
    (g : Matrix M R ℝ) (rows : Matrix M M ℝ) (cols : Matrix R R ℝ)
    (hr : rows*g=g) (hc : g*cols=g) : rows*g*cols=g := by rw [hr,hc]

theorem demand_right_contraction {M R N : Type*} [Fintype M] [Fintype R]
    (g : Matrix M R ℝ) (q : Matrix N R ℝ) (rows : Matrix M M ℝ)
    (cols : Matrix R R ℝ) (hr : rows*g=g) (hc : g*cols=g) :
    (rows*g*cols)*q.transpose=g*q.transpose := by
  rw [demand_two_sided_matrix g rows cols hr hc]

theorem demand_joint_scalar (x w q slope value dx dw db dq g : ℝ) :
    g*(slope*(dx*w+x*dw+db)*q+value*dq) =
    dx*(slope*g*q*w)+dw*(x*slope*g*q)+db*(slope*g*q)+dq*(value*g) := by ring

theorem demand_complete_cost (m k n r a c : ℝ) :
    (6*m*n*(k+r))-(2*m*n*(k+r)+6*a*n*k+4*a*n*c+2*m*r) =
    4*m*n*(k+r)-6*a*n*k-4*a*n*c-2*m*r := by ring

theorem demand_cost_admission (forward native candidate scan : ℝ)
    (h : candidate+scan < native) : forward+candidate+scan < forward+native := by linarith

theorem demand_error_pairing (a b g error bound : ℝ)
    (he : |a-b| ≤ error) (hg : |g| ≤ bound) : |a*g-b*g| ≤ error*bound := by
  have hnon : 0 ≤ error := le_trans (abs_nonneg _) he
  calc
    |a*g-b*g| = |a-b| * |g| := by rw [← sub_mul,abs_mul]
    _ ≤ error*bound := mul_le_mul he hg (abs_nonneg _) hnon

end OjasXMath
