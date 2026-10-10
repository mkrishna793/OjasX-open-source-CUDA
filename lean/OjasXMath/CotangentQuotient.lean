import OjasXMath.GradientFactors
import OjasXMath.ObjectiveCompiler

/-! Observable cotangents: quotient by the kernel of the requested pullback.
The projection and defect identities are dimension-general. Obtaining their
coefficients cheaply is an additional algorithmic obligation, not an axiom. -/
namespace OjasXMath
open scoped BigOperators

def observableEquivalent {E F : Type*} (observe : E → F) (a b : E) : Prop :=
  observe a = observe b

theorem observable_equivalence {E F : Type*} (observe : E → F) :
    Equivalence (observableEquivalent observe) := by
  constructor
  · intro a; rfl
  · intro a b h; exact h.symm
  · intro a b c h₁ h₂; exact h₁.trans h₂

theorem observable_downstream {E F G : Type*} (observe : E → F) (next : F → G)
    (a b : E) (h : observableEquivalent observe a b) :
    observableEquivalent (next ∘ observe) a b := congrArg next h

theorem observable_linear_kernel {E F : Type*}
    [AddCommGroup E] [Module ℝ E] [AddCommGroup F] [Module ℝ F]
    (observe : E →ₗ[ℝ] F) (a b : E) :
    observableEquivalent observe a b ↔ observe (a-b) = 0 := by
  simp [observableEquivalent, map_sub, sub_eq_zero]

def cotangentRepresentative {M N : Type*} [Fintype M] [Fintype N]
    (p : Matrix M M ℝ) (q : Matrix N N ℝ) (d : Matrix M N ℝ) : Matrix M N ℝ :=
  p*d+d*q-p*d*q

theorem quotient_preserves_weight {M K N : Type*}
    [Fintype M] [Fintype N]
    (x : Matrix M K ℝ) (p : Matrix M M ℝ) (q : Matrix N N ℝ) (d : Matrix M N ℝ)
    (hp : x.transpose*p=x.transpose) :
    x.transpose*cotangentRepresentative p q d = x.transpose*d := by
  simp only [cotangentRepresentative, Matrix.mul_sub, Matrix.mul_add]
  simp only [← Matrix.mul_assoc, hp]
  abel

theorem quotient_preserves_input {M K N : Type*}
    [Fintype M] [Fintype N]
    (w : Matrix K N ℝ) (p : Matrix M M ℝ) (q : Matrix N N ℝ) (d : Matrix M N ℝ)
    (hq : q*w.transpose=w.transpose) :
    cotangentRepresentative p q d*w.transpose = d*w.transpose := by
  simp only [cotangentRepresentative, Matrix.sub_mul, Matrix.add_mul]
  simp only [Matrix.mul_assoc, hq]
  abel

theorem quotient_two_factor_representation {M N A B : Type*}
    [Fintype M] [Fintype N] [Fintype A] [Fintype B]
    (u : Matrix M A ℝ) (a : Matrix A M ℝ)
    (b : Matrix N B ℝ) (v : Matrix B N ℝ) (d : Matrix M N ℝ) :
    cotangentRepresentative (u*a) (b*v) d =
      u*(a*d) + (d*b-u*(a*d*b))*v := by
  simp only [cotangentRepresentative, Matrix.sub_mul, Matrix.mul_assoc]
  abel

theorem quotient_factor_width (a b : ℕ) :
    Fintype.card (Fin a ⊕ Fin b) = a+b := by simp

theorem quotient_residual_invisible {M K N : Type*}
    [Fintype M] [Fintype N]
    (x : Matrix M K ℝ) (w : Matrix K N ℝ)
    (p : Matrix M M ℝ) (q : Matrix N N ℝ) (d : Matrix M N ℝ)
    (hp : x.transpose*p=x.transpose) (hq : q*w.transpose=w.transpose) :
    x.transpose*(d-cotangentRepresentative p q d)=0 ∧
      (d-cotangentRepresentative p q d)*w.transpose=0 := by
  constructor
  · rw [Matrix.mul_sub, quotient_preserves_weight x p q d hp]; simp
  · rw [Matrix.sub_mul, quotient_preserves_input w p q d hq]; simp

/- Arbitrary nonlinear primal and derivative defects. These are two separate
certificates; equal values alone do not establish equal derivatives. -/
theorem defect_actual_derivative (f : ℝ → ℝ) (x c slopeDefect : ℝ)
    (hf : HasDerivAt f (c+slopeDefect) x) :
    HasDerivAt (fun t => f t-c*t) slopeDefect x := by
  convert hf.sub ((hasDerivAt_id x).const_mul c) using 1 <;> first | rfl | simp

theorem defect_scalar_joint_dagger (x w dx dw db c e g : ℝ) :
    ((c+e)*(dx*w+x*dw+db))*g =
      dx*(c*g*w+e*g*w)+dw*(x*c*g+x*e*g)+db*(c*g+e*g) := by ring

theorem defect_projection_gradient {M K N R : Type*}
    [Fintype M] [Fintype K]
    (x : Matrix M K ℝ) (w : Matrix K N ℝ)
    (bias defect : Matrix M N ℝ) (g : Matrix M R ℝ) :
    (x*w+bias+defect).transpose*g =
      w.transpose*(x.transpose*g)+bias.transpose*g+defect.transpose*g := by
  simp only [Matrix.transpose_add, Matrix.add_mul, Matrix.transpose_mul, Matrix.mul_assoc]

theorem defect_input_gradient {M K N R : Type*}
    [Fintype N] [Fintype R]
    (w : Matrix K N ℝ) (g : Matrix M R ℝ) (q : Matrix N R ℝ)
    (defect : Matrix M N ℝ) :
    (g*q.transpose+defect)*w.transpose = g*(w*q).transpose+defect*w.transpose := by
  rw [Matrix.add_mul, factor_input_pullback]

theorem defect_weight_gradient {M K N R : Type*}
    [Fintype M] [Fintype R]
    (x : Matrix M K ℝ) (g : Matrix M R ℝ) (q : Matrix N R ℝ)
    (defect : Matrix M N ℝ) :
    x.transpose*(g*q.transpose+defect) = (x.transpose*g)*q.transpose+x.transpose*defect := by
  rw [Matrix.mul_add, factor_weight_pullback]

theorem defect_support_elision {I : Type*} [Fintype I] [DecidableEq I]
    (support : Finset I) (defect weight : I → ℝ)
    (hzero : ∀ i, i ∉ support → defect i=0) :
    (∑ i, defect i*weight i) = ∑ i ∈ support, defect i*weight i := by
  symm
  apply Finset.sum_subset (Finset.subset_univ _)
  intro i hi hnot
  rw [hzero i hnot, zero_mul]

theorem defect_joint_cost (m k n r e : ℝ) :
    (6*m*n*k+6*m*n*r) -
      (2*m*n*k+2*m*n*r+4*m*k*r+6*k*n*r+4*e*(k+r)) =
    4*m*n*(k+r)-4*m*k*r-6*k*n*r-4*e*(k+r) := by ring

theorem defect_cost_pays_certificate (native candidate certificate : ℝ)
    (h : candidate+certificate ≤ native) : candidate ≤ native-certificate := by linarith

theorem defect_dense_certificate_exists {I : Type*} [Fintype I]
    (value slope z c : I → ℝ) :
    (∀ i, value i=c i*z i+(value i-c i*z i)) ∧
    (∀ i, slope i=c i+(slope i-c i)) := by
  constructor <;> intro i <;> ring

/- Interval certificates include the selected native slope at kinks. They do
not assert classical differentiability at zero or at clipping endpoints. -/
theorem relu_positive_interval (lo z : ℝ) (hlo : 0 < lo) (hz : lo ≤ z) :
    max z 0 = z ∧ (if 0 < z then (1 : ℝ) else 0) = 1 := by
  have h : 0 < z := lt_of_lt_of_le hlo hz
  simp [max_eq_left (le_of_lt h), h]

theorem relu_nonpositive_interval (hi z : ℝ) (hhi : hi ≤ 0) (hz : z ≤ hi) :
    max z 0 = 0 ∧ (if 0 < z then (1 : ℝ) else 0) = 0 := by
  have h : z ≤ 0 := le_trans hz hhi
  simp [max_eq_right h, not_lt.mpr h]

theorem leaky_negative_interval (hi z a : ℝ) (hhi : hi ≤ 0) (hz : z ≤ hi) :
    (if 0 < z then z else a*z) = a*z ∧
      (if 0 < z then (1 : ℝ) else a) = a := by
  have h : ¬ 0 < z := not_lt.mpr (le_trans hz hhi)
  simp [h]

theorem clip_positive_interval (lo z : ℝ) (hlo : 1 < lo) (hz : lo ≤ z) :
    max (-1) (min z 1) = 1 ∧ (if -1 < z ∧ z < 1 then (1 : ℝ) else 0) = 0 := by
  have h : 1 ≤ z := le_of_lt (lt_of_lt_of_le hlo hz)
  constructor
  · rw [min_eq_right h]; norm_num
  · simp [not_lt.mpr h]

theorem clip_negative_interval (hi z : ℝ) (hhi : hi < -1) (hz : z ≤ hi) :
    max (-1) (min z 1) = -1 ∧ (if -1 < z ∧ z < 1 then (1 : ℝ) else 0) = 0 := by
  have h : z ≤ -1 := le_of_lt (lt_of_le_of_lt hz hhi)
  constructor
  · rw [min_eq_left (by linarith), max_eq_left h]
  · simp [not_lt.mpr h]

theorem clip_middle_interval (lo hi z : ℝ)
    (hlo : -1 < lo) (hhi : hi < 1) (hl : lo ≤ z) (hh : z ≤ hi) :
    max (-1) (min z 1) = z ∧ (if -1 < z ∧ z < 1 then (1 : ℝ) else 0) = 1 := by
  have h₁ : -1 < z := lt_of_lt_of_le hlo hl
  have h₂ : z < 1 := lt_of_le_of_lt hh hhi
  simp [min_eq_left (le_of_lt h₂), max_eq_right (le_of_lt h₁), h₁, h₂]

end OjasXMath
