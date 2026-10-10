import OjasXMath.CertifiedBackprop
import Mathlib.Analysis.InnerProductSpace.Adjoint
import Mathlib.Tactic.Linarith

/-! Abstract exact-real backpropagation. Input spaces can contain model inputs
and all trainable parameters together. No efficient implementation is implied
by a noncomputable operator or by the existence of its adjoint. -/
namespace OjasXMath
noncomputable section
variable {E F G S T : Type*}
variable [NormedAddCommGroup E] [NormedAddCommGroup F] [NormedAddCommGroup G]
variable [InnerProductSpace ℝ E] [InnerProductSpace ℝ F] [InnerProductSpace ℝ G]
variable [CompleteSpace E] [CompleteSpace F] [CompleteSpace G]

structure AbstractStage (E F S : Type*)
    [NormedAddCommGroup E] [NormedAddCommGroup F]
    [InnerProductSpace ℝ E] [InnerProductSpace ℝ F] where
  primal : E → F
  derivative : E → (E →L[ℝ] F)
  derivative_valid : ∀ x, HasFDerivAt primal (derivative x) x
  encode : E → S
  residualDerivative : S → (E →L[ℝ] F)
  sufficient : ∀ x, residualDerivative (encode x) = derivative x

def AbstractStage.reverse (f : AbstractStage E F S) (s : S) (g : F) : E :=
  (f.residualDerivative s).adjoint g

theorem abstract_dagger_valid (f : AbstractStage E F S) (x dx : E) (g : F) :
    inner ℝ dx (f.reverse (f.encode x) g) = inner ℝ (f.derivative x dx) g := by
  simp only [AbstractStage.reverse, f.sufficient]
  exact ContinuousLinearMap.adjoint_inner_right _ _ _

def AbstractStage.compose (f : AbstractStage E F S) (h : AbstractStage F G T) :
    AbstractStage E G (S × T) where
  primal := h.primal ∘ f.primal
  derivative x := (h.derivative (f.primal x)).comp (f.derivative x)
  derivative_valid x := (h.derivative_valid (f.primal x)).comp x (f.derivative_valid x)
  encode x := (f.encode x,h.encode (f.primal x))
  residualDerivative st := (h.residualDerivative st.2).comp (f.residualDerivative st.1)
  sufficient x := by simp only [f.sufficient,h.sufficient]

theorem abstract_reverse_chain (f : AbstractStage E F S) (h : AbstractStage F G T)
    (s : S) (t : T) (g : G) :
    (f.compose h).reverse (s,t) g = f.reverse s (h.reverse t g) := by
  simp [AbstractStage.reverse, AbstractStage.compose, ContinuousLinearMap.adjoint_comp]

theorem abstract_reverse_accumulation (f : AbstractStage E F S) (s : S) (g h : F) :
    f.reverse s (g+h) = f.reverse s g + f.reverse s h := by
  exact map_add _ _ _

theorem abstract_rewrite_preserves_all_gradients (f : AbstractStage E F S)
    (h : AbstractStage E F T) (same : f.primal = h.primal) (x : E) :
    (f.derivative x).adjoint = (h.derivative x).adjoint := by
  have hf := f.derivative_valid x
  rw [same] at hf
  exact congrArg ContinuousLinearMap.adjoint (hf.unique (h.derivative_valid x))

theorem abstract_rewrite_preserves_reverse (f : AbstractStage E F S)
    (h : AbstractStage E F T) (same : f.primal = h.primal) (x : E) (g : F) :
    f.reverse (f.encode x) g = h.reverse (h.encode x) g := by
  simp only [AbstractStage.reverse, f.sufficient, h.sufficient]
  exact congrArg (fun a : F →L[ℝ] E => a g)
    (abstract_rewrite_preserves_all_gradients f h same x)

omit [CompleteSpace E] [CompleteSpace F] in
theorem exact_primal_rewrite_derivative (f h : E → F)
    (same : f = h) (d : E →L[ℝ] F) (x : E) (hf : HasFDerivAt f d x) :
    HasFDerivAt h d x := by rwa [same] at hf

theorem abstract_sgd_preserved (f : AbstractStage E F S) (h : AbstractStage E F T)
    (same : f.primal = h.primal) (x : E) (g : F) (eta : ℝ) :
    x - eta • f.reverse (f.encode x) g = x - eta • h.reverse (h.encode x) g := by
  rw [abstract_rewrite_preserves_reverse f h same]

theorem abstract_optimizer_preserved {State : Type*}
    (update : E → E → State → E × State)
    (f : AbstractStage E F S) (h : AbstractStage E F T)
    (same : f.primal = h.primal) (x : E) (g : F) (state : State) :
    update x (f.reverse (f.encode x) g) state =
      update x (h.reverse (h.encode x) g) state := by
  rw [abstract_rewrite_preserves_reverse f h same]

omit [CompleteSpace E] in
theorem abstract_sgd_error (x a b : E) (eta delta : ℝ)
    (h : ‖a-b‖ ≤ delta) :
    ‖(x-eta • a)-(x-eta • b)‖ ≤ |eta| * delta := by
  calc
    _ = ‖eta • (b-a)‖ := by rw [smul_sub]; congr 1; abel
    _ = |eta| * ‖a-b‖ := by rw [norm_smul, Real.norm_eq_abs, norm_sub_rev]
    _ ≤ |eta| * delta := mul_le_mul_of_nonneg_left h (abs_nonneg eta)

end

theorem rounded_add_error (a b ah bh rounded ea eb localError : ℝ)
    (ha : |ah-a| ≤ ea) (hb : |bh-b| ≤ eb)
    (hr : |rounded-(ah+bh)| ≤ localError) :
    |rounded-(a+b)| ≤ localError+ea+eb := by
  calc
    _ = |(rounded-(ah+bh))+((ah-a)+(bh-b))| := by congr 1; ring
    _ ≤ |rounded-(ah+bh)| + |(ah-a)+(bh-b)| := abs_add_le _ _
    _ ≤ localError+(ea+eb) := add_le_add hr ((abs_add_le _ _).trans (add_le_add ha hb))
    _ = _ := by ring

theorem rounded_mul_error (a b ah bh rounded ea eb localError : ℝ)
    (ha : |ah-a| ≤ ea) (hb : |bh-b| ≤ eb)
    (hr : |rounded-ah*bh| ≤ localError) :
    |rounded-a*b| ≤ localError + |ah| * eb + |b| * ea := by
  calc
    _ = |(rounded-ah*bh)+(ah*(bh-b)+(ah-a)*b)| := by congr 1; ring
    _ ≤ |rounded-ah*bh| + |ah*(bh-b)+(ah-a)*b| := abs_add_le _ _
    _ ≤ localError+(|ah*(bh-b)| + |(ah-a)*b|) :=
      add_le_add hr (abs_add_le _ _)
    _ = localError+(|ah| * |bh-b| + |b| * |ah-a|) := by rw [abs_mul,abs_mul]; ring
    _ ≤ localError+(|ah| * eb + |b| * ea) :=
      add_le_add (le_refl localError) (add_le_add (mul_le_mul_of_nonneg_left hb (abs_nonneg ah))
        (mul_le_mul_of_nonneg_left ha (abs_nonneg b)))
    _ = _ := by ring

/- Complete additive arithmetic model. Residual words and transient workspace
must be checked separately: adding costs does not prove allocator peak memory. -/
structure ArithmeticCost where
  multiplications : Nat
  additions : Nat
deriving DecidableEq

def ArithmeticCost.total (c : ArithmeticCost) := c.multiplications+c.additions
def ArithmeticCost.withOptimizer (c o : ArithmeticCost) : ArithmeticCost :=
  ⟨c.multiplications+o.multiplications,c.additions+o.additions⟩

theorem complete_step_cost_guard (c native optimizer : ArithmeticCost)
    (hc : c.total ≤ native.total) :
    (c.withOptimizer optimizer).total ≤ (native.withOptimizer optimizer).total := by
  simp only [ArithmeticCost.total, ArithmeticCost.withOptimizer] at *
  omega

theorem complete_step_cost_choice (c native optimizer : ArithmeticCost) :
    ((if c.total ≤ native.total then c else native).withOptimizer optimizer).total ≤
      (native.withOptimizer optimizer).total := by
  split_ifs with h
  · exact complete_step_cost_guard c native optimizer h
  · exact Nat.le_refl _

/- Finite feature contraction for a nonlinear square block. No low-rank
assumption is made about the wide matrix; the input feature dimension grows. -/
theorem square_feature_contraction {K N : Type*} [Fintype K] [Fintype N]
    (x : K → ℝ) (a : K → N → ℝ) (b : N → ℝ) :
    (∑ j, (∑ i, x i*a i j)^2*b j) =
      ∑ i, ∑ l, x i*x l*(∑ j, a i j*a l j*b j) := by
  simp only [pow_two, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i hi
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro l hl
  apply Finset.sum_congr rfl
  intro j hj
  ring

theorem square_feature_zero_safe (x a b : ℝ) :
    (x*a)^2*b = x^2*(a^2*b) := by ring

theorem homogeneous_feature_tensor {K N : Type*} [Fintype K] [Fintype N]
    (degree : Nat) (x : K → ℝ) (a : K → N → ℝ) (b : N → ℝ) :
    (∑ j, (∑ i, x i*a i j)^degree*b j) =
      ∑ f : Fin degree → K, (∏ t, x (f t))*(∑ j, (∏ t, a (f t) j)*b j) := by
  have expand (j : N) : (∑ i, x i*a i j)^degree =
      ∑ f : Fin degree → K, ∏ t, x (f t)*a (f t) j := by
    calc
      _ = ∏ _t : Fin degree, ∑ i, x i*a i j := by simp
      _ = _ := Fintype.prod_sum (fun (_t : Fin degree) (i : K) => x i*a i j)
  simp_rw [expand, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro f hf
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j hj
  rw [Finset.prod_mul_distrib]
  ring

end OjasXMath
