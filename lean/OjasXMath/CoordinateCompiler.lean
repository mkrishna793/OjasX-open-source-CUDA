import OjasXMath.AbstractEngine
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
namespace OjasXMath

def coordRepeat {E : Type*} (f : E → E) : ℕ → E → E
  | 0, x => x
  | n+1, x => f (coordRepeat f n x)

theorem coordinate_domain_iteration {E : Type*} (f : E → E) (D : Set E)
    (stay : ∀ x ∈ D, f x ∈ D) (n : ℕ) (x : E) (hx : x ∈ D) :
    coordRepeat f n x ∈ D := by
  induction n with
  | zero => exact hx
  | succ n ih => exact stay _ ih

theorem coordinate_closed_iteration {E U : Type*} (f : E → E) (g : U → U)
    (encode : E → U) (D : Set E) (stay : ∀ x ∈ D, f x ∈ D)
    (closed : ∀ x ∈ D, encode (f x) = g (encode x))
    (n : ℕ) (x : E) (hx : x ∈ D) :
    encode (coordRepeat f n x) = coordRepeat g n (encode x) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [coordRepeat]
    rw [closed _ (coordinate_domain_iteration f D stay n x hx), ih]

theorem coordinate_decoded_iteration {E U : Type*} (f : E → E) (g : U → U)
    (encode : E → U) (decode : U → E) (D : Set E)
    (stay : ∀ x ∈ D, f x ∈ D) (closed : ∀ x ∈ D, encode (f x) = g (encode x))
    (recover : ∀ x ∈ D, decode (encode x) = x)
    (n : ℕ) (x : E) (hx : x ∈ D) :
    decode (coordRepeat g n (encode x)) = coordRepeat f n x := by
  rw [← coordinate_closed_iteration f g encode D stay closed n x hx]
  exact recover _ (coordinate_domain_iteration f D stay n x hx)

def coordSequence {P E : Type*} (f : P → E → E) : List P → E → E
  | [], x => x
  | p::ps, x => coordSequence f ps (f p x)

theorem coordinate_sequence_domain {P E : Type*} (f : P → E → E) (D : Set E)
    (stay : ∀ p x, x ∈ D → f p x ∈ D) (ps : List P) (x : E) (hx : x ∈ D) :
    coordSequence f ps x ∈ D := by
  induction ps generalizing x with
  | nil => exact hx
  | cons p ps ih => exact ih _ (stay p x hx)

theorem coordinate_sequence_closed {P E U : Type*} (f : P → E → E) (g : P → U → U)
    (encode : E → U) (D : Set E) (stay : ∀ p x, x ∈ D → f p x ∈ D)
    (closed : ∀ p x, x ∈ D → encode (f p x) = g p (encode x))
    (ps : List P) (x : E) (hx : x ∈ D) :
    encode (coordSequence f ps x) = coordSequence g ps (encode x) := by
  induction ps generalizing x with
  | nil => rfl
  | cons p ps ih =>
    simp only [coordSequence]
    rw [ih _ (stay p x hx), closed p x hx]

theorem coordinate_sequence_decoded {P E U : Type*} (f : P → E → E) (g : P → U → U)
    (encode : E → U) (decode : U → E) (D : Set E)
    (stay : ∀ p x, x ∈ D → f p x ∈ D)
    (closed : ∀ p x, x ∈ D → encode (f p x) = g p (encode x))
    (recover : ∀ x ∈ D, decode (encode x) = x) (ps : List P) (x : E) (hx : x ∈ D) :
    decode (coordSequence g ps (encode x)) = coordSequence f ps x := by
  rw [← coordinate_sequence_closed f g encode D stay closed ps x hx]
  exact recover _ (coordinate_sequence_domain f D stay ps x hx)

theorem coordinate_affine_composition (a b c d x : ℝ) :
    c*(a*x+b)+d = (c*a)*x+(c*b+d) := by ring

theorem coordinate_affine_associativity (a b c d e f x : ℝ) :
    (e*c*a)*x+(e*(c*b+d)+f) = e*(c*(a*x+b)+d)+f := by ring

theorem coordinate_affine_all_parameter_dagger
    (a b c d x da db dc dd dx g : ℝ) :
    (c*a*dx+c*x*da+c*db+(a*x+b)*dc+dd)*g =
    dx*(a*c*g)+da*(x*c*g)+db*(c*g)+dc*((a*x+b)*g)+dd*g := by ring

theorem log_homogeneous_closed (a x : ℝ) (p : ℕ) (ha : 0 < a) (hx : 0 < x) :
    Real.log ((a*x)^p) = (p:ℝ)*Real.log x+(p:ℝ)*Real.log a := by
  rw [Real.log_pow, Real.log_mul ha.ne' hx.ne']
  ring

theorem log_homogeneous_decoded (a x : ℝ) (p : ℕ) (ha : 0 < a) (hx : 0 < x) :
    Real.exp ((p:ℝ)*Real.log x+(p:ℝ)*Real.log a) = (a*x)^p := by
  rw [← log_homogeneous_closed a x p ha hx]
  exact Real.exp_log (pow_pos (mul_pos ha hx) _)

theorem log_homogeneous_input_derivative (a x : ℝ) (p : ℕ)
    (ha : 0 < a) (hx : 0 < x) :
    HasDerivAt (fun t : ℝ => Real.exp ((p:ℝ)*Real.log t+(p:ℝ)*Real.log a))
      ((a*x)^p*((p:ℝ)/x)) x := by
  have h := ((Real.hasDerivAt_log hx.ne').const_mul (p:ℝ)).add_const ((p:ℝ)*Real.log a)
  have he := h.exp
  simpa only [log_homogeneous_decoded a x p ha hx, div_eq_mul_inv] using he

theorem log_homogeneous_weight_derivative (a x : ℝ) (p : ℕ)
    (ha : 0 < a) (hx : 0 < x) :
    HasDerivAt (fun t : ℝ => Real.exp ((p:ℝ)*Real.log x+(p:ℝ)*Real.log t))
      ((a*x)^p*((p:ℝ)/a)) a := by
  have h := (hasDerivAt_const a ((p:ℝ)*Real.log x)).add
    ((Real.hasDerivAt_log ha.ne').const_mul (p:ℝ))
  have he := h.exp
  simpa only [Pi.add_apply, log_homogeneous_decoded a x p ha hx, div_eq_mul_inv, zero_add] using he

theorem koopman_triangular_closed (x y : ℝ) :
    ((2*x)^2, 3*y+x^2) = (4*x^2, x^2+3*y) := by
  apply Prod.ext <;> dsimp <;> ring

theorem koopman_trainable_closed (a b c x y : ℝ) :
    ((a*x)^2, b*y+c*x^2) = (a^2*x^2, c*x^2+b*y) := by
  apply Prod.ext <;> dsimp <;> ring

theorem output_only_parameter_gradient_impossible :
    ¬ ∃ recover : ℝ → ℝ, ∀ x : ℝ, recover ((0:ℝ)*x+1) = x := by
  rintro ⟨recover,h⟩
  have h0 := h 0
  have h1 := h 1
  simp only [zero_mul, zero_add] at h0 h1
  rw [h0] at h1
  norm_num at h1
end OjasXMath
