import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fintype.Pi

namespace OjasXMath

/- Necessary information condition for any output-only pullback coefficient. -/
theorem output_only_requires_fiber_consistency {α β γ : Type*}
    (f : α → β) (d : α → γ) (h : β → γ)
    (factor : ∀ x, h (f x) = d x) (x y : α) (same : f x = f y) :
    d x = d y := by
  rw [← factor x, ← factor y, same]

theorem collision_forbids_output_only {α β γ : Type*}
    (f : α → β) (d : α → γ) (x y : α)
    (same : f x = f y) (different : d x ≠ d y) :
    ¬ ∃ h : β → γ, ∀ z, h (f z) = d z := by
  rintro ⟨h, hh⟩
  exact different (output_only_requires_fiber_consistency f d h hh x y same)

/- General residual criterion: the cotangent map may itself be the value γ.
   The default handles residual values that no input can produce. -/
theorem residual_sufficiency_iff {α β γ : Type*} [Nonempty γ]
    (encode : α → β) (needed : α → γ) :
    (∃ decode : β → γ, ∀ x, decode (encode x) = needed x) ↔
    (∀ x y, encode x = encode y → needed x = needed y) := by
  classical
  constructor
  · rintro ⟨decode, h⟩ x y same
    exact output_only_requires_fiber_consistency encode needed decode h x y same
  · intro fibers
    let decode := fun b => if hb : ∃ x, encode x = b then
      needed (Classical.choose hb) else Classical.choice (inferInstance : Nonempty γ)
    refine ⟨decode, ?_⟩
    intro x
    dsimp [decode]
    have hx : ∃ y, encode y = encode x := ⟨x, rfl⟩
    rw [dite_eq_left hx]
    exact fibers _ x (Classical.choose_spec hx)

/- Worst case: n independent binary flags cannot be losslessly encoded in
   fewer than n bits. This does not apply to a restricted/compressible domain. -/
theorem exact_boolean_encoding_requires_bits (n b : Nat)
    (encode : (Fin n → Bool) → (Fin b → Bool))
    (hinj : Function.Injective encode) : n ≤ b := by
  have h := Fintype.card_le_of_injective encode hinj
  simp only [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin] at h
  exact (Nat.pow_le_pow_iff_right (by decide : 1 < 2)).1 h

theorem square_has_derivative (x : ℝ) :
    HasDerivAt (fun z : ℝ => z*z) (2*x) x := by
  convert (hasDerivAt_id x).mul (hasDerivAt_id x) using 1
  · funext z
    rfl
  · simp only [id_eq, one_mul, mul_one]
    ring

theorem square_has_no_output_only_derivative :
    ¬ ∃ h : ℝ → ℝ, ∀ x : ℝ, h (x*x) = 2*x := by
  apply collision_forbids_output_only (fun x : ℝ => x*x) (fun x => 2*x) 1 (-1)
  · norm_num
  · norm_num

theorem product_has_no_output_only_left_derivative :
    ¬ ∃ h : ℝ → ℝ, ∀ p : ℝ×ℝ, h (p.1*p.2) = p.2 := by
  apply collision_forbids_output_only (fun p : ℝ×ℝ => p.1*p.2) (fun p => p.2) (1,2) (2,1)
  · norm_num
  · norm_num

theorem scalar_chain_derivative (f g : ℝ → ℝ) (x df dg : ℝ)
    (hf : HasDerivAt f df x) (hg : HasDerivAt g dg (f x)) :
    HasDerivAt (g ∘ f) (dg*df) x := hg.comp x hf

theorem reverse_scalar_chain (df dg dy : ℝ) :
    (dg*df)*dy = df*(dg*dy) := by ring

/- Backward needs only a branch flag once its exact threshold is known. -/
def thresholdPullback (positive : Bool) (dy : ℝ) : ℝ := if positive then dy else 0

theorem relu_mask_sufficient (flag : Bool) (dy : ℝ) :
    thresholdPullback flag dy = if flag then dy else 0 := rfl

/- Row-statistic RMSNorm pullback coefficient over real arithmetic.
   This theorem establishes algebraic rewriting, not a transcendental FP proof. -/
theorem rms_row_statistic_identity (r x g dot n : ℝ) :
    r*(g-(x*r/n)*(r*dot)) = r*g-x*r^3*dot/n := by
  ring

end OjasXMath
