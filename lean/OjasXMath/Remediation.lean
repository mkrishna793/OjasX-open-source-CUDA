import Mathlib

/-!
# Verified-core remediation contracts

These are exact mathematical statements about the repaired accumulator, packed
layout and protocol state. They are not a Python-to-Lean refinement proof, an
IEEE floating-point error bound, or a cryptographic proof of HMAC.
-/

open scoped BigOperators

namespace OjasXMath

/-- Dividing every microbatch gradient before summing is the same exact-real
arithmetic mean as dividing their sum once. -/
theorem microbatch_scaled_sum (n : ℕ) (g : Fin n → ℝ) :
    (∑ i, g i / (n : ℝ)) = (∑ i, g i) / (n : ℝ) := by
  rw [← Finset.sum_div]

/-- If a partial window stores `S / N`, multiplying by `N / r` recovers the
mean `S / r` of the `r` actually pending microbatches. -/
theorem accumulation_partial_flush (S N r : ℝ) (hN : N ≠ 0) (hr : r ≠ 0) :
    (S / N) * (N / r) = S / r := by
  field_simp

/-- The former trainer/optimizer pair divides by four twice; that is not the
required single mean for a nonzero accumulated gradient. -/
theorem old_double_scaling_not_mean (G : ℝ) (hG : G ≠ 0) :
    G / 4 / 4 ≠ G / 4 := by
  intro h
  apply hG
  nlinarith

/-- The row-wise native layout has an exact flat scale count whenever the
feature dimension is divisible by the group size. -/
theorem native_row_scale_reshape (N K G : ℕ) (h : G ∣ K) :
    N * K / G = N * (K / G) := by
  exact Nat.mul_div_assoc N h

/-- Exact integer storage for 32-wide compressed blocks has less than one full
block of padding. -/
theorem compressed_padding_bound_32 (n : ℕ) :
    n ≤ ((n + 31) / 32) * 32 ∧ ((n + 31) / 32) * 32 < n + 32 := by
  omega

/-- A nonempty input uses at least one compressed block. -/
theorem compressed_rows_positive_32 (n : ℕ) (hn : 0 < n) :
    0 < (n + 31) / 32 := by
  omega

/-- Accepting sequence `received` advances the expected sequence, so the same
frame cannot be accepted immediately again in that ordered session. -/
theorem mesh_ordered_session_rejects_replay (received sequence : ℕ)
    (h : sequence = received) : sequence ≠ received + 1 := by
  omega

/-- Small explicit state shared by the accumulator contract and its reset. -/
structure RemediationAccumulatorState where
  count : ℕ
  total : ℝ

def consumeRemediationAccumulator (_ : RemediationAccumulatorState) :
    RemediationAccumulatorState := ⟨0, 0⟩

theorem accumulation_consumption_clears_state (s : RemediationAccumulatorState) :
    (consumeRemediationAccumulator s).count = 0 ∧
      (consumeRemediationAccumulator s).total = 0 := by
  rfl

/-- Checked candidate selection: both validations must pass before trial data
can replace the native reference. -/
def selectCheckedCandidate {α : Type} (valueOk gradientOk : Bool)
    (trial native : α) : α :=
  if valueOk && gradientOk then trial else native

theorem accepted_checked_candidate_uses_trial {α : Type} (trial native : α) :
    selectCheckedCandidate true true trial native = trial := by
  rfl

theorem invalid_value_preserves_native {α : Type} (gradientOk : Bool)
    (trial native : α) :
    selectCheckedCandidate false gradientOk trial native = native := by
  rfl

theorem invalid_gradient_preserves_native {α : Type} (valueOk : Bool)
    (trial native : α) :
    selectCheckedCandidate valueOk false trial native = native := by
  cases valueOk <;> rfl

end OjasXMath
