/-! M4 composition obligations used by equality saturation. Matrix
reassociation and structured transpose obligations reuse the audited
`linear_chain_forward`, `dagger_composition`, and `dagger_involution` theorems.
Exact-real identities are not floating-point bitwise claims. -/
namespace OjasXMath

/-- Splitting at any activation boundary and fusing the append are the same
function composition. Replay and local residual obligations are handled by the
M2.5 stage theorems. -/
theorem m4_activation_chain_split_fuse {X Y Z W : Type}
    (f : X → Y) (g : Y → Z) (h : Z → W) (x : X) :
    h (g (f x)) = (h ∘ g) (f x) := rfl

/-- The executable selector keeps the original cost unless the candidate is
strictly better. This standalone model uses one scalar cost; the Z3 artifact
checks the full four-component lexicographic objective. -/
def selectCost (original candidate : Nat) : Nat :=
  if candidate < original then candidate else original

theorem m4_original_retained_without_strict_improvement
    (original candidate : Nat) (h : original ≤ candidate) :
    selectCost original candidate = original := by
  simp [selectCost, Nat.not_lt_of_ge h]

/-- A nonnegative error bound within budget forces a nonnegative budget. -/
theorem m4_bounded_error_budget_nonnegative (bound budget : Rat) :
    0 ≤ bound → bound ≤ budget → 0 ≤ budget :=
  fun hbound hle => Rat.le_trans hbound hle

end OjasXMath
