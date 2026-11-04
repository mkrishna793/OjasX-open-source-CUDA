/-! Dependency-free replay substitutions shared by the M2.5 certified-stage
development and standalone kernel checks. No floating-point or allocator claim
is made by these generic theorems. -/
namespace OjasXMath

/-- Replaying a tail residual from the head residual leaves the composed
pullback unchanged on every primal fiber. -/
theorem m25_replay_tail_substitution {X Y Z S T : Type}
    (primal : X → Y) (save : X → S) (saveTail : Y → T) (replay : S → T)
    (pull : S → Y → X) (tailPull : T → Z → Y)
    (hreplay : ∀ x, replay (save x) = saveTail (primal x)) (x : X) (g : Z) :
    pull (save x) (tailPull (replay (save x)) g) =
      pull (save x) (tailPull (saveTail (primal x)) g) := by
  rw [hreplay]

/-- The same substitution is invisible to an arbitrary downstream optimizer. -/
theorem m25_replay_optimizer_substitution {X Y Z S T State : Type}
    (primal : X → Y) (save : X → S) (saveTail : Y → T) (replay : S → T)
    (pull : S → Y → X) (tailPull : T → Z → Y)
    (hreplay : ∀ x, replay (save x) = saveTail (primal x))
    (update : X → X → State → X × State) (x : X) (g : Z) (state : State) :
    update x (pull (save x) (tailPull (replay (save x)) g)) state =
      update x (pull (save x) (tailPull (saveTail (primal x)) g)) state := by
  rw [hreplay]

/-- Selecting the minimum of two managed-memory candidates cannot be worse
than retaining the original candidate. -/
theorem m25_minimum_candidate_guard (native fused : Nat) :
    min native fused ≤ native := by
  exact Nat.min_le_left _ _

end OjasXMath
