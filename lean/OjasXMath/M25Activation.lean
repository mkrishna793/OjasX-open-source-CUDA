import OjasXMath.CertifiedBackprop
import OjasXMath.M25ReplayCore

/-! M2.5 replay re-encoding. A fused chain may discard the tail residual only
when a replay theorem reconstructs it from the retained head residual. Primitive
derivative contracts remain explicit; no floating-point or allocator claim is
made by these exact-real composition theorems. -/
namespace OjasXMath
noncomputable section

/-- Retain the head residual and replay the tail residual on reverse. -/
noncomputable def CertifiedStage.replayTail {I J K S T : Type*}
    [Fintype I] [Fintype J] [Fintype K]
    (f : CertifiedStage I J S) (h : CertifiedStage J K T)
    (replay : S → T)
    (hreplay : ∀ x, replay (f.forward x).1 = (h.forward (f.forward x).2).1) :
    CertifiedStage I K S :=
  (f.then h).reencode
    (fun residual => residual.1)
    (fun residual g => f.pullback residual (h.pullback (replay residual) g))
    (by
      intro x g
      change f.pullback (f.forward x).1
        (h.pullback (replay (f.forward x).1) g) =
        f.pullback (f.forward x).1
          (h.pullback (h.forward (f.forward x).2).1 g)
      rw [hreplay])

/-- Replay re-encoding leaves the primal composite unchanged. -/
theorem m25_replay_tail_primal {I J K S T : Type*}
    [Fintype I] [Fintype J] [Fintype K]
    (f : CertifiedStage I J S) (h : CertifiedStage J K T)
    (replay : S → T)
    (hreplay : ∀ x, replay (f.forward x).1 = (h.forward (f.forward x).2).1)
    (x : I → ℝ) :
    ((f.replayTail h replay hreplay).forward x).2 = ((f.then h).forward x).2 := rfl

/-- The fused pullback equals the original two-residual pullback. -/
theorem m25_replay_tail_pullback {I J K S T : Type*}
    [Fintype I] [Fintype J] [Fintype K]
    (f : CertifiedStage I J S) (h : CertifiedStage J K T)
    (replay : S → T)
    (hreplay : ∀ x, replay (f.forward x).1 = (h.forward (f.forward x).2).1)
    (x : I → ℝ) (g : K → ℝ) :
    (f.replayTail h replay hreplay).pullback
      ((f.replayTail h replay hreplay).forward x).1 g =
    (f.then h).pullback ((f.then h).forward x).1 g := by
  change f.pullback (f.forward x).1
    (h.pullback (replay (f.forward x).1) g) =
    f.pullback (f.forward x).1
      (h.pullback (h.forward (f.forward x).2).1 g)
  apply m25_replay_tail_substitution
  exact hreplay

/-- The fused differential has the original composite adjoint pairing. -/
theorem m25_replay_tail_dagger {I J K S T : Type*}
    [Fintype I] [Fintype J] [Fintype K]
    (f : CertifiedStage I J S) (h : CertifiedStage J K T)
    (replay : S → T)
    (hreplay : ∀ x, replay (f.forward x).1 = (h.forward (f.forward x).2).1)
    (x dx : I → ℝ) (g : K → ℝ) :
    scalarPair ((f.replayTail h replay hreplay).differential x dx) g =
      scalarPair dx ((f.then h).pullback ((f.then h).forward x).1 g) := by
  rw [← m25_replay_tail_pullback f h replay hreplay]
  exact (f.replayTail h replay hreplay).dagger_correct x dx g

/-- Consequently an arbitrary optimizer sees the same update and state. -/
theorem m25_replay_tail_optimizer {I J K S T State : Type*}
    [Fintype I] [Fintype J] [Fintype K]
    (f : CertifiedStage I J S) (h : CertifiedStage J K T)
    (replay : S → T)
    (hreplay : ∀ x, replay (f.forward x).1 = (h.forward (f.forward x).2).1)
    (update : (I → ℝ) → (I → ℝ) → State → (I → ℝ) × State)
    (x : I → ℝ) (g : K → ℝ) (state : State) :
    update x
      ((f.replayTail h replay hreplay).pullback
        ((f.replayTail h replay hreplay).forward x).1 g) state =
    update x ((f.then h).pullback ((f.then h).forward x).1 g) state := by
  rw [m25_replay_tail_pullback]

/-- The engine's minimum-peak candidate is no worse than the native candidate. -/
theorem m25_policy_choice_memory_guard (native fused : ℕ) :
    min native fused ≤ native := min_le_left _ _

end
end OjasXMath
