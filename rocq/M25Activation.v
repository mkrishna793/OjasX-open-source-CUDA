Require Import QArith.Qcanon Ring Lia.
Open Scope Qc_scope.

(* Replay re-encoding: the retained head residual must reconstruct the tail
   residual on every primal fiber. Local derivative/adjoint obligations are
   intentionally separate from this substitution theorem. *)
Theorem m25_replay_tail_pullback {X Y Z S T : Type}
  (primal : X -> Y) (save : X -> S) (saveTail : Y -> T) (replay : S -> T)
  (pull : S -> Y -> X) (tailPull : T -> Z -> Y)
  (H : forall x, replay (save x) = saveTail (primal x)) x g :
  pull (save x) (tailPull (replay (save x)) g) =
  pull (save x) (tailPull (saveTail (primal x)) g).
Proof. rewrite H. reflexivity. Qed.

Theorem m25_replay_tail_optimizer {X Y Z S T State : Type}
  (primal : X -> Y) (save : X -> S) (saveTail : Y -> T) (replay : S -> T)
  (pull : S -> Y -> X) (tailPull : T -> Z -> Y)
  (H : forall x, replay (save x) = saveTail (primal x))
  (update : X -> X -> State -> (X*State)%type) x g state :
  update x (pull (save x) (tailPull (replay (save x)) g)) state =
  update x (pull (save x) (tailPull (saveTail (primal x)) g)) state.
Proof. rewrite H. reflexivity. Qed.

Theorem m25_cotangent_chain (s1 s2 s3 g : Qc) :
  ((g*s3)*s2)*s1 = g*(s1*s2*s3).
Proof. ring. Qed.

Theorem m25_policy_choice_memory_guard (native fused : nat) :
  (Nat.min native fused <= native)%nat.
Proof.
  revert fused.
  induction native as [|n IH]; intros fused; destruct fused; simpl; lia.
Qed.

Fixpoint prefix_work (layers : nat) : nat :=
  match layers with O => O | S n => S n + prefix_work n end.

Theorem m25_cached_work_le_streaming (layers : nat) :
  (layers <= prefix_work layers)%nat.
Proof. induction layers; simpl; lia. Qed.

(* A negative control: fusion is not a universal peak-memory improvement. *)
Theorem m25_fusion_peak_not_universal :
  ~ (forall native fused : nat, (fused <= native)%nat).
Proof.
  intro H. specialize (H O (S O)). simpl in H. lia.
Qed.

Close Scope Qc_scope.
