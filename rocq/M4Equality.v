Require Import QArith.Qcanon Ring Lia Coq.Arith.PeanoNat.
Open Scope Qc_scope.

Definition m4_matrix := nat -> nat -> Qc.
Definition m4_transpose (a : m4_matrix) : m4_matrix := fun i j => a j i.

Theorem m4_activation_chain_split_fuse {X Y Z W : Type}
  (f : X -> Y) (g : Y -> Z) (h : Z -> W) x :
  h (g (f x)) = (fun y => h (g y)) (f x).
Proof. reflexivity. Qed.

Theorem m4_transpose_involution (a : m4_matrix) i j :
  m4_transpose (m4_transpose a) i j = a i j.
Proof. reflexivity. Qed.

(* Output-only square residuals lose the sign fiber: x and -x have the same
   output but require different primal inputs for a general pullback. *)
Theorem m4_square_output_residual_incomplete :
  exists x y : Qc, y = x*x /\ y = (-x)*(-x) /\ x <> -x.
Proof.
  exists (-(1 : Qc)), (1 : Qc). repeat split; try ring; discriminate.
Qed.

Definition m4_select_cost (original candidate : nat) : nat :=
  if Nat.ltb candidate original then candidate else original.

Theorem m4_original_retained_without_strict_improvement
  (original candidate : nat) :
  (original <= candidate)%nat ->
  m4_select_cost original candidate = original.
Proof.
  intros Hle. unfold m4_select_cost.
  destruct (Nat.ltb candidate original) eqn:Hlt; [| reflexivity].
  assert (H : (candidate < original)%nat) by
    (apply (proj1 (Nat.ltb_lt candidate original)); exact Hlt).
  lia.
Qed.

Theorem m4_bounded_error_budget_nonnegative (bound budget : Qc) :
  0 <= bound -> bound <= budget -> 0 <= budget.
Proof. intros Hbound Hle. eapply Qcle_trans; eauto. Qed.

Close Scope Qc_scope.
