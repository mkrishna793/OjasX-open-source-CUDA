From Stdlib Require Import QArith.Qcanon List.
From Ojasx Require Import Product.
Import ListNotations.
Open Scope Qc_scope.

Fixpoint gradient_update (xs gs : list Qc) (eta : Qc) : list Qc :=
  match xs, gs with
  | x::xt, g::gt => (x-eta*g)::gradient_update xt gt eta
  | _, _ => []
  end.

Definition reference_step xs target eta := gradient_update xs (reference_loss_gradient xs target) eta.
Definition fast_step xs target eta := gradient_update xs (fast_loss_gradient xs target) eta.

Theorem gradient_step_equivalent xs target eta :
  fast_step xs target eta = reference_step xs target eta.
Proof. unfold fast_step, reference_step. rewrite fast_loss_gradient_correct. reflexivity. Qed.

Lemma update_length xs gs eta : length xs = length gs ->
  length (gradient_update xs gs eta) = length xs.
Proof.
  revert gs. induction xs; intros [|g gs] H; simpl in *; try discriminate; [reflexivity|].
  f_equal. apply IHxs. injection H. trivial.
Qed.

Theorem fast_step_length xs target eta : length (fast_step xs target eta) = length xs.
Proof.
  unfold fast_step. apply update_length. rewrite fast_loss_gradient_correct.
  unfold reference_loss_gradient, scale. rewrite length_map, reference_gradient_length. reflexivity.
Qed.

Fixpoint iterate_steps (fuel : nat) (step : list Qc -> list Qc) (xs : list Qc) :=
  match fuel with O => xs | S n => iterate_steps n step (step xs) end.

Theorem training_trajectory_equivalent fuel xs target eta :
  iterate_steps fuel (fun w => fast_step w target eta) xs =
  iterate_steps fuel (fun w => reference_step w target eta) xs.
Proof.
  revert xs. induction fuel; intro xs; simpl; [reflexivity|].
  rewrite gradient_step_equivalent. apply IHfuel.
Qed.

(* A bounded safeguard: try at most fuel rates, halve on rejection,
   and return the original vector if no candidate passes. *)
Fixpoint safeguarded_step (fuel : nat) (xs : list Qc) (target eta : Qc) : list Qc :=
  match fuel with
  | O => xs
  | S n => let candidate := fast_step xs target eta in
    if Qclt_le_dec (squared_loss xs target) (squared_loss candidate target)
    then safeguarded_step n xs target (eta / (1+1)) else candidate
  end.

Theorem safeguarded_loss_nonincrease fuel xs target eta :
  squared_loss (safeguarded_step fuel xs target eta) target <= squared_loss xs target.
Proof.
  revert eta. induction fuel; intro eta; simpl; [apply Qcle_refl|].
  destruct (Qclt_le_dec _ _) as [H|H]; [apply IHfuel|exact H].
Qed.

Theorem safeguarded_length fuel xs target eta :
  length (safeguarded_step fuel xs target eta) = length xs.
Proof.
  revert eta. induction fuel; intro eta; simpl; [reflexivity|].
  destruct (Qclt_le_dec _ _); [apply IHfuel|apply fast_step_length].
Qed.
