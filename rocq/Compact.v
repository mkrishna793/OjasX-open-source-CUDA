From Stdlib Require Import QArith.Qcanon Ring Lia.
From Ojasx Require Import Matrix Structured.
Open Scope Qc_scope.

Definition compact_rms_pull n r invn (z q : vector) i :=
  r*(q i-z i*invn*qsum n (fun j => z j*q j)).
Definition compact_layer_pull n r invn (z q : vector) i :=
  r*(q i-invn*qsum n q-z i*invn*qsum n (fun j=>z j*q j)).

Theorem rms_residual_sufficient n r invn z gamma g i :
  structured_pull n 1 (fun j=>r*gamma j)
    (fun j _=>Qcopp(r*gamma j*z j*invn)) (fun j _=>z j) g i =
  compact_rms_pull n r invn z (fun j=>gamma j*g j) i.
Proof.
  unfold structured_pull, structured_push, factor_apply, compact_rms_pull. simpl.
  assert (H : qsum n (fun j=>Qcopp(r*gamma j*z j*invn)*g j) =
    Qcopp(r*invn)*qsum n (fun j=>z j*(gamma j*g j))).
  { rewrite <- qsum_mul_l. apply qsum_ext. intro j. ring. }
  rewrite H. ring.
Qed.

(* Logical scalar-value counts, excluding aliases, parameters, temporary
   buffers and retained differentiation graphs. *)
Theorem compact_rms_residual_budget (rows width : nat) :
  (1 <= width)%nat -> (rows*(width+1) <= 3*rows*width)%nat.
Proof. intro H. nia. Qed.

Theorem compact_layer_residual_budget (rows width : nat) :
  (1 <= width)%nat -> (rows*(width+1) <= 5*rows*width)%nat.
Proof. intro H. nia. Qed.

Theorem normalized_rms_rewrite r gamma x z : z=r*x ->
  Qcopp(r*r*r*gamma*x)*x = Qcopp(r*gamma*z)*z.
Proof. intro H. rewrite H. ring. Qed.

Theorem layer_residual_sufficient n r invn z gamma g i :
  structured_pull n 2 (fun j=>r*gamma j)
    (fun j h=>match h with O=>Qcopp(r*gamma j*invn) | _=>Qcopp(r*gamma j*z j*invn) end)
    (fun j h=>match h with O=>1 | _=>z j end) g i =
  compact_layer_pull n r invn z (fun j=>gamma j*g j) i.
Proof.
  unfold structured_pull, structured_push, factor_apply, compact_layer_pull. simpl.
  assert (H : qsum n (fun j=>Qcopp(r*gamma j*invn)*g j) =
    Qcopp(r*invn)*qsum n (fun j=>gamma j*g j)).
  { rewrite <- qsum_mul_l. apply qsum_ext. intro j. ring. }
  assert (K : qsum n (fun j=>Qcopp(r*gamma j*z j*invn)*g j) =
    Qcopp(r*invn)*qsum n (fun j=>z j*(gamma j*g j))).
  { rewrite <- qsum_mul_l. apply qsum_ext. intro j. ring. }
  rewrite H,K. ring.
Qed.

Theorem epsilon_radial_correction n r invn eps z i :
  invn*qsum n (fun j=>z j*z j)=1-eps*r*r ->
  compact_rms_pull n r invn z z i = eps*r*r*r*z i.
Proof. intro H. unfold compact_rms_pull.
  replace (z i*invn*qsum n (fun j=>z j*z j)) with (z i*(invn*qsum n (fun j=>z j*z j))) by ring.
  rewrite H. ring. Qed.

Definition affine_direction (z dz : matrix) (gamma dgamma dbeta : vector) : matrix :=
  fun i j=>gamma j*dz i j+dgamma j*z i j+dbeta j.
Definition gain_pull m (z g : matrix) j := qsum m (fun i=>z i j*g i j).
Definition bias_pull m (g : matrix) j := qsum m (fun i=>g i j).

Theorem affine_all_gradients m n z dz gamma dgamma dbeta g :
  pairing m n (affine_direction z dz gamma dgamma dbeta) g =
  pairing m n dz (fun i j=>gamma j*g i j) +
  vpair n dgamma (gain_pull m z g) + vpair n dbeta (bias_pull m g).
Proof.
  unfold pairing, affine_direction, vpair, gain_pull, bias_pull.
  transitivity (qsum m (fun i=>qsum n (fun j=>dz i j*(gamma j*g i j))) +
    qsum m (fun i=>qsum n (fun j=>dgamma j*(z i j*g i j))) +
    qsum m (fun i=>qsum n (fun j=>dbeta j*g i j))).
  - rewrite <- !qsum_add. apply qsum_ext. intro i.
    rewrite <- !qsum_add. apply qsum_ext. intro j. ring.
  - rewrite (qsum_swap m n (fun i j=>dgamma j*(z i j*g i j))).
    rewrite (qsum_swap m n (fun i j=>dbeta j*g i j)).
    f_equal; [f_equal|]; apply qsum_ext; intro j; apply qsum_mul_l.
Qed.
