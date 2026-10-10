From Stdlib Require Import QArith.Qcanon Ring Arith Bool Lia.
From Ojasx Require Import Matrix Structured.
Open Scope Qc_scope.

Lemma qsum_bounded_ext n f g : (forall i, (i<n)%nat -> f i=g i) -> qsum n f=qsum n g.
Proof.
  intro H. induction n; simpl; [reflexivity|].
  rewrite IHn, H; [reflexivity|lia|intros i K;apply H;lia].
Qed.

Definition delta i j : Qc := if Nat.eqb i j then 1 else 0.
Lemma qsum_delta n (f : vector) j : (j<n)%nat ->
  qsum n (fun i=>f i*delta j i)=f j.
Proof.
  induction n as [|n IH]; intro H; [lia|]. simpl.
  destruct (Nat.eq_dec j n) as [E|E].
  - subst j. unfold delta. rewrite Nat.eqb_refl.
    assert (Z : qsum n (fun i=>f i*(if Nat.eqb n i then 1 else 0))=0).
    { transitivity (qsum n (fun _=>0)); [|apply qsum_zero].
      apply qsum_bounded_ext. intros i K.
      assert (B : Nat.eqb n i=false) by (apply Nat.eqb_neq;lia).
      rewrite B. ring. }
    rewrite Z. ring.
  - rewrite IH by lia. unfold delta. assert (B : Nat.eqb j n=false) by (apply Nat.eqb_neq;assumption). rewrite B. ring.
Qed.

Definition gather (index : nat->nat) (x : vector) : vector := fun j=>x(index j).
Definition scatter_sum k (index : nat->nat) (g : vector) : vector :=
  fun i=>qsum k (fun j=>g j*delta (index j) i).

Theorem gather_first_order index x dx t j :
  gather index (fun i=>x i+t*dx i) j = gather index x j+t*gather index dx j.
Proof. reflexivity. Qed.

Theorem gather_scatter_adjoint n k index dx g :
  (forall j, (j<k)%nat -> (index j<n)%nat) ->
  vpair k (gather index dx) g = vpair n dx (scatter_sum k index g).
Proof.
  intro H. unfold vpair, gather, scatter_sum.
  transitivity (qsum k (fun j=>qsum n (fun i=>dx i*delta(index j)i)*g j)).
  - apply qsum_bounded_ext. intros j K. rewrite qsum_delta by (apply H;exact K). reflexivity.
  - transitivity (qsum k (fun j=>qsum n (fun i=>dx i*(g j*delta(index j)i)))).
    + apply qsum_ext. intro j. rewrite <- qsum_mul_r. apply qsum_ext. intro i. ring.
    + rewrite qsum_swap. apply qsum_ext. intro i. apply qsum_mul_l.
Qed.
