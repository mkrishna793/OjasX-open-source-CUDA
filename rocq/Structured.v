From Stdlib Require Import QArith.Qcanon Ring NArith Lia.
From Ojasx Require Import Matrix.
Open Scope Qc_scope.

Definition vector := nat -> Qc.
Definition vpair n (x y : vector) := qsum n (fun i => x i*y i).
Definition factor_apply n r (u v : matrix) (x : vector) : vector :=
  fun i => qsum r (fun h => u i h * qsum n (fun j => v j h*x j)).
Definition structured_push n r (d : vector) (u v : matrix) (x : vector) : vector :=
  fun i => d i*x i + factor_apply n r u v x i.
Definition structured_pull n r d u v g := structured_push n r d v u g.

Lemma vpair_add_l n a b c :
  vpair n (fun i => a i+b i) c = vpair n a c + vpair n b c.
Proof.
  unfold vpair. rewrite <- qsum_add. apply qsum_ext. intro i. ring.
Qed.
Lemma vpair_add_r n a b c :
  vpair n a (fun i => b i+c i) = vpair n a b + vpair n a c.
Proof.
  unfold vpair. rewrite <- qsum_add. apply qsum_ext. intro i. ring.
Qed.

Theorem factor_adjoint n r u v x g :
  vpair n (factor_apply n r u v x) g = vpair n x (factor_apply n r v u g).
Proof.
  unfold vpair, factor_apply.
  transitivity (qsum n (fun i => qsum r (fun h =>
    (u i h * qsum n (fun j => v j h*x j))*g i))).
  - apply qsum_ext. intro i. symmetry. apply qsum_mul_r.
  - rewrite qsum_swap.
    transitivity (qsum r (fun h =>
      (qsum n (fun j => v j h*x j))*(qsum n (fun i => u i h*g i)))).
    + apply qsum_ext. intro h. rewrite <- qsum_mul_l.
      apply qsum_ext. intro i. ring.
    + transitivity (qsum r (fun h => qsum n (fun j =>
        x j*(v j h*qsum n (fun i => u i h*g i))))).
      * apply qsum_ext. intro h. rewrite <- qsum_mul_r.
        apply qsum_ext. intro j. ring.
      * rewrite qsum_swap. apply qsum_ext. intro j. apply qsum_mul_l.
Qed.

Theorem structured_adjoint n r d u v x g :
  vpair n (structured_push n r d u v x) g = vpair n x (structured_pull n r d u v g).
Proof.
  unfold structured_pull, structured_push.
  rewrite vpair_add_l, vpair_add_r, factor_adjoint.
  f_equal. unfold vpair. apply qsum_ext. intro i. ring.
Qed.

Definition lowrank_dense r u v : matrix := fun i j => qsum r (fun h => u i h*v j h).

Theorem factor_equals_dense n r u v x i :
  factor_apply n r u v x i = qsum n (fun j => lowrank_dense r u v i j*x j).
Proof.
  unfold factor_apply, lowrank_dense.
  transitivity (qsum r (fun h => qsum n (fun j => u i h*(v j h*x j)))).
  - apply qsum_ext. intro h. symmetry. apply qsum_mul_l.
  - rewrite qsum_swap. apply qsum_ext. intro j. rewrite <- qsum_mul_r.
    apply qsum_ext. intro h. ring.
Qed.

Theorem structured_pull_additive n r d u v g h i :
  structured_pull n r d u v (fun j => g j+h j) i =
  structured_pull n r d u v g i + structured_pull n r d u v h i.
Proof.
  unfold structured_pull, structured_push, factor_apply.
  assert (E : forall a, qsum n (fun j => u j a*(g j+h j)) =
    qsum n (fun j => u j a*g j) + qsum n (fun j => u j a*h j)).
  { intro a. rewrite <- qsum_add. apply qsum_ext. intro j. ring. }
  assert (F : qsum r (fun a => v i a*qsum n (fun j => u j a*(g j+h j))) =
    qsum r (fun a => v i a*qsum n (fun j => u j a*g j))+
    qsum r (fun a => v i a*qsum n (fun j => u j a*h j))).
  { rewrite <- qsum_add. apply qsum_ext. intro a. rewrite E. ring. }
  rewrite F. ring.
Qed.

Definition softmax_pull n (p g : vector) : vector :=
  fun i => p i*(g i-qsum n (fun j => p j*g j)).

Theorem softmax_as_rank_one n p g i :
  softmax_pull n p g i = structured_pull n 1 p
    (fun i _ => p i) (fun i _ => Qcopp (p i)) g i.
Proof. unfold softmax_pull, structured_pull, structured_push, factor_apply. simpl. ring. Qed.

Lemma qsum_subtract n f g : qsum n (fun i => f i-g i) = qsum n f-qsum n g.
Proof. induction n; simpl; [ring|]. rewrite IHn. ring. Qed.

Theorem softmax_zero_sum n p g : qsum n p = 1 ->
  qsum n (softmax_pull n p g) = 0.
Proof.
  intro H. unfold softmax_pull.
  transitivity (qsum n (fun i => p i*g i) - qsum n (fun i => p i*qsum n (fun j => p j*g j))).
  - rewrite <- qsum_subtract. apply qsum_ext. intro i. ring.
  - rewrite qsum_mul_r, H. ring.
Qed.

(* Model counts include each sum's initial zero-addition. No cost to form
   factors, no transcendental work and no floating-point runtime claim. *)
Definition structured_work (n r : N) := (4*n*r+2*n)%N.
Definition explicit_dense_work (n : N) := (2*n*n)%N.

Theorem structured_work_bound n r : (2*r+1 <= n)%N ->
  (structured_work n r <= explicit_dense_work n)%N.
Proof. unfold structured_work, explicit_dense_work. intro H. nia. Qed.
