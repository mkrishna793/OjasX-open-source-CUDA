From Stdlib Require Import QArith.Qcanon Ring.
Open Scope Qc_scope.

Definition matrix := nat -> nat -> Qc.
Fixpoint qsum (n : nat) (f : nat -> Qc) : Qc :=
  match n with O => 0 | S k => qsum k f + f k end.
Definition mm (k : nat) (a b : matrix) : matrix :=
  fun i j => qsum k (fun h => a i h * b h j).
Definition transpose (a : matrix) : matrix := fun i j => a j i.

Lemma qsum_ext n f g : (forall i, f i = g i) -> qsum n f = qsum n g.
Proof. intro H. induction n; simpl; [reflexivity|]. rewrite IHn, H. reflexivity. Qed.
Lemma qsum_zero n : qsum n (fun _ => 0) = 0.
Proof. induction n; simpl; [reflexivity|]. rewrite IHn. ring. Qed.
Lemma qsum_add n f g : qsum n (fun i => f i + g i) = qsum n f + qsum n g.
Proof. induction n; simpl; [ring|]. rewrite IHn. ring. Qed.
Lemma qsum_mul_l n a f : qsum n (fun i => a*f i) = a*qsum n f.
Proof. induction n; simpl; [ring|]. rewrite IHn. ring. Qed.
Lemma qsum_mul_r n a f : qsum n (fun i => f i*a) = qsum n f*a.
Proof. induction n; simpl; [ring|]. rewrite IHn. ring. Qed.
Lemma qsum_swap m n f :
  qsum m (fun i => qsum n (fun j => f i j)) =
  qsum n (fun j => qsum m (fun i => f i j)).
Proof.
  induction m; simpl; [symmetry; apply qsum_zero|].
  rewrite qsum_add, IHm. reflexivity.
Qed.

Theorem mm_associative k n a b c i j :
  mm n (mm k a b) c i j = mm k a (mm n b c) i j.
Proof.
  unfold mm.
  transitivity (qsum n (fun h => qsum k (fun l => a i l*b l h*c h j))).
  - apply qsum_ext. intro h. symmetry. apply qsum_mul_r.
  - rewrite qsum_swap. apply qsum_ext. intro l.
    rewrite <- qsum_mul_l. apply qsum_ext. intro h. ring.
Qed.

Theorem mm_transpose k a b i j :
  transpose (mm k a b) i j = mm k (transpose b) (transpose a) i j.
Proof. unfold transpose, mm. apply qsum_ext. intro h. ring. Qed.

Lemma mm_ext_l k a b c : (forall i j, a i j = b i j) ->
  forall i j, mm k a c i j = mm k b c i j.
Proof. intros H i j. unfold mm. apply qsum_ext. intro h. rewrite H. reflexivity. Qed.
Lemma mm_ext_r k a b c : (forall i j, b i j = c i j) ->
  forall i j, mm k a b i j = mm k a c i j.
Proof. intros H i j. unfold mm. apply qsum_ext. intro h. rewrite H. reflexivity. Qed.

(* X:m*k, A:k*n, B:n*p, upstream G:m*p. Every trainable gradient is retained. *)
Theorem linear_pair_forward k n x a b i j :
  mm n (mm k x a) b i j = mm k x (mm n a b) i j.
Proof. apply mm_associative. Qed.

Theorem linear_pair_dx p n a b g i j :
  mm n (mm p g (transpose b)) (transpose a) i j =
  mm p g (transpose (mm n a b)) i j.
Proof.
  rewrite mm_associative. apply mm_ext_r. intros r s.
  symmetry. apply mm_transpose.
Qed.

Theorem linear_pair_da m p x b g i j :
  mm m (transpose x) (mm p g (transpose b)) i j =
  mm p (mm m (transpose x) g) (transpose b) i j.
Proof. symmetry. apply mm_associative. Qed.

Theorem linear_pair_db m k x a g i j :
  mm m (transpose (mm k x a)) g i j =
  mm k (transpose a) (mm m (transpose x) g) i j.
Proof.
  transitivity (mm m (mm k (transpose a) (transpose x)) g i j).
  - apply mm_ext_l. intros r s. apply mm_transpose.
  - apply mm_associative.
Qed.

Definition perturb_matrix (a da : matrix) (t : Qc) : matrix :=
  fun i j => a i j + t*da i j.

Theorem mm_first_order k a b da db t i j :
  mm k (perturb_matrix a da t) (perturb_matrix b db t) i j =
  mm k a b i j + t*(mm k da b i j + mm k a db i j) + t*t*mm k da db i j.
Proof.
  unfold mm, perturb_matrix. induction k; simpl; [ring|]. rewrite IHk. ring.
Qed.

Definition pairing m n (a b : matrix) := qsum m (fun i => qsum n (fun j => a i j*b i j)).

Theorem mm_input_adjoint m k n da b g :
  pairing m n (mm k da b) g = pairing m k da (mm n g (transpose b)).
Proof.
  unfold pairing, mm, transpose. apply qsum_ext. intro i.
  transitivity (qsum n (fun j => qsum k (fun h => da i h*b h j*g i j))).
  - apply qsum_ext. intro j. symmetry. apply qsum_mul_r.
  - rewrite qsum_swap. apply qsum_ext. intro h.
    rewrite <- qsum_mul_l. apply qsum_ext. intro j. ring.
Qed.

Theorem mm_weight_adjoint m k n a db g :
  pairing m n (mm k a db) g = pairing k n db (mm m (transpose a) g).
Proof.
  unfold pairing, mm, transpose.
  transitivity (qsum m (fun i => qsum k (fun h => qsum n (fun j => a i h*db h j*g i j)))).
  - apply qsum_ext. intro i. rewrite <- qsum_swap.
    apply qsum_ext. intro j. symmetry. apply qsum_mul_r.
  - rewrite qsum_swap. apply qsum_ext. intro h. rewrite qsum_swap.
    apply qsum_ext. intro j. rewrite <- qsum_mul_l.
    apply qsum_ext. intro i. ring.
Qed.
