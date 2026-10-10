From Stdlib Require Import QArith.Qcanon Ring NArith Arith Lia.
From Ojasx Require Import Matrix.
Open Scope Qc_scope.

(* Finite exact-rational identities. The analytic differentiability part lives
   in Lean; these statements do not assert transcendental or IEEE semantics. *)
Theorem objective_rational_square (prediction target : Qc) :
 (prediction-target)*(prediction-target) =
 prediction*prediction-(1+1)*prediction*target+target*target.
Proof. ring. Qed.

Lemma objective_pair_product m k n h w y :
 pairing m n (mm k h w) y = pairing k n w (mm m (transpose h) y).
Proof.
 unfold pairing, mm, transpose.
 transitivity (qsum m (fun i => qsum k (fun a =>
   qsum n (fun j => h i a*w a j*y i j)))).
 - apply qsum_ext. intro i. rewrite <- qsum_swap.
   apply qsum_ext. intro j. symmetry. apply qsum_mul_r.
 - rewrite qsum_swap. apply qsum_ext. intro a. rewrite qsum_swap.
   apply qsum_ext. intro j. rewrite <- qsum_mul_l.
   apply qsum_ext. intro i. ring.
Qed.

Theorem objective_gram_energy m k n h w :
 pairing m n (mm k h w) (mm k h w) =
 pairing k k (mm m (transpose h) h) (mm n w (transpose w)).
Proof.
 rewrite objective_pair_product. unfold pairing.
 transitivity (qsum k (fun a => qsum n (fun j =>
   w a j * mm k (mm m (transpose h) h) w a j))).
 - apply qsum_ext. intro a. apply qsum_ext. intro j.
   rewrite mm_associative. reflexivity.
 - unfold mm, transpose. apply qsum_ext. intro a.
   transitivity (qsum n (fun j => qsum k (fun b =>
     w a j * ((qsum m (fun i => h i a*h i b))*w b j)))).
   + apply qsum_ext. intro j. symmetry. apply qsum_mul_l.
   + rewrite qsum_swap. apply qsum_ext. intro b.
     rewrite <- qsum_mul_l. apply qsum_ext. intro j. ring.
Qed.

Definition msub (a b : matrix) : matrix := fun i j => a i j-b i j.

Lemma objective_sum_squares n f g :
 qsum n (fun i => (f i-g i)*(f i-g i)) =
 qsum n (fun i => f i*f i)-(1+1)*qsum n (fun i=>f i*g i)+qsum n (fun i=>g i*g i).
Proof. induction n; simpl; [ring|]. rewrite IHn. ring. Qed.

Lemma objective_pair_square m n a y :
 pairing m n (msub a y) (msub a y) =
 pairing m n a a-(1+1)*pairing m n a y+pairing m n y y.
Proof.
 unfold pairing, msub. induction m; simpl; [ring|].
 rewrite IHm, objective_sum_squares. ring.
Qed.

Theorem objective_quadratic_gram_primal m k n h w y :
 pairing m n (msub (mm k h w) y) (msub (mm k h w) y) =
 pairing k k (mm m (transpose h) h) (mm n w (transpose w))-
 (1+1)*pairing k n w (mm m (transpose h) y)+pairing m n y y.
Proof.
 rewrite objective_pair_square, objective_gram_energy, (objective_pair_product m k n h w y).
 reflexivity.
Qed.

Theorem objective_gram_input (m k n : nat) h w y i a :
 mm n (msub (mm k h w) y) (transpose w) i a =
 msub (mm k h (mm n w (transpose w))) (mm n y (transpose w)) i a.
Proof.
 unfold msub, mm, transpose.
 assert (subsum : forall n f g,
   qsum n (fun j => f j-g j) = qsum n f-qsum n g).
 { intros z f g. induction z; simpl; [ring|]. rewrite IHz. ring. }
 transitivity (qsum n (fun j => (qsum k (fun b => h i b*w b j))*w a j)-
   qsum n (fun j => y i j*w a j)).
 - rewrite <- subsum. apply qsum_ext. intro j. ring.
 - f_equal.
   transitivity (qsum n (fun j => qsum k (fun b => h i b*w b j*w a j))).
   + apply qsum_ext. intro j. symmetry. apply qsum_mul_r.
   + rewrite qsum_swap. apply qsum_ext. intro b.
     rewrite <- qsum_mul_l. apply qsum_ext. intro j. ring.
Qed.

Theorem objective_gram_weight (m k n : nat) h w y a j :
 mm m (transpose h) (msub (mm k h w) y) a j =
 msub (mm k (mm m (transpose h) h) w) (mm m (transpose h) y) a j.
Proof.
 unfold msub, mm, transpose.
 assert (subsum : forall n f g,
   qsum n (fun i => f i-g i) = qsum n f-qsum n g).
 { intros z f g. induction z; simpl; [ring|]. rewrite IHz. ring. }
 transitivity (qsum m (fun i => h i a*qsum k (fun b => h i b*w b j))-
   qsum m (fun i => h i a*y i j)).
 - rewrite <- subsum. apply qsum_ext. intro i. ring.
 - f_equal.
   transitivity (qsum m (fun i => qsum k (fun b => h i a*(h i b*w b j)))).
   + apply qsum_ext. intro i. symmetry. apply qsum_mul_l.
   + rewrite qsum_swap. apply qsum_ext. intro b.
     rewrite <- qsum_mul_r. apply qsum_ext. intro i. ring.
Qed.

Fixpoint prefix_accumulate n (f : nat -> Qc) (initial : Qc) : Qc :=
 match n with O => initial | S k => prefix_accumulate k f initial + f k end.
Theorem objective_stream_accumulate n f initial :
 prefix_accumulate n f initial = initial+qsum n f.
Proof. induction n; simpl; [ring|]. rewrite IHn. ring. Qed.

Theorem objective_partition m n f :
 qsum (m+n) f = qsum m f+qsum n (fun i => f (m+i)%nat).
Proof.
 induction n; [rewrite Nat.add_0_r; simpl; ring|].
 rewrite Nat.add_succ_r. simpl. rewrite IHn. ring.
Qed.

Theorem objective_global_denominator a b d :
 a/d+b/d = (a+b)/d.
Proof. unfold Qcdiv. ring. Qed.

Theorem objective_fixed_weight_derivative w f df t :
 w*(f+t*df) = w*f+t*(w*df).
Proof. ring. Qed.

Theorem objective_zero_weight f df :
 (0*f,0*df) = (0,0).
Proof. f_equal; ring. Qed.

Theorem objective_optimizer_barrier {E S : Type}
 (update : E -> E -> S -> E*S) theta full streamed state :
 streamed=full -> update theta streamed state=update theta full state.
Proof. intros ->. reflexivity. Qed.

Theorem objective_cost_difference m k n :
 (1+1+1+1+1+1)*m*k*n-((1+1+1+1)*m*k*n+(1+1+1+1)*m*k*k+(1+1+1+1)*k*k*n) = (1+1)*k*(m*n-(1+1)*k*(m+n)).
Proof. ring. Qed.

Theorem objective_memory_schedule (persistent per_example batch chunk : N) :
 (chunk<=batch)%N ->
 (persistent+chunk*per_example<=persistent+batch*per_example)%N.
Proof. intro H. apply N.add_le_mono_l. now apply N.mul_le_mono_r. Qed.
