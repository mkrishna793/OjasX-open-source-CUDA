From Stdlib Require Import QArith.Qcanon Ring Arith Lia.
From Ojasx Require Import Matrix Structured Indexed.
Open Scope Qc_scope.

(* The framework stores proved contracts as data. An original pullback must
   already satisfy its primitive derivative contract. No missing global laws. *)
Module InvariantFoundation.
Record ExactRewrite (I O C G : Type) (primal : I -> O) (pull : I -> C -> G) := {
  execute : I -> O;
  reverse : I -> C -> G;
  execute_correct : forall x, execute x=primal x;
  reverse_correct : forall x cot, reverse x cot=pull x cot;
  work : I -> nat;
  reference_work : I -> nat;
  work_dominated : forall x, (work x <= reference_work x)%nat
}.
End InvariantFoundation.

Definition row_scale (a : vector) (x : matrix) : matrix := fun i j=>a i*x i j.
Definition matrix_plus (a b : matrix) : matrix := fun i j=>a i j+b i j.

Lemma invariant_mm_add_l k a b c i j :
 mm k (matrix_plus a b) c i j = mm k a c i j+mm k b c i j.
Proof. unfold mm, matrix_plus. rewrite <- qsum_add. apply qsum_ext. intro h. ring. Qed.

Lemma invariant_mm_add_r k a b c i j :
 mm k a (matrix_plus b c) i j = mm k a b i j+mm k a c i j.
Proof. unfold mm, matrix_plus. rewrite <- qsum_add. apply qsum_ext. intro h. ring. Qed.

Lemma invariant_row_scale_product k a b c i j :
 mm k (row_scale a b) c i j = row_scale a (mm k b c) i j.
Proof. unfold mm, row_scale. rewrite <- qsum_mul_l. apply qsum_ext. intro h. ring. Qed.

Lemma invariant_scaled_pair m a x p i j :
 mm m (transpose (row_scale a x)) p i j =
 mm m (transpose x) (row_scale a p) i j.
Proof. unfold mm, transpose, row_scale. apply qsum_ext. intro h. ring. Qed.

Theorem invariant_gram_matrix k n a v i j :
 mm n (mm k a v) (transpose (mm k a v)) i j =
 mm k (mm k a (mm n v (transpose v))) (transpose a) i j.
Proof.
 rewrite mm_associative.
 rewrite (mm_associative k k a (mm n v (transpose v)) (transpose a)).
 apply mm_ext_r. intros h l.
 transitivity (mm n v (mm k (transpose v) (transpose a)) h l).
 - apply mm_ext_r. intros u t. apply mm_transpose.
 - symmetry. apply mm_associative.
Qed.

Theorem invariant_projection k n a v q i j :
 mm n (mm k a v) q i j = mm k a (mm n v q) i j.
Proof. apply mm_associative. Qed.

Theorem invariant_radial_forward k n (f : Qc->Qc) a v q i j :
 f (mm n (mm k a v) (transpose (mm k a v)) i i) * mm n (mm k a v) q i j =
 f (mm k (mm k a (mm n v (transpose v))) (transpose a) i i) *
 mm k a (mm n v q) i j.
Proof. rewrite invariant_gram_matrix, (invariant_projection k n a v q i j). reflexivity. Qed.

(* The factors alpha and e are computed from each sample's actual scalar
   gate and coupling. They can differ across every row of the batch. *)
Definition invariant_native_dz (k r : nat) (a v q p : matrix) (alpha e : vector) : matrix :=
 matrix_plus (row_scale alpha (mm r p (transpose q)))
             (row_scale (fun i=>(1+1)*e i) (mm k a v)).

Theorem invariant_input_gradient k n r a v q p alpha e i j :
 mm n (invariant_native_dz k r a v q p alpha e) (transpose v) i j =
 mm r (row_scale alpha p) (transpose (mm n v q)) i j +
 row_scale (fun i=>(1+1)*e i) (mm k a (mm n v (transpose v))) i j.
Proof.
 unfold invariant_native_dz. rewrite invariant_mm_add_l.
 rewrite !invariant_row_scale_product.
 f_equal.
 - unfold row_scale. f_equal.
   rewrite mm_associative. apply mm_ext_r. intros h l.
   symmetry. apply mm_transpose.
 - unfold row_scale. f_equal. apply mm_associative.
Qed.

Lemma invariant_move_row_scale k m a e v i j :
 mm m (transpose a) (row_scale e (mm k a v)) i j =
 mm k (mm m (transpose a) (row_scale e a)) v i j.
Proof.
 rewrite mm_associative. unfold mm, row_scale, transpose.
 apply qsum_ext. intro h. rewrite <- !qsum_mul_l.
 apply qsum_ext. intro t. ring.
Qed.

Theorem invariant_weight_gradient m k r a v q p alpha e i j :
 mm m (transpose a) (invariant_native_dz k r a v q p alpha e) i j =
 mm r (mm m (transpose a) (row_scale alpha p)) (transpose q) i j +
 mm k (mm m (transpose a) (row_scale (fun h=>(1+1)*e h) a)) v i j.
Proof.
 unfold invariant_native_dz. rewrite invariant_mm_add_r.
 f_equal.
 - transitivity (mm m (transpose a) (mm r (row_scale alpha p) (transpose q)) i j).
   + apply mm_ext_r. intros h l. symmetry. apply invariant_row_scale_product.
   + symmetry. apply mm_associative.
 - apply invariant_move_row_scale.
Qed.

Theorem invariant_projection_gradient m k a v p alpha i j :
 mm m (transpose (row_scale alpha (mm k a v))) p i j =
 mm k (transpose v) (mm m (transpose a) (row_scale alpha p)) i j.
Proof.
 rewrite invariant_scaled_pair.
 transitivity (mm m (mm k (transpose v) (transpose a)) (row_scale alpha p) i j).
 - apply mm_ext_l. intros h l. apply mm_transpose.
 - apply mm_associative.
Qed.

Theorem invariant_local_dagger n z dz g alpha beta :
 vpair n (fun i=>alpha*dz i+beta*z i*vpair n z dz) g =
 vpair n dz (fun i=>alpha*g i+beta*z i*vpair n z g).
Proof.
 unfold vpair.
 transitivity (qsum n (fun i=>alpha*dz i*g i)+
               beta*qsum n (fun j=>z j*dz j)*qsum n (fun i=>z i*g i)).
 - rewrite <- qsum_mul_l, <- qsum_add.
   apply qsum_ext. intro i. ring.
 - assert (H : qsum n (fun j=>z j*dz j)=qsum n (fun j=>dz j*z j)).
   { apply qsum_ext. intro j. ring. }
   rewrite H.
   replace (beta*qsum n (fun j=>dz j*z j)*qsum n (fun i=>z i*g i)) with
     ((beta*qsum n (fun i=>z i*g i))*qsum n (fun j=>dz j*z j)) by ring.
   rewrite <- qsum_mul_l, <- qsum_add. apply qsum_ext. intro i. ring.
Qed.

Theorem invariant_chamber_factors chambers rank (p q masks : matrix) (labels : nat->nat) (i j : nat) :
 (labels i < chambers)%nat ->
 qsum rank (fun h=>p i h*q j h)*masks (labels i) j =
 qsum rank (fun h=>qsum chambers (fun c=>
   (p i h*delta (labels i) c)*(q j h*masks c j))).
Proof.
 intro H. rewrite <- qsum_mul_r. apply qsum_ext. intro h.
 transitivity (qsum chambers (fun c=>(p i h*q j h*masks c j)*delta (labels i) c)).
 - symmetry. apply qsum_delta. exact H.
 - apply qsum_ext. intro c. ring.
Qed.

Theorem invariant_trace_mask m n a g mask :
 pairing m n a (fun i j=>g i j*mask i j) =
 pairing m n (fun i j=>a i j*mask i j) g.
Proof. unfold pairing. apply qsum_ext. intro i. apply qsum_ext. intro j. ring. Qed.

Theorem invariant_common_cost (common candidate native : nat) :
 (candidate <= native)%nat -> (common+candidate <= common+native)%nat.
Proof. lia. Qed.

Theorem invariant_aligned_cubic n z i : vpair n z z=1 ->
 z i+(1+1)*z i*vpair n z z = (1+1+1)*z i.
Proof. intro H. rewrite H. ring. Qed.

Theorem invariant_norm_scale_cancellation a variance eps r :
 r*r*(a*a*variance+eps)=1 ->
 r-a*a*variance*r*r*r = eps*r*r*r.
Proof.
 intro H. replace r with (r*(r*r*(a*a*variance+eps))) at 1 by (rewrite H;ring).
 ring.
Qed.

(* Stable normalization cotangent, including a=0.  No rounded subtraction. *)
Theorem stable_norm_scale_cotangent a variance eps r dot :
 r*r*(a*a*variance+eps)=1 ->
 (r-a*a*variance*r*r*r)*dot = eps*r*r*r*dot.
Proof.
 intro H. rewrite (invariant_norm_scale_cancellation a variance eps r H). ring.
Qed.

Theorem stable_norm_ratio_reconstruction denominator eps r dot :
 r*r*denominator=1 ->
 denominator*(eps*r*r*r*dot)=eps*r*dot.
Proof.
 intro H.
 replace (denominator*(eps*r*r*r*dot)) with (eps*r*dot*(r*r*denominator)) by ring.
 rewrite H. ring.
Qed.
