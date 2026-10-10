From Stdlib Require Import QArith.Qcanon List Ring Lia.
Import ListNotations.
Open Scope Qc_scope.

(* Exact, division-free reference mathematics. No floating-point claims. *)
Fixpoint product (xs : list Qc) : Qc :=
  match xs with [] => 1 | x :: tail => x * product tail end.

Definition scale (p : Qc) (xs : list Qc) := map (fun x => p * x) xs.

Fixpoint reference_gradient (xs : list Qc) : list Qc :=
  match xs with
  | [] => []
  | x :: tail => product tail :: scale x (reference_gradient tail)
  end.

(* One traversal, three rational multiplications per input factor.
   The prefix can include an upstream cotangent. Zero factors are allowed. *)
Fixpoint product_pullback (xs : list Qc) (prefix : Qc) : Qc * list Qc :=
  match xs with
  | [] => (1, [])
  | x :: tail =>
    let '(suffix, gradients) := product_pullback tail (prefix * x) in
    (x * suffix, prefix * suffix :: gradients)
  end.

Lemma scale_compose a b xs : scale a (scale b xs) = scale (a*b) xs.
Proof. unfold scale. rewrite map_map. apply map_ext. intro x. ring. Qed.

Lemma scale_one xs : scale 1 xs = xs.
Proof. induction xs; simpl; [reflexivity|]. unfold scale in *. simpl in *. f_equal; [ring|assumption]. Qed.

Theorem product_pullback_correct xs p :
  product_pullback xs p = (product xs, scale p (reference_gradient xs)).
Proof.
  revert p. induction xs as [|x xs IH]; intro p; simpl; [reflexivity|].
  rewrite IH. simpl. rewrite scale_compose. reflexivity.
Qed.

(* Instrument the fast schedule's rational multiplications; integer bit
   complexity and rational normalization are not unit-time in reality. *)
Fixpoint counted_pullback (xs : list Qc) (prefix : Qc) : (Qc * list Qc) * nat :=
  match xs with
  | [] => ((1, []), O)
  | x::tail =>
    let '((suffix, gs), count) := counted_pullback tail (prefix*x) in
    ((x*suffix, prefix*suffix::gs), (count+3)%nat)
  end.

Theorem counted_pullback_correct xs p :
  counted_pullback xs p = (product_pullback xs p, (3*length xs)%nat).
Proof.
  revert p. induction xs as [|x xs IH]; intro p; simpl; [reflexivity|].
  rewrite IH. destruct (product_pullback xs (p*x)).
  f_equal. simpl. lia.
Qed.

Fixpoint reference_multiplications (n : nat) : nat :=
  match n with O => O | S k => (reference_multiplications k + 2*k)%nat end.

Theorem reference_quadratic_count n : (reference_multiplications n + n = n*n)%nat.
Proof. induction n; simpl; nia. Qed.

Theorem product_work_reduction n : (4 <= n)%nat ->
  (3*n <= reference_multiplications n)%nat.
Proof. intro H. pose proof (reference_quadratic_count n). nia. Qed.

Definition fast_gradient xs := snd (product_pullback xs 1).

Theorem fast_gradient_correct xs : fast_gradient xs = reference_gradient xs.
Proof. unfold fast_gradient. rewrite product_pullback_correct. simpl. apply scale_one. Qed.

Theorem reference_gradient_length xs : length (reference_gradient xs) = length xs.
Proof. induction xs; simpl; [reflexivity|]. unfold scale. rewrite length_map, IHxs. reflexivity. Qed.

Fixpoint dot (xs ys : list Qc) : Qc :=
  match xs, ys with x::xt, y::yt => x*y + dot xt yt | _, _ => 0 end.

Fixpoint tangent_product (xs ds : list Qc) : Qc :=
  match xs, ds with
  | x::xt, d::dt => d * product xt + x * tangent_product xt dt
  | _, _ => 0
  end.

Lemma dot_scale p xs ys : dot (scale p xs) ys = p * dot xs ys.
Proof.
  revert ys. induction xs as [|x xs IH]; intros [|y ys]; simpl; try ring.
  rewrite IH. ring.
Qed.

Theorem product_gradient_direction xs ds :
  dot (reference_gradient xs) ds = tangent_product xs ds.
Proof.
  revert ds. induction xs as [|x xs IH]; intros [|d ds]; simpl; try reflexivity.
  rewrite dot_scale, IH. ring.
Qed.

Fixpoint perturb (xs ds : list Qc) (t : Qc) : list Qc :=
  match xs, ds with x::xt, d::dt => (x+t*d)::perturb xt dt t | _, _ => [] end.

Fixpoint product_remainder (xs ds : list Qc) (t : Qc) : Qc :=
  match xs, ds with
  | x::xt, d::dt =>
    x*product_remainder xt dt t + d*tangent_product xt dt
      + t*d*product_remainder xt dt t
  | _, _ => 0
  end.

(* An exact polynomial identity certifies the first-order coefficient,
   including at zero factors. No division or nonzero-factor hypothesis. *)
Theorem product_first_order xs ds t : length xs = length ds ->
  product (perturb xs ds t) = product xs + t * dot (fast_gradient xs) ds
    + t*t*product_remainder xs ds t.
Proof.
  rewrite fast_gradient_correct, product_gradient_direction.
  revert ds. induction xs as [|x xs IH]; intros [|d ds] H; simpl in *;
    try discriminate; [ring|].
  injection H as H. rewrite IH by assumption. ring.
Qed.

Definition squared_loss xs target := (product xs - target) * (product xs - target).
Definition reference_loss_gradient xs target :=
  scale ((1+1)*(product xs-target)) (reference_gradient xs).
Definition fast_loss_gradient xs target :=
  let '(value, grads) := product_pullback xs 1 in scale ((1+1)*(value-target)) grads.

Theorem fast_loss_gradient_correct xs target :
  fast_loss_gradient xs target = reference_loss_gradient xs target.
Proof. unfold fast_loss_gradient, reference_loss_gradient.
  rewrite product_pullback_correct. simpl. rewrite scale_one. reflexivity. Qed.

Theorem squared_loss_first_order xs ds t target : length xs = length ds ->
  squared_loss (perturb xs ds t) target = squared_loss xs target
    + t * dot (fast_loss_gradient xs target) ds
    + t*t * ((1+1)*(product xs-target)*product_remainder xs ds t
      + tangent_product xs ds*tangent_product xs ds
      + (1+1)*t*tangent_product xs ds*product_remainder xs ds t
      + t*t*product_remainder xs ds t*product_remainder xs ds t).
Proof.
  intro H. unfold squared_loss.
  rewrite product_first_order by assumption.
  rewrite fast_loss_gradient_correct. unfold reference_loss_gradient.
  rewrite dot_scale, fast_gradient_correct, !product_gradient_direction. ring.
Qed.
