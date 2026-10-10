From Stdlib Require Import QArith.Qcanon Bool Ring.
From Ojasx Require Import Matrix.
Open Scope Qc_scope.

Definition gate (active : bool) (x : Qc) : Qc := if active then x else 0.
Definition positive_mask (x : Qc) : bool := if Qclt_le_dec 0 x then true else false.
Definition relu (x : Qc) := gate (positive_mask x) x.
(* This follows the usual zero-gradient convention at x=0.
   It does not assert that ReLU is differentiable at zero. *)
Definition relu_pullback (x upstream : Qc) := gate (positive_mask x) upstream.
Definition packed_relu_pullback (saved_mask : bool) (upstream : Qc) := gate saved_mask upstream.

Theorem relu_one_bit_sufficient x upstream :
  packed_relu_pullback (positive_mask x) upstream = relu_pullback x upstream.
Proof. reflexivity. Qed.

Theorem relu_fixed_region_exact x dx t : positive_mask (x+t*dx) = positive_mask x ->
  relu (x+t*dx) = relu x + t*relu_pullback x dx.
Proof.
  intro H. unfold relu, relu_pullback. rewrite H.
  unfold gate. destruct (positive_mask x); ring.
Qed.

Definition gate_columns (mask : nat -> bool) (a : matrix) : matrix :=
  fun i j => gate (mask j) (a i j).

Lemma mm_gate_columns k mask a b i j :
  mm k a (gate_columns mask b) i j = gate_columns mask (mm k a b) i j.
Proof.
  unfold mm, gate_columns, gate. destruct (mask j); [reflexivity|].
  transitivity (qsum k (fun _ => 0)); [apply qsum_ext; intro h; ring|apply qsum_zero].
Qed.

Lemma gate_transfer k mask a b i j :
  mm k (gate_columns mask a) (transpose b) i j =
  mm k a (transpose (gate_columns mask b)) i j.
Proof.
  unfold mm, gate_columns, transpose, gate. apply qsum_ext. intro h.
  destruct (mask h); ring.
Qed.

(* Shared means one hidden-unit mask across all rows in this batch.
   Arbitrary input-dependent masks do NOT satisfy this hypothesis. *)
Theorem shared_mask_forward k n mask x a b i j :
  mm n (gate_columns mask (mm k x a)) b i j =
  mm k x (mm n (gate_columns mask a) b) i j.
Proof.
  transitivity (mm n (mm k x (gate_columns mask a)) b i j).
  - apply mm_ext_l. intros r s. symmetry. apply mm_gate_columns.
  - apply mm_associative.
Qed.

Theorem shared_mask_dx p n mask a b g i j :
  mm n (gate_columns mask (mm p g (transpose b))) (transpose a) i j =
  mm p g (transpose (mm n (gate_columns mask a) b)) i j.
Proof. rewrite gate_transfer. apply linear_pair_dx. Qed.

Theorem shared_mask_da m p mask x b g i j :
  mm m (transpose x) (gate_columns mask (mm p g (transpose b))) i j =
  gate_columns mask (mm p (mm m (transpose x) g) (transpose b)) i j.
Proof. rewrite mm_gate_columns. unfold gate_columns. rewrite linear_pair_da. reflexivity. Qed.

Theorem shared_mask_db m k mask x a g i j :
  mm m (transpose (gate_columns mask (mm k x a))) g i j =
  mm k (transpose (gate_columns mask a)) (mm m (transpose x) g) i j.
Proof.
  transitivity (mm m (transpose (mm k x (gate_columns mask a))) g i j).
  - apply mm_ext_l. intros r s. unfold transpose. symmetry. apply mm_gate_columns.
  - apply linear_pair_db.
Qed.

(* Lossy activation storage counterexample: square outputs erase the sign,
   so output alone cannot determine the derivative at both +1 and -1. *)
Theorem square_output_not_sufficient :
  ~ exists recover : Qc -> Qc, forall x : Qc, recover (x*x) = (1+1)*x.
Proof.
  intros [recover H]. specialize (H 1) as Hpos. specialize (H (Qcopp 1)) as Hneg.
  replace (1*1) with 1 in Hpos by ring.
  replace ((Qcopp 1)*(Qcopp 1)) with 1 in Hneg by ring.
  rewrite Hpos in Hneg. vm_compute in Hneg. discriminate.
Qed.
