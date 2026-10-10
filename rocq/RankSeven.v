From Stdlib Require Import QArith.Qcanon Ring NArith Lia.
From Ojasx Require Import Matrix Structured.
Open Scope Qc_scope.

Record quad (T : Type) := Quad { a00 : T; a01 : T; a10 : T; a11 : T }.
Arguments Quad {T} _ _ _ _.
Arguments a00 {T} _.
Arguments a01 {T} _.
Arguments a10 {T} _.
Arguments a11 {T} _.

(* Strassen's existing seven-product identity. This is not a claim of a
   newly discovered matrix multiplication algorithm. *)
Definition seven {T} (add sub mul : T -> T -> T) (a b : quad T) : quad T :=
  let p1 := mul (add (a00 a) (a11 a)) (add (a00 b) (a11 b)) in
  let p2 := mul (add (a10 a) (a11 a)) (a00 b) in
  let p3 := mul (a00 a) (sub (a01 b) (a11 b)) in
  let p4 := mul (a11 a) (sub (a10 b) (a00 b)) in
  let p5 := mul (add (a00 a) (a01 a)) (a11 b) in
  let p6 := mul (sub (a10 a) (a00 a)) (add (a00 b) (a01 b)) in
  let p7 := mul (sub (a01 a) (a11 a)) (add (a10 b) (a11 b)) in
  Quad (add (sub (add p1 p4) p5) p7) (add p3 p5)
    (add p2 p4) (add (add (sub p1 p2) p3) p6).

Definition conventional {T} (add mul : T -> T -> T) (a b : quad T) : quad T :=
  Quad (add (mul (a00 a) (a00 b)) (mul (a01 a) (a10 b)))
    (add (mul (a00 a) (a01 b)) (mul (a01 a) (a11 b)))
    (add (mul (a10 a) (a00 b)) (mul (a11 a) (a10 b)))
    (add (mul (a10 a) (a01 b)) (mul (a11 a) (a11 b))).

Definition mul7 := seven Qcplus Qcminus Qcmult.
Definition mul2 := conventional Qcplus Qcmult.
Definition trans2 (a : quad Qc) := Quad (a00 a) (a10 a) (a01 a) (a11 a).
Definition add2 (a b : quad Qc) :=
  Quad (a00 a+a00 b) (a01 a+a01 b) (a10 a+a10 b) (a11 a+a11 b).
Definition scale2 (t : Qc) (a : quad Qc) := Quad (t*a00 a) (t*a01 a) (t*a10 a) (t*a11 a).
Definition pair2 (a b : quad Qc) := a00 a*a00 b+a01 a*a01 b+a10 a*a10 b+a11 a*a11 b.
Definition reverse7 a b g := (mul7 g (trans2 b), mul7 (trans2 a) g).

Theorem seven_scalar_correct a b : mul7 a b = mul2 a b.
Proof. destruct a,b. unfold mul7, mul2, seven, conventional. simpl. f_equal; ring. Qed.
Theorem seven_left_gradient a b g : fst (reverse7 a b g) = mul2 g (trans2 b).
Proof. unfold reverse7. simpl. apply seven_scalar_correct. Qed.
Theorem seven_right_gradient a b g : snd (reverse7 a b g) = mul2 (trans2 a) g.
Proof. unfold reverse7. simpl. apply seven_scalar_correct. Qed.

Theorem seven_joint_adjoint a b da db g :
  pair2 (add2 (mul2 da b) (mul2 a db)) g =
  pair2 da (fst (reverse7 a b g)) + pair2 db (snd (reverse7 a b g)).
Proof.
  rewrite seven_left_gradient, seven_right_gradient.
  destruct a,b,da,db,g. unfold pair2,add2,mul2,conventional,trans2. simpl. ring.
Qed.

Theorem seven_first_order a b da db t :
  mul7 (add2 a (scale2 t da)) (add2 b (scale2 t db)) =
  add2 (mul2 a b) (add2 (scale2 t (add2 (mul2 da b) (mul2 a db)))
    (scale2 (t*t) (mul2 da db))).
Proof.
  rewrite seven_scalar_correct. destruct a,b,da,db.
  unfold add2, scale2, mul2, conventional. simpl. f_equal; ring.
Qed.

Definition matrix_add (a b : matrix) : matrix := fun i j => a i j+b i j.
Definition matrix_sub (a b : matrix) : matrix := fun i j => a i j-b i j.
Definition block7 k := seven matrix_add matrix_sub (mm k).
Definition block_native k := conventional matrix_add (mm k).
Definition block_at (a : quad matrix) i j :=
  Quad (a00 a i j) (a01 a i j) (a10 a i j) (a11 a i j).

(* Arbitrary finite block inner dimension. The proof preserves operand
   order, so no invalid commutativity of matrix multiplication is used. *)
Theorem seven_blocks_correct k a b i j :
  block_at (block7 k a b) i j = block_at (block_native k a b) i j.
Proof.
  destruct a,b. unfold block_at, block7, block_native, seven, conventional,
    matrix_add, matrix_sub, mm. simpl.
  f_equal; repeat first [rewrite <- qsum_add | rewrite <- qsum_subtract];
    apply qsum_ext; intro h; ring.
Qed.

(* Uniform block costs M,A, for square equally-sized blocks. Forward plus
   both gradients use three products. Extra temporary storage is separate. *)
Definition ordinary_joint_work (mult add : N) := (24*mult+12*add)%N.
Definition seven_joint_work (mult add : N) := (21*mult+54*add)%N.
Definition choose_seven (mult add scratch budget : N) :=
  ((seven_joint_work mult add <=? ordinary_joint_work mult add)%N && (scratch <=? budget)%N)%bool.

Theorem seven_work_threshold mult add :
  (seven_joint_work mult add <= ordinary_joint_work mult add)%N <-> (14*add <= mult)%N.
Proof. unfold seven_joint_work, ordinary_joint_work. nia. Qed.

Theorem seven_guard_work mult add scratch budget : choose_seven mult add scratch budget = true ->
  (seven_joint_work mult add <= ordinary_joint_work mult add)%N.
Proof.
  unfold choose_seven. intro H. apply Bool.andb_true_iff in H. destruct H as [H _].
  apply N.leb_le. exact H.
Qed.
Theorem seven_guard_memory mult add scratch budget : choose_seven mult add scratch budget = true ->
  (scratch <= budget)%N.
Proof.
  unfold choose_seven. intro H. apply Bool.andb_true_iff in H. destruct H as [_ H].
  apply N.leb_le. exact H.
Qed.

Definition selected_product mult add scratch budget a b :=
  if choose_seven mult add scratch budget then mul7 a b else mul2 a b.
Theorem selected_product_correct mult add scratch budget a b :
  selected_product mult add scratch budget a b = mul2 a b.
Proof. unfold selected_product. destruct (choose_seven _ _ _ _); [apply seven_scalar_correct|reflexivity]. Qed.

(* Known Winograd rearrangement: same seven products, fifteen additions
   per multiplication instead of eighteen in the construction above. *)
Definition winograd {T} (add sub mul : T -> T -> T) (a b : quad T) : quad T :=
  let s1 := add (a10 a) (a11 a) in
  let s2 := sub s1 (a00 a) in
  let s3 := sub (a00 a) (a10 a) in
  let s4 := sub (a01 a) s2 in
  let t1 := sub (a01 b) (a00 b) in
  let t2 := sub (a11 b) t1 in
  let t3 := sub (a11 b) (a01 b) in
  let t4 := sub t2 (a10 b) in
  let p1 := mul (a00 a) (a00 b) in
  let p2 := mul (a01 a) (a10 b) in
  let p3 := mul s4 (a11 b) in
  let p4 := mul (a11 a) t4 in
  let p5 := mul s1 t1 in
  let p6 := mul s2 t2 in
  let p7 := mul s3 t3 in
  let u1 := add p1 p6 in
  let u2 := add u1 p7 in
  let u3 := add u1 p5 in
  Quad (add p1 p2) (add u3 p3) (sub u2 p4) (add u2 p5).
Definition mulw := winograd Qcplus Qcminus Qcmult.
Definition blockw k := winograd matrix_add matrix_sub (mm k).

Theorem winograd_scalar_correct a b : mulw a b = mul2 a b.
Proof. destruct a,b. unfold mulw, mul2, winograd, conventional. simpl. f_equal; ring. Qed.
Theorem winograd_blocks_correct k a b i j :
  block_at (blockw k a b) i j = block_at (block_native k a b) i j.
Proof.
  destruct a,b. unfold block_at, blockw, block_native, winograd, conventional,
    matrix_add, matrix_sub, mm. simpl.
  f_equal; repeat first [rewrite <- qsum_add | rewrite <- qsum_subtract];
    apply qsum_ext; intro h; ring.
Qed.
Definition reversew a b g := (mulw g (trans2 b), mulw (trans2 a) g).
Theorem winograd_left_gradient a b g : fst (reversew a b g) = mul2 g (trans2 b).
Proof. unfold reversew. simpl. apply winograd_scalar_correct. Qed.
Theorem winograd_right_gradient a b g : snd (reversew a b g) = mul2 (trans2 a) g.
Proof. unfold reversew. simpl. apply winograd_scalar_correct. Qed.
Definition winograd_joint_work (mult add : N) := (21*mult+45*add)%N.
Theorem winograd_work_threshold mult add :
  (winograd_joint_work mult add <= ordinary_joint_work mult add)%N <-> (11*add <= mult)%N.
Proof. unfold winograd_joint_work, ordinary_joint_work. nia. Qed.
Theorem winograd_improves_addition_model mult add :
  (winograd_joint_work mult add <= seven_joint_work mult add)%N.
Proof. unfold winograd_joint_work, seven_joint_work. nia. Qed.

Definition choose_winograd (mult add scratch budget : N) :=
  ((winograd_joint_work mult add <=? ordinary_joint_work mult add)%N && (scratch <=? budget)%N)%bool.
Theorem winograd_guard mult add scratch budget : choose_winograd mult add scratch budget = true ->
  (winograd_joint_work mult add <= ordinary_joint_work mult add /\ scratch <= budget)%N.
Proof.
  unfold choose_winograd. intro H. apply Bool.andb_true_iff in H. destruct H as [H K].
  split; apply N.leb_le; assumption.
Qed.
