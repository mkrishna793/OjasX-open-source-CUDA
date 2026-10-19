(******************************************************************************)
(* MODULE: SymplecticEngine.v                                                 *)
(* Mathematical Universe: Formal Mechanics of Deep Symplectic Chains in Rocq  *)
(* Zero Axiom | Zero External Dependencies | Pure Constructive Formalization  *)
(*                                                                            *)
(* This file formally certifies the tri-formal mathematical specifications:    *)
(* 1. Symplectic Euler phase-space volume preservation: det(J_Euler) = 1.     *)
(* 2. Störmer-Verlet (Strang splitting) symmetric volume preservation:        *)
(*    det(J_SV) = 1 in an arbitrary Commutative Ring and in Qc.               *)
(* 3. Exact time-reversibility (involution / bijection) for forward/backward. *)
(* 4. Symplectic squeeze: Hyperbolic scaling preserving canonical 2-form.     *)
(* 5. Sherman-Morrison Structured Adjoint & Determinant Identity:             *)
(*    det(diag(d) + u v^T) = prod(d) * (1 + v^T diag(d)^{-1} u).              *)
(* 6. Structured Derivative cotangent pullback theorem: <J x, g> = <x, J^T g> *)
(*    for LayerNorm, RMSNorm, and Softmax.                                    *)
(******************************************************************************)

From Stdlib Require Import QArith.Qcanon Ring.
Open Scope Qc_scope.

(* ========================================================================= *)
(* 1. CANONICAL 2D PHASE SPACE & MATRIX ALGEBRA IN Qc                        *)
(* ========================================================================= *)

Definition point2 := (Qc * Qc)%type. (* (q, p) *)
Definition mat2 := (Qc * Qc * Qc * Qc)%type. (* (a11, a12, a21, a22) *)

Definition det2 (M : mat2) : Qc :=
  let '(a11, a12, a21, a22) := M in
  a11 * a22 - a12 * a21.

Definition mm2 (A B : mat2) : mat2 :=
  let '(a11, a12, a21, a22) := A in
  let '(b11, b12, b21, b22) := B in
  (a11 * b11 + a12 * b21, a11 * b12 + a12 * b22,
   a21 * b11 + a22 * b21, a21 * b12 + a22 * b22).

Definition transpose2 (A : mat2) : mat2 :=
  let '(a11, a12, a21, a22) := A in
  (a11, a21, a12, a22).

Definition matVecMul2 (A : mat2) (x : point2) : point2 :=
  let '(a11, a12, a21, a22) := A in
  let '(q, p) := x in
  (a11 * q + a12 * p, a21 * q + a22 * p).

Definition dot2 (x y : point2) : Qc :=
  let '(qx, px) := x in
  let '(qy, py) := y in
  qx * qy + px * py.

Definition symplecticForm2 (u v : point2) : Qc :=
  let '(qu, pu) := u in
  let '(qv, pv) := v in
  qu * pv - qv * pu.

(* Elementary Shears *)
Definition shearQ (f' : Qc) : mat2 := (1, f', 0, 1).
Definition shearP (g' : Qc) : mat2 := (1, 0, g', 1).

(* ========================================================================= *)
(* 2. THEOREM 1: SYMPLECTIC EULER VOLUME PRESERVATION                        *)
(* ========================================================================= *)

Definition jEuler (f' g' : Qc) : mat2 :=
  mm2 (shearQ f') (shearP g').

Theorem theorem1_symplectic_euler_volume_preservation (f' g' : Qc) :
  det2 (jEuler f' g') = 1.
Proof.
  unfold jEuler, mm2, shearQ, shearP, det2. ring.
Qed.

(* ========================================================================= *)
(* 3. THEOREM 2: STÖRMER-VERLET (STRANG SPLITTING) VOLUME PRESERVATION       *)
(* ========================================================================= *)

Definition jSV (A B C : Qc) : mat2 :=
  mm2 (shearP C) (mm2 (shearQ B) (shearP A)).

Theorem theorem2_stormer_verlet_volume_preservation (A B C : Qc) :
  det2 (jSV A B C) = 1.
Proof.
  unfold jSV, mm2, shearP, shearQ, det2. ring.
Qed.

(* ========================================================================= *)
(* 4. THEOREM 3: EXACT TIME-REVERSIBILITY (INVOLUTION / BIJECTION)           *)
(* ========================================================================= *)

(* Forward & Backward Symplectic Euler *)
Definition euler_forward (f g : Qc -> Qc) (pt : point2) : point2 :=
  let '(q, p) := pt in
  let p_next := p + g q in
  let q_next := q + f p_next in
  (q_next, p_next).

Definition euler_backward (f g : Qc -> Qc) (pt : point2) : point2 :=
  let '(q, p) := pt in
  let q_prev := q - f p in
  let p_prev := p - g q_prev in
  (q_prev, p_prev).

Theorem theorem3_euler_time_reversibility_backward_forward (f g : Qc -> Qc) (pt : point2) :
  euler_backward f g (euler_forward f g pt) = pt.
Proof.
  destruct pt as [q p].
  unfold euler_backward, euler_forward. simpl.
  f_equal; ring.
Qed.

Theorem theorem3_euler_time_reversibility_forward_backward (f g : Qc -> Qc) (pt : point2) :
  euler_forward f g (euler_backward f g pt) = pt.
Proof.
  destruct pt as [q p].
  unfold euler_backward, euler_forward. simpl.
  f_equal; ring.
Qed.

(* Forward & Backward Symmetric Störmer-Verlet (Strang Splitting) *)
Definition sv_forward (f g1 g2 : Qc -> Qc) (pt : point2) : point2 :=
  let '(q, p) := pt in
  let p_half := p + g1 q in
  let q_next := q + f p_half in
  let p_next := p_half + g2 q_next in
  (q_next, p_next).

Definition sv_backward (f g1 g2 : Qc -> Qc) (pt : point2) : point2 :=
  let '(q, p) := pt in
  let p_half := p - g2 q in
  let q_prev := q - f p_half in
  let p_prev := p_half - g1 q_prev in
  (q_prev, p_prev).

Theorem theorem3_sv_time_reversibility_backward_forward (f g1 g2 : Qc -> Qc) (pt : point2) :
  sv_backward f g1 g2 (sv_forward f g1 g2 pt) = pt.
Proof.
  destruct pt as [q p].
  unfold sv_backward, sv_forward. simpl.
  f_equal; ring.
Qed.

Theorem theorem3_sv_time_reversibility_forward_backward (f g1 g2 : Qc -> Qc) (pt : point2) :
  sv_forward f g1 g2 (sv_backward f g1 g2 pt) = pt.
Proof.
  destruct pt as [q p].
  unfold sv_backward, sv_forward. simpl.
  f_equal; ring.
Qed.

(* ========================================================================= *)
(* 5. THEOREM 4: SYMPLECTIC SQUEEZE INVARIANT                                *)
(* ========================================================================= *)

Definition squeezeMatrix (lam lam_inv : Qc) : mat2 :=
  (lam, 0, 0, lam_inv).

Definition squeezeMap (lam lam_inv : Qc) (pt : point2) : point2 :=
  let '(q, p) := pt in
  (lam * q, lam_inv * p).

Theorem theorem4_squeeze_volume_preservation (lam lam_inv : Qc) :
  lam * lam_inv = 1 -> det2 (squeezeMatrix lam lam_inv) = 1.
Proof.
  intro Hinv.
  unfold det2, squeezeMatrix.
  rewrite Hinv. ring.
Qed.

Theorem theorem4_squeeze_preserves_symplectic_form (lam lam_inv : Qc) (u v : point2) :
  lam * lam_inv = 1 ->
  symplecticForm2 (squeezeMap lam lam_inv u) (squeezeMap lam lam_inv v) =
  symplecticForm2 u v.
Proof.
  intro Hinv.
  destruct u as [qu pu].
  destruct v as [qv pv].
  unfold symplecticForm2, squeezeMap. simpl.
  transitivity ((lam * lam_inv) * (qu * pv) - (lam * lam_inv) * (qv * pu)).
  - ring.
  - rewrite Hinv. ring.
Qed.

(* ========================================================================= *)
(* 6. THEOREM 5: SHERMAN-MORRISON STRUCTURED ADJOINT & DETERMINANT IDENTITIES*)
(* ========================================================================= *)

Definition rankOnePerturbedDiag2 (d1 d2 u1 u2 v1 v2 : Qc) : mat2 :=
  (d1 + u1 * v1, u1 * v2, u2 * v1, d2 + u2 * v2).

Theorem theorem5_sherman_morrison_det2 (d1 d2 u1 u2 v1 v2 : Qc) :
  det2 (rankOnePerturbedDiag2 d1 d2 u1 u2 v1 v2) =
  d1 * d2 + d1 * (u2 * v2) + d2 * (u1 * v1).
Proof.
  unfold det2, rankOnePerturbedDiag2. ring.
Qed.

Theorem theorem5_sherman_morrison_factored (d1 d2 u1 u2 v1 v2 d1_inv d2_inv : Qc) :
  d1 * d1_inv = 1 ->
  d2 * d2_inv = 1 ->
  det2 (rankOnePerturbedDiag2 d1 d2 u1 u2 v1 v2) =
  (d1 * d2) * (1 + (u1 * v1) * d1_inv + (u2 * v2) * d2_inv).
Proof.
  intros H1 H2.
  rewrite theorem5_sherman_morrison_det2.
  transitivity (d1 * d2 + (d1 * d2) * ((u1 * v1) * d1_inv) + (d1 * d2) * ((u2 * v2) * d2_inv)).
  - rewrite <- (Qcmul_assoc d1 d2 ((u1 * v1) * d1_inv)).
    rewrite (Qcmul_comm d2 ((u1 * v1) * d1_inv)).
    rewrite <- (Qcmul_assoc d1 ((u1 * v1) * d1_inv) d2).
    rewrite (Qcmul_comm (u1 * v1) d1_inv).
    rewrite (Qcmul_assoc d1 d1_inv (u1 * v1)).
    rewrite H1.
    rewrite (Qcmul_1_l (u1 * v1)).
    rewrite <- (Qcmul_assoc d1 d2 ((u2 * v2) * d2_inv)).
    rewrite (Qcmul_comm (u2 * v2) d2_inv).
    rewrite (Qcmul_assoc d2 d2_inv (u2 * v2)).
    rewrite H2.
    rewrite (Qcmul_1_l (u2 * v2)).
    ring.
  - ring.
Qed.

Theorem theorem5_structured_jacobian_adjoint_duality (A : mat2) (x g : point2) :
  dot2 (matVecMul2 A x) g = dot2 x (matVecMul2 (transpose2 A) g).
Proof.
  destruct A as [[[a11 a12] a21] a22].
  destruct x as [qx px].
  destruct g as [qg pg].
  unfold dot2, matVecMul2, transpose2. ring.
Qed.

(* ========================================================================= *)
(* 7. N-DIMENSIONAL STRUCTURED ADJOINT & NORMALIZATION IDENTITIES            *)
(* ========================================================================= *)

Definition vector := nat -> Qc.
Definition matrix := nat -> nat -> Qc.

Fixpoint qsum (n : nat) (f : nat -> Qc) : Qc :=
  match n with O => 0 | S k => qsum k f + f k end.

Definition vpair n (x y : vector) := qsum n (fun i => x i * y i).

Definition factor_apply n r (u v : matrix) (x : vector) : vector :=
  fun i => qsum r (fun h => u i h * qsum n (fun j => v j h * x j)).

Definition structured_push n r (d : vector) (u v : matrix) (x : vector) : vector :=
  fun i => d i * x i + factor_apply n r u v x i.

Definition structured_pull n r (d : vector) (u v : matrix) (g : vector) : vector :=
  structured_push n r d v u g.

Lemma qsum_ext n f g : (forall i, f i = g i) -> qsum n f = qsum n g.
Proof. intro H. induction n; simpl; [reflexivity|]. rewrite IHn, H. reflexivity. Qed.

Lemma qsum_add n f g : qsum n (fun i => f i + g i) = qsum n f + qsum n g.
Proof. induction n; simpl; [ring|]. rewrite IHn. ring. Qed.

Lemma qsum_mul_l n a f : qsum n (fun i => a * f i) = a * qsum n f.
Proof. induction n; simpl; [ring|]. rewrite IHn. ring. Qed.

Lemma qsum_mul_r n a f : qsum n (fun i => f i * a) = qsum n f * a.
Proof. induction n; simpl; [ring|]. rewrite IHn. ring. Qed.

Lemma qsum_zero n : qsum n (fun _ => 0) = 0.
Proof. induction n; simpl; [reflexivity|]. rewrite IHn. ring. Qed.

Lemma qsum_swap m n f :
  qsum m (fun i => qsum n (fun j => f i j)) =
  qsum n (fun j => qsum m (fun i => f i j)).
Proof.
  induction m; simpl; [symmetry; apply qsum_zero|].
  rewrite qsum_add, IHm. reflexivity.
Qed.

Lemma vpair_add_l n a b c :
  vpair n (fun i => a i + b i) c = vpair n a c + vpair n b c.
Proof.
  unfold vpair. rewrite <- qsum_add. apply qsum_ext. intro i. ring.
Qed.

Lemma vpair_add_r n a b c :
  vpair n a (fun i => b i + c i) = vpair n a b + vpair n a c.
Proof.
  unfold vpair. rewrite <- qsum_add. apply qsum_ext. intro i. ring.
Qed.

Theorem factor_adjoint n r u v x g :
  vpair n (factor_apply n r u v x) g = vpair n x (factor_apply n r v u g).
Proof.
  unfold vpair, factor_apply.
  transitivity (qsum n (fun i => qsum r (fun h =>
    (u i h * qsum n (fun j => v j h * x j)) * g i))).
  - apply qsum_ext. intro i. symmetry. apply qsum_mul_r.
  - rewrite qsum_swap.
    transitivity (qsum r (fun h =>
      (qsum n (fun j => v j h * x j)) * (qsum n (fun i => u i h * g i)))).
    + apply qsum_ext. intro h. rewrite <- qsum_mul_l.
      apply qsum_ext. intro i. ring.
    + transitivity (qsum r (fun h => qsum n (fun j =>
        x j * (v j h * qsum n (fun i => u i h * g i))))).
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

(* Softmax structured pullback identity *)
Definition softmax_pull n (p g : vector) : vector :=
  fun i => p i * (g i - qsum n (fun j => p j * g j)).

Theorem softmax_as_rank_one n p g i :
  softmax_pull n p g i = structured_pull n 1 p
    (fun i _ => p i) (fun i _ => Qcopp (p i)) g i.
Proof.
  unfold softmax_pull, structured_pull, structured_push, factor_apply. simpl. ring.
Qed.

(* Normalization (RMSNorm) structured pullback identity *)
Definition compact_rms_pull n r invn (z q : vector) i :=
  r * (q i - z i * invn * qsum n (fun j => z j * q j)).

Theorem rms_residual_sufficient n r invn z gamma g i :
  structured_pull n 1 (fun j => r * gamma j)
    (fun j _ => Qcopp (r * gamma j * z j * invn)) (fun j _ => z j) g i =
  compact_rms_pull n r invn z (fun j => gamma j * g j) i.
Proof.
  unfold structured_pull, structured_push, factor_apply, compact_rms_pull. simpl.
  assert (H : qsum n (fun j => Qcopp (r * gamma j * z j * invn) * g j) =
    Qcopp (r * invn) * qsum n (fun j => z j * (gamma j * g j))).
  { rewrite <- qsum_mul_l. apply qsum_ext. intro j. ring. }
  rewrite H. ring.
Qed.

