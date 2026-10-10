From Stdlib Require Import NArith Bool List Ring QArith.Qcanon.
From Ojasx Require Import Matrix.
Import ListNotations.
Open Scope N_scope.

(* Leading GEMM FLOPs, two operations per multiply-accumulate.
   This is not an elapsed-time cost model. All arithmetic is unbounded N. *)
Definition native_work m k n p : N := 6*m*k*n + 6*m*n*p.
Definition contracted_work m k n p : N := 6*p*k*(m+n).

(* Each node below is evaluated once in the specified joint forward/reverse
   DAG. Materializing/reusing those nodes is a backend obligation. *)
Definition gemm_shape := (N*N*N)%type.
Definition native_schedule m k n p : list gemm_shape :=
  [(m,k,n); (m,n,p); (m,p,n); (m,n,k); (k,m,n); (n,m,p)].
Definition contracted_schedule m k n p : list gemm_shape :=
  [(k,n,p); (m,k,p); (m,p,k); (k,m,p); (k,p,n); (n,k,p)].
Fixpoint schedule_work (nodes : list gemm_shape) : N :=
  match nodes with
  | [] => 0
  | (r,h,c)::rest => 2*r*h*c + schedule_work rest
  end.

Theorem native_schedule_cost m k n p :
  schedule_work (native_schedule m k n p) = native_work m k n p.
Proof. unfold native_schedule. cbn [schedule_work]. unfold native_work. ring. Qed.
Theorem contracted_schedule_cost m k n p :
  schedule_work (contracted_schedule m k n p) = contracted_work m k n p.
Proof. unfold contracted_schedule. cbn [schedule_work]. unfold contracted_work. ring. Qed.

Definition native_tape m n : N := m*n.
Definition contracted_tape k p : N := k*p.
Definition choose_contracted m k n p : bool :=
  (contracted_work m k n p <=? native_work m k n p) &&
  (contracted_tape k p <=? native_tape m n).
Definition selected_work m k n p :=
  if choose_contracted m k n p then contracted_work m k n p else native_work m k n p.
Definition selected_tape m k n p :=
  if choose_contracted m k n p then contracted_tape k p else native_tape m n.

Theorem planned_work_no_increase m k n p : selected_work m k n p <= native_work m k n p.
Proof.
  unfold selected_work. destruct (choose_contracted m k n p) eqn:H; [|apply N.le_refl].
  unfold choose_contracted in H. apply andb_true_iff in H. destruct H as [H _].
  apply N.leb_le. exact H.
Qed.
Theorem planned_tape_no_increase m k n p : selected_tape m k n p <= native_tape m n.
Proof.
  unfold selected_tape. destruct (choose_contracted m k n p) eqn:H; [|apply N.le_refl].
  unfold choose_contracted in H. apply andb_true_iff in H. destruct H as [_ H].
  apply N.leb_le. exact H.
Qed.

Definition planned_forward (m k n p : nat) x a b : matrix :=
  if choose_contracted (N.of_nat m) (N.of_nat k) (N.of_nat n) (N.of_nat p)
  then mm k x (mm n a b) else mm n (mm k x a) b.

Theorem planned_forward_correct m k n p x a b i j :
  planned_forward m k n p x a b i j = mm n (mm k x a) b i j.
Proof.
  unfold planned_forward. destruct (choose_contracted _ _ _ _);
    [symmetry; apply linear_pair_forward|reflexivity].
Qed.

(* One shared narrow core T=X^T G provides both parameter gradients. *)
Definition planned_reverse (m k n p : nat) x a b g : matrix * matrix * matrix :=
  if choose_contracted (N.of_nat m) (N.of_nat k) (N.of_nat n) (N.of_nat p)
  then let core := mm m (transpose x) g in
    (mm p g (transpose (mm n a b)), mm p core (transpose b), mm k (transpose a) core)
  else (mm n (mm p g (transpose b)) (transpose a),
    mm m (transpose x) (mm p g (transpose b)), mm m (transpose (mm k x a)) g).

Theorem planned_reverse_dx m k n p x a b g i j :
  fst (fst (planned_reverse m k n p x a b g)) i j =
  mm n (mm p g (transpose b)) (transpose a) i j.
Proof.
  unfold planned_reverse. destruct (choose_contracted _ _ _ _); simpl;
    [symmetry; apply linear_pair_dx|reflexivity].
Qed.
Theorem planned_reverse_da m k n p x a b g i j :
  snd (fst (planned_reverse m k n p x a b g)) i j =
  mm m (transpose x) (mm p g (transpose b)) i j.
Proof.
  unfold planned_reverse. destruct (choose_contracted _ _ _ _); simpl;
    [symmetry; apply linear_pair_da|reflexivity].
Qed.
Theorem planned_reverse_db m k n p x a b g i j :
  snd (planned_reverse m k n p x a b g) i j = mm m (transpose (mm k x a)) g i j.
Proof.
  unfold planned_reverse. destruct (choose_contracted _ _ _ _); simpl;
    [symmetry; apply linear_pair_db|reflexivity].
Qed.

Close Scope N_scope.
Open Scope Qc_scope.
Definition matrix_sgd (w dw : matrix) (eta : Qc) : matrix :=
  fun i j => w i j - eta*dw i j.
Definition planned_parameter_updates m k n p x a b g eta : matrix * matrix :=
  (matrix_sgd a (snd (fst (planned_reverse m k n p x a b g))) eta,
   matrix_sgd b (snd (planned_reverse m k n p x a b g)) eta).

Theorem planned_sgd_a m k n p x a b g eta i j :
  fst (planned_parameter_updates m k n p x a b g eta) i j =
  a i j - eta * mm m (transpose x) (mm p g (transpose b)) i j.
Proof. unfold planned_parameter_updates, matrix_sgd. simpl. rewrite planned_reverse_da. reflexivity. Qed.
Theorem planned_sgd_b m k n p x a b g eta i j :
  snd (planned_parameter_updates m k n p x a b g eta) i j =
  b i j - eta * mm m (transpose (mm k x a)) g i j.
Proof. unfold planned_parameter_updates, matrix_sgd. simpl. rewrite planned_reverse_db. reflexivity. Qed.
