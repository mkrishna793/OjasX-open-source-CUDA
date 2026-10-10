From Stdlib Require Import QArith.Qcanon Ring.
From Ojasx Require Import Matrix Structured Contracts.
Open Scope Qc_scope.

Definition structured_contract (n r : nat) (d : vector) (u v : matrix) :
  adjoint_contract vector vector (vpair n) (vpair n).
Proof.
  refine {|push := structured_push n r d u v; pull := structured_pull n r d u v|}.
  apply structured_adjoint.
Defined.

Definition matrix_weight_contract m k n (a : matrix) :
  adjoint_contract matrix matrix (pairing k n) (pairing m n).
Proof.
  refine {|push := fun db => mm k a db; pull := fun g => mm m (transpose a) g|}.
  intros db g. exact (mm_weight_adjoint m k n a db g).
Defined.

Definition matrix_input_contract m k n (b : matrix) :
  adjoint_contract matrix matrix (pairing m k) (pairing m n).
Proof.
  refine {|push := fun da => mm k da b; pull := fun g => mm n g (transpose b)|}.
  intros da g. exact (mm_input_adjoint m k n da b g).
Defined.

Theorem structured_composition n r s d u v e a b dx g :
  vpair n (structured_push n s e a b (structured_push n r d u v dx)) g =
  vpair n dx (structured_pull n r d u v (structured_pull n s e a b g)).
Proof. rewrite !structured_adjoint. reflexivity. Qed.

Theorem shared_loss_pullback n r d u v g h i :
  pull (structured_contract n r d u v) (fun j => g j+h j) i =
  pull (structured_contract n r d u v) g i + pull (structured_contract n r d u v) h i.
Proof. apply structured_pull_additive. Qed.

Theorem branch_accumulates n r s d u v e a b dx g h :
  vpair n (structured_push n r d u v dx) g +
  vpair n (structured_push n s e a b dx) h =
  vpair n dx (fun i => structured_pull n r d u v g i+structured_pull n s e a b h i).
Proof. rewrite vpair_add_r, !structured_adjoint. reflexivity. Qed.
