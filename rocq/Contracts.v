From Stdlib Require Import QArith.Qcanon List.
Open Scope Qc_scope.

(* A contract for the adjoint of a supplied local derivative. Establishing
   that [push] is the derivative of a primal operation is a leaf obligation. *)
Record adjoint_contract (X Y : Type) (px : X -> X -> Qc) (py : Y -> Y -> Qc) := {
  push : X -> Y;
  pull : Y -> X;
  adjoint_law : forall dx gy, py (push dx) gy = px dx (pull gy)
}.
Arguments push {X Y px py} _ _.
Arguments pull {X Y px py} _ _.
Arguments adjoint_law {X Y px py} _ _ _.

Definition compose_contract {X Y Z px py pz}
  (f : adjoint_contract X Y px py) (g : adjoint_contract Y Z py pz) :
  adjoint_contract X Z px pz.
Proof.
  refine {| push := fun dx => push g (push f dx); pull := fun gz => pull f (pull g gz) |}.
  intros dx gz. rewrite (adjoint_law g), (adjoint_law f). reflexivity.
Defined.

Definition identity_contract {X} (px : X -> X -> Qc) : adjoint_contract X X px px.
Proof. refine {|push := fun x => x; pull := fun x => x|}. reflexivity. Defined.

Theorem composed_pullback {X Y Z px py pz}
  (f : adjoint_contract X Y px py) (g : adjoint_contract Y Z py pz) gz :
  pull (compose_contract f g) gz = pull f (pull g gz).
Proof. reflexivity. Qed.

Definition pair_pairing {Y Z} (py : Y -> Y -> Qc) (pz : Z -> Z -> Qc)
  (a b : Y*Z) := py (fst a) (fst b) + pz (snd a) (snd b).

Definition branch_contract {X Y Z px py pz} (add : X -> X -> X)
  (add_law : forall x a b, px x (add a b) = px x a + px x b)
  (f : adjoint_contract X Y px py) (g : adjoint_contract X Z px pz) :
  adjoint_contract X (Y*Z) px (pair_pairing py pz).
Proof.
  refine {|push := fun x => (push f x, push g x);
    pull := fun yz => add (pull f (fst yz)) (pull g (snd yz))|}.
  intros x [y z]. unfold pair_pairing. simpl.
  rewrite add_law, (adjoint_law f), (adjoint_law g). reflexivity.
Defined.

(* A saved residual is sufficient precisely when it reproduces the pullback.
   Storage reduction itself, and a derivative contract, must be established
   separately for each primitive. *)
Theorem residual_substitution_composes {X Y Z S}
  (save : Y -> S) (back : Y -> Z -> Y) (recover : S -> Z -> Y)
  (before : X -> Y -> X) (H : forall y g, recover (save y) g = back y g) x y g :
  before x (recover (save y) g) = before x (back y g).
Proof. rewrite H. reflexivity. Qed.

Theorem cotangent_accumulation {X Y px py} (addx : X -> X -> X) (addy : Y -> Y -> Y)
  (f : adjoint_contract X Y px py)
  (Hx : forall x a b, px x (addx a b) = px x a + px x b)
  (Hy : forall y a b, py y (addy a b) = py y a + py y b)
  (separate : forall a b, (forall x, px x a = px x b) -> a = b) g h :
  pull f (addy g h) = addx (pull f g) (pull f h).
Proof.
  apply separate. intro x. rewrite Hx, <- !(adjoint_law f), Hy. reflexivity.
Qed.

Fixpoint repeat_contract {X px} (steps : nat) (f : adjoint_contract X X px px) :
  adjoint_contract X X px px :=
  match steps with O => identity_contract px | S n => compose_contract f (repeat_contract n f) end.

Theorem repeated_adjoint {X px} steps (f : adjoint_contract X X px px) dx g :
  px (push (repeat_contract steps f) dx) g = px dx (pull (repeat_contract steps f) g).
Proof. apply adjoint_law. Qed.
