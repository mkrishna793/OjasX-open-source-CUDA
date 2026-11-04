Require Import QArith.Qcanon Ring Lia Coq.Lists.List.
Import ListNotations.
Open Scope Qc_scope.

(* Exact rational M0/M1 representation laws. No IEEE or allocator claim. *)
Record uniform (I : Type) := Uniform { scalar : Qc }.
Arguments Uniform {I} _.
Arguments scalar {I} _.

Definition decode {I : Type} (u : uniform I) : I -> Qc :=
  fun _ => scalar u.
Definition materialize {I : Type} (u : uniform I) : I -> Qc := decode u.
Definition from_scalar {I : Type} (g : Qc) : uniform I := Uniform g.
Definition scale {I : Type} (s : Qc) (u : uniform I) : uniform I :=
  Uniform (s * scalar u).
Definition mean {I : Type} (n : Qc) (u : uniform I) : uniform I :=
  Uniform (scalar u / n).
Definition add_uniform {I : Type} (u v : uniform I) : uniform I :=
  Uniform (scalar u + scalar v).
Definition reindex {I J : Type} (f : J -> I) (u : uniform I) : uniform J :=
  Uniform (scalar u).

Theorem structured_sum_uniform {I : Type} (g : Qc) (i : I) :
  decode (from_scalar g) i = g.
Proof. reflexivity. Qed.

Theorem structured_mean_uniform {I : Type} (n g : Qc) (i : I) :
  n <> 0 -> decode (mean n (from_scalar g)) i = g/n.
Proof. intros Hn. reflexivity. Qed.

Theorem structured_uniform_add {I : Type} (u v : uniform I) (i : I) :
  decode (add_uniform u v) i = decode u i + decode v i.
Proof. reflexivity. Qed.

Theorem structured_uniform_reindex {I J K : Type}
  (f : J -> I) (g : K -> J) (u : uniform I) (k : K) :
  decode (reindex (fun x => f (g x)) u) k =
  decode (reindex g (reindex f u)) k.
Proof. reflexivity. Qed.

Theorem structured_reindex_identity {I : Type} (u : uniform I) (i : I) :
  decode (reindex (fun x => x) u) i = decode u i.
Proof. reflexivity. Qed.

Theorem structured_materialize_identity {I : Type} (u : uniform I) (i : I) :
  materialize u i = decode u i.
Proof. reflexivity. Qed.

Close Scope Qc_scope.

(* A region with extents [n_1,...,n_k] stores k scalar witnesses. Its logical
   dense bytes are e*sum(n_i), its managed bytes are e*k, and its exact saving
   is e*sum(n_i-1). Nonempty extents make the natural-number subtraction exact. *)
Definition region_logical (e : nat) (sizes : list nat) : nat :=
  e * fold_right Nat.add 0 sizes.
Definition region_managed (e : nat) (sizes : list nat) : nat :=
  e * length sizes.
Definition region_saving (e : nat) (sizes : list nat) : nat :=
  e * fold_right Nat.add 0 (map Nat.pred sizes).

Theorem structured_uniform_region_storage (e : nat) (sizes : list nat) :
  (forall n, In n sizes -> n > 0) ->
  region_logical e sizes = region_managed e sizes + region_saving e sizes.
Proof.
  induction sizes as [| n sizes IH]; intros Hnonempty.
  - unfold region_logical, region_managed, region_saving. simpl. lia.
  - unfold region_logical, region_managed, region_saving in *. simpl in *.
    assert (Hn : n > 0) by (apply Hnonempty; left; reflexivity).
    assert (Htail : forall m, In m sizes -> m > 0) by
      (intros m Hm; apply Hnonempty; right; exact Hm).
    specialize (IH Htail). destruct n; [lia|]. simpl in *. nia.
Qed.
