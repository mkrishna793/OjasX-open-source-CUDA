From Stdlib Require Import QArith.Qcanon Ring Arith Lia.
From Ojasx Require Import Matrix ObjectiveCompiler.
Open Scope Qc_scope.

(* Exact rational cotangent algebra. Analytic exp/log derivatives are checked
   in Lean. No transcendental, IEEE or universal speed claim is assumed here. *)
Theorem loss_weighted_primal (a z logpartition : Qc) :
 -(a*(z-logpartition)) = a*logpartition-a*z.
Proof. ring. Qed.

Theorem loss_local_cotangent (mass probability weight direction : Qc) :
 mass*(probability*direction)-weight*direction =
 (mass*probability-weight)*direction.
Proof. ring. Qed.

Theorem loss_cotangent_sum n (p a : nat -> Qc) :
 qsum n (fun i => qsum n a*p i-a i) = qsum n a*qsum n p-qsum n a.
Proof.
 assert (subsum : forall k f g,
   qsum k (fun i => f i-g i) = qsum k f-qsum k g).
 { intros k f g. induction k; simpl; [ring|]. rewrite IHk. ring. }
 rewrite subsum, qsum_mul_l. reflexivity.
Qed.

Theorem loss_cotangent_zero_sum n p a (hp : qsum n p = 1) :
 qsum n (fun i => qsum n a*p i-a i) = 0.
Proof. rewrite loss_cotangent_sum, hp. ring. Qed.

Theorem loss_selected_work (n a k v : Qc) :
 (1+1+1+1+1+1)*n*k*v-(1+1+1+1+1+1)*a*k*v = (1+1+1+1+1+1)*(n-a)*k*v.
Proof. ring. Qed.

Theorem loss_replay_work (a k v : Qc) :
 (1+1+1+1+1+1)*a*k*v+(1+1)*a*k*v = (1+1+1+1+1+1+1+1)*a*k*v.
Proof. ring. Qed.

Theorem loss_partition_replay m n (f : nat -> Qc) :
 qsum (m+n) f = qsum m f+qsum n (fun i => f (m+i)%nat).
Proof. apply objective_partition. Qed.

Close Scope Qc_scope.
Theorem loss_chunk_storage (n b v : nat) (hb : b <= n) : b*v <= n*v.
Proof. nia. Qed.

Theorem loss_resident_parameter_bound (p g m master : nat) : p <= p+g+m+master.
Proof. lia. Qed.
