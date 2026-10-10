From Stdlib Require Import QArith.Qcanon Ring Arith Lia.
From Ojasx Require Import Matrix.
Open Scope Qc_scope.

Theorem factor_joint_dagger a x da dx db slope g :
 (slope*(a*dx+da*x+db))*g = dx*(a*slope*g)+da*(x*slope*g)+db*(slope*g).
Proof. ring. Qed.

Theorem factor_hadamard_product r s p q u v :
 qsum r (fun h=>p h*q h)*qsum s (fun t=>u t*v t) =
 qsum r (fun h=>qsum s (fun t=>(p h*u t)*(q h*v t))).
Proof.
 rewrite <- qsum_mul_r. apply qsum_ext. intro h.
 rewrite <- qsum_mul_l. apply qsum_ext. intro t. ring.
Qed.

Theorem factor_bias_pullback m r (p q : matrix) (j : nat) :
 qsum m (fun i=>qsum r (fun h=>p i h*q j h)) =
 qsum r (fun h=>qsum m (fun i=>p i h)*q j h).
Proof. rewrite qsum_swap. apply qsum_ext. intro h. apply qsum_mul_r. Qed.

Theorem factor_input_pullback n r w p q i j :
 mm n (mm r p (transpose q)) (transpose w) i j =
 mm r p (transpose (mm n w q)) i j.
Proof.
 rewrite mm_associative. apply mm_ext_r. intros a b.
 symmetry. apply mm_transpose.
Qed.

Theorem factor_weight_pullback m r x p q i j :
 mm m (transpose x) (mm r p (transpose q)) i j =
 mm r (mm m (transpose x) p) (transpose q) i j.
Proof. symmetry. apply mm_associative. Qed.

Theorem factor_relu_rank_growth a b c d :
 a*c=1 -> a*d=0 -> b*c=0 -> b*d=1 -> False.
Proof.
 intros Hac Had Hbc Hbd.
 assert (E : (a*c)*(b*d)=(a*d)*(b*c)) by ring.
 rewrite Hac, Had, Hbc, Hbd in E. discriminate E.
Qed.

Theorem factor_cost_guard common candidate native :
 (candidate <= native)%nat -> (common+candidate <= common+native)%nat.
Proof. lia. Qed.
