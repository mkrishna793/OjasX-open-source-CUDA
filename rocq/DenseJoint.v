From Stdlib Require Import QArith.Qcanon Ring.
From Ojasx Require Import Matrix.
Open Scope Qc_scope.

(* Computable rational algebra; no assertion about transcendentals or IEEE. *)
Record ResidualContract (E R V G : Type) := {
 encode : E -> R;
 value : E -> V;
 pull : E -> G -> E;
 decode_value : R -> V;
 decode_pull : R -> G -> E;
 value_exact : forall x, decode_value (encode x)=value x;
 pull_exact : forall x g, decode_pull (encode x) g=pull x g
}.
Arguments encode {E R V G}.
Arguments pull {E R V G}.
Arguments decode_pull {E R V G}.
Arguments pull_exact {E R V G}.
Theorem dense_residual_pull E R V G (c : ResidualContract E R V G) x g :
 decode_pull c (encode c x) g=pull c x g.
Proof. apply pull_exact. Qed.

Theorem dense_coordinate_pair (s dz g : Qc) : g*(s*dz)=(s*g)*dz.
Proof. ring. Qed.
Theorem dense_parameters (x w q value slope dx dw db dq g : Qc) :
 g*(slope*(dx*w+x*dw+db)*q+value*dq)=
 dx*(slope*g*q*w)+dw*(x*slope*g*q)+db*(slope*g*q)+dq*(value*g).
Proof. ring. Qed.
Theorem dense_swiglu_parameters
 (x w u q value slope up dx dw du db dub dq g : Qc) :
 g*((slope*(dx*w+x*dw+db)*up+value*(dx*u+x*du+dub))*q+value*up*dq)=
 dx*(slope*g*up*q*w+value*g*q*u)+dw*(x*slope*g*up*q)+db*(slope*g*up*q)+
 du*(x*value*g*q)+dub*(value*g*q)+dq*(value*up*g).
Proof. ring. Qed.
Theorem dense_rank7_scaled_saving (m k n : Qc) :
 let c2 := 1+1 in let c3 := c2+1 in let c4 := c3+1 in
 let c5 := c4+1 in let c6 := c5+1 in let c8 := c4+c4 in
 let c9 := c8+1 in let c12 := c4*c3 in let c21 := c6*c3+c3 in
 c4*(c6*m*k*n-m*k-k*n-m*n)-
 (c21*m*k*n+c8*m*k+c8*k*n+c5*m*n)=
 c3*m*k*n-c12*m*k-c12*k*n-c9*m*n.
Proof. cbv beta iota zeta; ring. Qed.
Theorem dense_residual_tape (elements bytes : Qc) :
 (1+1)*elements*bytes-elements*bytes=elements*bytes.
Proof. ring. Qed.
Theorem dense_swiglu_tape (elements bytes : Qc) :
 (1+1+1+1)*elements*bytes-(1+1)*elements*bytes=(1+1)*elements*bytes.
Proof. ring. Qed.
Theorem dense_bias_pair n (g : nat -> Qc) db :
 qsum n (fun i => g i*db)=db*qsum n g.
Proof. induction n; simpl; [ring|rewrite IHn; ring]. Qed.
