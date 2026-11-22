From Stdlib Require Import QArith.Qcanon Ring Field Bool.
From Ojasx Require Import Matrix.
Open Scope Qc_scope.

(* Exact rational models of the executable first-order core. These are
   algebraic adjoint identities, not claims about rounded floating arithmetic. *)
Theorem core_linear_pairing (x w dx dw db g : Qc) :
  g*(dx*w+x*dw+db)=dx*(g*w)+dw*(x*g)+db*g.
Proof. ring. Qed.

Theorem core_add_pullback (a b g : Qc) : g*(a+b)=a*g+b*g.
Proof. ring. Qed.

Theorem core_sum_pullback n (x : nat -> Qc) (g : Qc) :
  g*qsum n x=qsum n (fun i=>x i*g).
Proof.
  rewrite <- qsum_mul_l. apply qsum_ext. intro i. ring.
Qed.

Theorem core_fanout_accumulation (x g h : Qc) : x*g+x*h=x*(g+h).
Proof. ring. Qed.

(* Raw derivative inputs and normalized Taylor coefficients: the order-four
   convolution is exactly the binomial form of Leibniz' product rule. *)
#[local] Definition q2 : Qc := 1 + 1.
#[local] Definition q3 : Qc := 1 + 1 + 1.
#[local] Definition q4 : Qc := q2 * q2.
#[local] Definition q6 : Qc := q2 * q3.
#[local] Definition q24 : Qc := q4 * q6.

Theorem core_normalized_product_order_four
  (a0 a1 a2 a3 a4 b0 b1 b2 b3 b4 : Qc) :
  q24*(a0*(b4/q24)+a1*(b3/q6)+(a2/q2)*(b2/q2)+(a3/q6)*b1+(a4/q24)*b0)=
  a0*b4+q4*a1*b3+q6*a2*b2+q4*a3*b1+a4*b0.
Proof. unfold q24, q6, q4, q3, q2. field; repeat split; vm_compute; discriminate. Qed.

Theorem core_jet_horner_order_four (d c0 c1 c2 c3 c4 : Qc) :
  c0+d*(c1+d*(c2+d*(c3+d*c4)))=
  c0+c1*d+c2*(d*d)+c3*(d*d*d)+c4*(d*d*d*d).
Proof. ring. Qed.

Definition linearDemand (rx rw rb : bool) (x w dx dw db g : Qc) : Qc :=
  (if rx then dx*(g*w) else 0)+
  (if rw then dw*(x*g) else 0)+
  (if rb then db*g else 0).

Theorem core_linear_demand_all (x w dx dw db g : Qc) :
  linearDemand true true true x w dx dw db g=g*(dx*w+x*dw+db).
Proof. unfold linearDemand. ring. Qed.

Theorem core_linear_demand_weight_only (x w dx dw db g : Qc) :
  linearDemand false true false x w dx dw db g=dw*(x*g).
Proof. unfold linearDemand. ring. Qed.

Theorem core_linear_demand_input_only (x w dx dw db g : Qc) :
  linearDemand true false false x w dx dw db g=dx*(g*w).
Proof. unfold linearDemand. ring. Qed.

Theorem core_linear_demand_none (x w dx dw db g : Qc) :
  linearDemand false false false x w dx dw db g=0.
Proof. unfold linearDemand. ring. Qed.

Close Scope Qc_scope.
