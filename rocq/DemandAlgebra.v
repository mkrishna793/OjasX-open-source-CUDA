From Stdlib Require Import QArith.Qcanon Ring.
From Ojasx Require Import Matrix ObjectiveCompiler.
Open Scope Qc_scope.

Theorem demand_zero (a : Qc) : a*0=0.
Proof. ring. Qed.

Theorem demand_joint (x w b q slope value dx dw db dq g : Qc) :
 g*(slope*(dx*w+x*dw+db)*q+value*dq)=
 dx*(slope*g*q*w)+dw*(x*slope*g*q)+db*(slope*g*q)+dq*(value*g).
Proof. ring. Qed.

Theorem demand_partition m n (f : nat -> Qc) :
 qsum (m+n) f=qsum m f+qsum n (fun i=>f (m+i)%nat).
Proof. apply objective_partition. Qed.

Theorem demand_work (m k n r a c : Qc) :
 (1+1+1+1+1+1)*m*n*(k+r)-((1+1)*m*n*(k+r)+(1+1+1+1+1+1)*a*n*k+(1+1+1+1)*a*n*c+(1+1)*m*r)=
 (1+1+1+1)*m*n*(k+r)-(1+1+1+1+1+1)*a*n*k-(1+1+1+1)*a*n*c-(1+1)*m*r.
Proof. ring. Qed.
