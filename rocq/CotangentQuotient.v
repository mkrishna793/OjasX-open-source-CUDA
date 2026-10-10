From Stdlib Require Import QArith.Qcanon Ring.
From Ojasx Require Import Matrix ObjectiveCompiler.
Open Scope Qc_scope.

Theorem observable_scalar_kernel (a b observation : Qc)
 (h : observation*a=observation*b) : observation*(a-b)=0.
Proof. transitivity (observation*a-observation*b); [ring|]. rewrite h. ring. Qed.

Theorem defect_scalar_joint (x w dx dw db c e g : Qc) :
 ((c+e)*(dx*w+x*dw+db))*g =
 dx*(c*g*w+e*g*w)+dw*(x*c*g+x*e*g)+db*(c*g+e*g).
Proof. ring. Qed.

Theorem defect_value_certificate (value z c offset : Qc) :
 value=c*z+offset+(value-(c*z+offset)).
Proof. ring. Qed.

Theorem defect_slope_certificate (slope c : Qc) : slope=c+(slope-c).
Proof. ring. Qed.

Theorem defect_input_contraction n r w p q defect i k :
 mm n (fun i j => mm r p (transpose q) i j+defect i j) (transpose w) i k =
 mm r p (transpose (mm n w q)) i k+mm n defect (transpose w) i k.
Proof.
 unfold mm at 1. transitivity
 (qsum n (fun j => mm r p (transpose q) i j*transpose w j k)+
  qsum n (fun j => defect i j*transpose w j k)).
 - rewrite <- qsum_add. apply qsum_ext. intro j. ring.
 - change (mm n (mm r p (transpose q)) (transpose w) i k+mm n defect (transpose w) i k =
     mm r p (transpose (mm n w q)) i k+mm n defect (transpose w) i k).
   rewrite linear_pair_dx. reflexivity.
Qed.

Theorem defect_weight_contraction m r x p q defect k j :
 mm m (transpose x) (fun i j => mm r p (transpose q) i j+defect i j) k j =
 mm r (mm m (transpose x) p) (transpose q) k j+mm m (transpose x) defect k j.
Proof.
 unfold mm at 1. transitivity
 (qsum m (fun i => transpose x k i*mm r p (transpose q) i j)+
  qsum m (fun i => transpose x k i*defect i j)).
 - rewrite <- qsum_add. apply qsum_ext. intro i. ring.
 - change (mm m (transpose x) (mm r p (transpose q)) k j+mm m (transpose x) defect k j =
     mm r (mm m (transpose x) p) (transpose q) k j+mm m (transpose x) defect k j).
   rewrite <- mm_associative. reflexivity.
Qed.

Theorem streamed_nonlinear_moments m n (f : nat -> Qc) :
 qsum (m+n) f=qsum m f+qsum n (fun i => f (m+i)%nat).
Proof. apply objective_partition. Qed.
