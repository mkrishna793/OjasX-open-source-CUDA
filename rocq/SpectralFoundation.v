From Stdlib Require Import QArith.Qcanon Ring Arith Lia.
From Ojasx Require Import Matrix GradientFactors.
Open Scope Qc_scope.

Module SpectralFoundation.
(* Matching primal and derivative are supplied as proved contracts. A local
   activation-error bound does not replace whole-program gradient admission. *)
Record PolynomialContract (I O G : Type) (original polynomial : I->O)
  (original_pull polynomial_pull : I->G) := {
  execute : I->O;
  execute_polynomial : forall x,execute x=polynomial x;
  reverse : I->G;
  reverse_polynomial : forall x,reverse x=polynomial_pull x;
  value_error : I->Qc;
  gradient_error : I->Qc;
  value_distance : O->O->Qc;
  gradient_distance : G->G->Qc;
  value_bound : forall x,value_distance (original x) (execute x)<=value_error x;
  gradient_bound : forall x,gradient_distance (original_pull x) (reverse x)<=gradient_error x
}.
End SpectralFoundation.

Theorem spectral_shared_projection k n features coefficients q i j :
 mm n (mm k features coefficients) q i j = mm k features (mm n coefficients q) i j.
Proof. apply mm_associative. Qed.

Theorem spectral_projection_gradient m s features coefficients p i j :
 mm m (transpose (mm s features coefficients)) p i j =
 mm s (transpose coefficients) (mm m (transpose features) p) i j.
Proof.
 transitivity (mm m (mm s (transpose coefficients) (transpose features)) p i j).
 - apply mm_ext_l. intros h l. apply mm_transpose.
 - apply mm_associative.
Qed.

Theorem spectral_common_gradient_contraction m n f c p i j :
 mm n f (mm m c p) i j = mm m (mm n f c) p i j.
Proof. symmetry. apply mm_associative. Qed.

Theorem spectral_loss_cotangent_error g e p dp :
 (g+e)*(p+dp)-g*p=e*p+g*dp+e*dp.
Proof. ring. Qed.

Theorem spectral_cost_guard common candidate reference :
 (candidate<=reference)%nat -> (common+candidate<=common+reference)%nat.
Proof. lia. Qed.

Theorem spectral_bias_fold_quadratic x w b c :
 c*(x*w+b)*(x*w+b)=x*x*(c*w*w)+x*(c*(1+1)*w*b)+c*b*b.
Proof. ring. Qed.

Theorem spectral_bias_fold_quadratic_grad_w x w b c p :
 p*c*(1+1)*(x*w+b)*x = p*(x*x*c*(1+1)*w+x*c*(1+1)*b).
Proof. ring. Qed.

Theorem spectral_bias_fold_quadratic_grad_b x w b c p :
 p*c*(1+1)*(x*w+b)=p*(x*c*(1+1)*w+c*(1+1)*b).
Proof. ring. Qed.
