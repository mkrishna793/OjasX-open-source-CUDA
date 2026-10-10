From Stdlib Require Import QArith.Qcanon Ring List.
Open Scope Qc_scope.
Definition affine_action (a b x : Qc) := a*x+b.
Theorem chart_action_compose a b c d x :
 affine_action c d (affine_action a b x) = affine_action (c*a) (c*b+d) x.
Proof. unfold affine_action; ring. Qed.
Theorem chart_action_associative a b c d e f x :
 (e*c*a)*x+(e*(c*b+d)+f) = e*(c*(a*x+b)+d)+f.
Proof. ring. Qed.
Theorem chart_action_all_parameters a b c x da db dc dd dx g :
 (c*a*dx+c*x*da+c*db+(a*x+b)*dc+dd)*g =
 dx*(a*c*g)+da*(x*c*g)+db*(c*g)+dc*((a*x+b)*g)+dd*g.
Proof. ring. Qed.
Theorem koopman_triangular_closure x y :
 ((1+1)*x)*((1+1)*x) = (1+1+1+1)*(x*x) /\
 (1+1+1)*y+x*x = x*x+(1+1+1)*y.
Proof. split; ring. Qed.
Theorem output_not_parameter_sufficient :
 ~ exists recover : Qc -> Qc, forall x : Qc, recover (0*x+1) = x.
Proof.
 intros [recover H]. specialize (H 0) as H0. specialize (H 1) as H1.
 replace (0*0+1) with 1 in H0 by ring.
 replace (0*1+1) with 1 in H1 by ring.
 rewrite H0 in H1. discriminate.
Qed.

Theorem koopman_trainable_closure a b c x y :
 (a*x)*(a*x) = (a*a)*(x*x) /\
 b*y+c*(x*x) = c*(x*x)+b*y.
Proof. split; ring. Qed.

Fixpoint run_actions {P E : Type} (f : P -> E -> E) (ps : list P) (x : E) : E :=
 match ps with nil => x | cons p rest => run_actions f rest (f p x) end.
Theorem coordinate_sequence_commutes {P E U : Type}
 (f : P -> E -> E) (g : P -> U -> U) (encode : E -> U)
 (H : forall p x, encode (f p x) = g p (encode x)) ps x :
 encode (run_actions f ps x) = run_actions g ps (encode x).
Proof.
 revert x. induction ps as [|p rest IH]; intro x; simpl; [reflexivity|].
 rewrite IH, H. reflexivity.
Qed.
Theorem coordinate_sequence_recovers {P E U : Type}
 (f : P -> E -> E) (g : P -> U -> U) (encode : E -> U) (decode : U -> E)
 (H : forall p x, encode (f p x) = g p (encode x))
 (R : forall x, decode (encode x) = x) ps x :
 decode (run_actions g ps (encode x)) = run_actions f ps x.
Proof. rewrite <- (coordinate_sequence_commutes f g encode H ps x). apply R. Qed.
