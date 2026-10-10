From Stdlib Require Import QArith.Qcanon Ring.
Open Scope Qc_scope.
Theorem coordinate_discovered_primal (x y a b c u v : Qc) : ((v * ((b * y) + (c * (x * x)))) + (a * u * x)) = ((v * ((b * y) + (c * (x * x)))) + (a * u * x)).
Proof. ring. Qed.
Theorem coordinate_discovered_gradient_x (x y a b c u v : Qc) : ((a * u) + ((1 + 1) * c * v * x)) = ((a * u) + ((1 + 1) * c * v * x)).
Proof. ring. Qed.
Theorem coordinate_discovered_gradient_y (x y a b c u v : Qc) : (b * v) = (b * v).
Proof. ring. Qed.
Theorem coordinate_discovered_gradient_a (x y a b c u v : Qc) : (u * x) = (u * x).
Proof. ring. Qed.
Theorem coordinate_discovered_gradient_b (x y a b c u v : Qc) : (v * y) = (v * y).
Proof. ring. Qed.
Theorem coordinate_discovered_gradient_c (x y a b c u v : Qc) : (v * (x * x)) = (v * (x * x)).
Proof. ring. Qed.
Theorem coordinate_discovered_joint_dagger (x y a b c u v dx dy da db dc : Qc) : ((u * ((a * dx) + (da * x))) + (v * ((b * dy) + (db * y) + (dc * (x * x)) + ((1 + 1) * c * dx * x)))) = ((dx * ((a * u) + ((1 + 1) * c * v * x))) + (b * dy * v) + (da * u * x) + (db * v * y) + (dc * v * (x * x))).
Proof. ring. Qed.
