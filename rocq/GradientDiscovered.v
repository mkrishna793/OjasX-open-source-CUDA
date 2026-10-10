From Stdlib Require Import QArith.Qcanon Ring.
Open Scope Qc_scope.
Theorem factor_discovered_degree_1_joint (x a b upstream dx da db : Qc) : ((db * upstream) + (a * dx * upstream) + (da * upstream * x)) = (upstream * (db + (a * dx) + (da * x))).
Proof. ring. Qed.
Theorem factor_discovered_degree_2_joint (x a b upstream dx da db : Qc) : ((db * upstream * (((1 + 1) * b) + ((1 + 1) * a * x))) + ((1 + 1) * a * dx * upstream * (b + (a * x))) + ((1 + 1) * da * upstream * x * (b + (a * x)))) = ((1 + 1) * upstream * (b + (a * x)) * (db + (a * dx) + (da * x))).
Proof. ring. Qed.
Theorem factor_discovered_degree_3_joint (x a b upstream dx da db : Qc) : (((1 + 1 + 1) * db * upstream * ((b + (a * x)) * (b + (a * x)))) + ((1 + 1 + 1) * a * dx * upstream * ((b + (a * x)) * (b + (a * x)))) + ((1 + 1 + 1) * da * upstream * x * ((b + (a * x)) * (b + (a * x))))) = ((1 + 1 + 1) * upstream * ((b + (a * x)) * (b + (a * x))) * (db + (a * dx) + (da * x))).
Proof. ring. Qed.
