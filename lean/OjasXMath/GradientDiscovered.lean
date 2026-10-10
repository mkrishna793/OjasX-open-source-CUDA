import OjasXMath.GradientFactors
import Mathlib.Analysis.Calculus.Deriv.Pow
namespace OjasXMath
theorem factor_discovered_degree_1_joint (x a b upstream dx da db : ℝ) : a*dx*upstream + da*upstream*x + db*upstream = upstream*(a*dx + da*x + db) := by ring
theorem factor_discovered_degree_1_derivative_x (x a b upstream : ℝ) :
 HasDerivAt (fun t : ℝ => upstream*(a*t + b)) (a*upstream) x := by
 convert ((hasDerivAt_const x (upstream)).mul ((hasDerivAt_const x (b)).add ((hasDerivAt_const x (a)).mul (hasDerivAt_id x)))) using 1 <;> (try funext t) <;> dsimp <;> ring
theorem factor_discovered_degree_1_derivative_a (x a b upstream : ℝ) :
 HasDerivAt (fun t : ℝ => upstream*(b + t*x)) (upstream*x) a := by
 convert ((hasDerivAt_const a (upstream)).mul ((hasDerivAt_const a (b)).add ((hasDerivAt_id a).mul (hasDerivAt_const a (x))))) using 1 <;> (try funext t) <;> dsimp <;> ring
theorem factor_discovered_degree_1_derivative_b (x a b upstream : ℝ) :
 HasDerivAt (fun t : ℝ => upstream*(a*x + t)) (upstream) b := by
 convert ((hasDerivAt_const b (upstream)).mul ((hasDerivAt_id b).add (hasDerivAt_const b (a*x)))) using 1 <;> (try funext t) <;> dsimp <;> ring
theorem factor_discovered_degree_2_joint (x a b upstream dx da db : ℝ) : 2*a*dx*upstream*(a*x + b) + 2*da*upstream*x*(a*x + b) + db*upstream*(2*a*x + 2*b) = 2*upstream*(a*x + b)*(a*dx + da*x + db) := by ring
theorem factor_discovered_degree_2_derivative_x (x a b upstream : ℝ) :
 HasDerivAt (fun t : ℝ => upstream*(a*t + b)^2) (2*a*upstream*(a*x + b)) x := by
 convert ((hasDerivAt_const x (upstream)).mul (((hasDerivAt_const x (b)).add ((hasDerivAt_const x (a)).mul (hasDerivAt_id x))).pow 2)) using 1 <;> (try funext t) <;> dsimp <;> ring
theorem factor_discovered_degree_2_derivative_a (x a b upstream : ℝ) :
 HasDerivAt (fun t : ℝ => upstream*(b + t*x)^2) (2*upstream*x*(a*x + b)) a := by
 convert ((hasDerivAt_const a (upstream)).mul (((hasDerivAt_const a (b)).add ((hasDerivAt_id a).mul (hasDerivAt_const a (x)))).pow 2)) using 1 <;> (try funext t) <;> dsimp <;> ring
theorem factor_discovered_degree_2_derivative_b (x a b upstream : ℝ) :
 HasDerivAt (fun t : ℝ => upstream*(a*x + t)^2) (upstream*(2*a*x + 2*b)) b := by
 convert ((hasDerivAt_const b (upstream)).mul (((hasDerivAt_id b).add (hasDerivAt_const b (a*x))).pow 2)) using 1 <;> (try funext t) <;> dsimp <;> ring
theorem factor_discovered_degree_3_joint (x a b upstream dx da db : ℝ) : 3*a*dx*upstream*(a*x + b)^2 + 3*da*upstream*x*(a*x + b)^2 + 3*db*upstream*(a*x + b)^2 = 3*upstream*(a*x + b)^2*(a*dx + da*x + db) := by ring
theorem factor_discovered_degree_3_derivative_x (x a b upstream : ℝ) :
 HasDerivAt (fun t : ℝ => upstream*(a*t + b)^3) (3*a*upstream*(a*x + b)^2) x := by
 convert ((hasDerivAt_const x (upstream)).mul (((hasDerivAt_const x (b)).add ((hasDerivAt_const x (a)).mul (hasDerivAt_id x))).pow 3)) using 1 <;> (try funext t) <;> dsimp <;> ring
theorem factor_discovered_degree_3_derivative_a (x a b upstream : ℝ) :
 HasDerivAt (fun t : ℝ => upstream*(b + t*x)^3) (3*upstream*x*(a*x + b)^2) a := by
 convert ((hasDerivAt_const a (upstream)).mul (((hasDerivAt_const a (b)).add ((hasDerivAt_id a).mul (hasDerivAt_const a (x)))).pow 3)) using 1 <;> (try funext t) <;> dsimp <;> ring
theorem factor_discovered_degree_3_derivative_b (x a b upstream : ℝ) :
 HasDerivAt (fun t : ℝ => upstream*(a*x + t)^3) (3*upstream*(a*x + b)^2) b := by
 convert ((hasDerivAt_const b (upstream)).mul (((hasDerivAt_id b).add (hasDerivAt_const b (a*x))).pow 3)) using 1 <;> (try funext t) <;> dsimp <;> ring
end OjasXMath
