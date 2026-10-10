import OjasXMath.CoordinateCompiler
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Tactic.Ring
set_option autoImplicit false
namespace OjasXMath
theorem coordinate_discovered_primal (x y a b c u v : ℝ) : a*u*x + v*(b*y + c*x^2) = a*u*x + v*(b*y + c*x^2) := by ring
theorem coordinate_discovered_gradient_x (x y a b c u v : ℝ) : a*u + 2*c*v*x = a*u + 2*c*v*x := by ring
theorem coordinate_discovered_gradient_y (x y a b c u v : ℝ) : b*v = b*v := by ring
theorem coordinate_discovered_gradient_a (x y a b c u v : ℝ) : u*x = u*x := by ring
theorem coordinate_discovered_gradient_b (x y a b c u v : ℝ) : v*y = v*y := by ring
theorem coordinate_discovered_gradient_c (x y a b c u v : ℝ) : v*x^2 = v*x^2 := by ring
theorem coordinate_actual_derivative_x (x y a b c u v : ℝ) :
  HasDerivAt (fun t : ℝ => a*t*u + v*(b*y + c*t^2)) (a*u + 2*c*v*x) x := by
  convert (((hasDerivAt_const x (v)).mul ((hasDerivAt_const x (b*y)).add ((hasDerivAt_const x (c)).mul ((hasDerivAt_id x).pow 2)))).add (((hasDerivAt_const x (a)).mul (hasDerivAt_const x (u))).mul (hasDerivAt_id x))) using 1 <;> (try funext t) <;> dsimp <;> ring
theorem coordinate_actual_derivative_y (x y a b c u v : ℝ) :
  HasDerivAt (fun t : ℝ => a*u*x + v*(b*t + c*x^2)) (b*v) y := by
  convert (((hasDerivAt_const y (v)).mul (((hasDerivAt_const y (b)).mul (hasDerivAt_id y)).add (hasDerivAt_const y (c*x^2)))).add (hasDerivAt_const y (a*u*x))) using 1 <;> (try funext t) <;> dsimp <;> ring
theorem coordinate_actual_derivative_a (x y a b c u v : ℝ) :
  HasDerivAt (fun t : ℝ => t*u*x + v*(b*y + c*x^2)) (u*x) a := by
  convert ((hasDerivAt_const a (v*(b*y + c*x^2))).add (((hasDerivAt_id a).mul (hasDerivAt_const a (u))).mul (hasDerivAt_const a (x)))) using 1 <;> (try funext t) <;> dsimp <;> ring
theorem coordinate_actual_derivative_b (x y a b c u v : ℝ) :
  HasDerivAt (fun t : ℝ => a*u*x + v*(c*x^2 + t*y)) (v*y) b := by
  convert (((hasDerivAt_const b (v)).mul (((hasDerivAt_id b).mul (hasDerivAt_const b (y))).add (hasDerivAt_const b (c*x^2)))).add (hasDerivAt_const b (a*u*x))) using 1 <;> (try funext t) <;> dsimp <;> ring
theorem coordinate_actual_derivative_c (x y a b c u v : ℝ) :
  HasDerivAt (fun t : ℝ => a*u*x + v*(b*y + t*x^2)) (v*x^2) c := by
  convert (((hasDerivAt_const c (v)).mul ((hasDerivAt_const c (b*y)).add ((hasDerivAt_id c).mul (hasDerivAt_const c (x^2))))).add (hasDerivAt_const c (a*u*x))) using 1 <;> (try funext t) <;> dsimp <;> ring
theorem coordinate_discovered_joint_dagger (x y a b c u v dx dy da db dc : ℝ) : u*(a*dx + da*x) + v*(b*dy + 2*c*dx*x + db*y + dc*x^2) = b*dy*v + da*u*x + db*v*y + dc*v*x^2 + dx*(a*u + 2*c*v*x) := by ring
end OjasXMath
