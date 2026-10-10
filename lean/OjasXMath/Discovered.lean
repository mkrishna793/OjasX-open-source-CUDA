import Mathlib.Basic.Real.Basic
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Tactic.Ring

set_option autoImplicit false

namespace OjasXMath

theorem discovered_degree_2_primal (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    b0*(a00*x0 + a10*x1)^2 + b1*(a01*x0 + a11*x1)^2 = x0^2*(a00^2*b0 + a01^2*b1) + x0*x1*(2*a00*a10*b0 + 2*a01*a11*b1) + x1^2*(a10^2*b0 + a11^2*b1) := by ring

theorem discovered_degree_2_gradient_x0 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    2*a00*b0*(a00*x0 + a10*x1) + 2*a01*b1*(a01*x0 + a11*x1) = 2*x0*(a00^2*b0 + a01^2*b1) + x1*(2*a00*a10*b0 + 2*a01*a11*b1) := by ring

theorem discovered_degree_2_gradient_x1 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    2*a10*b0*(a00*x0 + a10*x1) + 2*a11*b1*(a01*x0 + a11*x1) = x0*(2*a00*a10*b0 + 2*a01*a11*b1) + 2*x1*(a10^2*b0 + a11^2*b1) := by ring

theorem discovered_degree_2_gradient_a00 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    2*b0*x0*(a00*x0 + a10*x1) = 2*a00*b0*x0^2 + 2*a10*b0*x0*x1 := by ring

theorem discovered_degree_2_gradient_a01 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    2*b1*x0*(a01*x0 + a11*x1) = 2*a01*b1*x0^2 + 2*a11*b1*x0*x1 := by ring

theorem discovered_degree_2_gradient_a10 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    2*b0*x1*(a00*x0 + a10*x1) = 2*a00*b0*x0*x1 + 2*a10*b0*x1^2 := by ring

theorem discovered_degree_2_gradient_a11 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    2*b1*x1*(a01*x0 + a11*x1) = 2*a01*b1*x0*x1 + 2*a11*b1*x1^2 := by ring

theorem discovered_degree_2_gradient_b0 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    (a00*x0 + a10*x1)^2 = a00^2*x0^2 + 2*a00*a10*x0*x1 + a10^2*x1^2 := by ring

theorem discovered_degree_2_gradient_b1 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    (a01*x0 + a11*x1)^2 = a01^2*x0^2 + 2*a01*a11*x0*x1 + a11^2*x1^2 := by ring

theorem discovered_degree_2_native_derivative_x0 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => b0*(a00*t + a10*x1)^2 + b1*(a01*t + a11*x1)^2) (2*a00*b0*(a00*x0 + a10*x1) + 2*a01*b1*(a01*x0 + a11*x1)) x0 := by
  convert (((hasDerivAt_const x0 (b0)).mul ((((hasDerivAt_const x0 (a00)).mul (hasDerivAt_id x0)).add (hasDerivAt_const x0 (a10*x1))).pow 2)).add ((hasDerivAt_const x0 (b1)).mul ((((hasDerivAt_const x0 (a01)).mul (hasDerivAt_id x0)).add (hasDerivAt_const x0 (a11*x1))).pow 2))) using 1 <;> (try funext t) <;> dsimp <;> ring

theorem discovered_degree_2_lifted_derivative_x0 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => t^2*(a00^2*b0 + a01^2*b1) + t*x1*(2*a00*a10*b0 + 2*a01*a11*b1) + x1^2*(a10^2*b0 + a11^2*b1)) (2*a00*b0*(a00*x0 + a10*x1) + 2*a01*b1*(a01*x0 + a11*x1)) x0 := by
  have h := discovered_degree_2_native_derivative_x0 x0 x1 a00 a01 a10 a11 b0 b1
  have same : (fun t : ℝ => b0*(a00*t + a10*x1)^2 + b1*(a01*t + a11*x1)^2) = (fun t : ℝ => t^2*(a00^2*b0 + a01^2*b1) + t*x1*(2*a00*a10*b0 + 2*a01*a11*b1) + x1^2*(a10^2*b0 + a11^2*b1)) := by
    funext t
    ring
  rw [same] at h
  exact h

theorem discovered_degree_2_native_derivative_x1 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => b0*(a00*x0 + a10*t)^2 + b1*(a01*x0 + a11*t)^2) (2*a10*b0*(a00*x0 + a10*x1) + 2*a11*b1*(a01*x0 + a11*x1)) x1 := by
  convert (((hasDerivAt_const x1 (b0)).mul (((hasDerivAt_const x1 (a00*x0)).add ((hasDerivAt_const x1 (a10)).mul (hasDerivAt_id x1))).pow 2)).add ((hasDerivAt_const x1 (b1)).mul (((hasDerivAt_const x1 (a01*x0)).add ((hasDerivAt_const x1 (a11)).mul (hasDerivAt_id x1))).pow 2))) using 1 <;> (try funext t) <;> dsimp <;> ring

theorem discovered_degree_2_lifted_derivative_x1 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => t^2*(a10^2*b0 + a11^2*b1) + t*x0*(2*a00*a10*b0 + 2*a01*a11*b1) + x0^2*(a00^2*b0 + a01^2*b1)) (2*a10*b0*(a00*x0 + a10*x1) + 2*a11*b1*(a01*x0 + a11*x1)) x1 := by
  have h := discovered_degree_2_native_derivative_x1 x0 x1 a00 a01 a10 a11 b0 b1
  have same : (fun t : ℝ => b0*(a00*x0 + a10*t)^2 + b1*(a01*x0 + a11*t)^2) = (fun t : ℝ => t^2*(a10^2*b0 + a11^2*b1) + t*x0*(2*a00*a10*b0 + 2*a01*a11*b1) + x0^2*(a00^2*b0 + a01^2*b1)) := by
    funext t
    ring
  rw [same] at h
  exact h

theorem discovered_degree_2_native_derivative_a00 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => b0*(a10*x1 + t*x0)^2 + b1*(a01*x0 + a11*x1)^2) (2*b0*x0*(a00*x0 + a10*x1)) a00 := by
  convert (((hasDerivAt_const a00 (b0)).mul ((((hasDerivAt_id a00).mul (hasDerivAt_const a00 (x0))).add (hasDerivAt_const a00 (a10*x1))).pow 2)).add (hasDerivAt_const a00 (b1*(a01*x0 + a11*x1)^2))) using 1 <;> (try funext t) <;> dsimp <;> ring

theorem discovered_degree_2_lifted_derivative_a00 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => x0^2*(a01^2*b1 + b0*t^2) + x0*x1*(2*a01*a11*b1 + 2*a10*b0*t) + x1^2*(a10^2*b0 + a11^2*b1)) (2*b0*x0*(a00*x0 + a10*x1)) a00 := by
  have h := discovered_degree_2_native_derivative_a00 x0 x1 a00 a01 a10 a11 b0 b1
  have same : (fun t : ℝ => b0*(a10*x1 + t*x0)^2 + b1*(a01*x0 + a11*x1)^2) = (fun t : ℝ => x0^2*(a01^2*b1 + b0*t^2) + x0*x1*(2*a01*a11*b1 + 2*a10*b0*t) + x1^2*(a10^2*b0 + a11^2*b1)) := by
    funext t
    ring
  rw [same] at h
  exact h

theorem discovered_degree_2_native_derivative_a01 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => b0*(a00*x0 + a10*x1)^2 + b1*(a11*x1 + t*x0)^2) (2*b1*x0*(a01*x0 + a11*x1)) a01 := by
  convert ((hasDerivAt_const a01 (b0*(a00*x0 + a10*x1)^2)).add ((hasDerivAt_const a01 (b1)).mul ((((hasDerivAt_id a01).mul (hasDerivAt_const a01 (x0))).add (hasDerivAt_const a01 (a11*x1))).pow 2))) using 1 <;> (try funext t) <;> dsimp <;> ring

theorem discovered_degree_2_lifted_derivative_a01 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => x0^2*(a00^2*b0 + b1*t^2) + x0*x1*(2*a00*a10*b0 + 2*a11*b1*t) + x1^2*(a10^2*b0 + a11^2*b1)) (2*b1*x0*(a01*x0 + a11*x1)) a01 := by
  have h := discovered_degree_2_native_derivative_a01 x0 x1 a00 a01 a10 a11 b0 b1
  have same : (fun t : ℝ => b0*(a00*x0 + a10*x1)^2 + b1*(a11*x1 + t*x0)^2) = (fun t : ℝ => x0^2*(a00^2*b0 + b1*t^2) + x0*x1*(2*a00*a10*b0 + 2*a11*b1*t) + x1^2*(a10^2*b0 + a11^2*b1)) := by
    funext t
    ring
  rw [same] at h
  exact h

theorem discovered_degree_2_native_derivative_a10 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => b0*(a00*x0 + t*x1)^2 + b1*(a01*x0 + a11*x1)^2) (2*b0*x1*(a00*x0 + a10*x1)) a10 := by
  convert (((hasDerivAt_const a10 (b0)).mul (((hasDerivAt_const a10 (a00*x0)).add ((hasDerivAt_id a10).mul (hasDerivAt_const a10 (x1)))).pow 2)).add (hasDerivAt_const a10 (b1*(a01*x0 + a11*x1)^2))) using 1 <;> (try funext t) <;> dsimp <;> ring

theorem discovered_degree_2_lifted_derivative_a10 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => x0^2*(a00^2*b0 + a01^2*b1) + x0*x1*(2*a00*b0*t + 2*a01*a11*b1) + x1^2*(a11^2*b1 + b0*t^2)) (2*b0*x1*(a00*x0 + a10*x1)) a10 := by
  have h := discovered_degree_2_native_derivative_a10 x0 x1 a00 a01 a10 a11 b0 b1
  have same : (fun t : ℝ => b0*(a00*x0 + t*x1)^2 + b1*(a01*x0 + a11*x1)^2) = (fun t : ℝ => x0^2*(a00^2*b0 + a01^2*b1) + x0*x1*(2*a00*b0*t + 2*a01*a11*b1) + x1^2*(a11^2*b1 + b0*t^2)) := by
    funext t
    ring
  rw [same] at h
  exact h

theorem discovered_degree_2_native_derivative_a11 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => b0*(a00*x0 + a10*x1)^2 + b1*(a01*x0 + t*x1)^2) (2*b1*x1*(a01*x0 + a11*x1)) a11 := by
  convert ((hasDerivAt_const a11 (b0*(a00*x0 + a10*x1)^2)).add ((hasDerivAt_const a11 (b1)).mul (((hasDerivAt_const a11 (a01*x0)).add ((hasDerivAt_id a11).mul (hasDerivAt_const a11 (x1)))).pow 2))) using 1 <;> (try funext t) <;> dsimp <;> ring

theorem discovered_degree_2_lifted_derivative_a11 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => x0^2*(a00^2*b0 + a01^2*b1) + x0*x1*(2*a00*a10*b0 + 2*a01*b1*t) + x1^2*(a10^2*b0 + b1*t^2)) (2*b1*x1*(a01*x0 + a11*x1)) a11 := by
  have h := discovered_degree_2_native_derivative_a11 x0 x1 a00 a01 a10 a11 b0 b1
  have same : (fun t : ℝ => b0*(a00*x0 + a10*x1)^2 + b1*(a01*x0 + t*x1)^2) = (fun t : ℝ => x0^2*(a00^2*b0 + a01^2*b1) + x0*x1*(2*a00*a10*b0 + 2*a01*b1*t) + x1^2*(a10^2*b0 + b1*t^2)) := by
    funext t
    ring
  rw [same] at h
  exact h

theorem discovered_degree_2_native_derivative_b0 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => b1*(a01*x0 + a11*x1)^2 + t*(a00*x0 + a10*x1)^2) ((a00*x0 + a10*x1)^2) b0 := by
  convert (((hasDerivAt_id b0).mul (hasDerivAt_const b0 ((a00*x0 + a10*x1)^2))).add (hasDerivAt_const b0 (b1*(a01*x0 + a11*x1)^2))) using 1 <;> (try funext t) <;> dsimp <;> ring

theorem discovered_degree_2_lifted_derivative_b0 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => x0^2*(a00^2*t + a01^2*b1) + x0*x1*(2*a00*a10*t + 2*a01*a11*b1) + x1^2*(a10^2*t + a11^2*b1)) ((a00*x0 + a10*x1)^2) b0 := by
  have h := discovered_degree_2_native_derivative_b0 x0 x1 a00 a01 a10 a11 b0 b1
  have same : (fun t : ℝ => b1*(a01*x0 + a11*x1)^2 + t*(a00*x0 + a10*x1)^2) = (fun t : ℝ => x0^2*(a00^2*t + a01^2*b1) + x0*x1*(2*a00*a10*t + 2*a01*a11*b1) + x1^2*(a10^2*t + a11^2*b1)) := by
    funext t
    ring
  rw [same] at h
  exact h

theorem discovered_degree_2_native_derivative_b1 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => b0*(a00*x0 + a10*x1)^2 + t*(a01*x0 + a11*x1)^2) ((a01*x0 + a11*x1)^2) b1 := by
  convert ((hasDerivAt_const b1 (b0*(a00*x0 + a10*x1)^2)).add ((hasDerivAt_id b1).mul (hasDerivAt_const b1 ((a01*x0 + a11*x1)^2)))) using 1 <;> (try funext t) <;> dsimp <;> ring

theorem discovered_degree_2_lifted_derivative_b1 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => x0^2*(a00^2*b0 + a01^2*t) + x0*x1*(2*a00*a10*b0 + 2*a01*a11*t) + x1^2*(a10^2*b0 + a11^2*t)) ((a01*x0 + a11*x1)^2) b1 := by
  have h := discovered_degree_2_native_derivative_b1 x0 x1 a00 a01 a10 a11 b0 b1
  have same : (fun t : ℝ => b0*(a00*x0 + a10*x1)^2 + t*(a01*x0 + a11*x1)^2) = (fun t : ℝ => x0^2*(a00^2*b0 + a01^2*t) + x0*x1*(2*a00*a10*b0 + 2*a01*a11*t) + x1^2*(a10^2*b0 + a11^2*t)) := by
    funext t
    ring
  rw [same] at h
  exact h

theorem discovered_degree_3_primal (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    b0*(a00*x0 + a10*x1)^3 + b1*(a01*x0 + a11*x1)^3 = x0^3*(a00^3*b0 + a01^3*b1) + x0^2*x1*(3*a00^2*a10*b0 + 3*a01^2*a11*b1) + x0*x1^2*(3*a00*a10^2*b0 + 3*a01*a11^2*b1) + x1^3*(a10^3*b0 + a11^3*b1) := by ring

theorem discovered_degree_3_gradient_x0 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    3*a00*b0*(a00*x0 + a10*x1)^2 + 3*a01*b1*(a01*x0 + a11*x1)^2 = 3*x0^2*(a00^3*b0 + a01^3*b1) + 2*x0*x1*(3*a00^2*a10*b0 + 3*a01^2*a11*b1) + x1^2*(3*a00*a10^2*b0 + 3*a01*a11^2*b1) := by ring

theorem discovered_degree_3_gradient_x1 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    3*a10*b0*(a00*x0 + a10*x1)^2 + 3*a11*b1*(a01*x0 + a11*x1)^2 = x0^2*(3*a00^2*a10*b0 + 3*a01^2*a11*b1) + 2*x0*x1*(3*a00*a10^2*b0 + 3*a01*a11^2*b1) + 3*x1^2*(a10^3*b0 + a11^3*b1) := by ring

theorem discovered_degree_3_gradient_a00 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    3*b0*x0*(a00*x0 + a10*x1)^2 = 3*a00^2*b0*x0^3 + 6*a00*a10*b0*x0^2*x1 + 3*a10^2*b0*x0*x1^2 := by ring

theorem discovered_degree_3_gradient_a01 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    3*b1*x0*(a01*x0 + a11*x1)^2 = 3*a01^2*b1*x0^3 + 6*a01*a11*b1*x0^2*x1 + 3*a11^2*b1*x0*x1^2 := by ring

theorem discovered_degree_3_gradient_a10 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    3*b0*x1*(a00*x0 + a10*x1)^2 = 3*a00^2*b0*x0^2*x1 + 6*a00*a10*b0*x0*x1^2 + 3*a10^2*b0*x1^3 := by ring

theorem discovered_degree_3_gradient_a11 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    3*b1*x1*(a01*x0 + a11*x1)^2 = 3*a01^2*b1*x0^2*x1 + 6*a01*a11*b1*x0*x1^2 + 3*a11^2*b1*x1^3 := by ring

theorem discovered_degree_3_gradient_b0 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    (a00*x0 + a10*x1)^3 = a00^3*x0^3 + 3*a00^2*a10*x0^2*x1 + 3*a00*a10^2*x0*x1^2 + a10^3*x1^3 := by ring

theorem discovered_degree_3_gradient_b1 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    (a01*x0 + a11*x1)^3 = a01^3*x0^3 + 3*a01^2*a11*x0^2*x1 + 3*a01*a11^2*x0*x1^2 + a11^3*x1^3 := by ring

theorem discovered_degree_3_native_derivative_x0 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => b0*(a00*t + a10*x1)^3 + b1*(a01*t + a11*x1)^3) (3*a00*b0*(a00*x0 + a10*x1)^2 + 3*a01*b1*(a01*x0 + a11*x1)^2) x0 := by
  convert (((hasDerivAt_const x0 (b0)).mul ((((hasDerivAt_const x0 (a00)).mul (hasDerivAt_id x0)).add (hasDerivAt_const x0 (a10*x1))).pow 3)).add ((hasDerivAt_const x0 (b1)).mul ((((hasDerivAt_const x0 (a01)).mul (hasDerivAt_id x0)).add (hasDerivAt_const x0 (a11*x1))).pow 3))) using 1 <;> (try funext t) <;> dsimp <;> ring

theorem discovered_degree_3_lifted_derivative_x0 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => t^3*(a00^3*b0 + a01^3*b1) + t^2*x1*(3*a00^2*a10*b0 + 3*a01^2*a11*b1) + t*x1^2*(3*a00*a10^2*b0 + 3*a01*a11^2*b1) + x1^3*(a10^3*b0 + a11^3*b1)) (3*a00*b0*(a00*x0 + a10*x1)^2 + 3*a01*b1*(a01*x0 + a11*x1)^2) x0 := by
  have h := discovered_degree_3_native_derivative_x0 x0 x1 a00 a01 a10 a11 b0 b1
  have same : (fun t : ℝ => b0*(a00*t + a10*x1)^3 + b1*(a01*t + a11*x1)^3) = (fun t : ℝ => t^3*(a00^3*b0 + a01^3*b1) + t^2*x1*(3*a00^2*a10*b0 + 3*a01^2*a11*b1) + t*x1^2*(3*a00*a10^2*b0 + 3*a01*a11^2*b1) + x1^3*(a10^3*b0 + a11^3*b1)) := by
    funext t
    ring
  rw [same] at h
  exact h

theorem discovered_degree_3_native_derivative_x1 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => b0*(a00*x0 + a10*t)^3 + b1*(a01*x0 + a11*t)^3) (3*a10*b0*(a00*x0 + a10*x1)^2 + 3*a11*b1*(a01*x0 + a11*x1)^2) x1 := by
  convert (((hasDerivAt_const x1 (b0)).mul (((hasDerivAt_const x1 (a00*x0)).add ((hasDerivAt_const x1 (a10)).mul (hasDerivAt_id x1))).pow 3)).add ((hasDerivAt_const x1 (b1)).mul (((hasDerivAt_const x1 (a01*x0)).add ((hasDerivAt_const x1 (a11)).mul (hasDerivAt_id x1))).pow 3))) using 1 <;> (try funext t) <;> dsimp <;> ring

theorem discovered_degree_3_lifted_derivative_x1 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => t^3*(a10^3*b0 + a11^3*b1) + t^2*x0*(3*a00*a10^2*b0 + 3*a01*a11^2*b1) + t*x0^2*(3*a00^2*a10*b0 + 3*a01^2*a11*b1) + x0^3*(a00^3*b0 + a01^3*b1)) (3*a10*b0*(a00*x0 + a10*x1)^2 + 3*a11*b1*(a01*x0 + a11*x1)^2) x1 := by
  have h := discovered_degree_3_native_derivative_x1 x0 x1 a00 a01 a10 a11 b0 b1
  have same : (fun t : ℝ => b0*(a00*x0 + a10*t)^3 + b1*(a01*x0 + a11*t)^3) = (fun t : ℝ => t^3*(a10^3*b0 + a11^3*b1) + t^2*x0*(3*a00*a10^2*b0 + 3*a01*a11^2*b1) + t*x0^2*(3*a00^2*a10*b0 + 3*a01^2*a11*b1) + x0^3*(a00^3*b0 + a01^3*b1)) := by
    funext t
    ring
  rw [same] at h
  exact h

theorem discovered_degree_3_native_derivative_a00 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => b0*(a10*x1 + t*x0)^3 + b1*(a01*x0 + a11*x1)^3) (3*b0*x0*(a00*x0 + a10*x1)^2) a00 := by
  convert (((hasDerivAt_const a00 (b0)).mul ((((hasDerivAt_id a00).mul (hasDerivAt_const a00 (x0))).add (hasDerivAt_const a00 (a10*x1))).pow 3)).add (hasDerivAt_const a00 (b1*(a01*x0 + a11*x1)^3))) using 1 <;> (try funext t) <;> dsimp <;> ring

theorem discovered_degree_3_lifted_derivative_a00 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => x0^3*(a01^3*b1 + b0*t^3) + x0^2*x1*(3*a01^2*a11*b1 + 3*a10*b0*t^2) + x0*x1^2*(3*a01*a11^2*b1 + 3*a10^2*b0*t) + x1^3*(a10^3*b0 + a11^3*b1)) (3*b0*x0*(a00*x0 + a10*x1)^2) a00 := by
  have h := discovered_degree_3_native_derivative_a00 x0 x1 a00 a01 a10 a11 b0 b1
  have same : (fun t : ℝ => b0*(a10*x1 + t*x0)^3 + b1*(a01*x0 + a11*x1)^3) = (fun t : ℝ => x0^3*(a01^3*b1 + b0*t^3) + x0^2*x1*(3*a01^2*a11*b1 + 3*a10*b0*t^2) + x0*x1^2*(3*a01*a11^2*b1 + 3*a10^2*b0*t) + x1^3*(a10^3*b0 + a11^3*b1)) := by
    funext t
    ring
  rw [same] at h
  exact h

theorem discovered_degree_3_native_derivative_a01 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => b0*(a00*x0 + a10*x1)^3 + b1*(a11*x1 + t*x0)^3) (3*b1*x0*(a01*x0 + a11*x1)^2) a01 := by
  convert ((hasDerivAt_const a01 (b0*(a00*x0 + a10*x1)^3)).add ((hasDerivAt_const a01 (b1)).mul ((((hasDerivAt_id a01).mul (hasDerivAt_const a01 (x0))).add (hasDerivAt_const a01 (a11*x1))).pow 3))) using 1 <;> (try funext t) <;> dsimp <;> ring

theorem discovered_degree_3_lifted_derivative_a01 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => x0^3*(a00^3*b0 + b1*t^3) + x0^2*x1*(3*a00^2*a10*b0 + 3*a11*b1*t^2) + x0*x1^2*(3*a00*a10^2*b0 + 3*a11^2*b1*t) + x1^3*(a10^3*b0 + a11^3*b1)) (3*b1*x0*(a01*x0 + a11*x1)^2) a01 := by
  have h := discovered_degree_3_native_derivative_a01 x0 x1 a00 a01 a10 a11 b0 b1
  have same : (fun t : ℝ => b0*(a00*x0 + a10*x1)^3 + b1*(a11*x1 + t*x0)^3) = (fun t : ℝ => x0^3*(a00^3*b0 + b1*t^3) + x0^2*x1*(3*a00^2*a10*b0 + 3*a11*b1*t^2) + x0*x1^2*(3*a00*a10^2*b0 + 3*a11^2*b1*t) + x1^3*(a10^3*b0 + a11^3*b1)) := by
    funext t
    ring
  rw [same] at h
  exact h

theorem discovered_degree_3_native_derivative_a10 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => b0*(a00*x0 + t*x1)^3 + b1*(a01*x0 + a11*x1)^3) (3*b0*x1*(a00*x0 + a10*x1)^2) a10 := by
  convert (((hasDerivAt_const a10 (b0)).mul (((hasDerivAt_const a10 (a00*x0)).add ((hasDerivAt_id a10).mul (hasDerivAt_const a10 (x1)))).pow 3)).add (hasDerivAt_const a10 (b1*(a01*x0 + a11*x1)^3))) using 1 <;> (try funext t) <;> dsimp <;> ring

theorem discovered_degree_3_lifted_derivative_a10 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => x0^3*(a00^3*b0 + a01^3*b1) + x0^2*x1*(3*a00^2*b0*t + 3*a01^2*a11*b1) + x0*x1^2*(3*a00*b0*t^2 + 3*a01*a11^2*b1) + x1^3*(a11^3*b1 + b0*t^3)) (3*b0*x1*(a00*x0 + a10*x1)^2) a10 := by
  have h := discovered_degree_3_native_derivative_a10 x0 x1 a00 a01 a10 a11 b0 b1
  have same : (fun t : ℝ => b0*(a00*x0 + t*x1)^3 + b1*(a01*x0 + a11*x1)^3) = (fun t : ℝ => x0^3*(a00^3*b0 + a01^3*b1) + x0^2*x1*(3*a00^2*b0*t + 3*a01^2*a11*b1) + x0*x1^2*(3*a00*b0*t^2 + 3*a01*a11^2*b1) + x1^3*(a11^3*b1 + b0*t^3)) := by
    funext t
    ring
  rw [same] at h
  exact h

theorem discovered_degree_3_native_derivative_a11 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => b0*(a00*x0 + a10*x1)^3 + b1*(a01*x0 + t*x1)^3) (3*b1*x1*(a01*x0 + a11*x1)^2) a11 := by
  convert ((hasDerivAt_const a11 (b0*(a00*x0 + a10*x1)^3)).add ((hasDerivAt_const a11 (b1)).mul (((hasDerivAt_const a11 (a01*x0)).add ((hasDerivAt_id a11).mul (hasDerivAt_const a11 (x1)))).pow 3))) using 1 <;> (try funext t) <;> dsimp <;> ring

theorem discovered_degree_3_lifted_derivative_a11 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => x0^3*(a00^3*b0 + a01^3*b1) + x0^2*x1*(3*a00^2*a10*b0 + 3*a01^2*b1*t) + x0*x1^2*(3*a00*a10^2*b0 + 3*a01*b1*t^2) + x1^3*(a10^3*b0 + b1*t^3)) (3*b1*x1*(a01*x0 + a11*x1)^2) a11 := by
  have h := discovered_degree_3_native_derivative_a11 x0 x1 a00 a01 a10 a11 b0 b1
  have same : (fun t : ℝ => b0*(a00*x0 + a10*x1)^3 + b1*(a01*x0 + t*x1)^3) = (fun t : ℝ => x0^3*(a00^3*b0 + a01^3*b1) + x0^2*x1*(3*a00^2*a10*b0 + 3*a01^2*b1*t) + x0*x1^2*(3*a00*a10^2*b0 + 3*a01*b1*t^2) + x1^3*(a10^3*b0 + b1*t^3)) := by
    funext t
    ring
  rw [same] at h
  exact h

theorem discovered_degree_3_native_derivative_b0 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => b1*(a01*x0 + a11*x1)^3 + t*(a00*x0 + a10*x1)^3) ((a00*x0 + a10*x1)^3) b0 := by
  convert (((hasDerivAt_id b0).mul (hasDerivAt_const b0 ((a00*x0 + a10*x1)^3))).add (hasDerivAt_const b0 (b1*(a01*x0 + a11*x1)^3))) using 1 <;> (try funext t) <;> dsimp <;> ring

theorem discovered_degree_3_lifted_derivative_b0 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => x0^3*(a00^3*t + a01^3*b1) + x0^2*x1*(3*a00^2*a10*t + 3*a01^2*a11*b1) + x0*x1^2*(3*a00*a10^2*t + 3*a01*a11^2*b1) + x1^3*(a10^3*t + a11^3*b1)) ((a00*x0 + a10*x1)^3) b0 := by
  have h := discovered_degree_3_native_derivative_b0 x0 x1 a00 a01 a10 a11 b0 b1
  have same : (fun t : ℝ => b1*(a01*x0 + a11*x1)^3 + t*(a00*x0 + a10*x1)^3) = (fun t : ℝ => x0^3*(a00^3*t + a01^3*b1) + x0^2*x1*(3*a00^2*a10*t + 3*a01^2*a11*b1) + x0*x1^2*(3*a00*a10^2*t + 3*a01*a11^2*b1) + x1^3*(a10^3*t + a11^3*b1)) := by
    funext t
    ring
  rw [same] at h
  exact h

theorem discovered_degree_3_native_derivative_b1 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => b0*(a00*x0 + a10*x1)^3 + t*(a01*x0 + a11*x1)^3) ((a01*x0 + a11*x1)^3) b1 := by
  convert ((hasDerivAt_const b1 (b0*(a00*x0 + a10*x1)^3)).add ((hasDerivAt_id b1).mul (hasDerivAt_const b1 ((a01*x0 + a11*x1)^3)))) using 1 <;> (try funext t) <;> dsimp <;> ring

theorem discovered_degree_3_lifted_derivative_b1 (x0 x1 a00 a01 a10 a11 b0 b1 : ℝ) :
    HasDerivAt (fun t : ℝ => x0^3*(a00^3*b0 + a01^3*t) + x0^2*x1*(3*a00^2*a10*b0 + 3*a01^2*a11*t) + x0*x1^2*(3*a00*a10^2*b0 + 3*a01*a11^2*t) + x1^3*(a10^3*b0 + a11^3*t)) ((a01*x0 + a11*x1)^3) b1 := by
  have h := discovered_degree_3_native_derivative_b1 x0 x1 a00 a01 a10 a11 b0 b1
  have same : (fun t : ℝ => b0*(a00*x0 + a10*x1)^3 + t*(a01*x0 + a11*x1)^3) = (fun t : ℝ => x0^3*(a00^3*b0 + a01^3*t) + x0^2*x1*(3*a00^2*a10*b0 + 3*a01^2*a11*t) + x0*x1^2*(3*a00*a10^2*b0 + 3*a01*a11^2*t) + x1^3*(a10^3*b0 + a11^3*t)) := by
    funext t
    ring
  rw [same] at h
  exact h

end OjasXMath
