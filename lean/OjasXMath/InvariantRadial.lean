import OjasXMath.RadialCore
import OjasXMath.GradientFactors
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Pow

namespace OjasXMath
noncomputable section

/- True calculus at every input, including zero, for squared-norm gates.
   A supplied scalar derivative is required; no global missing laws. -/
theorem invariant_energy_derivative {N : Type*} [Fintype N]
    (z dz : N → ℝ) (c : ℝ) :
    HasDerivAt (fun t : ℝ => c*∑ j, (z j+t*dz j)^2)
      (2*c*∑ j, z j*dz j) 0 := by
  have each (j : N) : HasDerivAt (fun t : ℝ => (z j+t*dz j)^2) (2*z j*dz j) 0 := by
    convert ((hasDerivAt_const (0:ℝ) (z j)).add
      ((hasDerivAt_id (0:ℝ)).mul_const (dz j))).pow 2 using 1 <;> (try funext t) <;> dsimp <;> ring
  have hs := (HasDerivAt.fun_sum (u := Finset.univ) (fun j _ => each j)).const_mul c
  convert hs using 1 <;> (try funext t) <;> (try dsimp) <;> simp only [Finset.sum_mul, Finset.mul_sum] <;> ring

theorem invariant_radial_actual_derivative {N : Type*} [Fintype N]
    (z dz : N → ℝ) (c : ℝ) (f : ℝ → ℝ) (slope : ℝ) (j : N)
    (hf : HasDerivAt f slope (c*∑ l, (z l)^2)) :
    HasDerivAt (fun t : ℝ => f (c*∑ l, (z l+t*dz l)^2)*(z j+t*dz j))
      (f (c*∑ l, (z l)^2)*dz j+2*c*slope*z j*(∑ l, z l*dz l)) 0 := by
  have he := invariant_energy_derivative z dz c
  have hf0 : HasDerivAt f slope ((fun t : ℝ => c*∑ l, (z l+t*dz l)^2) 0) := by
    simpa using hf
  have hc := (hf0.comp 0 he).mul
    ((hasDerivAt_const (0:ℝ) (z j)).add ((hasDerivAt_id (0:ℝ)).mul_const (dz j)))
  convert hc using 1 <;> (try funext t) <;> dsimp <;> simp only [zero_mul, add_zero] <;> ring

theorem invariant_local_dagger {N : Type*} [Fintype N]
    (z dz g : N → ℝ) (a beta : ℝ) :
    (∑ i, (a*dz i+beta*z i*(∑ j, z j*dz j))*g i) =
      ∑ i, dz i*(a*g i+beta*z i*(∑ j, z j*g j)) := by
  have h := radial_rank_one_pullback z dz g a (-beta)
  simpa only [neg_mul, sub_neg_eq_add] using h

theorem invariant_gram_matrix {M K N : Type*} [Fintype K] [Fintype N]
    (a : Matrix M K ℝ) (v : Matrix K N ℝ) :
    (a*v)*(a*v).transpose = (a*(v*v.transpose))*a.transpose := by
  rw [Matrix.transpose_mul]
  simp only [Matrix.mul_assoc]

theorem invariant_projection {M K N P : Type*} [Fintype K] [Fintype N]
    (a : Matrix M K ℝ) (v : Matrix K N ℝ) (q : Matrix N P ℝ) :
    (a*v)*q = a*(v*q) := Matrix.mul_assoc a v q

theorem invariant_radial_forward {M K N P : Type*} [Fintype K] [Fintype N]
    (a : Matrix M K ℝ) (v : Matrix K N ℝ) (q : Matrix N P ℝ)
    (f : ℝ → ℝ) (c : ℝ) (i : M) (j : P) :
    f (c*((a*v)*(a*v).transpose) i i)*((a*v)*q) i j =
    f (c*((a*(v*v.transpose))*a.transpose) i i)*(a*(v*q)) i j := by
  rw [invariant_gram_matrix, Matrix.mul_assoc a v q]

theorem invariant_input_correction {M K N : Type*} [Fintype K] [Fintype N]
    (a : Matrix M K ℝ) (v : Matrix K N ℝ) :
    (a*v)*v.transpose = a*(v*v.transpose) := Matrix.mul_assoc a v v.transpose

theorem invariant_weight_correction {M K N : Type*} [Fintype M] [Fintype K]
    (a : Matrix M K ℝ) (v : Matrix K N ℝ) (e : Matrix M M ℝ) [Fintype N] :
    a.transpose*(e*(a*v)) = (a.transpose*(e*a))*v := by
  simp only [Matrix.mul_assoc]

theorem invariant_q_gradient {M K N P : Type*} [Fintype M] [Fintype K]
    (a : Matrix M K ℝ) (v : Matrix K N ℝ) (r : Matrix M P ℝ) :
    (a*v).transpose*r = v.transpose*(a.transpose*r) := by
  rw [Matrix.transpose_mul, Matrix.mul_assoc]

theorem invariant_orthogonal_gram {K N : Type*} [Fintype K] [Fintype N] [DecidableEq N]
    (v : Matrix K N ℝ) (o : Matrix N N ℝ) (ho : o*o.transpose=1) :
    (v*o)*(v*o).transpose = v*v.transpose := by
  rw [Matrix.transpose_mul, ← Matrix.mul_assoc, Matrix.mul_assoc v o o.transpose, ho]
  simp

theorem invariant_orthogonal_projection {K N P : Type*} [Fintype N] [DecidableEq N]
    (v : Matrix K N ℝ) (o : Matrix N N ℝ) (q : Matrix N P ℝ)
    (ho : o*o.transpose=1) :
    (v*o)*(o.transpose*q) = v*q := by
  rw [← Matrix.mul_assoc, Matrix.mul_assoc v o o.transpose, ho]
  simp

theorem invariant_reciprocal_actual (s : ℝ) (hs : 0 ≤ s) :
    HasDerivAt (fun t : ℝ => (1+t)⁻¹) (-((1+s)⁻¹)^2) s := by
  have hne : (1:ℝ)+s ≠ 0 := ne_of_gt (by linarith)
  convert ((hasDerivAt_const s (1:ℝ)).add (hasDerivAt_id s)).inv hne using 1 <;>
    (try funext t) <;> dsimp <;> field_simp [hne] <;> ring

theorem invariant_sigmoid_actual (a b s : ℝ) :
    HasDerivAt (fun t : ℝ => cacheSigmoid (a*t+b))
      (a*cacheSigmoid (a*s+b)*(1-cacheSigmoid (a*s+b))) s := by
  convert (cache_sigmoid_hasDerivAt (a*s+b)).comp s
    (((hasDerivAt_const s a).mul (hasDerivAt_id s)).add (hasDerivAt_const s b)) using 1 <;>
    (try funext t) <;> (try dsimp) <;> ring

theorem invariant_sigmoid_parameter_a (a b s : ℝ) :
    HasDerivAt (fun t : ℝ => cacheSigmoid (t*s+b))
      (s*cacheSigmoid (a*s+b)*(1-cacheSigmoid (a*s+b))) a := by
  convert (cache_sigmoid_hasDerivAt (a*s+b)).comp a
    (((hasDerivAt_id a).mul_const s).add (hasDerivAt_const a b)) using 1 <;>
    (try funext t) <;> (try dsimp) <;> ring

theorem invariant_sigmoid_parameter_b (a b s : ℝ) :
    HasDerivAt (fun t : ℝ => cacheSigmoid (a*s+t))
      (cacheSigmoid (a*s+b)*(1-cacheSigmoid (a*s+b))) b := by
  convert (cache_sigmoid_hasDerivAt (a*s+b)).comp b
    ((hasDerivAt_const b (a*s)).add (hasDerivAt_id b)) using 1 <;>
    (try funext t) <;> (try dsimp) <;> ring

theorem invariant_cost_threshold (m k n p : ℝ)
    (hkp : 0 < k+p) (hshape : k*(m+n) < m*n) :
    6*k*(k+p)*(m+n) < 6*m*n*(k+p) := by
  have h := mul_lt_mul_of_pos_right hshape hkp
  nlinarith

theorem invariant_energy_nonnegative {N : Type*} [Fintype N]
    (z : N → ℝ) (c : ℝ) (hc : 0 ≤ c) : 0 ≤ c*∑ j, (z j)^2 := by positivity

theorem invariant_aligned_cubic {N : Type*} [Fintype N]
    (z : N → ℝ) (h : ∑ j, (z j)^2=1) (i : N) :
    z i+2*z i*(∑ j, z j*z j)=3*z i := by
  simp_rw [← pow_two]
  rw [h]
  ring

theorem invariant_norm_scale_cancellation (a variance eps r : ℝ)
    (h : r*r*(a*a*variance+eps)=1) :
    r-a*a*variance*r*r*r = eps*r*r*r := by
  calc
    _ = r*(r*r*(a*a*variance+eps))-a*a*variance*r*r*r := by rw [h]; ring
    _ = _ := by ring

theorem stable_scaled_norm_actual_derivative (a variance eps : ℝ)
    (hv : 0 ≤ variance) (he : 0 < eps) :
    HasDerivAt (fun t : ℝ => t / Real.sqrt (t*t*variance+eps))
      (eps / (Real.sqrt (a*a*variance+eps))^3) a := by
  have hp : 0 < a*a*variance+eps :=
    add_pos_of_nonneg_of_pos (mul_nonneg (mul_self_nonneg a) hv) he
  have hq : HasDerivAt (fun t : ℝ => t*t*variance+eps) (2*a*variance) a := by
    convert (((hasDerivAt_id a).mul (hasDerivAt_id a)).mul_const variance).add_const eps using 1 <;>
      (try funext t) <;> dsimp <;> ring
  have h := radial_normalization_derivative hq (hasDerivAt_id a) hp
  have hn : Real.sqrt (a*a*variance+eps) ≠ 0 := ne_of_gt (Real.sqrt_pos.2 hp)
  have hs := Real.sq_sqrt (le_of_lt hp)
  convert h using 1
  dsimp
  field_simp [hn]
  simp only [id_eq, pow_two] at *
  nlinarith [hs]

theorem stable_scaled_norm_overflow_avoiding_form (d eps : ℝ) (hd : 0 < d) :
    eps / (Real.sqrt d)^3 = (eps/d) / Real.sqrt d := by
  have hn : Real.sqrt d ≠ 0 := ne_of_gt (Real.sqrt_pos.2 hd)
  calc
    _ = (eps/(Real.sqrt d)^2) / Real.sqrt d := by field_simp
    _ = _ := by rw [Real.sq_sqrt (le_of_lt hd)]

end
end OjasXMath
