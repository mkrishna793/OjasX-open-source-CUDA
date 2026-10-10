import OjasXMath.Algebra
import OjasXMath.DemandAlgebra
namespace OjasXMath
open scoped BigOperators

/- A residual is sufficient when it reconstructs the primitive's required
values and pullback. Constructing it and decoding it must be charged. -/
structure DenseResidual (E R F G : Type*) where
  encode : E → R
  value : E → F
  pull : E → G → E
  decodeValue : R → F
  decodePull : R → G → E
  value_exact : ∀ x, decodeValue (encode x) = value x
  pull_exact : ∀ x g, decodePull (encode x) g = pull x g

theorem dense_residual_preserves_value {E R F G : Type*}
    (c : DenseResidual E R F G) (x : E) :
    c.decodeValue (c.encode x) = c.value x := c.value_exact x

theorem dense_residual_preserves_pull {E R F G : Type*}
    (c : DenseResidual E R F G) (x : E) (g : G) :
    c.decodePull (c.encode x) g = c.pull x g := c.pull_exact x g

def preactivationResidual (f df : ℝ → ℝ) : DenseResidual ℝ ℝ ℝ ℝ where
  encode := id
  value := f
  pull := fun z g => df z*g
  decodeValue := f
  decodePull := fun z g => df z*g
  value_exact := by intro z; rfl
  pull_exact := by intro z g; rfl

theorem dense_preactivation_no_slope_condition (f df : ℝ → ℝ) (z g : ℝ) :
    (preactivationResidual f df).decodePull z g = df z*g := rfl

theorem dense_actual_activation_derivative (f : ℝ → ℝ) (z slope g : ℝ)
    (hf : HasDerivAt f slope z) :
    HasDerivAt (fun t => g*f t) (g*slope) z := hf.const_mul g

theorem dense_actual_affine_activation (f : ℝ → ℝ)
    (x w b slope dx dw db : ℝ) (hf : HasDerivAt f slope (x*w+b)) :
    HasDerivAt (fun t => f ((x+t*dx)*(w+t*dw)+(b+t*db)))
      (slope*(dx*w+x*dw+db)) 0 := by
  have hx : HasDerivAt (fun t : ℝ => x+t*dx) dx 0 := by
    convert (((hasDerivAt_id (0:ℝ)).mul_const dx).const_add x) using 1 <;> simp
  have hw : HasDerivAt (fun t : ℝ => w+t*dw) dw 0 := by
    convert (((hasDerivAt_id (0:ℝ)).mul_const dw).const_add w) using 1 <;> simp
  have hb : HasDerivAt (fun t : ℝ => b+t*db) db 0 := by
    convert (((hasDerivAt_id (0:ℝ)).mul_const db).const_add b) using 1 <;> simp
  have hz := (hx.fun_mul hw).fun_add hb
  have hz' : HasDerivAt (fun t : ℝ => (x+t*dx)*(w+t*dw)+(b+t*db))
      (dx*w+x*dw+db) 0 := by simpa using hz
  have hf' : HasDerivAt f slope ((x+0*dx)*(w+0*dw)+(b+0*db)) := by
    simpa using hf
  exact hf'.comp 0 hz'

theorem dense_coordinate_dagger {M N : Type*} [Fintype M] [Fintype N]
    (s dz g : Matrix M N ℝ) :
    (∑ i, ∑ j, g i j*(s i j*dz i j)) =
      ∑ i, ∑ j, (s i j*g i j)*dz i j := by
  apply Finset.sum_congr rfl
  intro i hi
  apply Finset.sum_congr rfl
  intro j hj
  ring

theorem dense_all_parameter_scalar (x w b q value slope dx dw db dq g : ℝ) :
    g*(slope*(dx*w+x*dw+db)*q+value*dq) =
      dx*(slope*g*q*w)+dw*(x*slope*g*q)+db*(slope*g*q)+dq*(value*g) := by ring

theorem dense_swiglu_all_parameter_scalar
    (x w u q value slope up dx dw du db dub dq g : ℝ) :
    g*((slope*(dx*w+x*dw+db)*up+value*(dx*u+x*du+dub))*q+value*up*dq) =
      dx*(slope*g*up*q*w+value*g*q*u)+
      dw*(x*slope*g*up*q)+db*(slope*g*up*q)+
      du*(x*value*g*q)+dub*(value*g*q)+dq*(value*up*g) := by ring

theorem dense_bias_dagger {M N : Type*} [Fintype M] [Fintype N]
    (g : Matrix M N ℝ) (db : N → ℝ) :
    (∑ i, ∑ j, g i j*db j) = ∑ j, db j*(∑ i, g i j) := by
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j hj
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i hi
  ring

def denseLocalCotangent {M N : Type*} (s g : Matrix M N ℝ) : Matrix M N ℝ :=
  fun i j => s i j * g i j

theorem dense_coordinate_then_matrix {M K N : Type*}
    [Fintype M] [Fintype K] [Fintype N]
    (x : Matrix M K ℝ) (w : Matrix K N ℝ)
    (dx : Matrix M K ℝ) (dw : Matrix K N ℝ) (s g : Matrix M N ℝ) :
    (∑ i, ∑ j, (s i j * (dx*w+x*dw) i j)*g i j) =
      (∑ i, ∑ k, dx i k * ((denseLocalCotangent s g*w.transpose) i k)) +
      (∑ k, ∑ j, dw k j * ((x.transpose*denseLocalCotangent s g) k j)) := by
  calc
    _ = ∑ i, ∑ j, (s i j*g i j)*(dx*w+x*dw) i j := by
      apply Finset.sum_congr rfl
      intro i hi
      apply Finset.sum_congr rfl
      intro j hj
      ring
    _ = _ := matrix_product_pullback x w dx dw (denseLocalCotangent s g)

/- Fixed one-level circuit count including one unary negation and optional
encoder replay. Allocation, packing and GPU latency are different metrics. -/
theorem dense_rank7_cached_cost (m k n : ℝ) :
    (21/4:ℝ)*m*k*n+m*k+k*n+m*n+m*n/4 =
      (21/4:ℝ)*m*k*n+m*k+k*n+(5/4:ℝ)*m*n := by ring

theorem dense_rank7_replayed_cost (m k n : ℝ) :
    (21/4:ℝ)*m*k*n+m*k+k*n+(5/4:ℝ)*m*n+(m*k+k*n) =
      (21/4:ℝ)*m*k*n+2*m*k+2*k*n+(5/4:ℝ)*m*n := by ring

theorem dense_rank7_cached_saving (m k n : ℝ) :
    (6*m*k*n-m*k-k*n-m*n) -
      ((21/4:ℝ)*m*k*n+m*k+k*n+(5/4:ℝ)*m*n) =
      (3/4:ℝ)*m*k*n-2*m*k-2*k*n-(9/4:ℝ)*m*n := by ring

theorem dense_rank7_replayed_saving (m k n : ℝ) :
    (6*m*k*n-m*k-k*n-m*n) -
      ((21/4:ℝ)*m*k*n+2*m*k+2*k*n+(5/4:ℝ)*m*n) =
      (3/4:ℝ)*m*k*n-3*m*k-3*k*n-(9/4:ℝ)*m*n := by ring

theorem dense_rank7_square_replay_admission (n : ℝ) (hn : 11 < n) :
    (21/4:ℝ)*n^3+(21/4:ℝ)*n^2 < 6*n^3-3*n^2 := by
  have hp : 0 < n := by linarith
  have hsq : 0 < n^2 := sq_pos_of_pos hp
  have hh : 0 < n^2*(n-11) := mul_pos hsq (by linarith)
  nlinarith

theorem dense_residual_half_tape (elements bytes : ℝ) :
    (2*elements*bytes-elements*bytes) = (2*elements*bytes)/2 := by ring

theorem dense_swiglu_half_tape (elements bytes : ℝ) :
    (4*elements*bytes-2*elements*bytes) = (4*elements*bytes)/2 := by ring

theorem dense_residual_no_projection_replay (matrix pointwise : ℝ) :
    matrix+pointwise-matrix = pointwise := by ring

theorem dense_residual_cost_guard (native decode saved : ℝ)
    (h : decode < saved) : native-saved+decode < native := by linarith

/- Relative error needs a lower bound on the observed result: magnitude bounds
on inputs alone do not protect cancellation. -/
theorem dense_absolute_error_from_channels (a b g ea eb : ℝ)
    (ha : |a| ≤ ea) (hb : |b| ≤ eb) :
    |g*(a+b)| ≤ |g| * (ea+eb) := by
  rw [abs_mul]
  exact mul_le_mul_of_nonneg_left
    (le_trans (abs_add_le a b) (add_le_add ha hb)) (abs_nonneg g)

end OjasXMath
