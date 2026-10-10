import OjasXMath.StreamedDagger

/-! Algebraic tangent projection results for the Euclidean sphere and Stiefel
constraint. These do not certify retractions, float constraint preservation,
optimizer convergence or an arbitrary Riemannian metric. -/
namespace OjasXMath

def sphereProject {I : Type*} [Fintype I] (x g : I → ℝ) : I → ℝ :=
  fun i => g i - x i * scalarPair x g

theorem sphere_projection_pairing {I : Type*} [Fintype I]
    (x g v : I → ℝ) :
    scalarPair v (sphereProject x g) =
      scalarPair v g - scalarPair v x * scalarPair x g := by
  simp only [sphereProject, scalarPair, mul_sub, Finset.sum_sub_distrib,
    Finset.sum_mul, mul_assoc]

theorem sphere_projection_tangent {I : Type*} [Fintype I]
    (x g : I → ℝ) (hunit : scalarPair x x = 1) :
    scalarPair x (sphereProject x g) = 0 := by
  rw [sphere_projection_pairing, hunit]
  ring

theorem sphere_projection_preserves_direction {I : Type*} [Fintype I]
    (x g v : I → ℝ) (htangent : scalarPair v x = 0) :
    scalarPair v (sphereProject x g) = scalarPair v g := by
  rw [sphere_projection_pairing, htangent]
  ring

theorem sphere_projection_idempotent {I : Type*} [Fintype I]
    (x g : I → ℝ) (hunit : scalarPair x x = 1) :
    sphereProject x (sphereProject x g) = sphereProject x g := by
  funext i
  simp only [sphereProject, sphere_projection_tangent x g hunit, mul_zero, sub_zero]

noncomputable def symmetricPart {N : Type*} (a : Matrix N N ℝ) : Matrix N N ℝ :=
  (1/2:ℝ) • (a+a.transpose)

theorem symmetric_part_transpose {N : Type*} (a : Matrix N N ℝ) :
    (symmetricPart a).transpose = symmetricPart a := by
  simp [symmetricPart, add_comm]

theorem symmetric_part_double {N : Type*} (a : Matrix N N ℝ) :
    symmetricPart a + symmetricPart a = a+a.transpose := by
  ext i j
  simp only [symmetricPart, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
  ring

noncomputable def stiefelProject {M N : Type*} [Fintype M] [Fintype N]
    (x g : Matrix M N ℝ) : Matrix M N ℝ :=
  g-x*symmetricPart (x.transpose*g)

theorem stiefel_projection_tangent {M N : Type*} [Fintype M] [Fintype N]
    [DecidableEq N] (x g : Matrix M N ℝ) (horth : x.transpose*x=1) :
    x.transpose*stiefelProject x g+(stiefelProject x g).transpose*x=0 := by
  have hsym := symmetric_part_transpose (x.transpose*g)
  have hdouble := symmetric_part_double (x.transpose*g)
  have htranspose : (x.transpose*g).transpose=g.transpose*x := by simp
  simp only [stiefelProject, Matrix.transpose_sub, Matrix.transpose_mul,
    Matrix.mul_sub, Matrix.sub_mul]
  rw [← Matrix.mul_assoc, horth, one_mul, hsym, Matrix.mul_assoc, horth, mul_one]
  rw [htranspose] at hdouble
  calc
    _ = (x.transpose*g+g.transpose*x) -
      (symmetricPart (x.transpose*g)+symmetricPart (x.transpose*g)) := by abel
    _ = 0 := by rw [hdouble]; simp

end OjasXMath
