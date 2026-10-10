import OjasXMath.Algebra
import OjasXMath.Residuals
import Mathlib.Analysis.Calculus.Deriv.Add

/-! A finite bilinear circuit with arbitrary encoders/decoder. Its cotangent
    channels can be consumed once, with no requirement to retain encoded
    operands across the forward/backward boundary. Floating schedules are
    separate implementations of these exact commutative-ring statements. -/
namespace OjasXMath

def scalarPair {I R : Type*} [Fintype I] [CommRing R] (x y : I → R) : R :=
  ∑ i, x i*y i

def linearApply {I J R : Type*} [Fintype I] [CommRing R]
    (c : J → I → R) (x : I → R) : J → R := fun j => ∑ i, c j i*x i

def linearAdjoint {I J R : Type*} [Fintype J] [CommRing R]
    (c : J → I → R) (g : J → R) : I → R := fun i => ∑ j, c j i*g j

theorem linear_circuit_dagger {I J R : Type*} [Fintype I] [Fintype J] [CommRing R]
    (c : J → I → R) (x : I → R) (g : J → R) :
    scalarPair (linearApply c x) g = scalarPair x (linearAdjoint c g) := by
  simp only [scalarPair, linearApply, linearAdjoint, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i hi
  apply Finset.sum_congr rfl
  intro j hj
  ring

theorem pointwise_bilinear_dagger {I R : Type*} [Fintype I] [CommRing R]
    (a b da db g : I → R) :
    scalarPair (fun i => da i*b i+a i*db i) g =
      scalarPair da (fun i => g i*b i)+scalarPair db (fun i => g i*a i) := by
  simp only [scalarPair, add_mul, Finset.sum_add_distrib]
  congr 1 <;> apply Finset.sum_congr rfl <;> intro i hi <;> ring

def circuitTangent {A B C Q R : Type*} [Fintype A] [Fintype B] [Fintype Q] [CommRing R]
    (u : Q → A → R) (v : Q → B → R) (w : C → Q → R)
    (a da : A → R) (b db : B → R) : C → R :=
  linearApply w (fun q => linearApply u da q*linearApply v b q +
    linearApply u a q*linearApply v db q)

def bilinearValue {A B C Q R : Type*} [Fintype A] [Fintype B] [Fintype Q] [CommRing R]
    (u : Q → A → R) (v : Q → B → R) (w : C → Q → R)
    (a : A → R) (b : B → R) : C → R :=
  linearApply w (fun q => linearApply u a q*linearApply v b q)

theorem linear_apply_affine {I J R : Type*} [Fintype I] [CommRing R]
    (c : J → I → R) (a da : I → R) (t : R) (j : J) :
    linearApply c (fun i => a i+t*da i) j = linearApply c a j+t*linearApply c da j := by
  simp only [linearApply, mul_add, Finset.sum_add_distrib, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  ring

theorem circuit_direction_expansion {A B C Q R : Type*}
    [Fintype A] [Fintype B] [Fintype Q] [CommRing R]
    (u : Q → A → R) (v : Q → B → R) (w : C → Q → R)
    (a da : A → R) (b db : B → R) (t : R) (j : C) :
    bilinearValue u v w (fun i => a i+t*da i) (fun i => b i+t*db i) j =
      bilinearValue u v w a b j+t*circuitTangent u v w a da b db j+
      (t*t)*bilinearValue u v w da db j := by
  unfold bilinearValue
  simp_rw [linear_apply_affine]
  change (∑ q, w j q*((linearApply u a q+t*linearApply u da q)*
    (linearApply v b q+t*linearApply v db q))) =
    (∑ q, w j q*(linearApply u a q*linearApply v b q))+
    t*(∑ q, w j q*(linearApply u da q*linearApply v b q+
      linearApply u a q*linearApply v db q))+
    (t*t)*(∑ q, w j q*(linearApply u da q*linearApply v db q))
  calc
    _ = ∑ q, ((w j q*(linearApply u a q*linearApply v b q)+
          t*(w j q*(linearApply u da q*linearApply v b q+
            linearApply u a q*linearApply v db q)))+
          (t*t)*(w j q*(linearApply u da q*linearApply v db q))) := by
      apply Finset.sum_congr rfl
      intro q hq
      ring
    _ = _ := by simp only [Finset.sum_add_distrib, ← Finset.mul_sum]

theorem circuit_direction_derivative {A B C Q : Type*}
    [Fintype A] [Fintype B] [Fintype Q]
    (u : Q → A → ℝ) (v : Q → B → ℝ) (w : C → Q → ℝ)
    (a da : A → ℝ) (b db : B → ℝ) (j : C) :
    HasDerivAt (fun t : ℝ => bilinearValue u v w
      (fun i => a i+t*da i) (fun i => b i+t*db i) j)
      (circuitTangent u v w a da b db j) 0 := by
  have heq : (fun t : ℝ => bilinearValue u v w
      (fun i => a i+t*da i) (fun i => b i+t*db i) j) =
      (fun t => bilinearValue u v w a b j+t*circuitTangent u v w a da b db j+
        (t*t)*bilinearValue u v w da db j) := by
    funext t
    exact circuit_direction_expansion u v w a da b db t j
  rw [heq]
  have h : HasDerivAt
      (fun t : ℝ => bilinearValue u v w a b j+t*circuitTangent u v w a da b db j+
        (t*t)*bilinearValue u v w da db j)
      ((0+1*circuitTangent u v w a da b db j)+(2*0)*bilinearValue u v w da db j) 0 :=
    ((hasDerivAt_const (0:ℝ) (bilinearValue u v w a b j)).add
      ((hasDerivAt_id (0:ℝ)).mul_const (circuitTangent u v w a da b db j))).add
      ((square_has_derivative 0).mul_const (bilinearValue u v w da db j))
  simpa only [mul_zero, zero_mul, one_mul, zero_add, add_zero] using h

def circuitPullback {A B C Q R : Type*} [Fintype A] [Fintype B] [Fintype C]
    [Fintype Q] [CommRing R]
    (u : Q → A → R) (v : Q → B → R) (w : C → Q → R)
    (a : A → R) (b : B → R) (g : C → R) : (A → R) × (B → R) :=
  (linearAdjoint u (fun q => linearAdjoint w g q*linearApply v b q),
   linearAdjoint v (fun q => linearAdjoint w g q*linearApply u a q))

/- The unified bilinear-circuit dagger theorem. Gradients factor through ONE
   decoder cotangent channel per product; no activation inverse is involved. -/
theorem unified_streamed_bilinear_dagger {A B C Q R : Type*}
    [Fintype A] [Fintype B] [Fintype C] [Fintype Q] [CommRing R]
    (u : Q → A → R) (v : Q → B → R) (w : C → Q → R)
    (a da : A → R) (b db : B → R) (g : C → R) :
    scalarPair (circuitTangent u v w a da b db) g =
      scalarPair da (circuitPullback u v w a b g).1 +
      scalarPair db (circuitPullback u v w a b g).2 := by
  dsimp only [circuitTangent, circuitPullback]
  rw [linear_circuit_dagger, pointwise_bilinear_dagger]
  rw [linear_circuit_dagger, linear_circuit_dagger]

/- An actual analytic derivative of the scalarized output, for EVERY direction.
   Together with the pairing identity this characterizes the real cotangents. -/
theorem unified_circuit_gradient_direction {A B C Q : Type*}
    [Fintype A] [Fintype B] [Fintype C] [Fintype Q]
    (u : Q → A → ℝ) (v : Q → B → ℝ) (w : C → Q → ℝ)
    (a da : A → ℝ) (b db : B → ℝ) (g : C → ℝ) :
    HasDerivAt (fun t : ℝ => scalarPair (bilinearValue u v w
      (fun i => a i+t*da i) (fun i => b i+t*db i)) g)
      (scalarPair da (circuitPullback u v w a b g).1+
       scalarPair db (circuitPullback u v w a b g).2) 0 := by
  have h := HasDerivAt.fun_sum (u := Finset.univ)
    (fun j _ => (circuit_direction_derivative u v w a da b db j).mul_const (g j))
  change HasDerivAt (fun t : ℝ => scalarPair (bilinearValue u v w
      (fun i => a i+t*da i) (fun i => b i+t*db i)) g)
      (scalarPair (circuitTangent u v w a da b db) g) 0 at h
  rwa [unified_streamed_bilinear_dagger] at h

def streamAccumulate {I R : Type*} [Add R] (contribution : I → R)
    (indices : List I) (initial : R) : R :=
  indices.foldl (fun acc i => acc+contribution i) initial

theorem streamed_accumulation_correct {I R : Type*} [AddCommMonoid R]
    (contribution : I → R) (indices : List I) (initial : R) :
    streamAccumulate contribution indices initial = initial+(indices.map contribution).sum := by
  induction indices generalizing initial with
  | nil => simp [streamAccumulate]
  | cons i rest ih =>
      change streamAccumulate contribution rest (initial+contribution i) = _
      rw [ih]
      simp [List.map_cons, List.sum_cons, add_assoc]

/- Storage statements exclude required output/cotangent buffers and transient
   tiles. They are a cost model, not a proof of a runtime allocator. -/
theorem encoded_operand_tape_elimination (a b : Nat) :
    2*(a+b)-(a+b) = a+b := by simp [two_mul]

theorem streamed_scratch_independent_of_channels (channels tile : Nat) (h : 1 ≤ channels) :
    tile ≤ channels*tile := by
  calc
    tile = 1*tile := (one_mul _).symm
    _ ≤ channels*tile := Nat.mul_le_mul_right tile h

/- Operand-only tape reconstructs channels with pointwise additions, not GEMMs.
   The extra seven quarter-block additions per side include both reconstruction
   and encoder adjoints; casts/sign changes/copies are outside this model. -/
theorem operand_only_backward_cost (m k n : ℝ) :
    7*(2*(m/2)*(n/2)*(k/2)-(m/2)*(k/2)) +
      7*(2*(k/2)*(m/2)*(n/2)-(k/2)*(n/2)) +
      4*(m/2)*(n/2)+14*(m/2)*(k/2)+14*(k/2)*(n/2) =
    (7/2:ℝ)*m*k*n+m*n+(7/4:ℝ)*(m*k+k*n) := by ring

theorem operand_only_joint_cost (m k n : ℝ) :
    ((7/4:ℝ)*m*k*n+m*k+k*n)+
      ((7/2:ℝ)*m*k*n+m*n+(7/4:ℝ)*(m*k+k*n)) =
    (21/4:ℝ)*m*k*n+(11/4:ℝ)*(m*k+k*n)+m*n := by ring

theorem operand_only_square_crossover (n : ℝ) (hn : (38/3:ℝ) < n) :
    (21/4:ℝ)*n^3+(13/2:ℝ)*n^2 < 6*n^3-3*n^2 := by
  have hpos : 0 < n := by linarith
  have hsq : 0 < n^2 := sq_pos_of_pos hpos
  have hprod : 0 < n^2*(n-38/3) := mul_pos hsq (by linarith)
  nlinarith

end OjasXMath
