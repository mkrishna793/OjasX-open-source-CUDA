import Mathlib.Data.Matrix.Mul
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.NoncommRing
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith

/-! Exact algebra. These theorems do not assert IEEE-754 equality. -/
namespace OjasXMath

@[ext] structure Block2 (R : Type*) where
  a11 : R
  a12 : R
  a21 : R
  a22 : R
deriving Repr

def stdBlockMul {R : Type*} [Ring R] (a b : Block2 R) : Block2 R :=
  ⟨a.a11*b.a11+a.a12*b.a21, a.a11*b.a12+a.a12*b.a22,
   a.a21*b.a11+a.a22*b.a21, a.a21*b.a12+a.a22*b.a22⟩

def winogradBlockMul {R : Type*} [Ring R] (a b : Block2 R) : Block2 R :=
  let s1 := a.a21+a.a22
  let s2 := s1-a.a11
  let s3 := a.a11-a.a21
  let s4 := a.a12-s2
  let t1 := b.a12-b.a11
  let t2 := b.a22-t1
  let t3 := b.a22-b.a12
  let t4 := t2-b.a21
  let p1 := a.a11*b.a11
  let p2 := a.a12*b.a21
  let p3 := s4*b.a22
  let p4 := a.a22*t4
  let p5 := s1*t1
  let p6 := s2*t2
  let p7 := s3*t3
  let u1 := p1+p6
  let u2 := u1+p7
  ⟨p1+p2, (u1+p5)+p3, u2-p4, u2+p5⟩

theorem winograd_block_equivalence {R : Type*} [Ring R] (a b : Block2 R) :
    winogradBlockMul a b = stdBlockMul a b := by
  apply Block2.ext <;> dsimp [winogradBlockMul, stdBlockMul] <;> noncomm_ring

@[ext] structure Channels7 (R : Type*) where
  c1 : R
  c2 : R
  c3 : R
  c4 : R
  c5 : R
  c6 : R
  c7 : R
deriving Repr

def decoderPullback {R : Type*} [AddCommGroup R] (g : Block2 R) : Channels7 R :=
  let v1 := g.a21+g.a22
  let v2 := g.a12+g.a22
  let v3 := g.a12+v1
  let v4 := g.a11+v3
  ⟨v4,g.a11,g.a12,-g.a21,v2,v3,v1⟩

theorem decoder_four_additions {R : Type*} [AddCommGroup R] (g : Block2 R) :
    decoderPullback g =
      ⟨g.a11+g.a12+g.a21+g.a22,g.a11,g.a12,-g.a21,
       g.a12+g.a22,g.a12+g.a21+g.a22,g.a21+g.a22⟩ := by
  apply Channels7.ext <;> dsimp [decoderPullback] <;> abel

theorem decoder_adjoint_pairing (p : Channels7 ℝ) (g : Block2 ℝ) :
    (p.c1+p.c2)*g.a11 + (p.c1+p.c3+p.c5+p.c6)*g.a12 +
    (p.c1-p.c4+p.c6+p.c7)*g.a21 + (p.c1+p.c5+p.c6+p.c7)*g.a22 =
    p.c1*(decoderPullback g).c1 + p.c2*(decoderPullback g).c2 +
    p.c3*(decoderPullback g).c3 + p.c4*(decoderPullback g).c4 +
    p.c5*(decoderPullback g).c5 + p.c6*(decoderPullback g).c6 +
    p.c7*(decoderPullback g).c7 := by
  dsimp [decoderPullback]
  ring

/- Reverse the shared encoder circuits instead of summing transpose columns
   independently: seven additions/subtractions each, rather than ten. -/
def encoderAPullback {R : Type*} [AddCommGroup R] (c : Channels7 R) : Block2 R :=
  let ds2 := c.c6-c.c3
  let ds1 := c.c5+ds2
  ⟨c.c1-ds2+c.c7, c.c2+c.c3, ds1-c.c7, c.c4+ds1⟩

def encoderBPullback {R : Type*} [AddCommGroup R] (c : Channels7 R) : Block2 R :=
  let dt2 := c.c6+c.c4
  let dt1 := c.c5-dt2
  ⟨c.c1-dt1, dt1-c.c7, c.c2-c.c4, c.c3+dt2+c.c7⟩

theorem encoder_a_seven_additions {R : Type*} [AddCommGroup R] (c : Channels7 R) :
    encoderAPullback c =
      ⟨c.c1+c.c3-c.c6+c.c7, c.c2+c.c3,
       -c.c3+c.c5+c.c6-c.c7, -c.c3+c.c4+c.c5+c.c6⟩ := by
  apply Block2.ext <;> dsimp [encoderAPullback] <;> abel

theorem encoder_b_seven_additions {R : Type*} [AddCommGroup R] (c : Channels7 R) :
    encoderBPullback c =
      ⟨c.c1-c.c5+c.c6+c.c4, c.c5-c.c6-c.c4-c.c7,
       c.c2-c.c4, c.c3+c.c6+c.c4+c.c7⟩ := by
  apply Block2.ext <;> dsimp [encoderBPullback] <;> abel

theorem dagger_involution {m n : Type*} (a : Matrix m n ℝ) :
    a.transpose.transpose = a := rfl

theorem dagger_composition {m n p : Type*} [Fintype n]
    (a : Matrix n m ℝ) (b : Matrix p n ℝ) :
    (b*a).transpose = a.transpose*b.transpose := by
  exact Matrix.transpose_mul b a

theorem matrix_product_pullback
    {m k n : Type*} [Fintype m] [Fintype k] [Fintype n]
    (a : Matrix m k ℝ) (b : Matrix k n ℝ)
    (da : Matrix m k ℝ) (db : Matrix k n ℝ) (g : Matrix m n ℝ) :
    (∑ i, ∑ j, g i j * ((da*b+a*db) i j)) =
    (∑ i, ∑ t, da i t * ((g*b.transpose) i t)) +
    (∑ t, ∑ j, db t j * ((a.transpose*g) t j)) := by
  simp only [Matrix.add_apply, Matrix.mul_apply, Matrix.transpose_apply,
    mul_add, Finset.sum_add_distrib, Finset.mul_sum]
  congr 1
  · calc
      _ = ∑ i, ∑ t, ∑ j, g i j * (da i t * b t j) := by
        apply Finset.sum_congr rfl
        intro i hi
        rw [Finset.sum_comm]
      _ = _ := by
        apply Finset.sum_congr rfl
        intro i hi
        apply Finset.sum_congr rfl
        intro t ht
        apply Finset.sum_congr rfl
        intro j hj
        ring
  · calc
      _ = ∑ i, ∑ t, ∑ j, g i j * (a i t * db t j) := by
        apply Finset.sum_congr rfl
        intro i hi
        rw [Finset.sum_comm]
      _ = ∑ t, ∑ i, ∑ j, g i j * (a i t * db t j) := by
        rw [Finset.sum_comm]
      _ = ∑ t, ∑ j, ∑ i, g i j * (a i t * db t j) := by
        apply Finset.sum_congr rfl
        intro t ht
        rw [Finset.sum_comm]
      _ = _ := by
        apply Finset.sum_congr rfl
        intro t ht
        apply Finset.sum_congr rfl
        intro j hj
        apply Finset.sum_congr rfl
        intro i hi
        ring

theorem cotangent_fanout {R : Type*} [AddCommMonoid R] (g1 g2 g3 : R) :
    g1+(g2+g3) = (g1+g2)+g3 := (add_assoc _ _ _).symm

theorem shared_projection_pullback {m k n : Type*} [Fintype n]
    (g1 g2 : Matrix m n ℝ) (w : Matrix n k ℝ) :
    (g1+g2)*w = g1*w+g2*w := Matrix.add_mul _ _ _

/- S/A requires division by two and commutative real scalars, not arbitrary rings. -/
theorem sab_scalar_reconstruction (w wt : ℝ) :
    (w+wt)/2 + (w-wt)/2 = w := by ring

theorem one_level_cost (m k n : ℝ) :
    7*(2*(m/2)*(k/2)*(n/2)-(m/2)*(n/2)) +
      4*(m/2)*(k/2)+4*(k/2)*(n/2)+7*(m/2)*(n/2) =
    (7/4:ℝ)*m*k*n+m*k+k*n := by ring

theorem square_arithmetic_crossover (n : ℝ) (hn : 12 < n) :
    (7/4:ℝ)*n^3+2*n^2 < 2*n^3-n^2 := by
  have hpos : 0 < n := by linarith
  have hsq : 0 < n^2 := sq_pos_of_pos hpos
  have hprod : 0 < n^2*(n-12) := mul_pos hsq (by linarith)
  nlinarith

theorem two_level_cost (n : ℝ) :
    49*(2*(n/4)^3-(n/4)^2)+15*(n/2)^2+105*(n/4)^2 =
    (49/32:ℝ)*n^3+(29/4:ℝ)*n^2 := by ring

theorem one_level_backward_cost (m k n : ℝ) :
    7*(2*(m/2)*(n/2)*(k/2)-(m/2)*(k/2)) +
      7*(2*(k/2)*(m/2)*(n/2)-(k/2)*(n/2)) +
      4*(m/2)*(n/2)+7*(m/2)*(k/2)+7*(k/2)*(n/2) =
    (7/2:ℝ)*m*k*n+m*n := by ring

theorem one_level_joint_cost (m k n : ℝ) :
    ((7/4:ℝ)*m*k*n+m*k+k*n)+((7/2:ℝ)*m*k*n+m*n) =
    (21/4:ℝ)*m*k*n+m*k+k*n+m*n := by ring

theorem square_joint_crossover (n : ℝ) (hn : 8 < n) :
    (21/4:ℝ)*n^3+3*n^2 < 6*n^3-3*n^2 := by
  have hpos : 0 < n := by linarith
  have hsq : 0 < n^2 := sq_pos_of_pos hpos
  have hprod : 0 < n^2*(n-8) := mul_pos hsq (by linarith)
  nlinarith

/- Conditional scheduler model: seven quadrant buffers and ONE live child.
   This is not a certificate for the Python training tape or all GPU caches. -/
noncomputable def depthFirstWorkspace : Nat → ℝ → ℝ
  | 0, _ => 0
  | d+1, n => (7/4:ℝ)*n^2 + depthFirstWorkspace d (n/2)

theorem depth_first_workspace_bound (d : Nat) :
    ∀ n : ℝ, depthFirstWorkspace d n ≤ (7/3:ℝ)*n^2 := by
  induction d with
  | zero => intro n; simp [depthFirstWorkspace]; positivity
  | succ d ih =>
      intro n
      have h := ih (n/2)
      simp only [depthFirstWorkspace]
      nlinarith

end OjasXMath
