import OjasXMath.Contraction
import OjasXMath.Residuals

/-! Exact-real nonlinear structure. A shared ReLU mask is a runtime premise,
not an assertion about ordinary random batches. Pairing identities require
primitive derivative contracts; they are not IEEE execution certificates. -/
namespace OjasXMath

theorem joint_tree_node_cost (a b c : ℝ) :
    (2*a*b*c-a*c)+(2*a*b*c-a*b)+(2*a*b*c-b*c) =
    6*a*b*c-(a*c+a*b+b*c) := by ring

theorem three_weight_chain_forward {M K N P Q : Type*}
    [Fintype K] [Fintype N] [Fintype P]
    (a : Matrix M K ℝ) (b : Matrix K N ℝ) (c : Matrix N P ℝ) (d : Matrix P Q ℝ) :
    ((a*b)*c)*d = a*(b*(c*d)) := by simp only [Matrix.mul_assoc]

theorem relu_value_gate (x : ℝ) :
    max 0 x = x*(if 0 < x then 1 else 0) := by
  by_cases h : 0 < x
  · simp [h, max_eq_right (le_of_lt h)]
  · simp [h, max_eq_left (le_of_not_gt h)]

theorem relu_positive_derivative (x : ℝ) (hx : 0 < x) :
    HasDerivAt (fun z : ℝ => max 0 z) 1 x := by
  apply (hasDerivAt_id x).congr_of_eventuallyEq
  filter_upwards [eventually_gt_nhds hx] with y hy
  simp [max_eq_right (le_of_lt hy)]

theorem relu_negative_derivative (x : ℝ) (hx : x < 0) :
    HasDerivAt (fun z : ℝ => max 0 z) 0 x := by
  apply (hasDerivAt_const x (0 : ℝ)).congr_of_eventuallyEq
  filter_upwards [eventually_lt_nhds hx] with y hy
  simp [max_eq_left (le_of_lt hy)]

theorem common_relu_mask_forward {M N : Type*} [Fintype N] [DecidableEq N]
    (z : Matrix M N ℝ) (active : N → Prop) [DecidablePred active]
    (h : ∀ i j, 0 < z i j ↔ active j) :
    (fun i j => max 0 (z i j)) = z * Matrix.diagonal (fun j => if active j then (1 : ℝ) else 0) := by
  ext i j
  rw [Matrix.mul_diagonal, relu_value_gate]
  simp only [h i j]

theorem common_mask_input_pullback {M K N P : Type*}
    [Fintype N] [Fintype P] [DecidableEq N]
    (b : Matrix K N ℝ) (c : Matrix N P ℝ) (g : Matrix M P ℝ) (d : N → ℝ) :
    ((g*c.transpose)*Matrix.diagonal d)*b.transpose =
      g*(b*Matrix.diagonal d*c).transpose := by
  simp only [Matrix.transpose_mul, Matrix.diagonal_transpose, Matrix.mul_assoc]

theorem common_mask_first_weight_pullback {M K N P : Type*}
    [Fintype M] [Fintype P] [Fintype N] [DecidableEq N]
    (a : Matrix M K ℝ) (c : Matrix N P ℝ) (g : Matrix M P ℝ) (d : N → ℝ) :
    a.transpose*((g*c.transpose)*Matrix.diagonal d) =
      ((a.transpose*g)*c.transpose)*Matrix.diagonal d := by
  simp only [Matrix.mul_assoc]

theorem common_mask_second_weight_pullback {M K N P : Type*}
    [Fintype M] [Fintype K] [Fintype N] [DecidableEq N]
    (a : Matrix M K ℝ) (b : Matrix K N ℝ) (g : Matrix M P ℝ) (d : N → ℝ) :
    ((a*b)*Matrix.diagonal d).transpose*g =
      Matrix.diagonal d*b.transpose*(a.transpose*g) := by
  simp only [Matrix.transpose_mul, Matrix.diagonal_transpose, Matrix.mul_assoc]

/- Exact factors for a row/column-separable activation derivative mask.
The factor width is Q×P; no SVD or approximation is supplied by this theorem. -/
theorem separable_mask_factor_expansion {M N Q P : Type*} [Fintype Q] [Fintype P]
    (u : M → Q → ℝ) (v : N → Q → ℝ) (g : M → P → ℝ) (c : N → P → ℝ)
    (i : M) (j : N) :
    (∑ q, u i q*v j q)*(∑ l, g i l*c j l) =
      ∑ q, ∑ l, (u i q*g i l)*(v j q*c j l) := by
  simp only [Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro q _
  apply Finset.sum_congr rfl
  intro l _
  ring

/- Diagonal-minus-rank-one maps are self-adjoint. Specializing d=v=p
gives the normalized softmax VJP once its derivative contract is supplied. -/
theorem diagonal_rankone_pairing {N : Type*} [Fintype N]
    (d v u g : N → ℝ) :
    (∑ i, (d i*u i-v i*(∑ j, v j*u j))*g i) =
      ∑ i, u i*(d i*g i-v i*(∑ j, v j*g j)) := by
  have hl : (∑ i, (d i*u i-v i*(∑ j, v j*u j))*g i) =
      (∑ i, d i*u i*g i)-(∑ j, v j*u j)*(∑ i, v i*g i) := by
    simp only [sub_mul, Finset.sum_sub_distrib]
    congr 1
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring
  have hr : (∑ i, u i*(d i*g i-v i*(∑ j, v j*g j))) =
      (∑ i, d i*u i*g i)-(∑ j, v j*g j)*(∑ i, v i*u i) := by
    simp only [mul_sub, Finset.sum_sub_distrib]
    congr 1
    · apply Finset.sum_congr rfl
      intro i _
      ring
    · rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i _
      ring
  rw [hl, hr]
  ring

theorem softmax_pullback_sum_zero {N : Type*} [Fintype N]
    (p g : N → ℝ) (hp : ∑ i, p i = 1) :
    (∑ i, p i*(g i-(∑ j, p j*g j))) = 0 := by
  simp only [mul_sub, Finset.sum_sub_distrib]
  rw [← Finset.sum_mul, hp]
  ring

/- Attention row identity reuses O=P V for the softmax reduction scalar. -/
theorem attention_row_scalar {N D : Type*} [Fintype N] [Fintype D]
    (p : N → ℝ) (v : N → D → ℝ) (g : D → ℝ) :
    (∑ j, p j*(∑ k, g k*v j k)) =
      ∑ k, g k*(∑ j, p j*v j k) := by
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k _
  apply Finset.sum_congr rfl
  intro j _
  ring

/- LayerNorm's mean direction vanishes; epsilon does not change this identity.
The normalized direction is generally not null when epsilon is positive. -/
theorem layernorm_shift_null {N : Type*} [Fintype N]
    (c g : N → ℝ) (r invn : ℝ)
    (hcount : invn*(∑ _i : N, (1 : ℝ)) = 1) (hcenter : ∑ i, c i = 0) :
    (∑ i, r*(g i-invn*(∑ j, g j)-r^2*c i*invn*(∑ j, c j*g j))) = 0 := by
  have hterm : ∀ i, r*(g i-invn*(∑ j, g j)-r^2*c i*invn*(∑ j, c j*g j)) =
      r*g i-(r*(∑ j, g j)*invn)*1-(r*r^2*invn*(∑ j, c j*g j))*c i := by
    intro i
    ring
  simp_rw [hterm]
  rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib,
      ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum]
  calc
    _ = r*(∑ j, g j)-r*(∑ j, g j)*(invn*(∑ _i : N, (1 : ℝ)))-
        (r*r^2*invn*(∑ j, c j*g j))*(∑ i, c i) := by ring
    _ = 0 := by rw [hcount, hcenter]; ring

/- LayerNorm supplied-Jacobian expansion through adjacent matrices. The
diagonal/low-rank coefficients must come from the actual forward statistics. -/
theorem layernorm_gram_input_reassociation {M K N P : Type*}
    [Fintype M] [Fintype K] [Fintype N] [Fintype P]
    (a : Matrix M K ℝ) (b : Matrix K N ℝ) (g : Matrix M P ℝ)
    (c : Matrix P N ℝ) (r alpha : Matrix M M ℝ) (u : Matrix M N ℝ) :
    (((r*g)*c+u)-(alpha*a)*b)*b.transpose =
      (r*g)*(c*b.transpose)+u*b.transpose-(alpha*a)*(b*b.transpose) := by
  simp only [Matrix.sub_mul, Matrix.add_mul, Matrix.mul_assoc]

theorem layernorm_gram_weight_reassociation {M K N P : Type*}
    [Fintype M] [Fintype K] [Fintype P]
    (a : Matrix M K ℝ) (b : Matrix K N ℝ) (g : Matrix M P ℝ)
    (c : Matrix P N ℝ) (r alpha : Matrix M M ℝ) (u : Matrix M N ℝ) :
    a.transpose*(((r*g)*c+u)-(alpha*a)*b) =
      (a.transpose*(r*g))*c+a.transpose*u-(a.transpose*(alpha*a))*b := by
  simp only [Matrix.mul_sub, Matrix.mul_add, Matrix.mul_assoc]

theorem layernorm_shared_weight_cotangent {M K P : Type*}
    [Fintype M] [DecidableEq M]
    (a : Matrix M K ℝ) (g : Matrix M P ℝ) (r : M → ℝ) :
    a.transpose*(Matrix.diagonal r*g) =
      (g.transpose*(Matrix.diagonal r*a)).transpose := by
  simp only [Matrix.transpose_mul, Matrix.diagonal_transpose,
      Matrix.transpose_transpose, Matrix.mul_assoc]

theorem layernorm_second_weight_reassociation {M K N P : Type*}
    [Fintype M] [Fintype K] [Fintype N]
    (a : Matrix M K ℝ) (b : Matrix K N ℝ) (g : Matrix M P ℝ)
    (r : Matrix M M ℝ) (gamma : Matrix N N ℝ) (u v : Matrix M N ℝ) :
    g.transpose*(((r*a)*b+u)*gamma+v) =
      ((g.transpose*(r*a))*b+g.transpose*u)*gamma+g.transpose*v := by
  simp only [Matrix.mul_add, Matrix.add_mul, Matrix.mul_assoc]

theorem layernorm_structured_leading_cost (m k n p : ℝ) :
    2*p*m*k+2*p*k*n+2*n*p*k+2*p*n*k+2*m*p*k+
      2*k*n*k+2*m*k*k+2*k*m*k+2*n*k*k =
    6*p*n*k+4*m*p*k+4*n*k^2+4*m*k^2 := by ring

theorem layernorm_leading_more_than_half (m k n p : ℝ)
    (hm : 0 < m) (hk : 0 < k) (hn : 0 < n) (hp : 0 < p)
    (hmk : 8*k ≤ m) (hnk : 8*k ≤ n) :
    2*(6*p*n*k+4*m*p*k+4*n*k^2+4*m*k^2) < 4*m*n*(k+p) := by
  have h1 := mul_le_mul_of_nonneg_right hmk (le_of_lt (mul_pos hn hk))
  have h2 := mul_le_mul_of_nonneg_right hnk (le_of_lt (mul_pos hm hk))
  have h3 := mul_le_mul_of_nonneg_right hmk (le_of_lt (mul_pos hn hp))
  have h4 := mul_le_mul_of_nonneg_right hnk (le_of_lt (mul_pos hm hp))
  have h5 := mul_pos (mul_pos hm hn) hk
  nlinarith

theorem linear_row_mean {N K : Type*} [Fintype N] [Fintype K]
    (x : K → ℝ) (w : N → K → ℝ) (invn : ℝ) :
    invn*(∑ j, ∑ k, x k*w j k) = ∑ k, x k*(invn*(∑ j, w j k)) := by
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k _
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem linear_row_centering {K : Type*} [Fintype K]
    (x w mean : K → ℝ) :
    (∑ k, x k*(w k-mean k)) = (∑ k, x k*w k)-(∑ k, x k*mean k) := by
  simp only [mul_sub, Finset.sum_sub_distrib]

theorem centered_weight_gram_moment {N K : Type*} [Fintype N] [Fintype K]
    (x : K → ℝ) (w : N → K → ℝ) :
    (∑ j, (∑ k, x k*w j k)^2) =
      ∑ k, ∑ l, x k*(∑ j, w j k*w j l)*x l := by
  have h : ∀ j, (∑ k, x k*w j k)^2 = ∑ k, ∑ l, (x k*w j k)*(x l*w j l) := by
    intro j
    rw [pow_two, Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro k _
    rw [Finset.mul_sum]
  simp_rw [h]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k _
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro l _
  simp only [Finset.mul_sum, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem gram_normalized_forward_reassociation {M K N P : Type*}
    [Fintype M] [Fintype K] [Fintype N]
    (a : Matrix M K ℝ) (b : Matrix K N ℝ) (c : Matrix N P ℝ)
    (r : Matrix M M ℝ) (gamma : Matrix N N ℝ) (u v : Matrix M N ℝ) :
    ((r*(a*b-u))*gamma+v)*c =
      (r*a)*(b*gamma*c)-(r*u)*(gamma*c)+v*c := by
  simp only [Matrix.mul_sub, Matrix.sub_mul, Matrix.add_mul, Matrix.mul_assoc]

theorem gram_normalized_joint_leading_cost (m k n p : ℝ) :
    6*k*n*k+6*m*k*k+6*p*n*k+6*m*k*p =
      6*(n*k^2+m*k^2+n*p*k+m*p*k) := by ring

end OjasXMath
