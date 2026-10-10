import OjasXMath.Algebra

/-! Contraction reassociation with known factors; no truncation or activation
crossing. Equalities are exact ring algebra, not rounded-bit statements. -/
namespace OjasXMath

/- G=L R^T is a provided cotangent factorization. -/
theorem factored_cotangent_input {M K N Q : Type*} [Fintype N] [Fintype Q]
    (b : Matrix K N ℝ) (l : Matrix M Q ℝ) (r : Matrix N Q ℝ) :
    (l*r.transpose)*b.transpose = l*(b*r).transpose := by
  rw [Matrix.transpose_mul, Matrix.mul_assoc]

theorem factored_cotangent_weight {M K N Q : Type*} [Fintype M] [Fintype Q]
    (a : Matrix M K ℝ) (l : Matrix M Q ℝ) (r : Matrix N Q ℝ) :
    a.transpose*(l*r.transpose) = (a.transpose*l)*r.transpose := by
  rw [Matrix.mul_assoc]

/- For Y=(A B) C, the joint schedule shares P=B C and T=A^T G. -/
theorem linear_chain_forward {M K N P : Type*} [Fintype K] [Fintype N]
    (a : Matrix M K ℝ) (b : Matrix K N ℝ) (c : Matrix N P ℝ) :
    (a*b)*c = a*(b*c) := Matrix.mul_assoc a b c

theorem linear_chain_input_pullback {M K N P : Type*} [Fintype N] [Fintype P]
    (b : Matrix K N ℝ) (c : Matrix N P ℝ) (g : Matrix M P ℝ) :
    (g*c.transpose)*b.transpose = g*(b*c).transpose := by
  rw [Matrix.transpose_mul, Matrix.mul_assoc]

theorem linear_chain_first_weight_pullback {M K N P : Type*}
    [Fintype M] [Fintype P]
    (a : Matrix M K ℝ) (c : Matrix N P ℝ) (g : Matrix M P ℝ) :
    a.transpose*(g*c.transpose) = (a.transpose*g)*c.transpose := by
  rw [Matrix.mul_assoc]

theorem linear_chain_second_weight_pullback {M K N P : Type*}
    [Fintype M] [Fintype K]
    (a : Matrix M K ℝ) (b : Matrix K N ℝ) (g : Matrix M P ℝ) :
    (a*b).transpose*g = b.transpose*(a.transpose*g) := by
  rw [Matrix.transpose_mul, Matrix.mul_assoc]

/- Full input + both weight gradients; conventional scalar operation counts.
Bias, copies, dispatch, memory traffic, precision and hardware units excluded. -/
theorem linear_chain_native_joint_cost (m k n p : ℝ) :
    (2*m*k*n-m*n)+(2*m*n*p-m*p)+(2*m*p*n-m*n)+
      (2*m*n*k-m*k)+(2*n*m*k-n*k)+(2*p*m*n-p*n) =
      6*m*k*n+6*m*n*p-(2*m*n+m*p+m*k+n*k+p*n) := by ring

theorem linear_chain_reassociated_joint_cost (m k n p : ℝ) :
    (2*p*n*k-p*k)+(2*m*k*p-m*p)+(2*p*m*k-p*k)+
      (2*m*p*k-m*k)+(2*n*p*k-n*k)+(2*p*k*n-p*n) =
      6*p*k*(m+n)-(2*p*k+m*p+m*k+n*k+p*n) := by ring

theorem linear_chain_cost_difference (m k n p : ℝ) :
    (6*m*k*n+6*m*n*p-(2*m*n+m*p+m*k+n*k+p*n))-
      (6*p*k*(m+n)-(2*p*k+m*p+m*k+n*k+p*n)) =
      6*(m*k*n+m*n*p-p*k*n-m*p*k)-2*m*n+2*p*k := by ring

theorem factored_cotangent_leading_cost (m k n r : ℝ) :
    2*k*n*r+2*m*r*k+2*k*m*r+2*k*r*n = 4*k*r*(m+n) := by ring

/- Dense outputs still impose storage even with smaller arithmetic. -/
theorem linear_chain_saved_residual_fraction (m n p k : ℝ) (h : 0 < m*n) :
    (m*n-p*k)/(m*n) = 1-(p*k)/(m*n) := by
  rw [sub_div, div_self (ne_of_gt h)]

/- A physically meaningful conditional roofline lower bound. Capacity/rate
and transfer estimates must be supplied by a target; this is not a timing fit. -/
theorem execution_roofline_lower_bound (time flops bytes peak bandwidth : ℝ)
    (hp : 0 < peak) (hb : 0 < bandwidth)
    (hcompute : flops ≤ time*peak) (hio : bytes ≤ time*bandwidth) :
    max (flops/peak) (bytes/bandwidth) ≤ time := by
  apply max_le
  · exact (div_le_iff₀ hp).2 hcompute
  · exact (div_le_iff₀ hb).2 hio

/- More than half of leading full-training arithmetic is removed under these
shape assumptions. This theorem says nothing about latency or rounded bits. -/
theorem linear_chain_leading_more_than_half (m k n p : ℝ)
    (hm : 0 < m) (hk : 0 < k) (hn : 0 < n) (hp : 0 < p)
    (hpm : 4*p ≤ m) (hpn : 4*p ≤ n) :
    2*(6*p*k*(m+n)) < 6*m*k*n+6*m*n*p := by
  have h1 := mul_le_mul_of_nonneg_right hpm (le_of_lt (mul_pos hn hk))
  have h2 := mul_le_mul_of_nonneg_right hpn (le_of_lt (mul_pos hm hk))
  have h3 := mul_pos (mul_pos hm hn) hp
  nlinarith

/- Replicated row biases are a special case of these matrix offsets. -/
theorem affine_chain_forward {M K N P : Type*} [Fintype K] [Fintype N]
    (a : Matrix M K ℝ) (b : Matrix K N ℝ) (c : Matrix N P ℝ)
    (h : Matrix M N ℝ) (z : Matrix M P ℝ) :
    (a*b+h)*c+z = a*(b*c)+h*c+z := by
  rw [Matrix.add_mul, Matrix.mul_assoc]

theorem affine_chain_second_weight_pullback {M K N P : Type*}
    [Fintype M] [Fintype K]
    (a : Matrix M K ℝ) (b : Matrix K N ℝ) (h : Matrix M N ℝ)
    (g : Matrix M P ℝ) :
    (a*b+h).transpose*g = b.transpose*(a.transpose*g)+h.transpose*g := by
  rw [Matrix.transpose_add, Matrix.transpose_mul, Matrix.add_mul, Matrix.mul_assoc]

end OjasXMath
