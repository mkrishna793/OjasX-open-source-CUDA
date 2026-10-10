import OjasXMath.SpectralFoundation
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Chebyshev.RootsExtrema
namespace OjasXMath
theorem spectral_chebyshev_explicit_9 (x : ℝ) : (Polynomial.Chebyshev.T ℝ 9).eval x = 256*x^9 - 576*x^7 + 432*x^5 - 120*x^3 + 9*x := by
  have h2 := Polynomial.Chebyshev.T_add_two ℝ (0:ℤ)
  have h3 := Polynomial.Chebyshev.T_add_two ℝ (1:ℤ)
  have h4 := Polynomial.Chebyshev.T_add_two ℝ (2:ℤ)
  have h5 := Polynomial.Chebyshev.T_add_two ℝ (3:ℤ)
  have h6 := Polynomial.Chebyshev.T_add_two ℝ (4:ℤ)
  have h7 := Polynomial.Chebyshev.T_add_two ℝ (5:ℤ)
  have h8 := Polynomial.Chebyshev.T_add_two ℝ (6:ℤ)
  have h9 := Polynomial.Chebyshev.T_add_two ℝ (7:ℤ)
  norm_num at h2 h3 h4 h5 h6 h7 h8 h9
  rw [h9, h8, h7, h6, h5, h4, h3, h2]
  norm_num [Polynomial.eval_sub, Polynomial.eval_mul, Polynomial.eval_X, Polynomial.eval_ofNat] <;> ring
theorem spectral_chebyshev_explicit_11 (x : ℝ) : (Polynomial.Chebyshev.T ℝ 11).eval x = 1024*x^11 - 2816*x^9 + 2816*x^7 - 1232*x^5 + 220*x^3 - 11*x := by
  have h2 := Polynomial.Chebyshev.T_add_two ℝ (0:ℤ)
  have h3 := Polynomial.Chebyshev.T_add_two ℝ (1:ℤ)
  have h4 := Polynomial.Chebyshev.T_add_two ℝ (2:ℤ)
  have h5 := Polynomial.Chebyshev.T_add_two ℝ (3:ℤ)
  have h6 := Polynomial.Chebyshev.T_add_two ℝ (4:ℤ)
  have h7 := Polynomial.Chebyshev.T_add_two ℝ (5:ℤ)
  have h8 := Polynomial.Chebyshev.T_add_two ℝ (6:ℤ)
  have h9 := Polynomial.Chebyshev.T_add_two ℝ (7:ℤ)
  have h10 := Polynomial.Chebyshev.T_add_two ℝ (8:ℤ)
  have h11 := Polynomial.Chebyshev.T_add_two ℝ (9:ℤ)
  norm_num at h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  rw [h11, h10, h9, h8, h7, h6, h5, h4, h3, h2]
  norm_num [Polynomial.eval_sub, Polynomial.eval_mul, Polynomial.eval_X, Polynomial.eval_ofNat] <;> ring
theorem spectral_two_chebyshev_bound (x a b : ℝ) (hx : |x| ≤ 1) :
  |a*(Polynomial.Chebyshev.T ℝ 9).eval x+b*(Polynomial.Chebyshev.T ℝ 11).eval x| ≤ |a|+|b| := by
  have h9 := Polynomial.Chebyshev.abs_eval_T_real_le_one 9 hx
  have h11 := Polynomial.Chebyshev.abs_eval_T_real_le_one 11 hx
  calc
    _ ≤ |a*(Polynomial.Chebyshev.T ℝ 9).eval x|+|b*(Polynomial.Chebyshev.T ℝ 11).eval x| := abs_add_le _ _
    _ ≤ |a|+|b| := by
      simp only [abs_mul]
      have ha := mul_le_mul_of_nonneg_left h9 (abs_nonneg a)
      have hb := mul_le_mul_of_nonneg_left h11 (abs_nonneg b)
      nlinarith
theorem spectral_chebyshev_gelu_tail (x prefactor : ℝ) (hx : |x| ≤ 1) :
  |-prefactor*x^11/3520 + 5*prefactor*x^9/1728 - 17*prefactor*x^7/3072 + prefactor*x^5/256 - 155*prefactor*x^3/147456 + 19*prefactor*x/245760| ≤ 83*|prefactor|/9732096 := by
  calc
    _ = |(73*prefactor/8847360)*(Polynomial.Chebyshev.T ℝ 9).eval x+(-prefactor/3604480)*(Polynomial.Chebyshev.T ℝ 11).eval x| := by
      rw [spectral_chebyshev_explicit_9, spectral_chebyshev_explicit_11]
      congr 1
      ring
    _ ≤ |(73*prefactor/8847360)|+|(-prefactor/3604480)| := spectral_two_chebyshev_bound _ _ _ hx
    _ = _ := by simp only [abs_mul, abs_div, abs_neg]; norm_num <;> ring
theorem spectral_chebyshev_gelu_primal_actual (x prefactor : ℝ) : HasDerivAt (fun t : ℝ => -131*prefactor*t^8/57344 + 187*prefactor*t^6/7680 - 98149*prefactor*t^4/589824 + 491501*prefactor*t^2/491520 + t/2) (-131*prefactor*x^7/7168 + 187*prefactor*x^5/1280 - 98149*prefactor*x^3/147456 + 491501*prefactor*x/245760 + 1/2) x := by
  convert ((((((hasDerivAt_const x (1/2)).mul (hasDerivAt_id x)).add (((hasDerivAt_const x (-98149/589824)).mul (hasDerivAt_const x (prefactor))).mul ((hasDerivAt_id x).pow 4))).add (((hasDerivAt_const x (-131/57344)).mul (hasDerivAt_const x (prefactor))).mul ((hasDerivAt_id x).pow 8))).add (((hasDerivAt_const x (187/7680)).mul (hasDerivAt_const x (prefactor))).mul ((hasDerivAt_id x).pow 6))).add (((hasDerivAt_const x (491501/491520)).mul (hasDerivAt_const x (prefactor))).mul ((hasDerivAt_id x).pow 2))) using 1 <;> (try funext t) <;> (try dsimp) <;> ring
theorem spectral_chebyshev_silu_tail (x prefactor : ℝ) (hx : |x| ≤ 1) :
  |-691*x^11/26611200 + 31*x^9/145152 - 3029*x^7/7741440 + 2099*x^5/7741440 - 1789*x^3/24772608 + 817*x/154828800| ≤ 4747/8174960640 := by
  calc
    _ = |(4127/7431782400)*(Polynomial.Chebyshev.T ℝ 9).eval x+(-691/27249868800)*(Polynomial.Chebyshev.T ℝ 11).eval x| := by
      rw [spectral_chebyshev_explicit_9, spectral_chebyshev_explicit_11]
      congr 1
      ring
    _ ≤ |(4127/7431782400)|+|(-691/27249868800)| := spectral_two_chebyshev_bound _ _ _ hx
    _ = _ := by simp only [abs_mul, abs_div, abs_neg]; norm_num <;> ring
theorem spectral_chebyshev_silu_primal_actual (x prefactor : ℝ) : HasDerivAt (fun t : ℝ => -10027*t^8/61931520 + 94669*t^6/46448640 - 2062595*t^4/99090432 + 77413583*t^2/309657600 + t/2) (-10027*x^7/7741440 + 94669*x^5/7741440 - 2062595*x^3/24772608 + 77413583*x/154828800 + 1/2) x := by
  convert ((((((hasDerivAt_const x (1/2)).mul (hasDerivAt_id x)).add ((hasDerivAt_const x (-2062595/99090432)).mul ((hasDerivAt_id x).pow 4))).add ((hasDerivAt_const x (-10027/61931520)).mul ((hasDerivAt_id x).pow 8))).add ((hasDerivAt_const x (94669/46448640)).mul ((hasDerivAt_id x).pow 6))).add ((hasDerivAt_const x (77413583/309657600)).mul ((hasDerivAt_id x).pow 2))) using 1 <;> (try funext t) <;> (try dsimp) <;> ring
end OjasXMath
