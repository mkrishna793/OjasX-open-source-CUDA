import OjasXMath.ReversibleOptic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Inv

/-! Cached branch sufficient statistics remove projection replay. These are
exact-real identities and conditional cost statements, not IEEE or speed proofs. -/
namespace OjasXMath

theorem cached_coupling_reconstruction {A B R T : Type*}
    [AddCommGroup A] [AddCommGroup B]
    (ef : B → R) (eg : A → T) (vf : R → A) (vg : T → B) (x : A × B) :
    let y := couplingForward (vf ∘ ef) (vg ∘ eg) x
    (y.1-vf (ef x.2), y.2-vg (eg y.1)) = x := by
  simp [couplingForward]

noncomputable def cacheSigmoid (x : ℝ) : ℝ := (1+Real.exp (-x))⁻¹

theorem cache_sigmoid_hasDerivAt (x : ℝ) :
    HasDerivAt cacheSigmoid (cacheSigmoid x*(1-cacheSigmoid x)) x := by
  have hne : (1:ℝ)+Real.exp (-x) ≠ 0 := ne_of_gt (by positivity)
  have he := (Real.hasDerivAt_exp (-x)).comp x (hasDerivAt_id x).neg
  have hi := ((hasDerivAt_const x (1:ℝ)).add he).inv hne
  convert hi using 1 <;> first | rfl | (simp [cacheSigmoid]; field_simp [hne]; ring)

theorem cache_silu_hasDerivAt (x : ℝ) :
    HasDerivAt (fun t : ℝ => t*cacheSigmoid t)
      (cacheSigmoid x*(1+x*(1-cacheSigmoid x))) x := by
  convert (hasDerivAt_id x).mul (cache_sigmoid_hasDerivAt x) using 1 <;>
    first | rfl | (simp; ring)

theorem cached_branch_direction_derivative (σ : ℝ → ℝ) (a b x da db dx slope : ℝ)
    (hσ : HasDerivAt σ slope (a*x+b)) :
    HasDerivAt (fun t : ℝ => σ ((a+t*da)*(x+t*dx)+(b+t*db)))
      (slope*(a*dx+da*x+db)) 0 := by
  have ha := (hasDerivAt_const (0:ℝ) a).add ((hasDerivAt_id (0:ℝ)).mul_const da)
  have hx := (hasDerivAt_const (0:ℝ) x).add ((hasDerivAt_id (0:ℝ)).mul_const dx)
  have hb := (hasDerivAt_const (0:ℝ) b).add ((hasDerivAt_id (0:ℝ)).mul_const db)
  have hz := (ha.mul hx).add hb
  have hs : HasDerivAt σ slope ((a+(0:ℝ)*da)*(x+(0:ℝ)*dx)+(b+(0:ℝ)*db)) := by
    simpa using hσ
  have hc := hs.comp (0:ℝ) hz
  convert hc using 1
  · rfl
  · simp only [Pi.add_apply, id_eq, zero_add, one_mul, zero_mul, add_zero]
    ring

theorem cached_nonlinear_coupling_joint_dagger
    (a c x2 y1 dx1 dx2 da db dc dd u v scale df dg : ℝ) :
    let dy1 := dx1+scale*df*(a*dx2+da*x2+db)
    let dy2 := dx2+scale*dg*(c*dy1+dc*y1+dd)
    let δg := scale*dg*v
    let adj1 := u+c*δg
    let δf := scale*df*adj1
    dy1*u+dy2*v = dx1*adj1+dx2*(v+a*δf)+
      da*(x2*δf)+db*δf+dc*(y1*δg)+dd*δg := by
  dsimp
  ring

theorem cached_silu_joint_value_vjp (z s incoming scale state : ℝ) :
    (state-scale*(z*s), (scale*incoming)*(s*(1+z*(1-s)))) =
    (state-scale*z*s, incoming*scale*(s+z*s*(1-s))) := by
  congr 1 <;> ring

theorem cached_finite_affine_pointwise_dagger {I J : Type*} [Fintype I] [Fintype J]
    (w dw : J → I → ℝ) (x dx : I → ℝ) (db slope g : J → ℝ) (scale : ℝ) :
    scalarPair (fun j => scale*slope j*(linearApply w dx j+linearApply dw x j+db j)) g =
      scalarPair dx (linearAdjoint w (fun j => scale*g j*slope j))+
      scalarPair x (linearAdjoint dw (fun j => scale*g j*slope j))+
      scalarPair db (fun j => scale*g j*slope j) := by
  have hp : scalarPair (fun j => scale*slope j*(linearApply w dx j+linearApply dw x j+db j)) g =
      scalarPair (linearApply w dx) (fun j => scale*g j*slope j)+
      scalarPair (linearApply dw x) (fun j => scale*g j*slope j)+
      scalarPair db (fun j => scale*g j*slope j) := by
    simp only [scalarPair, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [hp, linear_circuit_dagger, linear_circuit_dagger]

theorem cached_finite_weight_cotangent_pairing {I J : Type*} [Fintype I] [Fintype J]
    (dw : J → I → ℝ) (x : I → ℝ) (δ : J → ℝ) :
    scalarPair (linearApply dw x) δ = ∑ j, ∑ i, dw j i*(δ j*x i) := by
  simp only [scalarPair, linearApply, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro j _
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem cached_projection_work_saving (n c : ℕ) (hc : c ≤ n) :
    (8*n-2*c)+2*c = 8*n := by omega

theorem fully_cached_joint_gemm_count (n : ℕ) : 8*n-2*n = 6*n := by omega

theorem cache_budget_monotone_work (n c small : ℕ) (h : small ≤ c) :
    8*n-2*c ≤ 8*n-2*small := by omega

theorem full_cache_retained_storage_saving (n m d : ℕ)
    (hn : 2 ≤ n) (hm : 0 < m) (hd : 0 < d) :
    (2+2*n)*m*d < (4*n-1)*m*d := by
  have hc : 2+2*n < 4*n-1 := by omega
  exact Nat.mul_lt_mul_of_pos_right (Nat.mul_lt_mul_of_pos_right hc hm) hd

theorem cached_reconstruction_error_additive (x value forwardError valueError subtractError : ℝ) :
    |((x+value+forwardError)-(value+valueError)+subtractError)-x| ≤
      |forwardError|+|valueError|+|subtractError| := by
  have he : ((x+value+forwardError)-(value+valueError)+subtractError)-x =
      forwardError-valueError+subtractError := by ring
  rw [he]
  have ha := abs_add_le (forwardError-valueError) subtractError
  have hs := abs_sub forwardError valueError
  linarith

end OjasXMath
