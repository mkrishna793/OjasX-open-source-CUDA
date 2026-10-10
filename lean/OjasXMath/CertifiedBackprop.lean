import OjasXMath.Multilinear
import Mathlib.Analysis.Calculus.FDeriv.Comp
import Mathlib.Analysis.Calculus.FDeriv.Linear
import Mathlib.Analysis.Calculus.FDeriv.Prod

/-! Primitive contracts include an actual Frechet derivative and its adjoint
pairing. Composition discharges both obligations; arbitrary pullback functions
are not accepted as correct derivatives merely because they compose. -/
namespace OjasXMath

structure CertifiedStage (I J S : Type*) [Fintype I] [Fintype J] where
  forward : (I → ℝ) → S × (J → ℝ)
  differential : (I → ℝ) → ((I → ℝ) →L[ℝ] (J → ℝ))
  derivative_correct : ∀ x, HasFDerivAt (fun z => (forward z).2) (differential x) x
  pullback : S → (J → ℝ) → (I → ℝ)
  dagger_correct : ∀ x dx g, scalarPair (differential x dx) g =
    scalarPair dx (pullback (forward x).1 g)


noncomputable def certifiedLinearStage {I J : Type*} [Fintype I] [Fintype J]
    (l : (I → ℝ) →L[ℝ] (J → ℝ)) (p : (J → ℝ) → (I → ℝ))
    (h : ∀ dx g, scalarPair (l dx) g = scalarPair dx (p g)) :
    CertifiedStage I J Unit where
  forward x := ((),l x)
  differential _ := l
  derivative_correct _x := l.hasFDerivAt
  pullback _ := p
  dagger_correct _ dx g := h dx g

theorem certified_linear_derivative {I J : Type*} [Fintype I] [Fintype J]
    (l : (I → ℝ) →L[ℝ] (J → ℝ)) (p : (J → ℝ) → (I → ℝ))
    (h : ∀ dx g, scalarPair (l dx) g = scalarPair dx (p g)) (x : I → ℝ) :
    HasFDerivAt (fun z => ((certifiedLinearStage l p h).forward z).2) l x :=
  (certifiedLinearStage l p h).derivative_correct x


/- A certified family of scalar activations retains the primal input. Each
activation must supply its scalar derivative theorem; no inverse is assumed. -/
noncomputable def certifiedElementwiseStage {I : Type*} [Fintype I]
    (phi dphi : I → ℝ → ℝ)
    (hphi : ∀ i z, HasDerivAt (phi i) (dphi i z) z) :
    CertifiedStage I I (I → ℝ) where
  forward x := (x,fun i => phi i (x i))
  differential x := ContinuousLinearMap.pi (fun i =>
    dphi i (x i) • (ContinuousLinearMap.proj i : (I → ℝ) →L[ℝ] ℝ))
  derivative_correct x := hasFDerivAt_pi.mpr (fun i =>
    (hphi i (x i)).comp_hasFDerivAt x (hasFDerivAt_apply i x))
  pullback x g := fun i => g i*dphi i (x i)
  dagger_correct x dx g := by
    change (∑ i, (dphi i (x i)*dx i)*g i) = ∑ i, dx i*(g i*dphi i (x i))
    apply Finset.sum_congr rfl
    intro i hi
    ring

theorem certified_elementwise_derivative {I : Type*} [Fintype I]
    (phi dphi : I → ℝ → ℝ)
    (hphi : ∀ i z, HasDerivAt (phi i) (dphi i z) z) (x : I → ℝ) :
    HasFDerivAt (fun z => ((certifiedElementwiseStage phi dphi hphi).forward z).2)
      ((certifiedElementwiseStage phi dphi hphi).differential x) x :=
  (certifiedElementwiseStage phi dphi hphi).derivative_correct x

noncomputable def certifiedSquareStage (I : Type*) [Fintype I] :
    CertifiedStage I I (I → ℝ) :=
  certifiedElementwiseStage (fun _ z => z*z) (fun _ z => 2*z)
    (fun _ z => square_has_derivative z)

theorem certified_square_pullback {I : Type*} [Fintype I] (x g : I → ℝ) :
    (certifiedSquareStage I).pullback ((certifiedSquareStage I).forward x).1 g =
      fun i => g i*(2*x i) := rfl


noncomputable def certifiedProductStage (I : Type*) [Fintype I] :
    CertifiedStage I Unit (I → ℝ) where
  forward x := (x,fun _ => ∏ i, x i)
  differential x := ContinuousLinearMap.pi (fun _ : Unit =>
    ∑ i, otherFactors x i • (ContinuousLinearMap.proj i : (I → ℝ) →L[ℝ] ℝ))
  derivative_correct x := hasFDerivAt_pi.mpr
    (fun _ => finite_product_frechet_derivative x)
  pullback x g := productPullback x (g ())
  dagger_correct x dx g := by
    simpa [scalarPair, productTangent, productPullback,
      sum_apply, smul_apply, smul_eq_mul,
      mul_comm] using joint_product_dagger x dx (g ())

theorem certified_product_pullback {I : Type*} [Fintype I] (x : I → ℝ) (g : Unit → ℝ) :
    (certifiedProductStage I).pullback ((certifiedProductStage I).forward x).1 g =
      productPullback x (g ()) := rfl

noncomputable def CertifiedStage.then {I J K S T : Type*}
    [Fintype I] [Fintype J] [Fintype K]
    (f : CertifiedStage I J S) (h : CertifiedStage J K T) :
    CertifiedStage I K (S × T) where
  forward x :=
    let a := f.forward x
    let b := h.forward a.2
    ((a.1,b.1),b.2)
  differential x := (h.differential (f.forward x).2).comp (f.differential x)
  derivative_correct x := by
    have hh : HasFDerivAt (fun z : J → ℝ => (h.forward z).2)
      (h.differential (f.forward x).2) (f.forward x).2 :=
      h.derivative_correct (f.forward x).2
    exact hh.comp x (f.derivative_correct x)
  pullback s g := f.pullback s.1 (h.pullback s.2 g)
  dagger_correct x dx g := by
    change scalarPair (h.differential (f.forward x).2 (f.differential x dx)) g = _
    rw [h.dagger_correct, f.dagger_correct]

/- The exact output of a certified composition has a genuine derivative. -/
theorem certified_chain_derivative {I J K S T : Type*}
    [Fintype I] [Fintype J] [Fintype K]
    (f : CertifiedStage I J S) (h : CertifiedStage J K T) (x : I → ℝ) :
    HasFDerivAt (fun z => ((f.then h).forward z).2)
      ((h.differential (f.forward x).2).comp (f.differential x)) x :=
  (f.then h).derivative_correct x

theorem certified_chain_dagger {I J K S T : Type*}
    [Fintype I] [Fintype J] [Fintype K]
    (f : CertifiedStage I J S) (h : CertifiedStage J K T)
    (x dx : I → ℝ) (g : K → ℝ) :
    scalarPair ((f.then h).differential x dx) g =
      scalarPair dx ((f.then h).pullback ((f.then h).forward x).1 g) :=
  (f.then h).dagger_correct x dx g

theorem certified_chain_associative {I J K L S T U : Type*}
    [Fintype I] [Fintype J] [Fintype K] [Fintype L]
    (f : CertifiedStage I J S) (h : CertifiedStage J K T)
    (k : CertifiedStage K L U) (s : S) (t : T) (u : U) (g : L → ℝ) :
    ((f.then h).then k).pullback ((s,t),u) g =
      (f.then (h.then k)).pullback (s,(t,u)) g := rfl

theorem scalar_pair_separates {I : Type*} [Fintype I]
    (a b : I → ℝ) (h : ∀ dx, scalarPair dx a = scalarPair dx b) : a = b := by
  classical
  funext i
  have hi := h (fun j => if j = i then 1 else 0)
  simpa [scalarPair] using hi

theorem certified_pullback_unique {I J S : Type*} [Fintype I] [Fintype J]
    (f : CertifiedStage I J S) (x : I → ℝ) (g : J → ℝ) (candidate : I → ℝ)
    (h : ∀ dx, scalarPair (f.differential x dx) g = scalarPair dx candidate) :
    candidate = f.pullback (f.forward x).1 g := by
  apply scalar_pair_separates
  intro dx
  rw [← h, f.dagger_correct]

theorem certified_fanout_accumulates {I J S : Type*} [Fintype I] [Fintype J]
    (f : CertifiedStage I J S) (x : I → ℝ) (g h : J → ℝ) :
    f.pullback (f.forward x).1 (g+h) =
      f.pullback (f.forward x).1 g + f.pullback (f.forward x).1 h := by
  apply scalar_pair_separates
  intro dx
  rw [← f.dagger_correct]
  simp only [scalarPair, Pi.add_apply, mul_add, Finset.sum_add_distrib]
  exact congrArg₂ (·+·) (f.dagger_correct x dx g) (f.dagger_correct x dx h)

noncomputable def CertifiedStage.reencode {I J S T : Type*} [Fintype I] [Fintype J]
    (f : CertifiedStage I J S) (encode : S → T) (recover : T → (J → ℝ) → (I → ℝ))
    (h : ∀ x g, recover (encode (f.forward x).1) g = f.pullback (f.forward x).1 g) :
    CertifiedStage I J T where
  forward x := (encode (f.forward x).1,(f.forward x).2)
  differential := f.differential
  derivative_correct := f.derivative_correct
  pullback := recover
  dagger_correct x dx g := by
    rw [h, f.dagger_correct]

theorem certified_residual_reencoding {I J S T : Type*} [Fintype I] [Fintype J]
    (f : CertifiedStage I J S) (encode : S → T) (recover : T → (J → ℝ) → (I → ℝ))
    (h : ∀ x g, recover (encode (f.forward x).1) g = f.pullback (f.forward x).1 g)
    (x dx : I → ℝ) (g : J → ℝ) :
    scalarPair ((f.reencode encode recover h).differential x dx) g =
      scalarPair dx ((f.reencode encode recover h).pullback
        ((f.reencode encode recover h).forward x).1 g) :=
  (f.reencode encode recover h).dagger_correct x dx g

/- Numerical error propagation under an explicit local Lipschitz bound.
This theorem does not assert a bound for a particular float instruction. -/
theorem backward_error_composition {E F G : Type*}
    [NormedAddCommGroup E] [NormedAddCommGroup F] [NormedAddCommGroup G]
    (p phat : F → E) (q qhat : G → F) (g : G) (epsP epsQ lip : ℝ)
    (hlip : 0 ≤ lip)
    (hP : ‖phat (qhat g)-p (qhat g)‖ ≤ epsP)
    (hQ : ‖qhat g-q g‖ ≤ epsQ)
    (hbound : ‖p (qhat g)-p (q g)‖ ≤ lip*‖qhat g-q g‖) :
    ‖phat (qhat g)-p (q g)‖ ≤ epsP+lip*epsQ := by
  calc
    _ = ‖(phat (qhat g)-p (qhat g))+(p (qhat g)-p (q g))‖ := by
      congr 1
      abel
    _ ≤ ‖phat (qhat g)-p (qhat g)‖+‖p (qhat g)-p (q g)‖ := norm_add_le _ _
    _ ≤ epsP+lip*epsQ := add_le_add hP
      (hbound.trans (mul_le_mul_of_nonneg_left hQ hlip))

end OjasXMath
