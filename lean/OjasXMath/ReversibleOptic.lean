import OjasXMath.CertifiedBackprop

/-! Reconstructible optics and correctly typed cotangent relations. Algebraic
reconstruction is over reals/groups; no IEEE inverse or traffic claim follows. -/
namespace OjasXMath

def couplingForward {A B : Type*} [AddCommGroup A] [AddCommGroup B]
    (f : B → A) (g : A → B) (x : A × B) : A × B :=
  (x.1+f x.2, x.2+g (x.1+f x.2))

def couplingInverse {A B : Type*} [AddCommGroup A] [AddCommGroup B]
    (f : B → A) (g : A → B) (y : A × B) : A × B :=
  (y.1-f (y.2-g y.1), y.2-g y.1)

theorem coupling_inverse_forward {A B : Type*} [AddCommGroup A] [AddCommGroup B]
    (f : B → A) (g : A → B) (x : A × B) :
    couplingInverse f g (couplingForward f g x) = x := by
  simp [couplingInverse, couplingForward]

theorem coupling_forward_inverse {A B : Type*} [AddCommGroup A] [AddCommGroup B]
    (f : B → A) (g : A → B) (y : A × B) :
    couplingForward f g (couplingInverse f g y) = y := by
  simp [couplingInverse, couplingForward]

theorem coupling_branch_rewrite {A B : Type*} [AddCommGroup A] [AddCommGroup B]
    (f f' : B → A) (g g' : A → B)
    (hf : ∀ x, f x = f' x) (hg : ∀ y, g y = g' y) :
    couplingForward f g = couplingForward f' g' ∧
      couplingInverse f g = couplingInverse f' g' := by
  have hfe : f = f' := funext hf
  have hge : g = g' := funext hg
  subst f'
  subst g'
  exact ⟨rfl,rfl⟩

theorem reversible_stage_output_pullback {I J S : Type*} [Fintype I] [Fintype J]
    (stage : CertifiedStage I J S) (recover : (J → ℝ) → S)
    (h : ∀ x, recover (stage.forward x).2 = (stage.forward x).1)
    (x : I → ℝ) (g : J → ℝ) :
    stage.pullback (recover (stage.forward x).2) g =
      stage.pullback (stage.forward x).1 g := by rw [h]

theorem reconstructed_stage_adjoint {I J S : Type*} [Fintype I] [Fintype J]
    (stage : CertifiedStage I J S) (recover : (J → ℝ) → S)
    (h : ∀ x, recover (stage.forward x).2 = (stage.forward x).1)
    (x dx : I → ℝ) (g : J → ℝ) :
    scalarPair (stage.differential x dx) g =
      scalarPair dx (stage.pullback (recover (stage.forward x).2) g) := by
  rw [h]
  exact stage.dagger_correct x dx g

theorem coupling_scalar_joint_pullback (a c x2 y1 dx1 dx2 da db dc dd u v : ℝ) :
    let dy1 := dx1+a*dx2+da*x2+db
    let dy2 := dx2+c*dy1+dc*y1+dd
    dy1*u+dy2*v =
      dx1*(u+c*v)+dx2*(v+a*(u+c*v))+
      da*(x2*(u+c*v))+db*(u+c*v)+dc*(y1*v)+dd*v := by
  dsimp
  ring

theorem affine_coupling_direction_derivative (a b c d x1 x2 dx1 dx2 da db dc dd : ℝ) :
    HasDerivAt
      (fun t : ℝ => (x2+t*dx2)+(c+t*dc)*
        ((x1+t*dx1)+(a+t*da)*(x2+t*dx2)+(b+t*db))+(d+t*dd))
      (dx2+c*(dx1+a*dx2+da*x2+db)+dc*(x1+a*x2+b)+dd) 0 := by
  have ha := (hasDerivAt_const (0:ℝ) a).add ((hasDerivAt_id (0:ℝ)).mul_const da)
  have hc := (hasDerivAt_const (0:ℝ) c).add ((hasDerivAt_id (0:ℝ)).mul_const dc)
  have hx1 := (hasDerivAt_const (0:ℝ) x1).add ((hasDerivAt_id (0:ℝ)).mul_const dx1)
  have hx2 := (hasDerivAt_const (0:ℝ) x2).add ((hasDerivAt_id (0:ℝ)).mul_const dx2)
  have hb := (hasDerivAt_const (0:ℝ) b).add ((hasDerivAt_id (0:ℝ)).mul_const db)
  have hd := (hasDerivAt_const (0:ℝ) d).add ((hasDerivAt_id (0:ℝ)).mul_const dd)
  convert (hx2.add (hc.mul ((hx1.add (ha.mul hx2)).add hb))).add hd using 1 <;>
    first | rfl | (simp; ring)

def canonicalPair {I : Type*} [Fintype I]
    (q p q' p' : I → ℝ) : ℝ := scalarPair q p'-scalarPair q' p

theorem cotangent_relation_canonical_pair {I J : Type*} [Fintype I] [Fintype J]
    (w : J → I → ℝ) (q q' : I → ℝ) (p p' : J → ℝ) :
    canonicalPair q (linearAdjoint w p) q' (linearAdjoint w p') =
      canonicalPair (linearApply w q) p (linearApply w q') p' := by
  unfold canonicalPair
  rw [linear_circuit_dagger, linear_circuit_dagger]

theorem naive_dagger_bundle_scales_canonical_pair (q p q' p' : ℝ) :
    (2*q)*(2*p')-(2*q')*(2*p) = 4*(q*p'-q'*p) := by ring

theorem isometric_linear_reconstruction {I J B : Type*}
    [Fintype I] [Fintype J] [DecidableEq I]
    (w : Matrix J I ℝ) (x : Matrix I B ℝ)
    (h : w.transpose*w = 1) : w.transpose*(w*x) = x := by
  rw [← Matrix.mul_assoc, h, Matrix.one_mul]

theorem reconstruction_chain_error_bound (f : ℝ → ℝ) (x1 x2 eg ef gain : ℝ)
    (hf : |f x2-f (x2-eg)| ≤ gain*|eg|) :
    |(x1+f x2-f (x2-eg)-ef)-x1| ≤ gain*|eg|+|ef| := by
  have he : (x1+f x2-f (x2-eg)-ef)-x1 = (f x2-f (x2-eg))-ef := by ring
  rw [he]
  have ht := abs_sub (f x2-f (x2-eg)) ef
  linarith

end OjasXMath
