import OjasXMath.StreamedDagger

/-! Joint pullbacks for any finite number of operands, with different finite
coordinate sets. Excluded products use no division, allowing zero operands. -/
namespace OjasXMath

noncomputable def otherFactors {I R : Type*} [Fintype I] [CommRing R]
    (e : I → R) (i : I) : R := by
  classical
  exact ∏ j ∈ Finset.univ.erase i, e j

noncomputable def productTangent {I R : Type*} [Fintype I] [CommRing R]
    (e de : I → R) : R := ∑ i, de i * otherFactors e i

noncomputable def productPullback {I R : Type*} [Fintype I] [CommRing R]
    (e : I → R) (g : R) : I → R := fun i => g * otherFactors e i

theorem joint_product_dagger {I R : Type*} [Fintype I] [CommRing R]
    (e de : I → R) (g : R) :
    productTangent e de * g = scalarPair de (productPullback e g) := by
  simp only [productTangent, productPullback, scalarPair, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro i hi
  ring

theorem finite_product_direction_derivative {I : Type*} [Fintype I]
    (e de : I → ℝ) :
    HasDerivAt (fun t : ℝ => ∏ i, (e i + t*de i)) (productTangent e de) 0 := by
  classical
  have h := HasDerivAt.fun_finsetProd (u := Finset.univ)
    (fun i _ => (hasDerivAt_const (0:ℝ) (e i)).add
      ((hasDerivAt_id (0:ℝ)).mul_const (de i)))
  simpa [productTangent, otherFactors, smul_eq_mul, mul_comm] using h


theorem finite_product_frechet_derivative {I : Type*} [Fintype I] (e : I → ℝ) :
    HasFDerivAt (fun x : I → ℝ => ∏ i, x i)
      (∑ i, otherFactors e i •
        (ContinuousLinearMap.proj i : (I → ℝ) →L[ℝ] ℝ)) e := by
  classical
  simpa only [otherFactors] using
    (hasFDerivAt_finsetProd (𝕜 := ℝ) (u := Finset.univ) (x := e))

noncomputable def multilinearValue {I C Q : Type*} {A : I → Type*}
    [Fintype I] [Fintype Q] [∀ i, Fintype (A i)]
    (u : (i : I) → Q → A i → ℝ) (w : C → Q → ℝ)
    (a : (i : I) → A i → ℝ) : C → ℝ :=
  linearApply w (fun q => ∏ i, linearApply (u i) (a i) q)

noncomputable def multilinearTangent {I C Q : Type*} {A : I → Type*}
    [Fintype I] [Fintype Q] [∀ i, Fintype (A i)]
    (u : (i : I) → Q → A i → ℝ) (w : C → Q → ℝ)
    (a da : (i : I) → A i → ℝ) : C → ℝ :=
  linearApply w (fun q => productTangent
    (fun i => linearApply (u i) (a i) q) (fun i => linearApply (u i) (da i) q))

noncomputable def multilinearPullback {I C Q : Type*} {A : I → Type*}
    [Fintype I] [Fintype C] [Fintype Q] [∀ i, Fintype (A i)]
    (u : (i : I) → Q → A i → ℝ) (w : C → Q → ℝ)
    (a : (i : I) → A i → ℝ) (g : C → ℝ) : (i : I) → A i → ℝ :=
  fun i => linearAdjoint (u i) (fun q => linearAdjoint w g q *
    otherFactors (fun j => linearApply (u j) (a j) q) i)

theorem multilinear_direction_derivative {I C Q : Type*} {A : I → Type*}
    [Fintype I] [Fintype Q] [∀ i, Fintype (A i)]
    (u : (i : I) → Q → A i → ℝ) (w : C → Q → ℝ)
    (a da : (i : I) → A i → ℝ) (c : C) :
    HasDerivAt (fun t : ℝ => multilinearValue u w
      (fun i j => a i j + t*da i j) c) (multilinearTangent u w a da c) 0 := by
  have h := HasDerivAt.fun_sum (u := Finset.univ) (fun q _ =>
    (finite_product_direction_derivative
      (fun i => linearApply (u i) (a i) q)
      (fun i => linearApply (u i) (da i) q)).const_mul (w c q))
  unfold multilinearValue multilinearTangent
  simp_rw [linear_apply_affine]
  exact h

theorem joint_multilinear_dagger {I C Q : Type*} {A : I → Type*}
    [Fintype I] [Fintype C] [Fintype Q] [∀ i, Fintype (A i)]
    (u : (i : I) → Q → A i → ℝ) (w : C → Q → ℝ)
    (a da : (i : I) → A i → ℝ) (g : C → ℝ) :
    scalarPair (multilinearTangent u w a da) g =
      ∑ i, scalarPair (da i) (multilinearPullback u w a g i) := by
  unfold multilinearTangent multilinearPullback
  rw [linear_circuit_dagger]
  simp only [scalarPair, productTangent, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i hi
  change (∑ q, linearApply (u i) (da i) q *
    otherFactors (fun j => linearApply (u j) (a j) q) i * linearAdjoint w g q) =
      scalarPair (da i) (linearAdjoint (u i) (fun q => linearAdjoint w g q *
        otherFactors (fun j => linearApply (u j) (a j) q) i))
  rw [← linear_circuit_dagger]
  simp only [scalarPair]
  apply Finset.sum_congr rfl
  intro q hq
  ring

theorem joint_multilinear_gradient_direction {I C Q : Type*} {A : I → Type*}
    [Fintype I] [Fintype C] [Fintype Q] [∀ i, Fintype (A i)]
    (u : (i : I) → Q → A i → ℝ) (w : C → Q → ℝ)
    (a da : (i : I) → A i → ℝ) (g : C → ℝ) :
    HasDerivAt (fun t : ℝ => scalarPair
      (multilinearValue u w (fun i j => a i j+t*da i j)) g)
      (∑ i, scalarPair (da i) (multilinearPullback u w a g i)) 0 := by
  have h := HasDerivAt.fun_sum (u := Finset.univ) (fun c _ =>
    (multilinear_direction_derivative u w a da c).mul_const (g c))
  change HasDerivAt _ (scalarPair (multilinearTangent u w a da) g) 0 at h
  rwa [joint_multilinear_dagger] at h

theorem prefix_suffix_exclusion {R : Type*} [CommMonoid R]
    (before after : List R) :
    (before ++ after).prod = before.prod * after.prod := List.prod_append

/- Multiplication-only scalar model for r >= 2: r-2 prefix, r-2 suffix,
r-2 interior and r upstream products; no encoder/decoder or memory cost. -/
theorem joint_prefix_suffix_cost (r : ℝ) :
    (r-2)+(r-2)+(r-2)+r = 4*r-6 := by ring

theorem joint_prefix_suffix_saving (r : ℝ) :
    r*(r-1)-(4*r-6) = (r-2)*(r-3) := by ring

theorem joint_prefix_suffix_strict_saving (r : ℝ) (hr : 3 < r) :
    4*r-6 < r*(r-1) := by
  have h := mul_pos (show 0 < r-2 by linarith) (show 0 < r-3 by linarith)
  nlinarith

/- Rolling cotangent reuses the product of all factors to the right.
This is standard reverse accumulation; reconstructing prefixes has a cost. -/
theorem rolling_cotangent_pairing {R : Type*} [CommRing R]
    (before after : List R) (g : R) :
    g*(before ++ after).prod = before.prod*(g*after.prod) := by
  rw [List.prod_append]
  ring

theorem rolling_cotangent_cost (r : ℝ) :
    (r-2)+2*(r-1) = 3*r-4 := by ring

theorem rolling_cotangent_saving (r : ℝ) :
    r*(r-1)-(3*r-4) = (r-2)^2 := by ring

theorem rolling_reconstruction_overhead (r : ℝ) :
    (3*r-4)-(2*r-2) = r-2 := by ring

theorem rolling_cotangent_strict_saving (r : ℝ) (hr : 2 < r) :
    3*r-4 < r*(r-1) := by
  have h := sq_pos_of_pos (show 0 < r-2 by linarith)
  nlinarith

end OjasXMath
