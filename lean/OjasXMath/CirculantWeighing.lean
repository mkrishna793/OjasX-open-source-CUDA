/-!
Circulant Weighing Matrix Orthogonality Certificates for OjasX3 (`ojasx3/lattice/transforms.py`).

Pure dependency-free Lean 4 formalization (zero `sorry`, zero Mathlib imports,
zero `native_decide` trust escapes):
  1. Universal `CW(2m, 4)` (`cw_weight4_support`) exact autocorrelation orthogonality
     for all `m ≥ 2` (`cw4_dot_diag`, `cw4_dot_off_diag`).
  2. Certified `CW(28, 16)` and `CW(56, 16)` (`cw_weight16_support`) kernel-decided
     circulant orthogonality (`W W^T = 16 I`).
  3. Certified `CW(120, 36)` (`CW120_36_POS`, `CW120_36_NEG`) support validity and
     kernel-decided circulant orthogonality (`W W^T = 36 I`, 21 `+1` and 15 `-1` taps).
-/
namespace OjasXMath

/-- Exact first row of `CW(2m, 4)` (`cw_weight4_support(2*m)`):
    `+1` on `{1, m, m + 1}`, `-1` on `{0}`, `0` elsewhere. -/
def cw4Entry (m idx : Nat) : Int :=
  if idx == 0 then -1
  else if idx == 1 || idx == m || idx == m + 1 then 1
  else 0

theorem cw4Entry_zero_idx (m : Nat) : cw4Entry m 0 = -1 := rfl

theorem cw4Entry_pos (m idx : Nat) (h0 : idx ≠ 0)
    (hp : idx = 1 ∨ idx = m ∨ idx = m + 1) : cw4Entry m idx = 1 := by
  simp only [cw4Entry, beq_iff_eq, Bool.or_eq_true, or_assoc]
  simp [h0, hp]

theorem cw4Entry_off (m idx : Nat) (h0 : idx ≠ 0) (h1 : idx ≠ 1)
    (hm : idx ≠ m) (hm1 : idx ≠ m + 1) : cw4Entry m idx = 0 := by
  simp only [cw4Entry, beq_iff_eq, Bool.or_eq_true, or_assoc]
  simp [h0, h1, hm, hm1]

/-- Sparse circulant autocorrelation `∑_{i=0}^{2m-1} a[i] * a[(i + s) mod 2m]`
    for `CW(2m, 4)`. -/
def cw4Dot (m s : Nat) : Int :=
  let n := 2 * m
  (-1) * cw4Entry m (s % n) +
  cw4Entry m ((1 + s) % n) +
  cw4Entry m ((m + s) % n) +
  cw4Entry m ((m + 1 + s) % n)

/-- Diagonal norm identity for `CW(2m, 4)`: `⟨row 0, row 0⟩ = 4` for all `m ≥ 2`. -/
theorem cw4_dot_diag (m : Nat) (hm : 2 ≤ m) (s : Nat) (hs : s % (2 * m) = 0) :
    cw4Dot m s = 4 := by
  have h1 : (1 + s) % (2 * m) = 1 := by
    rw [Nat.add_mod, hs, Nat.add_zero, Nat.mod_mod]
    exact Nat.mod_eq_of_lt (by omega)
  have h2 : (m + s) % (2 * m) = m := by
    rw [Nat.add_mod, hs, Nat.add_zero, Nat.mod_mod]
    exact Nat.mod_eq_of_lt (by omega)
  have h3 : (m + 1 + s) % (2 * m) = m + 1 := by
    rw [Nat.add_mod, hs, Nat.add_zero, Nat.mod_mod]
    exact Nat.mod_eq_of_lt (by omega)
  simp only [cw4Dot, hs, h1, h2, h3]
  rw [cw4Entry_zero_idx,
      cw4Entry_pos m 1 (by omega) (Or.inl rfl),
      cw4Entry_pos m m (by omega) (Or.inr (Or.inl rfl)),
      cw4Entry_pos m (m + 1) (by omega) (Or.inr (Or.inr rfl))]
  omega

/-- Off-diagonal orthogonality for `CW(2m, 4)`: `⟨row 0, row s⟩ = 0` for all `m ≥ 2` and `s ≢ 0 (mod 2m)`. -/
theorem cw4_dot_off_diag (m : Nat) (hm : 2 ≤ m) (s : Nat) (hs : s % (2 * m) ≠ 0) :
    cw4Dot m s = 0 := by
  have hs1 : (1 + s) % (2 * m) = (1 + s % (2 * m)) % (2 * m) := by
    rw [Nat.add_mod, Nat.mod_eq_of_lt (by omega : 1 < 2 * m)]
  have hsm : (m + s) % (2 * m) = (m + s % (2 * m)) % (2 * m) := by
    rw [Nat.add_mod, Nat.mod_eq_of_lt (by omega : m < 2 * m)]
  have hsm1 : (m + 1 + s) % (2 * m) = (m + 1 + s % (2 * m)) % (2 * m) := by
    rw [Nat.add_mod, Nat.mod_eq_of_lt (by omega : m + 1 < 2 * m)]
  simp only [cw4Dot, hs1, hsm, hsm1]
  generalize hr_def : s % (2 * m) = r at hs ⊢
  have hr_lt : r < 2 * m := hr_def ▸ Nat.mod_lt s (by omega)
  have hr_pos : 0 < r := Nat.pos_of_ne_zero hs
  by_cases hr1 : r = 1
  · rw [hr1]
    by_cases hm2 : m = 2
    · subst hm2; decide
    · have ha : (1 + 1) % (2 * m) = 2 := Nat.mod_eq_of_lt (by omega)
      have hb : (m + 1) % (2 * m) = m + 1 := Nat.mod_eq_of_lt (by omega)
      have hc : (m + 1 + 1) % (2 * m) = m + 2 := Nat.mod_eq_of_lt (by omega)
      rw [ha, hb, hc,
          cw4Entry_pos m 1 (by omega) (Or.inl rfl),
          cw4Entry_off m 2 (by omega) (by omega) (by omega) (by omega),
          cw4Entry_pos m (m + 1) (by omega) (Or.inr (Or.inr rfl)),
          cw4Entry_off m (m + 2) (by omega) (by omega) (by omega) (by omega)]
      omega
  · by_cases hrm1 : r = m - 1
    · rw [hrm1]
      have ha : (1 + (m - 1)) % (2 * m) = m := by
        rw [show 1 + (m - 1) = m by omega]; exact Nat.mod_eq_of_lt (by omega)
      have hb : (m + (m - 1)) % (2 * m) = 2 * m - 1 := by
        rw [show m + (m - 1) = 2 * m - 1 by omega]; exact Nat.mod_eq_of_lt (by omega)
      have hc : (m + 1 + (m - 1)) % (2 * m) = 0 := by
        rw [show m + 1 + (m - 1) = 2 * m by omega]; exact Nat.mod_self (2 * m)
      rw [ha, hb, hc,
          cw4Entry_off m (m - 1) (by omega) (by omega) (by omega) (by omega),
          cw4Entry_pos m m (by omega) (Or.inr (Or.inl rfl)),
          cw4Entry_off m (2 * m - 1) (by omega) (by omega) (by omega) (by omega),
          cw4Entry_zero_idx]
      omega
    · by_cases hrm : r = m
      · rw [hrm]
        have ha : (1 + m) % (2 * m) = m + 1 := by
          rw [show 1 + m = m + 1 by omega]; exact Nat.mod_eq_of_lt (by omega)
        have hb : (m + m) % (2 * m) = 0 := by
          rw [show m + m = 2 * m by omega]; exact Nat.mod_self (2 * m)
        have hc : (m + 1 + m) % (2 * m) = 1 := by
          rw [show m + 1 + m = 2 * m + 1 by omega, Nat.add_mod, Nat.mod_self, Nat.zero_add, Nat.mod_mod]
          exact Nat.mod_eq_of_lt (by omega)
        rw [ha, hb, hc,
            cw4Entry_pos m m (by omega) (Or.inr (Or.inl rfl)),
            cw4Entry_pos m (m + 1) (by omega) (Or.inr (Or.inr rfl)),
            cw4Entry_zero_idx,
            cw4Entry_pos m 1 (by omega) (Or.inl rfl)]
        omega
      · by_cases hrmp1 : r = m + 1
        · rw [hrmp1]
          by_cases hm2 : m = 2
          · subst hm2; decide
          · have ha : (1 + (m + 1)) % (2 * m) = m + 2 := by
              rw [show 1 + (m + 1) = m + 2 by omega]
              exact Nat.mod_eq_of_lt (by omega : m + 2 < 2 * m)
            have hb : (m + (m + 1)) % (2 * m) = 1 := by
              rw [show m + (m + 1) = 2 * m + 1 by omega, Nat.add_mod, Nat.mod_self, Nat.zero_add, Nat.mod_mod]
              exact Nat.mod_eq_of_lt (by omega)
            have hc : (m + 1 + (m + 1)) % (2 * m) = 2 := by
              rw [show m + 1 + (m + 1) = 2 * m + 2 by omega, Nat.add_mod, Nat.mod_self, Nat.zero_add, Nat.mod_mod]
              exact Nat.mod_eq_of_lt (by omega : 2 < 2 * m)
            rw [ha, hb, hc,
                cw4Entry_pos m (m + 1) (by omega) (Or.inr (Or.inr rfl)),
                cw4Entry_off m (m + 2) (by omega) (by omega) (by omega) (by omega),
                cw4Entry_pos m 1 (by omega) (Or.inl rfl),
                cw4Entry_off m 2 (by omega) (by omega) (by omega) (by omega)]
            omega
        · by_cases hr2m1 : r = 2 * m - 1
          · rw [hr2m1]
            by_cases hm2 : m = 2
            · subst hm2; decide
            · have ha : (1 + (2 * m - 1)) % (2 * m) = 0 := by
                rw [show 1 + (2 * m - 1) = 2 * m by omega]; exact Nat.mod_self (2 * m)
              have hb : (m + (2 * m - 1)) % (2 * m) = m - 1 := by
                rw [show m + (2 * m - 1) = 2 * m + (m - 1) by omega, Nat.add_mod, Nat.mod_self, Nat.zero_add, Nat.mod_mod]
                exact Nat.mod_eq_of_lt (by omega : m - 1 < 2 * m)
              have hc : (m + 1 + (2 * m - 1)) % (2 * m) = m := by
                rw [show m + 1 + (2 * m - 1) = 2 * m + m by omega, Nat.add_mod, Nat.mod_self, Nat.zero_add, Nat.mod_mod]
                exact Nat.mod_eq_of_lt (by omega : m < 2 * m)
              rw [ha, hb, hc,
                  cw4Entry_off m (2 * m - 1) (by omega) (by omega) (by omega) (by omega),
                  cw4Entry_zero_idx,
                  cw4Entry_off m (m - 1) (by omega) (by omega) (by omega) (by omega),
                  cw4Entry_pos m m (by omega) (Or.inr (Or.inl rfl))]
              omega
          · have ha : (1 + r) % (2 * m) = 1 + r := Nat.mod_eq_of_lt (by omega)
            have hb : (m + r) % (2 * m) = if r < m then m + r else r - m := by
              split
              · exact Nat.mod_eq_of_lt (by omega)
              · rw [show m + r = 2 * m + (r - m) by omega, Nat.add_mod, Nat.mod_self, Nat.zero_add, Nat.mod_mod]
                exact Nat.mod_eq_of_lt (by omega)
            have hc : (m + 1 + r) % (2 * m) = if r + 1 < m then m + 1 + r else r + 1 - m := by
              split
              · exact Nat.mod_eq_of_lt (by omega)
              · rw [show m + 1 + r = 2 * m + (r + 1 - m) by omega, Nat.add_mod, Nat.mod_self, Nat.zero_add, Nat.mod_mod]
                exact Nat.mod_eq_of_lt (by omega)
            rw [ha, hb, hc,
                cw4Entry_off m r (by omega) (by omega) (by omega) (by omega),
                cw4Entry_off m (1 + r) (by omega) (by omega) (by omega) (by omega)]
            by_cases hrm_lt : r < m
            · by_cases hr1m_lt : r + 1 < m
              · simp only [hrm_lt, hr1m_lt, ↓reduceIte]
                rw [cw4Entry_off m (m + r) (by omega) (by omega) (by omega) (by omega),
                    cw4Entry_off m (m + 1 + r) (by omega) (by omega) (by omega) (by omega)]
                omega
              · simp only [hrm_lt, hr1m_lt, ↓reduceIte]
                rw [cw4Entry_off m (m + r) (by omega) (by omega) (by omega) (by omega),
                    cw4Entry_off m (r + 1 - m) (by omega) (by omega) (by omega) (by omega)]
                omega
            · have hr1m_nlt : ¬(r + 1 < m) := by omega
              simp only [hrm_lt, hr1m_nlt, ↓reduceIte]
              rw [cw4Entry_off m (r - m) (by omega) (by omega) (by omega) (by omega),
                  cw4Entry_off m (r + 1 - m) (by omega) (by omega) (by omega) (by omega)]
              omega

/-- Increment bucket `k` in a difference-count histogram of length `n`. -/
def incAt : List Nat → Nat → List Nat
  | [], _ => []
  | x :: xs, 0 => (x + 1) :: xs
  | x :: xs, k + 1 => x :: incAt xs k

/-- Difference histogram `{(j - i) mod n | i ∈ a, j ∈ b}` as an `n`-element count vector. -/
def diffHist (n : Nat) (a b : List Nat) : List Nat :=
  a.foldl (fun acc i =>
    b.foldl (fun acc2 j =>
      incAt acc2 ((j + n - i % n) % n)) acc) (List.replicate n 0)

/-- Certifies that `(pos, neg)` forms a proper `CW(n, k)` circulant weighing matrix:
    disjoint supports inside `{0..n-1}`, total weight `k`, diagonal `k`, and
    equal positive/negative pair counts at every nonzero shift `1 ≤ s < n`. -/
def cwHistOrthogonal (n k : Nat) (pos neg : List Nat) : Bool :=
  pos.length + neg.length == k &&
  pos.all (fun i => i < n && !(neg.contains i)) &&
  neg.all (fun i => i < n) &&
  let pp := diffHist n pos pos
  let nn := diffHist n neg neg
  let pn := diffHist n pos neg
  let np := diffHist n neg pos
  let same := List.zipWith (· + ·) pp nn
  let opp := List.zipWith (· + ·) pn np
  same == (k :: opp.drop 1)

/-- Certified `CW(28, 16)` supports from `cw_weight16_support(28)`. -/
def cw28Pos : List Nat := [3, 4, 5, 8, 9, 14, 16, 17, 19, 23]
def cw28Neg : List Nat := [0, 1, 2, 15, 18, 22]

/-- Certified `CW(56, 16)` supports from `cw_weight16_support(56)`. -/
def cw56Pos : List Nat := [5, 8, 9, 16, 17, 28, 32, 33, 37, 45]
def cw56Neg : List Nat := [0, 1, 4, 29, 36, 44]

/-- Certified `CW(120, 36)` supports (`CW120_36_POS`, `CW120_36_NEG`). -/
def cw120Pos : List Nat :=
  [17, 26, 30, 37, 42, 54, 57, 58, 60, 62, 68, 70, 77, 84, 94, 97, 98, 106, 112, 116, 117]

def cw120Neg : List Nat :=
  [0, 2, 6, 8, 14, 18, 24, 50, 52, 56, 66, 74, 78, 102, 110]

theorem cw28_16_orthogonal : cwHistOrthogonal 28 16 cw28Pos cw28Neg = true := by
  decide +kernel

theorem cw56_16_orthogonal : cwHistOrthogonal 56 16 cw56Pos cw56Neg = true := by
  decide +kernel

theorem cw120_36_support_valid :
    cw120Pos.length = 21 ∧ cw120Neg.length = 15 ∧ cw120Pos.length + cw120Neg.length = 36 := by
  decide

theorem cw120_36_orthogonal : cwHistOrthogonal 120 36 cw120Pos cw120Neg = true := by
  decide +kernel

end OjasXMath
