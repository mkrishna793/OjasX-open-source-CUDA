import Mathlib.Init

/-! Residual-carrying reverse maps. Correctness of primitive derivatives is
    a separate obligation; these laws certify the composition mechanism. -/
namespace OjasXMath

structure ReverseMap (A dA B dB S : Type*) where
  forward : A → S × B
  pullback : S → dB → dA

def ReverseMap.then {A dA B dB C dC S T : Type*}
    (f : ReverseMap A dA B dB S) (g : ReverseMap B dB C dC T) :
    ReverseMap A dA C dC (S × T) where
  forward x :=
    let a := f.forward x
    let b := g.forward a.2
    ((a.1, b.1), b.2)
  pullback tape dz := f.pullback tape.1 (g.pullback tape.2 dz)

def identityReverseMap (A dA : Type*) : ReverseMap A dA A dA Unit where
  forward x := ((), x)
  pullback _ dx := dx

theorem reverse_composition_forward {A dA B dB C dC S T : Type*}
    (f : ReverseMap A dA B dB S) (g : ReverseMap B dB C dC T) (x : A) :
    (f.then g).forward x =
      (((f.forward x).1, (g.forward (f.forward x).2).1),
       (g.forward (f.forward x).2).2) := rfl

theorem reverse_composition_pullback {A dA B dB C dC S T : Type*}
    (f : ReverseMap A dA B dB S) (g : ReverseMap B dB C dC T)
    (s : S) (t : T) (dz : dC) :
    (f.then g).pullback (s, t) dz = f.pullback s (g.pullback t dz) := rfl

/- Associativity is modulo the canonical reassociation of saved tapes. -/
theorem reverse_composition_associative {A dA B dB C dC D dD S T U : Type*}
    (f : ReverseMap A dA B dB S) (g : ReverseMap B dB C dC T)
    (h : ReverseMap C dC D dD U) (s : S) (t : T) (u : U) (dd : dD) :
    ((f.then g).then h).pullback ((s, t), u) dd =
      (f.then (g.then h)).pullback (s, (t, u)) dd := rfl

theorem reverse_left_identity {A dA B dB S : Type*}
    (f : ReverseMap A dA B dB S) (s : S) (db : dB) :
    ((identityReverseMap A dA).then f).pullback ((), s) db = f.pullback s db := rfl

theorem reverse_right_identity {A dA B dB S : Type*}
    (f : ReverseMap A dA B dB S) (s : S) (db : dB) :
    (f.then (identityReverseMap B dB)).pullback (s, ()) db = f.pullback s db := rfl

end OjasXMath
