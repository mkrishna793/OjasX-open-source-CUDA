/-! Dependency-free M0/M1 uniform-cotangent representation laws.

`I` is an index type and a uniform cotangent stores only its scalar witness.
These are exact rational representation theorems, not IEEE floating-point or
allocator claims. Runtime bitwise checks live in the Python differential tests.
-/
namespace OjasXMath

structure UniformCotangent (I : Type) where
  scalar : Rat

/-- Dense decoding is the explicit materialization boundary. -/
def decode {I : Type} (u : UniformCotangent I) : I → Rat :=
  fun _ => u.scalar

def materialize {I : Type} (u : UniformCotangent I) : I → Rat :=
  decode u

def fromScalar {I : Type} (g : Rat) : UniformCotangent I :=
  ⟨g⟩

def scale {I : Type} (s : Rat) (u : UniformCotangent I) : UniformCotangent I :=
  ⟨s * u.scalar⟩

def mean {I : Type} (n : Rat) (u : UniformCotangent I) : UniformCotangent I :=
  ⟨u.scalar / n⟩

def add {I : Type} (u v : UniformCotangent I) : UniformCotangent I :=
  ⟨u.scalar + v.scalar⟩

/-- Contravariant transport along an index morphism. -/
def reindex {I J : Type} (_f : J → I) (u : UniformCotangent I) : UniformCotangent J :=
  ⟨u.scalar⟩

/-- The full-sum adjoint sends a scalar upstream to the constant ones field. -/
theorem structured_sum_uniform {I : Type} (g : Rat) (i : I) :
    decode (fromScalar g : UniformCotangent I) i = g := rfl

/-- A nonzero rational divisor gives the exact mean-adjoint scalar law. -/
theorem structured_mean_uniform {I : Type} (n g : Rat) (i : I) :
    n ≠ 0 → decode (mean n (fromScalar g) : UniformCotangent I) i = g / n :=
  fun _ => rfl

/-- Uniform addition needs one scalar addition and decodes pointwise. -/
theorem structured_uniform_add {I : Type} (u v : UniformCotangent I) (i : I) :
    decode (add u v) i = decode u i + decode v i := rfl

/-- Reindexing commutes with decoding and is functorial under composition. -/
theorem structured_uniform_reindex {I J K : Type}
    (f : J → I) (g : K → J) (u : UniformCotangent I) (k : K) :
    decode (reindex (f ∘ g) u) k = decode (reindex g (reindex f u)) k := rfl

theorem structured_reindex_identity {I : Type} (u : UniformCotangent I) (i : I) :
    decode (reindex id u) i = decode u i := rfl

/-- Materialization changes representation, not any logical entry. -/
theorem structured_materialize_identity {I : Type} (u : UniformCotangent I) (i : I) :
    materialize u i = decode u i := rfl

end OjasXMath
