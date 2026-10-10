# Corrected reversible-optic theorem and OjasX implementation

Speed-focused extension: [cached nonlinear reverse](CACHED_REVERSE_THEOREM.md) removes projection replay by retaining selected preactivations and adds a native Metal joint SiLU step. It trades the old depth-independent tape for a smaller native-count backward schedule and measured GPU speed gains. The original replay construction and measurements below retain their scope.

This extension evaluates the attached “Symplectic Dagger-Optic” idea, replaces unsupported claims with precise contracts, adds Lean/Z3 evidence and tests a first-order CPU reversible-chain prototype. It uses known reversible coupling mathematics; it does not claim a new universal backpropagation algorithm. Additive reversible networks are an established research direction: [The Reversible Residual Network](https://arxiv.org/abs/1707.04585).

GPU update: the same prototype now has sampled MPS FP32 [qualification](GPU_QUALIFICATION.md). Depth 64 saves 99.22% of fresh activation tape but runs about 1.58× slower than native GPU PyTorch. Earlier CPU measurements below retain their original scope.

## 1. Corrections to the proposed framework

**The paired operator is a mixed-domain operator.** In column convention W:V→U, the pair (x,bar_y) belongs to V×U*, while its output (Wx,Wᵀbar_y) belongs to U×V*. For rectangular W, this is not a map T*V→T*U. The two streams travel in opposite directions.

**The displayed bundle is not generally symplectic.** In the square scalar example W=2, diag(W,Wᵀ) scales both position and momentum by two. The canonical bilinear pairing is therefore multiplied by four. A genuine forward cotangent lift of an invertible linear map sends (x,p) to (Wx,W^(-T)p). The reverse derivative instead sends an output cotangent to an input cotangent using Wᵀ. Those directions must not be confused.

**A correctly typed relation does preserve the relevant pairing.** Define omega((q,p),(q',p'))=<q,p'>-<q',p>. For any rectangular W,

    omega((q,Wᵀp),(q',Wᵀp'))
      = omega((Wq,p),(Wq',p')).

Lean proves the arbitrary finite-coordinate identity. This is a linear cotangent relation, not an invertible phase-space operator for arbitrary W, and not a full nonlinear symplectic-flow theorem.

**Transpose reconstruction requires an isometry.** WᵀWx=x when WᵀW=I. Training normally changes W, so the constraint needs a deliberate constrained optimizer/retraction. A general invertible matrix uses an inverse, not its transpose. Nonlinear activation inversion is an additional requirement; ReLU, for example, loses information.

**Weight-gradient orientation matters.** For columns z=Wx+b, e=bar_y⊙sigma'(z), the gradients are bar_x=Wᵀe and grad_W=e xᵀ. For row batches z=XW+b, they are bar_X=E Wᵀ and grad_W=XᵀE. The evaluation pairing returns a scalar; it is not itself the full outer-product gradient.

**Memory traffic and computation are not eliminated by a name.** Bundling forward, input-gradient and parameter-gradient outputs still requires their arithmetic and reductions. A resident tile can be reused where scheduling and resource constraints permit; “exactly one weight fetch” and “one GEMM worth of work” are not universal consequences.

**Causality remains.** The incoming loss gradient becomes available after the relevant forward/loss computation. Pipelining another microbatch can improve utilization, but does not provide the current gradient earlier. Gradient accumulation and delayed optimizer updates must preserve the required weight versions.

## 2. Stronger construction: arbitrary nonlinear functions inside a reversible block

Split the state into two parts. Let F_theta:B→A and G_phi:A→B be arbitrary deterministic functions. Define

    y1 = x1 + F_theta(x2)
    y2 = x2 + G_phi(y1).

Reconstruct using

    x2 = y2 - G_phi(y1)
    x1 = y1 - F_theta(x2).

Neither F nor G needs to be invertible, linear or an isometry. Their input and output sizes must match the two additions. GELU, SiLU and other nonlinear functions can occur inside these branches. This does not let a compiler replace an arbitrary existing MLP by the coupling block without changing the model: it requires an existing coupling architecture or an explicit architecture choice.

Lean proves both inverse identities over arbitrary additive commutative groups. This is stronger than imposing orthogonality on every branch weight. The real-number inverse does not imply that rounded machine arithmetic has an exact inverse.

## 3. All gradients and reconstruction in one backward schedule

Let g1 and g2 be cotangents arriving at y1 and y2. Using derivatives evaluated at the original/reconstructed branch inputs:

    a      = g1 + DG_phi(y1)ᵀ g2
    bar_x1 = a
    bar_x2 = g2 + DF_theta(x2)ᵀ a
    bar_theta = D_theta F_theta(x2)ᵀ a
    bar_phi   = D_phi G_phi(y1)ᵀ g2.

Bias gradients are included in the parameter pullbacks. Branch parameters remain independent and trainable; sharing parameters accumulates the contributions from each use.

The improved schedule reuses each replay evaluation:

1. Replay G(y1) with a temporary local derivative tape.
2. Recover x2=y2-G(y1).
3. Pull back g2 through that same G evaluation, obtaining a and grad_phi; release G's local tape.
4. Replay F(x2) with a temporary local derivative tape.
5. Recover x1=y1-F(x2).
6. Pull back a through that same F evaluation, obtaining bar_x2 and grad_theta; release F's tape.
7. Continue with the previous block using (x1,x2) and their cotangents.

This avoids the naive method of evaluating branches once for reconstruction and again for gradient replay. It still performs additional forward branch evaluations compared with storing all original branch activations. Forward and backward do not happen before their data dependencies permit them.

## 4. Formal evidence delivered

`lean/OjasXMath/ReversibleOptic.lean` adds eleven statements:

- Left and right inverse identities for arbitrary additive coupling functions.
- Preservation of the forward and inverse when branch implementations are equal functions. This permits a separately justified Gram/contraction rewrite inside a branch without changing the coupling function over reals.
- Reconstruction-based equality of a certified stage's pullback.
- The resulting reconstructed adjoint pairing, using the stage's actual derivative contract.
- All scalar affine-coupling input and parameter cotangents as one directional pairing.
- An actual `HasDerivAt` theorem for an affine coupling path with independent input/parameter perturbations.
- The arbitrary finite-coordinate canonical cotangent relation.
- The scalar factor-four refutation of the naive dagger bundle's symplectic claim.
- Arbitrary finite-matrix isometric reconstruction under WᵀW=I.
- A conditional reconstruction-error bound.

These are explicit mathematical contracts, not a full Lean analytic certificate for arbitrary GELU/SiLU networks or a formal refinement proof of Python/GPU execution. General branch reversibility does not require differentiability; branch pullback correctness separately requires derivative contracts or an explicitly chosen nonsmooth convention.

`tools/verify_reversible_z3.py` adds eight nonvacuous scoped proofs: affine round trips, all affine input/parameter cotangents, a rectangular 2×3 canonical relation, scalar isometric reconstruction, a scalar inverse-transpose cotangent lift, a conditional error bound and checkpoint-boundary integer counting.

It also verifies three counterexamples:

1. W=2 makes the proposed diagonal bundle scale the canonical pairing by four.
2. Old forward y=2·1 cannot reconstruct x=1 using a newly updated weight 1.5.
3. In binary32 round-to-nearest ties-to-even, (1+2^24)-2^24 rounds to 0, not 1.

The third counterexample uses finite ordinary values. Real algebra alone cannot establish bitwise reversible activations.

## 5. Floating-point reconstruction and checkpoints

Suppose a replayed G value has error e_G. Then the recovered second state is x2-e_G. If F changes by at most gain*|e_G| under that perturbation and F replay has error e_F, the recovered first state's error is bounded by

    gain*|e_G| + |e_F|.

Lean proves this conditional scalar bound; Z3 checks its explicit inequality model. In a vector implementation, analogous norm bounds require actual branch Lipschitz constants and local rounding-error estimates. They are not automatically provided by this CPU prototype.

The inverse block's conditioning can amplify errors across depth. A practical compiler should choose between native retained activations, reversible replay and exact checkpoint boundaries. Saving boundaries every K blocks limits the reconstruction span but increases storage. A correction residual can restore information only if it actually retains what rounding discarded; it has no universal small-storage guarantee.

Checkpoint segmentation, RNG replay for dropout, mutable-state handling for BatchNorm and automatic conditioning-based selection are design requirements, not delivered features here. The implemented branches are fixed deterministic affine-plus-GELU/SiLU functions and preserve the original parameter version until backward completes.

## 6. Executable prototype and limitations

`ojasx2/reversible.py` implements a complete chain as one custom autograd operation. It retains the two final states plus references to the original parameters. Backward reconstructs states in reverse order and builds only the currently needed branch tape.

Use it only for an existing coupling-chain model:

```python
from ojasx2 import reversible_coupling_chain
# Each block contains (wf,bf,wg,bg), with square branch weights.
# Defaults: activation='gelu', scale=0.1, accuracy='strict'.
y1, y2 = reversible_coupling_chain(x1, x2, blocks, accuracy='algebraic')
loss = loss_function(y1, y2)
loss.backward()
optimizer.step()  # after reconstruction/gradient accumulation has completed
```

Strict mode evaluates the ordinary coupling graph and supports native higher derivatives. Algebraic replay is opt-in, experimentally tested on CPU FP32/FP64 and MPS FP32 without autocast, and supports first-order gradients only. It explicitly rejects higher-order requests rather than returning an incorrect derivative. Saved-tensor version checks reject parameter mutation before replay. It has no universal finite-value/conditioning guard or native fallback for an unstable replay: use strict mode when that reconstruction contract is not qualified.

This prototype is not an automatic FX conversion of ordinary layers, not a native GPU kernel and not support for arbitrary stateful/random branch functions. Its mathematical generality exceeds its deliberately narrow runtime scope.

## 7. What memory is reduced

For fixed-width, deterministic reversible chains, the inter-block activation state retained across the forward/backward boundary can be independent of depth. That is O(Bd) state, not literally zero bytes. A local replay workspace is also needed.

Weights, dense gradients, optimizer state and nonreversible boundaries remain. Parameter references in the autograd tape are not copied parameters, but those underlying tensors still occupy memory. If an application separately keeps every intermediate output alive, this implementation cannot reclaim them.

The measurement uses fresh autograd-retained storage excluding supplied inputs and parameters. It does not measure process RSS, total backward peak, or physical GPU allocation. A small saved tape does not prove a workload fits on a single GPU.

## 8. How this connects to the existing OjasX arithmetic tools

Reversibility attacks the lifetime of saved states. Gram/sufficient-statistic rewrites attack the size and arithmetic of a branch calculation. A qualified branch rewrite can be composed with a reversible architecture; Lean's branch-rewrite statement preserves its forward/inverse functions over reals.

The current prototype uses plain affine+activation branches, so it does not implement or benchmark that combined Gram-plus-reversible path. A future backend should select already-matching graph patterns, verify branch equivalence and numerical conditions, choose checkpoint boundaries, then lower replay/pullback contractions into target-specific kernels.

For ordinary existing networks that are not reversible, use the existing legal contraction/residual rules or checkpointing. Turning a generic dense/GELU network into a different reversible architecture changes its function and cannot be called a transparent backend optimization.

The implementation goal is therefore a choice of certified schedules: retain when needed, reconstruct where justified, and reduce arithmetic independently where a smaller sufficient core exists.


## 9. Latest CPU results

| Coupling depth | Native joint ms | Replay joint ms | Native fresh saved bytes | Replay fresh saved bytes |
|---|---:|---:|---:|---:|
| 4 | 0.616 | 1.072 | 245,760 | 32,768 |
| 16 | 2.291 | 3.951 | 1,032,192 | 32,768 |
| 64 | 9.213 | 15.675 | 4,177,920 | 32,768 |

At depth 64, fresh retained activation storage falls by 99.22%. Replay is 1.70× slower in this CPU measurement; no speedup is claimed. Its maximum relative-L2 output/gradient discrepancy is 3.49e-07 in the tested stable regime.

The synthetic depth-eight, 30-step AdamW trajectory qualifies with maximum gradient relative-L2 difference 2.05e-07 and maximum parameter absolute difference 2.98e-08. This is not a universal floating-point or convergence certificate.

Verification totals after MPS qualification: 139 Lean statements; 33 existing scoped Z3 checks plus eight reversible checks and three reversible counterexamples; 333 passing regression tests, including 59 MPS checks, with one CUDA skip. The reversible runtime has 17 CPU tests and sampled MPS coverage. No Python-to-Lean formal refinement or universal GPU certificate is claimed.
