# Unified streamed bilinear dagger: complete proofs and backend use

The new core statement characterizes the real gradient of any finite bilinear circuit with arbitrary linear encoders and decoder. The implementation specializes it to rank-7 matrix multiplication and combines it with a residual-memory policy. This is a verified formulation for OjasX, assembled from bilinear differentiation, adjoints and streaming; historical novelty has not been established.

All source proofs are in [StreamedDagger.lean](lean/OjasXMath/StreamedDagger.lean). The root [axiom audit](lean/OjasXMath.lean) includes every project theorem. This guide describes the 6 October phase. The current project-wide run checks **121 Lean theorems, 33 Z3 models and 209 CPU tests**, with two unavailable-device skips; [the 7 October mathematical extension](BACKPROP_MATH_STRUCTURE.md) adds certified composition and general multilinear rules. The subsequent [joint contraction optimization](CONTRACTION_OPTIMIZATION.md) supplies a different, measured linear-chain speedup; its gains must not be applied to the rank-7 measurements below.

## 1. One circuit and one joint cotangent rule

Let `a` and `b` be finite real coordinate vectors. Let `U,V,W` be arbitrary compatible linear encoders/decoder. Define

\[
e=Ua,\qquad f=Vb,\qquad y=W(e\odot f).
\]

For output cotangent `g`, compute the shared decoded cotangent

\[
d=W^Tg.
\]

Then the two input cotangents are

\[
\boxed{\bar a=U^T(d\odot f),\qquad \bar b=V^T(d\odot e).}
\]

The same `d_q` serves both gradients for channel `q`. A channel's contributions can be consumed before producing the next. Encoder values are computable from the original operands; retaining every encoded value across the forward/backward boundary is unnecessary.

The identity is general in channel count, dimensions and coefficients. It includes ordinary matrix multiplication after finite-coordinate expansion, and bilinear decompositions such as the supplied rank-7 scheme. For matrix-block implementations, the existing Frobenius-pairing theorem supplies the product cotangents `dA=G Bᵀ`, `dB=Aᵀ G`.

The abstract algebraic theorem works over commutative rings with the stated bilinear pairing. The actual analytic gradient theorem specializes to real coordinates. Complex Hermitian adjoints need conjugation and are outside the implementation contract.

## 2. Complete proof chain

For any directions `da,db`, the bilinear tangent is

\[
T=W\big((Uda)\odot(Vb)+(Ua)\odot(Vdb)\big).
\]

The exact expansion is

\[
F(a+t\,da,b+t\,db)=F(a,b)+tT+t^2 F(da,db).
\]

Lean proves this expansion, then proves `HasDerivAt` at `t=0` for every output coordinate. It differentiates the scalarized output by a finite-sum rule. This establishes an actual analytic derivative, rather than taking the tangent formula as an assumption.

The pairing calculation is:

\[
\begin{aligned}
\langle T,g\rangle
&=\langle (Uda)\odot f+e\odot(Vdb),W^Tg\rangle\\
&=\langle Uda,d\odot f\rangle+\langle Vdb,d\odot e\rangle\\
&=\langle da,U^T(d\odot f)\rangle+\langle db,V^T(d\odot e)\rangle.
\end{aligned}
\]

Consequently, for **every** pair of input directions,

\[
\left.\frac{d}{dt}\langle F(a+t\,da,b+t\,db),g\rangle\right|_{t=0}
=\langle da,\bar a\rangle+\langle db,\bar b\rangle.
\]

That is the real cotangent characterization checked by `unified_circuit_gradient_direction`. Its proof depends on the exact expansion, genuine scalar derivatives, finite summation and the unified pairing identity. `streamed_accumulation_correct` separately proves sequential accumulation equals the stated sum. All proofs compile; no project theorem adds an axiom or uses an unfinished proof placeholder.

This does not mechanically prove the Python program or a future GPU kernel refines the theorem. Python is checked by differential, analytic/numerical-gradient, registration, alias/version and workload tests. Floating accumulation changes are explicitly qualified by tests, not converted into a real-arithmetic bit guarantee.

## 3. Rank-7 specialization and streaming schedule

The reference implementation [streamed.py](ojasx2/streamed.py) stores only the operands required by requested gradients:

- Both input gradients requested: retain `A,B`.
- Only `dA` requested: retain `B`.
- Only `dB` requested: retain `A`.

Forward shares its four-add encoder circuits locally, but does not save encoded matrices. It writes the decoder into the four output quadrants and holds one product at a time. First-order backward reconstructs encoder channels with pointwise additions; it introduces **no additional GEMM recomputation**.

The backward product-channel order is `6,3,4,5,1,7,2`. It produces each requested A/B gradient product once. Unused gradient-output quadrants temporarily hold shared encoder cotangents. In particular, the pending B-channel-three contribution is stored in `B11` and consumed before that slot is overwritten. The final encoder pullbacks still use seven additions/subtractions per operand. The corresponding real schedules are independently checked by Z3.

Higher-order differentiation uses a differentiable dense pullback. The output-slot reuse policy is a first-order optimization; its memory/operation model must not be silently applied to a backward pass whose own graph is needed.

With both gradients requested and positive even dimensions `(m,k,n)`, the first-order scalar-operation model is

\[
T_F=\tfrac74mkn+mk+kn,
\]
\[
T_B=\tfrac72mkn+mn+\tfrac74(mk+kn),
\]
\[
\boxed{T_{F+B}=\tfrac{21}{4}mkn+\tfrac{11}{4}(mk+kn)+mn.}
\]

The extra quadratic term pays for reconstructing encoder channels. The cached scheme has a smaller arithmetic count, but retains encoded operands. For square dimensions the streamed count beats the dense count `6n³-3n²` when `n>38/3`, hence for even `n≥14`. This is a proved scalar model, excluding unary signs, copies, transfers, dispatches and device instruction behavior. The program still makes seven forward and fourteen backward GEMM calls, compared with three large dense calls.

In the measured both-gradient cached rank-7 case, unique operand-related saved storage is `2(mk+kn)` elements; operand-only storage is `mk+kn`. The new scheme removes the extra encoded tape, giving a 50% reduction relative to that cached scheme. Ordinary dense GEMM already saves only its needed operands, so this comparison does not establish a memory advantage over native dense GEMM.

## 4. Measurements, including the strongest baseline

CPU FP32 joint forward/backward runs tested square sizes 256, 512, 1024, 2048 and 4096. All sampled outputs and both gradients passed the stated numerical tolerance. Timing medians and tensor-memory timelines are in [medium benchmarks](reports/streamed-benchmark-cpu.json) and [large benchmarks](reports/streamed-benchmark-heavy.json).

For square size 4096:

| Variant | Joint median | Profiler incremental tensor peak |
|---|---:|---:|
| Native dense | 370.5 ms | 192 MiB |
| Cached rank-7 | 416.4 ms | 432 MiB |
| Operand-only streamed rank-7 | 393.2 ms | 272 MiB |

The streamed scheme reduces tracked incremental tensor peak by **37.04% versus cached rank-7**. Native dense remains the better measured CPU baseline. The raw profiler includes recorded tensor allocations, not all process residency, and profiling can perturb lifetime. Timings are local observations, not stable performance guarantees. The heavy benchmark overlapped a brief Core ML probe during initial execution, so these numbers are exploratory; a release/autotuning decision requires isolated repeated target measurements.

The practical memory backend uses a separate rewrite that preserves native GEMMs: a single-consumer `SiLU(gate)*up` becomes the registered SwiGLU operator. Its backward reconstructs one pointwise activation rather than retaining that activation's complete output. This increases pointwise computation while removing saved storage.

An eight-block residual FFN stack, batch 512, width 256 and hidden width 1024, achieved:

- **16 MiB fewer fresh saved activation bytes**, a **23.53%** reduction.
- Profiler incremental tensor peak **73.5 → 59.5 MiB**, a **19.05%** reduction.
- Bit-identical sampled outputs, input gradients and every parameter gradient.
- The same native GEMM calls; no projection recomputation.

The CPU memory-focused version was slower in this run because it adds pointwise recomputation. Single-block examples reduced saved bytes but did not reduce peak tensor allocation. The [memory-backend report](reports/memory-backend.json) keeps these distinctions explicit.

## 5. Use without changing the training loop

For the qualified CPU memory policy:

```python
import torch
from ojasx2 import make_backend

# Keep the ordinary model, optimizer, parameters, loss and backward calls.
compiled = torch.compile(
    model,
    backend=make_backend(accuracy="strict", memory_policy="pointwise"),
)
optimizer.zero_grad()
loss = loss_fn(compiled(inputs), targets)
loss.backward()
optimizer.step()
```

This backend performs actual FX node rewrites. It preserves parameter identity and ordinary optimizer use. The pointwise rule checks a single consumer and excludes inplace SiLU. Broadcasting, unsupported devices/dtypes and other patterns keep the native expression. Qualification currently covers CPU FP32/FP64; other targets need their own checks. The default `memory_policy="native"` performs no pointwise rewrite.

For numerical experiments with rank-7:

```python
backend = make_backend(accuracy="algebraic", minimum_inner=128)
compiled = torch.compile(model, backend=backend)
print(backend.evidence)
```

Compatible even 2D matmul/linear nodes use runtime eligibility guards; unsupported, small, odd or unfavorable arithmetic shapes retain native operations. The algebraic mode is an experiment, not a measured-profitability selector or a universal bounded-error contract. Strict mode keeps native GEMM. Model-specific error budgets, hardware timing and memory objectives must determine a production selection policy.

The explicit `shared_projections` helper also preserves cotangent demand: unused heads keep `None` gradients, so an optimizer such as AdamW leaves their weights unchanged. It retains original operands and reconstructs packed weights transiently instead of saving a concatenated weight buffer. These behaviors, contiguous output layout and higher-order differentiation are tested. Automatic graph-level projection grouping remains pending.

The backend emits an ATen/custom-operator graph. Native GPU code generation, global residual liveness, distributed/AMP qualification and proof-carrying compiler refinement remain work listed in [COMPILER_PLAN.md](COMPILER_PLAN.md).

## 6. On-chip lowering and Apple hardware

GEMM is already a computation performed by arithmetic execution units. The proposed hardware change is to keep encoded tiles, shared cotangent tiles and accumulators on chip, instead of writing complete encoded matrices to global memory. The Python reference allocates ATen tensors; it does not certify register/SRAM residency. An IO-aware design has precedent in [FlashAttention](https://arxiv.org/abs/2205.14135), but that paper's results do not validate this GEMM implementation.

Future lowering should parameterize tile sizes and hardware limits, generate encoder expressions during tile loads, consume each `d_q` for both requested products, reuse gradient output storage only under proven alias/lifetime conditions, and qualify matrix-unit accumulation precision. Z3's integer tile-partition check covers logical tails, not arbitrary machine overflow, barriers or race freedom. Such obligations need target-specific discharge and real device tests.

The [hardware probe](reports/hardware.json) confirms the laptop is an Apple M5 MacBook Air with 24 GB of memory and a 10-core Metal-capable GPU. This process reports zero Metal devices, and PyTorch reports MPS inaccessible. Core ML advertises a Neural Engine. Apple's [deployment guidance](https://machinelearning.apple.com/research/neural-engine-transformers) distinguishes GPU training workflows from Core ML's model execution across available engines.

Core ML tools 9.0 were installed in a separate project-local Python 3.12 environment. An explicit graph computes `A B`, `G Bᵀ` and `Aᵀ G`; all three FP16 results pass sampled tolerances against FP64 products of the quantized inputs. CPU-only and CPU+Neural-Engine configurations both execute, but their [compute plans](reports/coreml-probe-medium.json) select CPU. The runtime reports `IOSurfaceSharedEvent creation failed` and ANE compilation failure, even after the specific hardware-cache write permission was granted. Neural Engine execution has not been confirmed.

Reproduction commands on the laptop:

```sh
python -B tools/hardware_probe.py
python -B tools/bench_streamed.py --device mps
.tools/apple-venv/bin/python -B tools/coreml_probe.py --m 256 --k 512 --n 512
```

The Apple environment is optional and excluded from the source archive. Recreate it with Python 3.12 and `pip install coremltools==9.0`. A usable device execution context is still required; changing filesystem cache permissions did not resolve the Metal/IOSurface limitation observed here.
