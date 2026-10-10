# OjasX workload scaling and mathematical toolbox

The current extension broadens synthetic CPU qualification; it does not certify every ML workload. ML workloads vary in graph, dimensions, datatype, optimizer, dataset, hardware and training duration. A universal claim would require proofs of all supported operator contracts, compiler transformations and numerical semantics, plus realistic device measurements. The current proof and test boundaries are listed below.

## A. What was scaled

The RMS sufficient-core rule now accepts contiguous activations with leading batch/token dimensions, such as [batch, sequence, features] and additional leading axes. It views those leading axes as independent rows, runs the reduced graph, and restores the output shape. The existing arbitrary-finite row-index theorems apply to this mathematical reindexing. There is no new Lean proof of PyTorch reshape implementation correctness. Noncontiguous multi-axis inputs retain native evaluation rather than quietly allocate a large copy. Strided two-dimensional support remains.

New synthetic qualification includes eight graph families in FP32 and FP64:

| Family | What runs | RMS pattern rewrite sites | What this establishes |
|---|---|---:|---|
| Residual token network | Three RMS-core blocks with residual additions | 3 | Multi-axis inputs, deeper graph, upstream gradients |
| Shared-weight network | The same RMS-core block used twice | 2 | Accumulation into the original shared parameters |
| CNN with BatchNorm | Bias-free Conv2d, BatchNorm, ReLU, pooling, RMS-core head | 1 | Convolution stays native; selected dense head can use the rule |
| Recurrent network | Four unrolled tanh recurrent steps and RMS-core head | 1 | Backprop through time stays native; selected head can use the rule |
| Standard causal Transformer | RMSNorm, QKV, SDPA, residuals, SwiGLU | 0 | Correct native execution; no claimed RMS-core acceleration |
| Branched hidden activation | Wide tensor also used by a skip branch | 0 | The compiler respects other consumers |
| Tied embedding/decoder | Embedding, RMSNorm, shared output weight | 0 | Tied parameter gradients remain native |
| GELU/dropout network | Eligible RMS block followed by GELU, dropout and a head | 1 | Smooth activations and dropout are preserved outside the rewritten block |

Every positive family test checks outputs, all parameter gradients, three AdamW steps, two accumulated microbatches per step, parameter identity, optimizer moments and module buffers. Dropout RNG is matched between reference and candidate. There are 16 such dtype/family cases. These are synthetic small models, not downloaded pretrained architectures or real-data training studies. The recurrent case is an explicitly unrolled tanh recurrence, not every cuDNN/GRU/LSTM implementation.

Sixteen further dtype/shape cases cover varying batches, bottleneck widths, hidden widths, output widths and 2D/3D/4D activations. A separate numerical first/second-derivative check covers token-shaped inputs. They test full gradients against native PyTorch with an explicit scale-aware relative-L2 contract, not bitwise equality.

The separate scaling benchmark includes seven block shapes. The largest has m=2048, k=32, n=8192, p=64: its ordinary hidden activation has 16,777,216 entries, about 64 MiB in FP32 before other activation copies, gradients and workspace. It also measures complete residual and ordinary Transformer forward/gradient graphs. Compile time is excluded from their steady-state timing; Python/runtime guards are included. CPU timings and profiler tensor allocations do not certify physical GPU behavior.

## B. A failure that matters: local gradient accuracy versus training accuracy

The biased version of the Conv2d → BatchNorm model is a deliberately retained negative qualification. In exact arithmetic, a spatially constant convolution bias is removed by training-mode BatchNorm's centering. Its gradient can consequently be tiny in floating point.

AdamW scales a gradient using approximately g/(sqrt(v)+epsilon). When g and sqrt(v) are extremely small, a tiny difference between two gradient computations can affect the parameter update much more than a large relative-L2 gradient check suggests. The broad FP32 experiment found a parameter difference exceeding the declared 2e-4 absolute contract, even though the first local gradient comparison passed.

This is not proof that either graph has the wrong real-number derivative. It demonstrates that a local floating-point tolerance is insufficient to certify a complete optimizer trajectory. The default strict mode remains available; the algebraic backend does not automatically detect or protect every such optimizer-sensitive graph. Passing the bias-free control does not repair or qualify the biased original model. No optimizer epsilon or model bias was silently changed in the negative test.

`reports/workload-scaling.json` explicitly records this failure as qualified=false. The regression test passes by demonstrating and preserving the limitation. A green overall verification run means the positive checks pass and the expected counterexample is reproduced, not that every tested candidate model qualifies for algebraic mode.

## C. Tool 1: derivative maps, daggers and composition

Imagine each layer sends two things: its forward result and just enough saved information for its backward calculation. When the loss sends a gradient back, the layer uses that information to produce gradients for its inputs and parameters. That is the residual-carrying lens interface.

For a linear map with Euclidean coordinates, the dagger is the transpose. For a nonlinear map f, it is the transpose of the derivative at the actual input:

    backward(g) = Df(x)ᵀ g.

The composition law says that the pullback of a composition runs the individual pullbacks in reverse order. Branches add their cotangents; shared parameters receive contributions from all uses. These laws make transformations checkable and provide a common representation across hardware.

This is already the mathematical foundation of ordinary reverse-mode differentiation. Its presence alone does not halve backpropagation. Our certified-stage structure additionally requires an actual derivative contract and an adjoint pairing; composing two correct primitive contracts yields another correct stage. An arbitrary backward function does not qualify merely because it is written in a lens interface.

Status: Lean derivative/adjoint composition framework and executable Python lens reference. Not a universal certificate for every PyTorch operator or native compiler.

## D. Tool 2: joint tensor-contraction planning

Associativity allows the same product to be calculated in different orders:

    (XA)B = X(AB).

With X[m,k], A[k,n], B[n,p], a wide intermediate XA has m*n entries. AB instead has k*p entries. Which is cheaper depends on the dimensions. Planning forward alone is insufficient because the gradient costs may reverse the choice.

For the two-layer example, let G be the output gradient and T=XᵀG. Then

    Y=X(AB), dX=G(AB)ᵀ, dA=TBᵀ, dB=AᵀT.

Both original parameter gradients remain. We are reusing useful contractions, not merging trainable layers into one independent parameter.

The longer-chain dynamic planner compares binary multiplication trees using the sum of forward and demanded backward scalar costs. It accounts for frozen inputs/weights. Its optimality is within that declared finite tree family and cost model; it is not a proof of globally optimal GPU scheduling. The exhaustive small-tree tests check that implementation.

Status: real-algebra Lean identities, executable guarded two-layer and longer-chain paths, compiler support and differential/training tests. Arbitrary nonlinear boundaries are preserved.

## E. Tool 3: Gram matrices and sufficient statistics for normalization

RMSNorm needs a row's squared length. It does not need every hidden entry independently after the final projection. For Z=XA:

    ||xA||² = x(AAᵀ)xᵀ.

If n is large and k is small, AAᵀ is much smaller than a batch of hidden rows. Define Q=AAᵀ and P=A diag(gamma) B. Obtain the normalization scalar from xQxᵀ and the projected output from xP. All original parameters still influence the smaller graph and are differentiated through.

LayerNorm also subtracts the row mean. Centering A's columns gives a centered weight matrix C, so the centered row is xC and its variance comes from CCᵀ. The existing centered-weight Gram LayerNorm rule handles this additional statistic, with extra conditioning guards.

The RMS core's leading all-gradient model is

    native = 6mn(k+p)
    core   = 6k(k+p)(n+m).

It favors k < mn/(m+n). Huge n alone is not enough; a tiny batch or large k may make it unprofitable. The implementation calculates Q/P anew for the current weight version. It does not pretend their construction or gradients are free.

Status: finite-index and matrix reassociation proofs; analytic scalar normalization derivative; arbitrary finite rank-one pairing; experimental CPU implementations and supported RMS compiler recognition. Floating-point error guards remain heuristic. Dense weights, dense gradients and optimizer state remain.

## F. Tool 4: structured nonlinear derivatives

A dense Jacobian is often unnecessary. For RMSNorm without its learned scale:

    D(z)[v] = r*v - (r³/n)*z*dot(z,v).

This is a scaled identity plus one rank-one correction. LayerNorm adds a mean-removal correction. Softmax has

    J = diag(p)-p pᵀ,
    Jᵀg = p ⊙ (g-dot(p,g)).

These are vector operations instead of constructing an n×n Jacobian. PyTorch already uses efficient vector derivatives for these standard operators, so merely restating them is not a new speedup. The useful additional step is combining their structure with adjacent contractions and eliminating unnecessary wide intermediates.

For attention O=PV, the softmax row reduction satisfies dot(P_row, dP_row)=dot(O_row, dO_row). This can supply the row scalar from the output and its cotangent. It is a building block for tiled attention backward, not a delivered faster full attention kernel. Native SDPA is the current full-attention path.

Status: algebraic Lean pairing/row identities, reference routines and CPU tests; broad primitive numerical/formal coverage is incomplete.

## G. Tool 5: exact factorized cotangents and activation masks

If an incoming cotangent is exactly G=UVᵀ with a small factor width, matrix pullbacks can be reassociated around U and V without materializing G. The factorization must be actually available or cheaply derivable. Computing an expensive factorization and ignoring its cost would invalidate the saving. Truncating an SVD generally changes the gradient.

Likewise, a derivative mask with an exact separable representation can preserve a small contraction core. ReLU has a particularly useful conditional case: every row has the same activation mask. Then one common diagonal mask can be pushed into the weights. The implementation checks equality of masks; ordinary random masks usually differ and retain native evaluation.

GELU and SwiGLU normally have different derivatives for different hidden entries and rows. There is no current exact universal rule that collapses those arbitrary values to the same small Gram core. Approximate compression needs its own error/convergence contract.

Status: finite-index factor identities and selected runtime paths; no universal factor discovery or arbitrary activation compression.

## H. Tool 6: residual sufficiency and memory compression

The question is: what information must survive the forward pass so that backward can still be reconstructed?

ReLU usually needs one branch bit per value rather than the original FP32 input solely for its derivative. A packed bit mask stores eight branch bits per byte. Compared with four-byte values, that particular derivative residual can be up to 32 times smaller before packing overhead. If another layer already saves the activation, adding a mask can increase total memory, which is why whole-graph liveness matters.

For a general operation, output-only backward is possible only if all inputs producing the same retained output also require the same backward result. The sufficiency/fiber theorems formalize that condition. Squaring provides a simple counterexample: +a and -a produce the same output but opposite derivatives. Saving the output alone loses necessary information. Product gates similarly need information about their factors.

GELU/SiLU references retain suitable forward inputs; SwiGLU retains the information needed about both factors. Checkpointing/recomputation can trade retained memory for extra computation, but it is not free arithmetic reduction.

Status: Lean sufficiency/impossibility results, packed ReLU and residual-carrying implementations, saved-storage audit. A whole-model optimal liveness planner is pending.

## I. Tool 7: low-rank bilinear circuits and streamed adjoints

A bilinear operator can be written as input encoders, pointwise products and an output decoder. Its backward uses the transposes of those encoders/decoder and the ordinary product derivative. Shared work can be organized in one mathematical circuit while still respecting when the loss gradient becomes available.

The rank-seven 2×2 block matrix multiplication circuit replaces eight block products by seven, with additional additions and temporary storage. The project proves that algebraic circuit and its pullbacks. It also explores storing only original operands and regenerating encoded channels during backward.

Fewer multiplication blocks does not guarantee faster real GEMM: additions, conditioning, tile granularity, tensor instructions and traffic matter. Current Python experiments do not certify a production GPU replacement for optimized vendor GEMM.

Status: Lean general bilinear/multilinear gradient identities and rank-seven equivalence, experimental streamed CPU operators, scoped numerical/cost tests.

## J. Tool 8: geometry for constrained parameters

Some models constrain vectors to a sphere or matrices to orthonormal columns. A Euclidean gradient can be projected onto the allowed tangent space. For an orthonormal-column matrix W, one common tangent projection is

    G_tangent = G-W*sym(WᵀG).

The project has real geometric tangent/projection identities. This preserves relevant first-order geometric constraints; it does not by itself supply a complete optimizer/retraction or halve a generic dense gradient computation. Ordinary unconstrained PyTorch training does not use these constraints automatically.

Status: formal geometric building blocks, not a production manifold optimizer.

## K. Tool 9: cost, memory and hardware bounds

The roofline idea compares arithmetic demand with data movement: time cannot be smaller than the bottleneck implied by operations/compute rate and bytes/bandwidth under the stated model. A weight-reuse tile can reduce bytes while performing the same multiplications. A Gram rewrite can reduce multiplications and intermediates. Both can help, but in different ways.

On-chip memory is finite. Keeping more shared tiles and accumulators live can reduce occupancy. Tensor/matrix instructions have shape and layout requirements. A hardware lowering must count registers, shared memory, output-gradient storage, barriers and workspace. The paired dagger pipeline is a design possibility, not currently a tested Metal/CUDA kernel.

Overall training memory also contains weights, dense gradients, optimizer state and other graph branches. Shrinking one activation does not make those disappear. Whole-model acceleration depends on how much step time the optimized region contributes; isolated speedups cannot be applied to the complete workload unchanged.

Status: cost identities, conditional bounds and recorded tensor-storage measurements; native GPU lowering/global memory planning remains pending.

## L. What the software tools contribute

- Lean 4 checks proofs of precisely stated real-number/finite-index contracts. It does not measure GPU speed or automatically discover every profitable rewrite.
- Z3 checks encoded finite machine/model problems, such as packing/layout domains and counterexamples to floating-point equivalence. The current scoped models do not prove arbitrary PyTorch training programs.
- SymPy expands and differentiates selected symbolic examples to cross-check formulas. Those checks are useful but are not universal kernel-checked proofs.
- PyTorch autograd evaluates gradients of the implemented reduced graph. gradcheck/gradgradcheck compare numerical derivatives; differential tests compare with the ordinary graph.
- torch.compile and our FX pass recognize legal candidate patterns while keeping parameter identity, branch boundaries and native fallbacks.
- Timing, saved-tensor hooks and the CPU profiler measure observed behavior. Their results must retain scope, shape and failed numerical contracts.

The mathematics is a toolbox rather than one universal replacement equation. OjasX's useful next production capability is a planner that selects a legal, numerically qualified smaller computation for each graph region, then generates hardware-specific kernels. Standard nonlinear Transformer blocks, mixed precision, long real-data training, distributed optimizers and native GPU performance still need separate qualification.


## M. Latest scaling measurements

CPU FP32, two threads, median of seven randomized-order repetitions after warmup, forward plus all gradients. These measurements are synthetic and shape-specific. GPU execution, real-data convergence and process memory were not measured.

| Workload | Native ms | Candidate ms | Speedup | Incremental tensor peak reduction |
|---|---:|---:|---:|---:|
| 256-8-256-8 | 0.470 | 0.279 | 1.68× | 96.25% |
| 1024-16-1024-16 | 2.433 | 0.465 | 5.24× | 98.15% |
| 2048-32-4096-32 | 36.482 | 1.413 | 25.82× | 98.56% |
| 2048-32-8192-64 | 70.060 | 1.897 | 36.93× | 97.92% |
| 1024-128-512-128 | 2.135 | 1.287 | 1.66× | 75.94% |
| 64-64-512-32 (native fallback) | 0.256 | 0.278 | 0.92× | 0.00% |
| 1-16-1024-8 (native fallback) | 0.118 | 0.136 | 0.86× | 0.00% |
| residual_tokens | 0.850 | 0.850 | 1.00× | 85.82% |
| standard_transformer_swiglu | 0.431 | 0.450 | 0.96× | 0.00% |

For the largest block, modeled main-product scalar arithmetic falls from 9,629,138,944 to 187,099,136, or 98.06% less. The largest relative-L2 output/gradient discrepancy is 2.33e-06. The norm-based qualification does not imply every element meets the tighter entrywise comparison; both results are retained in the report.

The full residual graph used three rewrite sites and reduced incremental tensor peak by 85.82%, but its measured speedup was only about 1.00×. The standard Transformer used zero rewrite sites and had no memory reduction; its approximately 0.96× ratio reflects steady-state backend/wrapper overhead and timing variation, not arithmetic acceleration.

The negative biased-CNN case had a first-step maximum bias gradient of 3.04e-09, a maximum gradient difference of 5.89e-09, and bias parameter drift reaching 0.00104 after three AdamW steps. Its declared parameter absolute tolerance was 0.0002; it remains unqualified.

Final verification: 128 Lean statements audited, 33 scoped Z3 checks, 257 passing CPU regression tests and two device skips. One passing regression deliberately reproduces the negative optimizer qualification. The new batch-axis/runtime changes are checked by tests; no additional Lean theorem count is implied.
