# OjasX v0.3.0

OjasX is a PyTorch research backend for reducing retained activation storage through packed masks, projection replay, selective vocabulary loss, and mathematical graph rewrites. It includes Lean/Rocq specifications, optional native CPU and Apple Metal kernels, and experimental sequential partitions over TCP.

Mathematical proofs cover their stated specifications. Differential tests connect selected Python operations to those specifications. The proofs do not certify the optimized Python/C++/GPU implementations.

## Implemented behavior

- **Holonomic autograd:** linear/activation, SwiGLU, GeGLU, RMSNorm and chunked vocabulary-loss operations. Smooth activations replay projections during backward, trading extra computation for reduced retained intermediates. Retained tensors use `save_for_backward`.
- **Graph rewriting and routing:** `torch.compile(..., backend="holonomic")`, an AOT backend, and a workload router preserving GELU approximation and LeakyReLU slope.
- **Optional native kernels:** CPU and Apple Metal execution. CUDA, ROCm, SYCL and Ascend source templates are experimental generated source; the universal native runtime does not launch those vendors' kernels. Its non-Metal paths use CPU computation.
- **Portable reference operators:** missing or incompatible native libraries fall back to PyTorch numerical operators. Native source generation and chip-specific schedules require a compatible native TileLang library.
- **Experimental mesh:** contiguous assignments for a nonempty `nn.Sequential`, based on declared capacities. `port=0` selects a local device; a positive port connects to a preloaded worker at `host:port`. Remote workers own their parameters, gradients and optional optimizers.

Remote execution supports first-order autograd. Cross-partition shared parameters are rejected. Workers must load the assigned architecture and initial checkpoint; the coordinator checks their signatures. Preloading workers is explicit; arbitrary model code is not sent over the network. Capacity totals are caller-supplied estimates, not measured unified GPU memory. Optional int8 transport is lossy.

## Installation

Requires Python 3.9+ and PyTorch 2.8.

```bash
python -m pip install .
python -m pip install '.[test,benchmark]'
```

Default wheels exclude machine-specific native binaries and use the numerical fallback. Source checkouts may load a compatible existing native library. The loader selects the platform's extension and falls back on load failure. Set `OJASX_DISABLE_NATIVE=1` to force reference operators.

Optional native builds require an installed PyTorch and compiler, from the full source checkout:

```bash
OJASX_REBUILD_NATIVE=1 python -m pip install --no-build-isolation .
```

macOS builds CPU and Metal kernels; Linux builds the CPU core. Native TileLang source generation is unavailable in that Linux CPU-only build. Windows currently uses the PyTorch fallback. Failed optional builds also use the fallback.

## Quickstart

```python
import torch
import ojasx2

layer = ojasx2.HolonomicLinear(64, 128, act="silu")
x = torch.randn(32, 64, requires_grad=True)
layer(x).square().mean().backward()

router = ojasx2.SmartWorkloadRouter()
# For an existing model:
# model, report = router.optimize_model(model, batch_tokens=2048)
```

Savings and latency depend on shape, dtype, device and trainable parameters.

## Remote partitions

Both machines load the same architecture and checkpoint and use the same descriptors. `plan_mesh_partitions` discovers assignments without moving layers or opening sockets.

```python
nodes = [
    ojasx2.MeshNodeSpec("local", "cpu", 8.0, 1.0, device="cpu"),
    ojasx2.MeshNodeSpec("partner", "cpu", 8.0, 1.0,
                       device="cpu", host="192.168.1.20", port=5000),
]
```

On the partner, after loading `model` from the shared checkpoint:

```python
plan = ojasx2.plan_mesh_partitions(model, nodes)
partition = torch.nn.Sequential(*(model[i] for i in plan.node_assignments["partner"]))
# Move the partition to the worker device before constructing its optimizer.
optimizer = torch.optim.SGD(partition.parameters(), lr=1e-3)
worker = ojasx2.MeshRemoteWorkerServer(
    partition, host="0.0.0.0", port=5000, device="cpu",
    compress_int8=False, optimizer=optimizer,
)
try:
    worker.wait()
finally:
    worker.close()
```

On the coordinator, after loading the same initial model:

```python
mesh = ojasx2.HolonomicVirtualMeshEngine(model, nodes, compress_boundary=False)
del model  # Release the coordinator's original remote-partition template references.
# Build local optimizers after partitioning: remote parameters are worker-owned.
local_optimizer = torch.optim.SGD(mesh.parameters(), lr=1e-3)
local_optimizer.zero_grad(set_to_none=True)
mesh.zero_remote_grad()
loss = mesh(torch.randn(32, 64)).square().mean()  # Match your model input width.
loss.backward()
local_optimizer.step()
mesh.step_remote()
mesh.close()
```

Forward/backward requests use individual IDs, supporting multiple forwards and reverse-order backwards. Worker updates, gradient resets and train/eval changes are explicit and require pending backwards to finish. Inference under `torch.no_grad()` retains no remote tape. The coordinator initially needs its model/checkpoint template in memory; this does not demonstrate loading a model larger than its memory. Application-byte telemetry includes tensor-request opcodes, IDs, dtype metadata and frame headers; it excludes IP/TCP overhead and setup/optimizer control messages.

## Benchmarks and evidence

`tools/run_industrial_all_level_benchmark.py` is a project-maintained **local sample harness**, using PyTorch testing APIs and trace data. It is not an official PyTorch/Meta certification suite. Seeds fix generated inputs; timings vary with hardware, scheduling and environment.

| Level | Actual workload and scope |
|---|---|
| 1 | Native CPU operator checks and CPU/Metal mask comparison; RMSNorm forward/backward at [4,256,512]. The report records whether a Metal GPU was available. |
| 2 | Nonempty FP32 `nn.functional.linear` samples, with five activations. Checks forward and first-order input/weight/bias gradients. No second-derivative claim. |
| 3 | First 25 runnable BackendBench mm/addmm traces, with independent caps M≤512 and K,N≤1024. Tests contiguous FP32 synthetic linear/ReLU², rather than original dtype/stride/addmm semantics. Output and both operand gradients must agree before a case passes. |
| 4 | Custom RMSNorm/SwiGLU/residual/loss block: batch 4, sequence 128, width 256, hidden 512, vocabulary 4096. No attention or RoPE. Also tests a small LoRA-style model. |
| 5 | Same-process MPS/CPU or CPU/CPU TCP loopback and int8 application-frame compression. Declared 18+8 GB capacities are planning inputs, not measured two-laptop GPU pooling. |

`SavedTensorAudit` measures autograd-retained storage and deduplicates aliases. Its **fresh storage** metric excludes explicitly supplied inputs/parameters. “Zero fresh saved bytes” does not mean zero activation memory, peak memory or optimizer state. Replay backward allocates temporary buffers. First-order loss replay bounds vocabulary-logit intermediates; higher-order differentiation builds a replay graph and is not chunk-memory bounded.

Other scripts contain architecture-inspired synthetic workloads, kernel microbenchmarks and shape projections. “70B-scale SwiGLU” describes a projection shape, not execution of a complete 70B model. FakeTensor/Roofline estimates do not demonstrate physical GPU fit or measured out-of-memory behavior.

Committed reports are snapshots: check their scope, source hashes and hardware before applying their numbers to a modified checkout. Older Lean/Rocq reports cover the pre-fix source snapshot. Mathematical specifications are unchanged, but their source hashes do not freshly certify this revision.

## Reproduction

```bash
# Full suite, including independent-process worker regression tests.
pytest -q
# Explicit portable numerical implementation:
OJASX_DISABLE_NATIVE=1 pytest -q tests/test_review_regressions.py -k portable
# Requires optional native TileLang library and benchmark dependencies:
python tools/run_industrial_all_level_benchmark.py
# Single-laptop memory tradeoff, chunked LM-head replay, narrow linear-chain contraction, and loopback mesh harness:
python tools/run_two_laptop_physical_mesh.py --role local-proof
# Additional project-maintained sample harnesses:
python tools/run_official_benchmarks_suite.py
python tools/run_holonomic_opinfo_and_backendbench.py
```

The five-level script writes `reports/INDUSTRIAL_ALL_LEVEL_BENCHMARK_SCORE.json` with status, dimensions, topology, scope, timestamp and source hashes. Failed runs write a failed status rather than leaving a prior successful result presented as current.

`tools/run_two_laptop_physical_mesh.py` writes `reports/TWO_LAPTOP_PHYSICAL_MESH_REPORT.json` with source hashes and four scoped measurements: (1) a 12-block MLP comparing fresh autograd-saved storage plus post-backward persistent `.grad` buffers under `HolonomicGradFreeOptimizer`, (2) first-order chunked cross-entropy replay (`holonomic_lm_head_loss`) trading backward recomputation for zero fresh saved logit storage, (3) narrow-output two-linear reassociation (`linear_chain`), and (4) an analytical 16,384-token capacity projection paired with a 5-step, 512-token loopback TCP execution.

Formal specifications are in `lean/` and `rocq/`; runners and supporting documentation are in `tools/` and `docs/`. See `LICENSE` for AGPL-3.0 terms.
