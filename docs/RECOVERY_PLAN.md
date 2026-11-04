# OjasX low-RAM recovery and research plan

**Status:** active recovery, 2026-11-03. This document distinguishes executed evidence from design work. It does not treat the repository's representative-layer benchmark as a reproduction or refutation of anybody's separate 31B experiment.

## 1. What the claimed 31B/32B path actually did

### Historical root benchmark

`benchmark_32b_ojasx_e8_training.py` constructed one `E8LatticeLinear(16384, 32552)` containing 533,332,768 represented base weights, attached rank-8 LoRA adapters, and ran three random-target steps. It did **not** construct 60 layers or a complete transformer. Its 60-layer memory figures were extrapolations, and its printed summary contained hard-coded values not derived from its own calculations.

The script has now been corrected to report its real scope, actual losses/gradient norms, exact code-plus-scale storage, and OS high-water RSS. It remains useful as a large single-layer mechanical check. It is not full-model evidence.

### Historical HuggingFace recipe

The old recipe called `AutoModelForCausalLM.from_pretrained(...)` and then `pack_model_e8(model)`. That ordering first materializes the complete dense checkpoint. Whole-matrix packing also creates full-matrix floating-point temporaries, and `inplace=False` can deep-copy the model. This route cannot establish a 14 GB checkpoint-ingestion claim.

### Recovered checkpoint-direct route

`ojasx2.streaming_e8` now implements the missing order:

1. Read the local HuggingFace config without network access.
2. Initialize parameters on `meta`, while Accelerate keeps small buffers real.
3. Apply the model's weight-tying policy and reject ambiguous quantization of shared weights.
4. Replace selected `nn.Linear` modules with real `E8LatticeLinear` modules before loading weights.
5. Read `safetensors` row slices and pack each slice directly into its final packed-code and scale buffers.
6. Assign other parameters shard by shard, restore dense-only ties, and fail if any meta tensor remains.
7. Freeze the base and leave only LoRA A/B trainable.
8. Emit a persistent-tensor ledger; this is deliberately not called RSS or training peak.

Executed qualification so far:

- 9 streaming tests pass.
- Packed codes and FP16 scales are byte-for-byte equal to the legacy whole-matrix packer for Root E8 (1.0-bit codes), ternary E8 (1.5-bit codes), and 2-bit E8, with Hadamard on and off.
- A real sharded Llama checkpoint loaded with no meta tensors and completed forward/backward with gradients for every adapter.
- Tied LM-head quantization fails closed; explicitly excluding the head preserves dense tying and executes.
- The run driver completed a separate all-layer tiny-model training step with nonzero adapter movement and an RSS trace.

These are tiny-model integration results. They are **not yet a 31B execution or quality result**.

## 2. Memory accounting that must not be omitted

For `N` quantized weights, code bitrate `b`, group size `g`, and FP16 scales, the pre-padding storage is

```text
B = N*b/8 + 2*N/g
effective bits/weight = b + 16/g
```

| Code format | Group size | Effective bits/weight | Packed bytes for 31B weights |
|---|---:|---:|---:|
| Root E8, 1.0-bit codes | 32 | 1.50 | 5.8125 GB (5.413 GiB) |
| Ternary E8, 1.5-bit codes | 32 | 2.00 | 7.7500 GB (7.218 GiB) |
| 2-bit E8 | 64 | 2.25 | 8.71875 GB (8.119 GiB) |
| 2-bit E8 | 32 | 2.50 | 9.6875 GB (9.021 GiB) |

These figures exclude padding where applicable and all of the following:

- unquantized embeddings, norms, biases, excluded/tied heads, and other parameters;
- LoRA A/B and any trainable dense modules;
- small but real ternary/root lookup tables and runtime buffers;
- gradients and optimizer moments for trainable tensors;
- activations, LoRA autograd tape, attention state, logits, and loss workspaces;
- E8 packing blocks and the cached ephemeral dequantization workspace;
- Python, HuggingFace, allocator, tokenizer, and runtime overhead.

The global E8 workspace is reused, which can bound peak growth, but it is not instantly returned to the allocator/OS. The existing test named `test_ephemeral_workspace_memory_reclaim` checks only output shape/device after five passes; it does not measure reclamation.

`resource.ru_maxrss` is an OS high-water mark, not a current or constant RSS sample. Linux reports KiB and macOS reports bytes. The corrected tools also sample RSS, but sampling alone can miss a transient; both values are recorded.

## 3. Mathematical and runtime ledger

### E8/ternary/root codebooks

The 2-bit packer uses a parity-constrained, scaled four-level half-integer construction. The ternary packer uses the 3,281 even-parity vectors in `D8 ∩ {-1,0,+1}^8`; its 12-bit rank is real compression, with small lookup tables. Root E8 uses a 256-direction byte codebook with an MMSE group scale.

E8's sphere-packing optimality does **not** by itself prove optimal finite-codebook quantization error, optimal LLM downstream quality, or superiority at a fixed total bitrate including scales. Those are separate theoretical and empirical questions.

The Hadamard path applies orthogonal 32-point transforms with the matching inverse/forward pair in the linear operation. Streaming packing reproduces this transform block-for-block and is tested for exact packed-output parity.

### Frozen base and LoRA

Packed codes/scales are buffers, not trainable parameters. The custom E8 operation computes input gradients while ordinary autograd handles LoRA A/B. This is the sound memory-saving separation: represented base weights do not need dense gradients or dense optimizer states. It does not imply full-base fine-tuning.

### Execution backends

There are materially different paths behind the same layer API:

- optional native CPU/MPS kernels decode weights into registers/local accumulators;
- TileLang kernels use on-chip shared tiles, including a dequantized shared tile, rather than the portable full-layer DRAM fallback;
- the portable path reconstructs a full layer or 1024-row output tiles and invokes ordinary `F.linear`.

TileLang/native compilation failure can silently return to the portable path. Therefore an import check or a layer type is not evidence of which backend executed. A per-run backend ledger and target-hardware numerical/memory tests remain necessary. “Zero DRAM weight materialization” is not a portable property.

### HF loss and attention integration

Most HF causal LMs compute loss internally from `labels`; they do not automatically call an `nn.CrossEntropyLoss` module replaced elsewhere. Loading E8 layers alone therefore leaves the stock HF logits/loss path. Likewise, the new loader deliberately leaves HF attention unchanged. The run report states both facts explicitly.

### In-backward optimizer

Applying per-parameter Adafactor updates in post-accumulate hooks can release gradient storage early and is central to the low-RAM design. Exact current-step global-norm clipping is a separate issue: the current implementation applies the previous latched global norm during the next backward. Until redesigned or explicitly reformulated, runs that require exact current-step clipping should disable it. The new driver uses `max_grad_norm=None` and disables moment quantization so weight-ingestion evidence is not confounded by those mechanisms.

## 4. Reproduce the new route

Install the optional local-checkpoint dependencies:

```bash
python -m pip install -e '.[streaming]'
```

Load and audit a complete local checkpoint without training:

```bash
python tools/run_streaming_e8_finetune.py \
  --checkpoint /path/to/local/model \
  --bitrate 1.5 --group-size 32 --lora-rank 8 \
  --memory-budget-gib 14 \
  --output reports/model-e8-load.json
```

Run one synthetic mechanics step (not quality evidence):

```bash
python tools/run_streaming_e8_finetune.py \
  --checkpoint /path/to/local/model \
  --bitrate 1.5 --group-size 32 --lora-rank 8 \
  --max-steps 1 --sequence-length 32 --allow-synthetic \
  --memory-budget-gib 14 \
  --output reports/model-e8-synthetic-step.json
```

Use `--text-file /path/to/shard.txt` for the real tokenization path. The JSON is rewritten after every step and records complete-model status, losses, observed gradients, adapter movement, stock-HF integration scope, and RSS. A process killed by the OS cannot write a final record; the last persisted status and external OS logs must then be retained.

Current loader limits are intentional: local safetensors only, HF causal-LM architectures, row-aligned selected `nn.Linear` matrices, no remote custom code, explicit handling of shared/tied weights, and CPU destination tensors. Unsupported layouts fail rather than silently allocating a dense matrix.

## 5. Evidence required for a defensible 31B result

1. Preserve the exact local config, shard filenames/sizes or hashes, source revision, command, dependency versions, OS/hardware, and physically installed RAM.
2. Start a cold process; avoid unrelated memory-heavy work.
3. Run load-only first. Verify that every checkpoint tensor is assigned or packed and no meta tensor remains.
4. Run one complete-model synthetic step, then real-data steps. Record external/OS memory evidence as well as the built-in sampler.
5. Report persistent packed/dense/adapter bytes separately from runtime RSS and optimizer/workspace allocations.
6. Record the actual E8 backend for forward and backward; do not infer it from availability.
7. Save/reload adapters and evaluate the resulting model, not merely the training loss on random targets.
8. Compare against strong equal-adapter/equal-data baselines: at least 4-bit LoRA and an appropriate 8-bit or FP16 baseline where memory permits.
9. Measure held-out perplexity and downstream tasks, throughput, and peak memory. Include quantization-only reconstruction error but do not substitute it for model quality.
10. Ablate code bitrate, group size, Hadamard rotation, layer sensitivity, dense vs quantized LM head, and moment quantization.

A user's separate 31B run should be documented with this evidence if its artifacts are available. The old repository script is not a substitute for those artifacts.

## 6. How this can become a general research contribution

The defensible novelty is not “E8 is the densest lattice, therefore 31B training is solved.” A stronger contribution would combine:

- **layout-preserving checkpoint-direct lattice ingestion** with bounded conversion work and no dense source model;
- **frozen lattice bases plus exact adapter gradients** and early-gradient-release optimization;
- **verified execution-dependent memory bounds**, distinguishing register, SRAM, cached-workspace, and full-materialization paths;
- **quality-aware mixed lattice allocation**, assigning code format/group size by measured layer sensitivity under a total memory budget;
- **reproducible all-layer evidence** across model families, data, and low-RAM systems against modern quantized-LoRA baselines.

The central research question is finite, group-scaled lattice distortion and downstream adaptation quality per *total* bit, not sphere packing alone. A useful theory would bound reconstruction/output error for the actual truncated codebooks and Hadamard-transformed weight distribution. A useful systems result would show that the complete model, not one layer, trains within an external memory budget while remaining quality-competitive.

That is a promising direction, but generality and breakthrough status must be earned through the comparisons above.

## 7. Remaining recovery work

- Add per-layer/per-call backend execution telemetry and honest workspace high-water accounting.
- Add a qualified portable/native execution strategy for full-model CPU and accelerator runs.
- Add adapter save/resume and post-training evaluation.
- Integrate streaming loss and attention only where the HF call graph actually supports them; verify semantics.
- Repair or explicitly redefine current-step clipping and test optimizer mathematics independently.
- Continue the semantic file-by-file ledger for root experiments, first-party math, formal claims, and generated/native code. Vendored upstream files must be distinguished from OjasX-authored mathematics rather than counted as reviewed by inheritance.
