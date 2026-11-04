# Copyright (C) 2026 Ojas-Arya Open Foundation & Antigravity Prime Architecture
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)

"""Execute ONE representative 533M-parameter E8/LoRA linear layer.

The 60-layer figures printed here are arithmetic extrapolations only.  This
script does not instantiate, load, forward, or train a 32B model.  For the
checkpoint-direct all-model route, use ``tools/run_streaming_e8_finetune.py``
and retain its JSON/RSS evidence.  The historical function name is kept for
command compatibility, not as a claim of full-model verification.
"""

from __future__ import annotations

import resource
import sys
import time
from typing import Dict

import torch
import torch.nn.functional as F

from ojasx2.grad_free_optimizer import HolonomicGradFreeOptimizer
from ojasx2.math.lattice import E8LatticeLinear


def get_process_peak_rss_gib() -> float:
    """Return the OS process high-water RSS in GiB (not a current RSS sample)."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux and the other supported Unix platforms report KiB; macOS reports
    # bytes.  Dividing a Linux value as bytes would inflate it by 1024.
    bytes_value = value if sys.platform == "darwin" else value * 1024.0
    return bytes_value / (1024 ** 3)


def print_banner(text: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def run_32b_training_benchmark() -> Dict[str, object]:
    """Run the representative-layer check and return a machine-readable scope."""

    num_layers = 60
    in_dim = 16384
    out_dim = 32552
    group_size = 64
    bitrate = 2.0
    lora_rank = 8
    params_per_layer = in_dim * out_dim
    extrapolated_params = num_layers * params_per_layer
    effective_bits = bitrate + 16.0 / group_size

    print_banner("OJASX E8 REPRESENTATIVE-LAYER TRAINING CHECK (NOT FULL 32B)")
    print(f"Platform              : {sys.platform}")
    print(f"PyTorch               : {torch.__version__}")
    print(f"OS peak RSS at start  : {get_process_peak_rss_gib():.3f} GiB")
    print("\n[EXTRAPOLATED ARCHITECTURE — NOT INSTANTIATED]")
    print(f"  Layers represented  : {num_layers}")
    print(f"  Actual layer shape  : {in_dim} x {out_dim}")
    print(f"  Params per layer    : {params_per_layer:,}")
    print(f"  Extrapolated params : {extrapolated_params:,}")
    print(f"  Code + FP16 scales  : {effective_bits:.3f} bits/weight before padding")

    fp16_gib = extrapolated_params * 2 / (1024 ** 3)
    packed_gib = extrapolated_params * effective_bits / 8 / (1024 ** 3)
    print("\n[ARITHMETIC STORAGE COMPARISON — NOT RSS]")
    print(f"  Extrapolated FP16 weights        : {fp16_gib:.3f} GiB")
    print(f"  Extrapolated packed codes+scales : {packed_gib:.3f} GiB")

    print_banner("STEP 1: INSTANTIATE ONE FULL-SCALE 533M LAYER")
    start = time.perf_counter()
    layer = E8LatticeLinear(
        in_features=in_dim,
        out_features=out_dim,
        bias=False,
        rank=lora_rank,
        lora_alpha=16.0,
        group_size=group_size,
        bitrate=bitrate,
    )
    allocation_seconds = time.perf_counter() - start

    packed_bytes = int(layer.packed_weight.numel())
    scale_bytes = int(layer.block_scales.numel() * layer.block_scales.element_size())
    layer_bytes = packed_bytes + scale_bytes
    reduction = (params_per_layer * 2) / layer_bytes
    print(f"  Allocation time          : {allocation_seconds:.3f} s")
    print(f"  Actual packed code bytes : {packed_bytes:,}")
    print(f"  Actual FP16 scale bytes  : {scale_bytes:,}")
    print(f"  Actual layer storage     : {layer_bytes / 1024**2:.3f} MiB")
    print(f"  FP16/packed storage ratio: {reduction:.4f}x (not 8x with these scales)")
    print(f"  60x layer extrapolation  : {layer_bytes * num_layers / 1024**3:.3f} GiB")
    print(f"  OS process peak RSS      : {get_process_peak_rss_gib():.3f} GiB")

    trainable = [parameter for parameter in layer.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    print_banner("STEP 2: ATTACH IN-BACKWARD ADAFACTOR TO LORA ONLY")
    print(f"  Trainable adapter parameters: {trainable_count:,}")
    print(f"  Frozen represented base parameters: {params_per_layer:,}")
    optimizer = HolonomicGradFreeOptimizer(trainable, lr=0.01, algorithm="adafactor")

    torch.manual_seed(42)
    batch_size, seq_len = 2, 4
    inputs = torch.randn(batch_size, seq_len, in_dim)
    target = torch.randn(batch_size, seq_len, out_dim)
    initial_a = layer.lora_A.detach().clone()
    initial_b = layer.lora_B.detach().clone()
    observed_grad_norms: Dict[str, float] = {}

    def observe_grad(name: str):
        def hook(gradient: torch.Tensor) -> None:
            observed_grad_norms[name] = float(gradient.norm())
        return hook

    handles = [
        layer.lora_A.register_hook(observe_grad("lora_A")),
        layer.lora_B.register_hook(observe_grad("lora_B")),
    ]

    losses = []
    cosine_similarities = []
    deltas_a = []
    deltas_b = []
    print_banner("STEP 3: EXECUTE THREE REPRESENTATIVE-LAYER STEPS")
    for step in range(1, 4):
        step_start = time.perf_counter()
        output = layer(inputs)
        loss = F.mse_loss(output, target)
        cosine = float(F.cosine_similarity(output.flatten(), target.flatten(), dim=0).detach())
        optimizer.zero_grad()
        loss.backward()  # Registered post-accumulate hooks apply the update.
        delta_a = float((layer.lora_A.detach() - initial_a).norm())
        delta_b = float((layer.lora_B.detach() - initial_b).norm())
        losses.append(float(loss.detach()))
        cosine_similarities.append(cosine)
        deltas_a.append(delta_a)
        deltas_b.append(delta_b)
        print(
            f"  step={step} loss={float(loss):.6f} cosine={cosine:.6f} "
            f"delta_A={delta_a:.8f} delta_B={delta_b:.8f} "
            f"seconds={time.perf_counter() - step_start:.3f} "
            f"os_peak_rss_gib={get_process_peak_rss_gib():.3f}"
        )
    for handle in handles:
        handle.remove()

    checks = {
        "lora_A_moved": deltas_a[-1] > 0.0,
        "lora_B_moved": deltas_b[-1] > 0.0,
        "nonzero_finite_gradients": all(
            observed_grad_norms.get(name, 0.0) > 0.0
            and torch.isfinite(torch.tensor(observed_grad_norms[name])).item()
            for name in ("lora_A", "lora_B")
        ),
        "loss_decreased": losses[-1] < losses[0],
        "cosine_increased": cosine_similarities[-1] > cosine_similarities[0],
    }
    print_banner("STEP 4: REPORT CHECKS WITHOUT CHANGING THEIR SCOPE")
    for name, passed in checks.items():
        print(f"  {name}: {passed}")
    print(f"  Observed gradient norms: {observed_grad_norms}")
    print(f"  Final OS high-water RSS: {get_process_peak_rss_gib():.3f} GiB")
    print("  RSS high-water is not a time series and does not prove a constant footprint.")

    # These are the mechanical training requirements.  Three steps on one
    # random target are not a convergence or model-quality experiment.
    assert checks["lora_A_moved"] and checks["lora_B_moved"]
    assert checks["nonzero_finite_gradients"]
    print_banner("VERDICT: ONE 533M E8/LORA LAYER EXECUTED; FULL 32B MODEL NOT TESTED")
    return {
        "scope": "one_representative_e8_lora_linear_layer",
        "full_32b_model_verified": False,
        "instantiated_linear_layers": 1,
        "extrapolated_linear_layers": num_layers,
        "actual_layer_parameters": params_per_layer,
        "extrapolated_parameters": extrapolated_params,
        "effective_bits_per_weight_before_padding": effective_bits,
        "actual_packed_and_scale_bytes": layer_bytes,
        "checks": checks,
        "losses": losses,
        "cosine_similarities": cosine_similarities,
        "observed_gradient_norms": observed_grad_norms,
        "os_peak_rss_gib": get_process_peak_rss_gib(),
    }


if __name__ == "__main__":
    run_32b_training_benchmark()
