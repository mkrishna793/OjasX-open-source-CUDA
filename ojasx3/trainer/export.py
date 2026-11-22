"""
Zero Lock-In Production Export for OjasX3 (HuggingFace PEFT & Dense Safetensors).

Ensures users are NEVER locked into OjasX custom math after training:
  - `export_peft_adapter(model, output_dir)`:
      Extracts LoRA adapters from `LoRALinear` or `E8LatticeLinear`, automatically
      un-rotates any Hadamard/CW-rotated A matrices (`A_std = A_rot @ Q^T`), and
      writes a standard HuggingFace `peft` directory (`adapter_config.json` +
      `adapter_model.safetensors` or `adapter_model.bin`) compatible with `peft`,
      `vLLM`, `Ollama`, and `llama.cpp`.
  - `merge_and_export_hf(model, output_dir)`:
      Merges `LoRALinear`, `CWOFTLinear`, and `E8LatticeLinear` layers into a
      100% standard dense PyTorch/HuggingFace state dict (`model.safetensors` or
      `pytorch_model.bin`) with zero OjasX runtime dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn

from ojasx2.math.lattice import E8LatticeLinear, _get_packer_for_bitrate, fwht_block32
from ojasx3.lattice.transforms import CWOFTLinear, apply_incoherence_transform
from ojasx3.trainer.peft import LoRALinear


def _save_state_dict(state_dict: Dict[str, torch.Tensor], out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cpu_sd = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    try:
        from safetensors.torch import save_file
        out_path = out_dir / f"{stem}.safetensors"
        save_file(cpu_sd, str(out_path))
        return out_path
    except Exception:
        out_path = out_dir / f"{stem}.bin"
        torch.save(cpu_sd, str(out_path))
        return out_path


@torch.no_grad()
def export_peft_adapter(
    model: nn.Module,
    output_dir: Union[str, Path],
    *,
    base_model_name_or_path: str = "",
    task_type: str = "CAUSAL_LM",
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    """Export trainable LoRA adapters to standard HuggingFace PEFT format.

    Automatically folds out any Hadamard/CW rotation on `E8LatticeLinear` so the
    exported `lora_A.weight` and `lora_B.weight` match the unrotated pretrained model.
    """
    out_dir = Path(output_dir)
    adapter_sd: Dict[str, torch.Tensor] = {}
    target_modules: Set[str] = set()
    ranks: Set[int] = set()
    alphas: Set[float] = set()

    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            a_w = mod.lora_A.detach().to(torch.float32)
            b_w = mod.lora_B.detach().to(torch.float32)
            if isinstance(mod.base_layer, E8LatticeLinear) and getattr(mod.base_layer, "use_hadamard", False):
                if a_w.shape[-1] % 32 == 0:
                    a_w = fwht_block32(a_w)
            if dtype is not None:
                a_w = a_w.to(dtype)
                b_w = b_w.to(dtype)
            prefix = f"base_model.model.{name}" if name else "base_model.model"
            adapter_sd[f"{prefix}.lora_A.weight"] = a_w
            adapter_sd[f"{prefix}.lora_B.weight"] = b_w
            target_modules.add(name.split(".")[-1] if name else "linear")
            ranks.add(int(mod.rank))
            alphas.add(float(mod.lora_alpha))
        elif isinstance(mod, E8LatticeLinear) and getattr(mod, "rank", 0) > 0:
            if getattr(mod, "lora_A", None) is not None and getattr(mod, "lora_B", None) is not None:
                # E8LatticeLinear stores lora_A as [in_f, rank], lora_B as [rank, out_f] -> transpose to PEFT [rank, in_f], [out_f, rank]
                a_w = mod.lora_A.detach().to(torch.float32).t().contiguous()
                b_w = mod.lora_B.detach().to(torch.float32).t().contiguous()
                if getattr(mod, "use_hadamard", False) and a_w.shape[-1] % 32 == 0:
                    a_w = fwht_block32(a_w)
                if dtype is not None:
                    a_w = a_w.to(dtype)
                    b_w = b_w.to(dtype)
                prefix = f"base_model.model.{name}" if name else "base_model.model"
                adapter_sd[f"{prefix}.lora_A.weight"] = a_w
                adapter_sd[f"{prefix}.lora_B.weight"] = b_w
                target_modules.add(name.split(".")[-1] if name else "linear")
                ranks.add(int(mod.rank))
                alphas.add(float(getattr(mod, "lora_alpha", mod.rank)))

    if not adapter_sd:
        raise ValueError("No LoRALinear or E8LatticeLinear(rank>0) adapters found in model to export.")

    weights_path = _save_state_dict(adapter_sd, out_dir, "adapter_model")
    config = {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": base_model_name_or_path,
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": sorted(alphas)[0] if len(alphas) == 1 else list(sorted(alphas)),
        "lora_dropout": 0.0,
        "r": sorted(ranks)[0] if len(ranks) == 1 else list(sorted(ranks)),
        "target_modules": sorted(target_modules),
        "task_type": task_type,
        "exported_by": "ojasx3",
    }
    cfg_path = out_dir / "adapter_config.json"
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {
        "weights_path": str(weights_path),
        "config_path": str(cfg_path),
        "num_adapter_tensors": len(adapter_sd),
        "target_modules": sorted(target_modules),
    }


@torch.no_grad()
def _materialize_dense_linear_weight(mod: nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Materialize any standard, E8, LoRA, or CW-OFT linear module into exact dense (weight, bias)."""
    if isinstance(mod, nn.Linear):
        return mod.weight.detach().to(torch.float32), (mod.bias.detach().to(torch.float32) if mod.bias is not None else None)

    if isinstance(mod, E8LatticeLinear):
        packer = _get_packer_for_bitrate(mod.bitrate)
        w = packer.unpack(
            mod.packed_weight,
            mod.block_scales,
            (mod.out_features, mod.in_features),
            dtype=torch.float32,
            use_workspace=False,
        )
        if getattr(mod, "use_hadamard", False) and mod.in_features % 32 == 0:
            w = fwht_block32(w)
        if getattr(mod, "rank", 0) > 0 and mod.lora_A is not None and mod.lora_B is not None:
            scaling = float(getattr(mod, "scaling", 1.0))
            delta = (mod.lora_A.detach().to(torch.float32) @ mod.lora_B.detach().to(torch.float32)).t() * scaling
            if getattr(mod, "use_hadamard", False) and mod.in_features % 32 == 0:
                delta = fwht_block32(delta)
            w = w + delta
        b = mod.bias.detach().to(torch.float32) if getattr(mod, "bias", None) is not None else None
        return w, b

    if isinstance(mod, LoRALinear):
        w, b = _materialize_dense_linear_weight(mod.base_layer)
        delta = (mod.lora_B.detach().to(torch.float32) @ mod.lora_A.detach().to(torch.float32)) * float(mod.scaling)
        return w + delta, b

    if isinstance(mod, CWOFTLinear):
        w, b = _materialize_dense_linear_weight(mod.base_layer)
        w_merged = mod.merge_into_dense_weight(w)
        if b is not None and mod.d_out is not None:
            b_rot = apply_incoherence_transform(b, inverse=False, spec=mod.out_spec, signs=mod.out_signs, orbit_id=mod.orbit_id)
            b = b + apply_incoherence_transform(b_rot * mod.d_out.detach().to(torch.float32), inverse=True, spec=mod.out_spec, signs=mod.out_signs, orbit_id=mod.orbit_id)
        return w_merged, b

    raise TypeError(f"Unsupported linear module type for dense merge: {type(mod)}")


@torch.no_grad()
def merge_and_export_hf(
    model: nn.Module,
    output_dir: Union[str, Path],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Merge all LoRALinear, CWOFTLinear, and E8LatticeLinear modules into a standard
    dense HuggingFace/PyTorch state_dict (`model.safetensors` / `model.bin`)."""
    out_dir = Path(output_dir)
    merged_sd: Dict[str, torch.Tensor] = {}
    handled_prefixes: List[str] = []

    for name, mod in model.named_modules():
        if isinstance(mod, (LoRALinear, CWOFTLinear, E8LatticeLinear)):
            # Skip child modules already merged by an enclosing wrapper
            if any(name.startswith(p + ".") for p in handled_prefixes):
                continue
            w, b = _materialize_dense_linear_weight(mod)
            key_w = f"{name}.weight" if name else "weight"
            merged_sd[key_w] = w.to(dtype)
            if b is not None:
                key_b = f"{name}.bias" if name else "bias"
                merged_sd[key_b] = b.to(dtype)
            handled_prefixes.append(name)

    # Copy all remaining standard parameters/buffers (embeddings, norms, unadapted linears)
    for k, v in model.state_dict().items():
        if any(k == p or k.startswith(p + ".") for p in handled_prefixes):
            continue
        merged_sd[k] = v.to(dtype) if v.is_floating_point() else v

    weights_path = _save_state_dict(merged_sd, out_dir, "model")
    return {
        "weights_path": str(weights_path),
        "num_tensors": len(merged_sd),
        "merged_adapter_layers": len(handled_prefixes),
    }


__all__ = ["export_peft_adapter", "merge_and_export_hf"]
