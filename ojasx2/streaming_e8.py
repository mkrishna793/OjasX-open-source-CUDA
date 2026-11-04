"""Bounded, checkpoint-direct E8 packing for large frozen-backbone fine-tuning.

The historical ``pack_model_e8`` API is an *in-memory model transform*: the
source model already owns dense weights and the packer creates whole-matrix
intermediates.  This module implements the missing ingestion route instead:

``safetensors row block -> existing E8 packer -> packed destination -> release``.

No claim about a model's training memory follows from packing alone.  The load
report below therefore accounts for persistent tensors only and explicitly
excludes activations, autograd tape, gradients, optimizer state, workspaces,
allocators, and interpreter overhead.  Those need a separate runtime trace.

The first implementation intentionally supports row-aligned matrices.  This is
the layout used by the existing fused E8 GEMM and by common decoder LLMs.  An
unaligned matrix is rejected rather than silently changing the legacy packing
layout or allocating a whole dense weight tensor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fnmatch
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch
from torch import nn

from .math.lattice import (
    E8LatticeLinear,
    E8LatticePacker,
    RootE8LatticePacker,
    TernaryE8LatticePacker,
    fwht_block32,
)

_SUPPORTED_BITRATES = {1.0, 1.5, 2.0}
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamingE8LoadReport:
    """Persistent-tensor ledger for a completed checkpoint-direct load.

    ``estimated_persistent_tensor_bytes`` is not RSS and is not a training
    peak.  In particular, it excludes every runtime and optimizer allocation.
    """

    checkpoint_dir: str
    architectures: Tuple[str, ...]
    bitrate: float
    group_size: int
    lora_rank: int
    row_block: int
    checkpoint_files: Tuple[str, ...]
    quantized_linear_layers: int
    quantized_source_parameters: int
    packed_weight_bytes: int
    packed_scale_bytes: int
    parameter_bytes: int
    other_buffer_bytes: int
    estimated_persistent_tensor_bytes: int
    trainable_parameters: int
    trainable_parameter_bytes: int
    dense_source_parameters: int
    max_source_row_block_bytes: int
    ignored_checkpoint_keys: Tuple[str, ...]
    memory_scope: str = (
        "persistent tensors only; excludes RSS, Python/allocator overhead, "
        "activations, autograd tape, gradients, optimizer state, and workspaces"
    )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _packer_for_bitrate(bitrate: float):
    if bitrate == 2.0:
        return E8LatticePacker
    if bitrate == 1.5:
        return TernaryE8LatticePacker
    if bitrate == 1.0:
        return RootE8LatticePacker
    raise ValueError(f"Unsupported E8 bitrate {bitrate!r}; expected one of {sorted(_SUPPORTED_BITRATES)}")


def _validate_row_layout(
    out_features: int,
    in_features: int,
    bitrate: float,
    group_size: int,
) -> None:
    _packer_for_bitrate(bitrate)
    if out_features <= 0 or in_features <= 0:
        raise ValueError("Streaming E8 requires non-empty two-dimensional weights")
    # All three current packers internally form 32-coordinate Hadamard/block
    # groups, even when the optional multiplicative Hadamard rotation is off.
    if group_size <= 0 or group_size % 32 != 0:
        raise ValueError("The current E8 packers require group_size to be a positive multiple of 32")
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} is not divisible by group_size={group_size}. "
            "Checkpoint-direct row packing would cross a scale-group boundary and "
            "change the legacy layout; use a compatible group size or the explicit "
            "whole-matrix conversion path."
        )


def _pack_row_block(
    block: torch.Tensor,
    bitrate: float,
    group_size: int,
    use_hadamard: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    packer = _packer_for_bitrate(bitrate)
    # Match E8LatticeLinear.from_linear: the inverse/forward Hadamard pair is
    # part of the linear layer's quantization transform, not a packer kwarg.
    source = fwht_block32(block) if use_hadamard else block
    packed, scales, _ = packer.pack(source, group_size=group_size)
    return packed, scales


def pack_weight_rowwise(
    weight: torch.Tensor,
    *,
    bitrate: float = 1.5,
    group_size: int = 32,
    row_block: int = 32,
    use_hadamard: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack an already materialized matrix with bounded packing intermediates.

    This helper is useful for migration and exact-parity tests.  It does not
    make the input matrix smaller and is not the checkpoint-direct loader.
    The returned byte/scales layout is identical to the corresponding
    whole-matrix packer whenever the row-alignment contract holds.
    """

    if weight.ndim != 2:
        raise ValueError(f"Expected a rank-2 weight, got shape {tuple(weight.shape)}")
    if row_block <= 0:
        raise ValueError("row_block must be positive")
    out_features, in_features = map(int, weight.shape)
    _validate_row_layout(out_features, in_features, bitrate, group_size)

    packed_parts: List[torch.Tensor] = []
    scale_parts: List[torch.Tensor] = []
    for start in range(0, out_features, row_block):
        block = weight[start : start + row_block]
        packed, scales = _pack_row_block(block, bitrate, group_size, use_hadamard)
        packed_parts.append(packed)
        scale_parts.append(scales)
    # These outputs are the required persistent packed representation, not a
    # dense reconstruction.  The loader below avoids even this output list by
    # copying each block directly into its final destination.
    return torch.cat(packed_parts), torch.cat(scale_parts)


def _copy_row_block_into_layer(
    layer: E8LatticeLinear,
    row_start: int,
    block: torch.Tensor,
    use_hadamard: bool,
) -> int:
    if block.ndim != 2 or block.shape[1] != layer.in_features:
        raise ValueError(
            f"Unexpected row block shape {tuple(block.shape)} for "
            f"{layer.out_features}x{layer.in_features} E8 layer"
        )
    if not block.dtype.is_floating_point:
        raise TypeError(f"E8 packing requires floating-point source weights, got {block.dtype}")
    rows = int(block.shape[0])
    packed, scales = _pack_row_block(
        block,
        float(layer.bitrate),
        int(layer.group_size),
        use_hadamard,
    )
    packed_per_row = packed.numel() // rows
    scales_per_row = scales.numel() // rows
    packed_begin = row_start * packed_per_row
    scales_begin = row_start * scales_per_row
    if packed_begin + packed.numel() > layer.packed_weight.numel():
        raise RuntimeError("Packed destination is smaller than the streamed source layout")
    if scales_begin + scales.numel() > layer.block_scales.numel():
        raise RuntimeError("Scale destination is smaller than the streamed source layout")
    with torch.no_grad():
        layer.packed_weight[packed_begin : packed_begin + packed.numel()].copy_(packed)
        layer.block_scales[scales_begin : scales_begin + scales.numel()].copy_(scales)
    return int(block.numel() * block.element_size())


def _stream_safetensor_weight(
    handle: object,
    key: str,
    layer: E8LatticeLinear,
    row_block: int,
    use_hadamard: bool,
) -> int:
    source = handle.get_slice(key)
    shape = tuple(map(int, source.get_shape()))
    expected = (int(layer.out_features), int(layer.in_features))
    if shape != expected:
        raise ValueError(f"Checkpoint tensor {key!r} has shape {shape}, expected {expected}")
    max_source_bytes = 0
    for start in range(0, expected[0], row_block):
        stop = min(start + row_block, expected[0])
        # PySafeSlice realizes only this row block as a Torch tensor.
        block = source[start:stop]
        source_bytes = _copy_row_block_into_layer(layer, start, block, use_hadamard)
        max_source_bytes = max(max_source_bytes, source_bytes)
        del block
    return max_source_bytes


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _set_named_parameter(model: nn.Module, name: str, parameter: nn.Parameter) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], parameter)


def _checkpoint_files(checkpoint_dir: Path) -> Tuple[Path, ...]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty weight_map in {index_path}")
        names = sorted({str(filename) for filename in weight_map.values()})
        files = tuple(checkpoint_dir / filename for filename in names)
    elif (checkpoint_dir / "model.safetensors").is_file():
        files = (checkpoint_dir / "model.safetensors",)
    else:
        files = tuple(sorted(checkpoint_dir.glob("*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint found under {checkpoint_dir}")
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(f"Checkpoint index references missing file: {file}")
        if file.resolve().parent != checkpoint_dir.resolve():
            raise ValueError("Checkpoint shard paths must remain inside the checkpoint directory")
    return files


def load_hf_checkpoint_e8(
    checkpoint_dir: str,
    *,
    bitrate: float = 1.5,
    group_size: int = 32,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    row_block: int = 32,
    use_hadamard: bool = False,
    adapter_dtype: torch.dtype = torch.float32,
    exclude_modules: Iterable[str] = (),
    allow_unexpected_keys: bool = False,
) -> Tuple[nn.Module, StreamingE8LoadReport]:
    """Construct and populate a HF causal LM without materializing dense linears.

    Parameters are initially meta tensors; small buffers are initialized on CPU.
    Every selected ``nn.Linear`` is replaced by a real packed E8 module before
    checkpoint ingestion.  Source weights are read in row blocks and copied
    directly into packed storage.  Non-quantized parameters are assigned one
    shard at a time.  All base parameters are frozen; only E8 LoRA A/B tensors
    are left trainable.

    Weight tying/shared linear weights fail closed when a selected member would
    be quantized independently.  For a tied dense LM head, explicitly exclude
    ``lm_head`` and accept the accounted dense-head memory instead.  Missing,
    duplicate, unexpected, or residual-meta tensors are errors by default.

    This function requires the optional ``transformers``, ``accelerate``, and
    ``safetensors`` dependencies.  It performs no network access: the config
    and all shards must already exist in ``checkpoint_dir``.
    """

    if row_block <= 0:
        raise ValueError("row_block must be positive")
    if lora_rank < 0:
        raise ValueError("lora_rank must be non-negative")
    _packer_for_bitrate(bitrate)
    if group_size <= 0 or group_size % 32 != 0:
        raise ValueError("group_size must be a positive multiple of 32")
    root = Path(checkpoint_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Local checkpoint directory does not exist: {root}")

    try:
        from accelerate import init_empty_weights
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "Checkpoint-direct E8 loading requires the 'streaming' optional dependencies: "
            "transformers, accelerate, and safetensors"
        ) from exc

    files = _checkpoint_files(root)
    config = AutoConfig.from_pretrained(
        str(root),
        local_files_only=True,
        trust_remote_code=False,
    )
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    # Some HF meta-initialization paths defer tying until weight load.  Apply
    # the architecture's own policy now so shared weights are visible to the
    # safety check and can be restored after shard-wise assignment.
    model.tie_weights()

    exclusions = tuple(exclude_modules)
    linear_modules: List[Tuple[str, nn.Linear]] = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    selected_names = {
        name for name, _ in linear_modules if not _matches_any(name, exclusions)
    }
    selected_weight_keys = {f"{name}.weight" for name in selected_names}

    parameter_owners: Dict[int, List[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parameter_owners.setdefault(id(parameter), []).append(name)
    shared_groups = [names for names in parameter_owners.values() if len(names) > 1]
    for names in shared_groups:
        if any(name in selected_weight_keys for name in names):
            raise ValueError(
                "A selected E8 linear shares its weight with another module: "
                + ", ".join(names)
                + ". Independent packing would break weight tying. Exclude the tied "
                "head/module explicitly, or add a separately validated tying policy."
            )

    weight_to_layer: Dict[str, E8LatticeLinear] = {}
    bias_to_layer: Dict[str, E8LatticeLinear] = {}
    quantized_source_parameters = 0

    def replace_children(parent: nn.Module, prefix: str = "") -> None:
        nonlocal quantized_source_parameters
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                if full_name not in selected_names:
                    continue
                _validate_row_layout(
                    int(child.out_features),
                    int(child.in_features),
                    float(bitrate),
                    int(group_size),
                )
                layer = E8LatticeLinear(
                    int(child.in_features),
                    int(child.out_features),
                    bias=child.bias is not None,
                    rank=int(lora_rank),
                    lora_alpha=float(lora_alpha),
                    bitrate=float(bitrate),
                    group_size=int(group_size),
                    use_hadamard=bool(use_hadamard),
                    device=torch.device("cpu"),
                    dtype=adapter_dtype,
                )
                setattr(parent, child_name, layer)
                weight_to_layer[f"{full_name}.weight"] = layer
                if layer.bias is not None:
                    layer.bias.requires_grad_(False)
                    bias_to_layer[f"{full_name}.bias"] = layer
                quantized_source_parameters += int(child.in_features * child.out_features)
            else:
                replace_children(child, full_name)

    replace_children(model)
    if not weight_to_layer:
        raise ValueError("No eligible nn.Linear modules were selected for E8 conversion")

    known_state_keys = set(model.state_dict())
    converted_state_keys = {
        f"{weight_key[:-7]}.{suffix}"
        for weight_key in weight_to_layer
        for suffix in ("packed_weight", "block_scales", "lora_A", "lora_B")
    }
    shared_state_keys = {name for names in shared_groups for name in names}
    # Real persistent buffers cannot be detected by a residual-meta check, so
    # every original dense state key needs explicit checkpoint coverage. Tied
    # aliases are validated as a group below because safetensors may store only
    # one member of the group.
    expected_dense_keys = known_state_keys - converted_state_keys - shared_state_keys
    expected_packed_keys = set(weight_to_layer) | set(bias_to_layer)
    seen: Set[str] = set()
    ignored: List[str] = []
    max_source_row_block_bytes = 0
    dense_source_parameters = 0

    for file in files:
        _LOGGER.info("Streaming checkpoint shard %s", file.name)
        dense: Dict[str, torch.Tensor] = {}
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in seen:
                    raise ValueError(f"Duplicate checkpoint tensor across shards: {key}")
                seen.add(key)
                if key in weight_to_layer:
                    source_bytes = _stream_safetensor_weight(
                        handle,
                        key,
                        weight_to_layer[key],
                        int(row_block),
                        bool(use_hadamard),
                    )
                    max_source_row_block_bytes = max(max_source_row_block_bytes, source_bytes)
                    _LOGGER.info(
                        "Packed %s directly from at most %d source rows per block",
                        key,
                        row_block,
                    )
                elif key in bias_to_layer:
                    layer = bias_to_layer[key]
                    tensor = handle.get_tensor(key)
                    if tuple(tensor.shape) != (int(layer.out_features),):
                        raise ValueError(f"Unexpected bias shape for {key}: {tuple(tensor.shape)}")
                    with torch.no_grad():
                        layer.bias.copy_(tensor)
                    dense_source_parameters += int(tensor.numel())
                    del tensor
                elif key in known_state_keys:
                    dense[key] = handle.get_tensor(key)
                    dense_source_parameters += int(dense[key].numel())
                elif allow_unexpected_keys:
                    ignored.append(key)
                else:
                    raise ValueError(
                        f"Unexpected checkpoint tensor {key!r} in {file.name}; "
                        "refusing a silent partial load"
                    )
        if dense:
            # assign=True replaces meta parameters with the checkpoint-owned
            # CPU tensors instead of allocating a second destination copy.
            dense_count = len(dense)
            model.load_state_dict(dense, strict=False, assign=True)
            dense.clear()
            _LOGGER.info("Assigned %d non-quantized tensors from %s", dense_count, file.name)

    missing_quantized = sorted(expected_packed_keys - seen)
    if missing_quantized:
        raise ValueError(
            "Checkpoint is missing tensors required by converted E8 layers: "
            + ", ".join(missing_quantized[:20])
        )
    missing_dense = sorted(expected_dense_keys - seen)
    if missing_dense:
        raise ValueError(
            "Checkpoint is missing required dense parameters or persistent buffers: "
            + ", ".join(missing_dense[:20])
        )
    for names in shared_groups:
        if not any(name in seen for name in names):
            raise ValueError(
                "Checkpoint contains no stored tensor for shared parameter group: "
                + ", ".join(names)
            )

    # Re-establish dense-only ties after shard-wise assignment.  Safetensors
    # commonly stores just one member of a tied pair.
    for names in shared_groups:
        candidates = []
        for name in names:
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            candidates.append(getattr(parent, parts[-1]))
        real = next((parameter for parameter in candidates if not parameter.is_meta), None)
        if real is None:
            continue  # The residual-meta check below gives the final error.
        for name in names:
            _set_named_parameter(model, name, real)

    meta_tensors = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    meta_tensors.extend(name for name, buffer in model.named_buffers() if buffer.is_meta)
    if meta_tensors:
        raise ValueError(
            "Checkpoint load left uninitialized meta tensors: "
            + ", ".join(meta_tensors[:30])
        )

    # Assignment and shared-weight restoration must not silently make the base
    # trainable.  Re-apply the training contract after all loading is complete.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in weight_to_layer.values():
        if layer.lora_A.numel():
            layer.lora_A.requires_grad_(True)
            layer.lora_B.requires_grad_(True)

    parameter_bytes = sum(
        int(parameter.numel() * parameter.element_size())
        for parameter in model.parameters()
    )
    trainable_parameters = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    trainable_parameter_bytes = sum(
        int(parameter.numel() * parameter.element_size())
        for parameter in model.parameters() if parameter.requires_grad
    )
    packed_weight_bytes = sum(int(layer.packed_weight.numel()) for layer in weight_to_layer.values())
    packed_scale_bytes = sum(
        int(layer.block_scales.numel() * layer.block_scales.element_size())
        for layer in weight_to_layer.values()
    )
    packed_buffer_ids = {
        id(buffer)
        for layer in weight_to_layer.values()
        for buffer in (layer.packed_weight, layer.block_scales)
    }
    other_buffer_bytes = sum(
        int(buffer.numel() * buffer.element_size())
        for buffer in model.buffers()
        if id(buffer) not in packed_buffer_ids
    )
    report = StreamingE8LoadReport(
        checkpoint_dir=str(root),
        architectures=tuple(getattr(config, "architectures", ()) or (type(model).__name__,)),
        bitrate=float(bitrate),
        group_size=int(group_size),
        lora_rank=int(lora_rank),
        row_block=int(row_block),
        checkpoint_files=tuple(file.name for file in files),
        quantized_linear_layers=len(weight_to_layer),
        quantized_source_parameters=quantized_source_parameters,
        packed_weight_bytes=packed_weight_bytes,
        packed_scale_bytes=packed_scale_bytes,
        parameter_bytes=parameter_bytes,
        other_buffer_bytes=other_buffer_bytes,
        estimated_persistent_tensor_bytes=(
            parameter_bytes + packed_weight_bytes + packed_scale_bytes + other_buffer_bytes
        ),
        trainable_parameters=trainable_parameters,
        trainable_parameter_bytes=trainable_parameter_bytes,
        dense_source_parameters=dense_source_parameters,
        max_source_row_block_bytes=max_source_row_block_bytes,
        ignored_checkpoint_keys=tuple(sorted(ignored)),
    )
    # A non-tensor attribute is safe for state_dict and keeps the provenance
    # alongside the loaded model for benchmark/report drivers.
    model.ojasx_streaming_report = report
    return model, report


__all__ = [
    "StreamingE8LoadReport",
    "load_hf_checkpoint_e8",
    "pack_weight_rowwise",
]
