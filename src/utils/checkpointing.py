import os
from typing import Any, Dict, Iterable, Optional

import torch


def _unwrap_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    Extract the actual model state_dict from common checkpoint formats.

    Supports formats like:
      - raw state_dict
      - {"state_dict": ...}
      - {"model": ...}
      - {"encoder": ...}
      - {"target_encoder": ...}
      - {"student": ...}
      - {"teacher": ...}
      - {"module": ...}
    """
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected checkpoint to be a dict-like object, got {type(ckpt)}")

    preferred_keys = [
        "state_dict",
        "model",
        "encoder",
        "target_encoder",
        "student",
        "teacher",
        "module",
    ]

    for key in preferred_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]

    if all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt

    raise KeyError(
        "Could not find a valid state_dict in checkpoint. "
        f"Available top-level keys: {list(ckpt.keys())}"
    )


def _strip_prefixes(
    state_dict: Dict[str, torch.Tensor],
    prefixes: Iterable[str],
) -> Dict[str, torch.Tensor]:
    """
    Strip any matching prefix repeatedly from keys.

    Example:
      module.backbone.blocks.0.attn.qkv.weight
      -> blocks.0.attn.qkv.weight
    """
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def _filter_state_dict_for_model(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Keep only keys that exist in the model and have matching shapes.
    """
    model_state = model.state_dict()
    filtered = {}

    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered[key] = value

    return filtered


def inspect_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Inspect checkpoint structure without loading into a model.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location)
    info: Dict[str, Any] = {
        "path": path,
        "type": type(ckpt).__name__,
    }

    if isinstance(ckpt, dict):
        info["top_level_keys"] = list(ckpt.keys())
        try:
            sd = _unwrap_state_dict(ckpt)
            sample_keys = list(sd.keys())[:20]
            info["state_dict_num_keys"] = len(sd)
            info["state_dict_sample_keys"] = sample_keys
        except Exception as exc:
            info["state_dict_error"] = str(exc)

    return info


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    map_location: str = "cpu",
    strict: bool = False,
    strip_prefixes: Optional[Iterable[str]] = None,
    filter_mismatched: bool = True,
) -> Dict[str, Any]:
    """
    Robust checkpoint loader for DINOv2 / V-JEPA2 style checkpoints.

    Args:
        model: Target model.
        path: Checkpoint path.
        map_location: torch.load map_location.
        strict: Passed to model.load_state_dict.
        strip_prefixes: Prefixes to remove from checkpoint keys.
        filter_mismatched: If True, only load keys present in the model
            with matching tensor shapes.

    Returns:
        Dict containing missing keys, unexpected keys, and load metadata.
    """
    if strip_prefixes is None:
        strip_prefixes = (
            "module.",
            "model.",
            "backbone.",
            "encoder.",
            "target_encoder.",
            "student.",
            "teacher.",
        )

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location)
    raw_state_dict = _unwrap_state_dict(ckpt)
    cleaned_state_dict = _strip_prefixes(raw_state_dict, strip_prefixes)

    if filter_mismatched:
        final_state_dict = _filter_state_dict_for_model(model, cleaned_state_dict)
    else:
        final_state_dict = cleaned_state_dict

    incompatible = model.load_state_dict(final_state_dict, strict=strict)

    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    loaded_keys = list(final_state_dict.keys())
    skipped_keys = [k for k in cleaned_state_dict.keys() if k not in final_state_dict]

    return {
        "missing": missing_keys,
        "unexpected": unexpected_keys,
        "loaded": loaded_keys,
        "skipped": skipped_keys,
        "num_loaded": len(loaded_keys),
        "num_skipped": len(skipped_keys),
    }