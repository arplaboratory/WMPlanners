
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional
import torch.nn as nn

@dataclass
class ModelBundle:
    model: nn.Module
    preprocess: Callable
    feature_dim: Optional[int]
    modality: str  # "image" or "video"

_BUILDERS: Dict[str, Callable[..., ModelBundle]] = {}

def register(name: str):
    def deco(fn):
        _BUILDERS[name] = fn
        return fn
    return deco

def create_model(name: str, **kwargs) -> ModelBundle:
    if name not in _BUILDERS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_BUILDERS.keys())}")
    return _BUILDERS[name](**kwargs)
