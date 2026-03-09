"""Global backbone registry – register and create model bundles by name."""

from __future__ import annotations

from typing import Callable, Dict

from .base import ModelBundle

_BUILDERS: Dict[str, Callable[..., ModelBundle]] = {}


def register(name: str):
    """Decorator that registers a backbone builder function under *name*."""

    def deco(fn: Callable[..., ModelBundle]):
        _BUILDERS[name] = fn
        return fn

    return deco


def create_model(name: str, **kwargs) -> ModelBundle:
    """Instantiate a registered backbone by *name*, forwarding ``**kwargs``."""
    if name not in _BUILDERS:
        raise KeyError(
            f"Unknown backbone '{name}'. Available: {sorted(_BUILDERS.keys())}"
        )
    return _BUILDERS[name](**kwargs)
