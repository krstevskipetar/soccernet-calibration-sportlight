"""Compatibility layer for legacy imports of the `src` package."""
import sys

from soccernet_calibration import datatools, models, utils

__all__ = ["datatools", "models", "utils"]

_aliases = {
    "datatools": datatools,
    "models": models,
    "utils": utils,
}

for name, module in _aliases.items():
    sys.modules.setdefault(f"src.{name}", module)


def __getattr__(name):
    if name in _aliases:
        return _aliases[name]
    raise AttributeError(f"module 'src' has no attribute '{name}'")


def __dir__():
    return sorted(__all__)
