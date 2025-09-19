"""Compatibility utilities for Torch-related functionality."""

from __future__ import annotations

import inspect
from typing import Any

from torch.optim import lr_scheduler


def ensure_reduce_lr_on_plateau_supports_verbose() -> None:
    """Patch ``ReduceLROnPlateau`` to accept the ``verbose`` argument if needed.

    Older versions of :mod:`torch` expose ``ReduceLROnPlateau`` without the
    ``verbose`` keyword argument.  Argus' callback always passes this keyword,
    which leads to a ``TypeError`` during scheduler construction.  This helper
    swaps in a thin wrapper that accepts the ``verbose`` keyword and forwards
    the remaining arguments to the original implementation.
    """
    signature = inspect.signature(lr_scheduler.ReduceLROnPlateau.__init__)
    if "verbose" in signature.parameters:
        return

    original_cls = lr_scheduler.ReduceLROnPlateau

    class _ReduceLROnPlateauCompat(original_cls):  # type: ignore[misc]
        def __init__(self, optimizer: Any, *args: Any, verbose: bool = False,
                     **kwargs: Any) -> None:
            super().__init__(optimizer, *args, **kwargs)
            self.verbose = verbose

        def _reduce_lr(self, epoch: int) -> None:  # type: ignore[override]
            previous_lrs = [group['lr'] for group in self.optimizer.param_groups]
            super()._reduce_lr(epoch)

            if not getattr(self, "verbose", False):
                return

            for index, (old_lr, group) in enumerate(
                zip(previous_lrs, self.optimizer.param_groups)
            ):
                new_lr = group['lr']
                if new_lr == old_lr:
                    continue
                print(
                    f"Epoch {epoch:5d}: reducing learning rate "
                    f"of group {index} to {new_lr:.4e}."
                )

    _ReduceLROnPlateauCompat.__name__ = original_cls.__name__
    _ReduceLROnPlateauCompat.__qualname__ = original_cls.__qualname__
    _ReduceLROnPlateauCompat.__doc__ = original_cls.__doc__
    _ReduceLROnPlateauCompat.__module__ = original_cls.__module__

    lr_scheduler.ReduceLROnPlateau = _ReduceLROnPlateauCompat

