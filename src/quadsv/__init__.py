"""Compatibility namespace for the former :mod:`quadsv` package."""

from importlib import import_module
from sys import modules
from warnings import warn

_sonic = import_module("sonic")

warn(
    "quadsv has been renamed to sonic; update imports to use 'sonic'",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = list(_sonic.__all__)
__version__ = _sonic.__version__

for _name in __all__:
    globals()[_name] = getattr(_sonic, _name)

_SUBMODULES = (
    "api",
    "_rasterize",
    "statistics",
    "utils",
    "kernels",
    "kernels.base",
    "kernels.fft",
    "kernels.matrix",
    "kernels.nufft",
    "detectors",
    "detectors.base",
    "detectors.grid",
    "detectors.irregular",
    "comparators",
    "comparators.base",
    "comparators.features",
    "comparators.grid",
    "comparators.irregular",
    "comparators.multisample",
    "comparators.normalization",
)

for _name in _SUBMODULES:
    _module = import_module(f"sonic.{_name}")
    modules[f"quadsv.{_name}"] = _module
    if "." not in _name:
        globals()[_name] = _module

del _module, _name, _sonic
